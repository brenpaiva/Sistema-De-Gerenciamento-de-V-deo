from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from .config import CameraConfig
from .frames import FramePacket
from .gpu import DeviceInfo
from .processor import FrameProcessor

logger = logging.getLogger(__name__)


class CameraWorker(QObject):
    """Capture, process, and emit frames for a single camera source."""
    frame_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str, str)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(
        self,
        camera: CameraConfig,
        device_info: DeviceInfo,
        reconnect_delay_seconds: float = 2.0,
        hardware_decode_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.camera = camera
        self.device_info = device_info
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.hardware_decode_mode = hardware_decode_mode
        self.processor = FrameProcessor(device_info, camera.render_size)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._sequence = 0
        self._last_status = ""
        self._hw_decode_logged = False
        self._hw_decode_report_logged = False

    def run(self) -> None:
        self._set_status("Iniciando")
        try:
            if self.camera.type == "synthetic":
                self._run_synthetic()
            else:
                self._run_capture()
        except Exception as exc:
            logger.exception("Worker crashed for %s", self.camera.id)
            self._set_status(f"Erro: {exc}")
        finally:
            self._set_status("Parado")
            self.finished.emit(self.camera.id)

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()

    def _run_synthetic(self) -> None:
        interval = 1.0 / self.camera.target_fps
        next_frame_at = time.perf_counter()

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                self._set_status("Pausado")
                time.sleep(0.05)
                continue

            self._set_status("Rodando")
            frame = self._make_synthetic_frame()
            self._emit_frame(frame)

            next_frame_at += interval
            sleep_for = next_frame_at - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_frame_at = time.perf_counter()

    def _run_capture(self) -> None:
        import cv2

        while not self._stop_event.is_set():
            self._set_status("Conectando")
            cap = self._open_capture(cv2)
            if not cap.isOpened():
                self._set_status("Erro de conexão")
                self.log_message.emit(f"{self.camera.name}: unable to open source {self.camera.url}")
                time.sleep(self.reconnect_delay_seconds)
                continue

            try:
                self._capture_loop(cap)
            finally:
                cap.release()

            if not self._stop_event.is_set():
                self._set_status("Reconectando")
                time.sleep(self.reconnect_delay_seconds)

    def _open_capture(self, cv2_module: object):
        source: int | str = self.camera.url
        if self.camera.type in {"device", "webcam"} and self.camera.url.isdigit():
            source = int(self.camera.url)

        if self.camera.type == "rtsp":
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

        cap = cv2_module.VideoCapture()
        try:
            cap.set(cv2_module.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2_module.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        except Exception:
            pass
        hw_decode_params = self._hardware_decode_open_params(cv2_module)
        opened = self._open_with_ffmpeg(cv2_module, cap, source, hw_decode_params)
        if not opened:
            cap.release()
            cap = cv2_module.VideoCapture()
            opened = self._open_with_ffmpeg(cv2_module, cap, source, hw_decode_params)
        if not cap.isOpened():
            cap.release()
            cap = cv2_module.VideoCapture(source)
        if cap.isOpened():
            self._log_hardware_decode_report(cv2_module, cap)

        try:
            cap.set(cv2_module.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _open_with_ffmpeg(self, cv2_module: object, cap: object, source: int | str, params: list[int]) -> bool:
        if params:
            try:
                return bool(cap.open(source, cv2_module.CAP_FFMPEG, params))
            except TypeError:
                self._log_hardware_decode("OpenCV sem suporte a open(..., params); tentando fallback sem parâmetros.")
            except Exception as exc:
                self._log_hardware_decode(f"falha ao abrir com parâmetros de HW decode: {exc}")

        return bool(cap.open(source, cv2_module.CAP_FFMPEG))

    def _hardware_decode_open_params(self, cv2_module: object) -> list[int]:
        if not self._should_try_hardware_decode():
            self._log_hardware_decode("decodificação por hardware desativada.")
            return []

        acceleration_prop = getattr(cv2_module, "CAP_PROP_HW_ACCELERATION", None)
        acceleration_any = getattr(cv2_module, "VIDEO_ACCELERATION_ANY", None)
        if acceleration_prop is None or acceleration_any is None:
            self._log_hardware_decode("OpenCV sem suporte exposto para CAP_PROP_HW_ACCELERATION.")
            return []

        params = [acceleration_prop, acceleration_any]

        device_prop = getattr(cv2_module, "CAP_PROP_HW_DEVICE", None)
        if device_prop is not None:
            params.extend([device_prop, 0])

        self._log_hardware_decode(
            f"tentativa de HW decode auto via parâmetros do FFmpeg/OpenCV (CAP_PROP_HW_ACCELERATION={acceleration_any})."
        )
        return params

    def _should_try_hardware_decode(self) -> bool:
        return self.hardware_decode_mode != "off" and self.camera.hardware_decode != "off"

    def _log_hardware_decode(self, message: str) -> None:
        if self._hw_decode_logged:
            return
        self._hw_decode_logged = True
        full_message = f"{self.camera.name}: {message}"
        logger.info(full_message)
        self.log_message.emit(full_message)

    def _log_hardware_decode_report(self, cv2_module: object, cap: object) -> None:
        if self._hw_decode_report_logged or not self._should_try_hardware_decode():
            return

        acceleration_prop = getattr(cv2_module, "CAP_PROP_HW_ACCELERATION", None)
        if acceleration_prop is None or not hasattr(cap, "get"):
            return

        self._hw_decode_report_logged = True
        try:
            value = cap.get(acceleration_prop)
        except Exception:
            logger.debug("Could not read CAP_PROP_HW_ACCELERATION for %s", self.camera.id, exc_info=True)
            return

        message = f"{self.camera.name}: HW decode reportado pelo OpenCV = {value}"
        logger.info(message)
        self.log_message.emit(message)

    def _capture_loop(self, cap: object) -> None:
        import cv2

        interval = 1.0 / self.camera.target_fps
        next_frame_at = time.perf_counter()
        self._set_status("Rodando")

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                self._set_status("Pausado")
                time.sleep(0.05)
                continue

            self._set_status("Rodando")
            ok, frame = cap.read()
            if not ok:
                if self.camera.loop and self.camera.type in {"file", "video"}:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            self._emit_frame(frame)
            next_frame_at += interval
            sleep_for = next_frame_at - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_frame_at = time.perf_counter()

    def _emit_frame(self, bgr_frame: np.ndarray) -> None:
        captured_at = time.perf_counter()
        try:
            rgb_frame = self.processor.process(bgr_frame)
        except Exception as exc:
            self.log_message.emit(f"{self.camera.name}: frame processing failed: {exc}")
            return
        processed_at = time.perf_counter()
        self._sequence += 1

        self.frame_ready.emit(
            FramePacket(
                camera_id=self.camera.id,
                camera_name=self.camera.name,
                sequence=self._sequence,
                frame=rgb_frame,
                captured_at=captured_at,
                processed_at=processed_at,
                backend=self.processor.backend,
            )
        )

    def _make_synthetic_frame(self) -> np.ndarray:
        import cv2

        width = max(self.camera.render_size[0], 160)
        height = max(self.camera.render_size[1], 90)
        hue = _stable_hue(self.camera.id)
        t = self._sequence / max(1.0, self.camera.target_fps)

        base = np.zeros((height, width, 3), dtype=np.uint8)
        gradient = np.linspace(18, 80, width, dtype=np.uint8)
        base[:, :, 0] = gradient
        base[:, :, 1] = 28 + (hue % 45)
        base[:, :, 2] = 34 + (hue % 60)

        line_x = int((math.sin(t * 1.7) * 0.5 + 0.5) * (width - 160)) + 80
        line_y = int((math.cos(t * 1.1) * 0.5 + 0.5) * (height - 120)) + 60

        cv2.rectangle(base, (32, 38), (width - 32, height - 38), (95, 135, 125), 2)
        cv2.line(base, (0, line_y), (width, line_y), (70, 160, 130), 2)
        cv2.line(base, (line_x, 0), (line_x, height), (140, 190, 120), 2)
        cv2.circle(base, (line_x, line_y), 34, (70, 210, 160), -1)

        timestamp = time.strftime("%H:%M:%S")
        scale = max(0.45, min(1.0, width / 960))
        thickness = max(1, int(round(scale * 2)))
        cv2.putText(base, self.camera.name, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 245, 235), thickness)
        cv2.putText(base, f"Stream sintético | {timestamp}", (20, height - 28), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, (205, 225, 215), thickness)
        cv2.putText(base, f"Frame {self._sequence:06d}", (max(20, width - 180), height - 28), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, (205, 225, 215), thickness)
        return base

    def _set_status(self, status: str) -> None:
        if status != self._last_status:
            self._last_status = status
            self.status_changed.emit(self.camera.id, status)


def _stable_hue(value: str) -> int:
    """Derive a stable hue value from an identifier."""
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int(digest[0])
