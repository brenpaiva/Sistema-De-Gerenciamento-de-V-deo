from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any

import numpy as np

from .config import CameraConfig
from .frames import FramePacket


logger = logging.getLogger(__name__)


@dataclass
class RecordingState:
    """Mutable state for an active recording session."""

    camera_id: str
    camera_name: str
    path: Path
    fps: float
    started_at: float = field(default_factory=time.time)
    frame_count: int = 0
    writer: Any | None = None
    frame_size: tuple[int, int] | None = None

    @property
    def duration_seconds(self) -> float:
        return max(0.0, time.time() - self.started_at)


class RecordingManager:
    """Manage asynchronous MP4 recording and snapshots."""

    def __init__(self, root: str | Path = "recordings") -> None:
        self.root = Path(root).expanduser().resolve()
        self._states: dict[str, RecordingState] = {}
        self._queue: Queue[tuple[str, np.ndarray] | None] = Queue(maxsize=120)
        self._lock = Lock()
        self._thread: Thread | None = None
        self.dropped_frames = 0

    def start(self, camera: CameraConfig) -> RecordingState:
        with self._lock:
            existing = self._states.get(camera.id)
            if existing:
                return existing

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            directory = self.root / camera.id
            directory.mkdir(parents=True, exist_ok=True)
            state = RecordingState(
                camera_id=camera.id,
                camera_name=camera.name,
                path=directory / f"{timestamp}.mp4",
                fps=max(1.0, camera.target_fps),
            )
            self._states[camera.id] = state
            self._ensure_thread()
            return state

    def stop(self, camera_id: str, drain: bool = True, timeout_seconds: float | None = None) -> RecordingState | None:
        if drain:
            self._wait_for_queue(timeout_seconds)
        with self._lock:
            state = self._states.pop(camera_id, None)
            if state and state.writer is not None:
                state.writer.release()
                state.writer = None
        return state

    def stop_all(self, drain: bool = True, timeout_seconds: float | None = None) -> list[RecordingState]:
        if drain:
            self._wait_for_queue(timeout_seconds)
        stopped = []
        with self._lock:
            for camera_id in list(self._states):
                state = self._states.pop(camera_id, None)
                if state and state.writer is not None:
                    state.writer.release()
                    state.writer = None
                if state:
                    stopped.append(state)
        return stopped

    def is_recording(self, camera_id: str) -> bool:
        with self._lock:
            return camera_id in self._states

    def active_states(self) -> tuple[RecordingState, ...]:
        with self._lock:
            return tuple(self._states.values())

    def write_frame(self, packet: FramePacket) -> None:
        frame = np.asarray(packet.frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("A gravação espera frames RGB com 3 canais.")

        with self._lock:
            if packet.camera_id not in self._states:
                return
            self._ensure_thread()

        if self._queue.full():
            self.dropped_frames += 1
            return
        self._queue.put((packet.camera_id, frame))

    def _write_frame_sync(self, camera_id: str, frame: np.ndarray) -> None:
        with self._lock:
            state = self._states.get(camera_id)
            if state is None:
                return

            import cv2

            height, width = frame.shape[:2]
            if state.writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                state.writer = cv2.VideoWriter(str(state.path), fourcc, state.fps, (width, height))
                state.frame_size = (width, height)
                if not state.writer.isOpened():
                    state.writer.release()
                    state.writer = None
                    raise RuntimeError(f"Não foi possível abrir o arquivo de gravação: {state.path}")

            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            state.writer.write(bgr)
            state.frame_count += 1

    def _wait_for_queue(self, timeout_seconds: float | None) -> bool:
        if timeout_seconds is None:
            self._queue.join()
            return True

        deadline = time.perf_counter() + max(0.0, timeout_seconds)
        while self._queue.unfinished_tasks:
            if time.perf_counter() >= deadline:
                logger.warning("Timeout ao aguardar fila de gravação esvaziar.")
                return False
            time.sleep(0.02)
        return True

    def _ensure_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = Thread(target=self._run_writer, name="gesec-recording-writer", daemon=True)
        self._thread.start()

    def _run_writer(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                camera_id, frame = item
                try:
                    self._write_frame_sync(camera_id, frame)
                except Exception:
                    logger.exception("Falha ao gravar frame da câmera %s", camera_id)
            finally:
                self._queue.task_done()

    def save_snapshot(self, camera: CameraConfig, frame: Any, root: str | Path = "snapshots") -> Path:
        import cv2

        frame_array = np.asarray(frame)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        directory = Path(root).expanduser().resolve() / camera.id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{timestamp}.png"
        bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"Não foi possível salvar o snapshot: {path}")
        return path
