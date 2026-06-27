from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from .gpu import DeviceInfo, resolve_device
from .processor import FrameProcessor


HIGH_LATENCY_P95_MS = 200.0


@dataclass(frozen=True)
class BenchmarkProfile:
    """Benchmark workload profile for a given camera layout."""
    key: str
    name: str
    camera_count: int
    target_fps_per_camera: float
    input_size: tuple[int, int]
    render_size: tuple[int, int]

    @property
    def target_total_fps(self) -> float:
        return self.camera_count * self.target_fps_per_camera


@dataclass(frozen=True)
class BenchmarkProgress:
    """Progress snapshot emitted while a benchmark is running."""
    elapsed_seconds: float
    frames: int
    total_fps: float
    fps_per_camera: float
    percent: float
    backend: str


@dataclass(frozen=True)
class BenchmarkResult:
    """Final benchmark metrics and environment details."""
    generated_at: str
    profile: BenchmarkProfile
    requested_backend: str
    selected_backend: str
    opencv_version: str
    cuda_device_count: int
    torch_cuda_available: bool
    torch_device_name: str
    frames: int
    elapsed_seconds: float
    total_fps: float
    fps_per_camera: float
    latency_avg_ms: float
    latency_p95_ms: float
    verdict: str
    platform: str
    python_version: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["profile"]["target_total_fps"] = self.profile.target_total_fps
        return data


BENCHMARK_PROFILES: dict[str, BenchmarkProfile] = {
    "2": BenchmarkProfile(
        key="2",
        name="2 câmeras",
        camera_count=2,
        target_fps_per_camera=20.0,
        input_size=(1280, 720),
        render_size=(640, 360),
    ),
    "4": BenchmarkProfile(
        key="4",
        name="4 câmeras",
        camera_count=4,
        target_fps_per_camera=12.0,
        input_size=(1280, 720),
        render_size=(640, 360),
    ),
    "9": BenchmarkProfile(
        key="9",
        name="9 câmeras",
        camera_count=9,
        target_fps_per_camera=10.0,
        input_size=(854, 480),
        render_size=(426, 240),
    ),
}


def classify_benchmark(total_fps: float, target_total_fps: float, latency_p95_ms: float) -> str:
    """Classify benchmark quality from throughput and tail latency."""
    if total_fps < target_total_fps or latency_p95_ms > HIGH_LATENCY_P95_MS:
        return "Ruim"
    if total_fps >= target_total_fps * 3.0:
        return "Excelente"
    if total_fps >= target_total_fps * 1.5:
        return "Bom"
    return "Aceitável"


def run_processing_benchmark(
    profile: BenchmarkProfile,
    requested_backend: str,
    duration_seconds: float,
    stop_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[BenchmarkProgress], None] | None = None,
    device_info: DeviceInfo | None = None,
) -> BenchmarkResult:
    """Run a synthetic processing benchmark against the selected backend."""
    device = device_info or resolve_device(requested_backend)
    processor = FrameProcessor(device, profile.render_size)
    frames = _make_synthetic_frames(profile)
    stop = stop_requested or (lambda: False)
    duration = max(0.01, float(duration_seconds))

    processed = 0
    latencies_ms: list[float] = []
    started = time.perf_counter()
    next_progress = started + 0.25

    while not stop():
        elapsed = time.perf_counter() - started
        if elapsed >= duration:
            break

        frame = frames[processed % profile.camera_count]
        frame_started = time.perf_counter()
        processor.process(frame)
        latencies_ms.append((time.perf_counter() - frame_started) * 1000.0)
        processed += 1

        now = time.perf_counter()
        if progress_callback and now >= next_progress:
            progress_callback(_progress(profile, processor.backend, processed, started, now, duration))
            next_progress = now + 0.25

    finished = time.perf_counter()
    elapsed_seconds = max(0.000001, finished - started)
    total_fps = processed / elapsed_seconds
    fps_per_camera = total_fps / profile.camera_count
    latency_avg_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    latency_p95_ms = _percentile(latencies_ms, 0.95)
    verdict = classify_benchmark(total_fps, profile.target_total_fps, latency_p95_ms)

    if progress_callback:
        progress_callback(_progress(profile, processor.backend, processed, started, finished, duration))

    return BenchmarkResult(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        profile=profile,
        requested_backend=device.requested,
        selected_backend=processor.backend,
        opencv_version=device.opencv_version,
        cuda_device_count=device.cuda_device_count,
        torch_cuda_available=device.torch_cuda_available,
        torch_device_name=device.torch_device_name,
        frames=processed,
        elapsed_seconds=elapsed_seconds,
        total_fps=total_fps,
        fps_per_camera=fps_per_camera,
        latency_avg_ms=latency_avg_ms,
        latency_p95_ms=latency_p95_ms,
        verdict=verdict,
        platform=platform.platform(),
        python_version=platform.python_version(),
    )


def save_benchmark_report(result: BenchmarkResult, root: str | Path = "benchmarks") -> Path:
    """Persist a benchmark result as a timestamped JSON report."""
    directory = Path(root).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"{stamp}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


class BenchmarkWorker(QObject):
    """Qt worker that runs a benchmark without blocking the UI thread."""
    progress = pyqtSignal(object)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, profile: BenchmarkProfile, requested_backend: str, duration_seconds: float) -> None:
        super().__init__()
        self.profile = profile
        self.requested_backend = requested_backend
        self.duration_seconds = duration_seconds
        self._stop_requested = False

    def run(self) -> None:
        try:
            result = run_processing_benchmark(
                self.profile,
                self.requested_backend,
                self.duration_seconds,
                stop_requested=lambda: self._stop_requested,
                progress_callback=self.progress.emit,
            )
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)

    def stop(self) -> None:
        self._stop_requested = True


def _make_synthetic_frames(profile: BenchmarkProfile) -> list[np.ndarray]:
    """Create deterministic synthetic frames for a benchmark profile."""
    width, height = profile.input_size
    base_x = np.linspace(0, 255, width, dtype=np.uint8)
    frames: list[np.ndarray] = []
    for index in range(profile.camera_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (base_x + index * 17) % 255
        frame[:, :, 1] = (80 + index * 23) % 255
        frame[:, :, 2] = (120 + index * 31) % 255
        frames.append(frame)
    return frames


def _progress(
    profile: BenchmarkProfile,
    backend: str,
    frames: int,
    started: float,
    now: float,
    duration_seconds: float,
) -> BenchmarkProgress:
    """Build a progress event from benchmark timing information."""
    elapsed = max(0.000001, now - started)
    total_fps = frames / elapsed
    return BenchmarkProgress(
        elapsed_seconds=elapsed,
        frames=frames,
        total_fps=total_fps,
        fps_per_camera=total_fps / profile.camera_count,
        percent=min(100.0, (elapsed / duration_seconds) * 100.0),
        backend=backend,
    )


def _percentile(values: list[float], percentile: float) -> float:
    """Return a simple nearest-rank percentile for a list of values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return ordered[index]
