from gesec_viewer.benchmark import (
    BenchmarkProfile,
    classify_benchmark,
    run_processing_benchmark,
)
from gesec_viewer.gpu import DeviceInfo


def _device():
    return DeviceInfo(
        requested="cpu",
        backend="cpu",
        cuda_available=False,
        cuda_device_count=0,
        opencv_version="test",
        reason="test",
    )


def test_benchmark_classification_excellent():
    assert classify_benchmark(total_fps=300, target_total_fps=100, latency_p95_ms=5) == "Excelente"


def test_benchmark_classification_good():
    assert classify_benchmark(total_fps=150, target_total_fps=100, latency_p95_ms=5) == "Bom"


def test_benchmark_classification_acceptable():
    assert classify_benchmark(total_fps=100, target_total_fps=100, latency_p95_ms=5) == "Aceitável"


def test_benchmark_classification_bad_for_low_fps_or_high_latency():
    assert classify_benchmark(total_fps=99, target_total_fps=100, latency_p95_ms=5) == "Ruim"
    assert classify_benchmark(total_fps=500, target_total_fps=100, latency_p95_ms=250) == "Ruim"


def test_processing_benchmark_returns_valid_metrics():
    profile = BenchmarkProfile(
        key="test",
        name="Teste",
        camera_count=2,
        target_fps_per_camera=5,
        input_size=(160, 90),
        render_size=(80, 45),
    )

    result = run_processing_benchmark(profile, "cpu", duration_seconds=0.02, device_info=_device())

    assert result.frames > 0
    assert result.total_fps > 0
    assert result.fps_per_camera > 0
    assert result.selected_backend == "cpu"
    assert result.verdict in {"Excelente", "Bom", "Aceitável", "Ruim"}
