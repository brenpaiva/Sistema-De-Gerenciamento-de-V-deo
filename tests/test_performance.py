from gesec_viewer.config import CameraConfig
from gesec_viewer.performance import PerformancePolicy


def test_performance_policy_reduces_fps_and_size_for_nine_camera_layout():
    camera = CameraConfig(
        id="cam",
        name="Câmera",
        type="rtsp",
        url="rtsp://example/stream",
        target_fps=30,
        render_size=(1280, 720),
    )
    policy = PerformancePolicy()

    adapted = policy.adapt_camera(camera, 9)

    assert adapted.target_fps == 10
    assert adapted.render_size == (426, 240)
    assert policy.render_interval_ms(9) == 66


def test_performance_policy_keeps_small_camera_size():
    camera = CameraConfig(
        id="cam",
        name="Câmera",
        type="rtsp",
        url="rtsp://example/stream",
        target_fps=12,
        render_size=(426, 240),
    )
    policy = PerformancePolicy()

    adapted = policy.adapt_camera(camera, 1)

    assert adapted.target_fps == 12
    assert adapted.render_size == (426, 240)
