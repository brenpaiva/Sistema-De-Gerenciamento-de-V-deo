import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication

from gesec_viewer.camera_store import CameraStore
from gesec_viewer.gpu import DeviceInfo
from gesec_viewer.widgets import MainWindow


def _app():
    return QApplication.instance() or QApplication(sys.argv[:1])


def _device():
    return DeviceInfo(
        requested="cpu",
        backend="cpu",
        cuda_available=False,
        cuda_device_count=0,
        opencv_version="test",
        reason="test",
    )


def test_sidebar_removes_map_and_logs():
    app = _app()
    store = CameraStore("config/demo.yaml")
    window = MainWindow(store.load(), _device(), store)

    labels = [button.text() for button in window._nav_buttons.values()]

    assert "Mapa" not in labels
    assert "Logs" not in labels
    assert labels == ["Monitor", "Câmeras", "Gravação", "PTZ", "Eventos", "Configurações", "Benchmark", "Sobre"]
    window.close()
    app.quit()


def test_visible_and_capture_cameras_follow_layout_and_recording():
    app = _app()
    store = CameraStore("config/demo.yaml")
    window = MainWindow(store.load(), _device(), store)

    window._layout_slots = 2
    assert len(window._visible_cameras()) == 2
    assert len(window._capture_cameras()) == 2

    hidden_camera = window.app_config.cameras[3]
    window.recording_manager.start(hidden_camera)

    assert hidden_camera in window._capture_cameras()
    assert len(window._capture_cameras()) == 3
    window.recording_manager.stop_all()
    window.close()
    app.quit()


def test_rtsp_deduplication_plans_one_worker_for_duplicate_public_streams():
    app = _app()
    store = CameraStore("config/rtsp.example.yaml")
    window = MainWindow(store.load(), _device(), store)

    cameras = window.app_config.cameras[:9]
    plan = window._capture_plan(list(cameras))

    assert len(plan) == 1
    assert plan[0][0].id == "rtsp_public_01"
    assert plan[0][1] == [f"rtsp_public_{index:02d}" for index in range(2, 10)]
    window.close()
    app.quit()


def test_benchmark_start_creates_worker_and_stop_cancels(monkeypatch):
    class FakeBenchmarkWorker(QObject):
        progress = pyqtSignal(object)
        finished = pyqtSignal(object)
        failed = pyqtSignal(str)

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.stopped = False

        def run(self):
            return

        def stop(self):
            self.stopped = True

    app = _app()
    monkeypatch.setattr("gesec_viewer.widgets.BenchmarkWorker", FakeBenchmarkWorker)
    store = CameraStore("config/demo.yaml")
    window = MainWindow(store.load(), _device(), store)

    window.start_benchmark()

    assert window._benchmark_worker is not None
    assert window._benchmark_thread is not None

    window.stop_benchmark()

    assert window._benchmark_worker is None
    assert window._benchmark_thread is None
    window.close()
    app.quit()
