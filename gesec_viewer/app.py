from __future__ import annotations

import logging
import sys
from pathlib import Path

from .camera_store import CameraStore
from .gpu import resolve_device


def run_app(config_path: str | Path, gpu_mode: str, hw_decode: str = "auto") -> int:
    """Load the configured cameras, select a processing backend, and start the Qt app."""
    camera_store = CameraStore(config_path)
    app_config = camera_store.load()
    device_info = resolve_device(gpu_mode)

    logging.info("OpenCV version: %s", device_info.opencv_version)
    logging.info("CUDA available: %s (%s device(s))", device_info.cuda_available, device_info.cuda_device_count)
    logging.info(
        "PyTorch CUDA available: %s (%s, %s)",
        device_info.torch_cuda_available,
        device_info.torch_version,
        device_info.torch_device_name or "no device",
    )
    logging.info("OpenCL available/enabled: %s/%s", device_info.opencl_available, device_info.opencl_enabled)
    logging.info("Selected processing backend: %s - %s", device_info.backend, device_info.reason)
    logging.info("Hardware video decode policy: %s", hw_decode)

    # On Windows, importing PyQt before torch can prevent torch CUDA DLLs from loading.
    from PyQt5.QtGui import QIcon
    from PyQt5.QtWidgets import QApplication

    from .widgets import MainWindow

    qt_app = QApplication(sys.argv[:1])
    qt_app.setApplicationName("GESEC Mini VMS Viewer")

    icon_path = Path(__file__).resolve().parent.parent / "assets/logo.ico"
    if icon_path.exists():
        qt_app.setWindowIcon(QIcon(str(icon_path.resolve())))

    window = MainWindow(app_config, device_info, camera_store, hardware_decode_mode=hw_decode)
    window.show()
    return int(qt_app.exec_())
