import numpy as np

from gesec_viewer.gpu import DeviceInfo
from gesec_viewer.processor import FrameProcessor


def test_cpu_processor_resizes_and_converts_bgr_to_rgb():
    device = DeviceInfo(
        requested="cpu",
        backend="cpu",
        cuda_available=False,
        cuda_device_count=0,
        opencv_version="test",
        reason="test",
    )
    processor = FrameProcessor(device, (20, 10))
    frame = np.zeros((30, 40, 3), dtype=np.uint8)
    frame[:, :, 0] = 255

    result = processor.process(frame)

    assert result.shape == (10, 20, 3)
    assert result[0, 0, 2] == 255
    assert processor.backend == "cpu"


def test_opencl_processor_resizes_and_converts_bgr_to_rgb():
    device = DeviceInfo(
        requested="opencl",
        backend="opencl",
        cuda_available=False,
        cuda_device_count=0,
        opencv_version="test",
        reason="test",
        opencl_available=True,
        opencl_enabled=True,
    )
    processor = FrameProcessor(device, (20, 10))
    frame = np.zeros((30, 40, 3), dtype=np.uint8)
    frame[:, :, 1] = 128

    result = processor.process(frame)

    assert result.shape == (10, 20, 3)
    assert result[0, 0, 1] == 128
    assert processor.backend in {"opencl", "cpu"}


def test_torch_cuda_backend_falls_back_cleanly(monkeypatch):
    device = DeviceInfo(
        requested="torch",
        backend="torch-cuda",
        cuda_available=False,
        cuda_device_count=0,
        opencv_version="test",
        reason="test",
        torch_cuda_available=True,
        torch_version="test",
        torch_device_name="test",
    )
    processor = FrameProcessor(device, (20, 10))
    monkeypatch.setattr(processor, "_process_torch_cuda", lambda frame: (_ for _ in ()).throw(RuntimeError("simulated")))
    frame = np.zeros((30, 40, 3), dtype=np.uint8)
    frame[:, :, 2] = 64

    result = processor.process(frame)

    assert result.shape == (10, 20, 3)
    assert result[0, 0, 0] == 64
