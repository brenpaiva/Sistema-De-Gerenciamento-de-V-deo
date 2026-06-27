from gesec_viewer.config import CameraConfig
from gesec_viewer.gpu import DeviceInfo
from gesec_viewer.worker import CameraWorker


def _device():
    return DeviceInfo(
        requested="cpu",
        backend="cpu",
        cuda_available=False,
        cuda_device_count=0,
        opencv_version="test",
        reason="test",
    )


class _FakeCapture:
    def __init__(self):
        self.set_calls = []
        self.open_calls = []
        self.released = False

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def open(self, source, backend, params=None):
        self.open_calls.append((source, backend, params or []))
        return True

    def get(self, prop):
        return 21

    def isOpened(self):
        return True

    def release(self):
        self.released = True


class _FakeCv2:
    CAP_PROP_OPEN_TIMEOUT_MSEC = 10
    CAP_PROP_READ_TIMEOUT_MSEC = 11
    CAP_FFMPEG = 12
    CAP_PROP_BUFFERSIZE = 13
    CAP_PROP_HW_ACCELERATION = 20
    VIDEO_ACCELERATION_ANY = 21
    CAP_PROP_HW_DEVICE = 22

    def __init__(self):
        self.capture = _FakeCapture()

    def VideoCapture(self, *args):
        return self.capture


def test_worker_applies_hardware_decode_properties_when_supported():
    fake_cv2 = _FakeCv2()
    camera = CameraConfig(id="rtsp", name="RTSP", type="rtsp", protocol="rtsp", url="rtsp://camera/live")
    worker = CameraWorker(camera, _device(), hardware_decode_mode="auto")

    worker._open_capture(fake_cv2)

    assert fake_cv2.capture.open_calls[0][2] == [
        _FakeCv2.CAP_PROP_HW_ACCELERATION,
        _FakeCv2.VIDEO_ACCELERATION_ANY,
        _FakeCv2.CAP_PROP_HW_DEVICE,
        0,
    ]


def test_worker_skips_hardware_decode_when_global_mode_is_off():
    fake_cv2 = _FakeCv2()
    camera = CameraConfig(id="rtsp", name="RTSP", type="rtsp", protocol="rtsp", url="rtsp://camera/live")
    worker = CameraWorker(camera, _device(), hardware_decode_mode="off")

    worker._open_capture(fake_cv2)

    assert fake_cv2.capture.open_calls[0][2] == []
