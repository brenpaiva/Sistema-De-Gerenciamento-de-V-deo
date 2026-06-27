from gesec_viewer.config import CameraConfig
from gesec_viewer.ptz import PtzController


class _Profile:
    token = "profile-token"


class _Media:
    def GetProfiles(self):
        return [_Profile()]


class _Ptz:
    def __init__(self):
        self.calls = []

    def ContinuousMove(self, payload):
        self.calls.append(("move", payload))

    def Stop(self, payload):
        self.calls.append(("stop", payload))

    def SetPreset(self, payload):
        self.calls.append(("set", payload))

    def GotoPreset(self, payload):
        self.calls.append(("goto", payload))


class _Client:
    def __init__(self, ptz):
        self.ptz = ptz

    def create_media_service(self):
        return _Media()

    def create_ptz_service(self):
        return self.ptz


def test_ptz_controller_sends_nudge_with_mocked_client():
    fake_ptz = _Ptz()
    camera = CameraConfig(
        id="ptz",
        name="PTZ",
        type="rtsp",
        protocol="rtsp",
        url="rtsp://camera/stream",
        onvif_host="10.0.0.30",
        ptz_enabled=True,
    )
    controller = PtzController(client_factory=lambda camera: _Client(fake_ptz))

    result = controller.nudge(camera, pan=1, tilt=-1, zoom=0, speed=0.5)

    assert result.ok is True
    assert fake_ptz.calls[0][0] == "move"
    assert fake_ptz.calls[0][1]["ProfileToken"] == "profile-token"
    assert fake_ptz.calls[0][1]["Velocity"]["PanTilt"] == {"x": 0.5, "y": -0.5}
    assert fake_ptz.calls[1][0] == "stop"


def test_ptz_controller_rejects_unconfigured_camera():
    camera = CameraConfig(id="fixed", name="Fixed", type="rtsp", url="rtsp://camera/stream")
    controller = PtzController(client_factory=lambda camera: _Client(_Ptz()))

    result = controller.nudge(camera, pan=1)

    assert result.ok is False
