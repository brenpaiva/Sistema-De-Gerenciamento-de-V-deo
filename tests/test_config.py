from pathlib import Path

from gesec_viewer.config import CameraConfig, load_config
from gesec_viewer.camera_store import CameraStore


def test_load_demo_config_has_four_synthetic_cameras():
    config = load_config("config/demo.yaml")

    assert config.title == "GESEC Mini VMS Viewer - Demo Sintética"
    assert len(config.cameras) == 4
    assert config.columns == 2
    assert config.cameras[0].type == "synthetic"
    assert config.cameras[0].url.startswith("synthetic://")
    assert config.cameras[0].render_size == (640, 360)


def test_file_source_is_resolved_relative_to_config(tmp_path):
    config_file = tmp_path / "demo.yaml"
    config_file.write_text(
        """
app:
  title: Demo
cameras:
  - name: File Cam
    type: file
    url: videos/cam.mp4
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    resolved = Path(config.cameras[0].url)
    assert tmp_path in resolved.parents
    assert resolved.parts[-2:] == ("videos", "cam.mp4")


def test_load_rtsp_onvif_camera_fields(tmp_path):
    config_file = tmp_path / "rtsp.yaml"
    config_file.write_text(
        """
app:
  title: VMS
cameras:
  - id: patio
    name: Patio
    type: rtsp
    protocol: rtsp
    url: rtsp://user:pass@10.0.0.10:554/stream1
    active: true
    loop: false
    target_fps: 25
    render_size: [1280, 720]
    onvif_host: 10.0.0.10
    onvif_port: 8899
    username: admin
    password: secret
    ptz_enabled: true
    hardware_decode: off
""",
        encoding="utf-8",
    )

    config = load_config(config_file)
    camera = config.cameras[0]

    assert camera.type == "rtsp"
    assert camera.protocol == "rtsp"
    assert camera.url.startswith("rtsp://")
    assert camera.onvif_host == "10.0.0.10"
    assert camera.onvif_port == 8899
    assert camera.username == "admin"
    assert camera.password == "secret"
    assert camera.ptz_enabled is True
    assert camera.active is True
    assert camera.loop is False
    assert camera.render_size == (1280, 720)
    assert camera.hardware_decode == "off"


def test_camera_config_defaults_hardware_decode_to_auto(tmp_path):
    config_file = tmp_path / "demo.yaml"
    config_file.write_text(
        """
app:
  title: Demo
cameras:
  - id: demo
    name: Demo
    type: synthetic
    url: synthetic://demo
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.cameras[0].hardware_decode == "auto"


def test_camera_store_persists_added_camera(tmp_path):
    default_config = tmp_path / "demo.yaml"
    user_config = tmp_path / "user.yaml"
    default_config.write_text(
        """
app:
  title: Demo
cameras:
  - id: demo
    name: Demo
    type: synthetic
    url: synthetic://demo
""",
        encoding="utf-8",
    )
    store = CameraStore(default_config, user_config)

    config = store.load()
    camera = CameraConfig(
        id="entrada",
        name="Entrada",
        type="rtsp",
        protocol="rtsp",
        url="rtsp://user:pass@10.0.0.20:554/live",
        onvif_host="10.0.0.20",
        ptz_enabled=True,
    )
    saved = store.save(store.add_or_update_camera(config, camera))
    reloaded = load_config(user_config)

    assert saved.source_path == str(user_config.resolve())
    assert user_config.exists()
    assert [item.id for item in reloaded.cameras] == ["demo", "entrada"]
    assert reloaded.cameras[1].onvif_host == "10.0.0.20"


def test_camera_store_load_respects_explicit_default_path(tmp_path):
    default_config = tmp_path / "demo.yaml"
    user_config = tmp_path / "user.yaml"
    default_config.write_text(
        """
app:
  title: Demo Sintética
cameras:
  - id: demo
    name: Demo
    type: synthetic
    url: synthetic://demo
""",
        encoding="utf-8",
    )
    user_config.write_text(
        """
app:
  title: Configuração Do Usuário
cameras:
  - id: user_rtsp
    name: Usuário
    type: rtsp
    url: rtsp://camera.local/live
""",
        encoding="utf-8",
    )

    config = CameraStore(default_config, user_config).load()

    assert config.title == "Demo Sintética"
    assert [camera.id for camera in config.cameras] == ["demo"]
