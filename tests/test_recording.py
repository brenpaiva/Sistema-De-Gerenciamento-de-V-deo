import numpy as np

from gesec_viewer.config import CameraConfig
from gesec_viewer.frames import FramePacket
from gesec_viewer.recording import RecordingManager


def test_recording_manager_writes_mp4(tmp_path):
    manager = RecordingManager(tmp_path / "recordings")
    camera = CameraConfig(id="demo", name="Demo", type="synthetic", url="synthetic://demo", target_fps=10)
    manager.start(camera)

    for sequence in range(3):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = sequence * 30
        packet = FramePacket(
            camera_id=camera.id,
            camera_name=camera.name,
            sequence=sequence,
            frame=frame,
            captured_at=0.0,
            processed_at=0.0,
            backend="cpu",
        )
        manager.write_frame(packet)

    state = manager.stop(camera.id)

    assert state is not None
    assert state.frame_count == 3
    assert state.path.exists()
    assert state.path.stat().st_size > 0


def test_recording_manager_can_stop_without_draining_queue(tmp_path, monkeypatch):
    manager = RecordingManager(tmp_path / "recordings")
    camera = CameraConfig(id="demo", name="Demo", type="synthetic", url="synthetic://demo", target_fps=10)
    manager.start(camera)

    def fail_if_waiting(timeout_seconds):
        raise AssertionError("stop(drain=False) não deve aguardar a fila")

    monkeypatch.setattr(manager, "_wait_for_queue", fail_if_waiting)
    state = manager.stop(camera.id, drain=False)

    assert state is not None
    assert state.camera_id == camera.id
