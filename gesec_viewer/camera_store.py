from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import AppConfig, CameraConfig, load_config, save_config


class CameraStore:
    """Load and persist camera configuration files."""
    def __init__(self, default_path: str | Path, user_path: str | Path | None = None) -> None:
        self.default_path = Path(default_path).expanduser().resolve()
        self.user_path = Path(user_path).expanduser().resolve() if user_path else self.default_path.parent / "user.yaml"

    def load(self) -> AppConfig:
        return load_config(self.default_path)

    def save(self, config: AppConfig) -> AppConfig:
        saved = replace(config, source_path=str(self.user_path))
        save_config(saved, self.user_path)
        return saved

    def add_or_update_camera(self, config: AppConfig, camera: CameraConfig) -> AppConfig:
        cameras = list(config.cameras)
        for index, current in enumerate(cameras):
            if current.id == camera.id:
                cameras[index] = camera
                break
        else:
            cameras.append(camera)
        return replace(config, cameras=tuple(cameras))

    def remove_camera(self, config: AppConfig, camera_id: str) -> AppConfig:
        cameras = tuple(camera for camera in config.cameras if camera.id != camera_id)
        return replace(config, cameras=cameras)
