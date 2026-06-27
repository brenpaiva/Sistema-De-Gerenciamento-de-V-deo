from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_RENDER_SIZE = (640, 360)


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for a single camera source."""
    id: str
    name: str
    url: str
    type: str = "synthetic"
    protocol: str = "synthetic"
    onvif_host: str = ""
    onvif_port: int = 80
    username: str = ""
    password: str = ""
    ptz_enabled: bool = False
    active: bool = True
    loop: bool = True
    target_fps: float = 30.0
    render_size: tuple[int, int] = DEFAULT_RENDER_SIZE
    hardware_decode: str = "auto"


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration loaded from disk."""
    title: str
    columns: int
    reconnect_delay_seconds: float
    cameras: tuple[CameraConfig, ...]
    render_size: tuple[int, int] = DEFAULT_RENDER_SIZE
    source_path: str = ""


def load_config(path: str | Path) -> AppConfig:
    """Load an application configuration from YAML or JSON."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    raw = _read_mapping(config_path)
    app_section = raw.get("app", {})
    cameras_section = raw.get("cameras", [])

    if not isinstance(cameras_section, list) or not cameras_section:
        raise ValueError("A configuração precisa incluir pelo menos uma câmera na lista 'cameras'.")

    default_render_size = _parse_render_size(app_section.get("render_size", DEFAULT_RENDER_SIZE))
    cameras = tuple(
        _parse_camera(index, item, config_path.parent, default_render_size)
        for index, item in enumerate(cameras_section, start=1)
    )

    return AppConfig(
        title=str(app_section.get("title", "GESEC Mini VMS Viewer")),
        columns=max(1, int(app_section.get("columns", 2))),
        reconnect_delay_seconds=max(0.5, float(app_section.get("reconnect_delay_seconds", 2.0))),
        cameras=cameras,
        render_size=default_render_size,
        source_path=str(config_path),
    )


def save_config(config: AppConfig, path: str | Path) -> None:
    """Write an application configuration to YAML or JSON."""
    config_path = Path(path).expanduser().resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = config_to_dict(config)

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML é necessário para gravar configurações YAML. Instale requirements.txt.") from exc

        config_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    elif suffix == ".json":
        config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        raise ValueError("A configuração deve ser salva como arquivo .yaml, .yml ou .json.")


def config_to_dict(config: AppConfig) -> dict:
    """Convert an AppConfig into a serializable dictionary."""
    return {
        "app": {
            "title": config.title,
            "columns": config.columns,
            "render_size": list(config.render_size),
            "reconnect_delay_seconds": config.reconnect_delay_seconds,
        },
        "cameras": [camera_to_dict(camera) for camera in config.cameras],
    }


def camera_to_dict(camera: CameraConfig) -> dict:
    """Convert a CameraConfig into a serializable dictionary."""
    return {
        "id": camera.id,
        "name": camera.name,
        "type": camera.type,
        "protocol": camera.protocol,
        "url": camera.url,
        "active": camera.active,
        "loop": camera.loop,
        "target_fps": camera.target_fps,
        "render_size": list(camera.render_size),
        "hardware_decode": camera.hardware_decode,
        "onvif_host": camera.onvif_host,
        "onvif_port": camera.onvif_port,
        "username": camera.username,
        "password": camera.password,
        "ptz_enabled": camera.ptz_enabled,
    }


def _read_mapping(path: Path) -> dict:
    """Read a YAML or JSON file and return its root mapping."""
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML é necessário para ler configurações YAML. Instale requirements.txt.") from exc

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("A configuração deve ser um arquivo .yaml, .yml ou .json.")

    if not isinstance(data, dict):
        raise ValueError("A raiz da configuração deve ser um mapa/objeto.")
    return data


def _parse_camera(index: int, raw: dict, config_dir: Path, default_render_size: tuple[int, int]) -> CameraConfig:
    """Validate and normalize a raw camera mapping."""
    if not isinstance(raw, dict):
        raise ValueError(f"Câmera #{index} deve ser um mapa/objeto.")

    name = str(raw.get("name") or f"Camera {index}")
    camera_id = str(raw.get("id") or _slugify(name) or f"camera_{index}")
    source_type = str(raw.get("type", "synthetic")).lower()
    protocol = str(raw.get("protocol") or source_type).lower()
    url = str(raw.get("url", "synthetic://demo"))

    return CameraConfig(
        id=camera_id,
        name=name,
        url=_resolve_url(source_type, url, config_dir),
        type=source_type,
        protocol=protocol,
        onvif_host=str(raw.get("onvif_host", "")),
        onvif_port=max(1, int(raw.get("onvif_port", 80))),
        username=str(raw.get("username", "")),
        password=str(raw.get("password", "")),
        ptz_enabled=_parse_bool(raw.get("ptz_enabled", raw.get("ptz", False))),
        active=_parse_bool(raw.get("active", True)),
        loop=_parse_bool(raw.get("loop", True)),
        target_fps=max(1.0, float(raw.get("target_fps", 30.0))),
        render_size=_parse_render_size(raw.get("render_size", default_render_size)),
        hardware_decode=_parse_hardware_decode(raw.get("hardware_decode", "auto")),
    )


def _parse_render_size(value: object) -> tuple[int, int]:
    """Parse a render size expressed as a tuple or WIDTHxHEIGHT string."""
    if isinstance(value, str):
        parts = value.lower().split("x", maxsplit=1)
    else:
        parts = list(value) if isinstance(value, (list, tuple)) else list(DEFAULT_RENDER_SIZE)

    if len(parts) != 2:
        raise ValueError("render_size deve ter largura e altura.")

    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("Os valores de render_size devem ser positivos.")
    return width, height


def _resolve_url(source_type: str, url: str, config_dir: Path) -> str:
    """Resolve relative file URLs against the configuration directory."""
    parsed = urlparse(url)
    if source_type in {"file", "video"} and not parsed.scheme:
        return str((config_dir / url).resolve())
    return url


def _parse_bool(value: object) -> bool:
    """Parse common truthy values from configuration input."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "sim", "on"}
    return bool(value)


def _parse_hardware_decode(value: object) -> str:
    """Parse the hardware decode policy for a camera."""
    if isinstance(value, bool):
        return "auto" if value else "off"
    normalized = str(value or "auto").strip().lower()
    if normalized not in {"auto", "off"}:
        raise ValueError("hardware_decode deve ser 'auto' ou 'off'.")
    return normalized


def _slugify(value: str) -> str:
    """Create a stable identifier from a human-readable name."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug
