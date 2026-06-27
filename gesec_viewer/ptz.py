from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .config import CameraConfig


@dataclass(frozen=True)
class PtzResult:
    """Result of a PTZ operation."""
    ok: bool
    message: str


class PtzController:
    """Small ONVIF/PTZ adapter used by the UI."""
    def __init__(self, client_factory: Callable[[CameraConfig], Any] | None = None) -> None:
        self.client_factory = client_factory

    def is_configured(self, camera: CameraConfig) -> bool:
        return bool(camera.ptz_enabled and camera.onvif_host)

    def probe(self, camera: CameraConfig) -> PtzResult:
        if not self.is_configured(camera):
            return PtzResult(False, "ONVIF/PTZ não configurado para esta câmera.")
        try:
            ptz, token = self._session(camera)
            if not token:
                return PtzResult(False, "Câmera ONVIF sem perfil de mídia.")
            if not ptz:
                return PtzResult(False, "Serviço PTZ não disponível.")
        except Exception as exc:
            return PtzResult(False, f"Falha ONVIF: {exc}")
        return PtzResult(True, "ONVIF/PTZ disponível.")

    def nudge(self, camera: CameraConfig, pan: float = 0.0, tilt: float = 0.0, zoom: float = 0.0, speed: float = 0.35) -> PtzResult:
        if not self.is_configured(camera):
            return PtzResult(False, "Câmera sem ONVIF/PTZ configurado.")
        try:
            ptz, token = self._session(camera)
            velocity = {
                "PanTilt": {"x": _clamp(pan * speed), "y": _clamp(tilt * speed)},
                "Zoom": {"x": _clamp(zoom * speed)},
            }
            ptz.ContinuousMove({"ProfileToken": token, "Velocity": velocity})
            time.sleep(0.18)
            ptz.Stop({"ProfileToken": token, "PanTilt": True, "Zoom": bool(zoom)})
        except Exception as exc:
            return PtzResult(False, f"Falha ao mover PTZ: {exc}")
        return PtzResult(True, "Comando PTZ enviado.")

    def set_preset(self, camera: CameraConfig, name: str) -> PtzResult:
        if not self.is_configured(camera):
            return PtzResult(False, "Câmera sem ONVIF/PTZ configurado.")
        if not name.strip():
            return PtzResult(False, "Informe o nome do preset.")
        try:
            ptz, token = self._session(camera)
            ptz.SetPreset({"ProfileToken": token, "PresetName": name.strip()})
        except Exception as exc:
            return PtzResult(False, f"Falha ao salvar preset: {exc}")
        return PtzResult(True, f"Preset '{name.strip()}' salvo.")

    def goto_preset(self, camera: CameraConfig, name: str) -> PtzResult:
        if not self.is_configured(camera):
            return PtzResult(False, "Câmera sem ONVIF/PTZ configurado.")
        if not name.strip():
            return PtzResult(False, "Informe o nome/token do preset.")
        try:
            ptz, token = self._session(camera)
            ptz.GotoPreset({"ProfileToken": token, "PresetToken": name.strip()})
        except Exception as exc:
            return PtzResult(False, f"Falha ao chamar preset: {exc}")
        return PtzResult(True, f"Preset '{name.strip()}' acionado.")

    def _session(self, camera: CameraConfig) -> tuple[Any, str]:
        client = self._client(camera)
        media = client.create_media_service()
        ptz = client.create_ptz_service()
        profiles = media.GetProfiles()
        if not profiles:
            return ptz, ""
        return ptz, profiles[0].token

    def _client(self, camera: CameraConfig) -> Any:
        if self.client_factory:
            return self.client_factory(camera)

        try:
            from onvif import ONVIFCamera
        except ImportError as exc:
            raise RuntimeError("Instale onvif-zeep para usar PTZ ONVIF.") from exc

        return ONVIFCamera(
            camera.onvif_host,
            camera.onvif_port,
            camera.username or None,
            camera.password or None,
        )


def _clamp(value: float) -> float:
    """Clamp PTZ velocity values to the supported range."""
    return max(-1.0, min(1.0, float(value)))
