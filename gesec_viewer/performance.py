from __future__ import annotations

from dataclasses import dataclass, replace

from .config import CameraConfig


@dataclass(frozen=True)
class PerformanceProfile:
    """Effective rendering profile for a camera under a layout policy."""
    layout_slots: int
    target_fps: float
    render_size: tuple[int, int]
    render_interval_ms: int


class PerformancePolicy:
    """Automatic light mode tuned for weak CPUs and multi-camera layouts."""

    _RULES = {
        1: (30.0, (854, 480), 16),
        2: (20.0, (640, 360), 25),
        4: (15.0, (480, 270), 40),
        9: (10.0, (426, 240), 66),
    }

    def profile_for(self, camera: CameraConfig, layout_slots: int) -> PerformanceProfile:
        slots = self._normalize_slots(layout_slots)
        max_fps, max_size, interval = self._RULES[slots]
        return PerformanceProfile(
            layout_slots=slots,
            target_fps=max(1.0, min(camera.target_fps, max_fps)),
            render_size=_bounded_size(camera.render_size, max_size),
            render_interval_ms=interval,
        )

    def adapt_camera(self, camera: CameraConfig, layout_slots: int) -> CameraConfig:
        profile = self.profile_for(camera, layout_slots)
        return replace(camera, target_fps=profile.target_fps, render_size=profile.render_size)

    def render_interval_ms(self, layout_slots: int) -> int:
        slots = self._normalize_slots(layout_slots)
        return self._RULES[slots][2]

    def _normalize_slots(self, layout_slots: int) -> int:
        if layout_slots <= 1:
            return 1
        if layout_slots <= 2:
            return 2
        if layout_slots <= 4:
            return 4
        return 9


def _bounded_size(current: tuple[int, int], maximum: tuple[int, int]) -> tuple[int, int]:
    """Clamp a size to a maximum while preserving aspect ratio."""
    width, height = current
    max_width, max_height = maximum
    if width <= max_width and height <= max_height:
        return width, height

    scale = min(max_width / width, max_height / height)
    bounded_width = min(max_width, max(2, int(round(width * scale))))
    bounded_height = min(max_height, max(2, int(round(height * scale))))
    return _even(bounded_width), _even(bounded_height)


def _even(value: int) -> int:
    """Ensure the value is even, which some video paths prefer."""
    return value if value % 2 == 0 else max(2, value - 1)
