from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FramePacket:
    """Processed frame data passed from workers to the UI."""
    camera_id: str
    camera_name: str
    sequence: int
    frame: Any
    captured_at: float
    processed_at: float
    backend: str
