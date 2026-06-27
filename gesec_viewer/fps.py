from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """Track rendering FPS over a sliding time window."""
    def __init__(self, window_seconds: float = 1.5) -> None:
        self.window_seconds = window_seconds
        self._ticks: deque[float] = deque()

    def tick(self, now: float | None = None) -> float:
        timestamp = time.perf_counter() if now is None else now
        self._ticks.append(timestamp)
        self._trim(timestamp)
        return self.fps(now=timestamp)

    def fps(self, now: float | None = None) -> float:
        timestamp = time.perf_counter() if now is None else now
        self._trim(timestamp)
        if len(self._ticks) < 2:
            return 0.0
        elapsed = self._ticks[-1] - self._ticks[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._ticks) - 1) / elapsed

    def reset(self) -> None:
        self._ticks.clear()

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._ticks and self._ticks[0] < cutoff:
            self._ticks.popleft()
