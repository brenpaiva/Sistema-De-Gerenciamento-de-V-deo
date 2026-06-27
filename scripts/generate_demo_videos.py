from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    """Generate local demo MP4 files for file-based camera tests."""
    parser = argparse.ArgumentParser(description="Gera dois MP4s locais para testar config/file.example.yaml.")
    parser.add_argument("--output", default="media")
    parser.add_argument("--seconds", type=int, default=12)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_video(output_dir / "demo_cam1.mp4", "Portaria Demo", args.seconds, args.fps, (42, 110, 90))
    _write_video(output_dir / "demo_cam2.mp4", "Garagem Demo", args.seconds, args.fps, (90, 76, 150))
    print(f"Videos generated in {output_dir.resolve()}")
    return 0


def _write_video(path: Path, label: str, seconds: int, fps: int, color: tuple[int, int, int]) -> None:
    """Render a synthetic MP4 test video with simple overlays."""
    width, height = 1280, 720
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    total = seconds * fps
    for index in range(total):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(15, 70, width, dtype=np.uint8)
        frame[:, :, 1] = color[1]
        frame[:, :, 2] = color[2]
        x = int((index / total) * (width - 180)) + 90
        y = int((np.sin(index / 20.0) * 0.5 + 0.5) * (height - 180)) + 90
        cv2.rectangle(frame, (40, 40), (width - 40, height - 40), color, 3)
        cv2.circle(frame, (x, y), 42, (210, 230, 190), -1)
        cv2.putText(frame, label, (64, 106), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (235, 245, 235), 3)
        cv2.putText(frame, f"Frame {index:05d}", (64, height - 72), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 235, 220), 2)
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    raise SystemExit(main())
