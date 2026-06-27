from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gesec_viewer.gpu import resolve_device
from gesec_viewer.processor import FrameProcessor


def main() -> int:
    """Run a simple local benchmark for the frame processor."""
    parser = argparse.ArgumentParser(description="Benchmark simples CPU vs CUDA do pipeline de processamento.")
    parser.add_argument("--gpu", choices=("auto", "cpu", "cuda", "opencl", "torch"), default="auto")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--render-size", default="640x360")
    args = parser.parse_args()

    width, height = _parse_size(args.size)
    render_size = _parse_size(args.render_size)
    device = resolve_device(args.gpu)
    processor = FrameProcessor(device, render_size)

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    frame[:, :, 1] = 80
    frame[:, :, 2] = 120

    start = time.perf_counter()
    for _ in range(args.frames):
        processor.process(frame)
    elapsed = time.perf_counter() - start

    fps = args.frames / elapsed if elapsed > 0 else 0.0
    print(f"Requested: {device.requested}")
    print(f"Selected backend: {processor.backend}")
    print(f"OpenCV: {device.opencv_version}")
    print(f"CUDA devices: {device.cuda_device_count}")
    print(f"PyTorch CUDA: {device.torch_cuda_available} {device.torch_device_name}")
    print(f"Processed frames: {args.frames}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"FPS: {fps:.2f}")
    return 0


def _parse_size(value: str) -> tuple[int, int]:
    """Parse a WIDTHxHEIGHT string into a size tuple."""
    width, height = value.lower().split("x", maxsplit=1)
    return int(width), int(height)


if __name__ == "__main__":
    raise SystemExit(main())
