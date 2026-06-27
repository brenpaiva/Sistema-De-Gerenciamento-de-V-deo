from __future__ import annotations

import argparse
import logging
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the viewer entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m gesec_viewer",
        description="Mini VMS viewer com PyQt5, OpenCV, threads e GPU opcional.",
    )
    parser.add_argument(
        "--config",
        default=Path("config/demo.yaml"),
        help="Arquivo YAML/JSON com as câmeras da demo.",
    )
    parser.add_argument(
        "--gpu",
        choices=("auto", "cpu", "cuda", "opencl", "torch"),
        default="auto",
        help="Backend de processamento. 'auto' usa OpenCV CUDA, PyTorch CUDA, OpenCL ou CPU.",
    )
    parser.add_argument(
        "--hw-decode",
        choices=("auto", "off"),
        default="auto",
        help="Tenta aceleração de decodificação por hardware no OpenCV/FFmpeg quando suportada.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Nível de logs no terminal.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and start the application."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    from .app import run_app

    return run_app(args.config, args.gpu, args.hw_decode)
