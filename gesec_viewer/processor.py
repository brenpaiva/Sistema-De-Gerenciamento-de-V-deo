from __future__ import annotations

import logging

import numpy as np

from .gpu import DeviceInfo

logger = logging.getLogger(__name__)


class FrameProcessor:
    """Process BGR frames into RGB render frames using the best available backend."""
    def __init__(self, device_info: DeviceInfo, render_size: tuple[int, int]) -> None:
        self.device_info = device_info
        self.render_size = render_size
        self._cuda_failed = False
        self._opencl_failed = False

    @property
    def backend(self) -> str:
        if self.device_info.backend == "cuda" and not self._cuda_failed:
            return "cuda"
        if self.device_info.backend == "torch-cuda":
            return "torch-cuda"
        if self.device_info.backend == "opencl" and not self._opencl_failed:
            return "opencl"
        return "cpu"

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame received.")

        if self.device_info.backend == "cuda" and not self._cuda_failed:
            try:
                return self._process_cuda(frame)
            except Exception as exc:
                self._cuda_failed = True
                logger.warning("CUDA processing failed once; falling back to CPU. Error: %s", exc)

        if self.device_info.backend == "torch-cuda":
            try:
                return self._process_torch_cuda(frame)
            except Exception as exc:
                logger.warning("PyTorch CUDA processing failed once; falling back to CPU. Error: %s", exc)
                return self._process_cpu(frame)

        if self.device_info.backend == "opencl" and not self._opencl_failed:
            try:
                return self._process_opencl(frame)
            except Exception as exc:
                self._opencl_failed = True
                logger.warning("OpenCL processing failed once; falling back to CPU. Error: %s", exc)

        return self._process_cpu(frame)

    def _process_cpu(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        resized = cv2.resize(frame, self.render_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb)

    def _process_cuda(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        gpu_resized = cv2.cuda.resize(gpu_frame, self.render_size)
        gpu_rgb = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2RGB)
        rgb = gpu_rgb.download()
        return np.ascontiguousarray(rgb)

    def _process_opencl(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        cv2.ocl.setUseOpenCL(True)
        gpu_frame = cv2.UMat(frame)
        gpu_resized = cv2.resize(gpu_frame, self.render_size, interpolation=cv2.INTER_AREA)
        gpu_rgb = cv2.cvtColor(gpu_resized, cv2.COLOR_BGR2RGB)
        rgb = gpu_rgb.get()
        return np.ascontiguousarray(rgb)

    def _process_torch_cuda(self, frame: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        target_width, target_height = self.render_size
        with torch.inference_mode():
            tensor = torch.from_numpy(frame).to("cuda", non_blocking=True)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            tensor = tensor[:, [2, 1, 0], :, :].float().div_(255.0)
            resized = functional.interpolate(
                tensor,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
            rgb = resized.mul(255.0).clamp_(0, 255).byte()
            rgb = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.ascontiguousarray(rgb)
