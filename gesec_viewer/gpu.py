from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    """Capabilities and backend choice for the current runtime."""
    requested: str
    backend: str
    cuda_available: bool
    cuda_device_count: int
    opencv_version: str
    reason: str
    opencl_available: bool = False
    opencl_enabled: bool = False
    torch_cuda_available: bool = False
    torch_version: str = "not installed"
    torch_device_name: str = ""


def resolve_device(requested: str) -> DeviceInfo:
    """Resolve the best available processing backend for the requested mode."""
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda", "opencl", "torch"}:
        raise ValueError("GPU mode must be one of: auto, cpu, cuda, opencl, torch.")

    import cv2

    cuda_count = _cuda_device_count(cv2)
    cuda_available = cuda_count > 0
    opencl_available, opencl_enabled = _opencl_status(cv2)
    torch_cuda_available, torch_version, torch_device_name = _torch_cuda_status()

    if requested == "cpu":
        backend = "cpu"
        reason = "CPU mode forced by CLI."
    elif requested == "cuda" and cuda_available:
        backend = "cuda"
        reason = "CUDA mode requested and OpenCV CUDA detected at least one CUDA device."
    elif requested == "cuda" and torch_cuda_available:
        backend = "torch-cuda"
        reason = "CUDA mode requested; OpenCV CUDA is unavailable, so PyTorch CUDA will process frames."
    elif requested == "cuda":
        backend = "cpu"
        reason = "CUDA mode requested, but neither OpenCV CUDA nor PyTorch CUDA are available. Falling back to CPU."
    elif requested == "opencl" and opencl_available:
        backend = "opencl"
        reason = "OpenCL mode requested and OpenCV reported OpenCL support."
    elif requested == "opencl":
        backend = "cpu"
        reason = "OpenCL mode requested, but OpenCV did not report OpenCL support. Falling back to CPU."
    elif requested == "torch" and torch_cuda_available:
        backend = "torch-cuda"
        reason = f"PyTorch CUDA mode requested and device '{torch_device_name}' is available."
    elif requested == "torch":
        backend = "cpu"
        reason = "PyTorch CUDA mode requested, but torch.cuda is not available. Falling back to CPU."
    elif cuda_available:
        backend = "cuda"
        reason = "Auto mode selected CUDA because OpenCV detected a CUDA device."
    elif torch_cuda_available:
        backend = "torch-cuda"
        reason = f"Auto mode selected PyTorch CUDA on '{torch_device_name}' because OpenCV CUDA is unavailable."
    elif opencl_available:
        backend = "opencl"
        reason = "Auto mode selected OpenCL because CUDA is not available in this OpenCV build."
    else:
        backend = "cpu"
        reason = "Auto mode selected CPU because neither CUDA nor OpenCL are available in this OpenCV runtime."

    return DeviceInfo(
        requested=requested,
        backend=backend,
        cuda_available=cuda_available,
        cuda_device_count=cuda_count,
        opencv_version=getattr(cv2, "__version__", "unknown"),
        reason=reason,
        opencl_available=opencl_available,
        opencl_enabled=opencl_enabled,
        torch_cuda_available=torch_cuda_available,
        torch_version=torch_version,
        torch_device_name=torch_device_name,
    )


def _cuda_device_count(cv2_module: object) -> int:
    """Return the number of CUDA devices exposed by OpenCV."""
    try:
        cuda_module = getattr(cv2_module, "cuda", None)
        if cuda_module is None:
            return 0
        return int(cuda_module.getCudaEnabledDeviceCount())
    except Exception:
        return 0


def _opencl_status(cv2_module: object) -> tuple[bool, bool]:
    """Return OpenCL availability and whether it is enabled."""
    try:
        ocl_module = getattr(cv2_module, "ocl", None)
        if ocl_module is None:
            return False, False
        available = bool(ocl_module.haveOpenCL())
        if available:
            ocl_module.setUseOpenCL(True)
        return available, bool(ocl_module.useOpenCL())
    except Exception:
        return False, False


def _torch_cuda_status() -> tuple[bool, str, str]:
    """Return PyTorch CUDA availability, version, and device name."""
    try:
        import torch
    except Exception:
        return False, "not installed", ""

    try:
        available = bool(torch.cuda.is_available())
        device_name = torch.cuda.get_device_name(0) if available else ""
        return available, getattr(torch, "__version__", "unknown"), device_name
    except Exception:
        return False, getattr(torch, "__version__", "unknown"), ""
