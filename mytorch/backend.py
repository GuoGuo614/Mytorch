"""Small NumPy/CuPy array backend used by MyTorch.

CuPy is deliberately imported only when a CUDA-specific function is called.
This keeps a CPU-only installation importable without optional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np


_DLL_DIRECTORY_HANDLES = []
_CUDA_COMPONENTS_CONFIGURED = False


def _configure_cuda_component_wheels():
    """Expose NVIDIA component-wheel DLLs before importing CuPy on Windows."""
    global _CUDA_COMPONENTS_CONFIGURED
    if _CUDA_COMPONENTS_CONFIGURED or os.name != "nt":
        return
    _CUDA_COMPONENTS_CONFIGURED = True
    bin_dirs = []
    for package in ("nvidia.cuda_runtime", "nvidia.cuda_nvrtc"):
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ModuleNotFoundError):
            spec = None
        if spec is not None and spec.origin:
            bin_dir = Path(spec.origin).resolve().parent / "bin"
            if bin_dir.is_dir():
                bin_dirs.append(str(bin_dir))
    if not bin_dirs:
        return
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(bin_dirs + [current_path])
    if hasattr(os, "add_dll_directory"):
        for directory in bin_dirs:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(directory))


def _load_cupy():
    _configure_cuda_component_wheels()
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CUDA support requires CuPy. Install MyTorch with "
            "`pip install -e '.[cuda]'`."
        ) from exc
    return cp


def _is_cupy_array(value: Any) -> bool:
    module = type(value).__module__.split(".", 1)[0]
    return module == "cupy"


@dataclass(frozen=True)
class Device:
    kind: str
    index: Optional[int] = None

    def __post_init__(self):
        if self.kind not in {"cpu", "cuda"}:
            raise ValueError(f"unknown device kind: {self.kind!r}")
        if self.kind == "cpu" and self.index is not None:
            raise ValueError("CPU device does not accept an index")

    def __repr__(self) -> str:
        return "cpu" if self.kind == "cpu" else f"cuda:{self.index}"

    __str__ = __repr__

    @property
    def xp(self):
        return np if self.kind == "cpu" else _load_cupy()

    def enabled(self) -> bool:
        return self.kind == "cpu" or is_cuda_available(self.index or 0)

    def asarray(self, value, dtype=None):
        return to_device(value, self, dtype=dtype)

    def empty(self, shape, dtype="float32"):
        with _device_context(self):
            return self.xp.empty(_normalize_shape(shape), dtype=dtype)

    def zeros(self, *shape, dtype="float32"):
        with _device_context(self):
            return self.xp.zeros(_normalize_shape(shape), dtype=dtype)

    def ones(self, *shape, dtype="float32"):
        with _device_context(self):
            return self.xp.ones(_normalize_shape(shape), dtype=dtype)

    def full(self, shape, fill_value, dtype="float32"):
        with _device_context(self):
            return self.xp.full(_normalize_shape(shape), fill_value, dtype=dtype)

    def rand(self, *shape, dtype="float32"):
        with _device_context(self):
            return self.xp.random.rand(*_normalize_shape(shape)).astype(dtype)

    def randn(self, *shape, dtype="float32"):
        with _device_context(self):
            return self.xp.random.randn(*_normalize_shape(shape)).astype(dtype)

    def one_hot(self, n, indices, dtype="float32"):
        with _device_context(self):
            return self.xp.eye(n, dtype=dtype)[indices]


class _device_context:
    def __init__(self, device: Device):
        self.device = device
        self.context = None

    def __enter__(self):
        if self.device.kind == "cuda":
            self.context = _load_cupy().cuda.Device(self.device.index)
            self.context.__enter__()

    def __exit__(self, exc_type, exc, traceback):
        if self.context is not None:
            return self.context.__exit__(exc_type, exc, traceback)
        return False


def _normalize_shape(shape):
    if isinstance(shape, tuple) and len(shape) == 1 and isinstance(shape[0], tuple):
        return shape[0]
    return tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)


_CPU = Device("cpu")


def cpu() -> Device:
    return _CPU


def cuda(index: int = 0) -> Device:
    if not is_cuda_available(index):
        raise RuntimeError(
            f"CUDA device cuda:{index} is unavailable. Install a CUDA 12 compatible "
            "CuPy build and check the NVIDIA driver/device."
        )
    return Device("cuda", index)


def default_device() -> Device:
    return cpu()


def all_devices():
    devices = [cpu()]
    if is_cuda_available(0):
        cp = _load_cupy()
        devices.extend(Device("cuda", i) for i in range(cp.cuda.runtime.getDeviceCount()))
    return devices


def is_cuda_available(index: int = 0) -> bool:
    try:
        cp = _load_cupy()
        if not 0 <= index < cp.cuda.runtime.getDeviceCount():
            return False
        with cp.cuda.Device(index):
            probe = cp.zeros(1, dtype=cp.float32)
            _ = probe + 1
            cp.cuda.get_current_stream().synchronize()
        return True
    except Exception:
        return False


def get_array_module(value):
    if hasattr(value, "realize_cached_data"):
        value = value.realize_cached_data()
    return _load_cupy() if _is_cupy_array(value) else np


def device_of(value) -> Device:
    if hasattr(value, "realize_cached_data"):
        value = value.realize_cached_data()
    if _is_cupy_array(value):
        return Device("cuda", int(value.device.id))
    return cpu()


def asarray(value, device=None, dtype=None):
    device = device_of(value) if device is None else device
    return to_device(value, device, dtype=dtype)


def asnumpy(value):
    if hasattr(value, "realize_cached_data"):
        value = value.realize_cached_data()
    if _is_cupy_array(value):
        return _load_cupy().asnumpy(value)
    return np.asarray(value)


def to_device(value, device: Device, dtype=None):
    if not isinstance(device, Device):
        raise TypeError(f"device must be a Device, got {type(device).__name__}")
    if device.kind == "cpu":
        value = asnumpy(value) if _is_cupy_array(value) else value
        return np.asarray(value, dtype=dtype)
    cp = _load_cupy()
    with cp.cuda.Device(device.index):
        return cp.asarray(value, dtype=dtype)


def empty(shape, *, device=None, dtype="float32"):
    return (device or cpu()).empty(shape, dtype=dtype)


def zeros(*shape, device=None, dtype="float32"):
    return (device or cpu()).zeros(*shape, dtype=dtype)


def ones(*shape, device=None, dtype="float32"):
    return (device or cpu()).ones(*shape, dtype=dtype)


def randn(*shape, device=None, dtype="float32"):
    return (device or cpu()).randn(*shape, dtype=dtype)
