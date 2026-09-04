"""CuPy-backed Triton driver registration.

Triton's stock CUDA driver obtains the active device and stream through
PyTorch.  MyTorch must not acquire a hidden PyTorch runtime dependency, so
this module supplies the same small driver contract from CuPy instead.
"""

import importlib.util
import os
from pathlib import Path


_installed_driver = None


def _configure_bundled_cuda():
    """Expose triton-windows' bundled headers/tools to its build helper."""
    if os.name != "nt" or os.environ.get("CUDA_PATH"):
        return
    spec = importlib.util.find_spec("triton")
    if spec is None or spec.origin is None:
        return
    cuda_root = Path(spec.origin).parent / "backends" / "nvidia"
    required = (
        cuda_root / "bin" / "ptxas.exe",
        cuda_root / "include" / "cuda.h",
        cuda_root / "lib" / "x64" / "cuda.lib",
    )
    if all(path.is_file() for path in required):
        os.environ["CUDA_PATH"] = str(cuda_root)


def ensure_cupy_triton_driver():
    """Install and return a Triton CUDA driver that never imports PyTorch."""
    global _installed_driver
    if _installed_driver is not None:
        return _installed_driver

    _configure_bundled_cuda()
    import cupy as cp
    from triton.backends.nvidia.driver import CudaDriver, CudaLauncher, CudaUtils
    from triton.runtime import driver as driver_config

    class CuPyCudaDriver(CudaDriver):
        def __init__(self):
            # Deliberately skip GPUDriver.__init__, which imports torch.
            self.utils = CudaUtils()
            self.launcher_cls = CudaLauncher

        @staticmethod
        def get_current_device():
            return cp.cuda.runtime.getDevice()

        @staticmethod
        def set_current_device(index):
            cp.cuda.runtime.setDevice(index)

        @staticmethod
        def get_current_stream(index):
            del index
            return cp.cuda.get_current_stream().ptr

        @staticmethod
        def get_device_capability(index):
            capability = cp.cuda.Device(index).compute_capability
            if isinstance(capability, bytes):
                capability = capability.decode("ascii")
            if isinstance(capability, str):
                return int(capability[:-1]), int(capability[-1])
            return tuple(capability)

        @staticmethod
        def is_active():
            return True

        @staticmethod
        def get_device_interface():
            return cp.cuda

    _installed_driver = CuPyCudaDriver()
    driver_config.set_active(_installed_driver)
    return _installed_driver
