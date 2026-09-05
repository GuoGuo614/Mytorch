"""Shared validation and lazy dispatch for optional Triton kernels."""

import importlib.util

from ..backend import device_of


IMPLEMENTATIONS = {"auto", "eager", "triton"}
SUPPORTED_DTYPES = {"float16", "float32"}
MAX_FUSED_COLUMNS = 65536


def is_triton_available():
    """Return whether a Triton package can be discovered without importing it."""
    try:
        return importlib.util.find_spec("triton") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _capability(array):
    capability = array.device.compute_capability
    if isinstance(capability, bytes):
        capability = capability.decode("ascii")
    if isinstance(capability, str):
        return int(capability[:-1]), int(capability[-1])
    return tuple(capability)


def triton_support_reason(arrays, *, operation, dimensions=None,
                          last_dim_reduction=False):
    """Return None when arrays satisfy the common fused-kernel contract."""
    if not arrays:
        return f"{operation} requires at least one array"
    if any(device_of(array).kind != "cuda" for array in arrays):
        return f"{operation} Triton path requires CUDA arrays"
    if not is_triton_available():
        return "Triton is not installed; install KernelLeaf with the triton extra"
    first_device = device_of(arrays[0])
    if any(device_of(array) != first_device for array in arrays[1:]):
        return f"{operation} arrays must be on the same CUDA device"
    if any(str(array.dtype) not in SUPPORTED_DTYPES for array in arrays):
        return f"{operation} Triton path supports float16 and float32 only"
    if any(not array.flags.c_contiguous for array in arrays):
        return f"{operation} Triton path requires contiguous arrays"
    if any(array.size == 0 for array in arrays):
        return f"{operation} Triton path does not support empty arrays"
    if dimensions is not None and arrays[0].ndim not in dimensions:
        expected = ", ".join(str(value) for value in sorted(dimensions))
        return f"{operation} Triton path requires ndim in {{{expected}}}"
    if last_dim_reduction and arrays[0].shape[-1] > MAX_FUSED_COLUMNS:
        return (
            f"{operation} last dimension exceeds the fused block limit "
            f"of {MAX_FUSED_COLUMNS}"
        )
    if _capability(arrays[0])[0] < 7:
        return f"{operation} Triton path requires CUDA capability 7.0 or newer"
    return None


def resolve_implementation(implementation, arrays, *, operation,
                           dimensions=None, last_dim_reduction=False):
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(
            f"implementation must be one of {sorted(IMPLEMENTATIONS)}, "
            f"got {implementation!r}"
        )
    if implementation == "eager":
        return "eager"
    reason = triton_support_reason(
        arrays,
        operation=operation,
        dimensions=dimensions,
        last_dim_reduction=last_dim_reduction,
    )
    if reason is None:
        try:
            from .driver import ensure_cupy_triton_driver
            ensure_cupy_triton_driver()
        except Exception as error:
            reason = f"Triton CUDA driver initialization failed: {error}"
        else:
            return "triton"
    if implementation == "auto":
        return "eager"
    raise RuntimeError(f"cannot use implementation='triton' for {operation}: {reason}")
