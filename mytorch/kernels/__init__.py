"""Optional fused kernels.

This package intentionally imports no Triton modules.  Individual kernel
modules are loaded only after dispatch selects the Triton implementation.
"""

from .runtime import is_triton_available

__all__ = ["is_triton_available"]
