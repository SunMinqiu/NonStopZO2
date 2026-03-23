"""Extension loader for zo_rng.

Tries to import the compiled C/CUDA extension. Falls back to a pure-Python
reference implementation if the native extension is not available.
"""

import torch

_has_native = False
_has_native_cuda = False

try:
    from zo_rng._ext_impl import generate_normal as _generate_normal_native
    from zo_rng._ext_impl import has_cuda as _has_cuda_native
    _has_native = True
    _has_native_cuda = _has_cuda_native()
except ImportError:
    pass


def _generate_normal(seed, counter, n, device_str, pool_id=0):
    """Generate n normal floats. Returns a 1-D float32 tensor."""
    if _has_native:
        want_cuda = device_str.startswith('cuda')

        if want_cuda and not _has_native_cuda:
            # Extension built CPU-only: generate on CPU, transfer
            result = _generate_normal_native(seed, counter, n, 'cpu', pool_id)
            return result.to(device_str)

        if want_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "zo_rng: CUDA requested but torch.cuda is not available")

        return _generate_normal_native(seed, counter, n, device_str, pool_id)

    # No native extension — fall back to pure-Python reference (no pool support)
    from .reference import generate_normal_reference
    result = generate_normal_reference(seed, counter, n)
    if device_str != 'cpu':
        result = result.to(device_str)
    return result
