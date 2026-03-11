"""zo_rng — Cross-device deterministic normal RNG based on Philox4x32-10.

Generates bit-exact identical float32 normal-distributed tensors from the
same seed, regardless of whether execution happens on CPU or GPU.

Usage:
    import zo_rng

    z = zo_rng.randn(seed=42, shape=(1024, 768), device='cuda')

    gen = zo_rng.Generator(seed=42)
    z1 = gen.randn((1024,))
    z2 = gen.randn((1024,))  # continues from where z1 left off
"""

import torch
from ._ext import _generate_normal

__all__ = ['Generator', 'randn']


class Generator:
    """Cross-device deterministic normal RNG based on Philox4x32-10.

    The generator maintains a (seed, counter) state. Each call to randn()
    advances the counter. Bit-exact identical output for the same
    (seed, counter, shape) on any device.
    """

    def __init__(self, seed: int):
        self.seed = seed & 0xFFFFFFFFFFFFFFFF  # uint64 range
        self.counter = 0

    def randn(self, shape, dtype=torch.float32, device='cpu'):
        """Generate normal-distributed tensor.

        Args:
            shape: Output tensor shape.
            dtype: Output dtype (default float32). Generation is always
                   in fp32; other dtypes are obtained by casting.
            device: 'cpu', 'cuda', or 'cuda:N'.

        Returns:
            Tensor of the given shape, dtype, and device.
        """
        n = 1
        for s in shape:
            n *= s

        result_fp32 = _generate_normal(self.seed, self.counter, n, str(device))
        self.counter += (n + 3) // 4  # each Philox call produces 4 values

        result = result_fp32.view(shape)
        if dtype != torch.float32:
            result = result.to(dtype)
        return result

    def state_dict(self):
        """Return generator state as a dict (for checkpointing)."""
        return {'seed': self.seed, 'counter': self.counter}

    def load_state_dict(self, d):
        """Restore generator state from a dict."""
        self.seed = d['seed']
        self.counter = d['counter']

    def __repr__(self):
        return f"zo_rng.Generator(seed={self.seed}, counter={self.counter})"


def randn(seed: int, shape, dtype=torch.float32, device='cpu'):
    """One-shot: generate a full tensor from seed (counter starts at 0).

    Args:
        seed: Integer seed.
        shape: Output tensor shape.
        dtype: Output dtype (default float32).
        device: 'cpu', 'cuda', or 'cuda:N'.

    Returns:
        Tensor of the given shape, dtype, and device.
    """
    gen = Generator(seed)
    return gen.randn(shape, dtype=dtype, device=device)
