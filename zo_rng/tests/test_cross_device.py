"""THE critical test: CPU and GPU must produce bit-exact identical results."""

import pytest
import torch

import zo_rng

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@requires_cuda
@pytest.mark.parametrize("seed", [0, 42, 12345, 2**31 - 1])
@pytest.mark.parametrize(
    "shape",
    [(100,), (64, 128), (3, 224, 224), (10_000_000,)],
    ids=["1d-100", "2d-64x128", "3d-3x224x224", "1d-10M"],
)
def test_cpu_gpu_identical(seed, shape):
    """Same seed must produce bit-exact same tensor on CPU and GPU."""
    z_cpu = zo_rng.randn(seed=seed, shape=shape, device='cpu')
    z_gpu = zo_rng.randn(seed=seed, shape=shape, device='cuda')
    assert torch.equal(z_cpu, z_gpu.cpu()), (
        f"Mismatch for seed={seed}, shape={shape}. "
        f"Max diff = {(z_cpu - z_gpu.cpu()).abs().max().item()}"
    )


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dtype_consistency(dtype):
    """fp16/bf16 conversion must also be identical across devices."""
    z_cpu = zo_rng.randn(seed=42, shape=(10000,), dtype=dtype, device='cpu')
    z_gpu = zo_rng.randn(seed=42, shape=(10000,), dtype=dtype, device='cuda')
    assert torch.equal(z_cpu, z_gpu.cpu()), (
        f"Mismatch for dtype={dtype}. "
        f"Max diff = {(z_cpu.float() - z_gpu.cpu().float()).abs().max().item()}"
    )


@requires_cuda
def test_generator_cross_device():
    """Generator state produces same continuation on CPU and GPU."""
    gen_cpu = zo_rng.Generator(seed=42)
    gen_gpu = zo_rng.Generator(seed=42)

    gen_cpu.randn((1000,), device='cpu')
    gen_gpu.randn((1000,), device='cuda')

    z_cpu = gen_cpu.randn((5000,), device='cpu')
    z_gpu = gen_gpu.randn((5000,), device='cuda')
    assert torch.equal(z_cpu, z_gpu.cpu())
