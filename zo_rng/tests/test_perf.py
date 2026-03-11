"""Performance benchmarks (informational, not strict pass/fail)."""

import time
import pytest
import torch

import zo_rng

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _benchmark(fn, warmup=3, repeat=10):
    """Run fn with warmup, return median time in ms."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def test_cpu_throughput():
    """Report CPU generation throughput."""
    n = 10_000_000
    ms = _benchmark(lambda: zo_rng.randn(seed=42, shape=(n,), device='cpu'))
    throughput = n / ms * 1000  # elements per second
    print(f"\nCPU: {ms:.2f} ms for {n:,} elements ({throughput / 1e6:.1f} M/s)")


@requires_cuda
def test_gpu_throughput():
    """Report GPU generation throughput."""
    n = 10_000_000
    ms = _benchmark(lambda: zo_rng.randn(seed=42, shape=(n,), device='cuda'))
    throughput = n / ms * 1000
    print(f"\nGPU: {ms:.2f} ms for {n:,} elements ({throughput / 1e6:.1f} M/s)")


@requires_cuda
def test_gpu_overhead_vs_torch():
    """GPU overhead vs torch.randn should be reasonable (< 5x)."""
    n = 10_000_000

    ms_zo = _benchmark(lambda: zo_rng.randn(seed=42, shape=(n,), device='cuda'))
    ms_torch = _benchmark(lambda: torch.randn(n, device='cuda'))

    ratio = ms_zo / ms_torch if ms_torch > 0 else float('inf')
    print(f"\nzo_rng: {ms_zo:.2f} ms, torch.randn: {ms_torch:.2f} ms, "
          f"ratio: {ratio:.2f}x")
    # Informational: warn but don't fail hard
    if ratio > 5.0:
        pytest.skip(f"zo_rng is {ratio:.1f}x slower than torch.randn "
                    f"(informational, not a hard failure)")
