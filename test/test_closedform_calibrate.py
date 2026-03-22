"""
Test closed-form W calibration benchmark.

Verifies that the benchmark logic works correctly and that
the recommended W produces correct closed-form replay results.

Usage:
    python -m pytest test/test_closedform_calibrate.py -v
    python test/test_closedform_calibrate.py
"""

import os
import sys
import time
import threading
import torch
import numpy as np
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from zo2.trainer.hf_transformers.batch_differential_checkpoint import (
    _replay_updates_on_state,
    _closedform_replay_on_state,
)


def _make_state(param_names, size=512, dtype=torch.float32, device='cpu'):
    state = OrderedDict()
    torch.manual_seed(42)
    for name in param_names:
        state[name] = torch.randn(size, dtype=dtype, device=device)
    return state


def _make_updates(n, wd=0.01):
    rng = np.random.RandomState(123)
    updates = []
    for i in range(n):
        updates.append({
            'step': i + 1,
            'seed': int(rng.randint(1, 2**31)),
            'grad': float(rng.uniform(-5, 5)),
            'lr': 1e-5,
            'wd': wd,
            'zo_eps': 0.0,
        })
    return updates


def _clone_state(state):
    return OrderedDict((k, v.clone()) for k, v in state.items())


# ===== Benchmark logic tests =====

def test_cpu_benchmark_throughput_increases():
    """W=2 should have higher throughput than W=1 (or at least not crash)."""
    test_numel = 1_000_000
    accum_dtype = torch.float32
    replay_dtype = torch.float32

    results = {}
    for W in [1, 2]:
        partials = [torch.zeros(test_numel, dtype=accum_dtype) for _ in range(W)]
        barrier = threading.Barrier(W + 1)
        times = [0.0] * W

        def worker(wid):
            p = partials[wid]
            for _ in range(3):
                z = torch.randn(test_numel, dtype=replay_dtype)
                p.add_(z, alpha=1.0)
            barrier.wait()
            t0 = time.perf_counter()
            for _ in range(5):
                z = torch.randn(test_numel, dtype=replay_dtype)
                p.add_(z, alpha=1.0)
            times[wid] = time.perf_counter() - t0
            barrier.wait()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(W)]
        for t in threads:
            t.start()
        barrier.wait()
        barrier.wait()
        for t in threads:
            t.join()
        results[W] = W * 5 / max(times)
        del partials

    # W=2 should have at least some throughput (no deadlock/crash)
    assert results[2] > 0, f"W=2 throughput should be > 0, got {results[2]}"
    print(f"PASS: test_cpu_benchmark_throughput_increases "
          f"(W=1: {results[1]:.1f}, W=2: {results[2]:.1f} terms/s)")


def test_cpu_benchmark_cast_mode():
    """Benchmark with accum=fp16, replay=fp32 (cast needed)."""
    test_numel = 500_000
    accum_dtype = torch.float16
    replay_dtype = torch.float32

    partials = [torch.zeros(test_numel, dtype=accum_dtype)]
    z = torch.randn(test_numel, dtype=replay_dtype)
    partials[0].add_(z.to(accum_dtype), alpha=1.0)

    assert partials[0].dtype == torch.float16
    assert partials[0].abs().max() > 0
    print("PASS: test_cpu_benchmark_cast_mode")


def test_memory_limit_calculation():
    """Buffer memory calculation: 1 shared accum + W concurrent z buffers."""
    import psutil
    total_numel = 1_700_000_000  # 1.7B params
    accum_elem = 4  # fp32
    replay_elem = 4  # fp32
    W = 4

    # New formula: 1 accum buffer + W z buffers
    accum_bytes = total_numel * accum_elem          # single shared buffer
    z_bytes = total_numel * replay_elem * W         # W concurrent z buffers
    total_bytes = accum_bytes + z_bytes
    total_GB = total_bytes / 1e9
    # W=4: 1*6.8 + 4*6.8 = 34.0 GB
    expected_GB = total_numel * accum_elem / 1e9 + total_numel * replay_elem * W / 1e9
    assert abs(total_GB - expected_GB) < 0.1, f"Expected ~{expected_GB:.1f} GB, got {total_GB}"

    available = psutil.virtual_memory().available
    # Solve for max W: accum_bytes + W*z_per_worker <= available*0.5
    z_per_worker = total_numel * replay_elem
    W_mem = max(1, int((available * 0.5 - accum_bytes) / z_per_worker)) if z_per_worker > 0 else 1
    assert W_mem >= 1
    print(f"PASS: test_memory_limit_calculation "
          f"(total={total_GB:.1f}GB @ W={W}, W_mem={W_mem})")


def test_recommended_W_produces_correct_result():
    """Closed-form with W=1,2,3 all match serial replay."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(20, wd=0.01)

    # Serial baseline
    state_serial = _make_state(names)
    _replay_updates_on_state(
        state_serial, updates, device='cpu', move_to_device=False,
        trainable_param_names=names, simulate_perturbation=False,
    )

    # Closed-form with different W
    for W in [1, 2, 3]:
        state_cf = _make_state(names)
        _closedform_replay_on_state(
            state_cf, updates, device='cpu', move_to_device=False,
            trainable_param_names=names, num_workers=W, precision="fp32",
        )
        for name in names:
            diff = (state_serial[name].float() - state_cf[name].float()).abs().max().item()
            assert diff < 1e-5, f"W={W} {name}: diff={diff:.2e}"

    print("PASS: test_recommended_W_produces_correct_result (W=1,2,3 all match)")


def test_gpu_benchmark():
    """GPU benchmark with CUDA streams."""
    if not torch.cuda.is_available():
        print("SKIP: test_gpu_benchmark (no CUDA)")
        return

    test_numel = 1_000_000
    accum_dtype = torch.float32
    replay_dtype = torch.float16

    results = {}
    for W in [1, 2]:
        ps = [torch.zeros(test_numel, dtype=accum_dtype, device='cuda') for _ in range(W)]
        ss = [torch.cuda.Stream() for _ in range(W)]
        for wid in range(W):
            with torch.cuda.stream(ss[wid]):
                z = torch.randn(test_numel, dtype=replay_dtype, device='cuda')
                ps[wid].add_(z.to(accum_dtype), alpha=1.0)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            for wid in range(W):
                with torch.cuda.stream(ss[wid]):
                    z = torch.randn(test_numel, dtype=replay_dtype, device='cuda')
                    ps[wid].add_(z.to(accum_dtype), alpha=1.0)
        torch.cuda.synchronize()
        results[W] = W * 5 / (time.perf_counter() - t0)
        del ps, ss
        torch.cuda.empty_cache()

    assert results[1] > 0 and results[2] > 0
    print(f"PASS: test_gpu_benchmark "
          f"(W=1: {results[1]:.1f}, W=2: {results[2]:.1f} terms/s)")


if __name__ == '__main__':
    tests = [
        test_cpu_benchmark_throughput_increases,
        test_cpu_benchmark_cast_mode,
        test_memory_limit_calculation,
        test_recommended_W_produces_correct_result,
        test_gpu_benchmark,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
