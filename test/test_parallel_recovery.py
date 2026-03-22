"""
Test pipelined producer-consumer recovery vs sequential replay.

Result should be bitwise-exact (torch.equal) since parameter updates remain
sequential — only z generation is parallelized/pipelined.

Usage:
    python -m pytest test/test_parallel_recovery.py -v
    # or directly:
    python test/test_parallel_recovery.py
"""

import os
import sys
import torch
import numpy as np
from collections import OrderedDict

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from zo2.trainer.hf_transformers.batch_differential_checkpoint import (
    _replay_updates_on_state,
    _parallel_replay_updates_on_state,
)


def _make_state(param_names, size=512, dtype=torch.float32, device='cpu'):
    """Create a deterministic test state dict."""
    state = OrderedDict()
    torch.manual_seed(42)
    for name in param_names:
        state[name] = torch.randn(size, dtype=dtype, device=device)
    return state


def _make_updates(n, wd=0.01, varying_lr=False):
    """Generate synthetic update records."""
    rng = np.random.RandomState(123)
    updates = []
    for i in range(n):
        updates.append({
            'step': i + 1,
            'seed': int(rng.randint(1, 2**31)),
            'grad': float(rng.uniform(-5, 5)),
            'lr': float(rng.uniform(1e-6, 1e-4)) if varying_lr else 1e-5,
            'wd': wd,
            'zo_eps': 1e-3,
        })
    return updates


def _clone_state(state):
    return OrderedDict((k, v.clone()) for k, v in state.items())


def _run_comparison(param_names, updates, simulate_perturbation=True,
                    zo2_mode=False, initial_prev_seed=None,
                    rng_device="native", replay_in_fp32=False,
                    dtype=torch.float32, num_workers=4):
    """Run sequential vs pipelined and assert bitwise equality."""
    state_seq = _make_state(param_names, dtype=dtype)
    state_par = _clone_state(state_seq)

    # Sequential
    _replay_updates_on_state(
        state_seq, updates, device='cpu', move_to_device=False,
        trainable_param_names=param_names,
        simulate_perturbation=simulate_perturbation,
        replay_in_fp32=replay_in_fp32,
        rng_device=rng_device,
        zo2_mode=zo2_mode,
        initial_prev_seed=initial_prev_seed,
    )

    # Pipelined
    os.environ['PARALLEL_RECOVERY_WORKERS'] = str(num_workers)
    _parallel_replay_updates_on_state(
        state_par, updates, device='cpu', move_to_device=False,
        trainable_param_names=param_names,
        simulate_perturbation=simulate_perturbation,
        replay_in_fp32=replay_in_fp32,
        rng_device=rng_device,
        zo2_mode=zo2_mode,
        initial_prev_seed=initial_prev_seed,
    )

    for name in param_names:
        if not torch.equal(state_seq[name], state_par[name]):
            diff = (state_seq[name] - state_par[name]).abs().max().item()
            raise AssertionError(
                f"MISMATCH {name}: max_diff={diff:.2e}\n"
                f"  seq[:5]={state_seq[name][:5].tolist()}\n"
                f"  par[:5]={state_par[name][:5].tolist()}"
            )
    return True


# ===== Test cases =====

def test_basic_no_wd():
    """All bias params (no weight decay)."""
    names = ['layer.0.bias', 'layer.1.bias']
    updates = _make_updates(20, wd=0.0)
    _run_comparison(names, updates, simulate_perturbation=False)
    print("PASS: test_basic_no_wd")


def test_basic_with_wd():
    """All weight params (with weight decay)."""
    names = ['layer.0.weight', 'layer.1.weight']
    updates = _make_updates(20, wd=0.01)
    _run_comparison(names, updates, simulate_perturbation=False)
    print("PASS: test_basic_with_wd")


def test_mixed_params():
    """Mix of weight and bias params."""
    names = ['layer.0.weight', 'layer.0.bias', 'layer.1.weight', 'layer_norm.weight']
    updates = _make_updates(30, wd=0.01)
    _run_comparison(names, updates, simulate_perturbation=False)
    print("PASS: test_mixed_params")


def test_with_perturbation_simulation():
    """simulate_perturbation=True should be bitwise exact."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(20, wd=0.01)
    _run_comparison(names, updates, simulate_perturbation=True)
    print("PASS: test_with_perturbation_simulation")


def test_varying_lr():
    """Different lr per step."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(25, wd=0.01, varying_lr=True)
    _run_comparison(names, updates, simulate_perturbation=True)
    print("PASS: test_varying_lr")


def test_grad_zero_steps():
    """Some steps with grad=0 (perturbation-only)."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(20, wd=0.01)
    # Set some grads to 0
    updates[0]['grad'] = 0.0
    updates[5]['grad'] = 0.0
    updates[19]['grad'] = 0.0
    _run_comparison(names, updates, simulate_perturbation=True)
    print("PASS: test_grad_zero_steps")


def test_single_worker():
    """With 1 worker (P=1, no overlap, should still match)."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(10)
    _run_comparison(names, updates, simulate_perturbation=True, num_workers=1)
    print("PASS: test_single_worker")


def test_many_workers():
    """With more workers than updates."""
    names = ['layer.0.weight']
    updates = _make_updates(3)
    _run_comparison(names, updates, simulate_perturbation=True, num_workers=16)
    print("PASS: test_many_workers")


def test_zo2_mode():
    """ZO2 mode: gradient uses prev step's seed."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(20, wd=0.01)
    _run_comparison(names, updates, simulate_perturbation=True,
                    zo2_mode=True, initial_prev_seed=99999)
    print("PASS: test_zo2_mode")


def test_zo2_no_perturbation():
    """ZO2 mode without perturbation simulation."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(15, wd=0.01)
    _run_comparison(names, updates, simulate_perturbation=False,
                    zo2_mode=True, initial_prev_seed=12345)
    print("PASS: test_zo2_no_perturbation")


def test_fp16_replay_in_fp32():
    """fp16 model with replay_in_fp32=True."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(15, wd=0.01)
    _run_comparison(names, updates, simulate_perturbation=True,
                    replay_in_fp32=True, dtype=torch.float16)
    print("PASS: test_fp16_replay_in_fp32")


def test_empty_updates():
    """Empty update list."""
    names = ['layer.0.weight']
    state_seq = _make_state(names)
    state_par = _clone_state(state_seq)
    _replay_updates_on_state(state_seq, [], trainable_param_names=names)
    _parallel_replay_updates_on_state(state_par, [], trainable_param_names=names)
    assert torch.equal(state_seq['layer.0.weight'], state_par['layer.0.weight'])
    print("PASS: test_empty_updates")


def test_rng_device_cpu():
    """rng_device='cpu' mode."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(15)
    _run_comparison(names, updates, simulate_perturbation=True, rng_device="cpu")
    print("PASS: test_rng_device_cpu")


# ===== Pipeline-specific tests =====

def test_pipeline_p2():
    """P=2 (minimum for overlap), must match sequential."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(20, wd=0.01)
    _run_comparison(names, updates, simulate_perturbation=True, num_workers=2)
    print("PASS: test_pipeline_p2")


def test_pipeline_p_equals_n():
    """P equals number of updates (all z pre-generated before any update)."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(8, wd=0.01)
    _run_comparison(names, updates, simulate_perturbation=True, num_workers=8)
    print("PASS: test_pipeline_p_equals_n")


def test_pipeline_p_exceeds_n():
    """P > n, should handle gracefully (only n producers launched)."""
    names = ['layer.0.weight']
    updates = _make_updates(3)
    _run_comparison(names, updates, simulate_perturbation=True, num_workers=100)
    print("PASS: test_pipeline_p_exceeds_n")


def test_pipeline_gpu_streams():
    """GPU mode with CUDA streams (if available)."""
    if not torch.cuda.is_available():
        print("SKIP: test_pipeline_gpu_streams (no CUDA)")
        return

    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(15, wd=0.01)

    state_seq = _make_state(names)
    state_par = _make_state(names)

    # Move to GPU
    for name in names:
        state_seq[name] = state_seq[name].cuda()
        state_par[name] = state_par[name].cuda()

    # Sequential on GPU
    _replay_updates_on_state(
        state_seq, updates, device='cuda', move_to_device=False,
        trainable_param_names=names, simulate_perturbation=True,
        rng_device='native',
    )

    # Pipelined on GPU (uses CUDA streams)
    os.environ['PARALLEL_RECOVERY_WORKERS'] = '4'
    _parallel_replay_updates_on_state(
        state_par, updates, device='cuda', move_to_device=False,
        trainable_param_names=names, simulate_perturbation=True,
        rng_device='native',
    )

    for name in names:
        if not torch.equal(state_seq[name], state_par[name]):
            diff = (state_seq[name] - state_par[name]).abs().max().item()
            raise AssertionError(f"GPU MISMATCH {name}: max_diff={diff:.2e}")

    print("PASS: test_pipeline_gpu_streams")


if __name__ == '__main__':
    # Ensure parallel dispatch is NOT used for sequential baseline
    os.environ.pop('PARALLEL_RECOVERY', None)

    tests = [
        test_basic_no_wd,
        test_basic_with_wd,
        test_mixed_params,
        test_with_perturbation_simulation,
        test_varying_lr,
        test_grad_zero_steps,
        test_single_worker,
        test_many_workers,
        test_zo2_mode,
        test_zo2_no_perturbation,
        test_fp16_replay_in_fp32,
        test_empty_updates,
        test_rng_device_cpu,
        # Pipeline-specific
        test_pipeline_p2,
        test_pipeline_p_equals_n,
        test_pipeline_p_exceeds_n,
        test_pipeline_gpu_streams,
    ]

    passed = 0
    failed = 0
    skipped = 0
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
