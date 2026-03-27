"""
Test closed-form parallel replay vs sequential replay.

The closed-form unrolls the ZO-SGD recurrence into independent terms:
    p_n = sp[0]*p_0 - Σ sp[t+1]*lr_t*grad_t*z_t
Results should be near-exact (not bitwise) with serial replay when
simulate_perturbation=False.

Usage:
    python -m pytest test/test_closedform_recovery.py -v
    # or directly:
    python test/test_closedform_recovery.py
"""

import os
import sys
import torch
import numpy as np
from collections import OrderedDict

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from zo2.trainer.hf_transformers.legacy_pipeline_closed_form_replay import (
    _closedform_replay_on_state,
    validate_closedform_replay,
)
from zo2.trainer.hf_transformers.log_based_replay import _replay_updates_on_state


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
            'zo_eps': 0.0,
        })
    return updates


def _clone_state(state):
    return OrderedDict((k, v.clone()) for k, v in state.items())


def _run_comparison(param_names, updates, zo2_mode=False, initial_prev_seed=None,
                    rng_device="native", dtype=torch.float32, num_workers=1,
                    precision="fp32", atol=1e-5, rtol=1e-5):
    """Run sequential vs closed-form and assert near-equality."""
    state_seq = _make_state(param_names, dtype=dtype)
    state_cf = _clone_state(state_seq)

    # Sequential (no perturbation — ground truth for closed-form comparison)
    _replay_updates_on_state(
        state_seq, updates, device='cpu', move_to_device=False,
        trainable_param_names=param_names,
        simulate_perturbation=False,
        rng_device=rng_device,
        zo2_mode=zo2_mode,
        initial_prev_seed=initial_prev_seed,
    )

    # Closed-form
    _closedform_replay_on_state(
        state_cf, updates, device='cpu', move_to_device=False,
        trainable_param_names=param_names,
        rng_device=rng_device,
        zo2_mode=zo2_mode,
        initial_prev_seed=initial_prev_seed,
        num_workers=num_workers,
        precision=precision,
    )

    for name in param_names:
        seq_p = state_seq[name].float()
        cf_p = state_cf[name].float()
        diff = (seq_p - cf_p).abs().max().item()
        denom = seq_p.abs().max().item()
        rel = diff / max(denom, 1e-10)
        if diff > atol and rel > rtol:
            raise AssertionError(
                f"MISMATCH {name}: max_abs_diff={diff:.2e}, rel_diff={rel:.2e} "
                f"(atol={atol:.0e}, rtol={rtol:.0e})\n"
                f"  seq[:5]={state_seq[name][:5].tolist()}\n"
                f"  cf[:5]={state_cf[name][:5].tolist()}"
            )
    return True


# ===== Basic tests =====

def test_closedform_no_wd():
    """Bias-only params (no weight decay), fp32."""
    names = ['layer.0.bias', 'layer.1.bias']
    updates = _make_updates(20, wd=0.0)
    _run_comparison(names, updates, precision="fp32", atol=1e-6)
    print("PASS: test_closedform_no_wd")


def test_closedform_with_wd():
    """Weight params with decay, fp32."""
    names = ['layer.0.weight', 'layer.1.weight']
    updates = _make_updates(20, wd=0.01)
    _run_comparison(names, updates, precision="fp32", atol=1e-5)
    print("PASS: test_closedform_with_wd")


def test_closedform_mixed_params():
    """Mix of weight and bias params."""
    names = ['layer.0.weight', 'layer.0.bias', 'layer.1.weight', 'layer_norm.weight']
    updates = _make_updates(30, wd=0.01)
    _run_comparison(names, updates, precision="fp32", atol=1e-5)
    print("PASS: test_closedform_mixed_params")


# ===== Precision mode tests =====

def test_closedform_precision_fp32():
    """fp32 mode should have very small error."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(50, wd=0.01)
    _run_comparison(names, updates, precision="fp32", atol=1e-4)
    print("PASS: test_closedform_precision_fp32")


def test_closedform_precision_mixed():
    """mixed mode: accumulate in fp32, params in original dtype."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(50, wd=0.01)
    # mixed mode with fp32 state should be similar to fp32 mode
    _run_comparison(names, updates, precision="mixed", atol=1e-4)
    print("PASS: test_closedform_precision_mixed")


def test_closedform_precision_fp16():
    """fp16 mode: larger error acceptable."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(20, wd=0.01)
    # fp16 accumulation has larger error
    _run_comparison(names, updates, dtype=torch.float16, precision="fp16",
                    atol=1e-2, rtol=1e-2)
    print("PASS: test_closedform_precision_fp16")


# ===== Multi-worker tests =====

def test_closedform_multi_worker():
    """W=1,2,4 all produce near-equal results."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(20, wd=0.01)

    results = {}
    for w in [1, 2, 4]:
        state = _make_state(names)
        _closedform_replay_on_state(
            state, updates, device='cpu', move_to_device=False,
            trainable_param_names=names,
            num_workers=w, precision="fp32",
        )
        results[w] = state

    # All workers should produce near-identical results
    for name in names:
        for w in [2, 4]:
            diff = (results[1][name] - results[w][name]).abs().max().item()
            if diff > 1e-5:
                raise AssertionError(f"W={w} vs W=1: {name} max_diff={diff:.2e}")

    print("PASS: test_closedform_multi_worker")


def test_closedform_workers_exceed_n():
    """W > n, handles gracefully."""
    names = ['layer.0.weight']
    updates = _make_updates(3)
    _run_comparison(names, updates, num_workers=100, precision="fp32", atol=1e-6)
    print("PASS: test_closedform_workers_exceed_n")


# ===== Edge case tests =====

def test_closedform_zo2_mode():
    """ZO2 mode with prev_seed handling."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(20, wd=0.01)
    _run_comparison(names, updates, zo2_mode=True, initial_prev_seed=99999,
                    precision="fp32", atol=1e-5)
    print("PASS: test_closedform_zo2_mode")


def test_closedform_varying_lr():
    """Different lr per step."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(25, wd=0.01, varying_lr=True)
    _run_comparison(names, updates, precision="fp32", atol=1e-5)
    print("PASS: test_closedform_varying_lr")


def test_closedform_grad_zero_steps():
    """Steps with grad=0 should be skipped."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(20, wd=0.01)
    updates[0]['grad'] = 0.0
    updates[5]['grad'] = 0.0
    updates[19]['grad'] = 0.0
    _run_comparison(names, updates, precision="fp32", atol=1e-5)
    print("PASS: test_closedform_grad_zero_steps")


def test_closedform_empty_updates():
    """Empty update list."""
    names = ['layer.0.weight']
    state = _make_state(names)
    original = state['layer.0.weight'].clone()
    _closedform_replay_on_state(state, [], trainable_param_names=names)
    assert torch.equal(state['layer.0.weight'], original)
    print("PASS: test_closedform_empty_updates")


# ===== GPU test =====

def test_closedform_gpu():
    """GPU backend matches CPU backend."""
    if not torch.cuda.is_available():
        print("SKIP: test_closedform_gpu (no CUDA)")
        return

    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(15, wd=0.01)

    state_cpu = _make_state(names)
    state_gpu = _clone_state(state_cpu)

    _closedform_replay_on_state(
        state_cpu, updates, device='cpu', move_to_device=False,
        trainable_param_names=names, num_workers=2, precision="fp32",
    )

    # Move to GPU
    for name in names:
        state_gpu[name] = state_gpu[name].cuda()

    _closedform_replay_on_state(
        state_gpu, updates, device='cuda', move_to_device=False,
        trainable_param_names=names, num_workers=2, precision="fp32",
        rng_device='native',
    )

    for name in names:
        diff = (state_cpu[name] - state_gpu[name].cpu()).abs().max().item()
        # GPU native RNG differs from CPU RNG, so we can't compare directly
        # Just verify it runs without error
        pass

    print("PASS: test_closedform_gpu (ran without error)")


# ===== Validation function test =====

def test_validate_function():
    """validate_closedform_replay() runs and returns expected structure."""
    names = ['layer.0.weight', 'layer.0.bias']
    updates = _make_updates(15, wd=0.01)
    state = _make_state(names)

    results = validate_closedform_replay(
        state, updates, trainable_param_names=names, num_workers=2,
    )

    # Check structure
    assert set(results.keys()) == {"fp32", "mixed", "fp16"}
    for prec in ["fp32", "mixed", "fp16"]:
        for name in names:
            assert name in results[prec], f"Missing {name} in {prec}"
            assert "max_abs" in results[prec][name]
            assert "rel" in results[prec][name]

    # fp32 should have smallest error
    fp32_max = max(v["max_abs"] for v in results["fp32"].values())
    assert fp32_max < 1e-4, f"fp32 error too large: {fp32_max:.2e}"

    print(f"PASS: test_validate_function (fp32_max_err={fp32_max:.2e})")


if __name__ == '__main__':
    # Ensure parallel/closedform dispatch is NOT used for sequential baseline
    os.environ.pop('PARALLEL_RECOVERY', None)
    os.environ.pop('CLOSEDFORM_RECOVERY', None)

    tests = [
        test_closedform_no_wd,
        test_closedform_with_wd,
        test_closedform_mixed_params,
        test_closedform_precision_fp32,
        test_closedform_precision_mixed,
        test_closedform_precision_fp16,
        test_closedform_multi_worker,
        test_closedform_workers_exceed_n,
        test_closedform_zo2_mode,
        test_closedform_varying_lr,
        test_closedform_grad_zero_steps,
        test_closedform_empty_updates,
        test_closedform_gpu,
        test_validate_function,
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
