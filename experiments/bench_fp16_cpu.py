#!/usr/bin/env python3
"""
Benchmark: why does CPU replay take 42s for 1 update on Qwen3-1.7B?
Hypothesis: fp16 torch.normal() and arithmetic on CPU is the bottleneck.

This script isolates each operation and measures it in fp16 vs fp32,
on CPU vs CUDA, to pinpoint the exact source of slowdown.
"""

import torch
import time
import sys

# Match Qwen3-1.7B: 310 trainable params, total 1.72B elements, fp16
# We use a few representative tensor sizes from the actual model
PARAM_SHAPES = {
    "embed_tokens": (151936, 2048),   # ~311M elements
    "q_proj": (2048, 2048),           # typical attention layer
    "k_proj": (512, 2048),
    "v_proj": (512, 2048),
    "o_proj": (2048, 2048),
    "gate_proj": (5632, 2048),        # MLP
    "up_proj": (5632, 2048),
    "down_proj": (2048, 5632),
    "lm_head": (151936, 2048),        # ~311M elements (tied with embed)
}

def make_params(dtype, device, scale=1):
    """Create a dict of parameter tensors mimicking Qwen3-1.7B structure.
    scale=1 for quick test, scale=10 to approximate full 1.7B params."""
    params = {}
    for name, shape in PARAM_SHAPES.items():
        params[name] = torch.randn(shape, dtype=dtype, device=device)
    # Duplicate attention/MLP layers to approximate full model size
    for i in range(scale):
        for name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            params[f"layers.{i}.{name}"] = torch.randn(
                PARAM_SHAPES[name], dtype=dtype, device=device
            )
    total = sum(p.numel() for p in params.values())
    size_mb = sum(p.numel() * p.element_size() for p in params.values()) / 1e6
    return params, total, size_mb


def bench_torch_normal(shape, dtype, device, repeats=3):
    """Benchmark torch.normal for a single tensor."""
    # warmup
    for _ in range(2):
        torch.manual_seed(42)
        z = torch.normal(mean=0, std=1, size=shape, dtype=dtype, device=device)
        if device == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.manual_seed(42)
        z = torch.normal(mean=0, std=1, size=shape, dtype=dtype, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


def bench_single_update(params, dtype, device, simulate_perturbation=False):
    """Benchmark one full _apply_single_update equivalent."""
    seed = 209652396
    grad = -47.96875
    lr = 1e-7
    zo_eps = 0.001
    param_names = list(params.keys())

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    if simulate_perturbation and zo_eps > 0:
        for scaling_factor in [1, -2, 1]:
            torch.manual_seed(seed)
            for name in param_names:
                p = params[name]
                z = torch.normal(mean=0, std=1, size=p.size(), dtype=p.dtype, device=p.device)
                p.data.add_(scaling_factor * z * zo_eps)

    torch.manual_seed(seed)
    for name in param_names:
        p = params[name]
        z = torch.normal(mean=0, std=1, size=p.size(), dtype=p.dtype, device=p.device)
        p.sub_(lr * grad * z)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed


def main():
    print("=" * 70)
    print("Benchmark: fp16 vs fp32 on CPU vs CUDA")
    print("=" * 70)

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    # ─── Test 1: torch.normal() for a single large tensor ───
    print("─── Test 1: torch.normal() single tensor (151936 x 2048 = 311M elements) ───")
    shape = (151936, 2048)
    for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
        t = bench_torch_normal(shape, dtype, "cpu")
        print(f"  CPU  {dtype_name}: {t:.4f}s")
    if has_cuda:
        for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
            t = bench_torch_normal(shape, dtype, "cuda")
            print(f"  CUDA {dtype_name}: {t:.4f}s")
    print()

    # ─── Test 2: torch.normal() for small tensor ───
    print("─── Test 2: torch.normal() small tensor (2048 x 2048 = 4M elements) ───")
    shape_small = (2048, 2048)
    for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
        t = bench_torch_normal(shape_small, dtype, "cpu")
        print(f"  CPU  {dtype_name}: {t:.6f}s")
    if has_cuda:
        for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
            t = bench_torch_normal(shape_small, dtype, "cuda")
            print(f"  CUDA {dtype_name}: {t:.6f}s")
    print()

    # ─── Test 3: Full update on representative param set ───
    # scale=10 gives roughly 28 attention/MLP layers ≈ Qwen3-1.7B
    print("─── Test 3: Full _apply_single_update (scale=10, ~0.9B elements) ───")
    print("  (Qwen3-1.7B has 1.72B elements; this is roughly half)")
    print()
    for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
        params, total, size_mb = make_params(dtype, "cpu", scale=10)
        print(f"  [{dtype_name}] {total/1e6:.1f}M params, {size_mb:.1f} MB")

        # Without perturbation (matching your 142147 experiment)
        t = bench_single_update(params, dtype, "cpu", simulate_perturbation=False)
        print(f"    CPU  no_perturb:   {t:.3f}s")

        t = bench_single_update(params, dtype, "cpu", simulate_perturbation=True)
        print(f"    CPU  with_perturb: {t:.3f}s")

        del params
        print()

    if has_cuda:
        print("  --- Same on CUDA ---")
        for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
            params, total, size_mb = make_params(dtype, "cuda", scale=10)
            print(f"  [{dtype_name}] {total/1e6:.1f}M params, {size_mb:.1f} MB")

            t = bench_single_update(params, dtype, "cuda", simulate_perturbation=False)
            print(f"    CUDA no_perturb:   {t:.4f}s")

            t = bench_single_update(params, dtype, "cuda", simulate_perturbation=True)
            print(f"    CUDA with_perturb: {t:.4f}s")

            del params
            torch.cuda.empty_cache()
            print()

    # ─── Test 4: Isolate torch.normal vs arithmetic ───
    print("─── Test 4: Isolate bottleneck — torch.normal vs add_ ───")
    shape = (151936, 2048)  # 311M elements (largest single param)
    for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
        # Pure torch.normal
        times_normal = []
        for _ in range(3):
            torch.manual_seed(42)
            t0 = time.perf_counter()
            z = torch.normal(mean=0, std=1, size=shape, dtype=dtype, device="cpu")
            times_normal.append(time.perf_counter() - t0)
        t_normal = min(times_normal)

        # Pure arithmetic (add_) with pre-generated z
        p = torch.randn(shape, dtype=dtype, device="cpu")
        z = torch.randn(shape, dtype=dtype, device="cpu")
        times_add = []
        for _ in range(3):
            t0 = time.perf_counter()
            p.add_(1e-7 * z)
            times_add.append(time.perf_counter() - t0)
        t_add = min(times_add)

        # Pure mul
        times_mul = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = 1e-7 * 47.0 * z
            times_mul.append(time.perf_counter() - t0)
        t_mul = min(times_mul)

        print(f"  CPU {dtype_name} (311M elements):")
        print(f"    torch.normal:  {t_normal:.4f}s")
        print(f"    add_:          {t_add:.4f}s")
        print(f"    scalar * z:    {t_mul:.4f}s")
        del p, z
    print()

    # ─── Test 5: The fix — fp32 replay then cast back ───
    print("─── Test 5: Proposed fix — replay in fp32 on CPU, then .half() ───")
    params_fp16, total, size_mb = make_params(torch.float16, "cpu", scale=10)
    print(f"  Original: {total/1e6:.1f}M params, {size_mb:.1f} MB (fp16)")

    # Simulate: cast to fp32, replay, cast back
    t0 = time.perf_counter()
    params_fp32 = {k: v.float() for k, v in params_fp16.items()}
    t_cast_up = time.perf_counter() - t0

    t_replay = bench_single_update(params_fp32, torch.float32, "cpu", simulate_perturbation=False)

    t0 = time.perf_counter()
    params_back = {k: v.half() for k, v in params_fp32.items()}
    t_cast_down = time.perf_counter() - t0

    t_total = t_cast_up + t_replay + t_cast_down
    print(f"  fp16→fp32 cast:  {t_cast_up:.3f}s")
    print(f"  fp32 replay:     {t_replay:.3f}s")
    print(f"  fp32→fp16 cast:  {t_cast_down:.3f}s")
    print(f"  Total:           {t_total:.3f}s")
    print()

    # Compare with direct fp16 replay
    t_fp16 = bench_single_update(params_fp16, torch.float16, "cpu", simulate_perturbation=False)
    print(f"  Direct fp16 replay: {t_fp16:.3f}s")
    print(f"  Speedup:            {t_fp16 / t_total:.1f}x")
    print()

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("If fp16 torch.normal is >> fp32 torch.normal on CPU,")
    print("then the 42s bottleneck is confirmed as fp16-on-CPU penalty.")
    print("Fix: cast to fp32 before replay, cast back after.")


if __name__ == "__main__":
    main()
