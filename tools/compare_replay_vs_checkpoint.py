#!/usr/bin/env python3
"""
Compare log-replay result vs a reference checkpoint, parameter by parameter.

Usage:
    python tools/compare_replay_vs_checkpoint.py \
        --replay-ckpt  <path-to-checkpoint-dir-with-log_metadata>  \
        --ref-ckpt     <path-to-reference-checkpoint>              \
        [--output-dir  <training-output-dir>]                      \
        [--model-name  <pretrained-model-name>]                    \
        [--device      cpu|cuda]                                   \
        [--top-k       10]

The script runs resume_from_log_based_bundle (pure log replay, LOG_BASED_CKPT=0)
on --replay-ckpt, loads the reference checkpoint from --ref-ckpt, then prints
per-parameter max|mean absolute difference and the overall maximum.
"""
import argparse
import os
import sys
import torch

# Ensure the repo root is importable
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from zo2.trainer.hf_transformers.log_based_resume import resume_from_log_based_bundle


def load_reference_state(path: str) -> dict:
    """Load a reference checkpoint (safetensors or .bin or directory)."""
    if os.path.isdir(path):
        safe = os.path.join(path, "model.safetensors")
        if os.path.exists(safe):
            from safetensors.torch import load_file
            return load_file(safe, device="cpu")
        bin_path = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(bin_path):
            return torch.load(bin_path, map_location="cpu", weights_only=True)
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {path}")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def main():
    parser = argparse.ArgumentParser(description="Compare log-replay vs reference checkpoint")
    parser.add_argument("--replay-ckpt", required=True, help="Checkpoint dir to replay (contains log_metadata.pt)")
    parser.add_argument("--ref-ckpt", required=True, help="Reference checkpoint path (dir or file)")
    parser.add_argument("--output-dir", default=None, help="Training output dir (parent of checkpoint dirs)")
    parser.add_argument("--model-name", default=None, help="Pretrained model name for loading initial weights")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--top-k", type=int, default=10, help="Show top-k params by max diff")
    parser.add_argument("--simulate-perturbation", type=int, default=1, help="1=simulate perturbation (default)")
    parser.add_argument("--rng-device", default="native", help="RNG device for replay")
    parser.add_argument("--zo2", action="store_true", help="Enable ZO2 mode")
    args = parser.parse_args()

    # 1. Log replay
    # Try to pre-load metadata — handle zo_log_checkpoint.pt / log_metadata.pt / optimizer.pt
    cached_opt_state = None
    ckpt_dir = args.replay_ckpt
    for meta_name in ("zo_log_checkpoint.pt", "log_metadata.pt", "optimizer.pt"):
        meta_path = os.path.join(ckpt_dir, meta_name)
        if os.path.exists(meta_path):
            cached_opt_state = torch.load(meta_path, map_location="cpu", weights_only=False)
            print(f"[1/3] Loaded metadata from: {meta_path}")
            break
    if cached_opt_state is None:
        print(f"[1/3] WARNING: No metadata found in {ckpt_dir}, resume will try other paths")
    else:
        # Force replay even if the checkpoint was saved as a full checkpoint
        cached_opt_state['is_full_checkpoint'] = False
        cached_opt_state['batch_size'] = 0
        cached_opt_state.setdefault('base_checkpoint', '__initial__')

    # Monkey-patch _replay_updates_on_state to print progress every 10 steps
    import zo2.trainer.hf_transformers.log_based_replay as _replay_mod
    import zo2.trainer.hf_transformers.log_based_resume as _resume_mod
    _orig_replay = _replay_mod._replay_updates_on_state

    def _patched_replay(state, updates, **kw):
        n = len(updates)
        print(f"      [Replay] Starting replay of {n} updates...", flush=True)
        _orig_apply = _replay_mod._apply_single_update
        _counter = [0]

        def _counting_apply(st, update, *a, **ka):
            result = _orig_apply(st, update, *a, **ka)
            _counter[0] += 1
            i = _counter[0]
            if i % 10 == 0 or i == n:
                step = update.get('step', '?')
                print(f"      [Replay] {i}/{n} updates done (step={step})", flush=True)
            return result

        _replay_mod._apply_single_update = _counting_apply
        try:
            result = _orig_replay(state, updates, **kw)
        finally:
            _replay_mod._apply_single_update = _orig_apply
        return result

    _replay_mod._replay_updates_on_state = _patched_replay
    _resume_mod._replay_updates_on_state = _patched_replay

    _total_updates = len(cached_opt_state.get('zo_update_history', [])) if cached_opt_state else '?'
    print(f"      Running log replay on: {args.replay_ckpt} ({_total_updates} updates)")
    bundle = resume_from_log_based_bundle(
        checkpoint_path=args.replay_ckpt,
        output_dir=args.output_dir,
        pretrained_model_name=args.model_name,
        device=args.device,
        simulate_perturbation=bool(args.simulate_perturbation),
        rng_device=args.rng_device,
        zo2_mode=args.zo2,
        cached_optimizer_state=cached_opt_state,
    )
    replayed = bundle.state_dict
    print(f"      Replayed to step {bundle.committed_step}, {len(replayed)} keys")

    # Move replayed to CPU for comparison
    replayed_cpu = {k: v.float().cpu() for k, v in replayed.items()}

    # 2. Load reference
    print(f"[2/3] Loading reference checkpoint: {args.ref_ckpt}")
    ref = load_reference_state(args.ref_ckpt)
    ref_cpu = {k: v.float().cpu() for k, v in ref.items()}
    print(f"      Reference has {len(ref_cpu)} keys")

    # 3. Compare
    print(f"[3/3] Comparing parameters...")
    common_keys = sorted(set(replayed_cpu.keys()) & set(ref_cpu.keys()))
    only_replay = sorted(set(replayed_cpu.keys()) - set(ref_cpu.keys()))
    only_ref = sorted(set(ref_cpu.keys()) - set(replayed_cpu.keys()))

    if only_replay:
        print(f"  WARNING: {len(only_replay)} keys only in replay: {only_replay[:5]}...")
    if only_ref:
        print(f"  WARNING: {len(only_ref)} keys only in ref: {only_ref[:5]}...")

    results = []
    global_max = 0.0
    global_max_name = ""

    for name in common_keys:
        r = replayed_cpu[name]
        c = ref_cpu[name]
        if r.shape != c.shape:
            print(f"  SHAPE MISMATCH: {name}: replay={r.shape} vs ref={c.shape}")
            continue
        diff = (r - c).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        results.append((name, max_diff, mean_diff, r.numel()))
        if max_diff > global_max:
            global_max = max_diff
            global_max_name = name

    # Sort by max_diff descending
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*80}")
    print(f"Top-{args.top_k} parameters by max absolute difference:")
    print(f"{'='*80}")
    print(f"  {'Parameter':<60s} {'MaxDiff':>18s} {'MeanDiff':>18s} {'Numel':>10s}")
    print(f"  {'-'*60} {'-'*18} {'-'*18} {'-'*10}")
    for name, mx, mn, numel in results[:args.top_k]:
        print(f"  {name:<60s} {mx:>18.12e} {mn:>18.12e} {numel:>10d}")

    n_exact = sum(1 for _, mx, _, _ in results if mx == 0.0)
    n_nonzero = len(results) - n_exact

    # Bitwise-exact check: compare original dtype tensors (not fp32-casted)
    bitwise_match = True
    for name in common_keys:
        r = replayed[name].cpu()
        c = ref[name]
        if r.dtype != c.dtype:
            c = c.to(r.dtype)
        if not torch.equal(r, c):
            bitwise_match = False
            break

    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Total common params: {len(results)}")
    print(f"  Exact match (diff=0): {n_exact}")
    print(f"  Non-zero diff: {n_nonzero}")
    print(f"  GLOBAL MAX DIFF: {global_max:.12e}  (param: {global_max_name})")
    if bitwise_match:
        print(f"  VERDICT: BITWISE EXACT MATCH")
    else:
        print(f"  VERDICT: NOT bitwise exact")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
