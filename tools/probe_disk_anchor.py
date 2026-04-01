#!/usr/bin/env python3
"""
Probe: measure disk persist time vs shadow commit interval.

Two modes:
  --shm_dir  : measure with existing flat files from a real training run
  --model    : generate dummy flat files matching a HuggingFace model size

Usage (Jupyter cell, bash):
    # Mode 1: use real flat files + trace
    !python tools/probe_disk_anchor.py \
        --shm_dir /dev/shm/zo_ckpt \
        --trace_jsonl <output_dir>/zo_trace.jsonl

    # Mode 2: dummy files for a specific model (no training needed)
    !python tools/probe_disk_anchor.py --model Qwen/Qwen3-1.7B
    !python tools/probe_disk_anchor.py --model facebook/opt-1.3b
    !python tools/probe_disk_anchor.py --model Qwen/Qwen3-8B --dtype fp16
"""

import argparse
import glob
import json
import os
import shutil
import statistics
import time


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_flat_files(shm_dir):
    """Find flat files, separated into model-only vs adam groups."""
    model_files = sorted(glob.glob(os.path.join(shm_dir, "*.flat.bin")))
    adam_files = sorted(
        glob.glob(os.path.join(shm_dir, "*.flat.adam_m.bin"))
        + glob.glob(os.path.join(shm_dir, "*.flat.adam_v.bin"))
    )
    meta_files = sorted(
        glob.glob(os.path.join(shm_dir, "*.flat.header.json"))
        + glob.glob(os.path.join(shm_dir, "*.flat.state.meta.json"))
        + glob.glob(os.path.join(shm_dir, "*.flat.adam.meta.json"))
        + glob.glob(os.path.join(shm_dir, "*.flat.generation.meta.json"))
    )
    return model_files, adam_files, meta_files


# ---------------------------------------------------------------------------
# Dummy flat file generation from HF model config
# ---------------------------------------------------------------------------

def _model_param_bytes(model_name, dtype):
    """Get total parameter bytes from HF model config (no download needed)."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    num_params = getattr(config, "num_parameters", None)
    if num_params is None:
        hidden = getattr(config, "hidden_size", None)
        layers = getattr(config, "num_hidden_layers", None)
        vocab = getattr(config, "vocab_size", None)
        intermediate = getattr(config, "intermediate_size", None)
        if hidden and layers and vocab:
            if intermediate is None:
                intermediate = 4 * hidden
            per_layer = (
                4 * hidden * hidden
                + 3 * hidden * intermediate
            )
            num_params = vocab * hidden + layers * per_layer + vocab * hidden
            print(f"  Estimated from config: {num_params / 1e9:.2f}B params")
        else:
            raise ValueError(f"Cannot estimate params for {model_name}, "
                             f"hidden={hidden} layers={layers} vocab={vocab}")
    else:
        print(f"  Config reports: {num_params / 1e9:.2f}B params")

    elem_size = 2 if dtype in ("fp16", "bf16") else 4
    return int(num_params) * elem_size, int(num_params), elem_size


def create_dummy_flat_files(tmpfs_dir, model_bytes, adam_elem_count):
    """Create dummy flat files on tmpfs with realistic sizes."""
    os.makedirs(tmpfs_dir, exist_ok=True)
    prefix = os.path.join(tmpfs_dir, "dummy_probe")

    files_created = {}
    chunk = 64 * 1024 * 1024

    model_path = prefix + ".flat.bin"
    with open(model_path, "wb") as f:
        remaining = model_bytes
        while remaining > 0:
            f.write(b"\x00" * min(chunk, remaining))
            remaining -= chunk
    files_created["model"] = [model_path]

    adam_bytes = adam_elem_count * 4
    adam_files = []
    for suffix in (".flat.adam_m.bin", ".flat.adam_v.bin"):
        path = prefix + suffix
        with open(path, "wb") as f:
            remaining = adam_bytes
            while remaining > 0:
                f.write(b"\x00" * min(chunk, remaining))
                remaining -= chunk
        adam_files.append(path)
    files_created["adam"] = adam_files

    meta_files = []
    for suffix in (".flat.header.json", ".flat.state.meta.json",
                   ".flat.adam.meta.json", ".flat.generation.meta.json"):
        path = prefix + suffix
        with open(path, "w") as f:
            json.dump({"probe": True}, f)
        meta_files.append(path)
    files_created["meta"] = meta_files

    return files_created


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _bytes_label(n):
    return f"{n / 1024**3:.2f} GB" if n >= 1024**3 else f"{n / 1024**2:.1f} MB"


def _run_copy(files, disk_dir, repeats):
    """Copy files with fsync, return list of elapsed ms."""
    total_bytes = sum(os.path.getsize(f) for f in files)
    times = []
    for i in range(repeats):
        for f in files:
            dst = os.path.join(disk_dir, os.path.basename(f))
            if os.path.exists(dst):
                os.remove(dst)

        t0 = time.perf_counter()
        for f in files:
            dst = os.path.join(disk_dir, os.path.basename(f))
            shutil.copy2(f, dst)
            fd = os.open(dst, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        bw = total_bytes / (elapsed_ms / 1000) / 1024**3 if elapsed_ms > 0 else 0
        print(f"    run {i+1}: {elapsed_ms:.0f} ms  ({bw:.2f} GB/s)")
        times.append(elapsed_ms)
    return times, total_bytes


def measure_persist(model_files, adam_files, meta_files, disk_dir, repeats=3, optimizer="both"):
    """Measure persist time for model-only and/or model+adam."""
    os.makedirs(disk_dir, exist_ok=True)

    all_files = model_files + (adam_files if optimizer != "sgd" else []) + meta_files
    print("Files:")
    for f in all_files:
        print(f"  {os.path.basename(f):55s} {_bytes_label(os.path.getsize(f))}")

    model_bytes = sum(os.path.getsize(f) for f in model_files)
    adam_bytes = sum(os.path.getsize(f) for f in adam_files)
    meta_bytes = sum(os.path.getsize(f) for f in meta_files)
    print(f"  model: {_bytes_label(model_bytes)}  adam: {_bytes_label(adam_bytes)}  "
          f"meta: {_bytes_label(meta_bytes)}  total: {_bytes_label(model_bytes + adam_bytes + meta_bytes)}")
    print()

    results = {}

    if optimizer in ("sgd", "both"):
        print("  [model only (SGD)]:")
        t, _ = _run_copy(model_files + meta_files, disk_dir, repeats)
        results["model"] = t

    if optimizer in ("adam", "both") and adam_files:
        print("  [model+adam (Adam)]:")
        t, _ = _run_copy(model_files + adam_files + meta_files, disk_dir, repeats)
        results["model_adam"] = t

    return results


# ---------------------------------------------------------------------------
# Trace analysis
# ---------------------------------------------------------------------------

def extract_trace_stats(trace_path):
    """Extract shadow_wait_update, shadow_commit, shadow_apply durations."""
    open_tokens = {}
    durations = {}
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            phase = r.get("phase")
            eid = r.get("event_id")
            event = r.get("event", "")
            if phase == "B":
                open_tokens[eid] = r
            elif phase == "E" and eid in open_tokens:
                begin = open_tokens.pop(eid)
                ms = r.get("duration_ms")
                if ms is None:
                    ms = (r["wall_time_ns"] - begin["wall_time_ns"]) / 1e6
                durations.setdefault(event, []).append(float(ms))
    return durations


def print_stats(name, values_ms):
    if not values_ms:
        print(f"  {name}: no data")
        return
    avg = statistics.mean(values_ms)
    mx = max(values_ms)
    mn = min(values_ms)
    print(f"  {name}: n={len(values_ms)}  avg={avg:.0f}ms  min={mn:.0f}ms  max={mx:.0f}ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--shm_dir", default=None,
                   help="tmpfs directory with existing flat files")
    g.add_argument("--model", default=None,
                   help="HuggingFace model name (generates dummy flat files)")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"],
                        help="model dtype for dummy mode (default: fp16)")
    parser.add_argument("--disk_dir", default="/tmp/probe_disk_anchor",
                        help="disk destination for persist test")
    parser.add_argument("--trace_jsonl", default="",
                        help="path to zo_trace.jsonl for idle window stats")
    parser.add_argument("--optimizer", default="both", choices=["sgd", "adam", "both"],
                        help="which optimizer layout to test: sgd (model only), adam (model+adam), both (default)")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    dummy_tmpfs_dir = None

    if args.model:
        print("=" * 60)
        print(f"Generating dummy flat files for: {args.model} ({args.dtype})")
        print("=" * 60)
        model_bytes, num_params, elem_size = _model_param_bytes(args.model, args.dtype)
        print(f"  Model state: {_bytes_label(model_bytes)}")
        print(f"  Adam m + v:  {_bytes_label(num_params * 4 * 2)} (fp32)")
        print(f"  Total:       {_bytes_label(model_bytes + num_params * 4 * 2)}")
        print()

        dummy_tmpfs_dir = "/dev/shm/probe_disk_anchor_dummy"
        print(f"  Writing dummy files to {dummy_tmpfs_dir} ...")
        created = create_dummy_flat_files(dummy_tmpfs_dir, model_bytes, num_params)
        model_files = created["model"]
        adam_files = created["adam"]
        meta_files = created["meta"]
        print("  Done.")
        print()
    else:
        model_files, adam_files, meta_files = find_flat_files(args.shm_dir)
        if not model_files:
            print(f"No flat files found in {args.shm_dir}")
            print("Run training with ENABLE_SHADOW=1 SHADOW_FLAT_COMMIT=1 first,")
            print("or use --model <hf_model_name> to generate dummy files.")
            return

    try:
        print("=" * 60)
        print("1. Disk persist time (tmpfs → disk, with fsync)")
        print("=" * 60)
        results = measure_persist(model_files, adam_files, meta_files,
                                  args.disk_dir, repeats=args.repeats,
                                  optimizer=args.optimizer)

        print()
        print("-" * 60)
        print("Persist summary:")
        for key, vals in results.items():
            print_stats(key, vals)

        # --- Trace stats ---
        if args.trace_jsonl and os.path.exists(args.trace_jsonl):
            print()
            print("=" * 60)
            print("2. Shadow timing from trace")
            print("=" * 60)
            durations = extract_trace_stats(args.trace_jsonl)

            print_stats("shadow_wait_update (idle window)", durations.get("shadow_wait_update"))
            print_stats("shadow_apply", durations.get("shadow_apply"))
            print_stats("shadow_commit", durations.get("shadow_commit"))

            idle = durations.get("shadow_wait_update", [])
            apply_d = durations.get("shadow_apply", [])
            commit_d = durations.get("shadow_commit", [])

            if idle:
                idle_avg = statistics.mean(idle)
                apply_avg = statistics.mean(apply_d) if apply_d else 0
                commit_avg = statistics.mean(commit_d) if commit_d else 0
                interval_avg = apply_avg + commit_avg + idle_avg

                print()
                print("=" * 60)
                print("3. Feasibility (can persist finish before next commit?)")
                print("=" * 60)
                print(f"  shadow cycle:  apply {apply_avg:.0f}ms + commit {commit_avg:.0f}ms "
                      f"+ wait {idle_avg:.0f}ms = {interval_avg:.0f}ms")
                print()

                for key, vals in results.items():
                    avg = statistics.mean(vals)
                    margin = interval_avg - avg
                    status = "OK" if margin > 0 else "TOO SLOW"
                    print(f"  [{key:20s}] {avg:8.0f}ms  vs  interval {interval_avg:.0f}ms  "
                          f"→ margin {margin:+.0f}ms  [{status}]")
        elif not args.trace_jsonl:
            print()
            print("Tip: add --trace_jsonl <output_dir>/zo_trace.jsonl to compare with shadow timing.")

    finally:
        if os.path.exists(args.disk_dir):
            shutil.rmtree(args.disk_dir)
            print(f"\nCleaned up {args.disk_dir}")
        if dummy_tmpfs_dir and os.path.exists(dummy_tmpfs_dir):
            shutil.rmtree(dummy_tmpfs_dir)
            print(f"Cleaned up {dummy_tmpfs_dir}")


if __name__ == "__main__":
    main()
