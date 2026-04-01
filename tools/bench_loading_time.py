#!/usr/bin/env python3
"""
Benchmark: L_disk vs L_cpu loading time.

L_disk = from_pretrained (SSD/HF cache) → GPU → first forward
L_cpu  = from_config + load_state_dict (CPU RAM) → GPU → first forward

Usage:
  # Measure both (run L_cpu first so L_disk gets cold-ish cache)
  python bench_loading_time.py --model Qwen/Qwen3-1.7B --dtype fp16

  # Evict page cache before L_disk (needs vmtouch installed)
  python bench_loading_time.py --model Qwen/Qwen3-1.7B --dtype fp16 --evict-cache

  # Only measure one
  python bench_loading_time.py --model Qwen/Qwen3-1.7B --mode disk
  python bench_loading_time.py --model Qwen/Qwen3-1.7B --mode cpu
"""
import argparse
import gc
import os
import subprocess
import time
from collections import OrderedDict

import torch


def get_dtype(name):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def get_model_class(model_name):
    if "opt" in model_name.lower():
        from transformers import OPTForCausalLM
        return OPTForCausalLM
    elif "qwen" in model_name.lower():
        from transformers import Qwen3ForCausalLM
        return Qwen3ForCausalLM
    elif "llama" in model_name.lower():
        from transformers import LlamaForCausalLM
        return LlamaForCausalLM
    else:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM


def get_dummy_input(model, device):
    """Create a minimal input for one forward pass."""
    return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=device)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evict_model_cache(model_name):
    """Evict HF model files from OS page cache using posix_fadvise (no root needed)."""
    from huggingface_hub import scan_cache_dir
    evicted_bytes = 0
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if model_name.replace("/", "--") in str(repo.repo_path):
                repo_path = str(repo.repo_path)
                for root, _dirs, files in os.walk(repo_path):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        try:
                            fd = os.open(fpath, os.O_RDONLY)
                            size = os.fstat(fd).st_size
                            os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
                            os.close(fd)
                            evicted_bytes += size
                        except OSError:
                            pass
                print(f"[cache] Evicted {evicted_bytes / 1e9:.2f} GB from page cache ({repo_path})")
                return True
    except Exception as e:
        print(f"[cache] Could not evict cache: {e}")
    return False


def copy_model_to_ssd(model_name, ssd_dir):
    """Copy entire HF hub cache dir to local SSD, return temp cache dir path."""
    import shutil

    # Find current HF cache root
    hf_cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    # Find the model's repo dir inside the cache
    model_dir_name = "models--" + model_name.replace("/", "--")
    src_path = os.path.join(hf_cache, model_dir_name)
    if not os.path.isdir(src_path):
        raise FileNotFoundError(f"Model not found at {src_path}. Run from_pretrained first.")

    ssd_cache = os.path.join(ssd_dir, f"hf_bench_{os.getpid()}", "hub")
    dst_path = os.path.join(ssd_cache, model_dir_name)
    print(f"[ssd] Copying {src_path} → {dst_path} ...")
    t0 = time.perf_counter()
    shutil.copytree(src_path, dst_path, symlinks=True)
    elapsed = time.perf_counter() - t0
    size_gb = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(dst_path)
        for f in files if not os.path.islink(os.path.join(r, f))
    ) / 1e9
    print(f"[ssd] Copied {size_gb:.2f} GB in {elapsed:.1f}s")
    # HF's cache_dir expects the directory containing "models--org--name"
    return ssd_cache


def cleanup_ssd_cache(ssd_cache):
    """Remove temp SSD cache dir."""
    import shutil
    if ssd_cache and os.path.isdir(ssd_cache):
        shutil.rmtree(ssd_cache, ignore_errors=True)
        print(f"[ssd] Cleaned up {ssd_cache}")


def measure_l_disk(model_name, dtype, device, evict=False, cache_dir=None):
    """Measure L_disk: from_pretrained → GPU → first forward."""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if evict:
        evict_model_cache(model_name)
        if cache_dir:
            # Also evict the SSD copy
            for root, _dirs, files in os.walk(cache_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    try:
                        fd = os.open(fpath, os.O_RDONLY)
                        size = os.fstat(fd).st_size
                        os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
                        os.close(fd)
                    except OSError:
                        pass

    from transformers import AutoConfig
    cls = get_model_class(model_name)

    t0 = time.perf_counter()

    # Phase 1: from_pretrained (disk → GPU)
    t_fp_start = time.perf_counter()
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    model = cls.from_pretrained(model_name, config=config, torch_dtype=dtype, cache_dir=cache_dir)
    if next(model.parameters()).device.type != device:
        model = model.to(device)
    cuda_sync()
    t_fp = time.perf_counter() - t_fp_start

    # Phase 2: first forward
    t_fwd_start = time.perf_counter()
    with torch.no_grad():
        model(get_dummy_input(model, device))
    cuda_sync()
    t_fwd = time.perf_counter() - t_fwd_start

    t_total = time.perf_counter() - t0

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "total": t_total,
        "from_pretrained": t_fp,
        "first_forward": t_fwd,
    }


def measure_l_cpu(model_name, dtype, device, cpu_state_dict):
    """Measure L_cpu: from_config + load_state_dict (from CPU RAM) → first forward."""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    from transformers import AutoConfig
    cls = get_model_class(model_name)

    t0 = time.perf_counter()

    # Phase 1: from_config (skeleton only, skip random init)
    t_cfg_start = time.perf_counter()
    config = AutoConfig.from_pretrained(model_name)
    from transformers import modeling_utils
    with modeling_utils.no_init_weights():
        model = cls(config).to(dtype=dtype, device=device)
    cuda_sync()
    t_cfg = time.perf_counter() - t_cfg_start

    # Phase 2: load_state_dict from CPU RAM
    t_load_start = time.perf_counter()
    model.load_state_dict(cpu_state_dict, strict=False)
    cuda_sync()
    t_load = time.perf_counter() - t_load_start

    # Phase 3: first forward
    t_fwd_start = time.perf_counter()
    with torch.no_grad():
        model(get_dummy_input(model, device))
    cuda_sync()
    t_fwd = time.perf_counter() - t_fwd_start

    t_total = time.perf_counter() - t0

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "total": t_total,
        "from_config": t_cfg,
        "load_state_dict": t_load,
        "first_forward": t_fwd,
    }


def print_result(label, result):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for k, v in result.items():
        print(f"  {k:25s} = {v:.3f}s")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark L_disk vs L_cpu loading time")
    parser.add_argument("--model", type=str, required=True, help="HF model name (e.g. Qwen/Qwen3-1.7B)")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "disk", "cpu"])
    parser.add_argument("--evict-cache", action="store_true", default=True,
                        help="Evict page cache before L_disk (default: on)")
    parser.add_argument("--no-evict-cache", dest="evict_cache", action="store_false",
                        help="Skip page cache eviction (measure warm-cache L_disk)")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each measurement N times")
    parser.add_argument("--ssd-dir", type=str, default=None,
                        help="Copy model to local SSD before L_disk benchmark, then clean up. "
                             "E.g. --ssd-dir /tmp")
    args = parser.parse_args()

    dtype = get_dtype(args.dtype)
    print(f"Model: {args.model}")
    print(f"Dtype: {args.dtype}")
    print(f"Device: {args.device}")

    # CUDA warmup
    if "cuda" in args.device:
        print("CUDA warmup...")
        _ = torch.zeros(1, device=args.device)
        torch.cuda.synchronize()

    # ============================================================
    # Order matters for fairness:
    #   1. L_disk FIRST (page cache is cold from previous workload)
    #   2. Use the loaded model to build cpu_state (free side-effect)
    #   3. L_cpu SECOND (reads from CPU RAM, not disk)
    # ============================================================

    cpu_state = None
    result_disk = None
    result_cpu = None
    ssd_cache = None

    # Copy model to local SSD if requested
    if args.ssd_dir and args.mode in ("both", "disk"):
        ssd_cache = copy_model_to_ssd(args.model, args.ssd_dir)

    try:
        # --- Step 1: Measure L_disk (cold cache) ---
        if args.mode in ("both", "disk"):
            result_disk = measure_l_disk(
                args.model, dtype, args.device,
                evict=args.evict_cache, cache_dir=ssd_cache,
            )
            source = f"local SSD ({args.ssd_dir})" if ssd_cache else "HF cache"
            print_result(f"L_disk: from_pretrained ({source} → GPU)", result_disk)

        # --- Step 2: Prepare CPU state_dict ---
        if args.mode in ("both", "cpu"):
            print("Preparing CPU state_dict for L_cpu...")
            from transformers import AutoConfig
            cls = get_model_class(args.model)
            config = AutoConfig.from_pretrained(args.model)
            tmp = cls.from_pretrained(args.model, config=config, torch_dtype=dtype)
            cpu_state = OrderedDict(
                (k, v.cpu().clone()) for k, v in tmp.state_dict().items()
            )
            state_mb = sum(v.numel() * v.element_size() for v in cpu_state.values()) / 1e6
            print(f"CPU state_dict ready: {len(cpu_state)} tensors, {state_mb:.1f} MB")
            del tmp
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # --- Step 3: Measure L_cpu (from CPU RAM) ---
        if args.mode in ("both", "cpu"):
            result_cpu = measure_l_cpu(args.model, dtype, args.device, cpu_state)
            print_result("L_cpu: from_config + load_state_dict (CPU RAM → GPU)", result_cpu)

        # --- Repeat (both warm cache now, for variance measurement) ---
        for i in range(1, args.repeat):
            tag = f" (run {i+1}/{args.repeat})"
            if args.mode in ("both", "disk"):
                result_disk = measure_l_disk(
                    args.model, dtype, args.device,
                    evict=args.evict_cache, cache_dir=ssd_cache,
                )
                print_result(f"L_disk{tag}", result_disk)
            if args.mode in ("both", "cpu"):
                result_cpu = measure_l_cpu(args.model, dtype, args.device, cpu_state)
                print_result(f"L_cpu{tag}", result_cpu)

        # Summary
        if args.mode == "both" and result_disk and result_cpu:
            diff = result_disk['total'] - result_cpu['total']
            print(f"\n>>> L_disk = {result_disk['total']:.3f}s")
            print(f">>> L_cpu  = {result_cpu['total']:.3f}s")
            print(f">>> L_disk - L_cpu = {diff:.3f}s")
    finally:
        if ssd_cache:
            cleanup_ssd_cache(ssd_cache)


if __name__ == "__main__":
    main()
