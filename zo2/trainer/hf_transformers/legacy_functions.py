"""
LEGACY / DISCARDED code kept only for reference.

This module contains:
  1. Deprecated pipeline replay and closed-form replay implementations.
  2. Async anchor checkpoint (GPU→pinned CPU→tmpfs→disk two-phase pipeline).
  3. Shadow rebase infrastructure (rebase payload flat read/write, retained
     updates replay, rebase command handlers).

They are intentionally isolated so routine maintenance and future coding
agents do not need to read them when working on the active checkpoint/replay
path.
"""

import hashlib
import logging
import mmap
import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import psutil
import torch

from ...utils.logging_controls import replay_step_time_log_enabled, resource_log_enabled, time_log_enabled
from ...utils.trace import directory_size_bytes, trace_begin, trace_end, trace_end_external, trace_instant
from .log_based_utils import (
    _atomic_save_state_dict_safetensors,
    _ensure_zo_shm_dir,
    _log_memory,
    _tie_state_dict_inplace,
)
from .log_based_shadow import (
    _build_adam_flat_layout,
    _build_shadow_flat_layout,
    _close_shadow_flat_views,
    _copy_shadow_flat_views_from_adam,
    _copy_shadow_flat_views_from_state,
    _deserialize_flat_layout,
    _ensure_shadow_flat_files,
    _load_shadow_bundle_flat,
    _load_shadow_replica,
    _open_shadow_flat_views,
    _read_shadow_flat_header,
    _rebase_payload_paths,
    _secondary_tied_keys,
    _serialize_flat_layout,
    _write_shadow_flat_header,
)

logger = logging.getLogger(__name__)


def _pipelined_replay_cpu(state, updates, param_names, rng_device,
                          num_producers, default_zo_eps, simulate_perturbation,
                          zo2_mode, seeds_info, replay_dtype):
    from . import log_based_replay as _replay

    P = num_producers
    n = len(updates)
    buffer = [None] * P
    ready = [threading.Event() for _ in range(P)]
    free = [threading.Event() for _ in range(P)]
    for e in free:
        e.set()

    error_holder = [None]

    def producer(slot_id):
        try:
            step = slot_id
            while step < n:
                free[slot_id].wait()
                free[slot_id].clear()
                grad_seed, perturb_seed = seeds_info[step]
                z = _replay._generate_z_for_one_step(grad_seed, param_names, state, rng_device, replay_dtype)
                z_perturb = None
                if zo2_mode and perturb_seed != grad_seed:
                    z_perturb = _replay._generate_z_for_one_step(
                        perturb_seed, param_names, state, rng_device, replay_dtype
                    )
                buffer[slot_id] = (z, z_perturb)
                ready[slot_id].set()
                step += P
        except Exception as e:
            error_holder[0] = e
            ready[slot_id].set()

    threads = []
    for i in range(min(P, n)):
        t = threading.Thread(target=producer, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    for step in range(n):
        slot = step % P
        ready[slot].wait()
        ready[slot].clear()
        if error_holder[0] is not None:
            raise error_holder[0]

        z, z_perturb = buffer[slot]
        _replay._apply_single_update_with_pregenerated_z(
            state, updates[step], param_names, z,
            z_perturb_dict=z_perturb,
            default_zo_eps=default_zo_eps,
            simulate_perturbation=simulate_perturbation,
            zo2_mode=zo2_mode,
        )
        buffer[slot] = None
        free[slot].set()

        if replay_step_time_log_enabled() and (step < 3 or step == n - 1):
            logger.info(f"[PipelinedReplay] update {step}: step={updates[step].get('step','?')}, "
                        f"seed={updates[step]['seed']}, grad={updates[step]['grad']:.6e}, "
                        f"lr={updates[step]['lr']}, wd={updates[step].get('wd', 0.0)}")
        elif replay_step_time_log_enabled() and step == 3:
            logger.info(f"[PipelinedReplay] ... ({n - 4} more updates) ...")

    for t in threads:
        t.join()
    if error_holder[0] is not None:
        raise error_holder[0]


def _pipelined_replay_gpu(state, updates, param_names, rng_device,
                          num_producers, default_zo_eps, simulate_perturbation,
                          zo2_mode, seeds_info, replay_dtype):
    from . import log_based_replay as _replay

    P = num_producers
    n = len(updates)
    streams = [torch.cuda.Stream() for _ in range(P)]
    ready_events = [torch.cuda.Event() for _ in range(P)]
    free_events = [torch.cuda.Event() for _ in range(P)]
    buffer = [None] * P

    for i in range(min(P, n)):
        grad_seed, perturb_seed = seeds_info[i]
        with torch.cuda.stream(streams[i]):
            z = _replay._generate_z_for_one_step(grad_seed, param_names, state, rng_device, replay_dtype)
            z_perturb = None
            if zo2_mode and perturb_seed != grad_seed:
                z_perturb = _replay._generate_z_for_one_step(
                    perturb_seed, param_names, state, rng_device, replay_dtype
                )
            buffer[i] = (z, z_perturb)
        ready_events[i].record(streams[i])

    default_stream = torch.cuda.current_stream()
    for step in range(n):
        slot = step % P
        default_stream.wait_event(ready_events[slot])

        z, z_perturb = buffer[slot]
        _replay._apply_single_update_with_pregenerated_z(
            state, updates[step], param_names, z,
            z_perturb_dict=z_perturb,
            default_zo_eps=default_zo_eps,
            simulate_perturbation=simulate_perturbation,
            zo2_mode=zo2_mode,
        )

        next_step = step + P
        if next_step < n:
            free_events[slot].record(default_stream)
            grad_seed, perturb_seed = seeds_info[next_step]
            with torch.cuda.stream(streams[slot]):
                streams[slot].wait_event(free_events[slot])
                z = _replay._generate_z_for_one_step(grad_seed, param_names, state, rng_device, replay_dtype)
                z_perturb = None
                if zo2_mode and perturb_seed != grad_seed:
                    z_perturb = _replay._generate_z_for_one_step(
                        perturb_seed, param_names, state, rng_device, replay_dtype
                    )
                buffer[slot] = (z, z_perturb)
            ready_events[slot].record(streams[slot])
        else:
            buffer[slot] = None

        if replay_step_time_log_enabled() and (step < 3 or step == n - 1):
            logger.info(f"[PipelinedReplay] update {step}: step={updates[step].get('step','?')}, "
                        f"seed={updates[step]['seed']}, grad={updates[step]['grad']:.6e}, "
                        f"lr={updates[step]['lr']}, wd={updates[step].get('wd', 0.0)}")
        elif replay_step_time_log_enabled() and step == 3:
            logger.info(f"[PipelinedReplay] ... ({n - 4} more updates) ...")

    torch.cuda.synchronize()


def _parallel_replay_updates_on_state(
    state: OrderedDict,
    updates: list,
    device: str = 'cpu',
    move_to_device: bool = True,
    trainable_param_names: list = None,
    default_zo_eps: float = 0.0,
    simulate_perturbation: bool = True,
    replay_in_fp32: bool = False,
    rng_device: str = "native",
    zo2_mode: bool = False,
    initial_prev_seed: int = None,
) -> OrderedDict:
    from . import log_based_replay as _replay

    if not updates:
        return state

    P = 1
    env_workers = os.environ.get('PARALLEL_RECOVERY_WORKERS', None)
    if env_workers is not None:
        P = max(1, int(env_workers))

    if time_log_enabled():
        logger.info(f"[PipelinedReplay] {len(updates)} updates, P={P}, "
                    f"rng_device={rng_device}, zo2_mode={zo2_mode}, "
                    f"simulate_perturbation={simulate_perturbation}")

    actual_device = 'cpu'
    if move_to_device and device == 'cuda' and torch.cuda.is_available():
        for key in state:
            if state[key].device.type != 'cuda':
                state[key] = state[key].cuda()
        actual_device = 'cuda'
    elif len(state) > 0:
        actual_device = next(iter(state.values())).device.type

    original_dtype = None
    replay_dtype = None
    if replay_in_fp32 and actual_device == 'cpu':
        sample = next(iter(state.values()))
        if sample.dtype in (torch.float16, torch.bfloat16):
            original_dtype = sample.dtype
            replay_dtype = torch.float32
            for key in state:
                state[key] = state[key].float()
            if time_log_enabled():
                logger.info(f"[PipelinedReplay] Upcast {original_dtype} -> fp32 for CPU replay")

    if actual_device == 'cpu' and torch.cuda.is_available() and rng_device != "zo_rng":
        logger.warning("[PipelinedReplay] WARNING: Replaying on CPU but CUDA is available. "
                       "Use device='cuda' or ZO_RNG_DEVICE=zo_rng for exact reconstruction.")

    param_names = trainable_param_names if trainable_param_names is not None else list(state.keys())
    model_bytes = sum(state[nm].numel() * state[nm].element_size() for nm in param_names)
    z_sets_per_step = 2 if zo2_mode else 1
    buffer_bytes = model_bytes * P * z_sets_per_step
    available_bytes = psutil.virtual_memory().available
    if resource_log_enabled() and buffer_bytes > available_bytes * 0.5:
        logger.warning(f"[PipelinedReplay] Ring buffer ~{buffer_bytes / 1e9:.1f} GB "
                       f"but only {available_bytes / 1e9:.1f} GB available. "
                       f"Consider reducing PARALLEL_RECOVERY_WORKERS.")

    seeds_info = []
    for i, update in enumerate(updates):
        if zo2_mode:
            if i == 0:
                prev_seed = initial_prev_seed if initial_prev_seed is not None else update['seed']
            else:
                prev_seed = updates[i - 1]['seed']
            seeds_info.append((prev_seed, update['seed']))
        else:
            seeds_info.append((update['seed'], update['seed']))

    t_start = time.time()
    _pip_proc = psutil.Process(os.getpid()) if resource_log_enabled() else None
    _pip_cpu0, _pip_gpu0 = _log_memory("pipelined start", _pip_proc, actual_device) if _pip_proc is not None else (None, None)

    use_gpu_pipeline = (actual_device == 'cuda' and rng_device == 'native')
    if use_gpu_pipeline:
        if time_log_enabled():
            logger.info(f"[PipelinedReplay] Using GPU mode (CUDA streams)")
        _pipelined_replay_gpu(
            state, updates, param_names, rng_device,
            num_producers=P, default_zo_eps=default_zo_eps,
            simulate_perturbation=simulate_perturbation,
            zo2_mode=zo2_mode, seeds_info=seeds_info,
            replay_dtype=replay_dtype,
        )
    else:
        if time_log_enabled():
            logger.info(f"[PipelinedReplay] Using CPU mode (threads)")
        _pipelined_replay_cpu(
            state, updates, param_names, rng_device,
            num_producers=P, default_zo_eps=default_zo_eps,
            simulate_perturbation=simulate_perturbation,
            zo2_mode=zo2_mode, seeds_info=seeds_info,
            replay_dtype=replay_dtype,
        )

    if _pip_proc is not None:
        _log_memory("pipelined done", _pip_proc, actual_device, _pip_cpu0, _pip_gpu0)

    if original_dtype is not None:
        for key in state:
            state[key] = state[key].to(original_dtype)
        if time_log_enabled():
            logger.info(f"[PipelinedReplay] Downcast fp32 -> {original_dtype}")

    t_elapsed = time.time() - t_start
    mode_str = "GPU/CUDA-streams" if use_gpu_pipeline else "CPU/threads"
    if time_log_enabled():
        logger.info(f"[PipelinedReplay] Completed: {len(updates)} updates in {t_elapsed:.3f}s "
                    f"(P={P}, mode={mode_str}, device={actual_device})")
    return state


def _closedform_cpu(state, param_names, terms, rng_device, num_workers,
                    accum_dtype, replay_dtype):
    from . import log_based_replay as _replay

    W = min(num_workers, len(terms))
    if W == 0:
        return {}

    total_sum = {}
    for name in param_names:
        param = state[name]
        total_sum[name] = torch.zeros(param.shape, dtype=accum_dtype, device=param.device)

    lock = threading.Lock()
    error_holder = [None]

    def worker_fn(worker_id):
        try:
            idx = worker_id
            while idx < len(terms):
                _step_idx, coeff_wd, coeff_nowd, grad_seed = terms[idx]
                z_dict = _replay._generate_z_for_one_step(
                    grad_seed, param_names, state, rng_device, replay_dtype
                )
                for name in param_names:
                    z = z_dict[name]
                    c = coeff_wd if _replay._is_wd_param(name) else coeff_nowd
                    if accum_dtype != z.dtype:
                        z = z.to(accum_dtype)
                        z_dict[name] = z
                    z.mul_(c)
                with lock:
                    for name in param_names:
                        total_sum[name].add_(z_dict[name])
                del z_dict
                idx += W
        except Exception as e:
            error_holder[0] = e

    threads = []
    for i in range(W):
        t = threading.Thread(target=worker_fn, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if error_holder[0] is not None:
        raise error_holder[0]
    return total_sum


def _closedform_gpu(state, param_names, terms, rng_device, num_workers,
                    accum_dtype, replay_dtype):
    from . import log_based_replay as _replay

    W = min(num_workers, len(terms))
    if W == 0:
        return {}

    streams = [torch.cuda.Stream() for _ in range(W)]
    total_sum = {}
    for name in param_names:
        param = state[name]
        total_sum[name] = torch.zeros(param.shape, dtype=accum_dtype, device=param.device)

    for batch_start in range(0, len(terms), W):
        batch_end = min(batch_start + W, len(terms))
        batch_size = batch_end - batch_start
        z_dicts = [None] * batch_size

        for i in range(batch_size):
            _step_idx, coeff_wd, coeff_nowd, grad_seed = terms[batch_start + i]
            with torch.cuda.stream(streams[i]):
                z_dicts[i] = _replay._generate_z_for_one_step(
                    grad_seed, param_names, state, rng_device, replay_dtype
                )
                for name in param_names:
                    z = z_dicts[i][name]
                    c = coeff_wd if _replay._is_wd_param(name) else coeff_nowd
                    if accum_dtype != z.dtype:
                        z = z.to(accum_dtype)
                        z_dicts[i][name] = z
                    z.mul_(c)

        torch.cuda.synchronize()
        for i in range(batch_size):
            for name in param_names:
                total_sum[name].add_(z_dicts[i][name])
        del z_dicts

    return total_sum


def _closedform_replay_on_state(
    state: OrderedDict,
    updates: list,
    device: str = 'cpu',
    move_to_device: bool = True,
    trainable_param_names: list = None,
    rng_device: str = "native",
    zo2_mode: bool = False,
    initial_prev_seed: int = None,
    num_workers: int = 1,
    precision: str = "mixed",
) -> OrderedDict:
    from . import log_based_replay as _replay

    if not updates:
        return state

    n = len(updates)
    W = num_workers
    if time_log_enabled():
        logger.info(f"[ClosedForm] {n} updates, W={W}, precision={precision}, "
                    f"rng_device={rng_device}, zo2_mode={zo2_mode}")

    actual_device = 'cpu'
    if move_to_device and device == 'cuda' and torch.cuda.is_available():
        for key in state:
            if state[key].device.type != 'cuda':
                state[key] = state[key].cuda()
        actual_device = 'cuda'
    elif len(state) > 0:
        actual_device = next(iter(state.values())).device.type

    sample = next(iter(state.values()))
    original_dtype = sample.dtype

    if precision == "fp32":
        accum_dtype = torch.float32
        target_dtype = torch.float32
        replay_dtype = torch.float32
        if original_dtype != torch.float32:
            for key in state:
                state[key] = state[key].float()
            if time_log_enabled():
                logger.info(f"[ClosedForm] Upcast {original_dtype} -> fp32")
    elif precision == "fp16":
        accum_dtype = original_dtype
        target_dtype = original_dtype
        replay_dtype = torch.float32 if (actual_device == 'cpu' and original_dtype != torch.float32) else None
    elif precision == "mixed":
        accum_dtype = torch.float32
        target_dtype = original_dtype
        replay_dtype = torch.float32 if (actual_device == 'cpu' and original_dtype != torch.float32) else None
    else:
        raise ValueError(f"Unknown precision mode: {precision}")

    if actual_device == 'cpu' and torch.cuda.is_available() and rng_device != "zo_rng":
        logger.warning("[ClosedForm] WARNING: Replaying on CPU but CUDA is available. "
                       "Use device='cuda' or ZO_RNG_DEVICE=zo_rng for exact reconstruction.")

    param_names = trainable_param_names if trainable_param_names is not None else list(state.keys())
    total_numel = sum(state[nm].numel() for nm in param_names)
    accum_elem_size = torch.tensor([], dtype=accum_dtype).element_size()
    replay_es = torch.tensor([], dtype=replay_dtype if replay_dtype is not None else original_dtype).element_size()
    total_buffer = total_numel * accum_elem_size + total_numel * replay_es * W
    available_bytes = psutil.virtual_memory().available
    if resource_log_enabled() and total_buffer > available_bytes * 0.5:
        logger.warning(f"[ClosedForm] Worker buffers ~{total_buffer / 1e9:.1f} GB "
                       f"but only {available_bytes / 1e9:.1f} GB available.")

    has_any_wd = any(u.get('wd', 0.0) != 0 for u in updates)
    sp = [1.0] * (n + 1)
    if has_any_wd:
        for i in range(n - 1, -1, -1):
            sp[i] = sp[i + 1] * (1.0 - updates[i]['lr'] * updates[i].get('wd', 0.0))
    sp_0 = sp[0]

    seeds_info = []
    for i, update in enumerate(updates):
        if zo2_mode:
            if i == 0:
                prev_seed = initial_prev_seed if initial_prev_seed is not None else update['seed']
            else:
                prev_seed = updates[i - 1]['seed']
            seeds_info.append((prev_seed, update['seed']))
        else:
            seeds_info.append((update['seed'], update['seed']))

    terms = []
    for t in range(n):
        grad = updates[t]['grad']
        if grad == 0:
            continue
        lr = updates[t]['lr']
        coeff_wd = sp[t + 1] * lr * grad
        coeff_nowd = lr * grad
        terms.append((t, coeff_wd, coeff_nowd, seeds_info[t][0]))

    if time_log_enabled():
        logger.info(f"[ClosedForm] {len(terms)} non-zero terms out of {n} updates"
                    f" (sp[0]={sp_0:.10f})")

    t_start = time.time()
    _cf_proc = psutil.Process(os.getpid()) if resource_log_enabled() else None
    _cf_cpu0, _cf_gpu0 = _log_memory("closedform start", _cf_proc, actual_device) if _cf_proc is not None else (None, None)

    use_gpu = (actual_device == 'cuda' and rng_device == 'native')
    if len(terms) == 0:
        total_sum = {}
    elif use_gpu:
        if time_log_enabled():
            logger.info(f"[ClosedForm] Using GPU mode (CUDA streams)")
        total_sum = _closedform_gpu(state, param_names, terms, rng_device, W, accum_dtype, replay_dtype)
    else:
        if time_log_enabled():
            logger.info(f"[ClosedForm] Using CPU mode (threads)")
        total_sum = _closedform_cpu(state, param_names, terms, rng_device, W, accum_dtype, replay_dtype)

    if _cf_proc is not None:
        _log_memory("closedform after accumulation", _cf_proc, actual_device, _cf_cpu0, _cf_gpu0)

    if not total_sum:
        total_sum = {
            name: torch.zeros(state[name].shape, dtype=accum_dtype, device=state[name].device)
            for name in param_names
        }

    for name in param_names:
        p0 = state[name]
        ts = total_sum[name]
        if _replay._is_wd_param(name) and has_any_wd:
            result = sp_0 * p0.to(accum_dtype) - ts
        else:
            result = p0.to(accum_dtype) - ts
        state[name] = result.to(target_dtype)

    t_elapsed = time.time() - t_start
    mode_str = "GPU/CUDA-streams" if use_gpu else "CPU/threads"
    if time_log_enabled():
        logger.info(f"[ClosedForm] Completed: {n} updates in {t_elapsed:.3f}s "
                    f"(W={W}, precision={precision}, mode={mode_str}, device={actual_device})")
    if _cf_proc is not None:
        _log_memory("closedform done", _cf_proc, actual_device, _cf_cpu0, _cf_gpu0)
    return state


def validate_closedform_replay(
    state: OrderedDict,
    updates: list,
    trainable_param_names: list = None,
    rng_device: str = "native",
    zo2_mode: bool = False,
    initial_prev_seed: int = None,
    num_workers: int = 1,
    device: str = 'cpu',
) -> dict:
    from . import log_based_replay as _replay

    def _clone(s):
        return OrderedDict((k, v.clone()) for k, v in s.items())

    param_names = trainable_param_names if trainable_param_names is not None else list(state.keys())
    state_serial = _clone(state)
    _replay._replay_updates_on_state(
        state_serial, updates, device=device, move_to_device=True,
        trainable_param_names=param_names,
        simulate_perturbation=False,
        rng_device=rng_device, zo2_mode=zo2_mode,
        initial_prev_seed=initial_prev_seed,
    )

    results = {}
    for prec in ["fp32", "mixed", "fp16"]:
        state_cf = _clone(state)
        _closedform_replay_on_state(
            state_cf, updates, device=device, move_to_device=True,
            trainable_param_names=param_names,
            rng_device=rng_device, zo2_mode=zo2_mode,
            initial_prev_seed=initial_prev_seed,
            num_workers=num_workers,
            precision=prec,
        )

        prec_results = {}
        for name in param_names:
            serial_p = state_serial[name].float()
            cf_p = state_cf[name].float()
            diff = (serial_p - cf_p).abs()
            max_abs = diff.max().item()
            denom = serial_p.abs().max().item()
            rel = max_abs / max(denom, 1e-10)
            prec_results[name] = {"max_abs": max_abs, "rel": rel}

        results[prec] = prec_results

    logger.info(f"[ClosedForm Validation] {len(updates)} updates, W={num_workers}")
    for prec in ["fp32", "mixed", "fp16"]:
        max_abs_all = max(v["max_abs"] for v in results[prec].values())
        max_rel_all = max(v["rel"] for v in results[prec].values())
        logger.info(f"  {prec:6s}: max_abs={max_abs_all:.2e}, max_rel={max_rel_all:.2e}")

    return results


# =========================================================================
# Section A: Async Anchor Checkpoint
# (Originally in async_anchor_checkpoint.py)
# =========================================================================

ADAM_STATE_NAME = "adam_state.pt"


def _subprocess_write_checkpoint(state_dict, save_path, step):
    """Write checkpoint in forked subprocess (independent GIL).

    Called only in the child process after os.fork(). Must NOT touch any CUDA
    resources. Uses os._exit() to avoid running the parent's atexit handlers
    or CUDA cleanup.
    """
    try:
        _atomic_save_state_dict_safetensors(
            state_dict,
            save_path,
            metadata={
                "base_step": int(step),
                "committed_step": int(step),
            },
        )
    except Exception:
        os._exit(1)
    os._exit(0)


class _PersistJob:
    """A pending CPU→disk persist job."""
    __slots__ = [
        'step',
        'output_dir',
        'copy_start_event',
        'copy_done_event',
        'uses_cuda',
        'd2h_fallback_s',
        'adam_state',
        'adam_d2h_s',
        'd2h_trace_id',
        'd2h_started_ns',
    ]

    def __init__(
        self,
        step,
        output_dir,
        copy_start_event,
        copy_done_event,
        uses_cuda,
        d2h_fallback_s=0.0,
        adam_state=None,
        adam_d2h_s=0.0,
        d2h_trace_id=None,
        d2h_started_ns=None,
    ):
        self.step = step
        self.output_dir = output_dir
        self.copy_start_event = copy_start_event
        self.copy_done_event = copy_done_event
        self.uses_cuda = uses_cuda
        self.d2h_fallback_s = d2h_fallback_s
        self.adam_state = adam_state
        self.adam_d2h_s = adam_d2h_s
        self.d2h_trace_id = d2h_trace_id
        self.d2h_started_ns = d2h_started_ns


class AsyncAnchorCheckpointer:
    """Async full-model checkpoint writer for ZO training.

    Pre-allocates a single CPU pinned buffer matching the model size.
    At anchor steps, async-copies GPU params to the pinned buffer via a
    dedicated CUDA stream, then a background thread first publishes
    `<ZO_SHM_DIR>/zo_anchor_latest_<hash>.safetensors` and finally persists
    the same snapshot to disk as `model.safetensors`.

    Args:
        model: nn.Module (used to determine param shapes/dtypes)
        checkpoint_dir: root output directory
        tied_groups: list of tied weight groups from _detect_tied_weights().
            Secondary keys are excluded from the pinned buffer and saved checkpoint,
            matching HuggingFace's save_pretrained() convention.
    """

    def __init__(self, model, checkpoint_dir, tied_groups=None):
        self._checkpoint_dir = checkpoint_dir
        self._shm_dir = _ensure_zo_shm_dir()
        self._anchor_latest_path = (
            os.path.join(
                self._shm_dir,
                f"zo_anchor_latest_{hashlib.md5(checkpoint_dir.encode()).hexdigest()[:8]}.safetensors",
            )
        )

        # Compute excluded keys from tied weight groups (keep first, exclude rest)
        self._excluded_keys = set()
        if tied_groups:
            for group in tied_groups:
                for name in group[1:]:
                    self._excluded_keys.add(name)
            if resource_log_enabled():
                logger.info(f"[AsyncAnchor] Excluding tied keys from checkpoint: {self._excluded_keys}")

        # CUDA stream dedicated to checkpoint copies
        self._ckpt_stream = torch.cuda.Stream()

        # Pre-allocate single pinned buffer (excluding tied weight duplicates)
        self._pinned_buffer = OrderedDict()

        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            if name in self._excluded_keys:
                continue
            self._pinned_buffer[name] = torch.empty(
                tensor.shape, dtype=tensor.dtype,
                device='cpu', pin_memory=True,
            )

        # Buffer availability (protected by Condition for wait/notify)
        self._buffer_free = True
        self._lock = threading.Lock()
        self._buffer_cond = threading.Condition(self._lock)

        # Persist completion: at most 1 child process writing at a time.
        # Training thread waits on this before starting a new checkpoint.
        self._persist_done = threading.Event()
        self._persist_done.set()  # Initially: no persist in progress

        # Completion tracking (protected by _lock)
        self._latest_completed_step = -1
        self._latest_completed_path = None
        self._latest_published_step = -1
        self._latest_published_snapshot = None

        # Background persist thread
        self._persist_queue = queue.Queue()
        self._persist_thread = threading.Thread(
            target=self._persist_worker, daemon=True, name="async-anchor-persist"
        )
        self._persist_thread.start()

        # Stats
        self._anchors_saved = 0
        self._enqueue_times = []
        self._d2h_times = []
        self._cpu_persist_total_times = []

        total_bytes = sum(
            t.numel() * t.element_size() for t in self._pinned_buffer.values()
        )
        if resource_log_enabled():
            logger.info(
                f"[AsyncAnchor] Init: {total_bytes / 1e9:.2f} GB pinned buffer"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def try_save_full_checkpoint(self, step, model, output_dir) -> bool:
        """Start async full checkpoint save.

        Waits for the previous persist to finish (at most 1 ongoing),
        then starts the async GPU→CPU copy. Never skips.
        """
        t_lock_start = time.time()
        # Ensure previous persist is done before starting a new one
        persist_wait_token = trace_begin(
            panel="gpu_train",
            lane="blocking",
            event="wait_anchor_persist",
            step=int(step),
        )
        self._persist_done.wait()
        trace_end(persist_wait_token, step=int(step))
        with self._buffer_cond:
            buffer_wait_token = None
            while not self._buffer_free:
                if buffer_wait_token is None:
                    buffer_wait_token = trace_begin(
                        panel="gpu_train",
                        lane="blocking",
                        event="wait_anchor_buffer",
                        step=int(step),
                    )
                if time_log_enabled():
                    logger.info(
                        f"[AsyncAnchor] Waiting for buffer at step {step}..."
                    )
                self._buffer_cond.wait()
            trace_end(buffer_wait_token, step=int(step))
            self._buffer_free = False
        t_lock = time.time() - t_lock_start

        t_sd_start = time.time()
        state_dict = model.state_dict()
        t_sd = time.time() - t_sd_start

        pinned = self._pinned_buffer

        # Phase 1: async copy on dedicated CUDA stream
        t_copy_start = time.time()
        d2h_token = trace_begin(
            panel="gpu_train",
            lane="anchor_thread",
            event="anchor_d2h_copy",
            step=int(step),
        )
        has_cuda = any(t.is_cuda for t in state_dict.values())
        if has_cuda:
            copy_start_event = torch.cuda.Event(enable_timing=True)
            copy_done_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self._ckpt_stream):
                copy_start_event.record()
                for name in pinned:
                    pinned[name].copy_(state_dict[name], non_blocking=True)
                copy_done_event.record()
            # GPU-side barrier: default stream waits for copy to finish
            # before next training step modifies the parameters.
            # This is a GPU-side dependency only — the CPU thread continues
            # immediately, so training never stalls on the CPU side.
            torch.cuda.current_stream().wait_stream(self._ckpt_stream)
        else:
            copy_start_event = None
            copy_done_event = None
            # All tensors on CPU (e.g. ZO2 offloading) — direct copy
            for name in pinned:
                pinned[name].copy_(state_dict[name])
        t_copy = time.time() - t_copy_start

        adam_state = None
        adam_d2h_s = 0.0
        opt = getattr(model, "opt", None)
        if opt is not None and hasattr(opt, "get_adam_state") and hasattr(opt, "betas") and hasattr(opt, "adam_eps"):
            t_adam_d2h_start = time.time()
            adam_raw = opt.get_adam_state()
            adam_state = {
                'm': OrderedDict(
                    (name, tensor.detach().to(device='cpu', dtype=torch.float32).clone())
                    for name, tensor in adam_raw.get('m', {}).items()
                ),
                'v': OrderedDict(
                    (name, tensor.detach().to(device='cpu', dtype=torch.float32).clone())
                    for name, tensor in adam_raw.get('v', {}).items()
                ),
                't': int(adam_raw.get('t', 0)),
                'betas': tuple(adam_raw.get('betas', getattr(opt, 'betas', (0.9, 0.999)))),
                'adam_eps': float(adam_raw.get('adam_eps', getattr(opt, 'adam_eps', 1e-8))),
            }
            adam_d2h_s = time.time() - t_adam_d2h_start

        # Queue Phase 2 (CPU→disk) for background thread
        self._persist_queue.put(
            _PersistJob(
                step=step,
                output_dir=output_dir,
                copy_start_event=copy_start_event,
                copy_done_event=copy_done_event,
                uses_cuda=has_cuda,
                d2h_fallback_s=t_copy,
                adam_state=adam_state,
                adam_d2h_s=adam_d2h_s,
                d2h_trace_id=d2h_token.event_id if d2h_token is not None else None,
                d2h_started_ns=d2h_token.started_ns if d2h_token is not None else None,
            )
        )
        self._anchors_saved += 1
        t_total = time.time() - t_lock_start
        self._enqueue_times.append(t_total)
        trace_instant(
            panel="gpu_train",
            lane="anchor_thread",
            event="anchor_enqueue",
            step=int(step),
            triggered_by=d2h_token.event_id if d2h_token is not None else None,
            counters={
                "enqueue_cpu_ms": t_total * 1000.0,
                "wait_ms": t_lock * 1000.0,
                "state_dict_ms": t_sd * 1000.0,
                "launch_ms": t_copy * 1000.0,
                "pinned_buffer_mb": sum(t.numel() * t.element_size() for t in self._pinned_buffer.values()) / 1024**2,
            },
        )

        if time_log_enabled():
            logger.info(
                f"[AsyncAnchor] Queued anchor step {step} "
                f"(enqueue_cpu={t_total:.3f}s, wait={t_lock:.3f}s, state_dict={t_sd:.3f}s, launch={t_copy:.3f}s)"
            )
        return True

    def get_latest_completed_anchor_step(self) -> int:
        """Step number of the most recent fully-persisted anchor (-1 if none)."""
        with self._lock:
            return self._latest_completed_step

    def get_latest_completed_anchor_path(self):
        """Path of the most recent fully-persisted anchor (None if none)."""
        with self._lock:
            return self._latest_completed_path

    def get_latest_published_anchor_step(self) -> int:
        with self._lock:
            return self._latest_published_step

    def consume_latest_published_snapshot(self, min_step_exclusive: int = -1):
        """Return the newest published in-memory snapshot if it is newer than min_step_exclusive."""
        with self._lock:
            if self._latest_published_step <= min_step_exclusive:
                return None
            snapshot = self._latest_published_snapshot
            if snapshot is None:
                return None
            self._latest_published_snapshot = None
            return self._latest_published_step, snapshot

    def get_anchor_latest_path(self):
        return self._anchor_latest_path

    def wait_for_completion(self):
        """Block until all pending persists finish."""
        done = threading.Event()
        self._persist_queue.put(done)
        done.wait()

    def shutdown(self):
        """Wait for pending work and stop the persist thread."""
        self.wait_for_completion()
        self._persist_queue.put(None)  # sentinel to exit worker loop
        self._persist_thread.join(timeout=60)
        if time_log_enabled():
            logger.info(f"[AsyncAnchor] Shutdown complete. Stats: {self.stats}")

    @property
    def stats(self) -> dict:
        """Checkpoint statistics."""
        return {
            'anchors_saved': self._anchors_saved,
            'enqueue_cpu_count': len(self._enqueue_times),
            'avg_enqueue_cpu_time': (
                sum(self._enqueue_times) / len(self._enqueue_times)
                if self._enqueue_times else 0
            ),
            'd2h_count': len(self._d2h_times),
            'avg_d2h_time': (
                sum(self._d2h_times) / len(self._d2h_times)
                if self._d2h_times else 0
            ),
            'cpu_persist_total_count': len(self._cpu_persist_total_times),
            'avg_cpu_persist_total_time': (
                sum(self._cpu_persist_total_times) / len(self._cpu_persist_total_times)
                if self._cpu_persist_total_times else 0
            ),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _persist_worker(self):
        """Background thread: clone buffer, fork subprocess, blocking wait.

        At most one child process writes to disk at a time. The training
        thread gates new checkpoints via _persist_done.wait().
        """
        force_fsync = os.environ.get('FORCE_FSYNC', '0') == '1'

        while True:
            item = self._persist_queue.get()

            if item is None:
                break

            if isinstance(item, threading.Event):
                item.set()
                continue

            job = item
            # Block here until GPU→CPU DMA completes — this is the ONLY
            # place that calls synchronize, never in the training thread.
            if job.uses_cuda:
                job.copy_done_event.synchronize()
                model_d2h_s = job.copy_start_event.elapsed_time(job.copy_done_event) / 1000.0
            else:
                model_d2h_s = job.d2h_fallback_s
            d2h_total_s = model_d2h_s + job.adam_d2h_s
            if job.d2h_trace_id is not None:
                trace_end_external(
                    event_id=job.d2h_trace_id,
                    panel="gpu_train",
                    lane="anchor_thread",
                    event="anchor_d2h_copy",
                    started_ns=job.d2h_started_ns,
                    step=int(job.step),
                    counters={
                        "d2h_model_ms": model_d2h_s * 1000.0,
                        "d2h_adam_ms": job.adam_d2h_s * 1000.0,
                        "d2h_ms": d2h_total_s * 1000.0,
                    },
                )
            cpu_ready_t0 = time.time()

            # Phase 2a: Clone pinned buffer → regular CPU memory.
            snapshot = {name: tensor.clone() for name, tensor in self._pinned_buffer.items()}

            # Free pinned buffer IMMEDIATELY and wake up waiting try_save.
            with self._buffer_cond:
                self._buffer_free = True
                self._buffer_cond.notify()

            publish_token = trace_begin(
                panel="gpu_train",
                lane="anchor_thread",
                event="anchor_publish_latest",
                step=int(job.step),
                triggered_by=job.d2h_trace_id,
            )
            _atomic_save_state_dict_safetensors(
                snapshot,
                self._anchor_latest_path,
                metadata={
                    "base_step": int(job.step),
                    "committed_step": int(job.step),
                },
            )
            trace_end(publish_token, step=int(job.step))
            with self._lock:
                if job.step > self._latest_published_step:
                    self._latest_published_step = job.step
                    self._latest_published_snapshot = (snapshot, job.adam_state)

            # Phase 2b: Fork subprocess for disk I/O.
            os.makedirs(job.output_dir, exist_ok=True)
            save_path = os.path.join(job.output_dir, "model.safetensors")
            adam_path = os.path.join(job.output_dir, ADAM_STATE_NAME)

            self._persist_done.clear()  # Mark persist as in-progress
            t0 = time.time()
            persist_token = trace_begin(
                panel="gpu_train",
                lane="anchor_thread",
                event="anchor_persist",
                step=int(job.step),
                triggered_by=job.d2h_trace_id,
            )
            pid = os.fork()
            if pid == 0:
                _subprocess_write_checkpoint(snapshot, save_path, job.step)
                os._exit(1)  # Safety net
            else:
                del snapshot  # Parent doesn't need it; child has CoW copy

            # Blocking wait — at most 1 child at a time.
            _, status = os.waitpid(pid, 0)
            t_total = time.time() - t0
            cpu_total_s = time.time() - cpu_ready_t0
            child_ok = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
            if child_ok:
                adam_persist_s = 0.0
                if job.adam_state is not None:
                    t_adam_persist = time.time()
                    torch.save(job.adam_state, adam_path)
                    adam_persist_s = time.time() - t_adam_persist
                if force_fsync:
                    fd = os.open(save_path, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    if job.adam_state is not None:
                        fd = os.open(adam_path, os.O_RDONLY)
                        try:
                            os.fsync(fd)
                        finally:
                            os.close(fd)
                with self._lock:
                    if job.step > self._latest_completed_step:
                        self._latest_completed_step = job.step
                        self._latest_completed_path = job.output_dir
                trace_end(
                    persist_token,
                    step=int(job.step),
                    counters={
                        "d2h_model_ms": model_d2h_s * 1000.0,
                        "d2h_adam_ms": job.adam_d2h_s * 1000.0,
                        "d2h_ms": d2h_total_s * 1000.0,
                        "adam_persist_ms": adam_persist_s * 1000.0,
                        "cpu_total_ms": cpu_total_s * 1000.0,
                        "output_ckpt_used_mb": directory_size_bytes(job.output_dir) / 1024**2,
                    },
                    extra={"save_path": save_path},
                )
                if time_log_enabled():
                    logger.info(
                        f"[AsyncAnchor] Persisted step {job.step} "
                        f"(d2h_model={model_d2h_s:.3f}s, d2h_adam={job.adam_d2h_s:.3f}s, "
                        f"d2h_total={d2h_total_s:.3f}s, "
                        f"adam_persist={adam_persist_s:.3f}s, cpu_total={cpu_total_s:.3f}s, "
                        f"adam_t={int(job.adam_state.get('t', 0)) if job.adam_state is not None else 0}) "
                        f"→ {save_path}"
                    )
            else:
                if os.WIFSIGNALED(status):
                    logger.error(
                        f"[AsyncAnchor] Subprocess killed by signal "
                        f"{os.WTERMSIG(status)} for step {job.step}"
                    )
                else:
                    logger.error(
                        f"[AsyncAnchor] Subprocess exited with code "
                        f"{os.WEXITSTATUS(status) if os.WIFEXITED(status) else '?'} "
                        f"for step {job.step}"
                    )
                trace_end(
                    persist_token,
                    step=int(job.step),
                    counters={
                        "d2h_model_ms": model_d2h_s * 1000.0,
                        "d2h_adam_ms": job.adam_d2h_s * 1000.0,
                        "d2h_ms": d2h_total_s * 1000.0,
                        "cpu_total_ms": cpu_total_s * 1000.0,
                    },
                    extra={"child_ok": False},
                )
            self._d2h_times.append(d2h_total_s)
            self._cpu_persist_total_times.append(cpu_total_s)
            self._persist_done.set()  # Allow next checkpoint


# =========================================================================
# Section B: Rebase functions
# (Originally in log_based_shadow.py)
# =========================================================================

def _write_rebase_payload_flat(
    state_dict,
    header_path,
    *,
    base_step,
    committed_step,
    tied_groups=None,
    adam_state=None,
    param_names=None,
):
    save_state = OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if key not in _secondary_tied_keys(tied_groups)
    )
    layout = _build_shadow_flat_layout(save_state)
    payload_paths = _rebase_payload_paths(header_path)
    _ensure_shadow_flat_files((payload_paths["state_path"],), int(layout["total_bytes"]))
    fd, mm, views = _open_shadow_flat_views(layout, payload_paths["state_path"])
    try:
        _copy_shadow_flat_views_from_state(views, save_state)
        mm.flush()
    finally:
        _close_shadow_flat_views(fd, mm, views)

    has_adam = bool(adam_state is not None)
    adam_layout = {"entries": [], "total_bytes": 0}
    if has_adam:
        adam_layout = _build_adam_flat_layout(save_state, param_names or list(save_state.keys()))
        _ensure_shadow_flat_files((payload_paths["adam_m_path"],), int(adam_layout["total_bytes"]))
        _ensure_shadow_flat_files((payload_paths["adam_v_path"],), int(adam_layout["total_bytes"]))
        adam_m_fd, adam_m_mm, adam_m_views = _open_shadow_flat_views(adam_layout, payload_paths["adam_m_path"])
        adam_v_fd, adam_v_mm, adam_v_views = _open_shadow_flat_views(adam_layout, payload_paths["adam_v_path"])
        try:
            _copy_shadow_flat_views_from_adam(adam_m_views, (adam_state or {}).get("m"))
            _copy_shadow_flat_views_from_adam(adam_v_views, (adam_state or {}).get("v"))
            adam_m_mm.flush()
            adam_v_mm.flush()
        finally:
            _close_shadow_flat_views(adam_m_fd, adam_m_mm, adam_m_views)
            _close_shadow_flat_views(adam_v_fd, adam_v_mm, adam_v_views)

    _write_shadow_flat_header(
        header_path,
        {
            "kind": "rebase_payload_flat",
            "base_step": int(base_step),
            "committed_step": int(committed_step),
            "has_adam": has_adam,
            "adam_t": int((adam_state or {}).get("t", 0)) if has_adam else 0,
            "adam_beta1": float((adam_state or {}).get("betas", (0.0, 0.0))[0]) if has_adam else 0.0,
            "adam_beta2": float((adam_state or {}).get("betas", (0.0, 0.0))[1]) if has_adam else 0.0,
            "adam_eps": float((adam_state or {}).get("adam_eps", 0.0)) if has_adam else 0.0,
            "layout": _serialize_flat_layout(layout),
            "adam_layout": _serialize_flat_layout(adam_layout),
            "state_path": payload_paths["state_path"],
            "adam_m_path": payload_paths["adam_m_path"] if has_adam else "",
            "adam_v_path": payload_paths["adam_v_path"] if has_adam else "",
        },
    )
    return header_path


def _cleanup_rebase_payload_flat(header_path):
    if not header_path:
        return
    payload_paths = _rebase_payload_paths(header_path)
    for path in (
        payload_paths["state_path"],
        payload_paths["adam_m_path"],
        payload_paths["adam_v_path"],
        header_path,
    ):
        if path and os.path.exists(path):
            os.unlink(path)


def _load_rebase_payload_flat(header_path, tied_groups=None, cleanup=False):
    header = _read_shadow_flat_header(header_path)
    layout = _deserialize_flat_layout(header["layout"])
    fd, mm, views = _open_shadow_flat_views(layout, header["state_path"])
    try:
        state_dict = OrderedDict((name, tensor.clone()) for name, tensor in views.items())
    finally:
        _close_shadow_flat_views(fd, mm, views)
    if tied_groups:
        _tie_state_dict_inplace(state_dict, tied_groups)

    adam_state = None
    if bool(header.get("has_adam", False)):
        adam_layout = _deserialize_flat_layout(header["adam_layout"])
        adam_m_fd, adam_m_mm, adam_m_views = _open_shadow_flat_views(adam_layout, header["adam_m_path"])
        adam_v_fd, adam_v_mm, adam_v_views = _open_shadow_flat_views(adam_layout, header["adam_v_path"])
        try:
            adam_state = {
                "m": OrderedDict((name, tensor.clone()) for name, tensor in adam_m_views.items()),
                "v": OrderedDict((name, tensor.clone()) for name, tensor in adam_v_views.items()),
                "t": int(header.get("adam_t", 0)),
                "betas": (
                    float(header.get("adam_beta1", 0.9)),
                    float(header.get("adam_beta2", 0.999)),
                ),
                "adam_eps": float(header.get("adam_eps", 1e-8)),
            }
        finally:
            _close_shadow_flat_views(adam_m_fd, adam_m_mm, adam_m_views)
            _close_shadow_flat_views(adam_v_fd, adam_v_mm, adam_v_views)

    base_step = int(header.get("base_step", 0))
    committed_step = int(header.get("committed_step", 0))
    if cleanup:
        _cleanup_rebase_payload_flat(header_path)
    return state_dict, adam_state, base_step, committed_step


def _rebase_working_state(anchor_ref, tied_groups, _logger):
    if isinstance(anchor_ref, dict):
        rebased, adam_state, base_step, committed_step = _load_shadow_bundle_flat(
            anchor_ref,
            tied_groups=tied_groups,
        )
    elif isinstance(anchor_ref, str) and anchor_ref.endswith(".json"):
        rebased, adam_state, base_step, committed_step = _load_rebase_payload_flat(
            anchor_ref,
            tied_groups=tied_groups,
            cleanup=True,
        )
    else:
        rebased, base_step, committed_step = _load_shadow_replica(anchor_ref, tied_groups=tied_groups)
        adam_state = None
    if tied_groups:
        _tie_state_dict_inplace(rebased, tied_groups)
    _logger.info(f"[Shadow] Rebased from {anchor_ref} at step {committed_step}")
    return rebased, adam_state, base_step, committed_step


def _trim_retained_updates(retained_updates, floor_step):
    stale_steps = [step for step in retained_updates if step <= floor_step]
    for step in stale_steps:
        retained_updates.pop(step, None)


def _replay_retained_suffix(
    retained_updates,
    rebase_step,
    working_state,
    param_names,
    rng_device,
    simulate_perturbation,
    default_zo_eps,
    adam_state,
    _bdc,
    _logger,
):
    replayed_steps = []
    for step in sorted(retained_updates):
        if step <= rebase_step:
            continue
        update = retained_updates[step]
        z_dict = _bdc._generate_z_for_one_step(update["seed"], param_names, working_state, rng_device)
        try:
            _bdc._apply_single_update_with_pregenerated_z(
                working_state,
                update,
                param_names,
                z_dict,
                default_zo_eps=default_zo_eps,
                simulate_perturbation=simulate_perturbation,
                adam_state=adam_state,
                _diag_logger=_logger,
            )
        finally:
            del z_dict
        replayed_steps.append(step)
    if replayed_steps:
        _logger.info(
            f"[Shadow] Replay-after-rebase: base={rebase_step} "
            f"replayed={len(replayed_steps)} last={replayed_steps[-1]}"
        )
        return replayed_steps[-1]
    return rebase_step


# =========================================================================
# Section C: Anchor publisher methods
# (Originally methods of LogBasedCheckpointCallback in log_based_checkpoint.py)
# =========================================================================

@dataclass
class AnchorPublishTask:
    step: int
    base_checkpoint_state: OrderedDict
    adam_state: dict | None
    base_pending_seed: int
    created_at: float


# Originally LogBasedCheckpointCallback._publish_anchor_latest
# Kept for reference; takes `self` (a LogBasedCheckpointCallback instance) as first arg.
def legacy_publish_anchor_latest(self, state_dict, step, *, adam_state=None):
    if adam_state is None:
        adam_state = self.shadow_adam_state
    if self.use_shadow_flat_commit:
        os.makedirs(self.rebase_payload_dir, exist_ok=True)
        payload_path = self._rebase_payload_header_path(step)
        _write_rebase_payload_flat(
            state_dict,
            payload_path,
            base_step=step,
            committed_step=step,
            tied_groups=getattr(self, "_tied_weight_groups", []),
            adam_state=adam_state,
            param_names=self._trainable_param_names or list(state_dict.keys()),
        )
        with self.anchor_publish_condition:
            self._staged_rebase_payloads.add(payload_path)
        return payload_path
    if not self.anchor_latest_path:
        return None
    save_state = OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if key not in self._shadow_secondary_keys()
    )
    _atomic_save_state_dict_safetensors(
        save_state,
        self.anchor_latest_path,
        metadata={
            "base_step": int(step),
            "committed_step": int(step),
        },
    )
    return self.anchor_latest_path


# Originally LogBasedCheckpointCallback._queue_shadow_rebase
# Kept for reference; takes `self` (a LogBasedCheckpointCallback instance) as first arg.
def legacy_queue_shadow_rebase(self, step, path=None):
    if self.update_queue is None:
        return
    try:
        payload = {
            "cmd": "rebase",
            "step": int(step),
            "path": path or self.anchor_latest_path,
        }
        self.update_queue.put_nowait(payload)
    except Exception as e:
        if not getattr(self, "_queue_error_logged", False):
            logger.warning(f"[LogBased] Failed to send rebase to shadow: {e}")
            self._queue_error_logged = True


# Originally LogBasedCheckpointCallback._use_async_anchor_publisher
# Kept for reference; takes `self` (a LogBasedCheckpointCallback instance) as first arg.
def legacy_use_async_anchor_publisher(self):
    async_anchor = getattr(self, "_async_anchor", None) or getattr(self.trainer, "_async_anchor", None)
    return bool(
        self.enable_shadow and
        self.batch_size >= 1 and
        self.use_shadow_flat_commit and
        async_anchor is None
    )


# Originally LogBasedCheckpointCallback._check_anchor_publisher_health
# Kept for reference; takes `self` (a LogBasedCheckpointCallback instance) as first arg.
def legacy_check_anchor_publisher_health(self):
    if self.anchor_publish_failed is not None:
        raise RuntimeError(
            f"anchor publisher failed previously: {self.anchor_publish_failed}"
        )


# Originally LogBasedCheckpointCallback._start_anchor_publisher
# Kept for reference; takes `self` (a LogBasedCheckpointCallback instance) as first arg.
def legacy_start_anchor_publisher(self):
    if not self._use_async_anchor_publisher():
        return
    if self.anchor_publish_thread is not None and self.anchor_publish_thread.is_alive():
        return
    self.anchor_publish_stop = False
    self.anchor_publish_failed = None
    self.anchor_publish_latest_task = None
    self.anchor_publish_inflight_step = None
    self.anchor_publish_completed_step = -1
    self.anchor_publish_thread = threading.Thread(
        target=self._anchor_publisher_main,
        name="anchor-publisher",
        daemon=True,
    )
    self.anchor_publish_thread.start()
    logger.info("[AnchorPublisher] Started async anchor publisher thread")


# Originally LogBasedCheckpointCallback._stop_anchor_publisher
# Kept for reference; takes `self` (a LogBasedCheckpointCallback instance) as first arg.
def legacy_stop_anchor_publisher(self, timeout_s=60.0):
    thread = self.anchor_publish_thread
    if thread is None:
        return
    with self.anchor_publish_condition:
        self.anchor_publish_stop = True
        self.anchor_publish_condition.notify_all()
    thread.join(timeout=timeout_s)
    if thread.is_alive():
        logger.warning(
            f"[AnchorPublisher] Timed out waiting for async anchor publisher to stop after {timeout_s:.1f}s"
        )
    else:
        logger.info(
            f"[AnchorPublisher] Stopped (completed_step={self.anchor_publish_completed_step})"
        )
    self.anchor_publish_thread = None
    self._check_anchor_publisher_health()


# Originally LogBasedCheckpointCallback._submit_anchor_publish_task
# Kept for reference; takes `self` (a LogBasedCheckpointCallback instance) as first arg.
def legacy_submit_anchor_publish_task(self, task: AnchorPublishTask):
    self._check_anchor_publisher_health()
    with self.anchor_publish_condition:
        previous = self.anchor_publish_latest_task
        if previous is not None and previous.step != task.step:
            logger.info(
                f"[AnchorPublisher] Dropped stale pending anchor step={previous.step} newer_step={task.step}"
            )
        self.anchor_publish_latest_task = task
        self.anchor_publish_condition.notify_all()


# Originally LogBasedCheckpointCallback._publish_anchor_task
# Kept for reference; takes `self` (a LogBasedCheckpointCallback instance) as first arg.
def legacy_publish_anchor_task(self, task: AnchorPublishTask):
    t_total = time.time()
    payload_path = None
    publish_anchor_s = 0.0
    queue_rebase_s = 0.0
    t0 = time.time()
    payload_path = self._publish_anchor_latest(
        task.base_checkpoint_state,
        task.step,
        adam_state=task.adam_state,
    )
    publish_anchor_s = time.time() - t0
    if payload_path is not None:
        t0 = time.time()
        self._queue_shadow_rebase(task.step, path=payload_path)
        queue_rebase_s = time.time() - t0
        self._last_shadow_rebased_anchor_step = int(task.step)
    logger.info(
        f"[AnchorPublisher] step={task.step} publish_anchor={publish_anchor_s:.3f}s "
        f"queue_rebase={queue_rebase_s:.3f}s total={time.time() - t_total:.3f}s"
    )


# Originally LogBasedCheckpointCallback._anchor_publisher_main
# Kept for reference; takes `self` (a LogBasedCheckpointCallback instance) as first arg.
def legacy_anchor_publisher_main(self):
    while True:
        with self.anchor_publish_condition:
            while not self.anchor_publish_stop and self.anchor_publish_latest_task is None:
                self.anchor_publish_condition.wait()
            if self.anchor_publish_stop and self.anchor_publish_latest_task is None:
                return
            task = self.anchor_publish_latest_task
            self.anchor_publish_latest_task = None
            self.anchor_publish_inflight_step = task.step
        try:
            self._publish_anchor_task(task)
        except Exception as exc:
            self.anchor_publish_failed = exc
            logger.exception(f"[AnchorPublisher] step={task.step} failed: {exc}")
            return
        finally:
            with self.anchor_publish_condition:
                if self.anchor_publish_inflight_step == task.step:
                    self.anchor_publish_inflight_step = None
                self.anchor_publish_completed_step = max(self.anchor_publish_completed_step, int(task.step))


# Originally LogBasedCheckpointCallback._update_base_and_shadow
# Kept for reference; takes `self` (a LogBasedCheckpointCallback instance) as first arg.
def legacy_update_base_and_shadow(self, model, step):
    """Update base_checkpoint_state from current model (full step only for batch_size>=1).
    GPU → CPU clone, then publish anchor latest / notify shadow to rebase."""
    from .log_based_utils import _clone_state_dict_to_cpu, _step_diag_enabled, _step_exact_enabled
    from .log_based_utils import _log_adam_checksums, _log_adam_exact_fingerprint, _log_adam_exact_compare

    self._check_anchor_publisher_health()
    trace_token = trace_begin(
        panel="gpu_train",
        lane="blocking",
        event="full_checkpoint_refresh",
        step=int(step),
    )
    t0_total = time.time()
    t0 = time.time()
    base_checkpoint_state = _clone_state_dict_to_cpu(model.state_dict())
    clone_model_s = time.time() - t0

    clone_adam_s = 0.0
    shadow_adam_state = None
    live_adam_state = None
    if self.enable_shadow:
        t0 = time.time()
        opt = getattr(model, "opt", None)
        if opt is not None and hasattr(opt, "get_adam_state"):
            live_adam_state = opt.get_adam_state()
        shadow_adam_state = self._clone_shadow_adam_state(model)
        clone_adam_s = time.time() - t0
    self.base_checkpoint_state = base_checkpoint_state
    self.shadow_adam_state = shadow_adam_state
    self.active_base_step = int(step)
    self.base_checkpoint_step = int(step)
    self._active_base_pending_seed = self._pending_seed
    self._base_pending_seed = self._pending_seed

    submit_anchor_s = 0.0
    publish_anchor_s = 0.0
    queue_rebase_s = 0.0
    if self.batch_size >= 1 and self.enable_shadow:
        if self._use_async_anchor_publisher():
            t0 = time.time()
            self._submit_anchor_publish_task(
                AnchorPublishTask(
                    step=int(step),
                    base_checkpoint_state=base_checkpoint_state,
                    adam_state=shadow_adam_state,
                    base_pending_seed=int(self._base_pending_seed),
                    created_at=time.time(),
                )
            )
            submit_anchor_s = time.time() - t0
        else:
            t0 = time.time()
            anchor_path = self._publish_anchor_latest(self.base_checkpoint_state, step)
            publish_anchor_s = time.time() - t0
            if anchor_path is not None:
                t0 = time.time()
                self._queue_shadow_rebase(step, path=anchor_path)
                queue_rebase_s = time.time() - t0
                self._last_shadow_rebased_anchor_step = int(step)

    if time_log_enabled():
        logger.info(
            f"[LogBased FullCkpt] step={step} "
            f"clone_model={clone_model_s:.3f}s clone_adam={clone_adam_s:.3f}s "
            f"submit_anchor={submit_anchor_s:.3f}s "
            f"publish_anchor={publish_anchor_s:.3f}s queue_rebase={queue_rebase_s:.3f}s "
            f"total={time.time() - t0_total:.3f}s"
        )
    trace_end(
        trace_token,
        step=int(step),
        counters={
            "clone_model_ms": clone_model_s * 1000.0,
            "clone_adam_ms": clone_adam_s * 1000.0,
            "submit_anchor_ms": submit_anchor_s * 1000.0,
            "publish_anchor_ms": publish_anchor_s * 1000.0,
            "queue_rebase_ms": queue_rebase_s * 1000.0,
        },
    )
    if _step_diag_enabled() or _step_exact_enabled():
        if live_adam_state is not None:
            _log_adam_checksums(f"train_snapshot step={step}", live_adam_state)
            if _step_exact_enabled():
                _log_adam_exact_fingerprint(f"train_snapshot step={step}", live_adam_state)
        if shadow_adam_state is not None:
            _log_adam_checksums(f"shadow_snapshot step={step}", shadow_adam_state)
            if _step_exact_enabled():
                _log_adam_exact_fingerprint(f"shadow_snapshot step={step}", shadow_adam_state)
        if _step_exact_enabled() and live_adam_state is not None and shadow_adam_state is not None:
            _log_adam_exact_compare(
                f"train_vs_shadow_snapshot step={step}",
                live_adam_state,
                shadow_adam_state,
            )


# Originally LogBasedCheckpointCallback.on_async_anchor_persisted
# Kept for reference; takes `self` (a LogBasedCheckpointCallback instance) as first arg.
def legacy_on_async_anchor_persisted(self, step, checkpoint_path):
    self.base_checkpoint_path = checkpoint_path
    self.base_checkpoint_step = int(step)
    if self.shadow_adam_state is not None:
        logger.info(
            f"[AsyncAnchor] Base updated to step {step} with Adam base "
            f"(t={int(self.shadow_adam_state.get('t', 0))})"
        )
    with self.update_lock:
        for u in reversed(self.update_history):
            if u['step'] <= step:
                self._base_pending_seed = u['seed']
                break
        self.update_history = [u for u in self.update_history if u['step'] > step]
