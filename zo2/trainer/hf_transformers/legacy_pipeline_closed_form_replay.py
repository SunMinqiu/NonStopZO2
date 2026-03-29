"""
LEGACY / DISCARDED replay paths kept only for reference.

This module contains the deprecated pipeline replay and closed-form replay
implementations. They are intentionally isolated so routine maintenance and
future coding agents do not need to read them when working on the active
checkpoint/replay path.
"""

import logging
import os
import threading
import time
from collections import OrderedDict

import psutil
import torch

from ...utils.logging_controls import resource_log_enabled, time_log_enabled
from .log_based_utils import _log_memory

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

        if time_log_enabled() and (step < 3 or step == n - 1):
            logger.info(f"[PipelinedReplay] update {step}: step={updates[step].get('step','?')}, "
                        f"seed={updates[step]['seed']}, grad={updates[step]['grad']:.6e}, "
                        f"lr={updates[step]['lr']}, wd={updates[step].get('wd', 0.0)}")
        elif time_log_enabled() and step == 3:
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

        if time_log_enabled() and (step < 3 or step == n - 1):
            logger.info(f"[PipelinedReplay] update {step}: step={updates[step].get('step','?')}, "
                        f"seed={updates[step]['seed']}, grad={updates[step]['grad']:.6e}, "
                        f"lr={updates[step]['lr']}, wd={updates[step].get('wd', 0.0)}")
        elif time_log_enabled() and step == 3:
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
