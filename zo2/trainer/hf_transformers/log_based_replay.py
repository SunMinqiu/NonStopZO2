import logging
import os
import time
from collections import OrderedDict

import psutil
import torch

from ...optimizer.mezo_adam.shared import apply_mezo_adam_update
from ...utils.logging_controls import replay_step_time_log_enabled, resource_log_enabled, time_log_enabled
from ...utils.trace import trace_instant
from .log_based_utils import (
    _log_adam_checksums,
    _log_adam_exact_fingerprint,
    _log_memory,
    _log_state_checksums,
    _log_state_exact_fingerprint,
    _step_diag_enabled,
    _step_exact_enabled,
)
def _parallel_replay_updates_on_state(*args, **kwargs):
    from .legacy_pipeline_closed_form_replay import _parallel_replay_updates_on_state as _fn
    return _fn(*args, **kwargs)

def _closedform_replay_on_state(*args, **kwargs):
    from .legacy_pipeline_closed_form_replay import _closedform_replay_on_state as _fn
    return _fn(*args, **kwargs)

logger = logging.getLogger(__name__)

_replay_adam_state_cache = {}


def _generate_z_for_replay(param, rng_device="native", zo_gen=None):
    """Generate z noise for replay, respecting rng_device setting."""
    if rng_device == "zo_rng":
        return zo_gen.randn(param.shape, dtype=param.dtype, device=param.device)
    if rng_device == "cpu" and param.device.type != "cpu":
        z = torch.normal(mean=0, std=1, size=param.size(), dtype=torch.float32, device='cpu')
        return z.to(dtype=param.dtype, device=param.device)
    return torch.normal(mean=0, std=1, size=param.size(), dtype=param.dtype, device=param.device)


def _is_wd_param(name):
    """Return True if this parameter receives weight decay."""
    return ('bias' not in name and 'layer_norm' not in name
            and 'layernorm' not in name and 'ln' not in name)


def _generate_z_for_one_step(seed, param_names, state, rng_device, replay_dtype=None):
    """Generate z for all params for a single step."""
    z_dict = {}

    if rng_device == "zo_rng":
        import zo_rng
        zo_gen = zo_rng.Generator(seed)
        for name in param_names:
            param = state[name]
            dtype = replay_dtype if replay_dtype is not None else param.dtype
            z_dict[name] = zo_gen.randn(param.shape, dtype=dtype, device=param.device)
    else:
        sample_param = state[param_names[0]]
        if rng_device == "cpu" or sample_param.device.type == "cpu":
            gen = torch.Generator(device='cpu')
        else:
            gen = torch.Generator(device=sample_param.device)
        gen.manual_seed(seed)

        for name in param_names:
            param = state[name]
            dtype = replay_dtype if replay_dtype is not None else param.dtype

            if rng_device == "cpu" and param.device.type != "cpu":
                z = torch.normal(
                    mean=0, std=1, size=param.size(),
                    dtype=torch.float32, device='cpu', generator=gen
                )
                z_dict[name] = z.to(dtype=dtype, device=param.device)
            else:
                z_dict[name] = torch.normal(
                    mean=0, std=1, size=param.size(),
                    dtype=dtype, device=param.device, generator=gen
                )

    return z_dict


def _apply_single_update_with_pregenerated_z(state, update, param_names, z_dict,
                                             z_perturb_dict=None,
                                             default_zo_eps=0.0,
                                             simulate_perturbation=True,
                                             zo2_mode=False,
                                             adam_state=None,
                                             _diag_first_call=False,
                                             _diag_logger=None):
    """Apply one ZO update using pre-generated z tensors."""
    grad = update['grad']
    lr = update['lr']
    wd = update.get('wd', 0.0)
    zo_eps = update.get('zo_eps', default_zo_eps)
    z_for_perturb = z_perturb_dict if z_perturb_dict is not None else z_dict
    _lr_grad = float(lr * grad)

    if adam_state is not None:
        if simulate_perturbation and zo_eps > 0:
            for scaling_factor in [1, -2, 1]:
                _alpha = float(scaling_factor * zo_eps)
                for name in param_names:
                    state[name].data.add_(z_dict[name], alpha=_alpha)
        adam_state['t'] += 1
        apply_mezo_adam_update(
            ((name, state[name]) for name in param_names),
            get_z=lambda name, _param_tensor: z_dict[name],
            grad=grad,
            lr=lr,
            weight_decay=None,
            default_weight_decay=wd,
            betas=adam_state['betas'],
            adam_eps=adam_state['adam_eps'],
            t=adam_state['t'],
            m_state=adam_state['m'],
            v_state=adam_state['v'],
            state_key=lambda name, _param: name,
            diag_label=f"adam_apply step={update.get('step', '?')}",
            diag_logger=_diag_logger,
        )
        return

    if zo2_mode and grad != 0:
        for name in param_names:
            param = state[name]
            z = z_dict[name]
            if wd == 0.0:
                param.sub_(z, alpha=_lr_grad)
            elif _is_wd_param(name):
                tmp = z.mul(grad)
                tmp.add_(param, alpha=wd)
                param.sub_(tmp, alpha=lr)
            else:
                param.sub_(z, alpha=_lr_grad)

        if simulate_perturbation and zo_eps > 0:
            for scaling_factor in [1, -2, 1]:
                _alpha = float(scaling_factor * zo_eps)
                for name in param_names:
                    state[name].data.add_(z_for_perturb[name], alpha=_alpha)
    else:
        if simulate_perturbation and zo_eps > 0:
            for scaling_factor in [1, -2, 1]:
                _alpha = float(scaling_factor * zo_eps)
                for name in param_names:
                    state[name].data.add_(z_dict[name], alpha=_alpha)

        if grad != 0:
            for name in param_names:
                param = state[name]
                z = z_dict[name]
                if wd == 0.0:
                    param.sub_(z, alpha=_lr_grad)
                elif _is_wd_param(name):
                    tmp = z.mul(grad)
                    tmp.add_(param, alpha=wd)
                    param.sub_(tmp, alpha=lr)
                else:
                    param.sub_(z, alpha=_lr_grad)


def _set_replay_adam_state(adam_state):
    global _replay_adam_state_cache
    _replay_adam_state_cache = adam_state or {}


def _get_and_clear_replay_adam_state():
    global _replay_adam_state_cache
    result = _replay_adam_state_cache
    _replay_adam_state_cache = {}
    return result if result else None


def _load_adam_state_from_base(base_checkpoint_ref, fallback_optimizer_state=None):
    """Load Adam state from base checkpoint."""
    if base_checkpoint_ref == '__initial__':
        betas = fallback_optimizer_state.get('adam_betas', (0.9, 0.999)) if fallback_optimizer_state else (0.9, 0.999)
        adam_eps = fallback_optimizer_state.get('adam_eps_value', 1e-8) if fallback_optimizer_state else 1e-8
        return {'m': {}, 'v': {}, 't': 0, 'betas': betas, 'adam_eps': adam_eps}

    adam_sidecar_path = os.path.join(base_checkpoint_ref, "adam_state.pt")
    if os.path.exists(adam_sidecar_path):
        adam_state = torch.load(adam_sidecar_path, map_location='cpu', weights_only=False)
        if isinstance(adam_state, dict):
            logger.info(f"[Adam Replay] Loaded adam_state from sidecar {adam_sidecar_path}")
            return adam_state

    opt_path = os.path.join(base_checkpoint_ref, "optimizer.pt")
    if os.path.exists(opt_path):
        opt = torch.load(opt_path, map_location='cpu', weights_only=False)
        adam_state = opt.get('adam_state', None)
        if adam_state:
            return adam_state

    betas = fallback_optimizer_state.get('adam_betas', (0.9, 0.999)) if fallback_optimizer_state else (0.9, 0.999)
    adam_eps = fallback_optimizer_state.get('adam_eps_value', 1e-8) if fallback_optimizer_state else 1e-8
    logger.warning(f"[Adam Replay] No adam_state found in base {base_checkpoint_ref}, starting from t=0")
    return {'m': {}, 'v': {}, 't': 0, 'betas': betas, 'adam_eps': adam_eps}


def _apply_single_update(state, update, param_names, default_zo_eps=0.0,
                         simulate_perturbation=True, rng_device="native",
                         zo2_mode=False, prev_seed=None,
                         adam_state=None):
    """Apply one ZO update to a state dict in-place."""
    seed = update['seed']
    grad = update['grad']
    lr = update['lr']
    wd = update.get('wd', 0.0)
    zo_eps = update.get('zo_eps', default_zo_eps)

    def _reset_rng(rng_seed=None):
        s = rng_seed if rng_seed is not None else seed
        if rng_device == "zo_rng":
            import zo_rng
            return zo_rng.Generator(s)
        torch.manual_seed(s)
        return None

    t_start = time.time()
    t_z = 0.0
    t_update = 0.0

    if adam_state is not None:
        if simulate_perturbation and zo_eps > 0:
            for scaling_factor in [1, -2, 1]:
                zo_gen = _reset_rng()
                for name in param_names:
                    param = state[name]
                    _t0 = time.time()
                    z = _generate_z_for_replay(param, rng_device, zo_gen)
                    t_z += time.time() - _t0
                    _t0 = time.time()
                    param.data.add_(z, alpha=float(scaling_factor * zo_eps))
                    t_update += time.time() - _t0
        adam_state['t'] += 1

        zo_gen = _reset_rng()
        def _get_z(_name, param_tensor):
            nonlocal t_z
            _t0 = time.time()
            z = _generate_z_for_replay(param_tensor, rng_device, zo_gen)
            t_z += time.time() - _t0
            return z

        helper_t0 = time.time()
        z_before = t_z
        apply_mezo_adam_update(
            ((name, state[name]) for name in param_names),
            get_z=_get_z,
            grad=grad,
            lr=lr,
            weight_decay=None,
            default_weight_decay=wd,
            betas=adam_state['betas'],
            adam_eps=adam_state['adam_eps'],
            t=adam_state['t'],
            m_state=adam_state['m'],
            v_state=adam_state['v'],
            state_key=lambda name, _param: name,
            diag_label=f"replay_live step={update.get('step', '?')}",
            diag_logger=logger,
        )
        helper_elapsed = time.time() - helper_t0
        t_update += max(0.0, helper_elapsed - (t_z - z_before))

        return {'total': time.time() - t_start, 'z_gen': t_z, 'update': t_update}

    _lr_grad = float(lr * grad)

    if zo2_mode and grad != 0:
        grad_seed = prev_seed if prev_seed is not None else seed
        zo_gen = _reset_rng(grad_seed)
        for name in param_names:
            param = state[name]
            _t0 = time.time()
            z = _generate_z_for_replay(param, rng_device, zo_gen)
            t_z += time.time() - _t0
            _t0 = time.time()
            if wd == 0.0:
                param.sub_(z, alpha=_lr_grad)
            elif _is_wd_param(name):
                tmp = z.mul(grad)
                tmp.add_(param, alpha=wd)
                param.sub_(tmp, alpha=lr)
            else:
                param.sub_(z, alpha=_lr_grad)
            t_update += time.time() - _t0

        if simulate_perturbation and zo_eps > 0:
            for scaling_factor in [1, -2, 1]:
                zo_gen = _reset_rng(seed)
                for name in param_names:
                    param = state[name]
                    _t0 = time.time()
                    z = _generate_z_for_replay(param, rng_device, zo_gen)
                    t_z += time.time() - _t0
                    _t0 = time.time()
                    param.data.add_(z, alpha=float(scaling_factor * zo_eps))
                    t_update += time.time() - _t0
    else:
        if simulate_perturbation and zo_eps > 0:
            for scaling_factor in [1, -2, 1]:
                zo_gen = _reset_rng()
                for name in param_names:
                    param = state[name]
                    _t0 = time.time()
                    z = _generate_z_for_replay(param, rng_device, zo_gen)
                    t_z += time.time() - _t0
                    _t0 = time.time()
                    param.data.add_(z, alpha=float(scaling_factor * zo_eps))
                    t_update += time.time() - _t0

        if grad != 0:
            zo_gen = _reset_rng()
            for name in param_names:
                param = state[name]
                _t0 = time.time()
                z = _generate_z_for_replay(param, rng_device, zo_gen)
                t_z += time.time() - _t0
                _t0 = time.time()
                if wd == 0.0:
                    param.sub_(z, alpha=_lr_grad)
                elif _is_wd_param(name):
                    tmp = z.mul(grad)
                    tmp.add_(param, alpha=wd)
                    param.sub_(tmp, alpha=lr)
                else:
                    param.sub_(z, alpha=_lr_grad)
                t_update += time.time() - _t0

    return {'total': time.time() - t_start, 'z_gen': t_z, 'update': t_update}


def _replay_updates_on_state(
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
    adam_state: dict = None,
) -> OrderedDict:
    """Replay ZO updates on a state dict."""
    if adam_state is not None:
        if os.environ.get('PARALLEL_RECOVERY') == '1' or os.environ.get('CLOSEDFORM_RECOVERY') == '1':
            logger.warning("[Replay] Adam mode only supports serial replay, ignoring PARALLEL/CLOSEDFORM")
    else:
        if os.environ.get('PARALLEL_RECOVERY', '0') == '1':
            return _parallel_replay_updates_on_state(
                state, updates, device=device, move_to_device=move_to_device,
                trainable_param_names=trainable_param_names,
                default_zo_eps=default_zo_eps,
                simulate_perturbation=simulate_perturbation,
                replay_in_fp32=replay_in_fp32,
                rng_device=rng_device,
                zo2_mode=zo2_mode,
                initial_prev_seed=initial_prev_seed,
            )

        if os.environ.get('CLOSEDFORM_RECOVERY', '0') == '1':
            return _closedform_replay_on_state(
                state, updates, device=device, move_to_device=move_to_device,
                trainable_param_names=trainable_param_names,
                rng_device=rng_device,
                zo2_mode=zo2_mode,
                initial_prev_seed=initial_prev_seed,
                num_workers=int(os.environ.get('CLOSEDFORM_WORKERS', '1')),
                precision=os.environ.get('CLOSEDFORM_PRECISION', 'mixed'),
            )

    if not updates:
        return state

    actual_device = 'cpu'
    if move_to_device and device == 'cuda' and torch.cuda.is_available():
        _moved = {}
        for key in state:
            _cpu_id = id(state[key])
            if _cpu_id in _moved:
                state[key] = _moved[_cpu_id]
            elif state[key].device.type != 'cuda':
                _gpu_t = state[key].cuda()
                _moved[_cpu_id] = _gpu_t
                state[key] = _gpu_t
        actual_device = 'cuda'
    elif len(state) > 0:
        actual_device = next(iter(state.values())).device.type

    if adam_state is not None:
        for mv_key in ('m', 'v'):
            for name in adam_state.get(mv_key, {}):
                t_mv = adam_state[mv_key][name]
                if t_mv.device.type != actual_device:
                    adam_state[mv_key][name] = t_mv.to(actual_device)

    original_dtype = None
    if replay_in_fp32 and actual_device == 'cpu':
        sample = next(iter(state.values()))
        if sample.dtype == torch.float16 or sample.dtype == torch.bfloat16:
            original_dtype = sample.dtype
            for key in state:
                state[key] = state[key].float()
            logger.info(f"[Replay] Upcast {original_dtype} → fp32 for CPU replay")

    logger.info(f"[Replay] Replaying {len(updates)} updates on device={actual_device}"
                f" (simulate_perturbation={simulate_perturbation}, replay_in_fp32={replay_in_fp32},"
                f" rng_device={rng_device})")
    if actual_device == 'cpu' and torch.cuda.is_available() and rng_device != "zo_rng":
        logger.warning("[Replay] WARNING: Replaying on CPU but CUDA is available. "
                       "CPU and CUDA RNG produce different z for the same seed. "
                       "Use device='cuda' for exact reconstruction, or use ZO_RNG_DEVICE=zo_rng "
                       "for cross-device deterministic replay.")

    param_names = trainable_param_names if trainable_param_names is not None else list(state.keys())

    _seq_proc = psutil.Process(os.getpid()) if resource_log_enabled() else None
    _seq_cpu0, _seq_gpu0 = _log_memory("sequential start", _seq_proc, actual_device) if _seq_proc is not None else (None, None)
    _seq_quarter = max(1, len(updates) // 4)

    timings = []
    for i, update in enumerate(updates):
        prev_seed = (initial_prev_seed if i == 0 else updates[i - 1]['seed']) if zo2_mode else None
        timing = _apply_single_update(
            state, update, param_names, default_zo_eps=default_zo_eps,
            simulate_perturbation=simulate_perturbation, rng_device=rng_device,
            zo2_mode=zo2_mode, prev_seed=prev_seed,
            adam_state=adam_state
        )
        timings.append(timing)
        trace_instant(
            panel="gpu_train",
            lane="counters",
            event="replay_step",
            step=int(update.get("step", i + 1)),
            counters={
                "replay_total_ms": float(timing["total"] * 1000.0),
                "replay_z_ms": float(timing["z_gen"] * 1000.0),
                "replay_update_ms": float(timing["update"] * 1000.0),
            },
            extra={"device": actual_device},
        )

        if _seq_proc is not None and i > 0 and i % _seq_quarter == 0:
            _log_memory(f"sequential step {i}/{len(updates)}", _seq_proc, actual_device, _seq_cpu0, _seq_gpu0)

        if replay_step_time_log_enabled():
            logger.info(f"[Replay] update {i}: step={update.get('step','?')}, seed={update['seed']}, "
                        f"grad={update['grad']:.6e}, lr={update['lr']}, wd={update.get('wd', 0.0)}, "
                        f"zo_eps={update.get('zo_eps', default_zo_eps)}, "
                        f"time={timing['total']:.4f}s (z_gen={timing['z_gen']:.4f}s, update={timing['update']:.4f}s)")
        if _step_diag_enabled() or _step_exact_enabled():
            if _step_diag_enabled():
                _log_state_checksums(f"replay_live step={update.get('step','?')}", state)
                if adam_state is not None:
                    _log_adam_checksums(f"replay_live step={update.get('step','?')}", adam_state)
            if _step_exact_enabled():
                _log_state_exact_fingerprint(f"replay_live step={update.get('step','?')}", state)
                if adam_state is not None:
                    _log_adam_exact_fingerprint(f"replay_live step={update.get('step','?')}", adam_state)

    if timings and time_log_enabled():
        avg_total = sum(t['total'] for t in timings) / len(timings)
        avg_z = sum(t['z_gen'] for t in timings) / len(timings)
        avg_upd = sum(t['update'] for t in timings) / len(timings)
        total_z = sum(t['z_gen'] for t in timings)
        total_upd = sum(t['update'] for t in timings)
        logger.info(f"[Replay Timing] avg per step: total={avg_total:.4f}s, z_gen={avg_z:.4f}s, update={avg_upd:.4f}s")
        logger.info(f"[Replay Timing] total: z_gen={total_z:.3f}s, update={total_upd:.3f}s")

    if original_dtype is not None:
        for key in state:
            state[key] = state[key].to(original_dtype)
        logger.info(f"[Replay] Downcast fp32 → {original_dtype}")

    return state
