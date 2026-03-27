import logging
import os
import re
import time
from collections import OrderedDict

import psutil
import torch

from .log_based_replay import (
    _get_and_clear_replay_adam_state,
    _load_adam_state_from_base,
    _replay_updates_on_state,
    _set_replay_adam_state,
)
from .log_based_shadow import _load_shadow_replica
from .log_based_utils import (
    _DTYPE_MAP,
    _detect_tied_weights,
    _log_memory,
    _restore_tied_weights,
    _tie_state_dict_inplace,
)

logger = logging.getLogger(__name__)


def _state_checksums(state_dict, trainable_param_names=None):
    """Return a few checksum views for the same state dict."""
    all_sum = 0.0
    unique_sum = 0.0
    trainable_sum = 0.0
    seen_ptrs = set()

    for tensor in state_dict.values():
        all_sum += tensor.float().sum().item()
        ptr = tensor.data_ptr()
        if ptr not in seen_ptrs:
            seen_ptrs.add(ptr)
            unique_sum += tensor.float().sum().item()

    if trainable_param_names is not None:
        for name in trainable_param_names:
            if name in state_dict:
                trainable_sum += state_dict[name].float().sum().item()

    return {
        'all_sum': all_sum,
        'unique_sum': unique_sum,
        'trainable_sum': trainable_sum,
        'num_tensors': len(state_dict),
        'num_unique_ptrs': len(seen_ptrs),
    }


def _log_state_checksums(label, state_dict, trainable_param_names=None, tied_groups=None):
    """Emit consistent checksum diagnostics for resume/replay debugging."""
    stats = _state_checksums(state_dict, trainable_param_names)
    logger.info(
        f"[DIAG-CKSUM] {label}: "
        f"all={stats['all_sum']:.10e} "
        f"unique={stats['unique_sum']:.10e} "
        f"trainable={stats['trainable_sum']:.10e} "
        f"tensors={stats['num_tensors']} "
        f"unique_ptrs={stats['num_unique_ptrs']}"
    )

    for name in ('model.embed_tokens.weight', 'lm_head.weight'):
        if name in state_dict:
            logger.info(
                f"[DIAG-CKSUM] {label}: {name}={state_dict[name].float().sum().item():.10e}"
            )

    if tied_groups:
        for group in tied_groups[:2]:
            present = [name for name in group if name in state_dict]
            if len(present) >= 2:
                primary = state_dict[present[0]]
                for other_name in present[1:]:
                    other = state_dict[other_name]
                    logger.info(
                        f"[DIAG-TIE] {label}: {present[0]} vs {other_name} "
                        f"same_ptr={primary.data_ptr() == other.data_ptr()} "
                        f"max_abs_diff={(primary.float() - other.float()).abs().max().item():.10e}"
                    )


def load_log_based_checkpoint(checkpoint_dir, base_checkpoint_dir=None, device='cpu',
                              simulate_perturbation=True, replay_in_fp32=False):
    """Load a log-based checkpoint if model files exist locally."""
    safe_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(safe_path):
        from safetensors.torch import load_file
        state_dict = load_file(safe_path)
        _restore_tied_weights(state_dict, checkpoint_dir)
        return state_dict

    model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        _restore_tied_weights(state_dict, checkpoint_dir)
        return state_dict

    full_model_path = os.path.join(checkpoint_dir, "pytorch_model_full.bin")
    if os.path.exists(full_model_path):
        return torch.load(full_model_path, map_location='cpu', weights_only=True)

    return None


def _load_base_state(base_checkpoint_ref, pretrained_model_name, tied_groups, model_dtype,
                     output_dir=None):
    """Load the base model state dict for replay."""
    if base_checkpoint_ref == "__initial__":
        if output_dir is not None:
            initial_model_dir = os.path.join(output_dir, "initial_model")
            safe_path = os.path.join(initial_model_dir, "model.safetensors")
            bin_path = os.path.join(initial_model_dir, "pytorch_model.bin")
            if os.path.exists(safe_path):
                from safetensors.torch import load_file
                logger.info(f"[Resume] Loading initial model from cached {safe_path}")
                return load_file(safe_path), tied_groups
            if os.path.exists(bin_path):
                logger.info(f"[Resume] Loading initial model from cached {bin_path}")
                return torch.load(bin_path, map_location='cpu', weights_only=True), tied_groups

        if pretrained_model_name is None:
            raise ValueError(
                "This checkpoint uses differential mode from initial model. "
                "You must provide pretrained_model_name to load it."
            )
        try:
            from transformers import AutoModelForCausalLM
            dtype_kwargs = {'torch_dtype': model_dtype} if model_dtype is not None else {}
            logger.info(f"[Resume] Loading base model from HuggingFace: {pretrained_model_name} (dtype={model_dtype})")
            base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, **dtype_kwargs)
            if not tied_groups:
                tied_groups = _detect_tied_weights(base_model)
                if tied_groups:
                    logger.info(f"[Resume] Detected tied weight groups from model: {tied_groups}")
            base_state = base_model.state_dict()
            del base_model
        except Exception as e:
            raise FileNotFoundError(f"Cannot load pretrained model {pretrained_model_name}: {e}")
    else:
        base_state = load_log_based_checkpoint(base_checkpoint_ref, base_checkpoint_dir=pretrained_model_name)
        if base_state is None:
            raise FileNotFoundError(f"Cannot load base checkpoint from {base_checkpoint_ref}")

    return base_state, tied_groups


def resume_from_log_based(
    checkpoint_path: str,
    output_dir: str = None,
    pretrained_model_name: str = None,
    device: str = 'cpu',
    simulate_perturbation: bool = True,
    replay_in_fp32: bool = False,
    base_state_dict: OrderedDict = None,
    cached_optimizer_state: dict = None,
    rng_device: str = "native",
    zo2_mode: bool = False,
    shadow_path: str = None,
) -> OrderedDict:
    """Resume from log-based checkpoints."""
    ckpt_dir = os.path.dirname(checkpoint_path) if os.path.isfile(checkpoint_path) else checkpoint_path

    if output_dir is None:
        output_dir = os.path.dirname(ckpt_dir)

    match = re.search(r'checkpoint-(\d+)', ckpt_dir)
    if not match:
        raise ValueError(f"Cannot extract step from checkpoint path: {ckpt_dir}")
    target_step = int(match.group(1))

    logger.info(f"[Resume] Target checkpoint: {ckpt_dir} (step {target_step})")
    logger.info(f"[Resume] Replay device: {device}")

    optimizer_path = os.path.join(ckpt_dir, "optimizer.pt")

    if cached_optimizer_state is not None and isinstance(cached_optimizer_state, dict) and 'zo_update_history' in cached_optimizer_state:
        optimizer_state = cached_optimizer_state
        logger.info("[Resume] Using cached optimizer state (skipped disk I/O)")
    elif os.path.exists(optimizer_path):
        optimizer_state = torch.load(optimizer_path, map_location='cpu', weights_only=False)
        if not isinstance(optimizer_state, dict) or 'zo_update_history' not in optimizer_state:
            logger.info("[Resume] optimizer.pt has no zo_update_history, loading as regular checkpoint")
            return load_log_based_checkpoint(ckpt_dir, base_checkpoint_dir=pretrained_model_name)
        logger.info("[Resume] Found log-based checkpoint (optimizer.pt with zo_update_history)")
    else:
        logger.info("[Resume] No optimizer.pt found, loading as regular checkpoint")
        return load_log_based_checkpoint(ckpt_dir, base_checkpoint_dir=pretrained_model_name)

    batch_size = optimizer_state.get('batch_size', 0)
    base_checkpoint_ref = optimizer_state.get('base_checkpoint', '__initial__')
    updates = optimizer_state['zo_update_history']
    tied_groups = optimizer_state.get('tied_weights', [])
    trainable_param_names = optimizer_state.get('trainable_param_names', None)
    default_zo_eps = optimizer_state.get('zo_eps', 0.0)
    model_dtype_str = optimizer_state.get('model_dtype', None)
    pending_grad = optimizer_state.get('pending_grad', None)
    base_pending_seed = optimizer_state.get('base_pending_seed', None)
    is_full_checkpoint = optimizer_state.get('is_full_checkpoint', False)

    ckpt_rng_device = optimizer_state.get('rng_device', 'native')
    if rng_device == "native" and ckpt_rng_device != "native":
        rng_device = ckpt_rng_device
        logger.info(f"[Resume] Auto-detected rng_device={rng_device} from checkpoint")

    if not zo2_mode and optimizer_state.get('zo2', False):
        zo2_mode = True
        logger.info("[Resume] Auto-detected zo2_mode=True from checkpoint")
    if zo2_mode:
        logger.info("[Resume] ZO2 mode: will use prev-step seed for gradient, current seed for perturbation")

    model_dtype = _DTYPE_MAP.get(model_dtype_str, None)

    logger.info(f"[Resume] Checkpoint mode: batch_size={batch_size}, base_checkpoint={base_checkpoint_ref}")
    if model_dtype is not None:
        logger.info(f"[Resume] Model dtype from checkpoint: {model_dtype}")
    if tied_groups:
        logger.info(f"[Resume] Tied weight groups from checkpoint: {tied_groups}")

    if batch_size == 0 and base_checkpoint_ref != "__initial__":
        logger.warning(
            f"[Resume] batch_size=0 but base_checkpoint={base_checkpoint_ref}, "
            f"overriding to __initial__ (batch_size=0 always replays from initial model)"
        )
        base_checkpoint_ref = "__initial__"

    zo_method = optimizer_state.get('zo_method', 'mezo-sgd')
    is_adam = (zo_method == 'mezo-adam')
    adam_state = None

    if is_full_checkpoint:
        logger.info("[Resume] Target is a full checkpoint, loading directly")
        if is_adam:
            _set_replay_adam_state(None)
        return load_log_based_checkpoint(ckpt_dir, base_checkpoint_dir=pretrained_model_name)

    if is_adam:
        adam_state = _load_adam_state_from_base(base_checkpoint_ref, optimizer_state)
        logger.info(f"[Resume] Adam mode: loaded base adam state (t={adam_state.get('t', 0)}, "
                    f"betas={adam_state.get('betas')}, {len(adam_state.get('m', {}))} m/v entries)")

    if shadow_path:
        reconstructed, _shadow_base_step, shadow_step = _load_shadow_replica(
            shadow_path,
            tied_groups=tied_groups,
        )
        _log_state_checksums(
            f"shadow_loaded step={shadow_step}",
            reconstructed,
            trainable_param_names=trainable_param_names,
            tied_groups=tied_groups,
        )
        updates = [u for u in updates if u['step'] > shadow_step]
        logger.info(f"[Resume] Soft recovery: shadow at step {shadow_step}, replaying {len(updates)} lag updates")
    elif base_state_dict is not None and base_checkpoint_ref == "__initial__":
        logger.info("[Resume] Using pre-loaded base state dict in-place (no clone)")
        reconstructed = base_state_dict
    else:
        base_state, tied_groups = _load_base_state(
            base_checkpoint_ref, pretrained_model_name, tied_groups, model_dtype, output_dir=output_dir
        )
        reconstructed = base_state

    _log_state_checksums(
        f"base_ready source={base_checkpoint_ref}",
        reconstructed,
        trainable_param_names=trainable_param_names,
        tied_groups=tied_groups,
    )

    logger.info(f"[Resume] Replaying {len(updates)} updates (default_zo_eps={default_zo_eps})")
    if pending_grad is not None:
        logger.info(f"[Resume] pending_grad={pending_grad} (will be restored to opt.projected_grad)")

    if device == 'cuda' and torch.cuda.is_available():
        for key in reconstructed:
            if reconstructed[key].device.type != 'cuda':
                reconstructed[key] = reconstructed[key].cuda()
        torch.cuda.synchronize()

    if tied_groups:
        _tie_state_dict_inplace(reconstructed, tied_groups)
        _log_state_checksums(
            "after_tie_before_replay",
            reconstructed,
            trainable_param_names=trainable_param_names,
            tied_groups=tied_groups,
        )

    _mem_proc = psutil.Process(os.getpid())
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _mem_cpu0, _mem_gpu0 = _log_memory("before replay", _mem_proc, device)

    t_replay_start = time.time()
    _replay_updates_on_state(
        reconstructed, updates, device=device, move_to_device=False,
        trainable_param_names=trainable_param_names,
        default_zo_eps=default_zo_eps,
        simulate_perturbation=simulate_perturbation,
        replay_in_fp32=replay_in_fp32,
        rng_device=rng_device,
        zo2_mode=zo2_mode,
        initial_prev_seed=base_pending_seed,
        adam_state=adam_state,
    )
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.info(f"[Resume Replay] {len(updates)} updates replayed in {time.time() - t_replay_start:.3f}s (device={device})")
    _log_state_checksums(
        "after_replay",
        reconstructed,
        trainable_param_names=trainable_param_names,
        tied_groups=tied_groups,
    )

    if is_adam and adam_state is not None:
        _set_replay_adam_state(adam_state)
        logger.info(f"[Resume] Cached replayed Adam state: t={adam_state.get('t', 0)}")

    _log_memory("after replay", _mem_proc, device, _mem_cpu0, _mem_gpu0)

    if updates:
        last = updates[-1]
        logger.info(f"[VERIFY-RESUME] Last replayed update: step={last.get('step','?')}, "
                    f"seed={last['seed']}, grad={last['grad']:.6e}")
    if pending_grad is not None:
        logger.info(f"[VERIFY-RESUME] pending_grad={pending_grad} => first resumed step should apply this grad")

    logger.info(f"[Resume] Completed! Recovered to step {target_step}")
    return reconstructed
