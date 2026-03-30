import logging
import os
import re
import time
import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import psutil
import torch

from ...utils.logging_controls import (
    consistency_log_enabled,
    resource_log_enabled,
    state_diag_log_enabled,
    state_exact_log_enabled,
    time_log_enabled,
)
from ...utils.trace import configure_trace, default_trace_path, trace_enabled, trace_instant, trace_span
from .log_based_replay import (
    _load_adam_state_from_base,
    _replay_updates_on_state,
    _set_replay_adam_state,
)
from .log_based_shadow import (
    _build_adam_flat_layout,
    _build_shadow_flat_layout,
    _shadow_flat_meta_paths,
    _load_shadow_bundle_flat,
    _load_shadow_replica,
)
from .log_based_utils import (
    _DTYPE_MAP,
    _detect_tied_weights,
    _log_adam_checksums,
    _log_adam_exact_fingerprint,
    _log_memory,
    _state_exact_fingerprint as _shared_state_exact_fingerprint,
    _restore_tied_weights,
    _tie_state_dict_inplace,
)

logger = logging.getLogger(__name__)
LOG_METADATA_NAME = "log_metadata.pt"


@dataclass
class LogBasedRecoveryBundle:
    state_dict: OrderedDict
    adam_state: dict | None
    base_step: int
    committed_step: int
    pending_grad: float | None
    base_pending_seed: int | None
    shadow_used: bool


def _tensor_exact_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(cpu.numpy()).tobytes()).hexdigest()


def _state_exact_fingerprint(state_dict, trainable_param_names=None):
    # Use the same full-state hashing logic as shadow_durable/train_durable_ref.
    # Comparing hashes across resume and shadow logs is otherwise meaningless.
    return _shared_state_exact_fingerprint(state_dict)


def _log_state_exact_fingerprint(label, state_dict, trainable_param_names=None):
    if not state_exact_log_enabled():
        return
    fp, tensor_count = _state_exact_fingerprint(state_dict, trainable_param_names=trainable_param_names)
    logger.info(f"[DIAG-EXACT] {label}: sha256={fp} tensors={tensor_count}")


def _log_state_exact_compare(label, lhs, rhs, trainable_param_names=None):
    if not state_exact_log_enabled():
        return
    names = list(trainable_param_names) if trainable_param_names is not None else list(lhs.keys())
    missing_lhs = [name for name in names if name in rhs and name not in lhs]
    missing_rhs = [name for name in names if name in lhs and name not in rhs]
    if missing_lhs or missing_rhs:
        logger.info(
            f"[DIAG-EXACT] {label}: exact_match=False missing_lhs={len(missing_lhs)} "
            f"missing_rhs={len(missing_rhs)} first_missing_lhs={missing_lhs[:3]} "
            f"first_missing_rhs={missing_rhs[:3]}"
        )
        return

    diffs = 0
    first_name = None
    first_max_abs = None
    first_lhs_hash = None
    first_rhs_hash = None
    for name in names:
        if name not in lhs or name not in rhs:
            continue
        a = lhs[name].detach().cpu()
        b = rhs[name].detach().cpu()
        if not torch.equal(a, b):
            diffs += 1
            if first_name is None:
                first_name = name
                first_max_abs = (a.float() - b.float()).abs().max().item()
                first_lhs_hash = _tensor_exact_hash(a)
                first_rhs_hash = _tensor_exact_hash(b)
    if diffs == 0:
        logger.info(f"[DIAG-EXACT] {label}: exact_match=True")
    else:
        logger.info(
            f"[DIAG-EXACT] {label}: exact_match=False differing_tensors={diffs} "
            f"first_diff={first_name} first_max_abs={first_max_abs:.10e} "
            f"lhs_sha256={first_lhs_hash} rhs_sha256={first_rhs_hash}"
        )


def _state_exact_matches(lhs, rhs, trainable_param_names=None):
    names = list(trainable_param_names) if trainable_param_names is not None else list(lhs.keys())
    for name in names:
        if name not in lhs or name not in rhs:
            return False
        a = lhs[name].detach().cpu()
        b = rhs[name].detach().cpu()
        if not torch.equal(a, b):
            return False
    return True


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
    if not state_diag_log_enabled():
        return
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


def _flat_storage_from_header_path(header_path, template_state_dict, tied_groups, trainable_param_names, has_adam):
    stem = header_path.removesuffix(".header.json")
    meta_paths = _shadow_flat_meta_paths(header_path)
    storage = {
        "enabled": True,
        # Writer persists the authoritative flat layouts into the shadow header.
        # Keep these template-built layouts only as legacy fallback.
        "layout": _build_shadow_flat_layout(template_state_dict, tied_groups=tied_groups),
        "header_path": header_path,
        "buffer_paths": (f"{stem}.bin",),
        "state_meta_path": meta_paths["state_meta_path"],
        "adam_meta_path": meta_paths["adam_meta_path"],
        "has_adam": bool(has_adam),
    }
    if has_adam:
        storage["adam_layout"] = _build_adam_flat_layout(template_state_dict, trainable_param_names or [])
        storage["adam_m_buffer_paths"] = (f"{stem}.adam_m.bin",)
        storage["adam_v_buffer_paths"] = (f"{stem}.adam_v.bin",)
    else:
        storage["adam_layout"] = {"entries": [], "total_bytes": 0}
        storage["adam_m_buffer_paths"] = ()
        storage["adam_v_buffer_paths"] = ()
    return storage


def resume_from_log_based_bundle(
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
) -> LogBasedRecoveryBundle:
    """Resume from log-based checkpoints and return a structured recovery bundle."""
    if trace_enabled() and output_dir is not None:
        configure_trace(path=os.environ.get("ZO_TRACE_PATH") or default_trace_path(output_dir), process_role="train")
    ckpt_dir = os.path.dirname(checkpoint_path) if os.path.isfile(checkpoint_path) else checkpoint_path

    if output_dir is None:
        output_dir = os.path.dirname(ckpt_dir)

    match = re.search(r'checkpoint-(\d+)', ckpt_dir)
    if not match:
        raise ValueError(f"Cannot extract step from checkpoint path: {ckpt_dir}")
    target_step = int(match.group(1))

    if time_log_enabled():
        logger.info(f"[Resume] Target checkpoint: {ckpt_dir} (step {target_step})")
        logger.info(f"[Resume] Replay device: {device}")
    trace_instant(
        panel="gpu_train",
        lane="blocking",
        event="resume_begin",
        step=int(target_step),
        extra={"checkpoint_dir": ckpt_dir, "device": device},
    )

    optimizer_path = os.path.join(ckpt_dir, "optimizer.pt")
    metadata_path = os.path.join(ckpt_dir, LOG_METADATA_NAME)

    redo_source = None

    if cached_optimizer_state is not None and isinstance(cached_optimizer_state, dict) and 'zo_update_history' in cached_optimizer_state:
        optimizer_state = cached_optimizer_state
        redo_source = "cached_optimizer_state"
        logger.info("[Resume] Using cached redo state (source=cached_optimizer_state, skipped disk I/O)")
    elif os.path.exists(metadata_path):
        optimizer_state = torch.load(metadata_path, map_location='cpu', weights_only=False)
        if not isinstance(optimizer_state, dict):
            raise RuntimeError(f"{LOG_METADATA_NAME} is not a dict checkpoint: {metadata_path}")
        redo_source = LOG_METADATA_NAME
        logger.info(f"[Resume] Found lightweight scalar redo metadata (source={LOG_METADATA_NAME})")
    elif os.path.exists(optimizer_path):
        optimizer_state = torch.load(optimizer_path, map_location='cpu', weights_only=False)
        if not isinstance(optimizer_state, dict) or 'zo_update_history' not in optimizer_state:
            logger.info("[Resume] optimizer.pt has no zo_update_history, loading as regular checkpoint")
            state_dict = load_log_based_checkpoint(ckpt_dir, base_checkpoint_dir=pretrained_model_name)
            return LogBasedRecoveryBundle(
                state_dict=state_dict,
                adam_state=None,
                base_step=target_step,
                committed_step=target_step,
                pending_grad=None,
                base_pending_seed=None,
                shadow_used=False,
            )
        redo_source = "optimizer.pt"
        logger.info("[Resume] Found full log checkpoint redo (source=optimizer.pt with zo_update_history)")
    else:
        optimizer_state = None
        logger.info("[Resume] No optimizer.pt/log metadata found; continuing with shadow-or-regular recovery")

    if optimizer_state is None:
        if shadow_path:
            logger.info("[Resume] No durable redo found, attempting shadow-only recovery")
            template_state = load_log_based_checkpoint(ckpt_dir, base_checkpoint_dir=pretrained_model_name)
            tied_groups = []
            if shadow_path.endswith(".flat.header.json"):
                shadow_stem = shadow_path[: -len(".header.json")]
                has_adam_shadow = os.path.exists(f"{shadow_stem}.adam_m.bin") and os.path.exists(
                    f"{shadow_stem}.adam_v.bin"
                )
                flat_storage = _flat_storage_from_header_path(
                    shadow_path,
                    template_state,
                    tied_groups,
                    None,
                    has_adam=has_adam_shadow,
                )
                reconstructed, shadow_adam_state, shadow_base_step, shadow_step = _load_shadow_bundle_flat(
                    flat_storage,
                    tied_groups=tied_groups,
                )
                adam_state = shadow_adam_state if has_adam_shadow else None
            else:
                reconstructed, shadow_base_step, shadow_step = _load_shadow_replica(
                    shadow_path,
                    tied_groups=tied_groups,
                )
                adam_state = None
            _log_state_checksums("shadow_only_recovery", reconstructed)
            _log_state_exact_fingerprint("shadow_only_recovery", reconstructed)
            logger.info(
                f"[Resume] Shadow-only recovery succeeded at step {shadow_step}; "
                f"redo_source=shadow_only, replay_updates=0"
            )
            return LogBasedRecoveryBundle(
                state_dict=reconstructed,
                adam_state=adam_state,
                base_step=int(shadow_base_step if shadow_base_step is not None else target_step),
                committed_step=int(shadow_step if shadow_step is not None else target_step),
                pending_grad=None,
                base_pending_seed=None,
                shadow_used=True,
            )
        logger.info("[Resume] Falling back to regular checkpoint load (no shadow/redo available)")
        state_dict = load_log_based_checkpoint(ckpt_dir, base_checkpoint_dir=pretrained_model_name)
        return LogBasedRecoveryBundle(
            state_dict=state_dict,
            adam_state=None,
            base_step=target_step,
            committed_step=target_step,
            pending_grad=None,
            base_pending_seed=None,
            shadow_used=False,
        )

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
        if consistency_log_enabled():
            logger.info(f"[Resume] Auto-detected rng_device={rng_device} from checkpoint")

    if not zo2_mode and optimizer_state.get('zo2', False):
        zo2_mode = True
        if consistency_log_enabled():
            logger.info("[Resume] Auto-detected zo2_mode=True from checkpoint")
    if zo2_mode:
        if consistency_log_enabled():
            logger.info("[Resume] ZO2 mode: will use prev-step seed for gradient, current seed for perturbation")

    model_dtype = _DTYPE_MAP.get(model_dtype_str, None)

    if consistency_log_enabled():
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
    base_adam_state = None

    if is_full_checkpoint:
        logger.info("[Resume] Target is a full checkpoint, loading directly")
        if is_adam:
            _set_replay_adam_state(None)
        state_dict = load_log_based_checkpoint(ckpt_dir, base_checkpoint_dir=pretrained_model_name)
        return LogBasedRecoveryBundle(
            state_dict=state_dict,
            adam_state=None,
            base_step=target_step,
            committed_step=target_step,
            pending_grad=pending_grad,
            base_pending_seed=base_pending_seed,
            shadow_used=False,
        )

    if is_adam:
        adam_state = _load_adam_state_from_base(base_checkpoint_ref, optimizer_state)
        base_adam_state = adam_state
        if consistency_log_enabled():
            logger.info(f"[Resume] Adam mode: loaded base adam state (t={adam_state.get('t', 0)}, "
                        f"betas={adam_state.get('betas')}, {len(adam_state.get('m', {}))} m/v entries)")
        _log_adam_checksums(f"base_adam source={base_checkpoint_ref}", adam_state)
        _log_adam_exact_fingerprint(f"base_adam source={base_checkpoint_ref}", adam_state)

    base_ref_step = 0
    if base_checkpoint_ref != "__initial__":
        base_match = re.search(r'checkpoint-(\d+)', str(base_checkpoint_ref))
        if base_match:
            base_ref_step = int(base_match.group(1))

    reconstructed = None
    shadow_used = False
    shadow_adam_state = None
    shadow_base_step = None
    shadow_step = None
    shadow_loaded_state = None
    model_source = f"base:{base_checkpoint_ref}"
    model_source_step = int(base_ref_step)
    adam_source = f"base:{base_checkpoint_ref}" if is_adam else "none"
    adam_source_step = int(base_adam_state.get("t", 0)) if base_adam_state is not None else None
    base_template_state = None
    if shadow_path:
        if shadow_path.endswith(".flat.header.json"):
            if base_state_dict is not None and base_checkpoint_ref == "__initial__":
                template_state = base_state_dict
            else:
                template_state, tied_groups = _load_base_state(
                    base_checkpoint_ref, pretrained_model_name, tied_groups, model_dtype, output_dir=output_dir
                )
            base_template_state = template_state
            flat_storage = _flat_storage_from_header_path(
                shadow_path,
                template_state,
                tied_groups,
                trainable_param_names,
                has_adam=is_adam,
            )
            try:
                reconstructed, shadow_adam_state, shadow_base_step, shadow_step = _load_shadow_bundle_flat(
                    flat_storage,
                    tied_groups=tied_groups,
                )
            except RuntimeError as e:
                err = str(e)
                if "missing state metadata" in err or "missing adam metadata" in err:
                    logger.warning(
                        f"[Resume] Ignoring legacy shadow flat snapshot at {shadow_path}; "
                        "falling back to base checkpoint + replay. "
                        f"Reason: {err}"
                    )
                    shadow_path = None
                    shadow_adam_state = None
                    shadow_base_step = None
                    shadow_step = None
                    reconstructed = None
                else:
                    raise
            if shadow_path is not None:
                model_source = f"shadow:{shadow_path}"
                model_source_step = int(shadow_step) if shadow_step is not None else None
                if is_adam and shadow_adam_state is not None:
                    adam_state = shadow_adam_state
                    adam_source = f"shadow:{shadow_path}"
                    adam_source_step = int(shadow_adam_state.get("t", 0))
                    _log_adam_checksums(f"shadow_loaded step={shadow_step}", shadow_adam_state)
                    _log_adam_exact_fingerprint(f"shadow_loaded step={shadow_step}", shadow_adam_state)
        else:
            reconstructed, shadow_base_step, shadow_step = _load_shadow_replica(
                shadow_path,
                tied_groups=tied_groups,
            )
            model_source = f"shadow:{shadow_path}"
            model_source_step = int(shadow_step) if shadow_step is not None else None
        if shadow_path and reconstructed is not None:
            _log_state_checksums(
                f"shadow_loaded step={shadow_step}",
                reconstructed,
                trainable_param_names=trainable_param_names,
                tied_groups=tied_groups,
            )
            _log_state_exact_fingerprint(
                f"shadow_loaded step={shadow_step}",
                reconstructed,
                trainable_param_names=trainable_param_names,
            )
            shadow_loaded_state = reconstructed
            shadow_reject_reason = None
            shadow_matches_base = None
            if base_template_state is not None:
                shadow_matches_base = _state_exact_matches(
                    reconstructed,
                    base_template_state,
                    trainable_param_names=trainable_param_names,
                )
                if consistency_log_enabled():
                    logger.info(
                        "[Resume] Shadow-vs-base comparison: "
                        f"shadow_step={shadow_step}, base_step={base_ref_step}, "
                        f"exact_match={bool(shadow_matches_base)}"
                    )

            if base_ref_step > 0 and (shadow_step is None or int(shadow_step) < base_ref_step):
                details = [
                    f"shadow_step={shadow_step}",
                    f"base_step={base_ref_step}",
                    f"log_retains_only_steps_gt={base_ref_step}",
                ]
                if shadow_matches_base is not None:
                    details.append(f"shadow_state_matches_base={shadow_matches_base}")
                shadow_reject_reason = (
                    "shadow snapshot is behind the current base checkpoint; "
                    "this indicates an incomplete rebase or a mixed-generation shadow bundle. "
                    + ", ".join(details)
                )
            elif (
                base_ref_step > 0
                and shadow_step is not None
                and int(shadow_step) > base_ref_step
                and bool(shadow_matches_base)
            ):
                shadow_reject_reason = (
                    "shadow snapshot header/content mismatch: "
                    f"shadow_step={shadow_step}, base_step={base_ref_step}, "
                    "but shadow model content exactly matches the base checkpoint. "
                    "This indicates a mixed-generation shadow bundle, so soft recovery is unsafe."
                )
            if shadow_reject_reason is not None:
                logger.error(f"[Resume] Shadow snapshot rejected: {shadow_reject_reason}")
                raise RuntimeError(
                    "Mixed-generation shadow snapshot detected during soft recovery. "
                    "Refusing to continue with a potentially misaligned model/adam/log state. "
                    f"Details: {shadow_reject_reason}"
                )
            else:
                updates = [u for u in updates if u['step'] > shadow_step]
                logger.info(f"[Resume] Soft recovery: shadow at step {shadow_step}, replaying {len(updates)} lag updates")
                shadow_used = True
    if reconstructed is None and base_state_dict is not None and base_checkpoint_ref == "__initial__":
        if consistency_log_enabled():
            logger.info("[Resume] Using pre-loaded base state dict in-place (no clone)")
        reconstructed = base_state_dict
        model_source = "base:__initial__"
        model_source_step = 0
    elif reconstructed is None:
        base_state, tied_groups = _load_base_state(
            base_checkpoint_ref, pretrained_model_name, tied_groups, model_dtype, output_dir=output_dir
        )
        reconstructed = base_state
        model_source = f"base:{base_checkpoint_ref}"
        model_source_step = int(base_ref_step)

    base_ready_state = base_template_state if base_template_state is not None else reconstructed
    _log_state_checksums(
        f"base_ready source={base_checkpoint_ref}",
        base_ready_state,
        trainable_param_names=trainable_param_names,
        tied_groups=tied_groups,
    )
    _log_state_exact_fingerprint(
        f"base_ready source={base_checkpoint_ref}",
        base_ready_state,
        trainable_param_names=trainable_param_names,
    )
    if shadow_used and shadow_loaded_state is not None:
        _log_state_exact_compare(
            f"shadow_vs_base_before_replay shadow_step={shadow_step} base={base_checkpoint_ref}",
            shadow_loaded_state,
            base_ready_state,
            trainable_param_names=trainable_param_names,
        )

    if time_log_enabled():
        logger.info(
            f"[Resume] Replaying {len(updates)} updates "
            f"(redo_source={redo_source}, default_zo_eps={default_zo_eps})"
        )
    log_start_step = int(updates[0]["step"]) if updates else None
    log_end_step = int(updates[-1]["step"]) if updates else None
    if consistency_log_enabled():
        logger.info(
            "[Resume Sources] "
            f"model_source={model_source} model_step={model_source_step} | "
            f"adam_source={adam_source} adam_step={adam_source_step} | "
            f"log_source={redo_source} log_start_step={log_start_step} log_end_step={log_end_step} "
            f"log_updates={len(updates)} target_step={target_step}"
        )
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
        _log_state_exact_fingerprint(
            "after_tie_before_replay",
            reconstructed,
            trainable_param_names=trainable_param_names,
        )

    _mem_proc = psutil.Process(os.getpid()) if resource_log_enabled() else None
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _mem_cpu0, _mem_gpu0 = _log_memory("before replay", _mem_proc, device) if _mem_proc is not None else (None, None)

    t_replay_start = time.time()
    with trace_span(
        panel="gpu_train",
        lane="blocking",
        event="replay_updates",
        step=int(target_step),
        counters={"num_updates": len(updates)},
        extra={"source": "resume"},
    ):
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
    _log_state_checksums(
        "after_replay",
        reconstructed,
        trainable_param_names=trainable_param_names,
        tied_groups=tied_groups,
    )
    _log_state_exact_fingerprint(
        "after_replay",
        reconstructed,
        trainable_param_names=trainable_param_names,
    )

    if is_adam and adam_state is not None:
        _log_adam_checksums("after_replay", adam_state)
        _log_adam_exact_fingerprint("after_replay", adam_state)
        _set_replay_adam_state(adam_state)
        if consistency_log_enabled():
            logger.info(f"[Resume] Cached replayed Adam state: t={adam_state.get('t', 0)}")

    if _mem_proc is not None:
        _log_memory("after replay", _mem_proc, device, _mem_cpu0, _mem_gpu0)

    if updates:
        last = updates[-1]
        if consistency_log_enabled():
            logger.info(f"[VERIFY-RESUME] Last replayed update: step={last.get('step','?')}, "
                        f"seed={last['seed']}, grad={last['grad']:.6e}")
    if pending_grad is not None:
        if consistency_log_enabled():
            logger.info(f"[VERIFY-RESUME] pending_grad={pending_grad} => first resumed step should apply this grad")

    if time_log_enabled():
        logger.info(f"[Resume] Completed! Recovered to step {target_step}")
    trace_instant(
        panel="gpu_train",
        lane="blocking",
        event="resume_end",
        step=int(target_step),
        counters={"num_updates": len(updates)},
    )
    recovered_base_step = 0
    if shadow_used and shadow_base_step is not None:
        recovered_base_step = int(shadow_base_step)
    elif base_checkpoint_ref != "__initial__":
        base_match = re.search(r'checkpoint-(\d+)', str(base_checkpoint_ref))
        if base_match:
            recovered_base_step = int(base_match.group(1))
    return LogBasedRecoveryBundle(
        state_dict=reconstructed,
        adam_state=adam_state,
        base_step=recovered_base_step,
        committed_step=int(shadow_step if shadow_used and shadow_step is not None else target_step),
        pending_grad=pending_grad,
        base_pending_seed=base_pending_seed,
        shadow_used=shadow_used,
    )


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
    """Backward-compatible wrapper that returns only the recovered model state dict."""
    bundle = resume_from_log_based_bundle(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        pretrained_model_name=pretrained_model_name,
        device=device,
        simulate_perturbation=simulate_perturbation,
        replay_in_fp32=replay_in_fp32,
        base_state_dict=base_state_dict,
        cached_optimizer_state=cached_optimizer_state,
        rng_device=rng_device,
        zo2_mode=zo2_mode,
        shadow_path=shadow_path,
    )
    return bundle.state_dict
