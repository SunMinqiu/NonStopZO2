import hashlib
import logging

import torch

from ...utils.logging_controls import z_diag_log_enabled, z_exact_log_enabled

logger = logging.getLogger(__name__)


_NO_WD_SUBSTRINGS = ("bias", "layer_norm", "layernorm", "ln")
_DEFAULT_Z_TRACKED_NAMES = ("model.embed_tokens.weight", "lm_head.weight")


def _resolve_param_tensor(param):
    if isinstance(param, torch.nn.Parameter):
        return param.data
    return param


def _uses_weight_decay(name):
    return all(token not in name for token in _NO_WD_SUBSTRINGS)


def _z_diag_enabled():
    return z_diag_log_enabled()


def _z_exact_enabled():
    return z_exact_log_enabled()


def _log_z_stats(label, z_tensors, _logger=None):
    _log = _logger or logger
    if not z_tensors:
        return

    total = 0.0
    tracked = {}
    digest = hashlib.sha256()
    tensor_count = 0
    for name, tensor in z_tensors:
        cpu = tensor.detach().float().cpu().contiguous()
        total += cpu.sum().item()
        tensor_count += 1
        if name in _DEFAULT_Z_TRACKED_NAMES:
            tracked[name] = cpu.sum().item()
        if _z_exact_enabled():
            digest.update(name.encode("utf-8"))
            digest.update(str(cpu.dtype).encode("utf-8"))
            digest.update(str(tuple(cpu.shape)).encode("utf-8"))
            digest.update(memoryview(cpu.numpy()).tobytes())

    if not tracked and z_tensors:
        first_name, first_tensor = z_tensors[0]
        tracked[first_name] = first_tensor.detach().float().cpu().sum().item()

    _log.info(f"[Z-CKSUM] {label}: sum={total:.10e} tensors={tensor_count}")
    for name, value in tracked.items():
        _log.info(f"[Z-CKSUM] {label}: {name}={value:.10e}")
    if _z_exact_enabled():
        _log.info(f"[Z-EXACT] {label}: sha256={digest.hexdigest()} tensors={tensor_count}")


def apply_mezo_adam_update(
    named_params,
    *,
    get_z,
    grad,
    lr,
    weight_decay,
    default_weight_decay,
    betas,
    adam_eps,
    t,
    m_state,
    v_state,
    state_key,
    diag_label=None,
    diag_logger=None,
):
    beta1, beta2 = betas
    bias_correction1 = 1 - beta1 ** t
    bias_correction2 = 1 - beta2 ** t
    step_size = lr / bias_correction1
    z_tensors = [] if diag_label and _z_diag_enabled() else None

    for name, param in named_params:
        param_tensor = _resolve_param_tensor(param)
        z = get_z(name, param_tensor)
        if z_tensors is not None:
            z_tensors.append((name, z))
        g = (grad * z).float()

        key = state_key(name, param)
        if key not in m_state:
            m_state[key] = torch.zeros_like(param_tensor, dtype=torch.float32)
            v_state[key] = torch.zeros_like(param_tensor, dtype=torch.float32)

        m = m_state[key]
        v = v_state[key]
        m.mul_(beta1).add_(g, alpha=1 - beta1)

        # Avoid addcmul_ here: the step-1 divergence we observed is isolated to
        # the v update path, while m matches exactly. Keeping the math as
        # v = beta2 * v + (1 - beta2) * (g * g) but spelling it out with mul/add
        # lets train/shadow/replay share the exact same operator sequence.
        g_sq = g.mul(g)
        v.mul_(beta2).add_(g_sq, alpha=1 - beta2)

        denom = (v / bias_correction2).sqrt_().add_(adam_eps)
        update = m.div(denom).mul_(step_size)

        wd = weight_decay
        if wd is None:
            wd = default_weight_decay if _uses_weight_decay(name) else 0.0
        if wd != 0.0:
            update.add_(param_tensor, alpha=lr * wd)

        param_tensor.sub_(update.to(param_tensor.dtype))

    if z_tensors is not None:
        _log_z_stats(diag_label, z_tensors, _logger=diag_logger)
