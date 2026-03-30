import json
import hashlib
import logging
import os
import tempfile
from collections import OrderedDict

import psutil
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ...utils.logging_controls import (
    resource_log_enabled,
    state_diag_log_enabled,
    state_exact_log_enabled,
    thread_snapshot_log_enabled,
)

logger = logging.getLogger(__name__)
_DEFAULT_ADAM_TRACKED_NAMES = ("model.embed_tokens.weight", "lm_head.weight")
_DEFAULT_STATE_TRACKED_NAMES = ("model.embed_tokens.weight", "lm_head.weight")

DEFAULT_ZO_SHM_DIR = "/dev/shm/zo_ckpt"

_DTYPE_MAP = {
    'torch.float16': torch.float16,
    'torch.bfloat16': torch.bfloat16,
    'torch.float32': torch.float32,
}


def _get_zo_shm_dir():
    """Return the directory used for ZO tmpfs checkpoint artifacts."""
    return os.environ.get("ZO_SHM_DIR", DEFAULT_ZO_SHM_DIR)


def _ensure_zo_shm_dir():
    """Create and return the directory used for ZO tmpfs checkpoint artifacts."""
    shm_dir = _get_zo_shm_dir()
    os.makedirs(shm_dir, exist_ok=True)
    return shm_dir


def _thread_debug_enabled():
    return thread_snapshot_log_enabled()


def _step_diag_enabled():
    return state_diag_log_enabled()


def _step_exact_enabled():
    return state_exact_log_enabled()

def _clone_state_dict_to_cpu(state_dict, *, exclude_keys=None):
    """Clone a state dict to regular CPU tensors."""
    cloned = OrderedDict()
    excluded = exclude_keys or set()
    for key, value in state_dict.items():
        if key in excluded:
            continue
        cloned[key] = value.detach().cpu().clone()
    return cloned


def _atomic_save_state_dict_safetensors(state_dict, path, metadata=None):
    """Atomically write a state_dict to a safetensors file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=os.path.dirname(path),
    )
    os.close(fd)
    try:
        safe_metadata = None
        if metadata is not None:
            safe_metadata = {str(k): str(v) for k, v in metadata.items()}
        save_file(state_dict, tmp_path, metadata=safe_metadata)
        _fsync_file(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _load_state_dict_safetensors(path):
    """Load tensors from a safetensors file."""
    return OrderedDict(load_file(path))


def _load_state_dict_safetensors_with_metadata(path):
    """Load tensors and metadata from a safetensors file."""
    metadata = {}
    tensors = OrderedDict()
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = dict(f.metadata() or {})
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors, metadata


def _thread_snapshot(label, _logger=None, detail=False):
    """Print a snapshot of all thread pools in the current process."""
    if not _thread_debug_enabled():
        return -1
    pid = os.getpid()
    try:
        os_thr = len(os.listdir(f'/proc/{pid}/task'))
    except Exception:
        os_thr = -1
    aten = torch.get_num_threads()
    interop = torch.get_num_interop_threads()
    zo_thr = -1
    try:
        import zo_rng as _zr
        zo_thr = _zr.get_num_threads()
    except ImportError:
        pass
    msg = (f"[ThreadSnap] {label}: os_thr={os_thr} "
           f"aten={aten} zo_rng={zo_thr} interop={interop}")
    if detail:
        from collections import Counter
        names = Counter()
        try:
            for tid in os.listdir(f'/proc/{pid}/task'):
                try:
                    with open(f'/proc/{pid}/task/{tid}/comm') as f:
                        names[f.read().strip()] += 1
                except Exception:
                    names['<unknown>'] += 1
        except Exception:
            pass
        if names:
            parts = [f"{name}={cnt}" for name, cnt in names.most_common()]
            msg += f"\n  [ThreadDetail] {' '.join(parts)}"
        import threading
        py_threads = [(t.name, t.ident, t.daemon) for t in threading.enumerate()]
        msg += f"\n  [PyThreads] count={len(py_threads)}"
        for tname, tid, daemon in py_threads:
            msg += f"\n    {tname} (tid={tid}, daemon={daemon})"
    _log = _logger or logger
    _log.info(msg)
    return os_thr


def _fsync_file(path):
    """Flush file to disk if output_dir is on a local filesystem."""
    if os.environ.get('FORCE_FSYNC', '0') == '1':
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _fsync_directory(dirpath):
    """Fsync all files in a checkpoint directory."""
    if os.environ.get('FORCE_FSYNC', '0') != '1':
        return
    for fname in os.listdir(dirpath):
        fpath = os.path.join(dirpath, fname)
        if os.path.isfile(fpath):
            _fsync_file(fpath)


def _detect_tied_weights(model) -> list:
    """Detect groups of tied parameters by shared storage."""
    ptr_to_names = {}
    for module_name, module in model.named_modules():
        for param_name, param in module._parameters.items():
            if param is None:
                continue
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            ptr = param.data_ptr()
            if ptr not in ptr_to_names:
                ptr_to_names[ptr] = []
            ptr_to_names[ptr].append(full_name)

    return [names for names in ptr_to_names.values() if len(names) > 1]


def _tie_state_dict_inplace(state: OrderedDict, tied_groups: list) -> None:
    """Make tied parameter groups share the same tensor in a state dict."""
    for group in tied_groups:
        primary = None
        for name in group:
            if name in state:
                primary = name
                break

        if primary is None:
            continue

        for name in group:
            if name != primary:
                state[name] = state[primary]


def _system_stats():
    """Return CPU, memory, and GPU memory stats."""
    cpu_pct = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    mem_used_gb = mem.used / 1024**3
    mem_total_gb = mem.total / 1024**3
    if torch.cuda.is_available():
        gpu_alloc_mb = torch.cuda.memory_allocated() / 1024**2
        gpu_reserved_mb = torch.cuda.memory_reserved() / 1024**2
    else:
        gpu_alloc_mb = 0
        gpu_reserved_mb = 0
    return cpu_pct, mem_used_gb, mem_total_gb, gpu_alloc_mb, gpu_reserved_mb


def _log_memory(tag, proc, device_type, baseline_cpu=None, baseline_gpu=None):
    """Log CPU RSS and GPU memory at a labeled checkpoint."""
    if not resource_log_enabled():
        return None, None
    cpu_rss = proc.memory_info().rss
    parts = [f"CPU RSS={cpu_rss / 1e9:.2f} GB"]
    if baseline_cpu is not None:
        parts.append(f"(delta={(cpu_rss - baseline_cpu) / 1e9:+.2f} GB)")
    gpu_alloc = None
    if device_type == 'cuda' and torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated()
        gpu_peak = torch.cuda.max_memory_allocated()
        parts.append(f"GPU alloc={gpu_alloc / 1e9:.2f} GB")
        parts.append(f"GPU peak={gpu_peak / 1e9:.2f} GB")
        if baseline_gpu is not None:
            parts.append(f"(delta={(gpu_alloc - baseline_gpu) / 1e9:+.2f} GB)")
    logger.info(f"[Memory] {tag}: {', '.join(parts)}")
    return cpu_rss, gpu_alloc


def _restore_tied_weights(state_dict, checkpoint_dir):
    """Restore tied weights that were deduplicated during saving."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path, 'r') as f:
        config = json.load(f)
    if config.get('tie_word_embeddings', False):
        if 'model.embed_tokens.weight' in state_dict and 'lm_head.weight' not in state_dict:
            state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight']


def _restore_tied_weights_for_model(state_dict, model):
    """Restore tied-weight secondary keys using the live model/config as source of truth."""
    tied_groups = _detect_tied_weights(model)
    if tied_groups:
        _tie_state_dict_inplace(state_dict, tied_groups)
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "tie_word_embeddings", False):
        if 'model.embed_tokens.weight' in state_dict and 'lm_head.weight' not in state_dict:
            state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight']


def _state_checksums(state_dict, tracked_names=None):
    if state_dict is None:
        return {"sum": 0.0, "tensors": 0, "tracked": {}}
    total = 0.0
    count = 0
    for tensor in state_dict.values():
        if torch.is_tensor(tensor):
            total += tensor.float().sum().item()
            count += 1
    tracked = {}
    for name in tracked_names or _DEFAULT_STATE_TRACKED_NAMES:
        tensor = state_dict.get(name)
        if torch.is_tensor(tensor):
            tracked[name] = tensor.float().sum().item()
    return {"sum": total, "tensors": count, "tracked": tracked}


def _state_exact_fingerprint(state_dict):
    if state_dict is None:
        return "", 0
    digest = hashlib.sha256()
    count = 0
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            continue
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("utf-8"))
        digest.update(str(tuple(cpu.shape)).encode("utf-8"))
        digest.update(memoryview(cpu.numpy()).tobytes())
        count += 1
    return digest.hexdigest(), count


def _log_state_checksums(label, state_dict, tracked_names=None, _logger=None):
    if not state_diag_log_enabled():
        return
    _log = _logger or logger
    if state_dict is None:
        _log.info(f"[STATE-CKSUM] {label}: no_state")
        return
    stats = _state_checksums(state_dict, tracked_names=tracked_names)
    _log.info(
        f"[STATE-CKSUM] {label}: "
        f"sum={stats['sum']:.10e} tensors={stats['tensors']}"
    )
    for name, value in stats["tracked"].items():
        _log.info(f"[STATE-CKSUM] {label}: {name}={value:.10e}")


def _log_state_exact_fingerprint(label, state_dict, _logger=None):
    if not state_exact_log_enabled():
        return
    _log = _logger or logger
    if state_dict is None:
        _log.info(f"[STATE-EXACT] {label}: no_state")
        return
    fp, tensor_count = _state_exact_fingerprint(state_dict)
    _log.info(f"[STATE-EXACT] {label}: sha256={fp} tensors={tensor_count}")


def _log_state_exact_compare(label, lhs, rhs, _logger=None):
    if not state_exact_log_enabled():
        return
    _log = _logger or logger
    if lhs is None or rhs is None:
        _log.info(
            f"[STATE-EXACT] {label}: exact_match=False "
            f"lhs_is_none={lhs is None} rhs_is_none={rhs is None}"
        )
        return

    lhs_names = {name for name, tensor in lhs.items() if torch.is_tensor(tensor)}
    rhs_names = {name for name, tensor in rhs.items() if torch.is_tensor(tensor)}
    if lhs_names != rhs_names:
        _log.info(
            f"[STATE-EXACT] {label}: exact_match=False name_mismatch "
            f"lhs_only={sorted(lhs_names - rhs_names)[:3]} rhs_only={sorted(rhs_names - lhs_names)[:3]}"
        )
        return

    for name in sorted(lhs_names):
        a = lhs[name].detach().cpu()
        b = rhs[name].detach().cpu()
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            _log.info(
                f"[STATE-EXACT] {label}: exact_match=False meta_mismatch "
                f"name={name} lhs_dtype={a.dtype} rhs_dtype={b.dtype} "
                f"lhs_shape={tuple(a.shape)} rhs_shape={tuple(b.shape)}"
            )
            return
        if not torch.equal(a, b):
            diff = (a.float() - b.float()).abs().max().item()
            _log.info(
                f"[STATE-EXACT] {label}: exact_match=False first_diff={name} "
                f"max_abs_diff={diff:.10e}"
            )
            return
    _log.info(f"[STATE-EXACT] {label}: exact_match=True")


def _adam_maps(adam_state):
    if not isinstance(adam_state, dict):
        return OrderedDict(), OrderedDict()
    return adam_state.get("m", {}) or OrderedDict(), adam_state.get("v", {}) or OrderedDict()


def _adam_state_checksums(adam_state):
    m_map, v_map = _adam_maps(adam_state)
    m_sum = 0.0
    v_sum = 0.0
    for tensor in m_map.values():
        m_sum += tensor.float().sum().item()
    for tensor in v_map.values():
        v_sum += tensor.float().sum().item()
    return {
        "m_sum": m_sum,
        "v_sum": v_sum,
        "m_tensors": len(m_map),
        "v_tensors": len(v_map),
        "t": int((adam_state or {}).get("t", 0)),
        "betas": tuple((adam_state or {}).get("betas", (0.0, 0.0))),
        "adam_eps": float((adam_state or {}).get("adam_eps", 0.0)),
    }


def _adam_exact_fingerprint(adam_state):
    m_map, v_map = _adam_maps(adam_state)
    digest = hashlib.sha256()
    digest.update(str(int((adam_state or {}).get("t", 0))).encode("utf-8"))
    digest.update(str(tuple((adam_state or {}).get("betas", (0.0, 0.0)))).encode("utf-8"))
    digest.update(str(float((adam_state or {}).get("adam_eps", 0.0))).encode("utf-8"))
    count = 0
    for mv_key, mapping in (("m", m_map), ("v", v_map)):
        for name in sorted(mapping.keys()):
            tensor = mapping[name].detach().cpu().contiguous()
            digest.update(mv_key.encode("utf-8"))
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(memoryview(tensor.numpy()).tobytes())
            count += 1
    return digest.hexdigest(), count


def _log_adam_brief(label, adam_state, tracked_names=None, _logger=None):
    if not state_diag_log_enabled():
        return
    _log = _logger or logger
    if adam_state is None:
        _log.info(f"[ADAM-BRIEF] {label}: no_state")
        return
    m_map, v_map = _adam_maps(adam_state)
    stats = _adam_state_checksums(adam_state)
    _log.info(
        f"[ADAM-BRIEF] {label}: "
        f"t={stats['t']} betas={stats['betas']} eps={stats['adam_eps']} "
        f"m_tensors={stats['m_tensors']} v_tensors={stats['v_tensors']}"
    )
    names = []
    for name in (tracked_names or _DEFAULT_ADAM_TRACKED_NAMES):
        if name in m_map or name in v_map:
            names.append(name)
    if not names:
        first_name = next(iter(m_map.keys()), None) or next(iter(v_map.keys()), None)
        if first_name is not None:
            names.append(first_name)
    for name in names:
        if name in m_map:
            _log.info(f"[ADAM-BRIEF] {label}: m[{name}]={m_map[name].float().sum().item():.10e}")
        if name in v_map:
            _log.info(f"[ADAM-BRIEF] {label}: v[{name}]={v_map[name].float().sum().item():.10e}")


def _log_adam_checksums(label, adam_state, tracked_names=None, _logger=None):
    if not state_diag_log_enabled():
        return
    _log = _logger or logger
    if adam_state is None:
        _log.info(f"[ADAM-CKSUM] {label}: no_state")
        return
    m_map, v_map = _adam_maps(adam_state)
    stats = _adam_state_checksums(adam_state)
    _log.info(
        f"[ADAM-CKSUM] {label}: "
        f"t={stats['t']} betas={stats['betas']} eps={stats['adam_eps']} "
        f"m_sum={stats['m_sum']:.10e} v_sum={stats['v_sum']:.10e} "
        f"m_tensors={stats['m_tensors']} v_tensors={stats['v_tensors']}"
    )
    for name in tracked_names or _DEFAULT_ADAM_TRACKED_NAMES:
        if name in m_map:
            _log.info(f"[ADAM-CKSUM] {label}: m[{name}]={m_map[name].float().sum().item():.10e}")
        if name in v_map:
            _log.info(f"[ADAM-CKSUM] {label}: v[{name}]={v_map[name].float().sum().item():.10e}")


def _log_adam_exact_fingerprint(label, adam_state, _logger=None):
    if not state_exact_log_enabled():
        return
    _log = _logger or logger
    if adam_state is None:
        _log.info(f"[ADAM-EXACT] {label}: no_state")
        return
    fp, tensor_count = _adam_exact_fingerprint(adam_state)
    _log.info(f"[ADAM-EXACT] {label}: sha256={fp} tensors={tensor_count}")


def _log_adam_exact_compare(label, lhs, rhs, _logger=None):
    if not state_exact_log_enabled():
        return
    _log = _logger or logger
    if lhs is None or rhs is None:
        _log.info(
            f"[ADAM-EXACT] {label}: exact_match=False "
            f"lhs_is_none={lhs is None} rhs_is_none={rhs is None}"
        )
        return
    lhs_m, lhs_v = _adam_maps(lhs)
    rhs_m, rhs_v = _adam_maps(rhs)
    meta_keys = ("t", "betas", "adam_eps")
    meta_mismatch = {
        key: ((lhs or {}).get(key), (rhs or {}).get(key))
        for key in meta_keys
        if (lhs or {}).get(key) != (rhs or {}).get(key)
    }
    if meta_mismatch:
        _log.info(f"[ADAM-EXACT] {label}: exact_match=False meta_mismatch={meta_mismatch}")
        return
    for mv_key, lhs_map, rhs_map in (("m", lhs_m, rhs_m), ("v", lhs_v, rhs_v)):
        lhs_names = set(lhs_map.keys())
        rhs_names = set(rhs_map.keys())
        if lhs_names != rhs_names:
            _log.info(
                f"[ADAM-EXACT] {label}: exact_match=False {mv_key}_name_mismatch "
                f"lhs_only={sorted(lhs_names - rhs_names)[:3]} rhs_only={sorted(rhs_names - lhs_names)[:3]}"
            )
            return
        for name in sorted(lhs_names):
            a = lhs_map[name].detach().cpu()
            b = rhs_map[name].detach().cpu()
            if not torch.equal(a, b):
                diff = (a.float() - b.float()).abs().max().item()
                _log.info(
                    f"[ADAM-EXACT] {label}: exact_match=False first_diff={mv_key}[{name}] "
                    f"max_abs_diff={diff:.10e}"
                )
                return
    _log.info(f"[ADAM-EXACT] {label}: exact_match=True")
