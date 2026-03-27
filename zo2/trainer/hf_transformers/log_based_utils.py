import json
import logging
import os
import tempfile
from collections import OrderedDict

import psutil
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    'torch.float16': torch.float16,
    'torch.bfloat16': torch.bfloat16,
    'torch.float32': torch.float32,
}

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
