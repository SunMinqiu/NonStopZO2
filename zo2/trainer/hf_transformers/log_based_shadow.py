import logging
import json
import hashlib
import mmap
import os
import queue as queue_module
import shutil
import tempfile
import threading
import time
from collections import OrderedDict

import psutil
import torch

from ...utils.logging_controls import (
    loading_phase_log_enabled,
    resource_log_enabled,
    shadow_step_resource_log_enabled,
    shadow_step_time_log_enabled,
    time_log_enabled,
)
from ...utils.trace import (
    configure_trace,
    shm_usage_bytes,
    start_resource_sampler,
    stop_resource_sampler,
    trace_begin,
    trace_enabled,
    trace_end,
    trace_instant,
    trace_span,
)
from .log_based_utils import (
    _DTYPE_MAP,
    _atomic_save_state_dict_safetensors,
    _log_adam_brief,
    _log_adam_checksums,
    _log_adam_exact_fingerprint,
    _log_state_checksums,
    _log_state_exact_fingerprint,
    _load_state_dict_safetensors_with_metadata,
    _step_diag_enabled,
    _step_exact_enabled,
    _thread_snapshot,
    _tie_state_dict_inplace,
)

logger = logging.getLogger(__name__)


def _shadow_resource_counters(shadow_step_val, process):
    counters = {
        "shadow_rss_mb": process.memory_info().rss / 1024**2,
        "shadow_cpu_percent": process.cpu_percent(interval=None),
        "shadow_num_threads": int(process.num_threads()),
        "shadow_durable_step": int(shadow_step_val.value) if shadow_step_val is not None else -1,
        "zo_shm_used_mb": shm_usage_bytes() / 1024**2,
    }
    return counters


def _secondary_tied_keys(tied_groups):
    excluded = set()
    for group in tied_groups or []:
        for name in group[1:]:
            excluded.add(name)
    return excluded


def _log_durable_publish(step, state_dict, adam_state, _logger):
    label = f"shadow_durable step={int(step)}"
    if _step_diag_enabled():
        _log_state_checksums(label, state_dict, _logger=_logger)
    if _step_exact_enabled():
        _log_state_exact_fingerprint(label, state_dict, _logger=_logger)
    if adam_state is not None:
        if _step_diag_enabled():
            _log_adam_checksums(label, adam_state, _logger=_logger)
        if _step_exact_enabled():
            _log_adam_exact_fingerprint(label, adam_state, _logger=_logger)


def _build_shadow_flat_layout(state_dict, tied_groups=None):
    excluded = _secondary_tied_keys(tied_groups)
    entries = []
    offset = 0
    for name, tensor in state_dict.items():
        if name in excluded:
            continue
        tensor = tensor.detach()
        numel = int(tensor.numel())
        nbytes = int(numel * tensor.element_size())
        entries.append(
            {
                "name": name,
                "offset": offset,
                "numel": numel,
                "shape": tuple(tensor.shape),
                "dtype": tensor.dtype,
            }
        )
        offset += nbytes
    return {"entries": entries, "total_bytes": offset}


def _build_adam_flat_layout(state_dict, param_names):
    entries = []
    offset = 0
    for name in param_names or []:
        if name not in state_dict:
            continue
        tensor = state_dict[name].detach()
        numel = int(tensor.numel())
        nbytes = int(numel * 4)  # Adam m/v are always fp32
        entries.append(
            {
                "name": name,
                "offset": offset,
                "numel": numel,
                "shape": tuple(tensor.shape),
                "dtype": torch.float32,
            }
        )
        offset += nbytes
    return {"entries": entries, "total_bytes": offset}


def _serialize_flat_layout(layout):
    return {
        "entries": [
            {
                "name": entry["name"],
                "offset": int(entry["offset"]),
                "numel": int(entry["numel"]),
                "shape": list(entry["shape"]),
                "dtype": str(entry["dtype"]),
            }
            for entry in layout["entries"]
        ],
        "total_bytes": int(layout["total_bytes"]),
    }


def _deserialize_flat_layout(layout_payload):
    return {
        "entries": [
            {
                "name": entry["name"],
                "offset": int(entry["offset"]),
                "numel": int(entry["numel"]),
                "shape": tuple(entry["shape"]),
                "dtype": _DTYPE_MAP[entry["dtype"]],
            }
            for entry in layout_payload["entries"]
        ],
        "total_bytes": int(layout_payload["total_bytes"]),
    }


def _rebase_payload_paths(header_path):
    if header_path.endswith(".header.json"):
        stem = header_path[: -len(".header.json")]
    elif header_path.endswith(".json"):
        stem = header_path[: -len(".json")]
    else:
        stem = header_path
    return {
        "state_path": f"{stem}.bin",
        "adam_m_path": f"{stem}.adam_m.bin",
        "adam_v_path": f"{stem}.adam_v.bin",
    }


def _shadow_flat_meta_paths(header_path):
    if header_path.endswith(".header.json"):
        stem = header_path[: -len(".header.json")]
    elif header_path.endswith(".json"):
        stem = header_path[: -len(".json")]
    else:
        stem = header_path
    return {
        "state_meta_path": f"{stem}.state.meta.json",
        "adam_meta_path": f"{stem}.adam.meta.json",
        "expected_generation_path": f"{stem}.generation.meta.json",
    }


def _write_shadow_flat_header(header_path, header):
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(header_path)}.",
        suffix=".tmp",
        dir=os.path.dirname(header_path),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(header, f)
            f.flush()
        os.replace(tmp_path, header_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _read_shadow_flat_header(header_path):
    with open(header_path, "r") as f:
        return json.load(f)


_SHADOW_INTEGRITY_FULL_SHA256 = "full_sha256"
_SHADOW_INTEGRITY_HEADER_ONLY = "header_only"


def _normalize_shadow_integrity_mode(mode, payload=None):
    if mode in (_SHADOW_INTEGRITY_FULL_SHA256, _SHADOW_INTEGRITY_HEADER_ONLY):
        return mode
    payload = payload or {}
    if (
        payload.get("state_sha256")
        or payload.get("adam_m_sha256")
        or payload.get("adam_v_sha256")
    ):
        return _SHADOW_INTEGRITY_FULL_SHA256
    return _SHADOW_INTEGRITY_HEADER_ONLY


def _shadow_generation_meta(expected_generation, reason):
    return {
        "kind": "shadow_generation",
        "expected_generation": int(expected_generation),
        "reason": str(reason),
    }


def _read_expected_shadow_generation(expected_generation_path, default=None):
    if not expected_generation_path or not os.path.exists(expected_generation_path):
        return default
    payload = _read_shadow_flat_header(expected_generation_path)
    return int(payload.get("expected_generation", default))


def _shadow_bundle_header(
    *,
    base_step=0,
    committed_step=0,
    generation=0,
    has_adam=False,
    adam_state=None,
    layout=None,
    adam_layout=None,
    snapshot_state="ready",
    integrity_mode=_SHADOW_INTEGRITY_HEADER_ONLY,
):
    adam_state = adam_state or {}
    beta1, beta2 = adam_state.get("betas", (0.0, 0.0))
    header = {
        "snapshot_state": str(snapshot_state),
        "base_step": int(base_step),
        "committed_step": int(committed_step),
        "generation": int(generation),
        "has_adam": bool(has_adam),
        "integrity_mode": _normalize_shadow_integrity_mode(integrity_mode),
        "adam_t": int(adam_state.get("t", 0)) if has_adam else 0,
        "adam_beta1": float(beta1) if has_adam else 0.0,
        "adam_beta2": float(beta2) if has_adam else 0.0,
        "adam_eps": float(adam_state.get("adam_eps", 0.0)) if has_adam else 0.0,
    }
    if layout is not None:
        header["layout"] = _serialize_flat_layout(layout)
    if adam_layout is not None:
        header["adam_layout"] = _serialize_flat_layout(adam_layout)
    return header

def _shadow_component_meta(
    *,
    kind,
    generation,
    step,
    snapshot_state="ready",
    adam_t=None,
    integrity_mode=_SHADOW_INTEGRITY_HEADER_ONLY,
):
    payload = {
        "kind": str(kind),
        "snapshot_state": str(snapshot_state),
        "generation": int(generation),
        "committed_step": int(step),
        "integrity_mode": _normalize_shadow_integrity_mode(integrity_mode),
    }
    if adam_t is not None:
        payload["adam_t"] = int(adam_t)
    return payload


def _hash_named_tensors(named_tensors):
    digest = hashlib.sha256()
    count = 0
    for name, tensor in named_tensors:
        if not torch.is_tensor(tensor):
            continue
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("utf-8"))
        digest.update(str(tuple(cpu.shape)).encode("utf-8"))
        digest.update(memoryview(cpu.numpy()).tobytes())
        count += 1
    return digest.hexdigest(), count


def _ensure_shadow_flat_files(buffer_paths, total_bytes):
    if total_bytes <= 0:
        raise ValueError("shadow flat storage requires a positive total_bytes")
    for path in buffer_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, total_bytes)
        finally:
            os.close(fd)


def _open_shadow_flat_views(layout, buffer_path):
    total_bytes = int(layout["total_bytes"])
    fd = os.open(buffer_path, os.O_RDWR)
    mm = mmap.mmap(fd, total_bytes, access=mmap.ACCESS_WRITE)
    views = OrderedDict()
    for entry in layout["entries"]:
        tensor = torch.frombuffer(
            mm,
            dtype=entry["dtype"],
            count=entry["numel"],
            offset=entry["offset"],
        ).view(entry["shape"])
        views[entry["name"]] = tensor
    return fd, mm, views


def _close_shadow_flat_views(fd, mm, views=None):
    if views is not None:
        views.clear()
    try:
        mm.close()
    finally:
        os.close(fd)


def _copy_shadow_flat_views_from_state(views, state_dict):
    for name, target in views.items():
        target.copy_(state_dict[name], non_blocking=False)


def _copy_shadow_flat_views_from_adam(views, source_map):
    source_map = source_map or {}
    for name, target in views.items():
        src = source_map.get(name)
        if src is None:
            target.zero_()
        else:
            target.copy_(src.to(device="cpu", dtype=torch.float32), non_blocking=False)


def _ensure_shadow_bundle_flat_files(flat_storage):
    _ensure_shadow_flat_files(flat_storage["buffer_paths"], int(flat_storage["layout"]["total_bytes"]))
    if flat_storage.get("has_adam", False):
        adam_layout = flat_storage["adam_layout"]
        _ensure_shadow_flat_files(flat_storage["adam_m_buffer_paths"], int(adam_layout["total_bytes"]))
        _ensure_shadow_flat_files(flat_storage["adam_v_buffer_paths"], int(adam_layout["total_bytes"]))


def _open_shadow_bundle_flat_writer(flat_storage):
    _ensure_shadow_bundle_flat_files(flat_storage)
    header = (
        _read_shadow_flat_header(flat_storage["header_path"])
        if os.path.exists(flat_storage["header_path"])
        else _shadow_bundle_header(has_adam=flat_storage.get("has_adam", False))
    )

    fd, mm, tensor_views = _open_shadow_flat_views(flat_storage["layout"], flat_storage["buffer_paths"][0])
    fds = [fd]
    mmaps = [mm]
    views = [tensor_views]

    adam_m_fds, adam_m_mmaps, adam_m_views = [], [], []
    adam_v_fds, adam_v_mmaps, adam_v_views = [], [], []
    if flat_storage.get("has_adam", False):
        fd, mm, tensor_views = _open_shadow_flat_views(flat_storage["adam_layout"], flat_storage["adam_m_buffer_paths"][0])
        adam_m_fds.append(fd)
        adam_m_mmaps.append(mm)
        adam_m_views.append(tensor_views)
        fd, mm, tensor_views = _open_shadow_flat_views(flat_storage["adam_layout"], flat_storage["adam_v_buffer_paths"][0])
        adam_v_fds.append(fd)
        adam_v_mmaps.append(mm)
        adam_v_views.append(tensor_views)

    return {
        "header_path": flat_storage["header_path"],
        "state_meta_path": flat_storage.get("state_meta_path"),
        "adam_meta_path": flat_storage.get("adam_meta_path"),
        "expected_generation_path": flat_storage.get("expected_generation_path"),
        "buffer_paths": tuple(flat_storage["buffer_paths"]),
        "layout": flat_storage["layout"],
        "fds": fds,
        "mmaps": mmaps,
        "views": views,
        "generation": int(header.get("generation", 0)),
        "needs_full_integrity": False,
        "has_adam": bool(header.get("has_adam", flat_storage.get("has_adam", False))),
        "adam_layout": flat_storage.get("adam_layout"),
        "adam_m_buffer_paths": tuple(flat_storage.get("adam_m_buffer_paths", ())),
        "adam_v_buffer_paths": tuple(flat_storage.get("adam_v_buffer_paths", ())),
        "adam_m_fds": adam_m_fds,
        "adam_m_mmaps": adam_m_mmaps,
        "adam_m_views": adam_m_views,
        "adam_v_fds": adam_v_fds,
        "adam_v_mmaps": adam_v_mmaps,
        "adam_v_views": adam_v_views,
    }


def _close_shadow_bundle_flat_writer(flat_writer):
    for view_dict in flat_writer.get("views", []):
        view_dict.clear()
    for mm in flat_writer.get("mmaps", []):
        mm.close()
    for fd in flat_writer.get("fds", []):
        os.close(fd)

    for view_dict in flat_writer.get("adam_m_views", []):
        view_dict.clear()
    for mm in flat_writer.get("adam_m_mmaps", []):
        mm.close()
    for fd in flat_writer.get("adam_m_fds", []):
        os.close(fd)

    for view_dict in flat_writer.get("adam_v_views", []):
        view_dict.clear()
    for mm in flat_writer.get("adam_v_mmaps", []):
        mm.close()
    for fd in flat_writer.get("adam_v_fds", []):
        os.close(fd)


def _init_shadow_bundle_flat_storage(
    state_dict,
    flat_storage,
    base_step,
    committed_step,
    tied_groups=None,
    adam_state=None,
):
    flat_writer = _open_shadow_bundle_flat_writer(flat_storage)
    try:
        generation = _reserve_shadow_epoch(flat_writer, reason="init")
        integrity_mode = _SHADOW_INTEGRITY_FULL_SHA256
        _write_shadow_flat_header(
            flat_writer["header_path"],
            _shadow_bundle_header(
                base_step=base_step,
                committed_step=committed_step,
                generation=generation,
                has_adam=flat_writer.get("has_adam", False),
                adam_state=adam_state,
                layout=flat_writer["layout"],
                adam_layout=flat_writer.get("adam_layout"),
                snapshot_state="writing",
                integrity_mode=integrity_mode,
            ),
        )
        _write_shadow_flat_header(
            flat_writer["state_meta_path"],
            _shadow_component_meta(
                kind="state",
                generation=generation,
                step=committed_step,
                snapshot_state="writing",
                integrity_mode=integrity_mode,
            ),
        )
        if flat_writer.get("has_adam", False):
            _write_shadow_flat_header(
                flat_writer["adam_meta_path"],
                _shadow_component_meta(
                    kind="adam",
                    generation=generation,
                    step=committed_step,
                    snapshot_state="writing",
                    adam_t=int((adam_state or {}).get("t", 0)),
                    integrity_mode=integrity_mode,
                ),
            )
        _copy_shadow_flat_views_from_state(flat_writer["views"][0], state_dict)
        flat_writer["mmaps"][0].flush()
        state_sha256, _ = _hash_named_tensors(flat_writer["views"][0].items())
        _write_shadow_flat_header(
            flat_writer["state_meta_path"],
            {
                **_shadow_component_meta(
                    kind="state",
                    generation=generation,
                    step=committed_step,
                    snapshot_state="ready",
                    integrity_mode=integrity_mode,
                ),
                "state_sha256": state_sha256,
            },
        )
        if flat_writer.get("has_adam", False):
            _copy_shadow_flat_views_from_adam(flat_writer["adam_m_views"][0], (adam_state or {}).get("m"))
            _copy_shadow_flat_views_from_adam(flat_writer["adam_v_views"][0], (adam_state or {}).get("v"))
            flat_writer["adam_m_mmaps"][0].flush()
            flat_writer["adam_v_mmaps"][0].flush()
            adam_m_sha256, _ = _hash_named_tensors(flat_writer["adam_m_views"][0].items())
            adam_v_sha256, _ = _hash_named_tensors(flat_writer["adam_v_views"][0].items())
            _write_shadow_flat_header(
                flat_writer["adam_meta_path"],
                {
                    **_shadow_component_meta(
                        kind="adam",
                        generation=generation,
                        step=committed_step,
                        snapshot_state="ready",
                        adam_t=int((adam_state or {}).get("t", 0)),
                        integrity_mode=integrity_mode,
                    ),
                    "adam_m_sha256": adam_m_sha256,
                    "adam_v_sha256": adam_v_sha256,
                },
            )
        _write_shadow_flat_header(
            flat_writer["header_path"],
            _shadow_bundle_header(
                base_step=base_step,
                committed_step=committed_step,
                generation=generation,
                has_adam=flat_writer.get("has_adam", False),
                adam_state=adam_state,
                layout=flat_writer["layout"],
                adam_layout=flat_writer.get("adam_layout"),
                snapshot_state="ready",
                integrity_mode=integrity_mode,
            ),
        )
        flat_writer["generation"] = generation
        flat_writer["needs_full_integrity"] = False
    finally:
        _close_shadow_bundle_flat_writer(flat_writer)


def _open_shadow_flat_writer(flat_storage):
    return _open_shadow_bundle_flat_writer(flat_storage)


def _reserve_shadow_epoch(flat_writer, *, reason):
    current_generation = int(flat_writer.get("generation", 0))
    expected_generation = _read_expected_shadow_generation(
        flat_writer.get("expected_generation_path"),
        default=current_generation,
    )
    next_generation = max(current_generation, int(expected_generation or 0)) + 1
    expected_generation_path = flat_writer.get("expected_generation_path")
    if expected_generation_path:
        _write_shadow_flat_header(
            expected_generation_path,
            _shadow_generation_meta(next_generation, reason),
        )
    flat_writer["generation"] = next_generation
    flat_writer["needs_full_integrity"] = True
    return next_generation


def _commit_shadow_bundle_flat(state_dict, adam_state, flat_writer, base_step, committed_step):
    generation = int(flat_writer.get("generation", 0))
    if generation <= 0:
        generation = _reserve_shadow_epoch(flat_writer, reason="implicit_init")
    integrity_mode = (
        _SHADOW_INTEGRITY_FULL_SHA256
        if flat_writer.get("needs_full_integrity", False)
        else _SHADOW_INTEGRITY_HEADER_ONLY
    )
    _write_shadow_flat_header(
        flat_writer["header_path"],
        _shadow_bundle_header(
            base_step=base_step,
            committed_step=committed_step,
            generation=generation,
            has_adam=flat_writer.get("has_adam", False),
            adam_state=adam_state,
            layout=flat_writer["layout"],
            adam_layout=flat_writer.get("adam_layout"),
            snapshot_state="writing",
            integrity_mode=integrity_mode,
        ),
    )
    _write_shadow_flat_header(
        flat_writer["state_meta_path"],
        _shadow_component_meta(
            kind="state",
            generation=generation,
            step=committed_step,
            snapshot_state="writing",
            integrity_mode=integrity_mode,
        ),
    )
    if flat_writer.get("has_adam", False):
        _write_shadow_flat_header(
            flat_writer["adam_meta_path"],
            _shadow_component_meta(
                kind="adam",
                generation=generation,
                step=committed_step,
                snapshot_state="writing",
                adam_t=int((adam_state or {}).get("t", 0)),
                integrity_mode=integrity_mode,
            ),
        )
    target_views = flat_writer["views"][0]
    _copy_shadow_flat_views_from_state(target_views, state_dict)
    flat_writer["mmaps"][0].flush()
    state_meta = _shadow_component_meta(
        kind="state",
        generation=generation,
        step=committed_step,
        snapshot_state="ready",
        integrity_mode=integrity_mode,
    )
    if integrity_mode == _SHADOW_INTEGRITY_FULL_SHA256:
        state_sha256, _ = _hash_named_tensors(target_views.items())
        state_meta["state_sha256"] = state_sha256
    _write_shadow_flat_header(flat_writer["state_meta_path"], state_meta)
    if flat_writer.get("has_adam", False):
        _copy_shadow_flat_views_from_adam(
            flat_writer["adam_m_views"][0],
            (adam_state or {}).get("m"),
        )
        _copy_shadow_flat_views_from_adam(
            flat_writer["adam_v_views"][0],
            (adam_state or {}).get("v"),
        )
        flat_writer["adam_m_mmaps"][0].flush()
        flat_writer["adam_v_mmaps"][0].flush()
        adam_meta = _shadow_component_meta(
            kind="adam",
            generation=generation,
            step=committed_step,
            snapshot_state="ready",
            adam_t=int((adam_state or {}).get("t", 0)),
            integrity_mode=integrity_mode,
        )
        if integrity_mode == _SHADOW_INTEGRITY_FULL_SHA256:
            adam_m_sha256, _ = _hash_named_tensors(flat_writer["adam_m_views"][0].items())
            adam_v_sha256, _ = _hash_named_tensors(flat_writer["adam_v_views"][0].items())
            adam_meta["adam_m_sha256"] = adam_m_sha256
            adam_meta["adam_v_sha256"] = adam_v_sha256
        _write_shadow_flat_header(flat_writer["adam_meta_path"], adam_meta)
    _write_shadow_flat_header(
        flat_writer["header_path"],
        _shadow_bundle_header(
            base_step=base_step,
            committed_step=committed_step,
            generation=generation,
            has_adam=flat_writer.get("has_adam", False),
            adam_state=adam_state,
            layout=flat_writer["layout"],
            adam_layout=flat_writer.get("adam_layout"),
            snapshot_state="ready",
            integrity_mode=integrity_mode,
        ),
    )
    flat_writer["generation"] = generation
    flat_writer["needs_full_integrity"] = False
    return {
        "storage": "flat",
        "base_step": int(base_step),
        "committed_step": int(committed_step),
        "generation": int(generation),
        "integrity_mode": integrity_mode,
        "state_ready": True,
        "adam_ready": bool(flat_writer.get("has_adam", False)),
        "header_ready": True,
    }


def _commit_shadow_state_flat(state_dict, flat_writer, base_step, committed_step):
    _commit_shadow_bundle_flat(
        state_dict,
        adam_state=None,
        flat_writer=flat_writer,
        base_step=base_step,
        committed_step=committed_step,
    )


def _load_shadow_bundle_flat(flat_storage, tied_groups=None):
    import time as _time
    _t0_total = _time.perf_counter()

    # --- Phase 1: Read header JSON ---
    _t0 = _time.perf_counter()
    header = _read_shadow_flat_header(flat_storage["header_path"])
    _t_read_header = _time.perf_counter() - _t0

    if header.get("snapshot_state", "ready") != "ready":
        raise RuntimeError(
            f"shadow flat snapshot {flat_storage['header_path']} is incomplete "
            f"(snapshot_state={header.get('snapshot_state')})"
        )
    generation = int(header.get("generation", -1))
    expected_generation = _read_expected_shadow_generation(
        flat_storage.get("expected_generation_path"),
        default=generation,
    )
    if int(expected_generation) != generation:
        raise RuntimeError(
            f"shadow flat snapshot {flat_storage['header_path']} has stale or mismatched generation "
            f"(expected_generation={expected_generation}, snapshot_generation={generation})"
        )
    integrity_mode = _normalize_shadow_integrity_mode(header.get("integrity_mode"), header)

    # --- Phase 2: Deserialize layout ---
    _t0 = _time.perf_counter()
    layout = (
        _deserialize_flat_layout(header["layout"])
        if "layout" in header else flat_storage["layout"]
    )
    _t_deserialize_layout = _time.perf_counter() - _t0

    # --- Phase 3: Read state metadata JSON ---
    state_meta_path = flat_storage.get("state_meta_path")
    if not state_meta_path or not os.path.exists(state_meta_path):
        raise RuntimeError(
            f"shadow flat snapshot {flat_storage['header_path']} is missing state metadata. "
            "This usually means /dev/shm still contains a legacy single-buffer shadow snapshot "
            "from the old protocol. Delete the existing zo_shadow_latest_*.flat* files and rerun "
            "so shadow can be reinitialized with the new state/adam/header metadata format."
        )
    _t0 = _time.perf_counter()
    state_meta = _read_shadow_flat_header(state_meta_path)
    _t_read_state_meta = _time.perf_counter() - _t0

    if (
        state_meta.get("snapshot_state") != "ready"
        or int(state_meta.get("generation", -2)) != generation
        or int(state_meta.get("committed_step", -3)) != int(header.get("committed_step", -4))
        or _normalize_shadow_integrity_mode(state_meta.get("integrity_mode"), state_meta) != integrity_mode
    ):
        raise RuntimeError(
            f"shadow flat snapshot {flat_storage['header_path']} has inconsistent state metadata "
            f"(header_gen={generation}, state_gen={state_meta.get('generation')}, "
            f"header_step={header.get('committed_step')}, state_step={state_meta.get('committed_step')}, "
            f"state_snapshot={state_meta.get('snapshot_state')}, "
            f"header_integrity={integrity_mode}, state_integrity={state_meta.get('integrity_mode')})"
        )

    # --- Phase 4: mmap open + view creation for state ---
    _t0 = _time.perf_counter()
    fd, mm, views = _open_shadow_flat_views(layout, flat_storage["buffer_paths"][0])
    _t_mmap_open_state = _time.perf_counter() - _t0

    # --- Phase 5: Clone state tensors from mmap ---
    try:
        _t0 = _time.perf_counter()
        state_dict = OrderedDict(
            (name, tensor.clone()) for name, tensor in views.items()
        )
        _t_clone_state = _time.perf_counter() - _t0

        # --- Phase 6: SHA256 verification (state) ---
        _t0 = _time.perf_counter()
        loaded_state_sha256 = None
        if integrity_mode == _SHADOW_INTEGRITY_FULL_SHA256:
            loaded_state_sha256, _ = _hash_named_tensors(state_dict.items())
        _t_sha256_state = _time.perf_counter() - _t0
    finally:
        _t0 = _time.perf_counter()
        _close_shadow_flat_views(fd, mm, views)
        _t_mmap_close_state = _time.perf_counter() - _t0

    expected_state_sha256 = state_meta.get("state_sha256")
    if integrity_mode == _SHADOW_INTEGRITY_FULL_SHA256 and loaded_state_sha256 != expected_state_sha256:
        raise RuntimeError(
            f"shadow flat snapshot {flat_storage['header_path']} failed state content verification "
            f"(expected_state_sha256={expected_state_sha256}, loaded_state_sha256={loaded_state_sha256})"
        )

    # --- Phase 7: Tie weights ---
    _t0 = _time.perf_counter()
    if tied_groups:
        _tie_state_dict_inplace(state_dict, tied_groups)
    _t_tie_weights = _time.perf_counter() - _t0

    # Compute state buffer size for bandwidth calculation
    _state_bytes = sum(t.nelement() * t.element_size() for t in state_dict.values())

    adam_state = None
    _t_read_adam_meta = 0.0
    _t_mmap_open_adam = 0.0
    _t_clone_adam = 0.0
    _t_sha256_adam = 0.0
    _t_mmap_close_adam = 0.0
    _adam_bytes = 0
    if bool(header.get("has_adam", flat_storage.get("has_adam", False))):
        adam_layout = (
            _deserialize_flat_layout(header["adam_layout"])
            if "adam_layout" in header else flat_storage["adam_layout"]
        )
        adam_meta_path = flat_storage.get("adam_meta_path")
        if not adam_meta_path or not os.path.exists(adam_meta_path):
            raise RuntimeError(
                f"shadow flat snapshot {flat_storage['header_path']} is missing adam metadata. "
                "This usually means /dev/shm still contains a legacy single-buffer shadow snapshot "
                "from the old protocol. Delete the existing zo_shadow_latest_*.flat* files and rerun "
                "so shadow can be reinitialized with the new state/adam/header metadata format."
            )

        # --- Phase 8: Read adam metadata JSON ---
        _t0 = _time.perf_counter()
        adam_meta = _read_shadow_flat_header(adam_meta_path)
        _t_read_adam_meta = _time.perf_counter() - _t0

        if (
            adam_meta.get("snapshot_state") != "ready"
            or int(adam_meta.get("generation", -2)) != generation
            or int(adam_meta.get("committed_step", -3)) != int(header.get("committed_step", -4))
            or int(adam_meta.get("adam_t", -5)) != int(header.get("adam_t", -6))
            or _normalize_shadow_integrity_mode(adam_meta.get("integrity_mode"), adam_meta) != integrity_mode
        ):
            raise RuntimeError(
                f"shadow flat snapshot {flat_storage['header_path']} has inconsistent adam metadata "
                f"(header_gen={generation}, adam_gen={adam_meta.get('generation')}, "
                f"header_step={header.get('committed_step')}, adam_step={adam_meta.get('committed_step')}, "
                f"header_t={header.get('adam_t')}, adam_t={adam_meta.get('adam_t')}, "
                f"adam_snapshot={adam_meta.get('snapshot_state')}, "
                f"header_integrity={integrity_mode}, adam_integrity={adam_meta.get('integrity_mode')})"
            )

        # --- Phase 9: mmap open adam_m + adam_v ---
        _t0 = _time.perf_counter()
        adam_m_fd, adam_m_mm, adam_m_views = _open_shadow_flat_views(
            adam_layout,
            flat_storage["adam_m_buffer_paths"][0],
        )
        adam_v_fd, adam_v_mm, adam_v_views = _open_shadow_flat_views(
            adam_layout,
            flat_storage["adam_v_buffer_paths"][0],
        )
        _t_mmap_open_adam = _time.perf_counter() - _t0

        # --- Phase 10: Clone adam tensors ---
        try:
            _t0 = _time.perf_counter()
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
            _t_clone_adam = _time.perf_counter() - _t0

            _adam_bytes = sum(t.nelement() * t.element_size() for t in adam_state["m"].values()) + \
                          sum(t.nelement() * t.element_size() for t in adam_state["v"].values())

            # --- Phase 11: SHA256 verification (adam) ---
            _t0 = _time.perf_counter()
            loaded_adam_m_sha256 = None
            loaded_adam_v_sha256 = None
            if integrity_mode == _SHADOW_INTEGRITY_FULL_SHA256:
                loaded_adam_m_sha256, _ = _hash_named_tensors(adam_state["m"].items())
                loaded_adam_v_sha256, _ = _hash_named_tensors(adam_state["v"].items())
            _t_sha256_adam = _time.perf_counter() - _t0
        finally:
            _t0 = _time.perf_counter()
            _close_shadow_flat_views(adam_m_fd, adam_m_mm, adam_m_views)
            _close_shadow_flat_views(adam_v_fd, adam_v_mm, adam_v_views)
            _t_mmap_close_adam = _time.perf_counter() - _t0

        expected_adam_m_sha256 = adam_meta.get("adam_m_sha256")
        expected_adam_v_sha256 = adam_meta.get("adam_v_sha256")
        if integrity_mode == _SHADOW_INTEGRITY_FULL_SHA256 and loaded_adam_m_sha256 != expected_adam_m_sha256:
            raise RuntimeError(
                f"shadow flat snapshot {flat_storage['header_path']} failed adam m content verification "
                f"(expected_adam_m_sha256={expected_adam_m_sha256}, loaded_adam_m_sha256={loaded_adam_m_sha256})"
            )
        if integrity_mode == _SHADOW_INTEGRITY_FULL_SHA256 and loaded_adam_v_sha256 != expected_adam_v_sha256:
            raise RuntimeError(
                f"shadow flat snapshot {flat_storage['header_path']} failed adam v content verification "
                f"(expected_adam_v_sha256={expected_adam_v_sha256}, loaded_adam_v_sha256={loaded_adam_v_sha256})"
            )

    base_step = int(header.get("base_step", "0"))
    committed_step = int(header.get("committed_step", "0"))

    _t_total = _time.perf_counter() - _t0_total
    _total_bytes = _state_bytes + _adam_bytes
    _clone_total = _t_clone_state + _t_clone_adam
    _sha256_total = _t_sha256_state + _t_sha256_adam
    _bw_clone = (_total_bytes / _clone_total / 1e9) if _clone_total > 0 else 0.0
    if loading_phase_log_enabled():
        logger.info(
            f"[Shadow Load Timing] total={_t_total:.4f}s "
            f"| read_header={_t_read_header:.4f}s "
            f"deserialize_layout={_t_deserialize_layout:.4f}s "
            f"read_state_meta={_t_read_state_meta:.4f}s "
            f"| mmap_open_state={_t_mmap_open_state:.4f}s "
            f"clone_state={_t_clone_state:.4f}s ({_state_bytes/1e6:.1f}MB) "
            f"sha256_state={_t_sha256_state:.4f}s "
            f"mmap_close_state={_t_mmap_close_state:.4f}s "
            f"| tie_weights={_t_tie_weights:.4f}s"
        )
        if adam_state is not None:
            logger.info(
                f"[Shadow Load Timing] adam: "
                f"read_adam_meta={_t_read_adam_meta:.4f}s "
                f"mmap_open_adam={_t_mmap_open_adam:.4f}s "
                f"clone_adam={_t_clone_adam:.4f}s ({_adam_bytes/1e6:.1f}MB) "
                f"sha256_adam={_t_sha256_adam:.4f}s "
                f"mmap_close_adam={_t_mmap_close_adam:.4f}s"
            )
        logger.info(
            f"[Shadow Load Timing] summary: "
            f"clone_total={_clone_total:.4f}s sha256_total={_sha256_total:.4f}s "
            f"data={_total_bytes/1e6:.1f}MB clone_bw={_bw_clone:.1f}GB/s "
            f"num_tensors={len(state_dict)} integrity={integrity_mode}"
        )

    logger.info(
        f"[Shadow Recovery] Loaded {len(state_dict)} tensors from flat storage, "
        f"base_step={base_step} committed_step={committed_step}"
    )
    return state_dict, adam_state, base_step, committed_step


def _load_shadow_flat_replica(flat_storage, tied_groups=None):
    state_dict, _adam_state, base_step, committed_step = _load_shadow_bundle_flat(
        flat_storage,
        tied_groups=tied_groups,
    )
    return state_dict, base_step, committed_step


## Legacy rebase payload functions (_write_rebase_payload_flat, _cleanup_rebase_payload_flat,
## _load_rebase_payload_flat) have been moved to legacy_functions.py.


def _close_shadow_flat_writer(flat_writer):
    _close_shadow_bundle_flat_writer(flat_writer)


def _commit_shadow_state(
    state_dict,
    replica_path,
    base_step,
    committed_step,
    tied_groups=None,
    flat_writer=None,
    adam_state=None,
):
    if flat_writer is not None:
        return _commit_shadow_bundle_flat(
            state_dict,
            adam_state=adam_state,
            flat_writer=flat_writer,
            base_step=base_step,
            committed_step=committed_step,
        )
    save_state = OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if key not in _secondary_tied_keys(tied_groups)
    )
    _atomic_save_state_dict_safetensors(
        save_state,
        replica_path,
        metadata={
            "base_step": int(base_step),
            "committed_step": int(committed_step),
        },
    )
    return {
        "storage": "safetensors",
        "base_step": int(base_step),
        "committed_step": int(committed_step),
        "generation": None,
        "state_ready": True,
        "adam_ready": True,
        "header_ready": True,
    }


def _commit_shadow_succeeded(commit_result, *, require_adam):
    if not commit_result:
        return False
    if not commit_result.get("state_ready", False):
        return False
    if not commit_result.get("header_ready", False):
        return False
    if require_adam and not commit_result.get("adam_ready", False):
        return False
    return True


def _load_shadow_replica(replica_path, tied_groups=None):
    state_dict, metadata = _load_state_dict_safetensors_with_metadata(replica_path)
    if tied_groups:
        _tie_state_dict_inplace(state_dict, tied_groups)
    base_step = int(metadata.get("base_step", "0"))
    committed_step = int(metadata.get("committed_step", "0"))
    logger.info(
        f"[Shadow Recovery] Loaded {len(state_dict)} tensors from {replica_path}, "
        f"base_step={base_step} committed_step={committed_step}"
    )
    return state_dict, base_step, committed_step


def _clone_working_state(initial_state, tied_groups):
    working = OrderedDict((name, tensor.clone()) for name, tensor in initial_state.items())
    if tied_groups:
        _tie_state_dict_inplace(working, tied_groups)
    return working


## Legacy functions (_rebase_working_state, _trim_retained_updates, _replay_retained_suffix)
## have been moved to legacy_functions.py.


def _collect_flat_files(flat_storage):
    """Collect all flat file paths from a flat_storage descriptor."""
    if not flat_storage:
        return []
    files = []
    if flat_storage.get("header_path"):
        files.append(flat_storage["header_path"])
    if flat_storage.get("state_meta_path"):
        files.append(flat_storage["state_meta_path"])
    if flat_storage.get("adam_meta_path"):
        files.append(flat_storage["adam_meta_path"])
    if flat_storage.get("expected_generation_path"):
        files.append(flat_storage["expected_generation_path"])
    for path in flat_storage.get("buffer_paths", ()) or ():
        files.append(path)
    for path in flat_storage.get("adam_m_buffer_paths", ()) or ():
        files.append(path)
    for path in flat_storage.get("adam_v_buffer_paths", ()) or ():
        files.append(path)
    return [f for f in files if f and os.path.exists(f)]


def _disk_anchor_thread_main(
    flat_storage,
    output_dir,
    persist_trigger,
    persist_done,
    stop_event,
    disk_anchor_step_val,
    shadow_step_val,
    _logger,
):
    """Daemon thread: after each shadow commit, copy flat files to shadow_anchor-<step>/."""
    while not stop_event.is_set():
        if not persist_trigger.wait(timeout=0.1):
            continue
        persist_trigger.clear()

        flat_files = _collect_flat_files(flat_storage)
        step = int(shadow_step_val.value)
        final_dir = os.path.join(output_dir, f"shadow_anchor-{step}")
        tmp_dir = final_dir + ".tmp"

        trace_token = trace_begin(
            panel="cpu_shadow", lane="disk_anchor",
            event="disk_anchor_persist", step=step,
        )
        t0 = time.perf_counter()
        total_bytes = 0
        elapsed_ms = 0.0
        try:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)

            for src in flat_files:
                dst = os.path.join(tmp_dir, os.path.basename(src))
                shutil.copy2(src, dst)
                total_bytes += os.path.getsize(dst)
                fd = os.open(dst, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)

            if os.path.isdir(final_dir):
                shutil.rmtree(final_dir)
            os.rename(tmp_dir, final_dir)

            disk_anchor_step_val.value = step
        except Exception as exc:
            _logger.exception(f"[DiskAnchor] step={step} failed: {exc}")
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            trace_end(trace_token, step=step, counters={
                "duration_ms": elapsed_ms,
                "total_bytes": total_bytes,
            })
            persist_done.set()

        if time_log_enabled():
            _logger.info(f"[DiskAnchor] step={step} persist={elapsed_ms:.0f}ms bytes={total_bytes}")


def _shadow_process_main(
    update_queue,
    initial_state,
    initial_base_step,
    initial_committed_step,
    shadow_step_val,
    param_names,
    tied_groups,
    rng_device,
    simulate_perturbation,
    default_zo_eps,
    adam_config,
    use_pipeline,
    P,
    replica_path,
    commit_interval,
    flat_storage,
    ready_event=None,
    output_dir=None,
    disk_anchor_step_val=None,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    _logger = logging.getLogger(__name__ + ".shadow_process")
    boot_started_at = time.perf_counter()
    configure_trace(process_role="shadow")
    if trace_enabled():
        _trace_proc = psutil.Process(os.getpid())
        _trace_proc.cpu_percent(interval=None)
        trace_instant(
            panel="cpu_shadow",
            lane="shadow_main",
            event="shadow_process_start",
            counters={"initial_base_step": int(initial_base_step), "initial_committed_step": int(initial_committed_step)},
        )
        start_resource_sampler(
            panel="cpu_shadow",
            provider=lambda: _shadow_resource_counters(shadow_step_val, _trace_proc),
        )
    shadow_boot_token = trace_begin(
        panel="cpu_shadow",
        lane="shadow_main",
        event="shadow_boot",
        step=int(initial_committed_step),
    )

    torch.set_num_interop_threads(1)
    adam_state = None
    flat_writer = None
    load_flat_s = 0.0

    if flat_storage and flat_storage.get("enabled") and initial_state is not None:
        _logger.warning(
            "[Shadow Boot] flat snapshot is enabled; ignoring spawned initial_state payload and loading from flat storage"
        )

    if flat_storage and flat_storage.get("enabled"):
        t0_load_flat = time.perf_counter()
        loaded_state, loaded_adam_state, loaded_base_step, loaded_committed_step = _load_shadow_bundle_flat(
            flat_storage,
            tied_groups=tied_groups,
        )
        load_flat_s = time.perf_counter() - t0_load_flat
        initial_state = loaded_state
        initial_base_step = int(loaded_base_step)
        initial_committed_step = int(loaded_committed_step)
        adam_state = loaded_adam_state
        flat_writer = _open_shadow_bundle_flat_writer(flat_storage)
        if (_step_diag_enabled() or _step_exact_enabled()) and adam_state is not None:
            _log_adam_checksums(
                f"shadow_boot base_step={initial_base_step} committed_step={initial_committed_step}",
                adam_state,
                _logger=_logger,
            )
            if _step_exact_enabled():
                _log_adam_exact_fingerprint(
                    f"shadow_boot base_step={initial_base_step} committed_step={initial_committed_step}",
                    adam_state,
                    _logger=_logger,
                )
        if time_log_enabled():
            _logger.info(
                f"[Shadow BootTiming] load_flat={load_flat_s:.3f}s "
                f"base_step={initial_base_step} committed_step={initial_committed_step}"
            )
    elif initial_state is None:
        raise RuntimeError("shadow process received no initial_state and no flat snapshot to load from")
    elif adam_config is not None:
        adam_state = {
            "m": {},
            "v": {},
            "t": 0,
            "betas": adam_config["betas"],
            "adam_eps": adam_config["adam_eps"],
        }
        _logger.info(f"[Shadow Process] Adam state initialized: betas={adam_config['betas']}")

    try:
        n_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cores = os.cpu_count() or 64

    n_reserve = int(os.environ.get("SHADOW_RESERVE_THREADS", "1"))
    if use_pipeline and rng_device == "zo_rng":
        n_cons = int(os.environ.get("SHADOW_CONSUMER_THREADS", str(n_cores // 2)))
        c_prod = max(1, n_cores - n_reserve - n_cons)
        aten_threads = max(1, n_cons)
        import zo_rng as _zo_rng

        _zo_rng.set_num_threads(c_prod)
        torch.set_num_threads(aten_threads)
    elif use_pipeline:
        threads_per_op = max(1, n_cores // (P + 1))
        torch.set_num_threads(threads_per_op)
    else:
        serial_threads = max(1, n_cores - n_reserve)
        torch.set_num_threads(serial_threads)
        if rng_device == "zo_rng":
            import zo_rng as _zo_rng

            _zo_rng.set_num_threads(serial_threads)

    try:
        _os_threads = len(os.listdir(f"/proc/{os.getpid()}/task"))
    except Exception:
        _os_threads = -1
    try:
        _affinity = sorted(os.sched_getaffinity(0))
    except Exception:
        _affinity = []

    _zo_thr = 0
    if rng_device == "zo_rng":
        try:
            import zo_rng as _zo_mod

            _zo_thr = _zo_mod.get_num_threads()
        except Exception:
            pass

    _model_bytes = sum(initial_state[nm].numel() * initial_state[nm].element_size() for nm in param_names)
    _interop_thr = torch.get_num_interop_threads()
    if resource_log_enabled():
        _logger.info(
            f"[Shadow Boot] pid={os.getpid()}\n"
            f"  affinity={{{','.join(str(c) for c in _affinity[:5])},...}} ({len(_affinity)} CPUs)\n"
            f"  aten={torch.get_num_threads()}  zo_rng={_zo_thr}  interop={_interop_thr}  "
            f"OS_threads={_os_threads}\n"
            f"  model_bytes={_model_bytes / 1e9:.2f}GB\n"
            f"  pipeline={use_pipeline}  P={P}  rng={rng_device}  "
            f"  commit_interval={commit_interval}"
        )
    _thread_snapshot("Shadow BOOT", _logger, detail=True)

    if flat_storage and flat_storage.get("enabled"):
        total_bytes = int(flat_storage["layout"]["total_bytes"])
        if flat_storage.get("has_adam", False):
            total_bytes += int(flat_storage["adam_layout"]["total_bytes"]) * 2
        if resource_log_enabled():
            _logger.info(
                f"[Shadow Flat] single-buffer enabled total_bytes={total_bytes / 1e9:.2f}GB "
                f"(non-atomic commit; incomplete snapshot is fatal)"
            )

    try:
        if use_pipeline:
            _shadow_process_pipelined(
                update_queue,
                initial_state,
                initial_base_step,
                initial_committed_step,
                shadow_step_val,
                param_names,
                tied_groups,
                rng_device,
                simulate_perturbation,
                default_zo_eps,
                adam_state,
                P,
                replica_path,
                commit_interval,
                _logger,
                flat_writer,
                boot_started_at,
                ready_event,
                shadow_boot_token,
                flat_storage,
                output_dir,
                disk_anchor_step_val,
            )
        else:
            _shadow_process_serial(
                update_queue,
                initial_state,
                initial_base_step,
                initial_committed_step,
                shadow_step_val,
                param_names,
                tied_groups,
                rng_device,
                simulate_perturbation,
                default_zo_eps,
                adam_state,
                replica_path,
                commit_interval,
                _logger,
                flat_writer,
                boot_started_at,
                ready_event,
                shadow_boot_token,
                flat_storage,
                output_dir,
                disk_anchor_step_val,
            )
    finally:
        stop_resource_sampler()
        if flat_writer is not None:
            _close_shadow_bundle_flat_writer(flat_writer)


def _shadow_process_serial(
    update_queue,
    initial_state,
    initial_base_step,
    initial_committed_step,
    shadow_step_val,
    param_names,
    tied_groups,
    rng_device,
    simulate_perturbation,
    default_zo_eps,
    adam_state,
    replica_path,
    commit_interval,
    _logger,
    flat_writer=None,
    boot_started_at=None,
    ready_event=None,
    boot_trace_token=None,
    flat_storage=None,
    output_dir=None,
    disk_anchor_step_val=None,
):
    from . import log_based_replay as _bdc

    t0_clone = time.perf_counter()
    working_state = _clone_working_state(initial_state, tied_groups)
    clone_working_s = time.perf_counter() - t0_clone
    if time_log_enabled():
        _logger.info(f"[Shadow BootTiming] clone_working={clone_working_s:.3f}s mode=serial")
    if boot_started_at is not None and time_log_enabled():
        _logger.info(
            f"[Shadow BootTiming] ready_for_updates={time.perf_counter() - boot_started_at:.3f}s mode=serial"
        )
    if ready_event is not None:
        ready_event.set()
    trace_end(boot_trace_token, step=int(initial_committed_step))
    base_step = int(initial_base_step)
    durable_step = int(initial_committed_step)
    last_applied_step = durable_step
    desired_commit_step = durable_step
    pending_since_commit = 0

    disk_anchor_enabled = (
        output_dir is not None
        and flat_writer is not None
        and flat_storage is not None
        and disk_anchor_step_val is not None
    )
    persist_trigger = None
    persist_done = None
    disk_anchor_stop = None
    disk_anchor_thread = None
    if disk_anchor_enabled:
        persist_trigger = threading.Event()
        persist_done = threading.Event()
        persist_done.set()
        disk_anchor_stop = threading.Event()
        disk_anchor_thread = threading.Thread(
            target=_disk_anchor_thread_main,
            args=(flat_storage, output_dir,
                  persist_trigger, persist_done, disk_anchor_stop,
                  disk_anchor_step_val, shadow_step_val, _logger),
            daemon=True, name="disk-anchor",
        )
        disk_anchor_thread.start()

    def _commit_if_needed(target_step, reason):
        nonlocal durable_step, desired_commit_step, pending_since_commit
        if int(target_step) <= int(durable_step):
            return 0.0
        t0_commit = time.time()
        if disk_anchor_enabled:
            # Only emit a trace span when the wait would actually block; if
            # persist_done is already set, the wait is a no-op and producing
            # a zero-duration span just adds visual noise to the plot.
            if not persist_done.is_set():
                wait_token = trace_begin(
                    panel="cpu_shadow",
                    lane="shadow_main",
                    event="wait_disk_anchor",
                    step=int(target_step),
                )
                persist_done.wait()
                trace_end(wait_token, step=int(target_step))
            persist_done.clear()
        commit_token = trace_begin(
            panel="cpu_shadow",
            lane="shadow_main",
            event="shadow_commit",
            step=int(target_step),
            extra={"reason": reason},
        )
        commit_result = _commit_shadow_state(
            working_state,
            replica_path,
            base_step,
            int(target_step),
            tied_groups=tied_groups,
            flat_writer=flat_writer,
            adam_state=adam_state,
        )
        if not _commit_shadow_succeeded(commit_result, require_adam=adam_state is not None):
            if disk_anchor_enabled:
                persist_done.set()
            raise RuntimeError(f"[Shadow] {reason} commit failed: {commit_result}")
        durable_step = int(commit_result["committed_step"])
        desired_commit_step = durable_step
        shadow_step_val.value = durable_step
        pending_since_commit = 0
        _log_durable_publish(durable_step, working_state, adam_state, _logger)
        trace_end(
            commit_token,
            step=int(durable_step),
            counters={"durable_step": int(durable_step)},
            extra={"reason": reason},
        )
        if disk_anchor_enabled:
            persist_trigger.set()
        return time.time() - t0_commit

    wait_token = trace_begin(
        panel="cpu_shadow",
        lane="shadow_main",
        event="shadow_wait_update",
        step=int(durable_step),
    )
    while True:
        try:
            cmd = update_queue.get(timeout=0.05)
        except queue_module.Empty:
            continue
        trace_end(wait_token, step=int(durable_step))

        kind = cmd.get("cmd") if isinstance(cmd, dict) else None
        if kind == "stop":
            trace_instant(panel="cpu_shadow", lane="shadow_main", event="shadow_stop", step=int(durable_step))
            break
        if kind != "update":
            wait_token = trace_begin(
                panel="cpu_shadow",
                lane="shadow_main",
                event="shadow_wait_update",
                step=int(durable_step),
            )
            continue

        update = cmd["update"]
        step = int(update["step"])
        if step <= last_applied_step:
            wait_token = trace_begin(
                panel="cpu_shadow",
                lane="shadow_main",
                event="shadow_wait_update",
                step=int(durable_step),
            )
            continue

        t_start = time.time()
        z_dict = _bdc._generate_z_for_one_step(step_seed := update["seed"], param_names, working_state, rng_device)
        t_zgen = time.time() - t_start

        t0_apply = time.time()
        apply_token = trace_begin(
            panel="cpu_shadow",
            lane="shadow_main",
            event="shadow_apply",
            step=int(step),
        )
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
        t_apply = time.time() - t0_apply
        del z_dict
        trace_end(
            apply_token,
            step=int(step),
            counters={
                "apply_ms": t_apply * 1000.0,
                "zgen_ms": t_zgen * 1000.0,
            },
        )

        last_applied_step = step
        pending_since_commit += 1
        if (_step_diag_enabled() or _step_exact_enabled()) and adam_state is not None:
            if _step_diag_enabled():
                _log_adam_brief(f"shadow_live step={step}", adam_state, _logger=_logger)
                _log_adam_checksums(f"shadow_live step={step}", adam_state, _logger=_logger)
            if _step_exact_enabled():
                _log_adam_exact_fingerprint(f"shadow_live step={step}", adam_state, _logger=_logger)
        if _step_diag_enabled():
            _log_state_checksums(f"shadow_live step={step}", working_state, _logger=_logger)
            if _step_exact_enabled():
                _log_state_exact_fingerprint(f"shadow_live step={step}", working_state, _logger=_logger)

        t_commit = 0.0
        if pending_since_commit >= commit_interval:
            desired_commit_step = int(last_applied_step)
            t_commit = _commit_if_needed(desired_commit_step, "Periodic")

        if shadow_step_time_log_enabled():
            _logger.info(
                f"[Shadow Timing] step={step} grad={update['grad']:.6e} seed={step_seed} "
                f"| apply={t_apply * 1000:.0f}ms zgen={t_zgen * 1000:.0f}ms "
                f"commit={t_commit * 1000:.0f}ms pending={pending_since_commit} "
                f"| applied={last_applied_step} desired={desired_commit_step} durable={durable_step}"
            )
        if shadow_step_resource_log_enabled():
            _rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
            _logger.info(
                f"[Shadow Resource] step={step} durable={durable_step} RSS={_rss_gb:.1f}GB"
            )
        wait_token = trace_begin(
            panel="cpu_shadow",
            lane="shadow_main",
            event="shadow_wait_update",
            step=int(durable_step),
        )

    if last_applied_step > durable_step:
        desired_commit_step = int(last_applied_step)
        _commit_if_needed(desired_commit_step, "Final")
    if disk_anchor_enabled:
        try:
            persist_done.wait(timeout=30)
        except Exception:
            pass
        disk_anchor_stop.set()
        try:
            disk_anchor_thread.join(timeout=5)
        except Exception:
            pass
    _logger.info("[Shadow Process] Stopped (serial)")


def _shadow_process_pipelined(
    update_queue,
    initial_state,
    initial_base_step,
    initial_committed_step,
    shadow_step_val,
    param_names,
    tied_groups,
    rng_device,
    simulate_perturbation,
    default_zo_eps,
    adam_state,
    P,
    replica_path,
    commit_interval,
    _logger,
    flat_writer=None,
    boot_started_at=None,
    ready_event=None,
    boot_trace_token=None,
    flat_storage=None,
    output_dir=None,
    disk_anchor_step_val=None,
):
    from . import log_based_replay as _bdc

    shadow_bytes = sum(initial_state[nm].numel() * initial_state[nm].element_size() for nm in param_names)
    t0_clone = time.perf_counter()
    working_state = _clone_working_state(initial_state, tied_groups)
    clone_working_s = time.perf_counter() - t0_clone
    if time_log_enabled():
        _logger.info(f"[Shadow BootTiming] clone_working={clone_working_s:.3f}s mode=pipeline")
    if boot_started_at is not None and time_log_enabled():
        _logger.info(
            f"[Shadow BootTiming] ready_for_updates={time.perf_counter() - boot_started_at:.3f}s mode=pipeline"
        )
    if ready_event is not None:
        ready_event.set()
    trace_end(boot_trace_token, step=int(initial_committed_step))
    base_step = int(initial_base_step)
    durable_step = int(initial_committed_step)
    consumer_step = durable_step + 1
    last_applied_step = durable_step
    desired_commit_step = durable_step
    pending_since_commit = 0

    disk_anchor_enabled = (
        output_dir is not None
        and flat_writer is not None
        and flat_storage is not None
        and disk_anchor_step_val is not None
    )
    persist_trigger = None
    persist_done = None
    disk_anchor_stop = None
    disk_anchor_thread = None
    if disk_anchor_enabled:
        persist_trigger = threading.Event()
        persist_done = threading.Event()
        persist_done.set()
        disk_anchor_stop = threading.Event()
        disk_anchor_thread = threading.Thread(
            target=_disk_anchor_thread_main,
            args=(flat_storage, output_dir,
                  persist_trigger, persist_done, disk_anchor_stop,
                  disk_anchor_step_val, shadow_step_val, _logger),
            daemon=True, name="disk-anchor",
        )
        disk_anchor_thread.start()

    result_queue = queue_module.Queue(maxsize=max(1, P))
    producer_stop = threading.Event()
    producer_error = [None]
    internal_updates = {}
    internal_lock = threading.Lock()
    update_available_event = threading.Event()
    assign_lock = threading.Lock()
    next_step_to_assign = [consumer_step]

    producer_timing = {"duration_ms": 0.0}
    generation = [0]

    def producer(worker_id=0):
        try:
            local_generation = generation[0]
            while not producer_stop.is_set():
                with assign_lock:
                    step_idx = next_step_to_assign[0]
                    next_step_to_assign[0] += 1

                update = None
                while not producer_stop.is_set():
                    with internal_lock:
                        update = internal_updates.pop(step_idx, None)
                    if update is not None:
                        break
                    update_available_event.wait(timeout=0.05)
                    update_available_event.clear()

                if producer_stop.is_set():
                    break
                if update is None:
                    continue

                t0_zgen = time.monotonic()
                with trace_span(
                    panel="cpu_shadow",
                    lane=f"producer_{worker_id}",
                    event="shadow_generate",
                    step=step_idx,
                ):
                    z_dict = _bdc._generate_z_for_one_step(update["seed"], param_names, working_state, rng_device)
                producer_timing["duration_ms"] = (time.monotonic() - t0_zgen) * 1000

                while not producer_stop.is_set():
                    try:
                        result_queue.put((local_generation, step_idx, z_dict, update), timeout=0.05)
                        break
                    except queue_module.Full:
                        continue
        except Exception as exc:
            producer_error[0] = exc
            _logger.error(f"[Shadow Pipeline] Producer CRASHED: {exc}")

    def _restart_producers():
        producer_stop.clear()
        producer_error[0] = None
        threads = []
        for wid in range(P):
            t = threading.Thread(target=producer, args=(wid,), daemon=True)
            t.start()
            threads.append(t)
        return threads

    def _stop_producers(threads):
        producer_stop.set()
        update_available_event.set()
        for t in threads:
            t.join(timeout=2.0)

    def _drain_result_queue():
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except queue_module.Empty:
                break

    def _reseed_internal_updates():
        with internal_lock:
            internal_updates.clear()
        next_step_to_assign[0] = consumer_step
        update_available_event.clear()

    def _pause_and_commit(target_step, reason):
        nonlocal threads, durable_step, desired_commit_step, pending_since_commit
        if int(target_step) <= int(durable_step):
            return 0.0
        t0_commit = time.monotonic()
        if disk_anchor_enabled:
            # Only emit a trace span when the wait would actually block.
            if not persist_done.is_set():
                wait_token = trace_begin(
                    panel="cpu_shadow",
                    lane="shadow_main",
                    event="wait_disk_anchor",
                    step=int(target_step),
                )
                persist_done.wait()
                trace_end(wait_token, step=int(target_step))
            persist_done.clear()
        commit_token = trace_begin(
            panel="cpu_shadow",
            lane="shadow_main",
            event="shadow_commit",
            step=int(target_step),
            extra={"reason": reason, "mode": "pipeline"},
        )
        _stop_producers(threads)
        generation[0] += 1
        pending_results.clear()
        _drain_result_queue()
        with internal_lock:
            internal_updates.clear()
        update_available_event.clear()
        commit_result = _commit_shadow_state(
            working_state,
            replica_path,
            base_step,
            int(target_step),
            tied_groups=tied_groups,
            flat_writer=flat_writer,
            adam_state=adam_state,
        )
        if not _commit_shadow_succeeded(commit_result, require_adam=adam_state is not None):
            if disk_anchor_enabled:
                persist_done.set()
            raise RuntimeError(f"[Shadow Pipeline] {reason} commit failed: {commit_result}")
        durable_step = int(commit_result["committed_step"])
        desired_commit_step = durable_step
        shadow_step_val.value = durable_step
        pending_since_commit = 0
        _log_durable_publish(durable_step, working_state, adam_state, _logger)
        _reseed_internal_updates()
        threads = _restart_producers()
        trace_end(
            commit_token,
            step=int(durable_step),
            counters={"durable_step": int(durable_step)},
            extra={"reason": reason, "mode": "pipeline"},
        )
        if disk_anchor_enabled:
            persist_trigger.set()
        return (time.monotonic() - t0_commit) * 1000

    threads = _restart_producers()
    pending_results = {}
    wait_token = trace_begin(
        panel="cpu_shadow",
        lane="shadow_main",
        event="shadow_wait_update",
        step=int(durable_step),
    )

    while True:
        try:
            while True:
                cmd = update_queue.get_nowait()
                trace_end(wait_token, step=int(durable_step))
                kind = cmd.get("cmd") if isinstance(cmd, dict) else None
                if kind == "stop":
                    trace_instant(panel="cpu_shadow", lane="shadow_main", event="shadow_stop", step=int(durable_step))
                    _stop_producers(threads)
                    if last_applied_step > durable_step:
                        if disk_anchor_enabled:
                            persist_done.wait()
                            persist_done.clear()
                        commit_result = _commit_shadow_state(
                            working_state,
                            replica_path,
                            base_step,
                            last_applied_step,
                            tied_groups=tied_groups,
                            flat_writer=flat_writer,
                            adam_state=adam_state,
                        )
                        if not _commit_shadow_succeeded(commit_result, require_adam=adam_state is not None):
                            if disk_anchor_enabled:
                                persist_done.set()
                            raise RuntimeError(f"[Shadow Pipeline] Final stop commit failed: {commit_result}")
                        durable_step = int(commit_result["committed_step"])
                        desired_commit_step = durable_step
                        shadow_step_val.value = durable_step
                        _log_durable_publish(durable_step, working_state, adam_state, _logger)
                        if disk_anchor_enabled:
                            persist_trigger.set()
                    if disk_anchor_enabled:
                        try:
                            persist_done.wait(timeout=30)
                        except Exception:
                            pass
                        disk_anchor_stop.set()
                        try:
                            disk_anchor_thread.join(timeout=5)
                        except Exception:
                            pass
                    _logger.info("[Shadow Pipeline] Stopped")
                    return
                if kind == "update":
                    update = cmd["update"]
                    with internal_lock:
                        internal_updates[int(update["step"])] = update
                    update_available_event.set()
                    wait_token = trace_begin(
                        panel="cpu_shadow",
                        lane="shadow_main",
                        event="shadow_wait_update",
                        step=int(durable_step),
                    )
        except queue_module.Empty:
            pass

        if producer_error[0] is not None:
            raise producer_error[0]

        try:
            result_generation, step_idx, z_dict, update = result_queue.get(timeout=0.05)
        except queue_module.Empty:
            continue
        trace_end(wait_token, step=int(durable_step))

        if result_generation != generation[0]:
            del z_dict
            continue
        pending_results[step_idx] = (z_dict, update)
        while consumer_step in pending_results:
            z_dict, update = pending_results.pop(consumer_step)
            t0_apply = time.monotonic()
            apply_token = trace_begin(
                panel="cpu_shadow",
                lane="shadow_main",
                event="shadow_apply",
                step=int(consumer_step),
                extra={"mode": "pipeline"},
            )
            _bdc._apply_single_update_with_pregenerated_z(
                working_state,
                update,
                param_names,
                z_dict,
                default_zo_eps=default_zo_eps,
                simulate_perturbation=simulate_perturbation,
                zo2_mode=False,
                adam_state=adam_state,
                _diag_logger=_logger,
            )
            apply_ms = (time.monotonic() - t0_apply) * 1000
            del z_dict
            trace_end(
                apply_token,
                step=int(consumer_step),
                counters={
                    "apply_ms": apply_ms,
                    "zgen_ms": producer_timing["duration_ms"],
                },
                extra={"mode": "pipeline"},
            )

            last_applied_step = consumer_step
            consumer_step += 1
            pending_since_commit += 1
            if (_step_diag_enabled() or _step_exact_enabled()) and adam_state is not None:
                if _step_diag_enabled():
                    _log_adam_brief(f"shadow_live step={last_applied_step}", adam_state, _logger=_logger)
                    _log_adam_checksums(f"shadow_live step={last_applied_step}", adam_state, _logger=_logger)
                if _step_exact_enabled():
                    _log_adam_exact_fingerprint(f"shadow_live step={last_applied_step}", adam_state, _logger=_logger)
            if _step_diag_enabled():
                _log_state_checksums(f"shadow_live step={last_applied_step}", working_state, _logger=_logger)
                if _step_exact_enabled():
                    _log_state_exact_fingerprint(f"shadow_live step={last_applied_step}", working_state, _logger=_logger)

            commit_ms = 0.0
            if pending_since_commit >= commit_interval:
                desired_commit_step = int(last_applied_step)
                commit_ms = _pause_and_commit(desired_commit_step, "Periodic")

            if shadow_step_time_log_enabled():
                _logger.info(
                    f"[Shadow Timing] step={last_applied_step} grad={update['grad']:.6e} seed={update['seed']} "
                    f"| apply={apply_ms:.0f}ms zgen={producer_timing['duration_ms']:.0f}ms "
                    f"commit={commit_ms:.0f}ms pending={pending_since_commit} "
                    f"| applied={last_applied_step} desired={desired_commit_step} durable={durable_step}"
                )
            if shadow_step_resource_log_enabled():
                _rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
                _logger.info(
                    f"[Shadow Resource] step={last_applied_step} durable={durable_step} RSS={_rss_gb:.1f}GB"
                )
            wait_token = trace_begin(
                panel="cpu_shadow",
                lane="shadow_main",
                event="shadow_wait_update",
                step=int(durable_step),
            )
