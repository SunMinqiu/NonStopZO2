import logging
import json
import hashlib
import mmap
import os
import queue as queue_module
import tempfile
import threading
import time
from collections import OrderedDict

import psutil
import torch

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
def _secondary_tied_keys(tied_groups):
    excluded = set()
    for group in tied_groups or []:
        for name in group[1:]:
            excluded.add(name)
    return excluded


def _log_durable_publish(step, state_dict, adam_state, _logger):
    label = f"shadow_durable step={int(step)}"
    _log_state_checksums(label, state_dict, _logger=_logger)
    _log_state_exact_fingerprint(label, state_dict, _logger=_logger)
    if adam_state is not None:
        _log_adam_checksums(label, adam_state, _logger=_logger)
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
):
    adam_state = adam_state or {}
    beta1, beta2 = adam_state.get("betas", (0.0, 0.0))
    header = {
        "snapshot_state": str(snapshot_state),
        "base_step": int(base_step),
        "committed_step": int(committed_step),
        "generation": int(generation),
        "has_adam": bool(has_adam),
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


def _shadow_component_meta(*, kind, generation, step, snapshot_state="ready", adam_t=None):
    payload = {
        "kind": str(kind),
        "snapshot_state": str(snapshot_state),
        "generation": int(generation),
        "committed_step": int(step),
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
        "buffer_paths": tuple(flat_storage["buffer_paths"]),
        "layout": flat_storage["layout"],
        "fds": fds,
        "mmaps": mmaps,
        "views": views,
        "generation": int(header.get("generation", 0)),
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
        generation = int(flat_writer.get("generation", 0)) + 1
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
            ),
        )
        _write_shadow_flat_header(
            flat_writer["state_meta_path"],
            _shadow_component_meta(
                kind="state",
                generation=generation,
                step=committed_step,
                snapshot_state="writing",
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
            ),
        )
        flat_writer["generation"] = generation
    finally:
        _close_shadow_bundle_flat_writer(flat_writer)


def _open_shadow_flat_writer(flat_storage):
    return _open_shadow_bundle_flat_writer(flat_storage)


def _commit_shadow_bundle_flat(state_dict, adam_state, flat_writer, base_step, committed_step):
    generation = int(flat_writer.get("generation", 0)) + 1
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
        ),
    )
    _write_shadow_flat_header(
        flat_writer["state_meta_path"],
        _shadow_component_meta(
            kind="state",
            generation=generation,
            step=committed_step,
            snapshot_state="writing",
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
            ),
        )
    target_views = flat_writer["views"][0]
    _copy_shadow_flat_views_from_state(target_views, state_dict)
    flat_writer["mmaps"][0].flush()
    state_sha256, _ = _hash_named_tensors(target_views.items())
    _write_shadow_flat_header(
        flat_writer["state_meta_path"],
        {
            **_shadow_component_meta(
                kind="state",
                generation=generation,
                step=committed_step,
                snapshot_state="ready",
            ),
            "state_sha256": state_sha256,
        },
    )
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
        ),
    )
    flat_writer["generation"] = generation
    return {
        "storage": "flat",
        "base_step": int(base_step),
        "committed_step": int(committed_step),
        "generation": int(generation),
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
    header = _read_shadow_flat_header(flat_storage["header_path"])
    if header.get("snapshot_state", "ready") != "ready":
        raise RuntimeError(
            f"shadow flat snapshot {flat_storage['header_path']} is incomplete "
            f"(snapshot_state={header.get('snapshot_state')})"
        )
    generation = int(header.get("generation", -1))
    layout = (
        _deserialize_flat_layout(header["layout"])
        if "layout" in header else flat_storage["layout"]
    )
    state_meta_path = flat_storage.get("state_meta_path")
    if not state_meta_path or not os.path.exists(state_meta_path):
        raise RuntimeError(
            f"shadow flat snapshot {flat_storage['header_path']} is missing state metadata. "
            "This usually means /dev/shm still contains a legacy single-buffer shadow snapshot "
            "from the old protocol. Delete the existing zo_shadow_latest_*.flat* files and rerun "
            "so shadow can be reinitialized with the new state/adam/header metadata format."
        )
    state_meta = _read_shadow_flat_header(state_meta_path)
    if (
        state_meta.get("snapshot_state") != "ready"
        or int(state_meta.get("generation", -2)) != generation
        or int(state_meta.get("committed_step", -3)) != int(header.get("committed_step", -4))
    ):
        raise RuntimeError(
            f"shadow flat snapshot {flat_storage['header_path']} has inconsistent state metadata "
            f"(header_gen={generation}, state_gen={state_meta.get('generation')}, "
            f"header_step={header.get('committed_step')}, state_step={state_meta.get('committed_step')}, "
            f"state_snapshot={state_meta.get('snapshot_state')})"
        )
    fd, mm, views = _open_shadow_flat_views(layout, flat_storage["buffer_paths"][0])
    try:
        state_dict = OrderedDict(
            (name, tensor.clone()) for name, tensor in views.items()
        )
        loaded_state_sha256, _ = _hash_named_tensors(state_dict.items())
    finally:
        _close_shadow_flat_views(fd, mm, views)
    expected_state_sha256 = state_meta.get("state_sha256")
    if expected_state_sha256 and loaded_state_sha256 != expected_state_sha256:
        raise RuntimeError(
            f"shadow flat snapshot {flat_storage['header_path']} failed state content verification "
            f"(expected_state_sha256={expected_state_sha256}, loaded_state_sha256={loaded_state_sha256})"
        )
    if tied_groups:
        _tie_state_dict_inplace(state_dict, tied_groups)

    adam_state = None
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
        adam_meta = _read_shadow_flat_header(adam_meta_path)
        if (
            adam_meta.get("snapshot_state") != "ready"
            or int(adam_meta.get("generation", -2)) != generation
            or int(adam_meta.get("committed_step", -3)) != int(header.get("committed_step", -4))
            or int(adam_meta.get("adam_t", -5)) != int(header.get("adam_t", -6))
        ):
            raise RuntimeError(
                f"shadow flat snapshot {flat_storage['header_path']} has inconsistent adam metadata "
                f"(header_gen={generation}, adam_gen={adam_meta.get('generation')}, "
                f"header_step={header.get('committed_step')}, adam_step={adam_meta.get('committed_step')}, "
                f"header_t={header.get('adam_t')}, adam_t={adam_meta.get('adam_t')}, "
                f"adam_snapshot={adam_meta.get('snapshot_state')})"
            )
        adam_m_fd, adam_m_mm, adam_m_views = _open_shadow_flat_views(
            adam_layout,
            flat_storage["adam_m_buffer_paths"][0],
        )
        adam_v_fd, adam_v_mm, adam_v_views = _open_shadow_flat_views(
            adam_layout,
            flat_storage["adam_v_buffer_paths"][0],
        )
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
            loaded_adam_m_sha256, _ = _hash_named_tensors(adam_state["m"].items())
            loaded_adam_v_sha256, _ = _hash_named_tensors(adam_state["v"].items())
        finally:
            _close_shadow_flat_views(adam_m_fd, adam_m_mm, adam_m_views)
            _close_shadow_flat_views(adam_v_fd, adam_v_mm, adam_v_views)
        expected_adam_m_sha256 = adam_meta.get("adam_m_sha256")
        expected_adam_v_sha256 = adam_meta.get("adam_v_sha256")
        if expected_adam_m_sha256 and loaded_adam_m_sha256 != expected_adam_m_sha256:
            raise RuntimeError(
                f"shadow flat snapshot {flat_storage['header_path']} failed adam m content verification "
                f"(expected_adam_m_sha256={expected_adam_m_sha256}, loaded_adam_m_sha256={loaded_adam_m_sha256})"
            )
        if expected_adam_v_sha256 and loaded_adam_v_sha256 != expected_adam_v_sha256:
            raise RuntimeError(
                f"shadow flat snapshot {flat_storage['header_path']} failed adam v content verification "
                f"(expected_adam_v_sha256={expected_adam_v_sha256}, loaded_adam_v_sha256={loaded_adam_v_sha256})"
            )

    base_step = int(header.get("base_step", "0"))
    committed_step = int(header.get("committed_step", "0"))
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
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    _logger = logging.getLogger(__name__ + ".shadow_process")
    boot_started_at = time.perf_counter()

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
        _logger.info(
            f"[Shadow] threads: zo_rng={c_prod} + ATen={aten_threads} "
            f"= {c_prod + aten_threads} (n_cores={n_cores}, reserve={n_reserve})"
        )
    elif use_pipeline:
        threads_per_op = max(1, n_cores // (P + 1))
        torch.set_num_threads(threads_per_op)
    else:
        serial_threads = max(1, n_cores - n_reserve)
        torch.set_num_threads(serial_threads)
        if rng_device == "zo_rng":
            import zo_rng as _zo_rng

            _zo_rng.set_num_threads(serial_threads)
        _logger.info(
            f"[Shadow] serial zo_rng={serial_threads} ATen={serial_threads} "
            f"(alternating, n_cores={n_cores}, reserve={n_reserve})"
        )

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
    _logger.info(
        f"[Shadow Boot] pid={os.getpid()}\n"
        f"  affinity={{{','.join(str(c) for c in _affinity[:5])},...}} ({len(_affinity)} CPUs)\n"
        f"  aten={torch.get_num_threads()}  zo_rng={_zo_thr}  interop={_interop_thr}  "
        f"OS_threads={_os_threads}\n"
        f"  model_bytes={_model_bytes / 1e9:.2f}GB\n"
        f"  pipeline={use_pipeline}  P={P}  rng={rng_device}  "
        f"  commit_interval={commit_interval}\n"
        f"  load_flat={load_flat_s:.3f}s"
    )
    _thread_snapshot("Shadow BOOT", _logger, detail=True)

    if flat_storage and flat_storage.get("enabled"):
        total_bytes = int(flat_storage["layout"]["total_bytes"])
        if flat_storage.get("has_adam", False):
            total_bytes += int(flat_storage["adam_layout"]["total_bytes"]) * 2
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
            )
    finally:
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
):
    from . import log_based_checkpoint as _bdc

    _logger.info(f"[Shadow Process] Running in serial mode (params={len(param_names)}, rng={rng_device})")

    t0_clone = time.perf_counter()
    working_state = _clone_working_state(initial_state, tied_groups)
    clone_working_s = time.perf_counter() - t0_clone
    _logger.info(f"[Shadow BootTiming] clone_working={clone_working_s:.3f}s mode=serial")
    if boot_started_at is not None:
        _logger.info(
            f"[Shadow BootTiming] ready_for_updates={time.perf_counter() - boot_started_at:.3f}s mode=serial"
        )
    if ready_event is not None:
        ready_event.set()
    base_step = int(initial_base_step)
    durable_step = int(initial_committed_step)
    last_applied_step = durable_step
    desired_commit_step = durable_step
    pending_since_commit = 0
    retained_updates = {}

    def _commit_if_needed(target_step, reason):
        nonlocal durable_step, desired_commit_step, pending_since_commit
        if int(target_step) <= int(durable_step):
            return 0.0
        t0_commit = time.time()
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
            raise RuntimeError(f"[Shadow] {reason} commit failed: {commit_result}")
        durable_step = int(commit_result["committed_step"])
        desired_commit_step = durable_step
        shadow_step_val.value = durable_step
        pending_since_commit = 0
        _log_durable_publish(durable_step, working_state, adam_state, _logger)
        return time.time() - t0_commit

    while True:
        try:
            cmd = update_queue.get(timeout=0.05)
        except queue_module.Empty:
            continue

        kind = cmd.get("cmd") if isinstance(cmd, dict) else None
        if kind == "stop":
            break
        if kind == "rebase":
            anchor_ref = cmd.get("flat_storage") if isinstance(cmd, dict) else None
            if anchor_ref is None:
                anchor_ref = cmd["path"]
            working_state, rebased_adam_state, base_step, durable_step = _rebase_working_state(
                anchor_ref, tied_groups, _logger
            )
            if rebased_adam_state is not None:
                adam_state = rebased_adam_state
                if _step_diag_enabled() or _step_exact_enabled():
                    _log_adam_checksums(
                        f"shadow_rebase base_step={base_step} committed_step={durable_step}",
                        adam_state,
                        _logger=_logger,
                    )
                    if _step_exact_enabled():
                        _log_adam_exact_fingerprint(
                            f"shadow_rebase base_step={base_step} committed_step={durable_step}",
                            adam_state,
                            _logger=_logger,
                        )
            elif adam_state is not None:
                raise RuntimeError("[Shadow] Rebase source has no Adam state for MeZO-Adam")
            last_applied_step = _replay_retained_suffix(
                retained_updates,
                durable_step,
                working_state,
                param_names,
                rng_device,
                simulate_perturbation,
                default_zo_eps,
                adam_state,
                _bdc,
                _logger,
            )
            desired_commit_step = int(last_applied_step)
            pending_since_commit = max(0, desired_commit_step - durable_step)
            _commit_if_needed(desired_commit_step, "Rebase")
            _trim_retained_updates(retained_updates, durable_step)
            continue
        if kind != "update":
            continue

        update = cmd["update"]
        step = int(update["step"])
        retained_updates[step] = update
        if step <= last_applied_step:
            continue

        t_start = time.time()
        z_dict = _bdc._generate_z_for_one_step(step_seed := update["seed"], param_names, working_state, rng_device)
        t_zgen = time.time() - t_start

        t0_apply = time.time()
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

        _rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        _logger.info(
            f"[Shadow] step={step} grad={update['grad']:.6e} seed={step_seed} "
            f"| apply={t_apply * 1000:.0f}ms zgen={t_zgen * 1000:.0f}ms "
            f"commit={t_commit * 1000:.0f}ms pending={pending_since_commit} "
            f"| applied={last_applied_step} desired={desired_commit_step} durable={durable_step} RSS={_rss_gb:.1f}GB"
        )

    if last_applied_step > durable_step:
        desired_commit_step = int(last_applied_step)
        _commit_if_needed(desired_commit_step, "Final")
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
):
    from . import log_based_checkpoint as _bdc

    shadow_bytes = sum(initial_state[nm].numel() * initial_state[nm].element_size() for nm in param_names)
    _logger.info(f"[Shadow Pipeline] P={P} producers, shadow_copy={shadow_bytes / 1e9:.2f}GB")

    t0_clone = time.perf_counter()
    working_state = _clone_working_state(initial_state, tied_groups)
    clone_working_s = time.perf_counter() - t0_clone
    _logger.info(f"[Shadow BootTiming] clone_working={clone_working_s:.3f}s mode=pipeline")
    if boot_started_at is not None:
        _logger.info(
            f"[Shadow BootTiming] ready_for_updates={time.perf_counter() - boot_started_at:.3f}s mode=pipeline"
        )
    if ready_event is not None:
        ready_event.set()
    base_step = int(initial_base_step)
    durable_step = int(initial_committed_step)
    consumer_step = durable_step + 1
    last_applied_step = durable_step
    desired_commit_step = durable_step
    pending_since_commit = 0

    result_queue = queue_module.Queue(maxsize=max(1, P))
    producer_stop = threading.Event()
    producer_error = [None]
    internal_updates = {}
    internal_lock = threading.Lock()
    update_available_event = threading.Event()
    assign_lock = threading.Lock()
    next_step_to_assign = [consumer_step]

    producer_timing = {"duration_ms": 0.0}
    retained_updates = {}
    generation = [0]

    def producer():
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
        for _ in range(P):
            t = threading.Thread(target=producer, daemon=True)
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
            for step_idx, update in retained_updates.items():
                if int(step_idx) >= int(consumer_step):
                    internal_updates[int(step_idx)] = update
        next_step_to_assign[0] = consumer_step
        if internal_updates:
            update_available_event.set()
        else:
            update_available_event.clear()

    def _pause_and_commit(target_step, reason):
        nonlocal threads, durable_step, desired_commit_step, pending_since_commit
        if int(target_step) <= int(durable_step):
            return 0.0
        t0_commit = time.monotonic()
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
            raise RuntimeError(f"[Shadow Pipeline] {reason} commit failed: {commit_result}")
        durable_step = int(commit_result["committed_step"])
        desired_commit_step = durable_step
        shadow_step_val.value = durable_step
        pending_since_commit = 0
        _log_durable_publish(durable_step, working_state, adam_state, _logger)
        _reseed_internal_updates()
        threads = _restart_producers()
        return (time.monotonic() - t0_commit) * 1000

    threads = _restart_producers()
    pending_results = {}

    while True:
        try:
            while True:
                cmd = update_queue.get_nowait()
                kind = cmd.get("cmd") if isinstance(cmd, dict) else None
                if kind == "stop":
                    _stop_producers(threads)
                    if last_applied_step > durable_step:
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
                            raise RuntimeError(f"[Shadow Pipeline] Final stop commit failed: {commit_result}")
                        durable_step = int(commit_result["committed_step"])
                        desired_commit_step = durable_step
                        shadow_step_val.value = durable_step
                        _log_durable_publish(durable_step, working_state, adam_state, _logger)
                    _logger.info("[Shadow Pipeline] Stopped")
                    return
                if kind == "rebase":
                    _stop_producers(threads)
                    generation[0] += 1
                    with internal_lock:
                        internal_updates.clear()
                    pending_results.clear()
                    _drain_result_queue()
                    update_available_event.clear()
                    anchor_ref = cmd.get("flat_storage") if isinstance(cmd, dict) else None
                    if anchor_ref is None:
                        anchor_ref = cmd["path"]
                    working_state, rebased_adam_state, base_step, durable_step = _rebase_working_state(
                        anchor_ref, tied_groups, _logger
                    )
                    if rebased_adam_state is not None:
                        adam_state = rebased_adam_state
                        if _step_diag_enabled() or _step_exact_enabled():
                            _log_adam_checksums(
                                f"shadow_rebase base_step={base_step} committed_step={durable_step}",
                                adam_state,
                                _logger=_logger,
                            )
                            if _step_exact_enabled():
                                _log_adam_exact_fingerprint(
                                    f"shadow_rebase base_step={base_step} committed_step={durable_step}",
                                    adam_state,
                                    _logger=_logger,
                                )
                    elif adam_state is not None:
                        raise RuntimeError("[Shadow] Rebase source has no Adam state for MeZO-Adam")
                    last_applied_step = _replay_retained_suffix(
                        retained_updates,
                        durable_step,
                        working_state,
                        param_names,
                        rng_device,
                        simulate_perturbation,
                        default_zo_eps,
                        adam_state,
                        _bdc,
                        _logger,
                    )
                    desired_commit_step = int(last_applied_step)
                    consumer_step = desired_commit_step + 1
                    pending_since_commit = max(0, desired_commit_step - durable_step)
                    _pause_and_commit(desired_commit_step, "Rebase")
                    _trim_retained_updates(retained_updates, durable_step)
                    continue
                if kind == "update":
                    update = cmd["update"]
                    retained_updates[int(update["step"])] = update
                    with internal_lock:
                        internal_updates[int(update["step"])] = update
                    update_available_event.set()
        except queue_module.Empty:
            pass

        if producer_error[0] is not None:
            raise producer_error[0]

        try:
            result_generation, step_idx, z_dict, update = result_queue.get(timeout=0.05)
        except queue_module.Empty:
            continue

        if result_generation != generation[0]:
            del z_dict
            continue
        pending_results[step_idx] = (z_dict, update)
        while consumer_step in pending_results:
            z_dict, update = pending_results.pop(consumer_step)
            t0_apply = time.monotonic()
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

            _rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
            _logger.info(
                f"[Shadow] step={last_applied_step} grad={update['grad']:.6e} seed={update['seed']} "
                f"| apply={apply_ms:.0f}ms zgen={producer_timing['duration_ms']:.0f}ms "
                f"commit={commit_ms:.0f}ms pending={pending_since_commit} "
                f"| applied={last_applied_step} desired={desired_commit_step} durable={durable_step} RSS={_rss_gb:.1f}GB"
            )
