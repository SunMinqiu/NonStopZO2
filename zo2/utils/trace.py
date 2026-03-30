import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable


_TRACE_FD = None
_TRACE_PATH = None
_TRACE_RUN_ID = None
_TRACE_WRITE_LOCK = threading.Lock()
_TRACE_EVENT_COUNTER = 0
_RESOURCE_SAMPLER_THREAD = None
_RESOURCE_SAMPLER_STOP = None


def trace_enabled() -> bool:
    return os.environ.get("ZO_TRACE", "0") == "1"


def timeline_trace_enabled() -> bool:
    if not trace_enabled():
        return False
    return os.environ.get("ZO_TRACE_TIMELINE", "1") == "1"


def resource_trace_enabled() -> bool:
    if not trace_enabled():
        return False
    return os.environ.get("ZO_TRACE_RESOURCE", "1") == "1"


def trace_path() -> str | None:
    return os.environ.get("ZO_TRACE_PATH")


def trace_run_id() -> str | None:
    return os.environ.get("ZO_TRACE_RUN_ID")


def default_trace_path(output_dir: str) -> str:
    return os.path.join(output_dir, "zo_trace.jsonl")


def configure_trace(
    *,
    path: str | None = None,
    run_id: str | None = None,
    process_role: str | None = None,
) -> str | None:
    global _TRACE_FD, _TRACE_PATH, _TRACE_RUN_ID
    if not trace_enabled():
        return None

    if path is not None:
        os.environ["ZO_TRACE_PATH"] = path
    if run_id is not None:
        os.environ["ZO_TRACE_RUN_ID"] = run_id
    if process_role is not None:
        os.environ["ZO_TRACE_PROCESS_ROLE"] = process_role

    resolved_path = trace_path()
    if not resolved_path:
        return None
    resolved_run_id = trace_run_id() or uuid.uuid4().hex[:16]
    os.environ["ZO_TRACE_RUN_ID"] = resolved_run_id

    if _TRACE_FD is None or _TRACE_PATH != resolved_path:
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        _TRACE_FD = os.open(resolved_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
        _TRACE_PATH = resolved_path
        _TRACE_RUN_ID = resolved_run_id

    return resolved_path


def _next_event_id() -> str:
    global _TRACE_EVENT_COUNTER
    _TRACE_EVENT_COUNTER += 1
    return f"{os.getpid()}-{threading.get_ident()}-{_TRACE_EVENT_COUNTER}"


def _sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return str(value)


def _write_trace_record(record: dict[str, Any]) -> None:
    if not trace_enabled():
        return
    path = configure_trace()
    if path is None:
        return
    payload = (json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
    with _TRACE_WRITE_LOCK:
        os.write(_TRACE_FD, payload)


def trace_instant(
    *,
    panel: str,
    lane: str,
    event: str,
    step: int | None = None,
    counters: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    triggered_by: str | None = None,
) -> str | None:
    if not timeline_trace_enabled():
        return None
    event_id = _next_event_id()
    record = {
        "run_id": trace_run_id(),
        "event_id": event_id,
        "triggered_by": triggered_by,
        "wall_time_ns": time.time_ns(),
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "thread_name": threading.current_thread().name,
        "process_role": os.environ.get("ZO_TRACE_PROCESS_ROLE", "unknown"),
        "panel": panel,
        "lane": lane,
        "event": event,
        "phase": "I",
        "step": step,
        "counters": _sanitize(counters or {}),
        "extra": _sanitize(extra or {}),
    }
    _write_trace_record(record)
    return event_id


def trace_resource_instant(
    *,
    panel: str,
    counters: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    event: str = "resource_sample",
) -> str | None:
    if not resource_trace_enabled():
        return None
    event_id = _next_event_id()
    record = {
        "run_id": trace_run_id(),
        "event_id": event_id,
        "triggered_by": None,
        "wall_time_ns": time.time_ns(),
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "thread_name": threading.current_thread().name,
        "process_role": os.environ.get("ZO_TRACE_PROCESS_ROLE", "unknown"),
        "panel": panel,
        "lane": "resource",
        "event": event,
        "phase": "I",
        "step": None,
        "counters": _sanitize(counters or {}),
        "extra": _sanitize(extra or {}),
    }
    _write_trace_record(record)
    return event_id


@dataclass
class TraceToken:
    event_id: str
    panel: str
    lane: str
    event: str
    step: int | None
    started_ns: int


def trace_begin(
    *,
    panel: str,
    lane: str,
    event: str,
    step: int | None = None,
    counters: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    triggered_by: str | None = None,
) -> TraceToken | None:
    if not timeline_trace_enabled():
        return None
    started_ns = time.time_ns()
    token = TraceToken(
        event_id=_next_event_id(),
        panel=panel,
        lane=lane,
        event=event,
        step=step,
        started_ns=started_ns,
    )
    record = {
        "run_id": trace_run_id(),
        "event_id": token.event_id,
        "triggered_by": triggered_by,
        "wall_time_ns": started_ns,
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "thread_name": threading.current_thread().name,
        "process_role": os.environ.get("ZO_TRACE_PROCESS_ROLE", "unknown"),
        "panel": panel,
        "lane": lane,
        "event": event,
        "phase": "B",
        "step": step,
        "counters": _sanitize(counters or {}),
        "extra": _sanitize(extra or {}),
    }
    _write_trace_record(record)
    return token


def trace_end(
    token: TraceToken | None,
    *,
    step: int | None = None,
    counters: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if token is None or not timeline_trace_enabled():
        return
    ended_ns = time.time_ns()
    duration_ms = (ended_ns - token.started_ns) / 1_000_000.0
    record = {
        "run_id": trace_run_id(),
        "event_id": token.event_id,
        "triggered_by": None,
        "wall_time_ns": ended_ns,
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "thread_name": threading.current_thread().name,
        "process_role": os.environ.get("ZO_TRACE_PROCESS_ROLE", "unknown"),
        "panel": token.panel,
        "lane": token.lane,
        "event": token.event,
        "phase": "E",
        "step": token.step if step is None else step,
        "duration_ms": duration_ms,
        "counters": _sanitize(counters or {}),
        "extra": _sanitize(extra or {}),
    }
    _write_trace_record(record)


def trace_end_external(
    *,
    event_id: str | None,
    panel: str,
    lane: str,
    event: str,
    started_ns: int | None,
    step: int | None = None,
    counters: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if event_id is None or not timeline_trace_enabled():
        return
    ended_ns = time.time_ns()
    duration_ms = None
    if started_ns is not None:
        duration_ms = (ended_ns - started_ns) / 1_000_000.0
    record = {
        "run_id": trace_run_id(),
        "event_id": event_id,
        "triggered_by": None,
        "wall_time_ns": ended_ns,
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "thread_name": threading.current_thread().name,
        "process_role": os.environ.get("ZO_TRACE_PROCESS_ROLE", "unknown"),
        "panel": panel,
        "lane": lane,
        "event": event,
        "phase": "E",
        "step": step,
        "duration_ms": duration_ms,
        "counters": _sanitize(counters or {}),
        "extra": _sanitize(extra or {}),
    }
    _write_trace_record(record)


class trace_span:
    def __init__(
        self,
        *,
        panel: str,
        lane: str,
        event: str,
        step: int | None = None,
        counters: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        triggered_by: str | None = None,
    ):
        self._kwargs = {
            "panel": panel,
            "lane": lane,
            "event": event,
            "step": step,
            "counters": counters,
            "extra": extra,
            "triggered_by": triggered_by,
        }
        self._token = None

    def __enter__(self):
        self._token = trace_begin(**self._kwargs)
        return self._token

    def __exit__(self, exc_type, exc, tb):
        extra = {}
        if exc is not None:
            extra["error"] = f"{type(exc).__name__}: {exc}"
        trace_end(self._token, extra=extra or None)
        return False


def start_resource_sampler(
    *,
    panel: str,
    provider: Callable[[], dict[str, Any] | None],
    interval_s: float | None = None,
) -> None:
    global _RESOURCE_SAMPLER_THREAD, _RESOURCE_SAMPLER_STOP
    if not resource_trace_enabled():
        return
    if _RESOURCE_SAMPLER_THREAD is not None:
        return

    stop_event = threading.Event()
    _RESOURCE_SAMPLER_STOP = stop_event
    sample_interval_s = interval_s
    if sample_interval_s is None:
        sample_interval_s = float(os.environ.get("ZO_TRACE_RESOURCE_INTERVAL_SEC", "1.0"))

    def _run():
        sample_seq = 0
        while not stop_event.is_set():
            try:
                counters = provider() or {}
                counters["sample_seq"] = sample_seq
                trace_resource_instant(panel=panel, counters=counters, event="resource_sample")
            except Exception as exc:
                trace_resource_instant(
                    panel=panel,
                    event="resource_sample_error",
                    extra={"error": f"{type(exc).__name__}: {exc}"},
                )
            sample_seq += 1
            stop_event.wait(sample_interval_s)

    _RESOURCE_SAMPLER_THREAD = threading.Thread(
        target=_run,
        daemon=True,
        name=f"zo-trace-resource-{panel}",
    )
    _RESOURCE_SAMPLER_THREAD.start()


def stop_resource_sampler() -> None:
    global _RESOURCE_SAMPLER_THREAD, _RESOURCE_SAMPLER_STOP
    if _RESOURCE_SAMPLER_STOP is not None:
        _RESOURCE_SAMPLER_STOP.set()
    if _RESOURCE_SAMPLER_THREAD is not None:
        _RESOURCE_SAMPLER_THREAD.join(timeout=2.0)
    _RESOURCE_SAMPLER_THREAD = None
    _RESOURCE_SAMPLER_STOP = None


def directory_size_bytes(path: str | None) -> int:
    if not path or not os.path.exists(path):
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def shm_usage_bytes(path: str = "/dev/shm/zo_ckpt") -> int:
    if not os.path.isdir(path):
        return 0
    total = 0
    for entry in os.scandir(path):
        try:
            if entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
    return total
