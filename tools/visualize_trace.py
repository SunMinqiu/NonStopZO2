#!/usr/bin/env python3
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


_GB_KEYS = {
    "train_rss_mb",
    "shadow_rss_mb",
    "gpu0_alloc_mb",
    "gpu0_reserved_mb",
    "gpu0_peak_mb",
    "zo_shm_used_mb",
}

_DISPLAY_NAMES = {
    "train_cpu_percent": "train_cpu_percent",
    "train_rss_mb": "train_rss_gb",
    "shadow_cpu_percent": "shadow_cpu_percent",
    "shadow_rss_mb": "shadow_rss_gb",
    "gpu0_alloc_mb": "gpu_alloc_gb",
    "gpu0_reserved_mb": "gpu_reserved_gb",
    "gpu0_peak_mb": "gpu_peak_gb",
    "zo_shm_used_mb": "dram_gb",
    "shadow_apply_backlog": "shadow_apply_backlog",
    "shadow_durable_lag": "shadow_durable_lag",
    "anchor_lag": "anchor_lag",
    "update_history_len": "update_history_len",
}

_TIMELINE_VISIBLE = {
    "framework_overhead",
    "train_step",
    "checkpoint_cpu_serialize",
    "checkpoint_rng_save",
    "checkpoint_disk_persist",
    "checkpoint_model_persist",
    "checkpoint_save",
    "wait_shadow_ready",
    "recover_shadow",
    "replay_updates",
    "train_end_cleanup",
    "resume_begin",
    "resume_end",
    "shadow_boot",
    "shadow_wait_update",
    "shadow_apply",
    "shadow_commit",
    "shadow_stop",
    "shadow_generate",
    "shadow_idle",
    "disk_anchor_persist",
    "wait_disk_anchor",
}

_TOPLEVEL_TIMELINE_EVENTS = {
    "framework_overhead",
    "train_step",
    "wait_shadow_ready",
    "recover_shadow",
    "replay_updates",
    "train_end_cleanup",
    "resume_begin",
    "resume_end",
    "shadow_boot",
    "shadow_wait_update",
    "shadow_apply",
    "shadow_commit",
    "shadow_stop",
    "disk_anchor_persist",
}

_SHADOW_BLOCK_EVENTS = {
    "wait_shadow_ready",
    "recover_shadow",
    "replay_updates",
    "resume_begin",
    "resume_end",
}

_LEGACY_TIMELINE_VISIBLE = {
    "checkpoint_d2h_copy",
    "wait_anchor_persist",
    "wait_anchor_buffer",
    "full_checkpoint_refresh",
    "shadow_rebase",
    "anchor_d2h_copy",
    "anchor_publish_latest",
    "anchor_persist",
}


def _format_scalar(name, value):
    if name == "learning_rate":
        return f"{value:.3e}"
    return f"{value:.6f}"


def _percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def load_records(path):
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    records.sort(key=lambda item: (item.get("wall_time_ns", 0), item.get("phase", "")))
    return records


def build_spans(records):
    open_tokens = {}
    spans = []
    instants = []
    for record in records:
        phase = record.get("phase")
        event_id = record.get("event_id")
        if phase == "B":
            open_tokens[event_id] = record
        elif phase == "E":
            begin = open_tokens.pop(event_id, None)
            if begin is None:
                continue
            spans.append(
                {
                    "panel": record["panel"],
                    "lane": record["lane"],
                    "event": record["event"],
                    "start_ns": begin["wall_time_ns"],
                    "end_ns": record["wall_time_ns"],
                    "duration_ms": record.get("duration_ms"),
                    "step": record.get("step"),
                    "pid": record.get("pid"),
                    "tid": record.get("tid"),
                    "triggered_by": begin.get("triggered_by"),
                    "counters": record.get("counters"),
                }
            )
        elif phase == "I":
            instants.append(record)
    return spans, instants


def _derive_idle_spans(spans):
    """Derive `shadow_idle` = `shadow_wait_update` \\ union(`shadow_generate`).

    Real idle = consumer waiting AND no producer is generating.
    """
    wait = [s for s in spans if s["event"] == "shadow_wait_update"]
    gen = [s for s in spans if s["event"] == "shadow_generate"]
    if not wait:
        return []
    # Sort generators by start for efficient overlap lookup
    gen_sorted = sorted(gen, key=lambda g: g["start_ns"])
    idle = []
    for w in wait:
        ws, we = w["start_ns"], w["end_ns"]
        # Collect busy intervals intersecting w
        busy = []
        for g in gen_sorted:
            if g["end_ns"] <= ws:
                continue
            if g["start_ns"] >= we:
                break
            busy.append((max(ws, g["start_ns"]), min(we, g["end_ns"])))
        # Merge overlapping busy segments
        busy.sort()
        merged = []
        for s, e in busy:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        # Subtract merged from [ws, we]
        cur = ws
        for s, e in merged:
            if s > cur:
                idle.append({
                    "panel": w["panel"], "lane": w["lane"],
                    "event": "shadow_idle",
                    "start_ns": cur, "end_ns": s,
                    "duration_ms": (s - cur) / 1_000_000.0,
                    "step": w.get("step"), "pid": w.get("pid"),
                    "tid": w.get("tid"), "triggered_by": None,
                    "counters": None,
                })
            cur = max(cur, e)
        if cur < we:
            idle.append({
                "panel": w["panel"], "lane": w["lane"],
                "event": "shadow_idle",
                "start_ns": cur, "end_ns": we,
                "duration_ms": (we - cur) / 1_000_000.0,
                "step": w.get("step"), "pid": w.get("pid"),
                "tid": w.get("tid"), "triggered_by": None,
                "counters": None,
            })
    return idle


def load_trace(path):
    records = load_records(path)
    spans, instants = build_spans(records)
    spans.extend(_derive_idle_spans(spans))
    return {
        "path": str(Path(path).resolve()),
        "records": records,
        "spans": spans,
        "instants": instants,
    }


def _ns_to_s(ns, origin_ns):
    return (ns - origin_ns) / 1_000_000_000.0


def _origin_ns(records, spans, instants):
    all_times = [item["wall_time_ns"] for item in records]
    for span in spans:
        all_times.extend([span["start_ns"], span["end_ns"]])
    for instant in instants:
        all_times.append(instant["wall_time_ns"])
    if not all_times:
        raise RuntimeError("trace has no events to plot")
    return min(all_times)


def _extract_series(records, event_name, key, panel=None):
    xs = []
    ys = []
    for record in records:
        if record.get("phase") != "I":
            continue
        if record.get("event") != event_name:
            continue
        if panel is not None and record.get("panel") != panel:
            continue
        counters = record.get("counters") or {}
        if key not in counters or counters[key] is None:
            continue
        xs.append(record["wall_time_ns"])
        ys.append(counters[key])
    return xs, ys


def _extract_series_any(records, event_name, keys, panel=None):
    xs = []
    ys = []
    matched_key = None
    for key in keys:
        xs, ys = _extract_series(records, event_name, key, panel=panel)
        if xs:
            matched_key = key
            break
    return xs, ys, matched_key


def _to_display_unit(label, values):
    if label in _GB_KEYS:
        return [value / 1024.0 for value in values]
    return values


def _extract_step_series(records, event_name, key, panel=None):
    xs = []
    ys = []
    for record in records:
        if record.get("phase") != "I":
            continue
        if record.get("event") != event_name:
            continue
        if panel is not None and record.get("panel") != panel:
            continue
        step = record.get("step")
        counters = record.get("counters") or {}
        value = counters.get(key)
        if step is None or value is None:
            continue
        xs.append(step)
        ys.append(value)
    return xs, ys


def _group_span_durations_by_event(spans):
    grouped = defaultdict(list)
    for span in spans:
        # For async d2h spans, prefer the accurate d2h_ms counter over
        # the wall-clock span duration (which includes queue wait + synchronize).
        counters = span.get("counters") or {}
        if span["event"] == "anchor_d2h_copy" and "d2h_ms" in counters:
            duration_ms = counters["d2h_ms"]
        else:
            duration_ms = span.get("duration_ms")
            if duration_ms is None:
                duration_ms = (span["end_ns"] - span["start_ns"]) / 1_000_000.0
        grouped[span["event"]].append(float(duration_ms))
    return grouped


def _collect_event_samples(grouped, event_names):
    samples = []
    for event_name in event_names:
        samples.extend(grouped.get(event_name, []))
    return samples


def _extract_loading_phase_breakdown(records, source):
    """Extract the latest loading_phase_breakdown instant for a given source (fresh/recovery).
    Returns a dict of {phase_name: ms, ..., _extra: {...}} or None."""
    for record in reversed(records):
        if record.get("phase") != "I":
            continue
        if record.get("event") != "loading_phase_breakdown":
            continue
        extra = record.get("extra") or {}
        if extra.get("source") != source:
            continue
        counters = record.get("counters") or {}
        result = {k: float(v) for k, v in counters.items()}
        result["_extra"] = extra
        return result
    return None


def _extract_instant_counter(records, event_name, key, panel=None, extra_key=None, extra_value=None):
    values = []
    for record in records:
        if record.get("phase") != "I":
            continue
        if record.get("event") != event_name:
            continue
        if panel is not None and record.get("panel") != panel:
            continue
        if extra_key is not None:
            extra = record.get("extra") or {}
            if extra.get(extra_key) != extra_value:
                continue
        counters = record.get("counters") or {}
        value = counters.get(key)
        if value is None:
            continue
        values.append(float(value))
    return values


def _stat_block(samples):
    if not samples:
        return None
    return {
        "count": len(samples),
        "total_ms": float(sum(samples)),
        "avg_ms": float(statistics.mean(samples)),
        "p50_ms": float(_percentile(samples, 0.50)),
        "p95_ms": float(_percentile(samples, 0.95)),
        "max_ms": float(max(samples)),
    }


def _derive_replay_steady(samples):
    if not samples:
        return None
    if len(samples) == 1:
        steady = float(samples[0])
        cold = 0.0
    else:
        tail = [float(v) for v in samples[1:]]
        steady = float(statistics.mean(tail))
        cold = max(0.0, float(samples[0]) - steady)
    result = _stat_block(samples)
    result["steady_avg_ms"] = steady
    result["cold_start_ms"] = cold
    return result


def summarize_trace(trace_or_records, spans=None, instants=None):
    if isinstance(trace_or_records, dict):
        records = trace_or_records["records"]
        spans = trace_or_records["spans"]
        instants = trace_or_records["instants"]
    else:
        records = trace_or_records
        if spans is None or instants is None:
            spans, instants = build_spans(records)

    origin_ns = _origin_ns(records, spans, instants)
    end_ns = max(
        [item["wall_time_ns"] for item in records] +
        [item["end_ns"] for item in spans] +
        [item["wall_time_ns"] for item in instants]
    )
    total_wall_s = (end_ns - origin_ns) / 1_000_000_000.0

    span_groups = defaultdict(list)
    panel_totals = defaultdict(float)
    lane_totals = defaultdict(float)
    for span in spans:
        duration_ms = span.get("duration_ms")
        if duration_ms is None:
            duration_ms = (span["end_ns"] - span["start_ns"]) / 1_000_000.0
        key = (span["panel"], span["lane"], span["event"])
        span_groups[key].append(float(duration_ms))
        panel_totals[span["panel"]] += float(duration_ms)
        lane_totals[(span["panel"], span["lane"])] += float(duration_ms)
    span_by_event = _group_span_durations_by_event(spans)

    span_summary = []
    total_wall_ms = total_wall_s * 1000.0
    for (panel, lane, event), durations in sorted(span_groups.items()):
        total_ms = sum(durations)
        span_summary.append(
            {
                "panel": panel,
                "lane": lane,
                "event": event,
                "count": len(durations),
                "total_ms": total_ms,
                "avg_ms": statistics.mean(durations),
                "p50_ms": _percentile(durations, 0.50),
                "p95_ms": _percentile(durations, 0.95),
                "max_ms": max(durations),
                "share_pct": (100.0 * total_ms / total_wall_ms) if total_wall_ms > 0 else 0.0,
            }
        )

    resource_specs = {
        "train_cpu_percent": ("resource_sample", ["train_cpu_percent"], "gpu_train"),
        "train_rss_mb": ("resource_sample", ["train_rss_mb"], "gpu_train"),
        "shadow_cpu_percent": ("resource_sample", ["shadow_cpu_percent"], "cpu_shadow"),
        "shadow_rss_mb": ("resource_sample", ["shadow_rss_mb"], "cpu_shadow"),
        "gpu0_alloc_mb": ("resource_sample", ["gpu0_alloc_mb", "gpu_alloc_mb"], "gpu_train"),
        "gpu0_reserved_mb": ("resource_sample", ["gpu0_reserved_mb", "gpu_reserved_mb"], "gpu_train"),
        "gpu0_peak_mb": ("resource_sample", ["gpu0_peak_mb", "gpu_peak_mb"], "gpu_train"),
        "zo_shm_used_mb": ("resource_sample", ["zo_shm_used_mb"], None),
        "shadow_apply_backlog": ("train_progress", ["shadow_apply_backlog"], "gpu_train"),
        "shadow_durable_lag": ("train_progress", ["shadow_durable_lag"], "gpu_train"),
        "anchor_lag": ("train_progress", ["anchor_lag"], "gpu_train"),
        "update_history_len": ("train_progress", ["update_history_len"], "gpu_train"),
    }
    resource_summary = {}
    missing = []
    for label, (event_name, keys, panel) in resource_specs.items():
        xs, ys, matched_key = _extract_series_any(records, event_name, keys, panel=panel)
        if not xs:
            missing.append(label)
            continue
        display_ys = _to_display_unit(label, ys)
        resource_summary[label] = {
            "display_name": _DISPLAY_NAMES.get(label, label),
            "source_key": matched_key,
            "samples": len(display_ys),
            "avg": statistics.mean(display_ys),
            "p50": _percentile(display_ys, 0.50),
            "p95": _percentile(display_ys, 0.95),
            "max": max(display_ys),
            "min": min(display_ys),
            "unit": "GB" if label in _GB_KEYS else "",
        }

    scalar_specs = {
        "loss": ("train_scalar", "loss", "gpu_train"),
    }
    scalar_summary = {}
    last_scalars = {}
    for label, (event_name, key, panel) in scalar_specs.items():
        xs, ys = _extract_step_series(records, event_name, key, panel=panel)
        if not xs:
            continue
        scalar_summary[label] = {
            "samples": len(ys),
            "avg": statistics.mean(ys),
            "p50": _percentile(ys, 0.50),
            "p95": _percentile(ys, 0.95),
            "max": max(ys),
            "min": min(ys),
        }
        last_scalars[label] = {
            "step": xs[-1],
            "value": ys[-1],
        }

    replay_cuda = _extract_instant_counter(records, "replay_step", "replay_total_ms", panel="gpu_train", extra_key="device", extra_value="cuda")
    replay_cpu = _extract_instant_counter(records, "replay_step", "replay_total_ms", panel="gpu_train", extra_key="device", extra_value="cpu")
    first_step_latency_fresh = _extract_instant_counter(records, "first_step_latency", "program_to_first_step_ms", panel="gpu_train", extra_key="source", extra_value="fresh")
    _recovery_program_to_first = _extract_instant_counter(records, "first_step_latency", "program_to_first_step_ms", panel="gpu_train", extra_key="source", extra_value="recovery")
    _recovery_replay_ms = _extract_instant_counter(records, "replay_step", "replay_total_ms", panel="gpu_train", extra_key="device", extra_value="cuda")
    # L_cpu = program_to_first_step minus replay time (replay is accounted separately as t_r * D)
    _replay_total_for_recovery = sum(_recovery_replay_ms) if _recovery_replay_ms else 0.0
    first_step_latency_recovery = [p - _replay_total_for_recovery / max(1, len(_recovery_program_to_first)) for p in _recovery_program_to_first] if _recovery_program_to_first else []
    # Backwards compat: if no source tag found, fall back to untagged instants
    if not first_step_latency_fresh and not first_step_latency_recovery:
        first_step_latency_fresh = _extract_instant_counter(records, "first_step_latency", "program_to_first_step_ms", panel="gpu_train")
    recovery_load_shadow = _extract_instant_counter(records, "recovery_load", "load_cpu_to_gpu_ms", panel="gpu_train", extra_key="source", extra_value="shadow")

    # Extract loading phase breakdowns (emitted by _emit_loading_phase_breakdown)
    _loading_breakdown_fresh = _extract_loading_phase_breakdown(records, source="fresh")
    _loading_breakdown_recovery = _extract_loading_phase_breakdown(records, source="recovery")

    replay_cuda_stats = _derive_replay_steady(replay_cuda)
    replay_cpu_stats = _derive_replay_steady(replay_cpu)
    replay_cold_ms = 0.0
    if replay_cuda_stats is not None:
        replay_cold_ms = replay_cuda_stats["cold_start_ms"]
    elif replay_cpu_stats is not None:
        replay_cold_ms = replay_cpu_stats["cold_start_ms"]

    checkpoint_total_ms = float(sum(span_by_event.get("checkpoint_save", [])))
    framework_overhead_total_ms = float(sum(span_by_event.get("framework_overhead", [])))
    train_step_total_ms = float(sum(span_by_event.get("train_step", [])))
    shadow_block_total_ms = float(sum(sum(span_by_event.get(event, [])) for event in _SHADOW_BLOCK_EVENTS))
    train_step_count = len(span_by_event.get("train_step", []))
    derived_t_step_total_ms = max(0.0, total_wall_ms - checkpoint_total_ms - shadow_block_total_ms)

    checkpoint_d2h_samples = _collect_event_samples(
        span_by_event,
        [
            "anchor_d2h_copy",
            "checkpoint_d2h_copy",
        ],
    )
    checkpoint_persist_samples = _collect_event_samples(
        span_by_event,
        [
            "anchor_persist",
            "checkpoint_model_persist",
            "checkpoint_disk_persist",
            "checkpoint_rng_save",
        ],
    )

    named_time_metrics = {
        "checkpoint_total": {**(_stat_block(span_by_event.get("checkpoint_save", [])) or {}), "status": "exact"} if span_by_event.get("checkpoint_save") else None,
        "t_step": {
            "count": train_step_count,
            "total_ms": derived_t_step_total_ms,
            "avg_ms": (derived_t_step_total_ms / train_step_count) if train_step_count > 0 else 0.0,
            "status": "derived",
        } if train_step_count > 0 else None,
        "t_l": {**(_stat_block(span_by_event.get("log_send_cpu", [])) or {}), "status": "exact"} if span_by_event.get("log_send_cpu") else None,
        "t_d2h": {
            **(_stat_block(checkpoint_d2h_samples) or {}),
            "status": "exact",
        } if checkpoint_d2h_samples else None,
        "t_persist": {
            **(_stat_block(checkpoint_persist_samples) or {}),
            "status": "exact",
        } if checkpoint_persist_samples else None,
        "t_cp": None,  # derived below: commit_avg / commit_interval
        "t_r": {**replay_cuda_stats, "status": "derived"} if replay_cuda_stats is not None else None,
        "t_rc": {**(_stat_block(span_by_event.get("shadow_apply", [])) or {}), "status": "exact"} if span_by_event.get("shadow_apply") else None,
        "L_disk": {
            "count": len(first_step_latency_fresh),
            "total_ms": float(sum(first_step_latency_fresh)),
            "avg_ms": float(statistics.mean(first_step_latency_fresh)),
            "p50_ms": float(_percentile(first_step_latency_fresh, 0.50)),
            "p95_ms": float(_percentile(first_step_latency_fresh, 0.95)),
            "max_ms": float(max(first_step_latency_fresh)),
            "status": "exact",
        } if first_step_latency_fresh else None,
        "L_cpu": {
            "count": len(first_step_latency_recovery),
            "total_ms": float(sum(first_step_latency_recovery)),
            "avg_ms": float(statistics.mean(first_step_latency_recovery)),
            "p50_ms": float(_percentile(first_step_latency_recovery, 0.50)),
            "p95_ms": float(_percentile(first_step_latency_recovery, 0.95)),
            "max_ms": float(max(first_step_latency_recovery)),
            "status": "exact",
        } if first_step_latency_recovery else None,
        "L_cpu_cold": {
            "count": 1,
            "total_ms": float(replay_cold_ms),
            "avg_ms": float(replay_cold_ms),
            "p50_ms": float(replay_cold_ms),
            "p95_ms": float(replay_cold_ms),
            "max_ms": float(replay_cold_ms),
            "status": "derived",
        } if replay_cold_ms > 0.0 else None,
        "L_disk_breakdown": _loading_breakdown_fresh,
        "L_cpu_breakdown": _loading_breakdown_recovery,
    }

    # Derive t_cp = amortised shadow commit cost per training step
    shadow_commit_spans = span_by_event.get("shadow_commit", [])
    shadow_apply_spans = span_by_event.get("shadow_apply", [])
    raw_commit_stats = _stat_block(shadow_commit_spans)
    if raw_commit_stats and shadow_commit_spans and shadow_apply_spans:
        commit_count = len(shadow_commit_spans)
        apply_count = len(shadow_apply_spans)
        commit_interval = max(1, apply_count // commit_count) if commit_count > 0 else 1
        named_time_metrics["t_cp"] = {
            "count": apply_count,
            "total_ms": raw_commit_stats["total_ms"],
            "avg_ms": raw_commit_stats["avg_ms"] / commit_interval,
            "commit_interval": commit_interval,
            "status": "derived",
        }

    return {
        "overview": {
            "trace_start_ns": origin_ns,
            "trace_end_ns": end_ns,
            "total_wall_s": total_wall_s,
            "checkpoint_total_s": checkpoint_total_ms / 1000.0,
            "train_step_total_s": train_step_total_ms / 1000.0,
            "framework_overhead_total_s": framework_overhead_total_ms / 1000.0,
            "shadow_block_total_s": shadow_block_total_ms / 1000.0,
            "approx_sum_s": (checkpoint_total_ms + train_step_total_ms + framework_overhead_total_ms + shadow_block_total_ms) / 1000.0,
            "record_count": len(records),
            "span_count": len(spans),
            "instant_count": len(instants),
            "resource_sample_count": sum(1 for r in records if r.get("event") == "resource_sample"),
        },
        "panel_totals_ms": dict(sorted(panel_totals.items())),
        "lane_totals_ms": {
            f"{panel}/{lane}": total for (panel, lane), total in sorted(lane_totals.items())
        },
        "span_summary": span_summary,
        "resource_summary": resource_summary,
        "scalar_summary": scalar_summary,
        "last_scalars": last_scalars,
        "named_time_metrics": named_time_metrics,
        "missing_series": missing,
    }


def print_summary(summary, *, top_n=20):
    overview = summary["overview"]
    print("=== Trace Overview ===")
    print(f"total_wall_s: {overview['total_wall_s']:.3f}")
    print()

    print("=== Named Time Metrics ===")
    for name in ("checkpoint_total", "t_step", "t_l", "t_d2h", "t_persist", "t_r", "t_rc", "t_cp", "L_disk", "L_cpu"):
        stats = summary["named_time_metrics"].get(name)
        if not stats:
            print(f"{name}: missing")
            continue
        if name == "t_l":
            line = (
                f"{name}: total_ms={stats['total_ms']:.3f} "
                f"avg_ms={stats['avg_ms']:.3f}"
            )
        else:
            line = (
                f"{name}: total_s={stats['total_ms'] / 1000.0:.3f} "
                f"avg_s={stats['avg_ms'] / 1000.0:.3f}"
            )
        if "steady_avg_ms" in stats:
            line += f" steady_avg_s={stats['steady_avg_ms'] / 1000.0:.3f}"
        if "cold_start_ms" in stats:
            line += f" cold_start_s={stats['cold_start_ms'] / 1000.0:.3f}"
        line += f" count={stats['count']}"
        print(line)
        # Print phase breakdown under L_disk / L_cpu
        breakdown_key = f"{name}_breakdown" if name in ("L_disk", "L_cpu") else None
        breakdown = summary["named_time_metrics"].get(breakdown_key) if breakdown_key else None
        if breakdown:
            total_ms = breakdown.get("total_ms", 0.0)
            _phase_order = [
                "T_import", "T_main_setup", "T_config", "T_from_pretrained",
                "T_zo_init", "T_tokenizer", "T_tokenize_data", "T_trainer_init",
                "T_callback_setup", "T_resume_total", "T_cpu_to_gpu",
                "T_diag", "T_hf_inner",
            ]
            _extra = breakdown.get("_extra") or {}
            _weight_source = _extra.get("weight_source", "")
            _resume_source = _extra.get("resume_source", "")
            _inplace = _extra.get("inplace", False)
            for phase in _phase_order:
                phase_ms = breakdown.get(f"{phase}_ms")
                if phase_ms is None:
                    continue
                pct = phase_ms / total_ms * 100.0 if total_ms > 0 else 0.0
                ann = ""
                if phase == "T_from_pretrained" and _weight_source:
                    ann = f" [{_weight_source}]"
                elif phase == "T_resume_total" and _resume_source:
                    ann = f" [{_resume_source}]"
                elif phase == "T_cpu_to_gpu" and _inplace:
                    ann = " [inplace]"
                print(f"  {phase:22s} = {phase_ms / 1000.0:7.3f}s ({pct:5.1f}%){ann}")
    print()

    print("=== Resource Stats ===")
    for _name, stats in sorted(summary["resource_summary"].items()):
        print(
            f"{stats['display_name']}: avg={stats['avg']:.3f} max={stats['max']:.3f}"
        )
    if summary["last_scalars"]:
        print()
        print("=== Last Scalars ===")
        for name, stats in sorted(summary["last_scalars"].items()):
            print(f"{name}: step={stats['step']} value={_format_scalar(name, stats['value'])}")
    if summary["missing_series"]:
        print()
        print("=== Missing Series ===")
        for name in summary["missing_series"]:
            print(name)
    l_cpu_cold = summary["named_time_metrics"].get("L_cpu_cold")
    if l_cpu_cold:
        print()
        print(
            f"L_cpu_cold: total_s={l_cpu_cold['total_ms'] / 1000.0:.3f} "
            f"avg_s={l_cpu_cold['avg_ms'] / 1000.0:.3f} count={l_cpu_cold['count']}"
        )


def plot_timeline(
    trace_or_records,
    spans=None,
    instants=None,
    *,
    figsize=None,
    width=16.0,
    base_fontsize=None,
    save_path=None,
):
    """Plot the merged GPU + CPU timeline.

    Args:
        figsize: (w, h) tuple. If None, height is derived from `width` using
            the golden ratio (h = w / 1.618) so the figure has a clean
            ~1.618:1 aspect.
        width: used when `figsize` is None. Default 16 inches.
        base_fontsize: base font size used for lane (ytick) labels. Axis
            labels are drawn at `base_fontsize + 3`, the GPU/CPU section
            labels at `base_fontsize + 5`. If None, falls back to matplotlib
            rcParams["ytick.labelsize"] (which may be a float or a string
            like "medium" → resolved via FontProperties).
        save_path: optional path (str or pathlib.Path) to save the figure.
            If the path ends in a recognized matplotlib extension (e.g.
            ".pdf", ".png", ".svg") the format is inferred; otherwise PDF
            is used. Parent directories are created if missing.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot timeline figures") from exc

    # Resolve base font size. If the caller passed a number use it directly;
    # otherwise pull the default from rcParams and resolve symbolic sizes
    # ("small"/"medium"/"large") into absolute points via FontProperties.
    if base_fontsize is None:
        _rc_size = plt.rcParams.get("ytick.labelsize", 10)
        try:
            base_fontsize = float(_rc_size)
        except (TypeError, ValueError):
            base_fontsize = float(FontProperties(size=_rc_size).get_size_in_points())
    axis_label_size = base_fontsize + 3
    section_label_size = base_fontsize + 10

    # Golden-ratio figure by default.
    if figsize is None:
        _phi = 1.6180339887
        figsize = (float(width), float(width) / _phi)

    if isinstance(trace_or_records, dict):
        records = trace_or_records["records"]
        spans = trace_or_records["spans"]
        instants = trace_or_records["instants"]
    else:
        records = trace_or_records
        if spans is None or instants is None:
            spans, instants = build_spans(records)

    panel_order = ["gpu_train", "cpu_shadow"]
    panel_titles = {"gpu_train": "GPU / Train", "cpu_shadow": "CPU / Shadow"}
    # Events hidden from the timeline (setup/teardown noise + raw
    # shadow_wait_update which is replaced by derived shadow_idle).
    _HIDDEN_EVENTS = {
        "wait_shadow_ready", "shadow_boot", "shadow_stop",
        "resume_begin", "resume_end",
        "train_end_cleanup",
        "shadow_wait_update",
    }
    # Detect LOG_BASED_CKPT mode from trace_config instant event to decide
    # how to label the merged HF-checkpoint row.
    _merged_ckpt_mode = "step_log"  # default fallback
    for item in instants:
        if item.get("event") == "trace_config":
            _bs = (item.get("extra") or {}).get("batch_size")
            if _bs is not None:
                _merged_ckpt_mode = "step_log" if int(_bs) == 0 else "anchor_checkpoint"
            break
    # The 5 HF Trainer save events all happen together in _save_checkpoint
    # (trainer_state.json + scheduler.pt + rng_state.pth + optimizer.pt +
    # wrapper span). Merge them into a single row on the plot so it's not
    # visual noise. Print summary still shows them separately.
    _MERGED_CKPT_EVENTS = {
        "checkpoint_save",
        "checkpoint_cpu_serialize",
        "checkpoint_rng_save",
        "checkpoint_disk_persist",
        "checkpoint_model_persist",
    }
    _MERGED_CKPT_LABEL = _merged_ckpt_mode  # "step_log" or "anchor_checkpoint"
    # HF Trainer's "framework_overhead" span covers the bookkeeping between
    # train_step events (dataloader, callbacks, LR update, ...). For paper
    # plots, merge it onto the "train" row — they're back-to-back anyway.
    # Print summary keeps them separate.
    _MERGE_INTO_TRAIN = {"framework_overhead"}
    # Display-only rename: keeps underlying trace event names stable.
    _DISPLAY_NAMES = {
        "train_step": "train",
        "replay_updates": "replay",
        "shadow_apply": "update",
        "shadow_commit": "snapshot",
        "shadow_generate": "generate",
        "shadow_idle": "idle",
        "disk_anchor_persist": "persist",
        "wait_disk_anchor": "wait_persist",
        _MERGED_CKPT_LABEL: _MERGED_CKPT_LABEL,
    }
    # Unified row order for both panels. Each panel's rows are the subset
    # of this list that actually appear in that panel's events, so the
    # relative order below determines per-panel display order:
    #   GPU / Train:  loading → train → checkpoint → (recover_shadow) → replay
    #   CPU / Shadow: loading → generate → update → idle → snapshot
    #                 → (wait_persist) → persist
    # Legacy rows appended after so old traces still render.
    # Note: framework_overhead is merged into train_step (see _MERGE_INTO_TRAIN).
    row_order = [
        "loading",                 # both panels
        # --- GPU / Train rows ---
        "train_step",              # "train"
        _MERGED_CKPT_LABEL,        # "step_log" or "anchor_checkpoint"
        "recover_shadow",          # retry-only, sits between checkpoint and replay
        "replay_updates",          # "replay"
        # --- CPU / Shadow rows ---
        "shadow_generate",         # "generate"
        "shadow_apply",            # "update"
        "shadow_idle",             # "idle"
        "shadow_commit",           # "snapshot"
        "wait_disk_anchor",        # "wait_persist" (only if real blocking)
        "disk_anchor_persist",     # "persist"
        # --- Legacy rows (kept for rendering old traces; empty for new runs) ---
        "checkpoint_d2h_copy",
        "wait_anchor_persist",
        "wait_anchor_buffer",
        "full_checkpoint_refresh",
        "anchor_d2h_copy",
        "anchor_publish_latest",
        "anchor_persist",
        "shadow_rebase",
    ]
    colors = {
        "loading": "#9467bd",
        "shadow_generate": "#ff7f0e",
        "shadow_idle": "#e0e0e0",
        "framework_overhead": "#c7c7c7",
        "train_step": "#1f77b4",
        "zo_update_hook": "#ff7f0e",
        "checkpoint_save": "#2ca02c",
        "checkpoint_d2h_copy": "#98df8a",
        "checkpoint_cpu_serialize": "#ffbb78",
        "checkpoint_rng_save": "#c5b0d5",
        "checkpoint_disk_persist": "#8c564b",
        "checkpoint_model_persist": "#17becf",
        "wait_shadow_ready": "#d62728",
        "wait_anchor_persist": "#9467bd",
        "wait_anchor_buffer": "#8c564b",
        "recover_shadow": "#e377c2",
        "replay_updates": "#7f7f7f",
        "full_checkpoint_refresh": "#bcbd22",
        "shadow_boot": "#17becf",
        "shadow_wait_update": "#c7c7c7",
        "shadow_apply": "#1f77b4",
        "shadow_commit": "#2ca02c",
        "shadow_rebase": "#d62728",
        "anchor_d2h_copy": "#ff9896",
        "anchor_publish_latest": "#98df8a",
        "anchor_persist": "#aec7e8",
        "train_end_cleanup": "#c49c94",
        "resume_begin": "#9edae5",
        "resume_end": "#9edae5",
        # Phase 2: shadow-side disk anchor
        "disk_anchor_persist": "#17becf",
        "wait_disk_anchor": "#f7b6d2",
        # Merged HF checkpoint row (label varies by LOG_BASED_CKPT mode)
        "step_log": "#2ca02c",
        "anchor_checkpoint": "#bcbd22",
    }

    origin_ns = _origin_ns(records, spans, instants)

    # Find first train_step start — everything before it is collapsed into "loading".
    first_train_ns = None
    for item in spans:
        if item["event"] == "train_step":
            if first_train_ns is None or item["start_ns"] < first_train_ns:
                first_train_ns = item["start_ns"]

    # Detect failure+recovery windows in retry runs. The trace file is
    # append-mode, so retry runs accumulate in the same jsonl. A retry
    # produces `resume_begin` → recover_shadow / replay_updates → `resume_end`
    # → new train_step. The failure itself is the gap between the last
    # pre-crash train_step end and the first resume_begin.
    #
    # For each resume_begin, we derive two windows:
    #   - failure window  = [last train_step end before resume_begin,
    #                        resume_begin wall_time]
    #     → drawn as red axvspan across both panels
    #   - loading window  = [resume_begin wall_time,
    #                        first train_step start after the resume_end]
    #     → emitted as a synthetic "loading" span so it joins the existing
    #       loading row
    train_span_bounds = sorted(
        [(s["start_ns"], s["end_ns"]) for s in spans if s["event"] == "train_step"]
    )
    resume_begins = sorted(
        [int(i["wall_time_ns"]) for i in instants if i.get("event") == "resume_begin"]
    )
    # replay_updates is a concrete recovery sub-phase that gets its own row.
    # Subtract it from the synthetic "loading" window so the loading bar
    # reflects the REAL wall-clock time spent on other recovery work
    # (model skeleton build, base checkpoint load, HF Trainer setup, ...),
    # not the union that includes replay.
    replay_bounds = sorted(
        [(s["start_ns"], s["end_ns"]) for s in spans if s["event"] == "replay_updates"]
    )

    def _subtract_intervals(window_start, window_end, sub_intervals):
        """Return segments of [window_start, window_end] not covered by any
        interval in sub_intervals."""
        clipped = []
        for s, e in sub_intervals:
            cs = max(s, window_start)
            ce = min(e, window_end)
            if cs < ce:
                clipped.append((cs, ce))
        if not clipped:
            return [(window_start, window_end)]
        clipped.sort()
        merged = [list(clipped[0])]
        for s, e in clipped[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        result = []
        cursor = window_start
        for s, e in merged:
            if cursor < s:
                result.append((cursor, s))
            cursor = e
        if cursor < window_end:
            result.append((cursor, window_end))
        return result

    failure_windows: list[tuple[int, int]] = []  # (failure_start_ns, failure_end_ns)
    post_failure_loading: list[tuple[int, int]] = []  # segments not covered by sub-spans
    for rb_ns in resume_begins:
        # Failure start = end of last train_step before this resume_begin.
        prev_train_end = None
        for s_ns, e_ns in train_span_bounds:
            if e_ns <= rb_ns:
                prev_train_end = e_ns
            else:
                break
        if prev_train_end is None:
            continue  # resume_begin with no preceding train_step — skip
        failure_windows.append((prev_train_end, rb_ns))
        # Full recovery window = resume_begin → first train_step after.
        next_train_start = None
        for s_ns, e_ns in train_span_bounds:
            if s_ns >= rb_ns:
                next_train_start = s_ns
                break
        if next_train_start is None:
            continue
        # Loading = recovery window minus replay_updates spans within it.
        # Produces one or two disjoint segments (pre-replay setup + post-replay setup).
        for seg_start, seg_end in _subtract_intervals(
            rb_ns, next_train_start, replay_bounds
        ):
            post_failure_loading.append((seg_start, seg_end))

    # === Merged single-axis layout ===
    # Loading is a shared row at top. GPU rows follow (train / checkpoint /
    # replay / recover), then a horizontal divider, then CPU rows (generate /
    # update / idle / snapshot / persist / wait_persist). Failure window is
    # an axvspan across the whole figure so it visually connects both halves.
    _GPU_PANEL_EVENTS = {
        "train_step",
        _MERGED_CKPT_LABEL,
        "recover_shadow",
        "replay_updates",
        # legacy GPU
        "checkpoint_d2h_copy",
        "wait_anchor_persist",
        "wait_anchor_buffer",
        "full_checkpoint_refresh",
        "anchor_d2h_copy",
        "anchor_publish_latest",
        "anchor_persist",
    }
    _CPU_PANEL_EVENTS = {
        "shadow_generate",
        "shadow_apply",
        "shadow_idle",
        "shadow_commit",
        "wait_disk_anchor",
        "disk_anchor_persist",
        "shadow_rebase",
    }

    merged_spans = []
    for item in spans:
        if item["event"] not in (_TIMELINE_VISIBLE | _LEGACY_TIMELINE_VISIBLE):
            continue
        if item["event"] in _HIDDEN_EVENTS:
            continue
        if first_train_ns is not None and item["start_ns"] < first_train_ns:
            continue
        if item["event"] in _MERGED_CKPT_EVENTS:
            merged = dict(item)
            merged["event"] = _MERGED_CKPT_LABEL
            merged_spans.append(merged)
        elif item["event"] in _MERGE_INTO_TRAIN:
            merged = dict(item)
            merged["event"] = "train_step"
            merged_spans.append(merged)
        else:
            merged_spans.append(item)

    # Synthetic pre-train "loading" span (emit once, not per panel).
    if first_train_ns is not None and first_train_ns > origin_ns:
        merged_spans.append({
            "event": "loading",
            "start_ns": origin_ns,
            "end_ns": first_train_ns,
        })
    # Post-failure loading (one per retry).
    for load_start_ns, load_end_ns in post_failure_loading:
        merged_spans.append({
            "event": "loading",
            "start_ns": load_start_ns,
            "end_ns": load_end_ns,
        })

    merged_instants = [
        item for item in instants
        if item["event"] in (_TIMELINE_VISIBLE | _LEGACY_TIMELINE_VISIBLE)
        and item["event"] not in _HIDDEN_EVENTS
        and (first_train_ns is None or item.get("wall_time_ns", 0) >= first_train_ns)
    ]

    present = {item["event"] for item in merged_spans}
    present.update(
        item["event"]
        for item in merged_instants
        if item["lane"] not in ("resource", "counters", "meta")
    )
    row_keys = [key for key in row_order if key in present]
    y_positions = {key: idx * 10 for idx, key in enumerate(row_keys)}

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for item in merged_spans:
        y = y_positions.get(item["event"])
        if y is None:
            continue
        start_s = _ns_to_s(item["start_ns"], origin_ns)
        width_s = max(_ns_to_s(item["end_ns"], item["start_ns"]), 1e-6)
        ax.broken_barh(
            [(start_s, width_s)],
            (y, 8),
            facecolors=colors.get(item["event"], "#4c78a8"),
            edgecolors="black",
            linewidth=0.4,
            alpha=0.9,
        )

    for item in merged_instants:
        if item["lane"] in ("resource", "counters", "meta"):
            continue
        y = y_positions.get(item["event"])
        if y is None:
            continue
        x = _ns_to_s(item["wall_time_ns"], origin_ns)
        ax.plot([x], [y + 4], marker="|", markersize=12, color="black")

    # Red failure band: vertical extent = top of the first row ↔ bottom of
    # the last row (not full axis, so it doesn't go beyond the lanes).
    if row_keys and failure_windows:
        _row_top_y = 0
        _row_bottom_y = (len(row_keys) - 1) * 10 + 8
        for fail_start_ns, fail_end_ns in failure_windows:
            fail_start_s = _ns_to_s(fail_start_ns, origin_ns)
            fail_end_s = _ns_to_s(fail_end_ns, origin_ns)
            # firebrick (#b22222) — deeper/warmer red than default, not garish.
            ax.fill_betweenx(
                [_row_top_y, _row_bottom_y],
                fail_start_s, fail_end_s,
                color="#FF0000", alpha=0.6, zorder=0,
            )

    # Horizontal divider between GPU and CPU rows.
    gpu_row_indices = [i for i, k in enumerate(row_keys) if k in _GPU_PANEL_EVENTS]
    cpu_row_indices = [i for i, k in enumerate(row_keys) if k in _CPU_PANEL_EVENTS]

    ax.set_yticks([y_positions[key] + 4 for key in row_keys] or [0])
    ax.set_yticklabels(
        [_DISPLAY_NAMES.get(key, key) for key in row_keys] or ["no_events"],
        fontsize=base_fontsize,
        fontweight="bold",
    )
    for tick in ax.get_xticklabels():
        tick.set_fontsize(base_fontsize)
        tick.set_fontweight("bold")
    # Flip so row_order reads top-to-bottom.
    ax.invert_yaxis()
    ax.set_xlabel("Wall Time (s)", fontsize=axis_label_size, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # Place GPU/CPU section labels far to the left of the tick labels.
    # Font size is lane-label size + 3 to make them stand out.
    if gpu_row_indices and cpu_row_indices:
        last_gpu = max(gpu_row_indices)
        first_cpu = min(cpu_row_indices)
        divider_y = (last_gpu * 10 + 8 + first_cpu * 10) / 2
        ax.axhline(divider_y, color="black", linewidth=1.2, alpha=0.6)

        gpu_mid = (min(gpu_row_indices) * 10 + last_gpu * 10 + 8) / 2
        cpu_mid = (first_cpu * 10 + max(cpu_row_indices) * 10 + 8) / 2
        # Use axes-fraction x, data-coord y (get_yaxis_transform), well to
        # the left of the y tick labels so they don't overlap.
        ax.text(-0.12, gpu_mid, "GPU",
                transform=ax.get_yaxis_transform(),
                ha="right", va="center",
                fontsize=section_label_size, fontweight="bold")
        ax.text(-0.12, cpu_mid, "CPU",
                transform=ax.get_yaxis_transform(),
                ha="right", va="center",
                fontsize=section_label_size, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        _save_path = Path(save_path)
        _save_path.parent.mkdir(parents=True, exist_ok=True)
        # If no recognized extension, default to .pdf
        _known_exts = {".pdf", ".png", ".svg", ".jpg", ".jpeg", ".eps", ".ps"}
        if _save_path.suffix.lower() not in _known_exts:
            _save_path = _save_path.with_suffix(".pdf")
        fig.savefig(str(_save_path), bbox_inches="tight")

    return fig, ax


def plot_resources(trace_or_records, *, figsize=(18, 10)):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot resource figures") from exc

    if isinstance(trace_or_records, dict):
        records = trace_or_records["records"]
        spans = trace_or_records["spans"]
        instants = trace_or_records["instants"]
    else:
        records = trace_or_records
        spans, instants = build_spans(records)

    origin_ns = _origin_ns(records, spans, instants)
    width, total_height = figsize
    per_height = max(total_height / 4.0, 2.5)

    fig_mem, ax_mem = plt.subplots(1, 1, figsize=(width, per_height))
    fig_gpu, ax_gpu = plt.subplots(1, 1, figsize=(width, per_height))
    fig_cpu, ax_cpu = plt.subplots(1, 1, figsize=(width, per_height))
    fig_lag, ax_lag = plt.subplots(1, 1, figsize=(width, per_height))

    memory_specs = [
        ("gpu_train", ["train_rss_mb"], "Train RSS", "#2ca02c"),
        ("cpu_shadow", ["shadow_rss_mb"], "Shadow RSS", "#d62728"),
        (None, ["zo_shm_used_mb"], "DRAM", "#9467bd"),
    ]
    for panel, keys, label, color in memory_specs:
        xs, ys, _ = _extract_series_any(records, "resource_sample", keys, panel=panel)
        if not xs:
            continue
        ax_mem.plot([_ns_to_s(x, origin_ns) for x in xs], [y / 1024.0 for y in ys], label=label, color=color)

    ax_mem.set_ylabel("GB")
    ax_mem.grid(axis="x", linestyle="--", alpha=0.35)
    handles, labels = ax_mem.get_legend_handles_labels()
    if handles:
        ax_mem.legend(loc="upper left", ncol=3)

    gpu_xs, gpu_ys, _ = _extract_series_any(records, "resource_sample", ["gpu0_alloc_mb", "gpu_alloc_mb"], panel="gpu_train")
    if gpu_xs:
        ax_gpu.plot([_ns_to_s(x, origin_ns) for x in gpu_xs], [y / 1024.0 for y in gpu_ys], label="GPU alloc", color="#1f77b4")
    ax_gpu.set_ylabel("GB")
    ax_gpu.grid(axis="x", linestyle="--", alpha=0.35)
    handles, labels = ax_gpu.get_legend_handles_labels()
    if handles:
        ax_gpu.legend(loc="upper left", ncol=2)

    cpu_specs = [
        ("gpu_train", ["train_cpu_percent"], "Train CPU%", "#8c564b"),
        ("cpu_shadow", ["shadow_cpu_percent"], "Shadow CPU%", "#e377c2"),
    ]
    for panel, keys, label, color in cpu_specs:
        xs, ys, _ = _extract_series_any(records, "resource_sample", keys, panel=panel)
        if not xs:
            continue
        ax_cpu.plot([_ns_to_s(x, origin_ns) for x in xs], ys, label=label, color=color)

    ax_cpu.set_ylabel("CPU %")
    ax_cpu.grid(axis="x", linestyle="--", alpha=0.35)
    lines1, labels1 = ax_cpu.get_legend_handles_labels()
    if lines1:
        ax_cpu.legend(lines1, labels1, loc="upper left", ncol=3)

    lag_specs = [
        ("shadow_apply_backlog", "Shadow apply backlog", "#1f77b4"),
        ("shadow_durable_lag", "Shadow durable lag", "#ff7f0e"),
        ("anchor_lag", "Anchor lag", "#2ca02c"),
        ("update_history_len", "Update history len", "#d62728"),
    ]
    lag_ax2 = ax_lag.twinx()
    for key, label, color in lag_specs:
        xs, ys = _extract_series(records, "train_progress", key, panel="gpu_train")
        if not xs:
            continue
        axis = lag_ax2 if key == "update_history_len" else ax_lag
        axis.plot([_ns_to_s(x, origin_ns) for x in xs], ys, label=label, color=color)

    ax_lag.set_ylabel("Lag / Backlog")
    lag_ax2.set_ylabel("Log Length")
    ax_lag.grid(axis="x", linestyle="--", alpha=0.35)
    lines1, labels1 = ax_lag.get_legend_handles_labels()
    lines2, labels2 = lag_ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax_lag.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=2)
    ax_lag.set_xlabel("Wall Time Since Trace Start (s)")

    fig_mem.tight_layout()
    fig_gpu.tight_layout()
    fig_cpu.tight_layout()
    fig_lag.tight_layout()
    return (
        {
            "memory": fig_mem,
            "gpu": fig_gpu,
            "cpu": fig_cpu,
            "lag": fig_lag,
        },
        {
            "memory": ax_mem,
            "gpu": ax_gpu,
            "cpu": ax_cpu,
            "lag": ax_lag,
            "lag_aux": lag_ax2,
        },
    )


def plot_loss(trace_or_records, *, figsize=(18, 4)):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot loss figures") from exc

    if isinstance(trace_or_records, dict):
        records = trace_or_records["records"]
    else:
        records = trace_or_records

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    loss_xs, loss_ys = _extract_step_series(records, "train_scalar", "loss", panel="gpu_train")
    if loss_xs:
        ax.plot(loss_xs, loss_ys, color="#1f77b4", label="Loss")
    lr_xs, lr_ys = _extract_step_series(records, "train_scalar", "learning_rate", panel="gpu_train")
    if lr_xs:
        ax_lr = ax.twinx()
        ax_lr.plot(lr_xs, lr_ys, color="#ff7f0e", linestyle="--", label="Learning rate")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_lr.get_legend_handles_labels()
        if lines1 or lines2:
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", ncol=2)
        ax_lr.set_ylabel("Learning Rate")
    else:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig, ax


def plot_interactive(trace_or_records, spans=None, instants=None, *, height=1000, width=1500):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("plotly is required to plot interactive figures") from exc

    if isinstance(trace_or_records, dict):
        records = trace_or_records["records"]
        spans = trace_or_records["spans"]
        instants = trace_or_records["instants"]
    else:
        records = trace_or_records
        if spans is None or instants is None:
            spans, instants = build_spans(records)

    origin_ns = _origin_ns(records, spans, instants)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("GPU / Train Timeline", "CPU / Shadow Timeline", "Resources / Lag"),
    )

    for row, panel in ((1, "gpu_train"), (2, "cpu_shadow")):
        panel_spans = [item for item in spans if item["panel"] == panel]
        row_keys = sorted({item["event"] for item in panel_spans})
        y_positions = {key: idx for idx, key in enumerate(row_keys)}
        for item in panel_spans:
            start_s = _ns_to_s(item["start_ns"], origin_ns)
            end_s = _ns_to_s(item["end_ns"], origin_ns)
            y = y_positions[item["event"]]
            fig.add_trace(
                go.Scatter(
                    x=[start_s, end_s],
                    y=[y, y],
                    mode="lines",
                    line={"width": 12},
                    name=item["event"],
                    hovertemplate=(
                        f"{item['event']}<br>step={item.get('step')}"
                        f"<br>start={start_s:.3f}s<br>end={end_s:.3f}s"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(y_positions.values()),
            ticktext=list(y_positions.keys()),
            row=row,
            col=1,
        )

    series_specs = [
        ("gpu_train", "resource_sample", ["gpu0_alloc_mb", "gpu_alloc_mb"], "GPU alloc"),
        ("gpu_train", "resource_sample", ["gpu0_reserved_mb", "gpu_reserved_mb"], "GPU reserved"),
        ("gpu_train", "resource_sample", ["gpu0_peak_mb", "gpu_peak_mb"], "GPU peak"),
        ("gpu_train", "resource_sample", "train_rss_mb", "Train RSS"),
        ("cpu_shadow", "resource_sample", "shadow_rss_mb", "Shadow RSS"),
        (None, "resource_sample", "zo_shm_used_mb", "DRAM"),
        ("gpu_train", "train_progress", "shadow_apply_backlog", "Shadow apply backlog"),
        ("gpu_train", "train_progress", "shadow_durable_lag", "Shadow durable lag"),
        ("gpu_train", "train_progress", "anchor_lag", "Anchor lag"),
        ("gpu_train", "train_progress", "update_history_len", "Update history len"),
    ]
    for panel, event_name, key, label in series_specs:
        if isinstance(key, list):
            xs, ys, _ = _extract_series_any(records, event_name, key, panel=panel)
        else:
            xs, ys = _extract_series(records, event_name, key, panel=panel)
        if not xs:
            continue
        if event_name == "resource_sample" and (
            (isinstance(key, list) and any(k in _GB_KEYS for k in key)) or key in _GB_KEYS
        ):
            ys = [y / 1024.0 for y in ys]
        fig.add_trace(
            go.Scatter(
                x=[_ns_to_s(x, origin_ns) for x in xs],
                y=ys,
                mode="lines",
                name=label,
            ),
            row=3,
            col=1,
        )

    fig.update_xaxes(title_text="Wall Time Since Trace Start (s)", row=3, col=1)
    fig.update_layout(height=height, width=width, legend={"orientation": "h"})
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize NonStopZO2 structured trace JSONL output.")
    parser.add_argument("trace_jsonl", help="Path to zo_trace.jsonl")
    parser.add_argument("--html", action="store_true", help="Also show an interactive plotly figure")
    args = parser.parse_args()

    trace = load_trace(args.trace_jsonl)
    summary = summarize_trace(trace)
    print_summary(summary)
    timeline_fig, _ = plot_timeline(trace)
    resource_figs, _ = plot_resources(trace)
    loss_fig, _ = plot_loss(trace)
    timeline_fig.show()
    for fig in resource_figs.values():
        fig.show()
    loss_fig.show()
    if args.html:
        plot_interactive(trace).show()


if __name__ == "__main__":
    main()
