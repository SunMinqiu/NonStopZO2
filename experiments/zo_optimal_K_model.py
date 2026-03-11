"""
zo_optimal_K_model.py
======================
ZO Checkpoint 成本模型：计算 + 画图
被 cell 脚本 import 调用，不要直接运行。

数据结构（FS 分组）：
  EXPERIMENTS = {
    "Model / Dataset": {
      "ts": 每步训练时间,
      "total_steps": 总训练步数,
      "Info": [
        {
          "FS": 文件系统名,
          "avg_full_ckpt_time": full ckpt 平均写入时间,
          "avg_log_ckpt_time": log ckpt 平均写入时间,
          "e2e_train": [
            {"checkpoint_type": "full"/"log", "checkpoint numbers": N, "e2e_time_seconds": T},
          ],
          "full checkpoint resume time": 标量 (可选),
          "log checkpoint resume time": dict 或 list[dict],
            每个 dict: {"type": "simulation"/"no-simulation", "data": [
              {"replay_steps": K, "replay_time": T, "total_resume_time": T}, ...
            ]}
        },
      ],
    },
  }

恢复公式:
  U = polyfit(replay_steps, replay_time) 的斜率（包含 steps=0 数据点）
  cold_start_i = total_resume_time_i - U * replay_steps_i
  cold_start = mean(cold_start_i)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ---------- 尝试设置中文字体 ----------
for font_name in ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
                   'PingFang SC', 'Microsoft YaHei']:
    try:
        matplotlib.font_manager.findfont(font_name, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

# ---------- 全局字号放大 ----------
plt.rcParams.update({
    'font.size':        14,
    'axes.titlesize':   16,
    'axes.labelsize':   14,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  12,
})


# ============================================================
#  核心公式
# ============================================================

def K_star(C, U, M, ts, tl=0):
    """最优 Full checkpoint 间隔。K* = (1/t) * ( sqrt( C*(2*M*ts/U - C) ) - C )"""
    t = ts + tl
    inside = C * (2 * M * ts / U - C)
    inside = np.maximum(inside, 0)
    return np.maximum((np.sqrt(inside) - C) / t, 0)


def overhead(K, C, U, M, ts, tl=0, cold_start=0):
    """per-unit-time overhead（无量纲比例）。"""
    t = ts + tl
    T_cycle = K * t + C
    ckpt_cost = (C + K * tl) / T_cycle
    E_replay = U * K * (K * t / 2 + C) / T_cycle
    return ckpt_cost + (cold_start + E_replay) / M


def recovery_time(K, C, U, ts, tl=0, cold_start=0):
    """平均恢复时间 = cold_start + E[回放时间]"""
    t = ts + tl
    T_cycle = K * t + C
    E_replay = U * K * (K * t / 2 + C) / T_cycle
    return cold_start + E_replay


# ============================================================
#  数据工具函数
# ============================================================

_DEFAULT_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6',
                   '#f39c12', '#1abc9c', '#e67e22', '#34495e']
_DEFAULT_LS = ['-', '--', '-.', ':']
_FULL_COLOR = '#95a5a6'


def flatten_experiments(experiments):
    """
    将嵌套的 EXPERIMENTS 展平为 flat record 列表。
    每条 record 对应一个 (model, dataset, FS) 组合。
    """
    records = []
    for key, cfg in experiments.items():
        parts = key.split(" / ")
        model = parts[0].strip()
        dataset = parts[1].strip() if len(parts) > 1 else ""
        ts = cfg["ts"]
        total_steps = cfg["total_steps"]
        for info in cfg.get("Info", []):
            rec = {
                "model": model,
                "dataset": dataset,
                "FS": info["FS"],
                "ts": ts,
                "total_steps": total_steps,
                "avg_full_ckpt_time": info.get("avg_full_ckpt_time"),
                "avg_log_ckpt_time": info.get("avg_log_ckpt_time"),
                "e2e_train": info.get("e2e_train", []),
                "full_resume_time": info.get("full checkpoint resume time"),
                "log_resume_time_raw": info.get("log checkpoint resume time"),
            }
            records.append(rec)
    return records


def compute_log_resume_params(log_resume_raw):
    """
    从 "log checkpoint resume time" 原始数据计算回放参数。

    用 ALL 数据点（含 replay_steps=0）做 polyfit 拟合 U：
      2+ 个点: U = polyfit slope
      1 个点 (steps>0): U = replay_time / steps
      1 个点 (steps=0): U = 0

    cold_start = mean(total_resume_time_i - U * replay_steps_i)

    Returns: [{"type", "U", "cold_start", "data"}, ...]
    """
    if log_resume_raw is None:
        return []

    # 统一为 list
    if isinstance(log_resume_raw, dict):
        entries = [log_resume_raw]
    else:
        entries = list(log_resume_raw)

    results = []
    for entry in entries:
        sim_type = entry["type"]
        data = entry["data"]
        if not data:
            continue

        steps = np.array([d["replay_steps"] for d in data])
        times = np.array([d["replay_time"] for d in data])

        if len(data) >= 2:
            U = np.polyfit(steps, times, 1)[0]
        elif data[0]["replay_steps"] > 0:
            U = times[0] / steps[0]
        else:
            U = 0.0

        U = max(0.0, float(U))

        # cold_start from total_resume_time
        cs_list = []
        for d in data:
            trt = d.get("total_resume_time")
            if trt is not None:
                cs_list.append(trt - U * d["replay_steps"])
        cold_start = max(0.0, float(np.mean(cs_list))) if cs_list else 0.0

        results.append({
            "type": sim_type,
            "U": U,
            "cold_start": cold_start,
            "data": data,
        })

    return results


def _apply_filters(records, config, filter_keys):
    """按 config 中指定的维度过滤 records。None 值表示不过滤。"""
    filtered = list(records)
    for key in filter_keys:
        val = config.get(key)
        if val is not None:
            filtered = [r for r in filtered if r.get(key) == val]
    return filtered


# ============================================================
#  Plot 1: Cost Model
# ============================================================

def plot_cost_model(experiments, config, mtbf_config):
    """
    成本模型图（3 子图）。

    config 维度: model, dataset, FS, simulation
      设为具体值 → 固定（过滤）; None → 每个值画一条线

    Parameters
    ----------
    experiments : dict  — EXPERIMENTS 数据
    config : dict       — {"model", "dataset", "FS", "simulation"}
    mtbf_config : dict  — {"range_hours", "demo_hours", "table_values"}
    """
    records = flatten_experiments(experiments)
    records = _apply_filters(records, config, ["model", "dataset", "FS"])

    # 展开 simulation 维度
    scenarios = []
    for rec in records:
        params_list = compute_log_resume_params(rec["log_resume_time_raw"])
        for params in params_list:
            scenarios.append({
                "model": rec["model"],
                "dataset": rec["dataset"],
                "FS": rec["FS"],
                "simulation": params["type"],
                "C": rec["avg_full_ckpt_time"],
                "ts": rec["ts"],
                "tl": rec["avg_log_ckpt_time"] or 0,
                "U": params["U"],
                "cold_start": params["cold_start"],
            })

    # 过滤 simulation
    sim_val = config.get("simulation")
    if sim_val is not None:
        scenarios = [s for s in scenarios if s["simulation"] == sim_val]

    if not scenarios:
        print("plot_cost_model: No matching scenarios found.")
        return None

    # 生成 label：用未固定的维度
    unfixed = [d for d in ["model", "dataset", "FS", "simulation"]
               if config.get(d) is None]

    mtbf_h = np.linspace(mtbf_config["range_hours"][0],
                         mtbf_config["range_hours"][1], 500)
    mtbf_s = mtbf_h * 3600

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, s in enumerate(scenarios):
        label = " / ".join(str(s[d]) for d in unfixed) if unfixed else \
                f"{s['model']} / {s['dataset']}"
        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        ls = _DEFAULT_LS[i % len(_DEFAULT_LS)]
        C, U, ts, tl, cs = s["C"], s["U"], s["ts"], s["tl"], s["cold_start"]

        K_opt = K_star(C, U, mtbf_s, ts, tl)

        # (a) K* vs MTBF
        axes[0].plot(mtbf_h, K_opt, color=color, ls=ls, lw=2.5, label=label)

        # (b) Overhead %
        oh = np.array([overhead(k, C, U, m, ts, tl, cs)
                       for k, m in zip(K_opt, mtbf_s)]) * 100
        axes[1].plot(mtbf_h, oh, color=color, ls=ls, lw=2.5, label=label)

        # (c) overhead(K) at fixed MTBF
        M_demo = mtbf_config["demo_hours"] * 3600
        K_range = np.linspace(50, 16000, 1000)
        ovh = np.array([overhead(K, C, U, M_demo, ts, tl, cs) for K in K_range])
        Kopt = K_star(C, U, M_demo, ts, tl)
        axes[2].plot(K_range, ovh, color=color, ls=ls, lw=2, label=label)
        axes[2].axvline(x=Kopt, color=color, ls=':', alpha=0.4)
        axes[2].plot(Kopt, overhead(Kopt, C, U, M_demo, ts, tl, cs), 'o',
                     color=color, ms=7)
        axes[2].annotate(
            f'K*={Kopt:.0f}',
            xy=(Kopt, overhead(Kopt, C, U, M_demo, ts, tl, cs)),
            fontsize=12, color=color, ha='left', va='bottom',
            xytext=(Kopt + 100,
                    overhead(Kopt, C, U, M_demo, ts, tl, cs) + 0.002))

    # 标题
    fixed_parts = [f"{d}={config[d]}" for d in ["model", "dataset", "FS", "simulation"]
                   if config.get(d) is not None]
    title_suffix = "  |  " + ", ".join(fixed_parts) if fixed_parts else ""

    axes[0].set_xlabel('MTBF (hours)')
    axes[0].set_ylabel('Optimal K* (steps)')
    axes[0].set_title('(a) K* vs MTBF' + title_suffix, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('MTBF (hours)')
    axes[1].set_ylabel('Overhead (%)')
    axes[1].set_title('(b) Overhead at Optimal K*', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    axes[2].set_xlabel('K (steps)')
    axes[2].set_ylabel('Per-unit-time Overhead')
    axes[2].set_title(
        f'(c) Overhead vs K  (MTBF={mtbf_config["demo_hours"]}h)',
        fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(bottom=0)

    plt.tight_layout(pad=2.0)
    return fig


# ============================================================
#  Plot 2: E2E Comparison (Full vs Log)
# ============================================================

def plot_e2e_comparison(experiments, config):
    """
    E2E 训练时间对比（Full vs Log），归一化为每步 ckpt 1 次。

    config:
      group_by : "model" / "dataset" / "FS" — x 轴分组维度
      model, dataset, FS : 过滤值 (None = 不过滤)

    E2E 归一化公式:
      pure_train = measured_e2e - avg_ckpt_time × num_ckpts
      projected_e2e = pure_train + avg_ckpt_time × total_steps
    """
    records = flatten_experiments(experiments)
    group_by = config["group_by"]
    filter_keys = [d for d in ["model", "dataset", "FS"] if d != group_by]
    records = _apply_filters(records, config, filter_keys)

    if not records:
        print("plot_e2e_comparison: No matching records found.")
        return None

    # 按 group_by 分组
    groups = {}
    for rec in records:
        key = rec[group_by]
        if key not in groups:
            groups[key] = rec

    group_names = list(groups.keys())
    n = len(group_names)

    # 计算数据
    full_ckpt_times = []
    log_ckpt_times = []
    full_e2e = []
    log_e2e = []

    for name in group_names:
        rec = groups[name]
        total_steps = rec["total_steps"]
        avg_full = rec["avg_full_ckpt_time"] or 0
        avg_log = rec["avg_log_ckpt_time"] or 0
        full_ckpt_times.append(avg_full)
        log_ckpt_times.append(avg_log)

        # E2E
        full_e2e_val = 0
        log_e2e_val = 0
        for e in rec["e2e_train"]:
            measured = e["e2e_time_seconds"]
            num_ckpts = e["checkpoint numbers"]
            if e["checkpoint_type"] == "full":
                pure_train = measured - avg_full * num_ckpts
                full_e2e_val = pure_train + avg_full * total_steps
            else:
                pure_train = measured - avg_log * num_ckpts
                log_e2e_val = pure_train + avg_log * total_steps
        full_e2e.append(full_e2e_val)
        log_e2e.append(log_e2e_val)

    fig, axes = plt.subplots(1, 2, figsize=(max(13, n * 4), 6))
    x = np.arange(n)
    width = 0.35

    # (a) Per-checkpoint write time
    ax = axes[0]
    bars_full = ax.bar(x - width / 2, full_ckpt_times, width,
                       label='Full Checkpoint', color=_FULL_COLOR,
                       alpha=0.85, edgecolor='black', linewidth=0.5,
                       hatch='///')
    bars_log = ax.bar(x + width / 2, log_ckpt_times, width,
                      label='Log Checkpoint', color=_DEFAULT_COLORS[1],
                      alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars_full, full_ckpt_times):
        label_str = f'{val:.3f}s' if val < 1 else f'{val:.2f}s'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(full_ckpt_times) * 0.02,
                label_str, ha='center', va='bottom', fontsize=11,
                fontweight='bold')
    for bar, val in zip(bars_log, log_ckpt_times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(full_ckpt_times) * 0.02,
                f'{val:.4f}s', ha='center', va='bottom', fontsize=11,
                fontweight='bold')
    ax.set_ylabel('Avg Time per Checkpoint (s)')
    ax.set_title('(a) Per-Checkpoint Write Time', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # (b) Projected E2E time (every-step checkpointing)
    ax = axes[1]
    bars_full = ax.bar(x - width / 2, full_e2e, width,
                       label='Full Checkpoint', color=_FULL_COLOR,
                       alpha=0.85, edgecolor='black', linewidth=0.5,
                       hatch='///')
    bars_log = ax.bar(x + width / 2, log_e2e, width,
                      label='Log Checkpoint', color=_DEFAULT_COLORS[1],
                      alpha=0.85, edgecolor='black', linewidth=0.5)

    all_vals = full_e2e + log_e2e
    max_val = max(all_vals) if all_vals else 1
    for bar, val in zip(bars_full, full_e2e):
        _annotate_time(ax, bar, val, max_val)
    for bar, val in zip(bars_log, log_e2e):
        _annotate_time(ax, bar, val, max_val)

    total_steps = records[0]["total_steps"] if records else 0
    ax.set_ylabel('Projected E2E Time (s)')
    ax.set_title(
        f'(b) E2E Time (every-step ckpt, {total_steps} steps)',
        fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(pad=2.0)
    return fig


def _annotate_time(ax, bar, val, max_val):
    """在柱子顶部标注时间 (mm:ss 或秒)"""
    if val >= 60:
        m_, s_ = divmod(int(val), 60)
        if m_ >= 60:
            h_, m_ = divmod(m_, 60)
            txt = f'{h_}h{m_:02d}m'
        else:
            txt = f'{m_}:{s_:02d}'
    else:
        txt = f'{val:.1f}s'
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.02,
            txt, ha='center', va='bottom', fontsize=11, fontweight='bold')


# ============================================================
#  Plot 3: Resume Time Comparison (Full vs Log, stacked)
# ============================================================

def plot_resume_comparison(experiments, config):
    """
    恢复时间对比: Full resume vs Log resume (stacked bar)。

    config:
      group_by : "model" / "dataset" / "FS" — x 轴分组维度
      model, dataset, FS : 过滤值 (None = 不过滤)
      simulation : "simulation" / "no-simulation" / None — 过滤 log resume 类型

    Log resume bar 分为两段:
      底部: cold_start = total_resume_time - U * replay_steps
      顶部: replay = U * replay_steps
      其中 U 由 ALL 数据点 polyfit 拟合
    """
    records = flatten_experiments(experiments)
    group_by = config["group_by"]
    filter_keys = [d for d in ["model", "dataset", "FS"] if d != group_by]
    records = _apply_filters(records, config, filter_keys)

    if not records:
        print("plot_resume_comparison: No matching records found.")
        return None

    sim_filter = config.get("simulation")

    # 收集所有要画的 bar 数据
    bar_groups = []  # list of {"group_label", "bars": [...]}

    for rec in records:
        group_label = rec[group_by]
        bars = []

        # Full resume bar
        frt = rec.get("full_resume_time")
        if frt is not None:
            bars.append({
                "label": "Full",
                "cold_start": frt,
                "replay": 0.0,
                "total": frt,
            })

        # Log resume bars
        params_list = compute_log_resume_params(rec["log_resume_time_raw"])
        for params in params_list:
            if sim_filter is not None and params["type"] != sim_filter:
                continue
            U = params["U"]
            sim_label = f" ({params['type']})" if len(params_list) > 1 and sim_filter is None else ""
            for d in params["data"]:
                replay_portion = U * d["replay_steps"]
                total = d.get("total_resume_time", replay_portion + params["cold_start"])
                cold_portion = max(0.0, total - replay_portion)
                bars.append({
                    "label": f"Log (replay={d['replay_steps']}){sim_label}",
                    "cold_start": cold_portion,
                    "replay": replay_portion,
                    "total": total,
                })

        if bars:
            bar_groups.append({"group_label": group_label, "bars": bars})

    if not bar_groups:
        print("plot_resume_comparison: No resume data to plot.")
        return None

    # 画图: 每个 group 一组 clustered bars
    # 所有 group 的 bar label 取并集保持对齐
    all_bar_labels = []
    for bg in bar_groups:
        for b in bg["bars"]:
            if b["label"] not in all_bar_labels:
                all_bar_labels.append(b["label"])

    n_groups = len(bar_groups)
    n_bars = len(all_bar_labels)
    fig, ax = plt.subplots(figsize=(max(10, n_groups * n_bars * 1.2), 6))

    group_positions = np.arange(n_groups)
    width = 0.8 / max(n_bars, 1)

    cold_plotted = False
    replay_plotted = False

    for j, bl in enumerate(all_bar_labels):
        cold_vals = []
        replay_vals = []
        for bg in bar_groups:
            # 找到这个 bar label 在这个 group 里的数据
            matched = [b for b in bg["bars"] if b["label"] == bl]
            if matched:
                cold_vals.append(matched[0]["cold_start"])
                replay_vals.append(matched[0]["replay"])
            else:
                cold_vals.append(0)
                replay_vals.append(0)

        positions = group_positions + (j - (n_bars - 1) / 2) * width
        is_full = bl.startswith("Full")

        # Cold start / load portion (bottom)
        cold_label = "Cold Start / Load" if not cold_plotted else None
        bars_cold = ax.bar(
            positions, cold_vals, width,
            color=_FULL_COLOR,
            alpha=0.85, edgecolor='black', linewidth=0.5,
            hatch='///' if is_full else '',
            label=cold_label)
        if not cold_plotted:
            cold_plotted = True

        # Replay portion (top, only for log)
        if any(r > 0 for r in replay_vals):
            replay_label = "Replay" if not replay_plotted else None
            ax.bar(
                positions, replay_vals, width,
                bottom=cold_vals,
                color=_DEFAULT_COLORS[0],
                alpha=0.85, edgecolor='black', linewidth=0.5,
                label=replay_label)
            if not replay_plotted:
                replay_plotted = True

        # 标注总时间
        for pos, cv, rv in zip(positions, cold_vals, replay_vals):
            total = cv + rv
            if total > 0:
                ax.text(pos, total + 0.3, f'{total:.1f}s',
                        ha='center', va='bottom', fontsize=15,
                        fontweight='bold')

    ax.set_ylabel('Resume Time (s)')
    ax.set_title('Resume Time: Full vs Log Checkpoint', fontweight='bold')
    ax.set_xticks(group_positions)
    ax.set_xticklabels([bg["group_label"] for bg in bar_groups],
                       rotation=15, ha='right')

    # 第二层 x-tick: bar labels
    # 用 legend 区分 bar 类型
    # 添加 bar label 文字在底部
    for i, bg in enumerate(bar_groups):
        for j, bl in enumerate(all_bar_labels):
            pos = i + (j - (n_bars - 1) / 2) * width
            ax.text(pos, -0.02 * ax.get_ylim()[1], bl,
                    ha='center', va='top', fontsize=10, rotation=45,
                    color='gray')

    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    plt.tight_layout(pad=2.0)
    return fig
