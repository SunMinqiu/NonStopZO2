"""
zo_checkpoint_model.py
======================
ZO Checkpoint 成本模型：计算 + 画图
被 cell 脚本 import 调用，不要直接运行。

数据结构：统一的 experiment_groups 列表，每组包含:
  - name:             组名（如 "Log-based (GPU)"）
  - avg_ckpt_time:    每步 checkpoint 写入时间 → 自动用作 tl
  - total_ckpts:      checkpoint 总数（用于 strategy comparison）
  - e2e_time_seconds: 端到端训练时间
  - replay_data:      [{replay_steps, replay_time, full_resume_time}, ...]
                      至少 2 个点，用斜率拟合 U；每个点都有对应的 full_resume_time
  - color / ls:       可选，画图样式
  - ckpt_label:       可选，strategy comparison 去重用的标签

恢复公式:
  对每个数据点: cold_start_i = full_resume_time_i - U * replay_steps_i
  cold_start = mean(cold_start_i)
  （cold_start 包含模型加载、初始化、replay 冷启动等所有非回放开销）
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
#  从统一数据计算参数
# ============================================================

def compute_replay_params(experiment_groups):
    """
    从统一的 experiment_groups 计算回放参数。
    tl = avg_ckpt_time（自动获取，不需要手动填）。

    2+ 个点：用斜率拟合 U（消除 replay 冷启动），
             cold_start_i = full_resume_time_i - U * steps_i，取平均
             （cold_start 包含 replay 冷启动）
    1  个点：无法分离 replay 冷启动，U = replay_time / steps（含 replay 冷启动），
             cold_start = full_resume_time - replay_time（不含 replay 冷启动）

    Returns
    -------
    dict: {name: {"U", "cold_start", "tl", "full_resume_time", "name", ...}}
          full_resume_time 取 replay_steps 最大的那个点（代表实际恢复场景）
    """
    results = {}
    for g in experiment_groups:
        name = g["name"]
        tl = g.get("avg_ckpt_time", 0)
        replay_data = g.get("replay_data", [])

        # 过滤 replay_steps=0
        valid = [d for d in replay_data if d["replay_steps"] > 0]
        if not valid:
            continue

        steps = np.array([d["replay_steps"] for d in valid])
        times = np.array([d["replay_time"] for d in valid])

        if len(valid) < 2:
            # 只有 1 个点：无法消除 replay 冷启动
            # U 包含 replay 冷启动，cold_start 只算非 replay 部分
            U = times[0] / steps[0]
            frt = valid[0].get("full_resume_time")
            cold_start = max(0, frt - times[0]) if frt is not None else 0
        else:
            # 2+ 个点：斜率 = 真实每步 replay 时间（消除 replay 冷启动）
            U = np.polyfit(steps, times, 1)[0]
            # 每个点算 cold_start_i = full_resume_time_i - U * steps_i，取平均
            cs_list = []
            for d in valid:
                frt = d.get("full_resume_time")
                if frt is not None:
                    cs_list.append(frt - U * d["replay_steps"])
            cold_start = max(0, float(np.mean(cs_list))) if cs_list else 0

        # 取 replay_steps 最大的点的 full_resume_time 作为代表值
        max_point = max(valid, key=lambda d: d["replay_steps"])
        full_resume_time = max_point.get("full_resume_time")

        entry = {"U": U, "cold_start": cold_start, "tl": tl, "name": name,
                 "full_resume_time": full_resume_time}
        # 透传 color / ls
        for k in ("color", "ls"):
            if k in g:
                entry[k] = g[k]
        results[name] = entry

    return results


def get_unique_checkpoint_modes(experiment_groups):
    """
    从 experiment_groups 去重，用于 strategy comparison 柱状图。
    按 ckpt_label（若有）或 name 去重。
    """
    seen = set()
    result = []
    for g in experiment_groups:
        label = g.get("ckpt_label", g["name"])
        if label not in seen:
            seen.add(label)
            entry = dict(g)
            entry["name"] = label
            result.append(entry)
    return result


# ============================================================
#  打印计算结果
# ============================================================

def print_summary(C, ts, replay_scenarios,
                  experiment_groups,
                  total_steps, mtbf_table_values):
    """打印完整计算结果表格（tl 从每个 scenario 自动获取）"""

    print("=" * 80)
    print("ZO Checkpoint Cost Model — Summary")
    print("=" * 80)

    print(f"\n[Input Parameters]")
    print(f"  C  (Full checkpoint write time) = {C:.3f}s")
    print(f"  ts (pure training step time)    = {ts:.4f}s")
    for s in replay_scenarios:
        tl = s.get('tl', 0)
        t = ts + tl
        frt = s.get('full_resume_time')
        frt_str = f"{frt:.3f}s" if frt is not None else "N/A"
        print(f"  {s['name']:<34} U={s['U']:.4f}s/step  full_resume={frt_str}  cold_start={s['cold_start']:.3f}s  tl={tl:.4f}s  t={t:.4f}s")

    # ---- K* 表格 ----
    print(f"\n[Optimal K*]  K* = (1/t) * ( sqrt( C*(2*M*ts/U - C) ) - C )")
    for s in replay_scenarios:
        tl = s.get('tl', 0)
        print(f"\n  --- {s['name']} (tl={tl:.4f}s) ---")
        hdr = f"  {'MTBF(h)':<9} {'K*':<10} {'OH%':<10} {'Recovery(s)':<15}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for mh in mtbf_table_values:
            M = mh * 3600
            kc = K_star(C, s['U'], M, ts, tl)
            oh = overhead(kc, C, s['U'], M, ts, tl, s['cold_start']) * 100
            rec = recovery_time(kc, C, s['U'], ts, tl, s['cold_start'])
            print(f"  {mh:<9} {kc:<10.0f} {oh:<10.1f} {rec:<15.1f}")

    # ---- Checkpoint 模式写入开销（自动去重）----
    if experiment_groups:
        unique = get_unique_checkpoint_modes(experiment_groups)
        print(f"\n[Checkpoint Write Overhead] ({total_steps} steps)")
        for m in unique:
            total = m["avg_ckpt_time"] * m["total_ckpts"]
            print(f"  {m['name']:<28} {m['total_ckpts']:>5} x {m['avg_ckpt_time']:.4f}s = {total:.1f}s")


# ============================================================
#  画图函数
# ============================================================

_DEFAULT_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6',
                   '#f39c12', '#1abc9c', '#e67e22', '#34495e']
_DEFAULT_LS = ['-', '--', '-.', ':']


def plot_cost_model(C, ts, replay_scenarios,
                    mtbf_range_hours=(0.5, 24), mtbf_demo_hours=4):
    """
    三合一成本模型图（各 scenario 使用自己的 tl）:
      (a) K* vs MTBF
      (b) Overhead% vs MTBF
      (c) overhead(K) curve at a fixed MTBF
    """
    mtbf_h = np.linspace(mtbf_range_hours[0], mtbf_range_hours[1], 500)
    mtbf_s = mtbf_h * 3600

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, s in enumerate(replay_scenarios):
        U_s = s['U']
        cs = s['cold_start']
        tl = s.get('tl', 0)
        color = s.get('color', _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)])
        ls = s.get('ls', _DEFAULT_LS[i % len(_DEFAULT_LS)])
        label = s['name']

        K_opt = K_star(C, U_s, mtbf_s, ts, tl)

        # (a) K* vs MTBF
        axes[0].plot(mtbf_h, K_opt, color=color, ls=ls, lw=2.5, label=label)

        # (b) Overhead %
        oh = np.array([overhead(k, C, U_s, m, ts, tl, cs)
                        for k, m in zip(K_opt, mtbf_s)]) * 100
        axes[1].plot(mtbf_h, oh, color=color, ls=ls, lw=2.5, label=label)

        # (c) overhead(K) at fixed MTBF
        M_demo = mtbf_demo_hours * 3600
        K_range = np.linspace(50, 8000, 1000)
        ovh = np.array([overhead(K, C, U_s, M_demo, ts, tl, cs) for K in K_range])
        Kopt = K_star(C, U_s, M_demo, ts, tl)
        axes[2].plot(K_range, ovh, color=color, ls=ls, lw=2, label=label)
        axes[2].axvline(x=Kopt, color=color, ls=':', alpha=0.4)
        axes[2].plot(Kopt, overhead(Kopt, C, U_s, M_demo, ts, tl, cs), 'o',
                     color=color, ms=7)
        axes[2].annotate(f'K*={Kopt:.0f}', xy=(Kopt, overhead(Kopt, C, U_s, M_demo, ts, tl, cs)),
                         fontsize=12, color=color, ha='left', va='bottom',
                         xytext=(Kopt + 100, overhead(Kopt, C, U_s, M_demo, ts, tl, cs) + 0.002))

    # titles & labels
    ax = axes[0]
    ax.set_xlabel('MTBF (hours)')
    ax.set_ylabel('Optimal K* (steps)')
    ax.set_title('(a) Optimal Full Checkpoint Interval K* vs MTBF', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.set_xlabel('MTBF (hours)')
    ax.set_ylabel('Overhead (% of step time)')
    ax.set_title('(b) Checkpoint Overhead at Optimal K*', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    ax = axes[2]
    ax.set_xlabel('Full Checkpoint Interval K (steps)')
    ax.set_ylabel('Per-unit-time Overhead')
    ax.set_title(f'(c) Overhead vs K  (MTBF = {mtbf_demo_hours}h)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout(pad=2.0)
    return fig


def plot_strategy_comparison(experiment_groups, total_steps):
    """各 checkpoint 模式的单次耗时 + E2E 时间柱状图（自动按 ckpt_label 去重）"""
    modes = get_unique_checkpoint_modes(experiment_groups)
    for i, m in enumerate(modes):
        if "color" not in m:
            m["color"] = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]

    names = [m["name"] for m in modes]
    colors = [m["color"] for m in modes]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Per-checkpoint time
    ax = axes[0]
    per_ckpt = [m["avg_ckpt_time"] for m in modes]
    bars = ax.bar(names, per_ckpt, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, per_ckpt):
        label = f'{val:.3f}s' if val < 1 else f'{val:.2f}s'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(per_ckpt) * 0.02,
                label, ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg Time per Checkpoint (seconds)')
    ax.set_title('Per-Checkpoint Write Time', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # E2E time
    ax = axes[1]
    e2e = [m["e2e_time_seconds"] for m in modes]
    bars = ax.bar(names, e2e, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, e2e):
        m_, s_ = divmod(int(val), 60)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(e2e) * 0.02,
                f'{m_}:{s_:02d}', ha='center', va='bottom',
                fontsize=13, fontweight='bold')
    ax.set_ylabel('E2E Training Time (seconds)')
    ax.set_title(f'End-to-End Training Time ({total_steps} steps)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(pad=2.0)
    return fig


def plot_replay_speed(experiment_groups):
    """
    各组回放参数对比（按 name 分组）：
    - 左图：每步回放时间 U
    - 右图：完整恢复时间 full_resume_time
    """
    params = compute_replay_params(experiment_groups)
    if not params:
        print("No replay data to plot.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    names = list(params.keys())
    colors = [params[n].get('color', _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)])
              for i, n in enumerate(names)]

    # 左图: 每步回放时间 U
    ax = axes[0]
    U_vals = [params[n]["U"] for n in names]
    bars = ax.bar(names, U_vals, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, U_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(U_vals) * 0.02,
                f'{val:.4f}s', ha='center', va='bottom',
                fontsize=13, fontweight='bold')
    ax.set_ylabel('Per-step Replay Time U (seconds)')
    ax.set_title('Per-Step Replay Time U', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

    # 右图: 完整恢复时间
    ax = axes[1]
    frt_vals = [params[n].get("full_resume_time", 0) or 0 for n in names]
    bars = ax.bar(names, frt_vals, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, frt_vals):
        y_offset = max(frt_vals) * 0.02 if max(frt_vals) > 0 else 0.1
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_offset,
                f'{val:.3f}s', ha='center', va='bottom',
                fontsize=13, fontweight='bold')
    ax.set_ylabel('Full Resume Time (seconds)')
    ax.set_title('Full Resume Time', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    return fig
