"""
zo_checkpoint_model.py
======================
ZO Checkpoint 成本模型：计算 + 画图
被 cell 脚本 import 调用，不要直接运行。
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


# ============================================================
#  核心公式
# ============================================================

def K_star(C, U, M, ts):
    """
    最优 Full checkpoint 间隔（考虑 checkpoint 本身也占时间、也可能崩溃）
    K* = -C/ts + sqrt(2CM/(U*ts) + C^2/ts^2)
    """
    inside = 2 * C * M / (U * ts) + C**2 / ts**2
    inside = np.maximum(inside, 0)
    return -C / ts + np.sqrt(inside)


def overhead(K, C, U, M, ts, cold_start=0):
    """per-unit-time overhead（含 cold_start）"""
    T_cycle = K * ts + C
    ckpt_cost = C / T_cycle
    E_replay = cold_start + U * K * (K * ts / 2 + C) / T_cycle
    return ckpt_cost + E_replay / M


def recovery_time(K, C, U, ts, load_base, cold_start=0):
    """平均恢复时间（按崩溃位置加权）"""
    T_cycle = K * ts + C
    E_replay = U * K * (K * ts / 2 + C) / T_cycle
    return load_base + E_replay + cold_start


# ============================================================
#  打印计算结果
# ============================================================

def print_summary(C, ts, load_base, replay_scenarios,
                  checkpoint_modes, resume_measurements,
                  total_steps, mtbf_table_values):
    """打印完整计算结果表格"""

    print("=" * 80)
    print("ZO Checkpoint Cost Model — Summary")
    print("=" * 80)

    print(f"\n[Input Parameters]")
    print(f"  C  (Full checkpoint write time) = {C:.3f}s")
    print(f"  ts (training step time)         = {ts:.4f}s")
    print(f"  load_base                       = {load_base:.3f}s")
    for s in replay_scenarios:
        print(f"  {s['name']:<34} U={s['U']:.4f}s/step  cold_start={s['cold_start']:.3f}s")

    # ---- K* 表格 ----
    print(f"\n[Optimal K*]  K* = -C/ts + sqrt(2CM/(U*ts) + C²/ts²)")
    # header
    name_w = max(len(s['name']) for s in replay_scenarios) + 2
    for s in replay_scenarios:
        print(f"\n  --- {s['name']} ---")
        hdr = f"  {'MTBF(h)':<9} {'K*':<10} {'OH%':<10} {'Recovery(s)':<15}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for mh in mtbf_table_values:
            M = mh * 3600
            kc = K_star(C, s['U'], M, ts)
            oh = overhead(kc, C, s['U'], M, ts, s['cold_start']) / ts * 100
            rec = recovery_time(kc, C, s['U'], ts, load_base, s['cold_start'])
            print(f"  {mh:<9} {kc:<10.0f} {oh:<10.1f} {rec:<15.1f}")

    # ---- Checkpoint 模式写入开销 ----
    if checkpoint_modes:
        print(f"\n[Checkpoint Write Overhead] ({total_steps} steps)")
        for m in checkpoint_modes:
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
    三合一成本模型图，每个 replay scenario 一条线:
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
        color = s.get('color', _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)])
        ls = s.get('ls', _DEFAULT_LS[i % len(_DEFAULT_LS)])
        label = s['name']

        K_opt = K_star(C, U_s, mtbf_s, ts)

        # (a) K* vs MTBF
        axes[0].plot(mtbf_h, K_opt, color=color, ls=ls, lw=2.5, label=label)

        # (b) Overhead %
        oh = np.array([overhead(k, C, U_s, m, ts, cs)
                        for k, m in zip(K_opt, mtbf_s)]) / ts * 100
        axes[1].plot(mtbf_h, oh, color=color, ls=ls, lw=2.5, label=label)

        # (c) overhead(K) at fixed MTBF
        M_demo = mtbf_demo_hours * 3600
        K_range = np.linspace(50, 8000, 1000)
        ovh = np.array([overhead(K, C, U_s, M_demo, ts, cs) for K in K_range])
        Kopt = K_star(C, U_s, M_demo, ts)
        axes[2].plot(K_range, ovh, color=color, ls=ls, lw=2, label=label)
        axes[2].axvline(x=Kopt, color=color, ls=':', alpha=0.4)
        axes[2].plot(Kopt, overhead(Kopt, C, U_s, M_demo, ts, cs), 'o',
                     color=color, ms=7)
        axes[2].annotate(f'K*={Kopt:.0f}', xy=(Kopt, overhead(Kopt, C, U_s, M_demo, ts, cs)),
                         fontsize=9, color=color, ha='left', va='bottom',
                         xytext=(Kopt + 100, overhead(Kopt, C, U_s, M_demo, ts, cs) + 0.002))

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


def plot_strategy_comparison(checkpoint_modes, total_steps):
    """
    各 checkpoint 模式的单次耗时 + E2E 时间柱状图
    """
    for i, m in enumerate(checkpoint_modes):
        if "color" not in m:
            m["color"] = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]

    names = [m["name"] for m in checkpoint_modes]
    colors = [m["color"] for m in checkpoint_modes]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Per-checkpoint time
    ax = axes[0]
    per_ckpt = [m["avg_ckpt_time"] for m in checkpoint_modes]
    bars = ax.bar(names, per_ckpt, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, per_ckpt):
        label = f'{val:.3f}s' if val < 1 else f'{val:.2f}s'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(per_ckpt) * 0.02,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Avg Time per Checkpoint (seconds)')
    ax.set_title('Per-Checkpoint Write Time', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # E2E time
    ax = axes[1]
    e2e = [m["e2e_time_seconds"] for m in checkpoint_modes]
    bars = ax.bar(names, e2e, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, e2e):
        m_, s_ = divmod(int(val), 60)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(e2e) * 0.02,
                f'{m_}:{s_:02d}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('E2E Training Time (seconds)')
    ax.set_title(f'End-to-End Training Time ({total_steps} steps)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(pad=2.0)
    return fig


def plot_replay_speed(resume_measurements):
    """
    GPU vs CPU 回放速度对比柱状图
    """
    data = [r for r in resume_measurements if r["replay_steps"] > 0]
    if not data:
        print("No replay data to plot.")
        return None

    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(12, 5.5))

    labels = [f"{r['name']}\n({r['replay_steps']} steps)" for r in data]
    per_step = [r["replay_time"] / r["replay_steps"] for r in data]
    colors = ['#3498db' if r["device"] == "GPU" else '#e74c3c' for r in data]

    bars = ax.bar(labels, per_step, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, per_step):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(per_step) * 0.02,
                f'{val:.3f}s', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    legend_elements = [Patch(facecolor='#3498db', alpha=0.85, label='GPU'),
                       Patch(facecolor='#e74c3c', alpha=0.85, label='CPU')]
    ax.legend(handles=legend_elements, fontsize=11)
    ax.set_ylabel('Per-step Replay Time (seconds)')
    ax.set_title('Replay Speed: GPU vs CPU', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig
