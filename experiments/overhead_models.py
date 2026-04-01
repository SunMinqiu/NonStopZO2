"""
Overhead Models: Log-Only / Sync Baseline / Ours (No-Anchor Shadow)
====================================================================
Expects `param_sets` and `MTBF_hours_list` to be defined before exec().

Each entry in param_sets must contain:
  name, t_step, t_l, t_d2h, t_persist, t_r, t_rc, t_cp, L_cpu, L_disk
"""

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

MTBF_range = np.linspace(0.5, 24, 200)

# ====================================================================
# Model 1: Log-Only (no anchor, no shadow, replay from step 0)
#   overhead = t_l/t + L_disk/M + t_r/2
# ====================================================================

def overhead_logonly(M, p):
    t = p['t_step'] + p['t_l']
    return p['t_l'] / t + p['L_disk'] / M + p['t_r'] / 2

def overhead_logonly_decomposed(M, p):
    t = p['t_step'] + p['t_l']
    return p['t_l'] / t, p['L_disk'] / M, p['t_r'] / 2

# ====================================================================
# Model 2: Sync Baseline (full checkpoint every K steps, all blocking)
#   overhead(K) = (K*t_l + t_a)/T_cycle + E[T_recover]/M
# ====================================================================

def overhead_sync(K, M, p):
    t = p['t_step'] + p['t_l']
    t_a = p['t_d2h'] + p['t_persist']
    T_cycle = K * t + t_a
    E_train = p['L_disk'] + p['t_r'] * K / 2
    E_ckpt  = p['L_disk'] + p['t_r'] * K
    E_recover = (K * t / T_cycle) * E_train + (t_a / T_cycle) * E_ckpt
    return (K * p['t_l'] + t_a) / T_cycle + E_recover / M

def solve_sync(M, p):
    result = minimize_scalar(lambda K: overhead_sync(K, M, p),
                             bounds=(1, max(2, M)), method='bounded')
    return result.x, result.fun

def overhead_sync_decomposed(K, M, p):
    t = p['t_step'] + p['t_l']
    t_a = p['t_d2h'] + p['t_persist']
    T_cycle = K * t + t_a
    E_train = p['L_disk'] + p['t_r'] * K / 2
    E_ckpt  = p['L_disk'] + p['t_r'] * K
    log_cost    = K * p['t_l'] / T_cycle
    ckpt_block  = t_a / T_cycle
    load_cost   = ((K * t / T_cycle) * p['L_disk'] + (t_a / T_cycle) * p['L_disk']) / M
    replay_cost = ((K * t / T_cycle) * (p['t_r'] * K / 2) + (t_a / T_cycle) * (p['t_r'] * K)) / M
    return log_cost, ckpt_block, load_cost, replay_cost

# ====================================================================
# Model 3: Ours (CPU shadow, no anchor, no training blocking)
#   overhead = t_l/t + (L_cpu + t_r*(1 + N*/2)) / M
#   Requires CPU can catch up: t > t_rc + t_cp/N*
# ====================================================================

def compute_N_star(p):
    t = p['t_step'] + p['t_l']
    gap = t - p['t_rc']
    if gap <= 0:
        return np.inf
    return max(1, int(np.ceil(p['t_cp'] / gap)))

def compute_t_cpu(N, p):
    return p['t_rc'] + p['t_cp'] / N

def ours_viable(p):
    N = compute_N_star(p)
    if N == np.inf:
        return False
    t = p['t_step'] + p['t_l']
    return t > compute_t_cpu(N, p)

def overhead_ours(M, p):
    t = p['t_step'] + p['t_l']
    N = compute_N_star(p)
    if N == np.inf or not ours_viable(p):
        return np.inf
    return p['t_l'] / t + (p['L_cpu'] + p['t_r'] * (1 + N / 2)) / M

def overhead_ours_decomposed(M, p):
    t = p['t_step'] + p['t_l']
    N = compute_N_star(p)
    if N == np.inf or not ours_viable(p):
        return np.inf, np.inf, np.inf
    return p['t_l'] / t, p['L_cpu'] / M, p['t_r'] * (1 + N / 2) / M

# ====================================================================
# Compute & Plot
# ====================================================================
# plot_mode controls what to plot. Set in input cell before exec().
#   "by_model"     — one plot per model, all mechanisms on same axes
#   "by_mechanism"  — one plot per mechanism, all models on same axes
#   "all"           — single plot with everything (default)

if 'plot_mode' not in dir():
    plot_mode = "all"

ALL_MECHANISMS = ['Log-Only', 'Sync', 'Ours']

def _compute_all(p, MTBF_range):
    t = p['t_step'] + p['t_l']
    M_arr = MTBF_range * 3600 / t
    curves = {}
    curves['Log-Only'] = np.array([overhead_logonly(M, p) for M in M_arr])
    curves['Sync']     = np.array([solve_sync(M, p)[1] for M in M_arr])
    oh_ours = np.array([overhead_ours(M, p) for M in M_arr])
    if not np.all(np.isinf(oh_ours)):
        curves['Ours'] = oh_ours
    return curves

all_curves = {}
for p in param_sets:
    all_curves[p['name']] = _compute_all(p, MTBF_range)

print("=" * 80)
print("OVERHEAD MODEL COMPARISON")
print("=" * 80)

STYLES = {
    'Log-Only': {'linestyle': '--', 'linewidth': 1.0},
    'Sync':     {'linestyle': '-',  'linewidth': 1.5},
    'Ours':     {'linestyle': '-',  'linewidth': 2.0},
}

if plot_mode == "by_model":
    for p in param_sets:
        fig, ax = plt.subplots(figsize=(10, 5))
        curves = all_curves[p['name']]
        for mech, oh in curves.items():
            ax.plot(MTBF_range, oh * 100, label=mech, **STYLES[mech])
        ax.set_xlabel('MTBF (hours)')
        ax.set_ylabel('Optimal Overhead (%)')
        ax.set_title(f'{p["name"]} — Overhead vs MTBF')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()

elif plot_mode == "by_mechanism":
    mechs_present = set()
    for curves in all_curves.values():
        mechs_present.update(curves.keys())
    for mech in ALL_MECHANISMS:
        if mech not in mechs_present:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for pname, curves in all_curves.items():
            if mech in curves:
                ax.plot(MTBF_range, curves[mech] * 100, linewidth=1.5, label=pname)
        ax.set_xlabel('MTBF (hours)')
        ax.set_ylabel('Optimal Overhead (%)')
        ax.set_title(f'{mech} — All Models')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()

else:  # "all"
    fig, ax = plt.subplots(figsize=(12, 6))
    for p in param_sets:
        curves = all_curves[p['name']]
        for mech, oh in curves.items():
            ax.plot(MTBF_range, oh * 100, label=f'{p["name"]} {mech}', **STYLES[mech])
    ax.set_xlabel('MTBF (hours)')
    ax.set_ylabel('Optimal Overhead (%)')
    ax.set_title('Overhead Comparison: All Models × All Mechanisms')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()

# --- Summary tables ---

for p in param_sets:
    t = p['t_step'] + p['t_l']
    t_a = p['t_d2h'] + p['t_persist']
    N_star = compute_N_star(p)
    viable = ours_viable(p)

    print(f"\n{'=' * 80}")
    print(f"  {p['name']}")
    print(f"{'=' * 80}")
    print(f"  t={t:.5f}  t_d2h={p['t_d2h']}  t_persist={p['t_persist']}  t_a={t_a:.3f}")
    print(f"  t_r={p['t_r']}  t_rc={p['t_rc']}  t_cp={p['t_cp']}")
    print(f"  L_cpu={p['L_cpu']}  L_disk={p['L_disk']}")
    if N_star != np.inf:
        t_cpu = compute_t_cpu(N_star, p)
        print(f"  Ours: N*={N_star}  t_cpu={t_cpu:.5f}  viable={'YES' if viable else 'NO'}")
    else:
        print(f"  Ours: NOT VIABLE (t <= t_rc)")
    print()

    header = (f"  {'MTBF':>6} {'M':>7}"
              f" | {'Log-Only':>10}"
              f" | {'Sync K*':>8} {'Sync%':>8}"
              f" | {'Ours%':>9}")
    print(header)
    print(f"  {'-' * len(header)}")

    for mtbf_h in MTBF_hours_list:
        M_val = mtbf_h * 3600 / t

        oh_lo = overhead_logonly(M_val, p) * 100

        K_sync, oh_sync_val = solve_sync(M_val, p)
        oh_sync_pct = oh_sync_val * 100

        oh_ours_val = overhead_ours(M_val, p)
        oh_ours_pct = f"{oh_ours_val*100:9.4f}" if oh_ours_val != np.inf else "     N/A "

        print(f"  {mtbf_h:>5.1f}h {M_val:>7.0f}"
              f" | {oh_lo:>10.4f}"
              f" | {K_sync:>8.1f} {oh_sync_pct:>8.4f}"
              f" | {oh_ours_pct}")

    # Decomposition at MTBF = 3h
    mtbf_h = 3.0
    M_val = mtbf_h * 3600 / t
    print(f"\n  --- Decomposition at MTBF = {mtbf_h}h (M = {M_val:.0f}) ---")

    lo_log, lo_load, lo_replay = overhead_logonly_decomposed(M_val, p)
    print(f"\n  Log-Only:")
    print(f"    Log:    {lo_log*100:10.6f} %")
    print(f"    Load:   {lo_load*100:10.6f} %")
    print(f"    Replay: {lo_replay*100:10.6f} %  (t_r/2 — fixed floor)")
    print(f"    Total:  {(lo_log+lo_load+lo_replay)*100:10.4f} %")

    K_sync, _ = solve_sync(M_val, p)
    s_log, s_block, s_load, s_replay = overhead_sync_decomposed(K_sync, M_val, p)
    s_total = s_log + s_block + s_load + s_replay
    print(f"\n  Sync Baseline (K*={K_sync:.1f}):")
    print(f"    Log:         {s_log*100:10.6f} %  ({s_log/s_total*100:5.1f}%)")
    print(f"    Ckpt block:  {s_block*100:10.6f} %  ({s_block/s_total*100:5.1f}%)")
    print(f"    Disk load:   {s_load*100:10.6f} %  ({s_load/s_total*100:5.1f}%)")
    print(f"    Replay:      {s_replay*100:10.6f} %  ({s_replay/s_total*100:5.1f}%)")
    print(f"    Total:       {s_total*100:10.4f} %")

    if viable:
        na_log, na_load, na_replay = overhead_ours_decomposed(M_val, p)
        na_total = na_log + na_load + na_replay
        print(f"\n  Ours (N*={N_star}):")
        print(f"    Log:    {na_log*100:10.6f} %")
        print(f"    Load:   {na_load*100:10.6f} %")
        print(f"    Replay: {na_replay*100:10.6f} %  (1 + N*/2 = {1+N_star/2:.1f} steps)")
        print(f"    Total:  {na_total*100:10.4f} %")
    else:
        print(f"\n  Ours: NOT VIABLE")