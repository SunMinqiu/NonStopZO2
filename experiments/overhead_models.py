"""
Overhead Models: Log-Only / Log+Anchor / CPU Replay
=====================================================
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
# Model 2: Log+Anchor (full checkpoint every K steps, all blocking)
#   overhead(K) = (K*t_l + t_a)/T_cycle + E[T_recover]/M
# ====================================================================

def overhead_anchor(K, M, p):
    t = p['t_step'] + p['t_l']
    t_a = p['t_d2h'] + p['t_persist']
    T_cycle = K * t + t_a
    E_train = p['L_disk'] + p['t_r'] * K / 2
    E_ckpt  = p['L_disk'] + p['t_r'] * K
    E_recover = (K * t / T_cycle) * E_train + (t_a / T_cycle) * E_ckpt
    return (K * p['t_l'] + t_a) / T_cycle + E_recover / M

def solve_anchor(M, p):
    result = minimize_scalar(lambda K: overhead_anchor(K, M, p),
                             bounds=(1, max(2, M)), method='bounded')
    return result.x, result.fun

def overhead_anchor_decomposed(K, M, p):
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
# Model 3: CPU Replay (CPU shadow, no anchor, no training blocking)
#
#   t_cpu(N) = t_rc + t_cp/N
#
#   Two regimes:
#     t > t_cpu(N): CPU catches up, lag = 1 step, d(N) = 1 + N/2
#     t <= t_cpu(N): CPU cannot catch up, lag grows, d(N) = M/2*(1-t/t_cpu) + N/2
#
#   N constraints:
#     N_catchup = ceil(t_cp / (t - t_rc))  — smallest N for catchup
#     N_persist = ceil(t_persist / t_rc)    — smallest N so persist doesn't block replay
#     N_opt     = (sqrt(M*t*t_cp) - t_cp) / t_rc — closed-form in lag regime
#
#   In catchup regime, N* = max(N_catchup, N_persist): smallest N satisfying both.
#   If N_persist > N_catchup, we pay extra replay cost to avoid blocking persist.
# ====================================================================

def compute_t_cpu(N, p):
    return p['t_rc'] + p['t_cp'] / N

def _replay_distance(N, M, p):
    """Replay distance d(N) with proper regime check."""
    t = p['t_step'] + p['t_l']
    t_cpu = compute_t_cpu(N, p)
    if t > t_cpu:
        return 1 + N / 2
    else:
        return M / 2 * (1 - t / t_cpu) + N / 2

def _find_catchup_N(p):
    """Smallest integer N >= 1 where CPU catches up (t > t_cpu(N)).
    Returns (N_catchup, can_catchup)."""
    t = p['t_step'] + p['t_l']
    gap = t - p['t_rc']
    if gap <= 0:
        return np.inf, False
    N_catchup = max(1, int(np.ceil(p['t_cp'] / gap)))
    if t > compute_t_cpu(N_catchup, p):
        return N_catchup, True
    else:
        return np.inf, False

def _find_persist_N(p):
    """Smallest integer N so shadow persist doesn't block replay.
    t_persist <= N * t_rc  =>  N >= t_persist / t_rc"""
    if p['t_rc'] <= 0:
        return np.inf
    return max(1, int(np.ceil(p['t_persist'] / p['t_rc'])))

def compute_N_star(M, p):
    """Find optimal N* considering catchup, lag, and persist constraints.

    Returns dict with:
      N_star, regime, reason, d, t_stale,
      N_catchup, N_persist, N_opt, can_catchup, persist_blocks
    """
    t = p['t_step'] + p['t_l']

    N_catchup, can_catchup = _find_catchup_N(p)
    N_persist = _find_persist_N(p)

    # Lag regime closed-form
    val = M * t * p['t_cp']
    if val > 0 and p['t_rc'] > 0:
        N_opt_raw = (np.sqrt(val) - p['t_cp']) / p['t_rc']
        N_opt = max(1, N_opt_raw)
    else:
        N_opt = np.inf

    # Only keep N_opt if actually in lag regime
    if N_opt != np.inf and t > compute_t_cpu(N_opt, p):
        N_opt_valid = False
    else:
        N_opt_valid = N_opt != np.inf

    # Build candidates
    candidates = []

    if can_catchup:
        # Catchup: N must satisfy both catchup and persist
        N_catchup_final = max(N_catchup, N_persist)
        # Verify still in catchup regime (should be, since larger N => smaller t_cpu)
        if t > compute_t_cpu(N_catchup_final, p):
            if N_persist > N_catchup:
                reason = f'catchup (N raised {N_catchup}->{N_catchup_final} for persist)'
            else:
                reason = 'catchup'
            candidates.append((N_catchup_final, 'catchup', reason))

    if N_opt_valid:
        # Lag regime: also enforce persist constraint
        N_lag_final = max(N_opt, N_persist)
        reason_lag = 'lag'
        if N_persist > N_opt:
            reason_lag = f'lag (N raised {N_opt:.1f}->{N_lag_final} for persist)'
        candidates.append((N_lag_final, 'lag', reason_lag))

    if not candidates:
        return dict(N_star=np.inf, regime='none', reason='not viable', d=np.inf,
                    t_stale=np.inf, N_catchup=N_catchup, N_persist=N_persist,
                    N_opt=N_opt, can_catchup=can_catchup, persist_blocks=False)

    best = min(candidates, key=lambda x: _replay_distance(x[0], M, p))
    best_N, best_regime, best_reason = best
    d = _replay_distance(best_N, M, p)
    t_stale = best_N * p['t_rc']
    persist_blocks = (N_persist > N_catchup) if can_catchup else False

    return dict(N_star=best_N, regime=best_regime, reason=best_reason, d=d,
                t_stale=t_stale, N_catchup=N_catchup if can_catchup else np.inf,
                N_persist=N_persist, N_opt=N_opt if N_opt_valid else np.inf,
                can_catchup=can_catchup, persist_blocks=persist_blocks)

def overhead_cpu_replay(M, p):
    t = p['t_step'] + p['t_l']
    info = compute_N_star(M, p)
    if info['N_star'] == np.inf:
        return np.inf
    return p['t_l'] / t + (p['L_cpu'] + p['t_r'] * info['d']) / M

def overhead_cpu_replay_decomposed(M, p):
    t = p['t_step'] + p['t_l']
    info = compute_N_star(M, p)
    if info['N_star'] == np.inf:
        return np.inf, np.inf, np.inf
    return p['t_l'] / t, p['L_cpu'] / M, p['t_r'] * info['d'] / M

# ====================================================================
# Model 4: From CPU Persist (hard failure — load from disk persist)
#
#   Same N*, same shadow mechanism as CPU Replay.
#   Differences:
#     - Load from disk: L_disk instead of L_cpu
#     - Extra replay: t_persist/t_rc steps (persist may not have completed)
#
#   d_hard(N) = d_soft(N) + t_persist / t_rc
#   E[T_recover] = L_disk + t_r * d_hard
# ====================================================================

def _replay_distance_hard(N, M, p):
    """Hard failure replay distance = soft distance + t_persist/t_rc."""
    return _replay_distance(N, M, p) + p['t_persist'] / p['t_rc']

def overhead_cpu_persist(M, p):
    t = p['t_step'] + p['t_l']
    info = compute_N_star(M, p)
    if info['N_star'] == np.inf:
        return np.inf
    d_hard = _replay_distance_hard(info['N_star'], M, p)
    return p['t_l'] / t + (p['L_disk'] + p['t_r'] * d_hard) / M

def overhead_cpu_persist_decomposed(M, p):
    t = p['t_step'] + p['t_l']
    info = compute_N_star(M, p)
    if info['N_star'] == np.inf:
        return np.inf, np.inf, np.inf
    d_hard = _replay_distance_hard(info['N_star'], M, p)
    return p['t_l'] / t, p['L_disk'] / M, p['t_r'] * d_hard / M

# ====================================================================
# Compute & Plot
# ====================================================================
# plot_mode controls overhead-vs-MTBF plot grouping:
#   "by_model"     — one plot per model, all mechanisms on same axes
#   "by_mechanism"  — one plot per mechanism, all models on same axes
#   "all"           — single plot with everything (default)

if 'plot_mode' not in dir():
    plot_mode = "all"

# save_dir: set to a directory path to save PDF+PNG for each figure.
# Leave unset (or None) to only display inline.
if 'save_dir' not in dir():
    save_dir = None

# fig_width: figure width in inches (height auto-calculated from golden ratio)
if 'fig_width' not in dir():
    fig_width = 10
_golden = fig_width / 1.618

def _save_and_show(fig, name):
    """Save figure as PDF (if save_dir is set), then show."""
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'{name}.pdf'), bbox_inches='tight')
    plt.show()

ALL_MECHANISMS = ['Log-Only', 'Log+Anchor', 'From CPU Persist', 'CPU Replay']

def _compute_all(p, MTBF_range):
    t = p['t_step'] + p['t_l']
    M_arr = MTBF_range * 3600 / t
    curves = {}
    K_stars = []
    N_infos = []

    curves['Log-Only']    = np.array([overhead_logonly(M, p) for M in M_arr])

    anchor_results = [solve_anchor(M, p) for M in M_arr]
    curves['Log+Anchor']  = np.array([r[1] for r in anchor_results])
    K_stars = np.array([r[0] for r in anchor_results])

    oh_cr = np.array([overhead_cpu_replay(M, p) for M in M_arr])
    if not np.all(np.isinf(oh_cr)):
        curves['CPU Replay'] = oh_cr

    oh_cp = np.array([overhead_cpu_persist(M, p) for M in M_arr])
    if not np.all(np.isinf(oh_cp)):
        curves['From CPU Persist'] = oh_cp

    N_infos = [compute_N_star(M, p) for M in M_arr]

    return curves, K_stars, N_infos

# Precompute
all_data = {}
for p in param_sets:
    curves, K_stars, N_infos = _compute_all(p, MTBF_range)
    all_data[p['name']] = dict(curves=curves, K_stars=K_stars, N_infos=N_infos, params=p)

print("=" * 80)
print("OVERHEAD MODEL COMPARISON")
print("=" * 80)

STYLES = {
    'Log-Only':          {'linestyle': '-',  'linewidth': 2.5, 'color': '#2ca02c'},  # green
    'Log+Anchor':        {'linestyle': '-',  'linewidth': 2.5, 'color': '#1f77b4'},  # blue
    'CPU Replay':        {'linestyle': '-',  'linewidth': 2.5, 'color': '#d62728'},  # red
    'From CPU Persist':  {'linestyle': '--', 'linewidth': 2.5, 'color': '#ff7f0e'},  # orange dashed
}

# Global font size — set `font_size` in input cell to control the SMALLEST text.
# All other elements scale up from this base.
#   annotations / bar labels:  font_size      (base)
#   legend:                    font_size + 2
#   axis labels + tick labels: font_size + 3
#   title:                     font_size + 4
if 'font_size' not in dir():
    font_size = 12
_fs = font_size
plt.rcParams.update({
    'font.size':          _fs,
    'axes.titlesize':     _fs + 6,
    'axes.labelsize':     _fs + 6,
    'xtick.labelsize':    _fs + 3,
    'ytick.labelsize':    _fs + 3,
    'legend.fontsize':    _fs + 2,
    'font.weight':        'bold',
    'axes.titleweight':   'bold',
    'axes.labelweight':   'bold',
})

if 'MTBF_CUT' not in dir():
    MTBF_CUT = 4.0  # vertical cut line for annotations

def _annotate_at_cut(ax, mtbf_range, oh_arr, color, offset_y=0):
    """Annotate a line's value at MTBF_CUT with a label (single-line, used as fallback)."""
    idx = np.argmin(np.abs(mtbf_range - MTBF_CUT))
    val = oh_arr[idx]
    if np.isfinite(val):
        ax.annotate(f'{val*100:.2f}%',
                    xy=(MTBF_CUT, val * 100), xytext=(8, offset_y),
                    textcoords='offset points', color=color, va='center',
                    fontsize=_fs, fontweight='bold')

def _annotate_all_at_cut(ax, mtbf_range, curves_with_colors, min_gap_pt=18):
    """Annotate all lines at MTBF_CUT, automatically spreading labels to avoid overlap.

    curves_with_colors: list of (oh_arr, color)
    min_gap_pt: minimum vertical gap between labels in points
    """
    idx = np.argmin(np.abs(mtbf_range - MTBF_CUT))
    # Collect (y_data, color)
    items = []
    for oh_arr, color in curves_with_colors:
        val = oh_arr[idx]
        if np.isfinite(val):
            items.append((val * 100, color))
    if not items:
        return

    # Convert data y → display points for spreading
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    transform = ax.transData
    def y_to_pt(y):
        return transform.transform((0, y))[1] / (fig.dpi / 72)
    def pt_to_y(pt):
        return transform.inverted().transform((0, pt * (fig.dpi / 72)))[1]

    # Sort by y value (ascending)
    items.sort(key=lambda x: x[0])

    # Compute label positions in points, push up if too close
    data_ys = [y for y, _ in items]
    label_pts = [y_to_pt(data_ys[0])]
    for i in range(1, len(items)):
        natural_pt = y_to_pt(data_ys[i])
        label_pts.append(max(natural_pt, label_pts[-1] + min_gap_pt))

    # Draw with arrow connecting label to actual data point
    for (y_val, color), lbl_pt in zip(items, label_pts):
        offset_y = lbl_pt - y_to_pt(y_val)
        ax.annotate(f'{y_val:.2f}%',
                    xy=(MTBF_CUT, y_val), xytext=(8, offset_y),
                    textcoords='offset points', color=color, va='center',
                    fontsize=_fs, fontweight='bold')

def _add_cut_line(ax):
    ax.axvline(MTBF_CUT, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    # Mark on x-axis
    xlim = ax.get_xlim()
    current_ticks = [t for t in ax.get_xticks() if xlim[0] <= t <= xlim[1]]
    if MTBF_CUT not in current_ticks:
        current_ticks.append(MTBF_CUT)
        ax.set_xticks(sorted(current_ticks))
    ax.get_xticklabels()  # force redraw
    # Bold the cut tick
    for label in ax.get_xticklabels():
        if f'{MTBF_CUT:.0f}' in label.get_text() or f'{MTBF_CUT}' in label.get_text():
            label.set_color('gray')
            label.set_fontweight('bold')

# --- Plot 1: Overhead vs MTBF ---

def _mech_label(mech, pname, data, prefix=None):
    """Build legend label, appending K* at MTBF_CUT for Log+Anchor."""
    label = f'{prefix} {mech}' if prefix else mech
    idx = np.argmin(np.abs(MTBF_range - MTBF_CUT))
    if mech == 'Log+Anchor':
        K_val = data['K_stars'][idx]
        label += f' (K*={K_val:.0f})'
    elif mech in ('CPU Replay', 'From CPU Persist'):
        N_info = data['N_infos'][idx]
        if N_info['N_star'] != np.inf:
            label += f' (N*={N_info["N_star"]:.0f})'
    return label

if plot_mode == "by_model":
    for pname, data in all_data.items():
        fig, ax = plt.subplots(figsize=(fig_width, _golden))
        curves_colors = []
        for mech, oh in data['curves'].items():
            line, = ax.plot(MTBF_range, oh * 100, label=_mech_label(mech, pname, data), **STYLES[mech])
            curves_colors.append((oh, line.get_color()))
        _add_cut_line(ax)
        ax.set_xlabel('MTBF (hours)')
        ax.set_ylabel('Optimal WASTE (%)')
        ax.set_title(f'{pname} — WASTE vs MTBF')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        _annotate_all_at_cut(ax, MTBF_range, curves_colors)
        # _save_and_show(fig, f'overhead_{pname}')

elif plot_mode == "by_mechanism":
    mechs_present = set()
    for data in all_data.values():
        mechs_present.update(data['curves'].keys())
    for mech in ALL_MECHANISMS:
        if mech not in mechs_present:
            continue
        fig, ax = plt.subplots(figsize=(fig_width, _golden))
        curves_colors = []
        for pname, data in all_data.items():
            if mech in data['curves']:
                oh = data['curves'][mech]
                line, = ax.plot(MTBF_range, oh * 100, linewidth=2.0, label=pname)
                curves_colors.append((oh, line.get_color()))
        _add_cut_line(ax)
        ax.set_xlabel('MTBF (hours)')
        ax.set_ylabel('Optimal Overhead (%)')
        ax.set_title(f'{mech} — All Models')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        _annotate_all_at_cut(ax, MTBF_range, curves_colors)
        _save_and_show(fig, f'overhead_{mech.replace(" ", "_")}')

else:  # "all"
    fig, ax = plt.subplots(figsize=(fig_width, _golden))
    curves_colors = []
    for pname, data in all_data.items():
        for mech, oh in data['curves'].items():
            line, = ax.plot(MTBF_range, oh * 100, label=_mech_label(mech, pname, data, prefix=pname), **STYLES[mech])
            curves_colors.append((oh, line.get_color()))
    _add_cut_line(ax)
    ax.set_xlabel('MTBF (hours)')
    ax.set_ylabel('Optimal Overhead (%)')
    ax.set_title('Overhead Comparison: All Models × All Mechanisms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _annotate_all_at_cut(ax, MTBF_range, curves_colors)
    _save_and_show(fig, 'overhead_all')

# # --- Plot 2: Log+Anchor optimal K* vs MTBF ---

# fig, ax = plt.subplots(figsize=(fig_width, _golden))
# for pname, data in all_data.items():
#     line, = ax.plot(MTBF_range, data['K_stars'], linewidth=2.0, label=pname)
#     idx = np.argmin(np.abs(MTBF_range - MTBF_CUT))
#     val = data['K_stars'][idx]
#     ax.annotate(f'{val:.0f}', xy=(MTBF_CUT, val), xytext=(8, 0),
#                 textcoords='offset points', color=line.get_color(), va='center',
#                 fontsize=_fs, fontweight='bold')
# _add_cut_line(ax)
# ax.set_xlabel('MTBF (hours)')
# ax.set_ylabel('K* (optimal anchor interval, steps)')
# ax.set_title('Log+Anchor: Optimal K* vs MTBF')
# ax.legend()
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# _save_and_show(fig, 'kstar_vs_mtbf')

# --- Plot 3: CPU Replay N* bar chart ---

fig, ax = plt.subplots(figsize=(fig_width, _golden))
model_names = list(all_data.keys())
N_vals = []
N_catchups = []
N_persists = []
t_stales = []
for pname in model_names:
    p = all_data[pname]['params']
    t = p['t_step'] + p['t_l']
    M_3h = 3.0 * 3600 / t
    info = compute_N_star(M_3h, p)
    N_vals.append(info['N_star'] if info['N_star'] != np.inf else 0)
    N_catchups.append(info['N_catchup'] if info['N_catchup'] != np.inf else 0)
    N_persists.append(info['N_persist'])
    t_stales.append(info['t_stale'] if info['t_stale'] != np.inf else 0)

x = np.arange(len(model_names))
width = 0.25

bars1 = ax.bar(x - width, N_catchups, width, label='N_catchup', alpha=0.7)
bars2 = ax.bar(x, N_persists, width, label='N_persist', alpha=0.7)
bars3 = ax.bar(x + width, N_vals, width, label='N* (chosen)', alpha=0.9, edgecolor='black')

max_N = max(max(N_vals), max(N_catchups), max(N_persists))
ax.set_ylim(0, max_N * 1.8)  # extra headroom for annotations

for i, (n, ts) in enumerate(zip(N_vals, t_stales)):
    if n > 0:
        ax.text(x[i] + width, n + max_N * 0.05,
                f'N*={n:.0f}\nt_stale={ts:.2f}s',
                ha='center', va='bottom', fontsize=_fs, fontweight='bold')

ax.set_xlabel('Model')
ax.set_ylabel('N (commit interval, steps)')
ax.set_title('CPU Replay: N* Selection (at MTBF = 3h)')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
_save_and_show(fig, 'nstar_bar')

# --- Plot 4: WASTE bar chart at MTBF_CUT, stacked checkpoint / recovery ---

def _decompose3_at(mech, M, p):
    """Return (checkpoint, load, replay) overhead for a mechanism at given M."""
    if mech == 'Log-Only':
        log, load, replay = overhead_logonly_decomposed(M, p)
        return log, load, replay
    elif mech == 'Log+Anchor':
        K, _ = solve_anchor(M, p)
        log, ckpt_block, load, replay = overhead_anchor_decomposed(K, M, p)
        return log + ckpt_block, load, replay
    elif mech == 'CPU Replay':
        log, load, replay = overhead_cpu_replay_decomposed(M, p)
        return log, load, replay
    elif mech == 'From CPU Persist':
        log, load, replay = overhead_cpu_persist_decomposed(M, p)
        return log, load, replay
    return 0, 0, 0

fig, ax = plt.subplots(figsize=(fig_width, _golden))
model_names = list(all_data.keys())
mechs_in_data = []
for mech in ALL_MECHANISMS:
    if any(mech in all_data[pn]['curves'] for pn in model_names):
        mechs_in_data.append(mech)

n_mechs = len(mechs_in_data)
x = np.arange(len(model_names))
width = 0.8 / n_mechs
idx_cut = np.argmin(np.abs(MTBF_range - MTBF_CUT))

for j, mech in enumerate(mechs_in_data):
    ckpt_vals = []
    load_vals = []
    replay_vals = []
    for pname in model_names:
        p = all_data[pname]['params']
        t = p['t_step'] + p['t_l']
        M_val = MTBF_CUT * 3600 / t
        if mech in all_data[pname]['curves']:
            c, l, r = _decompose3_at(mech, M_val, p)
            ckpt_vals.append(c * 100 if np.isfinite(c) else 0)
            load_vals.append(l * 100 if np.isfinite(l) else 0)
            replay_vals.append(r * 100 if np.isfinite(r) else 0)
        else:
            ckpt_vals.append(0)
            load_vals.append(0)
            replay_vals.append(0)
    offset = (j - (n_mechs - 1) / 2) * width
    bot_load = np.array(ckpt_vals)
    bot_replay = bot_load + np.array(load_vals)
    ax.bar(x + offset, ckpt_vals, width, alpha=0.85,
           color=STYLES[mech]['color'])
    ax.bar(x + offset, load_vals, width, bottom=bot_load, alpha=0.55,
           color=STYLES[mech]['color'], hatch='...')
    ax.bar(x + offset, replay_vals, width, bottom=bot_replay, alpha=0.35,
           color=STYLES[mech]['color'], hatch='///')

# y-axis limit
all_totals = []
for pname in model_names:
    p = all_data[pname]['params']
    t = p['t_step'] + p['t_l']
    M_val = MTBF_CUT * 3600 / t
    for mech in mechs_in_data:
        if mech in all_data[pname]['curves']:
            c, l, r = _decompose3_at(mech, M_val, p)
            if all(np.isfinite(v) for v in (c, l, r)):
                all_totals.append((c + l + r) * 100)
ax.set_ylim(0, max(all_totals) * 1.3 if all_totals else 1)
ax.set_xlabel('Model')
ax.set_ylabel('WASTE (%)')
ax.set_title(f'WASTE Decomposition at MTBF = {MTBF_CUT}h')
ax.set_xticks(x)
ax.set_xticklabels(model_names)

# Two-column legend: color = mechanism, hatch = component
from matplotlib.patches import Patch
legend_handles = []
for mech in mechs_in_data:
    legend_handles.append(Patch(facecolor=STYLES[mech]['color'], alpha=0.85, label=mech))
legend_handles.append(Patch(facecolor='gray', alpha=0.85, label='checkpoint'))
legend_handles.append(Patch(facecolor='gray', alpha=0.55, hatch='...', label='load'))
legend_handles.append(Patch(facecolor='gray', alpha=0.35, hatch='///', label='replay'))
ax.legend(handles=legend_handles, ncol=2)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
_save_and_show(fig, 'waste_bar')

# --- Summary tables ---

for p in param_sets:
    pname = p['name']
    data = all_data[pname]
    t = p['t_step'] + p['t_l']
    t_a = p['t_d2h'] + p['t_persist']

    N_catchup, can_catchup = _find_catchup_N(p)
    N_persist = _find_persist_N(p)

    print(f"\n{'=' * 100}")
    print(f"  {pname}")
    print(f"{'=' * 100}")
    print(f"  t={t:.5f}  t_d2h={p['t_d2h']}  t_persist={p['t_persist']}  t_a={t_a:.3f}")
    print(f"  t_r={p['t_r']}  t_rc={p['t_rc']}  t_cp={p['t_cp']}")
    print(f"  L_cpu={p['L_cpu']}  L_disk={p['L_disk']}")
    print()
    print(f"  CPU Replay N constraints:")
    if can_catchup:
        print(f"    N_catchup = {N_catchup}  (smallest N for CPU to catch up)")
    else:
        print(f"    N_catchup = inf  (CPU cannot catch up: t={t:.4f} <= t_rc={p['t_rc']})")
    print(f"    N_persist = {N_persist}  (smallest N so persist fits: t_persist={p['t_persist']:.3f} <= N*t_rc={N_persist}*{p['t_rc']:.3f}={N_persist*p['t_rc']:.3f})")
    if can_catchup:
        N_final = max(N_catchup, N_persist)
        binding = "N_persist" if N_persist > N_catchup else "N_catchup"
        print(f"    => N* = max({N_catchup}, {N_persist}) = {N_final}  (binding: {binding})")
        print(f"    => t_stale = {N_final} * {p['t_rc']} = {N_final * p['t_rc']:.3f}s")
    print()

    header = (f"  {'MTBF':>6} {'M':>7}"
              f" | {'LogOnly':>9}"
              f" | {'Anch K*':>8} {'Anch%':>8}"
              f" | {'CR N*':>6} {'regime':>8} {'t_stale':>8} {'CR%':>9} {'Persist%':>9}")
    print(header)
    print(f"  {'-' * len(header)}")

    for mtbf_h in MTBF_hours_list:
        M_val = mtbf_h * 3600 / t

        oh_lo = overhead_logonly(M_val, p) * 100

        K_anch, oh_anch_val = solve_anchor(M_val, p)

        info = compute_N_star(M_val, p)
        oh_cr_val = overhead_cpu_replay(M_val, p)
        oh_cp_val = overhead_cpu_persist(M_val, p)

        if oh_cr_val != np.inf:
            cr_pct = f"{oh_cr_val*100:9.4f}"
            cp_pct = f"{oh_cp_val*100:9.4f}"
            n_str = f"{info['N_star']:>6.0f}" if info['N_star'] == int(info['N_star']) else f"{info['N_star']:>6.1f}"
            r_str = f"{info['regime']:>8}"
            ts_str = f"{info['t_stale']:>7.3f}s"
        else:
            cr_pct = "     N/A "
            cp_pct = "     N/A "
            n_str = "   N/A"
            r_str = "     N/A"
            ts_str = "     N/A"

        print(f"  {mtbf_h:>5.1f}h {M_val:>7.0f}"
              f" | {oh_lo:>9.4f}"
              f" | {K_anch:>8.1f} {oh_anch_val*100:>8.4f}"
              f" | {n_str} {r_str} {ts_str} {cr_pct} {cp_pct}")

    # Decomposition at MTBF = 3h
    mtbf_h = 3.0
    M_val = mtbf_h * 3600 / t
    info = compute_N_star(M_val, p)

    print(f"\n  --- Decomposition at MTBF = {mtbf_h}h (M = {M_val:.0f}) ---")

    lo_log, lo_load, lo_replay = overhead_logonly_decomposed(M_val, p)
    print(f"\n  Log-Only:")
    print(f"    Log:    {lo_log*100:10.6f} %")
    print(f"    Load:   {lo_load*100:10.6f} %")
    print(f"    Replay: {lo_replay*100:10.6f} %  (t_r/2 — fixed floor)")
    print(f"    Total:  {(lo_log+lo_load+lo_replay)*100:10.4f} %")

    K_anch, _ = solve_anchor(M_val, p)
    a_log, a_block, a_load, a_replay = overhead_anchor_decomposed(K_anch, M_val, p)
    a_total = a_log + a_block + a_load + a_replay
    print(f"\n  Log+Anchor (K*={K_anch:.1f}):")
    print(f"    Log:         {a_log*100:10.6f} %  ({a_log/a_total*100:5.1f}%)")
    print(f"    Ckpt block:  {a_block*100:10.6f} %  ({a_block/a_total*100:5.1f}%)")
    print(f"    Disk load:   {a_load*100:10.6f} %  ({a_load/a_total*100:5.1f}%)")
    print(f"    Replay:      {a_replay*100:10.6f} %  ({a_replay/a_total*100:5.1f}%)")
    print(f"    Total:       {a_total*100:10.4f} %")

    if info['N_star'] != np.inf:
        d_soft = _replay_distance(info['N_star'], M_val, p)
        d_hard = _replay_distance_hard(info['N_star'], M_val, p)
        persist_extra = p['t_persist'] / p['t_rc']

        cr_log, cr_load, cr_replay = overhead_cpu_replay_decomposed(M_val, p)
        cr_total = cr_log + cr_load + cr_replay
        print(f"\n  CPU Replay — soft (N*={info['N_star']:.0f}, {info['reason']}, d={d_soft:.1f} steps, t_stale={info['t_stale']:.3f}s):")
        print(f"    Log:    {cr_log*100:10.6f} %")
        print(f"    Load:   {cr_load*100:10.6f} %  (L_cpu={p['L_cpu']}s)")
        print(f"    Replay: {cr_replay*100:10.6f} %")
        print(f"    Total:  {cr_total*100:10.4f} %")

        cp_log, cp_load, cp_replay = overhead_cpu_persist_decomposed(M_val, p)
        cp_total = cp_log + cp_load + cp_replay
        print(f"\n  From CPU Persist — hard (N*={info['N_star']:.0f}, d={d_hard:.1f} steps = {d_soft:.1f} + {persist_extra:.1f} persist lag):")
        print(f"    Log:    {cp_log*100:10.6f} %")
        print(f"    Load:   {cp_load*100:10.6f} %  (L_disk={p['L_disk']}s)")
        print(f"    Replay: {cp_replay*100:10.6f} %")
        print(f"    Total:  {cp_total*100:10.4f} %")
    else:
        print(f"\n  CPU Replay: NOT VIABLE")
        print(f"  From CPU Persist: NOT VIABLE")