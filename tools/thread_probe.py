"""ThreadProbe: calibrate producer/consumer thread allocation for shadow pipeline.

Flow:
    Step 1: measure t_gen(generator) and t_update(consumer) curves; find t_update plateau.
    Step 2: within the plateau, for each consumer, auto-derive P via formula
            P = max(1, ceil(t_gen(gen_per)/t_update(consumer)))  with gen_per = (C-consumer)//P
            then run a concurrent mini-pipeline to measure real t_cpu.
    Step 3: pick best by measured t_cpu, derive N_min = ceil(t_commit/(t_gpu - t_cpu)),
            print env vars, save two PDFs + log.
"""
import math
import os
import sys
import threading
import time
import traceback

import matplotlib.pyplot as plt
import torch

from zo2.trainer.hf_transformers.log_based_tuning import (
    calibrate_producer_consumer,
    _interp_curve,
)
from zo2.trainer.hf_transformers.log_based_replay import (
    _generate_z_for_one_step,
    _apply_single_update_with_pregenerated_z,
)


# =============================================================================
# Logging helper
# =============================================================================
class _Tee:
    """Duplicate writes to multiple streams (e.g. stdout + log file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


# =============================================================================
# Step 1: measure curves + plateau
# =============================================================================
def measure_curves(state, param_names, *, C, t_train,
                   rng_device='zo_rng',
                   core_start, core_stop, core_step,
                   n_warmup, n_measure,
                   plateau_threshold,
                   adam_state=None):
    result = calibrate_producer_consumer(
        state, param_names, rng_device=rng_device,
        C=C, t_train=t_train,
        n_warmup=n_warmup, n_measure=n_measure,
        core_start=core_start,
        core_stop=core_stop if core_stop is not None else C - 1,
        core_step=core_step,
        adam_state=adam_state,
    )
    t_gen_curve = result['t_gen_curve']
    t_update_curve = result['t_update_curve']
    t_commit = result['t_commit']

    t_update_min = min(t_update_curve.values())
    plateau_points = sorted(
        n for n, t in t_update_curve.items()
        if t < t_update_min * plateau_threshold
    )
    n_low, n_high = plateau_points[0], plateau_points[-1]
    print(f"\nt_update plateau: consumer=[{n_low}, {n_high}]")

    return {
        't_gen_curve': t_gen_curve,
        't_update_curve': t_update_curve,
        't_commit': t_commit,
        'n_low': n_low,
        'n_high': n_high,
    }


# =============================================================================
# Step 2: concurrent fine scan
# =============================================================================
def _run_concurrent_test(P, cores_per_prod, consumer, state, param_names,
                         *, rng_device='zo_rng', zo_eps=1e-3,
                         n_steps=3, n_warmup=1, adam_state=None):
    """Pipelined concurrent test with real producer-consumer bandwidth contention.

    Producers run continuously in background (not lock-step), filling a queue.
    Consumer measures apply time while producers are actively generating on
    their cores — simulating real memory-bandwidth competition.
    """
    import queue as queue_module
    import zo_rng as _zo_rng
    _zo_rng.set_num_threads(cores_per_prod)
    torch.set_num_threads(consumer)

    result_queue = queue_module.Queue(maxsize=max(2, P))
    stop = threading.Event()
    err = [None]

    def producer(worker_id):
        try:
            step = worker_id
            while not stop.is_set():
                z = _generate_z_for_one_step(
                    1_000_000 + step, param_names, state, rng_device)
                while not stop.is_set():
                    try:
                        result_queue.put(z, timeout=0.05)
                        break
                    except queue_module.Full:
                        continue
                step += P
        except Exception as e:
            err[0] = e

    threads = [threading.Thread(target=producer, args=(i,), daemon=True)
               for i in range(P)]
    for t in threads:
        t.start()

    dummy = {'seed': 42, 'grad': 1e-4, 'lr': 1e-5, 'wd': 0.01, 'zo_eps': zo_eps}
    times = []
    for step in range(n_steps):
        z = result_queue.get(timeout=300)
        if err[0]:
            raise err[0]
        t0 = time.monotonic()
        _apply_single_update_with_pregenerated_z(
            state, dummy, param_names, z,
            default_zo_eps=zo_eps, simulate_perturbation=True, zo2_mode=False,
            adam_state=adam_state)
        t1 = time.monotonic()
        if step >= n_warmup:
            times.append(t1 - t0)

    stop.set()
    for t in threads:
        t.join(timeout=5)
    if err[0]:
        raise err[0]
    times.sort()
    return times[len(times) // 2]


def fine_search(curves, state, param_names, *, C,
                consumer_step, test_n_steps=3, test_n_warmup=1,
                rng_device='zo_rng', adam_state=None):
    """For each consumer in plateau, compute P via formula and measure real t_cpu."""
    t_gen_curve = curves['t_gen_curve']
    t_update_curve = curves['t_update_curve']
    n_low = curves['n_low']
    n_high = curves['n_high']

    def t_gen_at(g):
        return _interp_curve(t_gen_curve, max(1, int(g)))

    def t_upd_at(c):
        return _interp_curve(t_update_curve, max(1, int(c)))

    def pick_P(consumer):
        """Smallest P with P * t_update(consumer) >= t_gen(gen_per).
        Upper bound: P <= generator_total (when gen_per < 1 we give up)."""
        generator = C - consumer
        if generator < 1:
            return None
        t_upd = t_upd_at(consumer)
        P = 1
        while True:
            gen_per = generator // P
            if gen_per < 1:
                return None  # P > generator_total → infeasible
            if P * t_upd >= t_gen_at(gen_per):
                return P, gen_per
            P += 1

    consumers_to_test = list(range(n_low, n_high + 1, consumer_step))
    if n_high not in consumers_to_test:
        consumers_to_test.append(n_high)

    print(f"\nStep 2: concurrent scan over plateau=[{n_low}, {n_high}], step={consumer_step}")
    print(f"{'consumer':>9} {'generator':>10} {'P':>3} {'gen/P':>7} {'t_cpu':>10}")
    print(f"{'-' * 48}")

    results = []
    for consumer in consumers_to_test:
        picked = pick_P(consumer)
        if picked is None:
            print(f"{consumer:>9} {C - consumer:>10} {'-':>3} {'-':>7} infeasible")
            continue
        P, gen_per = picked
        t_measured = _run_concurrent_test(
            P, gen_per, consumer, state, param_names,
            rng_device=rng_device, n_steps=test_n_steps, n_warmup=test_n_warmup,
            adam_state=adam_state)
        results.append({
            'consumer': consumer,
            'generator': C - consumer,
            'P': P,
            'gen_per': gen_per,
            't_cpu': t_measured,
        })
        print(f"{consumer:>9} {C - consumer:>10} {P:>3} {gen_per:>7} {t_measured:>9.3f}s")

    if not results:
        raise RuntimeError("Step 2: no feasible configuration found")

    best = min(results, key=lambda r: r['t_cpu'])
    print(f"\nBest: consumer={best['consumer']}, generator={best['generator']}, "
          f"P={best['P']}, gen/P={best['gen_per']}, t_cpu={best['t_cpu']:.3f}s")
    return results, best


# =============================================================================
# Plotting
# =============================================================================
def _setup_rcparams(font_size):
    plt.rcParams.update({
        'font.size':          font_size,
        'axes.titlesize':     font_size + 6,
        'axes.labelsize':     font_size + 6,
        'xtick.labelsize':    font_size + 3,
        'ytick.labelsize':    font_size + 3,
        'legend.fontsize':    font_size + 2,
        'font.weight':        'bold',
        'axes.titleweight':   'bold',
        'axes.labelweight':   'bold',
    })


def plot_curves(curves, t_train, *, fig_width, font_size, save_path):
    _setup_rcparams(font_size)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width / 1.618))
    t_gen_curve = curves['t_gen_curve']
    t_update_curve = curves['t_update_curve']
    n_low = curves['n_low']
    n_high = curves['n_high']

    cs = sorted(t_gen_curve.keys())
    ns = sorted(t_update_curve.keys())
    ax.plot(cs, [t_gen_curve[c] for c in cs], 'b.-',
            label='t_generator', markersize=8, linewidth=2)
    ax.plot(ns, [t_update_curve[n] for n in ns], '.-',
            color='green', label='t_consumer', markersize=8, linewidth=2)
    ax.axvspan(n_low, n_high, alpha=0.15, color='green',
               label=f't_update plateau [{n_low}, {n_high}]')
    ax.axhline(t_train, color='red', ls='--', linewidth=2,
               label=f't_train={t_train:.3f}s')
    ax.set_xlabel('Number of threads')
    ax.set_ylabel('Time (s)')
    ax.set_title('Step 1: generator / consumer curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    print(f"[plot] saved curves → {save_path}")
    plt.show()
    plt.close(fig)


def plot_scan(results, best, t_train, *, fig_width, font_size, save_path):
    _setup_rcparams(font_size)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width / 1.618))
    xs = [r['consumer'] for r in results]
    ys = [r['t_cpu']    for r in results]
    ax.plot(xs, ys, 'o-', color='#1f77b4', markersize=8, linewidth=2,
            label='t_cpu (measured)')
    ax.plot([best['consumer']], [best['t_cpu']], marker='*', markersize=18,
            color='gold', markeredgecolor='black', markeredgewidth=1.5,
            linestyle='None', label=f"best (P={best['P']})")
    ax.axhline(t_train, color='red', ls='--', linewidth=2,
               label=f't_gpu={t_train:.3f}s')
    for r in results:
        ax.annotate(f"P={r['P']}",
                    xy=(r['consumer'], r['t_cpu']),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=font_size, fontweight='bold')
    ax.set_xlabel('consumer (threads)')
    ax.set_ylabel('t_cpu (s)')
    ax.set_title('Step 2: concurrent fine scan')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    print(f"[plot] saved scan → {save_path}")
    plt.show()
    plt.close(fig)


# =============================================================================
# Replot from log: override gpu_step_s, apply calibration offset
# =============================================================================
def replot_from_log(log_path, gpu_step_s, *, calibration=0.0,
                    fig_width=10, font_size=12, save_path=None):
    """Parse a ThreadProbe log, apply calibration offset, replot scan chart.

    Args:
        log_path:    path to *_ThreadProbe*.log
        gpu_step_s:  corrected t_gpu value (overrides the one in the log)
        calibration: offset added to every t_cpu value (can be negative)
        fig_width:   figure width in inches
        font_size:   base font size
        save_path:   if set, save PDF here; otherwise derive from log_path
    """
    import re

    # --- parse Step 2 scan table ---
    with open(log_path) as f:
        lines = f.readlines()

    # find t_commit
    t_commit = None
    for line in lines:
        m = re.search(r't_commit\s*=\s*([\d.]+)s', line)
        if m:
            t_commit = float(m.group(1))

    # parse scan table (header: " consumer  generator   P   gen/P      t_cpu")
    results = []
    in_scan = False
    for line in lines:
        if 'consumer' in line and 'generator' in line and 'gen/P' in line and 'Step 2' not in line:
            in_scan = True
            continue
        if in_scan and line.strip().startswith('-'):
            continue
        if in_scan and line.strip() == '':
            in_scan = False
            continue
        if in_scan:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    results.append({
                        'consumer':  int(parts[0]),
                        'generator': int(parts[1]),
                        'P':         int(parts[2]),
                        'gen_per':   int(parts[3]),
                        't_cpu':     float(parts[4].rstrip('s')),
                    })
                except ValueError:
                    continue

    if not results:
        raise RuntimeError(f"No scan data found in {log_path}")

    # apply calibration
    for r in results:
        r['t_cpu'] += calibration

    best = min(results, key=lambda r: r['t_cpu'])
    t_train = gpu_step_s

    # N_min
    if t_commit and best['t_cpu'] < t_train:
        slack = t_train - best['t_cpu']
        N_min = max(1, math.ceil(t_commit / slack))
    else:
        N_min = None

    # print summary
    cal_str = f", calibration={calibration:+.3f}s" if calibration else ""
    print(f"Replot: {log_path}")
    print(f"  t_gpu={t_train:.3f}s{cal_str}")
    print(f"  Best: consumer={best['consumer']}, P={best['P']}, "
          f"t_cpu={best['t_cpu']:.3f}s")
    if N_min is not None:
        print(f"  N_min = ceil({t_commit:.3f} / {t_train - best['t_cpu']:.3f}) = {N_min}")

    # --- plot ---
    _setup_rcparams(font_size)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width / 1.618))
    xs = [r['consumer'] for r in results]
    ys = [r['t_cpu']    for r in results]
    ax.plot(xs, ys, 'o-', color='#1f77b4', markersize=8, linewidth=2,
            label='t_cpu (measured)')
    ax.plot([best['consumer']], [best['t_cpu']], marker='*', markersize=18,
            color='gold', markeredgecolor='black', markeredgewidth=1.5,
            linestyle='None',
            label=f"best: threads_cons={best['consumer']}, threads_gen={best['gen_per']}, t_cpu={best['t_cpu']:.3f}s (P={best['P']})")
    ax.axhline(t_train, color='red', ls='--', linewidth=2,
               label=f't_gpu={t_train:.3f}s')
    # # annotate best with t_cpu value
    # ax.annotate(f"t_cpu={best['t_cpu']:.3f}s\nP={best['P']}",
    #             xy=(best['consumer'], best['t_cpu']),
    #             xytext=(12, -20), textcoords='offset points',
    #             ha='left', fontsize=font_size + 1, fontweight='bold',
    #             color='#d62728',
    #             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5))
    for r in results:
        ax.annotate(f"P={r['P']}",
                    xy=(r['consumer'], r['t_cpu']),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=font_size, fontweight='bold')
    ax.set_xlabel('consumer (threads)')
    ax.set_ylabel('t_cpu (s)')
    title = 'Concurrent fine scan'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = log_path.replace('.log', '_replot.pdf')
    fig.savefig(save_path, bbox_inches='tight')
    print(f"[plot] saved → {save_path}")
    plt.show()
    plt.close(fig)

    return {
        'results': results,
        'best': best,
        'N_min': N_min,
        't_commit': t_commit,
    }


# =============================================================================
# Top-level entry
# =============================================================================
def run_probe(
    model_name,
    gpu_step_s,
    *,
    output_dir,
    optimizer='sgd',
    reserve_cores=1,
    core_start=1,
    core_stop=None,
    core_step=3,
    n_warmup=1,
    n_measure=3,
    plateau_threshold=1.10,
    consumer_step=5,
    test_n_steps=3,
    test_n_warmup=1,
    fig_width=10,
    font_size=12,
    rng_device='zo_rng',
):
    """Load model, run all 3 steps, save log + 2 PDFs.

    Args:
        optimizer: 'sgd' or 'adam'. Adam mode builds m/v buffers
                   so t_update and t_commit reflect the real Adam cost.

    Outputs (all in `output_dir`, named with a path-safe model name):
        <MODEL>_ThreadProbe_<opt>.log          — full stdout mirror
        <MODEL>_ThreadProbe_<opt>_curves.pdf   — Step 1 curves
        <MODEL>_ThreadProbe_<opt>_scan.pdf     — Step 2 fine scan
    """
    from transformers import AutoModelForCausalLM

    assert optimizer in ('sgd', 'adam'), f"optimizer must be 'sgd' or 'adam', got {optimizer!r}"

    os.makedirs(output_dir, exist_ok=True)
    safe = model_name.replace('/', '_')
    log_path   = os.path.join(output_dir, f'{safe}_ThreadProbe_{optimizer}.log')
    curves_pdf = os.path.join(output_dir, f'{safe}_ThreadProbe_{optimizer}_curves.pdf')
    scan_pdf   = os.path.join(output_dir, f'{safe}_ThreadProbe_{optimizer}_scan.pdf')

    orig_stdout = sys.stdout
    log_file = open(log_path, 'w')
    sys.stdout = _Tee(orig_stdout, log_file)

    try:
        print(f"{'=' * 60}")
        print(f"ThreadProbe: {model_name}  (optimizer={optimizer})")
        print(f"{'=' * 60}")
        print(f"  output_dir={output_dir}")
        print(f"  log={log_path}")

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        state = {n: p.data.cpu().clone()
                 for n, p in model.named_parameters() if p.requires_grad}
        param_names = list(state.keys())
        model_bytes = sum(p.numel() * p.element_size() for p in state.values())
        del model

        # Build adam_state if needed
        adam_state = None
        if optimizer == 'adam':
            adam_state = {
                't': 0,
                'betas': (0.9, 0.999),
                'adam_eps': 1e-8,
                'm': {name: torch.zeros_like(state[name]) for name in param_names},
                'v': {name: torch.zeros_like(state[name]) for name in param_names},
            }
            # Extend state dict with m/v so t_commit measures realistic 3x size
            for name in param_names:
                state[f"_adam_m_{name}"] = adam_state['m'][name]
                state[f"_adam_v_{name}"] = adam_state['v'][name]
            adam_extra_bytes = model_bytes * 2
            print(f"\nModel: {len(param_names)} params, {model_bytes / 1e9:.2f} GB"
                  f"  (+ adam m/v: {adam_extra_bytes / 1e9:.2f} GB,"
                  f" total: {(model_bytes + adam_extra_bytes) / 1e9:.2f} GB)")
        else:
            print(f"\nModel: {len(param_names)} params, {model_bytes / 1e9:.2f} GB")

        t_train = gpu_step_s
        C = len(os.sched_getaffinity(0)) - reserve_cores
        print(f"C = {C}, t_train = {t_train:.3f}s")

        # Step 1
        curves = measure_curves(
            state, param_names, C=C, t_train=t_train, rng_device=rng_device,
            core_start=core_start, core_stop=core_stop, core_step=core_step,
            n_warmup=n_warmup, n_measure=n_measure,
            plateau_threshold=plateau_threshold,
            adam_state=adam_state,
        )

        # Step 2
        results, best = fine_search(
            curves, state, param_names, C=C,
            consumer_step=consumer_step,
            test_n_steps=test_n_steps, test_n_warmup=test_n_warmup,
            rng_device=rng_device,
            adam_state=adam_state,
        )

        t_commit = curves['t_commit']

        # Step 3: recommendation + N_min
        print(f"\n{'=' * 60}")
        print(f"Final Recommendation  (optimizer={optimizer})")
        print(f"{'=' * 60}")
        print(f"  t_cpu = {best['t_cpu']:.3f}s   (measured, concurrent)")
        print(f"  t_gpu = {t_train:.3f}s")

        if best['t_cpu'] < t_train:
            slack = t_train - best['t_cpu']
            N_min = max(1, math.ceil(t_commit / slack))
            print(f"  t_cpu < t_gpu → shadow keeps up")
            print(f"  N_min = ceil(t_commit={t_commit:.3f}s / slack={slack:.3f}s) = {N_min}")
        else:
            lag = best['t_cpu'] / t_train - 1
            N_min = None
            print(f"  t_cpu >= t_gpu → LAGGY by {lag:+.0%}")

        print(f"\n  Env vars:")
        print(f"    SHADOW_PIPELINE_WORKERS={best['P']}")
        print(f"    SHADOW_CONSUMER_THREADS={best['consumer']}")
        print(f"    SHADOW_GENERATOR_THREADS={best['gen_per']}   # cores per producer")
        print(f"    SHADOW_RESERVE_THREADS={reserve_cores}")
        if N_min is not None:
            print(f"    SHADOW_COMMIT_INTERVAL={N_min}")
        print(f"{'=' * 60}")

        # Plots
        plot_curves(curves, t_train,
                    fig_width=fig_width, font_size=font_size,
                    save_path=curves_pdf)
        plot_scan(results, best, t_train,
                  fig_width=fig_width, font_size=font_size,
                  save_path=scan_pdf)

        return {
            'curves': curves,
            'scan': results,
            'best': best,
            'N_min': N_min,
            'optimizer': optimizer,
            'log': log_path,
            'plots': {'curves': curves_pdf, 'scan': scan_pdf},
        }
    except Exception:
        traceback.print_exc()
        raise
    finally:
        sys.stdout = orig_stdout
        log_file.close()
