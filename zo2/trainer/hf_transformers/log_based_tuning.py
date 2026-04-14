import json
import logging
import multiprocessing as mp
import os
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

DEFAULT_ZO_SHM_DIR = "/dev/shm/zo_ckpt"


def _benchmark_curves_worker(shared_tensors, param_names, rng_device, C,
                             n_warmup, n_measure, zo_eps, adam_state,
                             core_points, result_dict,
                             measure_commit, commit_n_warmup,
                             commit_n_measure, commit_dir):
    """Measure sparse t_gen(c) and t_update(n) curves in a child process."""
    import torch

    from .log_based_replay import (
        _apply_single_update_with_pregenerated_z,
        _generate_z_for_one_step,
    )
    from .log_based_utils import _atomic_save_state_dict_safetensors

    torch.set_num_interop_threads(1)

    try:
        n_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cores = os.cpu_count() or 64

    _zo_rng = None
    if rng_device == "zo_rng":
        import zo_rng as _zo_rng

    state = OrderedDict()
    for name in param_names:
        state[name] = shared_tensors[name].clone()

    def _median(xs):
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    C_max = min(C, n_cores)
    points = sorted(set(c for c in core_points if 1 <= c < C_max))
    print(f"[BenchCurves] pid={os.getpid()} cores={n_cores} C={C_max} "
          f"points={len(points)} range=[{points[0]}..{points[-1]}] "
          f"warmup={n_warmup} measure={n_measure}", flush=True)

    # Adaptive measurement: if a single run is slow (>SLOW_THRESHOLD_S),
    # skip repeat measurements to save time.
    SLOW_THRESHOLD_S = 20.0

    t_gen_curve = {}
    for idx, c in enumerate(points):
        if _zo_rng is not None:
            _zo_rng.set_num_threads(c)
        else:
            torch.set_num_threads(c)

        times = []
        total_iters = n_warmup + n_measure
        for i in range(total_iters):
            seed = 1000000 + i
            t0 = time.monotonic()
            z = _generate_z_for_one_step(seed, param_names, state, rng_device)
            t1 = time.monotonic()
            if i >= n_warmup:
                times.append(t1 - t0)
            del z
            # Stop early if the first measured run is slow.
            if len(times) == 1 and times[0] > SLOW_THRESHOLD_S:
                break

        t_gen_curve[c] = _median(times)
        print(f"[BenchCurves] t_gen(generator={c}) = {t_gen_curve[c]:.3f}s  "
              f"[{idx+1}/{len(points)}] (n={len(times)})", flush=True)

    _prev_aten = torch.get_num_threads()
    torch.set_num_threads(C_max)
    if _zo_rng is not None:
        _zo_rng.set_num_threads(C_max)
    z_pregenerated = _generate_z_for_one_step(42, param_names, state, rng_device)
    torch.set_num_threads(_prev_aten)
    print(f"[BenchCurves] z pre-generated for t_update measurement", flush=True)

    t_update_curve = {}
    dummy_update = {'seed': 42, 'grad': 1e-4, 'lr': 1e-5, 'wd': 0.01, 'zo_eps': zo_eps}

    for idx, n in enumerate(points):
        torch.set_num_threads(n)

        times = []
        total_iters = n_warmup + n_measure
        for i in range(total_iters):
            t0 = time.monotonic()
            _apply_single_update_with_pregenerated_z(
                state, dummy_update, param_names, z_pregenerated,
                default_zo_eps=zo_eps,
                simulate_perturbation=True,
                zo2_mode=False,
                adam_state=adam_state,
            )
            t1 = time.monotonic()
            if i >= n_warmup:
                times.append(t1 - t0)
            # Stop early if the first measured run is slow.
            if len(times) == 1 and times[0] > SLOW_THRESHOLD_S:
                break

        t_update_curve[n] = _median(times)
        print(f"[BenchCurves] t_update(consumer={n}) = {t_update_curve[n]:.3f}s  "
              f"[{idx+1}/{len(points)}] (iters={len(times)})", flush=True)

    del z_pregenerated

    t_update_min = min(t_update_curve.values())
    plateau_threshold = 1.10
    plateau_points = sorted(
        n for n in t_update_curve
        if t_update_curve[n] < t_update_min * plateau_threshold
    )
    n_low, n_high = plateau_points[0], plateau_points[-1]
    print(f"[BenchCurves] t_update plateau: consumer=[{n_low}, {n_high}]", flush=True)

    t_commit = 0.0
    if measure_commit:
        effective_commit_dir = commit_dir if os.path.isdir(commit_dir) else "/tmp"
        commit_path = os.path.join(
            effective_commit_dir,
            f"zo_shadow_calib_{os.getpid()}.safetensors",
        )
        commit_times = []
        total_commit_iters = max(1, commit_n_warmup + commit_n_measure)
        for i in range(total_commit_iters):
            t0 = time.monotonic()
            _atomic_save_state_dict_safetensors(
                state,
                commit_path,
                metadata={"base_step": 0, "committed_step": 0},
            )
            t1 = time.monotonic()
            if i >= commit_n_warmup:
                commit_times.append(t1 - t0)
            try:
                os.remove(commit_path)
            except FileNotFoundError:
                pass
        t_commit = _median(commit_times)
        print(
            f"[BenchCurves] t_commit(path={effective_commit_dir}) = "
            f"{t_commit:.3f}s (warmup={commit_n_warmup}, measure={commit_n_measure})",
            flush=True,
        )

    result_dict['t_gen_json'] = json.dumps({str(k): v for k, v in t_gen_curve.items()})
    result_dict['t_update_json'] = json.dumps({str(k): v for k, v in t_update_curve.items()})
    result_dict['n_cores'] = n_cores
    result_dict['C_max'] = C_max
    result_dict['n_low'] = n_low
    result_dict['n_high'] = n_high
    result_dict['t_commit'] = t_commit


def _interp_curve(curve_dict, x):
    """Linear interpolation on a sparse {int: float} curve."""
    keys = sorted(curve_dict.keys())
    if x in curve_dict:
        return curve_dict[x]
    if x <= keys[0]:
        return curve_dict[keys[0]]
    if x >= keys[-1]:
        return curve_dict[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= x <= keys[i + 1]:
            x0, x1 = keys[i], keys[i + 1]
            y0, y1 = curve_dict[x0], curve_dict[x1]
            return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return curve_dict[keys[-1]]


def _normalize_commit_intervals(commit_interval, commit_intervals=None):
    """Return a de-duplicated ordered list of positive commit intervals."""
    raw_values = commit_intervals if commit_intervals is not None else [commit_interval]
    values = []
    seen = set()
    for raw in raw_values:
        interval = max(1, int(raw))
        if interval in seen:
            continue
        seen.add(interval)
        values.append(interval)
    if not values:
        raise ValueError("At least one commit interval is required")
    return values


def optimize_thread_allocation(t_gen_curve, t_update_curve, C, t_train,
                               n_sat_range=None,
                               t_commit=0.0, commit_interval=1):
    """P-first search for optimal (c, P) minimizing pipeline step time."""
    pareto = []
    all_configs = []
    commit_interval = max(1, int(commit_interval))
    t_commit_avg = t_commit / commit_interval

    # t_cpu = t_gen/P + t_update  (no commit — commit is amortized separately)
    # best = minimize t_cpu. P is unbounded above (up to C).
    for P in range(1, C + 1):
        best_t = float('inf')
        best_cfg = None

        for c in range(1, C // P + 1):
            c_cons = C - P * c
            if c_cons < 1:
                continue
            if n_sat_range is not None:
                if c_cons < n_sat_range[0] or c_cons > n_sat_range[1]:
                    continue

            t_gen_P = _interp_curve(t_gen_curve, c) / P
            t_upd = _interp_curve(t_update_curve, c_cons)
            t_cpu = t_gen_P + t_upd

            cfg = {
                'c': c, 'P': P, 'c_cons': c_cons,
                't_cpu': t_cpu,
                't_gen_P': t_gen_P, 't_update_val': t_upd,
                't_commit_val': t_commit,
            }
            all_configs.append(cfg)

            if t_cpu < best_t:
                best_t = t_cpu
                best_cfg = cfg

        if best_cfg:
            pareto.append(best_cfg)

    if not pareto:
        raise ValueError("No valid (c, P) configuration found")

    recommended = min(pareto, key=lambda p: p['t_cpu'])

    return {
        'pareto': pareto,
        'recommended': recommended,
        'all_configs': all_configs,
        'C': C,
        't_train': t_train,
        'best_generator': recommended['c'],
        'best_P': recommended['P'],
        'best_consumer': recommended['c_cons'],
        'best_t_cpu': recommended['t_cpu'],
        'best_t_gen_P': recommended['t_gen_P'],
        'best_t_update_val': recommended['t_update_val'],
        'best_t_commit_val': recommended['t_commit_val'],
    }


def calibrate_producer_consumer(state, param_names, rng_device="zo_rng",
                                C=None, t_train=None, dataloader_num_workers=0,
                                n_warmup=5, n_measure=8,
                                zo_eps=1e-3, adam_state=None,
                                core_start=1, core_stop=None, core_step=1,
                                commit_interval=1, commit_intervals=None,
                                measure_commit=True,
                                commit_n_warmup=1, commit_n_measure=2,
                                commit_dir=DEFAULT_ZO_SHM_DIR):
    """Benchmark t_gen/t_update curves and find optimal (c, P, c_cons)."""
    if t_train is None:
        raise ValueError("t_train (GPU step time in seconds) is required")
    commit_interval_values = _normalize_commit_intervals(commit_interval, commit_intervals)

    for k in state:
        if state[k].device.type != 'cpu':
            state[k] = state[k].cpu()

    try:
        n_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cores = os.cpu_count() or 64

    if C is None:
        C = n_cores - 1 - dataloader_num_workers
    C = max(3, C)

    _stop = core_stop if core_stop is not None else C - 1
    core_points = list(range(core_start, _stop + 1, core_step))
    if _stop not in core_points and _stop >= core_start:
        core_points.append(_stop)
    logger.info(f"[CalibratePC] core_points: {len(core_points)} points, "
                f"range=[{core_points[0]}..{core_points[-1]}], step={core_step}")

    shared_state = OrderedDict()
    for name in param_names:
        shared_state[name] = state[name].clone()

    _old_env = {}
    _thread_env = {
        'OMP_NUM_THREADS': str(n_cores),
        'OMP_WAIT_POLICY': 'passive',
        'GOMP_SPINCOUNT': '0',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'KMP_BLOCKTIME': '0',
    }
    for k, v in _thread_env.items():
        _old_env[k] = os.environ.get(k)
        os.environ[k] = v

    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    result_dict = manager.dict()

    n_points = len(core_points)
    timeout_s = max(300, n_points * 120 + 120)

    p = ctx.Process(
        target=_benchmark_curves_worker,
        args=(shared_state, param_names, rng_device, C,
              n_warmup, n_measure, zo_eps, adam_state,
              core_points, result_dict,
              measure_commit, commit_n_warmup,
              commit_n_measure, commit_dir),
        daemon=True,
    )
    logger.info(f"[CalibratePC] Spawning benchmark worker: C={C}, t_train={t_train:.3f}s, "
                f"points={n_points}, timeout={timeout_s}s, "
                f"n_warmup={n_warmup}, n_measure={n_measure}, "
                f"commit_intervals={commit_interval_values}, measure_commit={measure_commit}")
    p.start()
    p.join(timeout=timeout_s)

    for k in _thread_env:
        if _old_env[k] is not None:
            os.environ[k] = _old_env[k]
        else:
            os.environ.pop(k, None)

    if p.is_alive():
        logger.error("[CalibratePC] Worker timed out, killing...")
        p.kill()
        p.join(timeout=10)
        manager.shutdown()
        raise RuntimeError(f"calibrate_producer_consumer: worker timed out after {timeout_s}s")
    if p.exitcode != 0:
        manager.shutdown()
        raise RuntimeError(f"calibrate_producer_consumer: worker crashed with exitcode={p.exitcode}")

    if 't_gen_json' not in result_dict:
        manager.shutdown()
        raise RuntimeError("calibrate_producer_consumer: worker produced no results")
    t_gen_curve = {int(k): v for k, v in json.loads(result_dict['t_gen_json']).items()}
    t_update_curve = {int(k): v for k, v in json.loads(result_dict['t_update_json']).items()}
    n_low = int(result_dict.get('n_low', 1))
    n_high = int(result_dict.get('n_high', C - 1))
    t_commit = float(result_dict.get('t_commit', 0.0))
    manager.shutdown()

    logger.info(f"[CalibratePC] t_update plateau: n=[{n_low}, {n_high}]")

    scan_results = []
    selected_interval = None
    selected_opt = None
    for interval in commit_interval_values:
        opt = optimize_thread_allocation(
            t_gen_curve,
            t_update_curve,
            C,
            t_train,
            n_sat_range=(n_low, n_high),
            t_commit=t_commit,
            commit_interval=interval,
        )
        scan_entry = {
            'commit_interval': interval,
            'recommended': opt['recommended'],
            'best_generator': opt['best_generator'],
            'best_P': opt['best_P'],
            'best_consumer': opt['best_consumer'],
            'best_t_cpu': opt['best_t_cpu'],
            'best_t_gen_P': opt['best_t_gen_P'],
            'best_t_update_val': opt['best_t_update_val'],
            'best_t_commit_val': opt['best_t_commit_val'],
        }
        scan_results.append(scan_entry)
        if selected_opt is None or opt['best_t_cpu'] < selected_opt['best_t_cpu']:
            selected_interval = interval
            selected_opt = opt

    opt = selected_opt

    rec = opt['recommended']
    import math

    print(f"\n{'='*60}")
    print(f"Producer-Consumer Optimization (C={C}, t_gpu={t_train:.3f}s)")
    print(f"{'='*60}")
    print(f"{'P':>3} {'generator':>10} {'consumer':>9} {'t_cpu':>10}")
    print(f"{'-'*38}")
    for row in opt['pareto']:
        marker = ' <-- best' if row is rec else ''
        print(f"{row['P']:>3} {row['c']:>10} {row['c_cons']:>9} "
              f"{row['t_cpu']:>9.3f}s{marker}")

    print(f"\n  t_commit = {t_commit:.3f}s")

    # Auto-derive minimum commit interval N from: t_cpu + t_commit/N <= t_gpu
    if rec['t_cpu'] < t_train:
        slack = t_train - rec['t_cpu']
        if slack > 0:
            N_min = max(1, math.ceil(t_commit / slack))
            print(f"  t_cpu ({rec['t_cpu']:.3f}s) < t_gpu ({t_train:.3f}s) "
                  f"→ N_min = ceil({t_commit:.3f} / {slack:.3f}) = {N_min}")
        else:
            N_min = 1
            print(f"  t_cpu == t_gpu → N_min = 1")
    else:
        N_min = None
        print(f"  t_cpu ({rec['t_cpu']:.3f}s) >= t_gpu ({t_train:.3f}s) → laggy")

    print(f"\n  Recommended: P={rec['P']}, generator={rec['c']}, consumer={rec['c_cons']}, "
          f"t_cpu={rec['t_cpu']:.3f}s")
    print(f"{'='*60}\n")

    return {
        't_gen_curve': t_gen_curve,
        't_update_curve': t_update_curve,
        'n_low': n_low,
        'n_high': n_high,
        't_commit': t_commit,
        'commit_interval': selected_interval,
        'commit_intervals': commit_interval_values,
        'scan_results': scan_results,
        'N_min': N_min,
        **opt,
    }
