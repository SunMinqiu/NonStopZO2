import logging
import os
import queue as queue_module
import threading
import time
from collections import OrderedDict

import psutil
import torch

from .log_based_utils import (
    _atomic_save_state_dict_safetensors,
    _load_state_dict_safetensors_with_metadata,
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


def _commit_shadow_state(state_dict, replica_path, base_step, committed_step, tied_groups=None):
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


def _rebase_working_state(anchor_path, tied_groups, _logger):
    rebased, base_step, committed_step = _load_shadow_replica(anchor_path, tied_groups=tied_groups)
    if tied_groups:
        _tie_state_dict_inplace(rebased, tied_groups)
    _logger.info(f"[Shadow] Rebased from {anchor_path} at step {committed_step}")
    return rebased, base_step, committed_step


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
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    _logger = logging.getLogger(__name__ + ".shadow_process")

    torch.set_num_interop_threads(1)

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
        f"  commit_interval={commit_interval}"
    )
    _thread_snapshot("Shadow BOOT", _logger, detail=True)

    adam_state = None
    if adam_config is not None:
        adam_state = {
            "m": {},
            "v": {},
            "t": 0,
            "betas": adam_config["betas"],
            "adam_eps": adam_config["adam_eps"],
        }
        _logger.info(f"[Shadow Process] Adam state initialized: betas={adam_config['betas']}")

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
        )


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
):
    from . import log_based_checkpoint as _bdc

    _logger.info(f"[Shadow Process] Running in serial mode (params={len(param_names)}, rng={rng_device})")

    working_state = _clone_working_state(initial_state, tied_groups)
    base_step = int(initial_base_step)
    committed_step = int(initial_committed_step)
    last_applied_step = committed_step
    pending_since_commit = 0

    while True:
        try:
            cmd = update_queue.get(timeout=0.05)
        except queue_module.Empty:
            continue

        kind = cmd.get("cmd") if isinstance(cmd, dict) else None
        if kind == "stop":
            break
        if kind == "rebase":
            working_state, base_step, committed_step = _rebase_working_state(
                cmd["path"], tied_groups, _logger
            )
            last_applied_step = committed_step
            pending_since_commit = 0
            if adam_state is not None:
                adam_state["m"].clear()
                adam_state["v"].clear()
                adam_state["t"] = 0
            _commit_shadow_state(working_state, replica_path, base_step, committed_step, tied_groups=tied_groups)
            shadow_step_val.value = committed_step
            continue
        if kind != "update":
            continue

        update = cmd["update"]
        step = int(update["step"])
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
        )
        t_apply = time.time() - t0_apply
        del z_dict

        last_applied_step = step
        pending_since_commit += 1

        t_commit = 0.0
        if pending_since_commit >= commit_interval:
            t0_commit = time.time()
            committed_step = last_applied_step
            _commit_shadow_state(working_state, replica_path, base_step, committed_step, tied_groups=tied_groups)
            shadow_step_val.value = committed_step
            pending_since_commit = 0
            t_commit = time.time() - t0_commit

        _rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        _logger.info(
            f"[Shadow] step={step} grad={update['grad']:.6e} seed={step_seed} "
            f"| apply={t_apply * 1000:.0f}ms zgen={t_zgen * 1000:.0f}ms "
            f"commit={t_commit * 1000:.0f}ms pending={pending_since_commit} "
            f"| committed={shadow_step_val.value} RSS={_rss_gb:.1f}GB"
        )

    if last_applied_step > committed_step:
        _commit_shadow_state(working_state, replica_path, base_step, last_applied_step, tied_groups=tied_groups)
        shadow_step_val.value = last_applied_step
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
):
    from . import log_based_checkpoint as _bdc

    shadow_bytes = sum(initial_state[nm].numel() * initial_state[nm].element_size() for nm in param_names)
    _logger.info(f"[Shadow Pipeline] P={P} producers, shadow_copy={shadow_bytes / 1e9:.2f}GB")

    working_state = _clone_working_state(initial_state, tied_groups)
    base_step = int(initial_base_step)
    committed_step = int(initial_committed_step)
    consumer_step = committed_step + 1
    last_applied_step = committed_step
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

    def producer():
        try:
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
                        result_queue.put((step_idx, z_dict, update), timeout=0.05)
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

    threads = _restart_producers()
    pending_results = {}

    while True:
        try:
            while True:
                cmd = update_queue.get_nowait()
                kind = cmd.get("cmd") if isinstance(cmd, dict) else None
                if kind == "stop":
                    _stop_producers(threads)
                    if last_applied_step > committed_step:
                        _commit_shadow_state(
                            working_state,
                            replica_path,
                            base_step,
                            last_applied_step,
                            tied_groups=tied_groups,
                        )
                        shadow_step_val.value = last_applied_step
                    _logger.info("[Shadow Pipeline] Stopped")
                    return
                if kind == "rebase":
                    _stop_producers(threads)
                    with internal_lock:
                        internal_updates.clear()
                    pending_results.clear()
                    while not result_queue.empty():
                        try:
                            result_queue.get_nowait()
                        except queue_module.Empty:
                            break
                    update_available_event.clear()
                    working_state, base_step, committed_step = _rebase_working_state(
                        cmd["path"], tied_groups, _logger
                    )
                    last_applied_step = committed_step
                    consumer_step = committed_step + 1
                    next_step_to_assign[0] = consumer_step
                    pending_since_commit = 0
                    if adam_state is not None:
                        adam_state["m"].clear()
                        adam_state["v"].clear()
                        adam_state["t"] = 0
                    _commit_shadow_state(
                        working_state,
                        replica_path,
                        base_step,
                        committed_step,
                        tied_groups=tied_groups,
                    )
                    shadow_step_val.value = committed_step
                    threads = _restart_producers()
                    continue
                if kind == "update":
                    update = cmd["update"]
                    with internal_lock:
                        internal_updates[int(update["step"])] = update
                    update_available_event.set()
        except queue_module.Empty:
            pass

        if producer_error[0] is not None:
            raise producer_error[0]

        try:
            step_idx, z_dict, update = result_queue.get(timeout=0.05)
        except queue_module.Empty:
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
            )
            apply_ms = (time.monotonic() - t0_apply) * 1000
            del z_dict

            last_applied_step = consumer_step
            consumer_step += 1
            pending_since_commit += 1

            commit_ms = 0.0
            if pending_since_commit >= commit_interval:
                t0_commit = time.monotonic()
                committed_step = last_applied_step
                _commit_shadow_state(
                    working_state,
                    replica_path,
                    base_step,
                    committed_step,
                    tied_groups=tied_groups,
                )
                shadow_step_val.value = committed_step
                pending_since_commit = 0
                commit_ms = (time.monotonic() - t0_commit) * 1000

            _rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
            _logger.info(
                f"[Shadow] step={last_applied_step} grad={update['grad']:.6e} seed={update['seed']} "
                f"| apply={apply_ms:.0f}ms zgen={producer_timing['duration_ms']:.0f}ms "
                f"commit={commit_ms:.0f}ms pending={pending_since_commit} "
                f"| committed={shadow_step_val.value} RSS={_rss_gb:.1f}GB"
            )
