"""
Log-Based Checkpoint for ZO Training:
- Supports different checkpoint modes controlled by batch_size parameter:
  - batch_size = -1: Disabled (L0 baseline, uses default Trainer checkpoint)
  - batch_size = 0: Log-based (accumulate all updates from initial model)
  - batch_size >= 1: Full + Log (full checkpoint every N steps, log checkpoints in between)
    - Optional: enable_shadow for real-time shadow model on CPU (instant recovery)
- Controllable failure injection for testing

Environment variables:
  - FORCE_FSYNC=1: Force fsync after checkpoint writes
  - PARALLEL_RECOVERY=1: Enable pipelined producer-consumer replay recovery.
      P producer threads/streams generate z concurrently while 1 consumer
      applies updates serially via a ring buffer. Bitwise-exact with
      sequential replay.
  - PARALLEL_RECOVERY_WORKERS=P: Number of producers / ring buffer slots
      (default: 1). P>=2 needed for overlap. Use calibrate_producer_consumer()
      to determine optimal P.
  - CLOSEDFORM_RECOVERY=1: Enable closed-form parallel replay recovery.
      Unrolls the ZO-SGD recurrence into a sum of independent terms, enabling
      true data-parallel computation across W workers. Near-exact with serial
      replay (no perturbation simulation).
  - CLOSEDFORM_WORKERS=W: Number of parallel workers (default: 1).
  - CLOSEDFORM_PRECISION=mode: Precision mode for closed-form replay.
      "fp32" (all fp32), "fp16" (keep original dtype), "mixed" (default,
      accumulate in fp32, keep params in original dtype).
  - SHADOW_PIPELINE=1: Enable pipelined producer-consumer for the CPU shadow
      model during training. P producer threads pre-generate z tensors from
      logged seeds while 1 consumer thread applies updates using buffered z.
      Requires ENABLE_SHADOW=1. Falls back to serial shadow if not set.
      GIL is not a bottleneck: PyTorch ops and zo_rng release the GIL.
  - SHADOW_PIPELINE_WORKERS=P: Number of producer threads / ring buffer slots
      for the shadow pipeline (default: 2). Use calibrate_shadow_pipeline()
      to determine optimal P. Memory: P × model_size per z-buffer slot.
  - SHADOW_COMMIT_INTERVAL=N: Shadow commits a tmpfs replica every N updates
  (default: 1). Rebase always forces an immediate commit.
  - SHADOW_FLAT_COMMIT=1: Use preallocated flat tmpfs buffers for shadow
  commit/recover instead of writing a safetensors replica on every commit.
"""

import glob
import hashlib
import logging
import multiprocessing as mp
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import psutil
import torch
from transformers import TrainerCallback

from ...utils.logging_controls import (
    resource_log_enabled,
    time_log_enabled,
)
from .log_based_failure_injection import GPUFailureSimulator, format_gpu_fail_steps
from .log_based_resume import load_log_based_checkpoint
from .log_based_replay import (
    _get_and_clear_replay_adam_state,
    _replay_updates_on_state,
)
from .log_based_shadow import (
    _build_adam_flat_layout,
    _build_shadow_flat_layout,
    _shadow_flat_meta_paths,
    _cleanup_rebase_payload_flat,
    _init_shadow_bundle_flat_storage,
    _load_shadow_bundle_flat,
    _load_shadow_replica,
    _shadow_process_main,
    _write_rebase_payload_flat,
)
from .log_based_utils import (
    _atomic_save_state_dict_safetensors,
    _clone_state_dict_to_cpu,
    _log_adam_brief,
    _log_adam_checksums,
    _log_adam_exact_compare,
    _log_adam_exact_fingerprint,
    _log_state_checksums,
    _log_state_exact_compare,
    _log_state_exact_fingerprint,
    _detect_tied_weights,
    _fsync_file,
    _ensure_zo_shm_dir,
    _step_diag_enabled,
    _step_exact_enabled,
    _system_stats,
    _thread_debug_enabled,
    _thread_snapshot,
    _tie_state_dict_inplace,
)

logger = logging.getLogger(__name__)
LOG_METADATA_NAME = "log_metadata.pt"


@dataclass
class AnchorPublishTask:
    step: int
    base_checkpoint_state: OrderedDict
    adam_state: dict | None
    base_pending_seed: int
    created_at: float


class LogBasedCheckpointCallback(TrainerCallback):
    """
    Log-based checkpoint callback.

    Modes controlled by batch_size parameter:
    - batch_size = -1: Disabled, uses default Trainer checkpoint
    - batch_size = 0: Log-based - accumulate all updates from initial model
    - batch_size >= 1: Full + Log - full checkpoint every N steps, log checkpoints in between
      - enable_shadow: Real-time shadow model on CPU for instant recovery (only for batch_size>=1)
      - instant_recover: Recover from shadow on GPU failure (requires enable_shadow)
    """

    def __init__(self, batch_size=0, enable_shadow=False, instant_recover=False):
        """
        Args:
            batch_size: Checkpoint mode
                -1: Disabled
                0: Log-based (accumulate all updates)
                >=1: Full + Log (full checkpoint every N steps, log in between)
            enable_shadow: Enable real-time shadow model on CPU (effective for batch_size>=0, forced off for -1)
            instant_recover: Instantly recover from shadow on GPU failure (requires enable_shadow)
        """
        self.batch_size = batch_size
        # Shadow is allowed for batch_size>=0 (=0: log-based, >=1: full+log).
        # Only batch_size=-1 (disabled mode) forces shadow off.
        self.enable_shadow = enable_shadow if batch_size >= 0 else False
        self.instant_recover = instant_recover if self.enable_shadow else False

        # Base checkpoint state (CPU)
        self.base_checkpoint_state: OrderedDict = None
        self.active_base_step: int = 0
        self.base_checkpoint_path: str = None
        self.base_checkpoint_step: int = 0
        self.is_first_save = True
        self.save_count = 0
        self.initial_cpu_state_dict: OrderedDict = None
        self.initial_cpu_state_source: str = None
        self.initial_cpu_state_keys = None

        # Update history
        self.update_history = []
        self.update_lock = threading.Lock()
        self.last_saved_step = 0
        self.current_step = 0

        # Pending grad: the last computed projected_grad that hasn't been applied yet.
        # Saved in checkpoint so it can be restored on resume for the first step's zo_update.
        self._pending_grad = 0.0
        # Pending seed: the zo_random_seed from the step that computed _pending_grad.
        # Needed by ZO2 to reconstruct last_rstate (CUDA RNG state) on resume.
        self._pending_seed = 0
        # Base pending seed: the seed from the step that became the current base checkpoint.
        # Needed by ZO2 replay: the first entry after a base uses this as prev_seed for z generation.
        self._base_pending_seed = 0
        self._active_base_pending_seed = 0

        # Shadow model (real-time tracking)
        self.shadow_step = 0
        self.shadow_base_step = 0
        self.shadow_adam_state = None  # Adam m/v/t on CPU (None = SGD mode)

        # Multiprocessing shadow (zero contention with training)
        self.shadow_process = None       # mp.Process
        self.update_queue = None         # mp.Queue: training → shadow
        self.shadow_step_val = None      # mp.Value('i', lock=False): latest committed step
        self.shm_dir = None
        self.shadow_replica_path = None  # tmpfs committed shadow replica
        self.anchor_latest_path = None   # tmpfs async/sync anchor latest
        self.use_shadow_flat_commit = False
        self.shadow_flat_header_path = None
        self.shadow_flat_buffer_paths = ()
        self.shadow_flat_adam_m_buffer_paths = ()
        self.shadow_flat_adam_v_buffer_paths = ()
        self.shadow_flat_storage = None
        self.rebase_payload_dir = None
        self._staged_rebase_payloads = set()
        self.shadow_commit_interval = 1
        self._last_shadow_rebased_anchor_step = -1
        self.anchor_publish_thread = None
        self.anchor_publish_condition = threading.Condition()
        self.anchor_publish_latest_task = None
        self.anchor_publish_inflight_step = None
        self.anchor_publish_completed_step = -1
        self.anchor_publish_stop = False
        self.anchor_publish_failed = None

        # Output directory
        self.output_dir = None

        # Trainer reference
        self.trainer = None
        self._hook_registered = False

        # GPU failure simulator
        self.failure_simulator = GPUFailureSimulator()
        self.failure_simulator.callback = self

        # Model dtype (detected at training start for replay consistency)
        self.model_dtype = None

        # Trainable parameter names (from named_parameters() with requires_grad=True)
        # Used for replay consistency: ensures same iteration order and RNG sequence as training
        self._trainable_param_names = None

        # Timing statistics
        self.timing_stats = {
            'recoveries': [],
            'shadow_send_times': [],
            'checkpoint_total_times': [],
        }
        self.disk_log_save_count = 0
        self.full_anchor_save_count = 0

    def _run_hash(self):
        if not self.output_dir:
            return "no_output"
        return hashlib.md5(self.output_dir.encode()).hexdigest()[:8]

    def _shadow_replica_path(self):
        return os.path.join(self.shm_dir, f"zo_shadow_latest_{self._run_hash()}.safetensors")

    def _shadow_flat_header_storage_path(self):
        return os.path.join(self.shm_dir, f"zo_shadow_latest_{self._run_hash()}.flat.header.json")

    def _shadow_flat_buffer_storage_paths(self):
        stem = os.path.join(self.shm_dir, f"zo_shadow_latest_{self._run_hash()}.flat")
        return (f"{stem}.bin",)

    def _shadow_flat_adam_m_buffer_storage_paths(self):
        stem = os.path.join(self.shm_dir, f"zo_shadow_latest_{self._run_hash()}.flat.adam_m")
        return (f"{stem}.bin",)

    def _shadow_flat_adam_v_buffer_storage_paths(self):
        stem = os.path.join(self.shm_dir, f"zo_shadow_latest_{self._run_hash()}.flat.adam_v")
        return (f"{stem}.bin",)

    def _anchor_latest_path(self):
        return os.path.join(self.shm_dir, f"zo_anchor_latest_{self._run_hash()}.safetensors")

    def _rebase_payload_dir_path(self):
        return os.path.join(self.shm_dir, f"zo_rebase_payload_{self._run_hash()}")

    def _rebase_payload_header_path(self, step):
        return os.path.join(self.rebase_payload_dir, f"step_{int(step)}.header.json")

    def _build_flat_storage_descriptor(
        self,
        state_dict,
        *,
        header_path,
        buffer_paths,
        adam_m_buffer_paths=(),
        adam_v_buffer_paths=(),
        adam_state=None,
    ):
        meta_paths = _shadow_flat_meta_paths(header_path)
        descriptor = {
            "enabled": True,
            "layout": _build_shadow_flat_layout(
                state_dict,
                tied_groups=getattr(self, "_tied_weight_groups", []),
            ),
            "header_path": header_path,
            "state_meta_path": meta_paths["state_meta_path"],
            "adam_meta_path": meta_paths["adam_meta_path"],
            "buffer_paths": tuple(buffer_paths),
            "has_adam": bool(adam_state is not None),
        }
        if adam_state is not None:
            descriptor["adam_layout"] = _build_adam_flat_layout(
                state_dict,
                self._trainable_param_names or [],
            )
            descriptor["adam_m_buffer_paths"] = tuple(adam_m_buffer_paths)
            descriptor["adam_v_buffer_paths"] = tuple(adam_v_buffer_paths)
        else:
            descriptor["adam_layout"] = {"entries": [], "total_bytes": 0}
            descriptor["adam_m_buffer_paths"] = ()
            descriptor["adam_v_buffer_paths"] = ()
        return descriptor

    def _ensure_shadow_flat_storage(self, state_dict):
        if not self.use_shadow_flat_commit:
            return
        want_adam = bool(self.shadow_adam_state is not None)
        if self.shadow_flat_storage is None or self.shadow_flat_storage.get("has_adam", False) != want_adam:
            self.shadow_flat_storage = self._build_flat_storage_descriptor(
                state_dict,
                header_path=self.shadow_flat_header_path,
                buffer_paths=self.shadow_flat_buffer_paths,
                adam_m_buffer_paths=self.shadow_flat_adam_m_buffer_paths,
                adam_v_buffer_paths=self.shadow_flat_adam_v_buffer_paths,
                adam_state=self.shadow_adam_state,
            )

    def _shadow_storage_available(self):
        if not self.enable_shadow:
            return False
        if self.use_shadow_flat_commit:
            return bool(
                self.shadow_flat_header_path and
                os.path.exists(self.shadow_flat_header_path)
            )
        return bool(self.shadow_replica_path and os.path.exists(self.shadow_replica_path))

    def _shadow_secondary_keys(self):
        excluded = set()
        for group in getattr(self, "_tied_weight_groups", []) or []:
            for name in group[1:]:
                excluded.add(name)
        return excluded

    def _shadow_commit_interval_value(self):
        try:
            return max(1, int(os.environ.get("SHADOW_COMMIT_INTERVAL", "1")))
        except ValueError:
            return 1

    def _publish_anchor_latest(self, state_dict, step, *, adam_state=None):
        if adam_state is None:
            adam_state = self.shadow_adam_state
        if self.use_shadow_flat_commit:
            os.makedirs(self.rebase_payload_dir, exist_ok=True)
            payload_path = self._rebase_payload_header_path(step)
            _write_rebase_payload_flat(
                state_dict,
                payload_path,
                base_step=step,
                committed_step=step,
                tied_groups=getattr(self, "_tied_weight_groups", []),
                adam_state=adam_state,
                param_names=self._trainable_param_names or list(state_dict.keys()),
            )
            with self.anchor_publish_condition:
                self._staged_rebase_payloads.add(payload_path)
            return payload_path
        if not self.anchor_latest_path:
            return None
        save_state = OrderedDict(
            (key, value)
            for key, value in state_dict.items()
            if key not in self._shadow_secondary_keys()
        )
        _atomic_save_state_dict_safetensors(
            save_state,
            self.anchor_latest_path,
            metadata={
                "base_step": int(step),
                "committed_step": int(step),
            },
        )
        return self.anchor_latest_path

    def _queue_shadow_rebase(self, step, path=None):
        if self.update_queue is None:
            return
        try:
            payload = {
                "cmd": "rebase",
                "step": int(step),
                "path": path or self.anchor_latest_path,
            }
            self.update_queue.put_nowait(payload)
        except Exception as e:
            if not getattr(self, "_queue_error_logged", False):
                logger.warning(f"[LogBased] Failed to send rebase to shadow: {e}")
                self._queue_error_logged = True

    def _use_async_anchor_publisher(self):
        async_anchor = getattr(self, "_async_anchor", None) or getattr(self.trainer, "_async_anchor", None)
        return bool(
            self.enable_shadow and
            self.batch_size >= 1 and
            self.use_shadow_flat_commit and
            async_anchor is None
        )

    def _clone_shadow_adam_state(self, model):
        opt = getattr(model, 'opt', None)
        if opt is None:
            return None
        if not (hasattr(opt, 'get_adam_state') and hasattr(opt, 'betas') and hasattr(opt, 'adam_eps')):
            return None
        adam_state = opt.get_adam_state()
        return {
            'm': OrderedDict(
                (name, tensor.detach().to(device='cpu', dtype=torch.float32).clone())
                for name, tensor in adam_state.get('m', {}).items()
            ),
            'v': OrderedDict(
                (name, tensor.detach().to(device='cpu', dtype=torch.float32).clone())
                for name, tensor in adam_state.get('v', {}).items()
            ),
            't': int(adam_state.get('t', 0)),
            'betas': tuple(adam_state.get('betas', getattr(opt, 'betas', (0.9, 0.999)))),
            'adam_eps': float(adam_state.get('adam_eps', getattr(opt, 'adam_eps', 1e-8))),
        }

    def _check_anchor_publisher_health(self):
        if self.anchor_publish_failed is not None:
            raise RuntimeError(
                f"anchor publisher failed previously: {self.anchor_publish_failed}"
            )

    def _start_anchor_publisher(self):
        if not self._use_async_anchor_publisher():
            return
        if self.anchor_publish_thread is not None and self.anchor_publish_thread.is_alive():
            return
        self.anchor_publish_stop = False
        self.anchor_publish_failed = None
        self.anchor_publish_latest_task = None
        self.anchor_publish_inflight_step = None
        self.anchor_publish_completed_step = -1
        self.anchor_publish_thread = threading.Thread(
            target=self._anchor_publisher_main,
            name="anchor-publisher",
            daemon=True,
        )
        self.anchor_publish_thread.start()
        logger.info("[AnchorPublisher] Started async anchor publisher thread")

    def _stop_anchor_publisher(self, timeout_s=60.0):
        thread = self.anchor_publish_thread
        if thread is None:
            return
        with self.anchor_publish_condition:
            self.anchor_publish_stop = True
            self.anchor_publish_condition.notify_all()
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            logger.warning(
                f"[AnchorPublisher] Timed out waiting for async anchor publisher to stop after {timeout_s:.1f}s"
            )
        else:
            logger.info(
                f"[AnchorPublisher] Stopped (completed_step={self.anchor_publish_completed_step})"
            )
        self.anchor_publish_thread = None
        self._check_anchor_publisher_health()

    def _submit_anchor_publish_task(self, task: AnchorPublishTask):
        self._check_anchor_publisher_health()
        with self.anchor_publish_condition:
            previous = self.anchor_publish_latest_task
            if previous is not None and previous.step != task.step:
                logger.info(
                    f"[AnchorPublisher] Dropped stale pending anchor step={previous.step} newer_step={task.step}"
                )
            self.anchor_publish_latest_task = task
            self.anchor_publish_condition.notify_all()

    def _publish_anchor_task(self, task: AnchorPublishTask):
        t_total = time.time()
        payload_path = None
        publish_anchor_s = 0.0
        queue_rebase_s = 0.0
        t0 = time.time()
        payload_path = self._publish_anchor_latest(
            task.base_checkpoint_state,
            task.step,
            adam_state=task.adam_state,
        )
        publish_anchor_s = time.time() - t0
        if payload_path is not None:
            t0 = time.time()
            self._queue_shadow_rebase(task.step, path=payload_path)
            queue_rebase_s = time.time() - t0
            self._last_shadow_rebased_anchor_step = int(task.step)
        logger.info(
            f"[AnchorPublisher] step={task.step} publish_anchor={publish_anchor_s:.3f}s "
            f"queue_rebase={queue_rebase_s:.3f}s total={time.time() - t_total:.3f}s"
        )

    def _anchor_publisher_main(self):
        while True:
            with self.anchor_publish_condition:
                while not self.anchor_publish_stop and self.anchor_publish_latest_task is None:
                    self.anchor_publish_condition.wait()
                if self.anchor_publish_stop and self.anchor_publish_latest_task is None:
                    return
                task = self.anchor_publish_latest_task
                self.anchor_publish_latest_task = None
                self.anchor_publish_inflight_step = task.step
            try:
                self._publish_anchor_task(task)
            except Exception as exc:
                self.anchor_publish_failed = exc
                logger.exception(f"[AnchorPublisher] step={task.step} failed: {exc}")
                return
            finally:
                with self.anchor_publish_condition:
                    if self.anchor_publish_inflight_step == task.step:
                        self.anchor_publish_inflight_step = None
                    self.anchor_publish_completed_step = max(self.anchor_publish_completed_step, int(task.step))

    def _refresh_shadow_from_base(self, *, step=None, commit_now=True):
        if self.base_checkpoint_state is None:
            return
        step = int(self.active_base_step if step is None else step)
        self.shadow_base_step = step
        self.shadow_step = step
        if commit_now and self.enable_shadow:
            if self.use_shadow_flat_commit:
                self._ensure_shadow_flat_storage(self.base_checkpoint_state)
                _init_shadow_bundle_flat_storage(
                    self.base_checkpoint_state,
                    self.shadow_flat_storage,
                    step,
                    step,
                    tied_groups=getattr(self, "_tied_weight_groups", []),
                    adam_state=self.shadow_adam_state,
                )
            elif self.shadow_replica_path:
                save_state = OrderedDict(
                    (key, value)
                    for key, value in self.base_checkpoint_state.items()
                    if key not in self._shadow_secondary_keys()
                )
                _atomic_save_state_dict_safetensors(
                    save_state,
                    self.shadow_replica_path,
                    metadata={
                        "base_step": step,
                        "committed_step": step,
                    },
                )

    def _seed_for_step(self, step, fallback=None):
        default = self._base_pending_seed if fallback is None else fallback
        if step <= 0:
            return default
        with self.update_lock:
            for update in reversed(self.update_history):
                if int(update.get("step", -1)) <= step:
                    return int(update.get("seed", default))
        return default

    def _activate_base_state(self, state_dict, step, adam_state=None):
        if not isinstance(state_dict, OrderedDict):
            state_dict = OrderedDict(state_dict.items())
        if getattr(self, "_tied_weight_groups", None):
            _tie_state_dict_inplace(state_dict, self._tied_weight_groups)
        self.base_checkpoint_state = state_dict
        if adam_state is not None:
            self.shadow_adam_state = adam_state
        self.active_base_step = int(step)
        self._active_base_pending_seed = self._seed_for_step(self.active_base_step)

    def _log_shadow_storage_exact_compare(self, model, step):
        if not self.enable_shadow or model is None:
            return
        try:
            shadow_adam = None
            if self.use_shadow_flat_commit and self.shadow_flat_storage is not None:
                shadow_state, shadow_adam, _base_step, _committed_step = _load_shadow_bundle_flat(
                    self.shadow_flat_storage,
                    tied_groups=getattr(self, "_tied_weight_groups", []),
                )
            elif self.shadow_replica_path and os.path.exists(self.shadow_replica_path):
                shadow_state, _base_step, _committed_step = _load_shadow_replica(
                    self.shadow_replica_path,
                    tied_groups=getattr(self, "_tied_weight_groups", []),
                )
            else:
                return
        except Exception as exc:
            logger.warning(f"[STATE-EXACT] train_vs_shadow_storage step={step}: compare_failed={type(exc).__name__}")
            return

        live_state = OrderedDict(
            (name, tensor.detach())
            for name, tensor in model.state_dict().items()
            if torch.is_tensor(tensor)
        )
        _log_state_exact_compare(f"train_vs_shadow_storage step={step}", live_state, shadow_state)

        live_adam = None
        opt = getattr(model, "opt", None)
        if opt is not None and hasattr(opt, "get_adam_state") and hasattr(opt, "betas") and hasattr(opt, "adam_eps"):
            live_adam = self._clone_shadow_adam_state(model)
        if live_adam is not None or shadow_adam is not None:
            _log_adam_exact_compare(f"train_vs_shadow_storage step={step}", live_adam, shadow_adam)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Called at training start"""
        self.output_dir = args.output_dir
        self.shm_dir = _ensure_zo_shm_dir()
        self.shadow_replica_path = self._shadow_replica_path() if self.enable_shadow else None
        self.use_shadow_flat_commit = self.enable_shadow and os.environ.get("SHADOW_FLAT_COMMIT", "0") == "1"
        self.shadow_flat_header_path = (
            self._shadow_flat_header_storage_path() if self.use_shadow_flat_commit else None
        )
        self.shadow_flat_buffer_paths = (
            self._shadow_flat_buffer_storage_paths() if self.use_shadow_flat_commit else ()
        )
        self.shadow_flat_adam_m_buffer_paths = (
            self._shadow_flat_adam_m_buffer_storage_paths() if self.use_shadow_flat_commit else ()
        )
        self.shadow_flat_adam_v_buffer_paths = (
            self._shadow_flat_adam_v_buffer_storage_paths() if self.use_shadow_flat_commit else ()
        )
        if self.use_shadow_flat_commit and self.shadow_flat_header_path and os.path.exists(self.shadow_flat_header_path):
            meta_paths = _shadow_flat_meta_paths(self.shadow_flat_header_path)
            if not os.path.exists(meta_paths["state_meta_path"]) or not os.path.exists(meta_paths["adam_meta_path"]):
                logger.warning(
                    f"[LogBased] Removing legacy shadow flat snapshot without metadata at {self.shadow_flat_header_path}"
                )
                for path in (
                    self.shadow_flat_header_path,
                    *self.shadow_flat_buffer_paths,
                    *self.shadow_flat_adam_m_buffer_paths,
                    *self.shadow_flat_adam_v_buffer_paths,
                    meta_paths["state_meta_path"],
                    meta_paths["adam_meta_path"],
                ):
                    if path and os.path.exists(path):
                        try:
                            os.unlink(path)
                        except FileNotFoundError:
                            pass
        self.shadow_flat_storage = None
        self.anchor_latest_path = self._anchor_latest_path() if (self.batch_size >= 1 and not self.use_shadow_flat_commit) else None
        self.rebase_payload_dir = (
            self._rebase_payload_dir_path() if self.use_shadow_flat_commit and self.batch_size >= 1 else None
        )
        self._staged_rebase_payloads = set()
        if self.rebase_payload_dir and os.path.isdir(self.rebase_payload_dir):
            for payload_path in glob.glob(os.path.join(self.rebase_payload_dir, "*.header.json")):
                _cleanup_rebase_payload_flat(payload_path)
        self.shadow_commit_interval = self._shadow_commit_interval_value()

        if 'trainer' in kwargs:
            self.trainer = kwargs['trainer']

        # batch_size=-1: disabled, use default Trainer checkpoint — skip all setup
        if self.batch_size < 0:
            logger.info("[LogBased] batch_size=-1, disabled (using default Trainer checkpoint)")
            logger.info(f"[LogBased Config] LOG_BASED_CKPT={self.batch_size}")
            return

        if self.trainer and hasattr(self.trainer, 'zo') and self.trainer.zo:
            if not self._hook_registered:
                self.trainer.register_zo2_training_step_post_hook(self._zo_update_hook)
                self._hook_registered = True
                logger.info("[LogBased] Registered post-hook")

        # Detect model dtype for replay consistency
        if model is not None and self.model_dtype is None:
            for p in model.parameters():
                self.model_dtype = str(p.dtype)  # e.g. "torch.float16"
                break
            logger.info(f"[LogBased] Detected model dtype: {self.model_dtype}")

        log_based_resume = getattr(args, 'log_based_resume', '')

        if log_based_resume and model is not None:
            # Resume: model already reconstructed via replay — don't cache from scratch
            self._init_for_resume(model, state, log_based_resume)
        elif self.base_checkpoint_state is None and model is not None:
            # Fresh training: prefer injected CPU initial state captured at model load time.
            if self.enable_shadow:
                self._init_shadow_adam_state(model)
            if self.initial_cpu_state_dict is not None:
                model_state_keys = tuple(model.state_dict().keys())
                if model_state_keys != self.initial_cpu_state_keys:
                    raise RuntimeError(
                        "Injected initial CPU state keys do not match model.state_dict() keys at on_train_begin: "
                        f"source={self.initial_cpu_state_source}, "
                        f"injected={len(self.initial_cpu_state_keys)} keys, model={len(model_state_keys)} keys, "
                        f"first_injected_only={next((k for k in self.initial_cpu_state_keys if k not in model_state_keys), None)}, "
                        f"first_model_only={next((k for k in model_state_keys if k not in self.initial_cpu_state_keys), None)}"
                    )
                logger.info(
                    f"[LogBased] Using injected initial CPU state from {self.initial_cpu_state_source}"
                )
                self._initialize_model_metadata(model)
                self._finalize_initial_base(
                    self.initial_cpu_state_dict,
                    source=f"injected:{self.initial_cpu_state_source}",
                )
                self.initial_cpu_state_dict = None
                self.initial_cpu_state_source = None
                self.initial_cpu_state_keys = None
            else:
                self._cache_initial_model(model)

        # Sync current_step with trainer state so the hook reports correct step numbers
        self.current_step = state.global_step
        self.failure_simulator.advance_past(self.current_step)

        # Start shadow model as separate process (zero contention with training)
        if self.enable_shadow:
            self._start_shadow_process()
            if _step_diag_enabled() or _step_exact_enabled():
                self._log_shadow_storage_exact_compare(model, self.current_step)
            self._start_anchor_publisher()
            if self.instant_recover:
                logger.info("[LogBased] Mode: L3 (Instant Recovery) - shadow tracking + instant recovery")
            else:
                logger.info("[LogBased] Mode: L2 (CPU Shadow) - real-time shadow tracking")
        else:
            mode_desc = self._get_mode_description()
            logger.info(f"[LogBased] Mode: L1 ({mode_desc}) - on-demand reconstruction")

        effective_gpu_fail_step = format_gpu_fail_steps(
            self.failure_simulator.get_remaining_fail_steps()
        )
        effective_async_anchor = bool(getattr(self, "_async_anchor", None) or getattr(self.trainer, "_async_anchor", None))
        effective_rng_device = os.environ.get('ZO_RNG_DEVICE', 'native')
        if self.trainer is not None and hasattr(self.trainer, 'model'):
            _opt = getattr(self.trainer.model, 'opt', None)
            if _opt is not None:
                effective_rng_device = getattr(_opt, 'rng_device', effective_rng_device)

        # Log effective runtime configuration only.
        logger.info(
            f"[LogBased Effective Config]\n"
            f"  LOG_BASED_CKPT={self.batch_size}\n"
            f"  ENABLE_SHADOW={self.enable_shadow}\n"
            f"  INSTANT_RECOVER={self.instant_recover}\n"
            f"  FAILURE_TYPE={os.environ.get('FAILURE_TYPE', 'soft')}\n"
            f"  GPU_FAIL_STEP={effective_gpu_fail_step}\n"
            f"  ASYNC_ANCHOR={effective_async_anchor}\n"
            f"  ZO_RNG_DEVICE={effective_rng_device}\n"
            f"  SHADOW_PIPELINE={os.environ.get('SHADOW_PIPELINE', '0')}\n"
            f"  SHADOW_PIPELINE_WORKERS={os.environ.get('SHADOW_PIPELINE_WORKERS', '2')}\n"
            f"  SHADOW_COMMIT_INTERVAL={self.shadow_commit_interval}\n"
            f"  SHADOW_FLAT_COMMIT={self.use_shadow_flat_commit}\n"
            f"  SHADOW_RESERVE_THREADS={os.environ.get('SHADOW_RESERVE_THREADS', '1')}\n"
            f"  SHADOW_CONSUMER_THREADS={os.environ.get('SHADOW_CONSUMER_THREADS', 'auto')}\n"
            f"  LOG_BASED_SIMULATE_PERTURBATION={os.environ.get('LOG_BASED_SIMULATE_PERTURBATION', '1')}\n"
            f"  FORCE_FSYNC={os.environ.get('FORCE_FSYNC', '0')}\n"
            f"  LOG_BASED_REPLAY_DEVICE={os.environ.get('LOG_BASED_REPLAY_DEVICE', 'cuda')}\n"
            f"  LOG_BASED_REPLAY_FP32={os.environ.get('LOG_BASED_REPLAY_FP32', '0')}\n"
            f"  ZO_RNG_TRAIN_THREADS={os.environ.get('ZO_RNG_TRAIN_THREADS', 'unset')}"
        )
        # Thread env — show env var → actual value mapping for training process
        _zo_actual = -1
        try:
            import zo_rng as _zr_init
            _zo_actual = _zr_init.get_num_threads()
        except ImportError:
            pass
        if _thread_debug_enabled():
            logger.info(
                f"[Thread Env — Training Process]\n"
                f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}"
                f" → torch.get_num_threads()={torch.get_num_threads()}\n"
                f"  OMP_WAIT_POLICY={os.environ.get('OMP_WAIT_POLICY', 'unset')}\n"
                f"  MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', 'unset')}\n"
                f"  OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', 'unset')}\n"
                f"  NUMEXPR_NUM_THREADS={os.environ.get('NUMEXPR_NUM_THREADS', 'unset')}\n"
                f"  KMP_BLOCKTIME={os.environ.get('KMP_BLOCKTIME', 'unset')}\n"
                f"  GOMP_SPINCOUNT={os.environ.get('GOMP_SPINCOUNT', 'unset')}\n"
                f"  ZO_RNG_NUM_THREADS={os.environ.get('ZO_RNG_NUM_THREADS', 'unset')}"
                f" → zo_rng.get_num_threads()={_zo_actual}\n"
                f"  SHADOW_PIPELINE_WORKERS={os.environ.get('SHADOW_PIPELINE_WORKERS', '2')}\n"
                f"  SHADOW_COMMIT_INTERVAL={self.shadow_commit_interval}\n"
                f"  SHADOW_FLAT_COMMIT={self.use_shadow_flat_commit}\n"
                f"  SHADOW_CONSUMER_THREADS={os.environ.get('SHADOW_CONSUMER_THREADS', 'auto')}\n"
                f"  SHADOW_RESERVE_THREADS={os.environ.get('SHADOW_RESERVE_THREADS', '1')}"
            )
        _thread_snapshot("Train INIT", detail=True)

    def _get_mode_description(self) -> str:
        """Get mode description string"""
        if self.batch_size == -1:
            return "Disabled"
        elif self.batch_size == 0:
            return "Log-based (accumulate all)"
        else:
            return f"Full + Log (every {self.batch_size} steps)"

    def set_initial_cpu_state(self, state_dict: OrderedDict, source: str = "model_load"):
        """Inject an initial CPU state captured before the model is moved to GPU."""
        normalized = OrderedDict()
        for name, tensor in state_dict.items():
            if torch.is_tensor(tensor):
                if tensor.device.type == "cpu" and not tensor.requires_grad:
                    normalized[name] = tensor.detach()
                else:
                    normalized[name] = tensor.detach().to(device="cpu").clone()
            else:
                normalized[name] = tensor
        self.initial_cpu_state_dict = normalized
        self.initial_cpu_state_source = source
        self.initial_cpu_state_keys = tuple(normalized.keys())
        total_mb = sum(
            tensor.numel() * tensor.element_size()
            for tensor in normalized.values()
            if torch.is_tensor(tensor)
        ) / (1024 ** 2)
        logger.info(
            f"[LogBased] Received injected initial CPU state from {source} "
            f"({len(normalized)} tensors, {total_mb:.1f} MB)"
        )

    def _initialize_model_metadata(self, model):
        """Detect tied weights and capture the training parameter order."""
        self._tied_weight_groups = _detect_tied_weights(model)
        if self._tied_weight_groups:
            logger.info(f"[LogBased] Detected tied weight groups: {self._tied_weight_groups}")

        self._trainable_param_names = [
            name for name, p in model.named_parameters() if p.requires_grad
        ]
        total_params = sum(1 for _ in model.named_parameters())
        logger.info(f"[LogBased] Trainable params: {len(self._trainable_param_names)} / {total_params}")

    def _finalize_initial_base(self, state_dict: OrderedDict, source: str):
        """Activate a CPU initial base and publish all step-0 artifacts from it."""
        t_finalize_start = time.time()
        self.base_checkpoint_state = state_dict
        self.active_base_step = 0
        self.base_checkpoint_step = 0
        self.base_checkpoint_path = "__initial__"
        self._active_base_pending_seed = 0
        self._base_pending_seed = 0
        self.shadow_base_step = 0
        self.shadow_step = 0

        t_refresh_shadow = 0.0
        if self.enable_shadow:
            t0 = time.time()
            self._refresh_shadow_from_base(step=0, commit_now=True)
            t_refresh_shadow = time.time() - t0
        t_publish_anchor = 0.0
        if self.anchor_latest_path:
            t0 = time.time()
            self._publish_anchor_latest(self.base_checkpoint_state, 0)
            t_publish_anchor = time.time() - t0

        t_save_initial_model = 0.0
        if self.output_dir:
            initial_model_dir = os.path.join(self.output_dir, "initial_model")
            os.makedirs(initial_model_dir, exist_ok=True)
            excluded = self._shadow_secondary_keys()
            if excluded:
                save_state = OrderedDict(
                    (k, v) for k, v in self.base_checkpoint_state.items() if k not in excluded
                )
            else:
                save_state = self.base_checkpoint_state
            try:
                from safetensors.torch import save_file
                save_path = os.path.join(initial_model_dir, "model.safetensors")
                t0 = time.time()
                save_file(save_state, save_path)
                _fsync_file(save_path)
                t_save_initial_model = time.time() - t0
                logger.info(
                    f"[LogBased] Initial model saved to {initial_model_dir} "
                    f"(safetensors, excluded {len(excluded)} tied keys)"
                )
            except ImportError:
                save_path = os.path.join(initial_model_dir, "pytorch_model.bin")
                t0 = time.time()
                torch.save(save_state, save_path)
                _fsync_file(save_path)
                t_save_initial_model = time.time() - t0
                logger.info(
                    f"[LogBased] Initial model saved to {initial_model_dir} "
                    f"(pytorch_model.bin, excluded {len(excluded)} tied keys)"
                )

        mem_mb = self._get_memory_size()
        logger.info(f"[LogBased] Initial model base activated from {source} ({mem_mb:.1f} MB)")
        if time_log_enabled():
            logger.info(
                f"[LogBased InitBaseTiming] source={source} "
                f"refresh_shadow={t_refresh_shadow:.3f}s "
                f"publish_anchor={t_publish_anchor:.3f}s "
                f"save_initial_model={t_save_initial_model:.3f}s "
                f"total={time.time() - t_finalize_start:.3f}s"
            )
        self._log_memory_status()

    def _cache_initial_model(self, model):
        """Cache initial model to CPU memory and disk for fresh training."""
        logger.info("[LogBased] Caching initial model to CPU memory and disk...")
        t_start = time.time()

        self._initialize_model_metadata(model)
        base_state = _clone_state_dict_to_cpu(model.state_dict())
        self._finalize_initial_base(base_state, source="fallback_gpu_to_cpu_clone")

        mem_mb = self._get_memory_size()
        t_elapsed = time.time() - t_start
        if time_log_enabled():
            logger.info(f"[LogBased] Initial model cached ({mem_mb:.1f} MB) in {t_elapsed:.3f}s")
        self._log_memory_status()

    def _init_for_resume(self, model, state, log_based_resume):
        """Initialize callback state from an already-reconstructed (resumed) model.
        Unlike _cache_initial_model (for fresh training), this:
        - Loads update_history and base info from the checkpoint's optimizer.pt
        - Sets base_checkpoint_step correctly for batch_size>=1
        - Starts shadow from the already-reconstructed current model state
        """
        logger.info("[LogBased Resume] Initializing from resumed model...")
        t_start = time.time()

        # Detect tied weights
        self._tied_weight_groups = _detect_tied_weights(model)
        if self._tied_weight_groups:
            logger.info(f"[LogBased] Detected tied weight groups: {self._tied_weight_groups}")

        # Cache trainable param names
        self._trainable_param_names = [
            name for name, p in model.named_parameters() if p.requires_grad
        ]

        # Load log metadata once for all metadata (pending_grad + update_history + base info)
        opt_state = None
        meta_path = os.path.join(log_based_resume, LOG_METADATA_NAME)
        opt_path = os.path.join(log_based_resume, "optimizer.pt")
        if os.path.exists(meta_path):
            try:
                opt_state = torch.load(meta_path, map_location='cpu', weights_only=False)
                if not isinstance(opt_state, dict):
                    opt_state = None
                else:
                    logger.info(f"[LogBased Resume] Loaded lightweight log metadata from {LOG_METADATA_NAME}")
            except Exception as e:
                logger.warning(f"[LogBased Resume] Failed to load {LOG_METADATA_NAME}: {e}")
        if opt_state is None and os.path.exists(opt_path):
            try:
                opt_state = torch.load(opt_path, map_location='cpu', weights_only=False)
                if not isinstance(opt_state, dict) or 'zo_update_history' not in opt_state:
                    opt_state = None
            except Exception as e:
                logger.warning(f"[LogBased Resume] Failed to load optimizer.pt: {e}")

        # Restore pending_grad
        if hasattr(model, 'opt') and opt_state is not None:
            pg = opt_state.get('pending_grad', None)
            if pg is not None:
                model.opt.projected_grad = float(pg)
                logger.info(f"[LogBased Resume] Restored pending_grad={float(pg):.6e}")

                # ZO2 only: reconstruct last_rstate so the first resumed step's
                # zo_update can regenerate the correct perturbation vector z.
                if pg != 0 and hasattr(model.opt, 'rstate_queue'):
                    ps = opt_state.get('pending_seed', None)
                    if ps is None:
                        # Fallback: for checkpoints created before pending_seed was saved,
                        # the last entry in zo_update_history has the seed we need (step >= 2).
                        updates = opt_state.get('zo_update_history', [])
                        if updates:
                            ps = updates[-1]['seed']
                            logger.info(f"[LogBased Resume] No pending_seed in checkpoint, "
                                        f"using last update history seed={ps}")
                        else:
                            raise RuntimeError(
                                f"Checkpoint has pending_grad={pg} but no pending_seed "
                                "and empty zo_update_history. Cannot reconstruct RNG state "
                                "for ZO2 delayed update. Re-run baseline with updated code."
                            )
                    torch.cuda.manual_seed(ps)
                    model.opt.last_rstate = torch.cuda.get_rng_state()
                    model.opt.rstate_queue.append(model.opt.last_rstate.clone())
                    logger.info(f"[LogBased Resume] Reconstructed last_rstate from seed={ps}")
            else:
                logger.warning("[LogBased Resume] No pending_grad found, "
                               "first step will skip zo_update (projected_grad=0)")

        # Restore Adam state (only when optimizer is MeZOAdam)
        if hasattr(model, 'opt') and hasattr(model.opt, 'm'):
            # Priority: replayed adam state (log checkpoint) > checkpoint adam state (full checkpoint)
            replay_adam = _get_and_clear_replay_adam_state()
            if replay_adam:
                model.opt.restore_adam_state(replay_adam)
                logger.info(f"[LogBased Resume] Restored replayed Adam state: t={replay_adam.get('t', 0)}")
            elif opt_state and 'adam_state' in opt_state:
                model.opt.restore_adam_state(opt_state['adam_state'])
                logger.info(f"[LogBased Resume] Restored Adam state from checkpoint: "
                            f"t={opt_state['adam_state'].get('t', 0)}")
            if _step_diag_enabled() or _step_exact_enabled():
                _log_adam_checksums(f"train_restored step={state.global_step}", model.opt.get_adam_state())
                if _step_exact_enabled():
                    _log_adam_exact_fingerprint(f"train_restored step={state.global_step}", model.opt.get_adam_state())

        if self.enable_shadow:
            self._init_shadow_adam_state(model)
            if (
                (_step_diag_enabled() or _step_exact_enabled())
                and hasattr(model, 'opt')
                and hasattr(model.opt, 'get_adam_state')
                and self.shadow_adam_state is not None
            ):
                _log_adam_checksums(f"shadow_init step={state.global_step}", self.shadow_adam_state)
                if _step_exact_enabled():
                    _log_adam_exact_fingerprint(f"shadow_init step={state.global_step}", self.shadow_adam_state)
                    _log_adam_exact_compare(
                        f"train_vs_shadow_init step={state.global_step}",
                        model.opt.get_adam_state(),
                        self.shadow_adam_state,
                    )

        # Load update history and base info
        if opt_state is not None:
            is_full_ckpt = opt_state.get('is_full_checkpoint', False)
            self._base_pending_seed = opt_state.get('base_pending_seed', 0)

            if self.batch_size >= 1:
                if is_full_ckpt:
                    # Full checkpoint: this IS the new base, history starts empty
                    self.base_checkpoint_path = log_based_resume
                    self.base_checkpoint_step = state.global_step
                    self.update_history = []
                    logger.info(f"[LogBased Resume] Full checkpoint → new base at step {state.global_step}")
                else:
                    # Log checkpoint: inherit base and history from metadata
                    self.update_history = list(opt_state['zo_update_history'])
                    self.base_checkpoint_path = opt_state.get('base_checkpoint', '__initial__')
                    match = re.search(r'checkpoint-(\d+)', str(self.base_checkpoint_path))
                    self.base_checkpoint_step = int(match.group(1)) if match else 0
                    logger.info(f"[LogBased Resume] Log checkpoint → base={self.base_checkpoint_path} "
                               f"(step {self.base_checkpoint_step}), {len(self.update_history)} updates")
            else:
                # batch_size=0: always accumulate all updates from initial model
                self.update_history = list(opt_state['zo_update_history'])
                logger.info(f"[LogBased Resume] Loaded {len(self.update_history)} updates (accumulative)")

        # batch_size=0: base is always __initial__
        if self.batch_size == 0:
            self.base_checkpoint_path = "__initial__"
            self.base_checkpoint_step = 0

        self._activate_base_state(_clone_state_dict_to_cpu(model.state_dict()), state.global_step)
        self.shadow_base_step = int(state.global_step)
        self.shadow_step = int(state.global_step)
        if self.enable_shadow:
            self._refresh_shadow_from_base(step=state.global_step, commit_now=True)
        if self.anchor_latest_path and self.batch_size >= 1:
            self._publish_anchor_latest(self.base_checkpoint_state, state.global_step)
            self._last_shadow_rebased_anchor_step = int(state.global_step)

        self.is_first_save = False
        mem_mb = self._get_memory_size()
        t_elapsed = time.time() - t_start
        if time_log_enabled():
            logger.info(f"[LogBased Resume] Initialized ({mem_mb:.1f} MB) in {t_elapsed:.3f}s")

    def _init_shadow_adam_state(self, model):
        """Initialize shadow_adam_state if the optimizer is MeZO-Adam.

        Clone the current optimizer Adam state onto CPU so shadow flat storage,
        rebase, and soft-recovery start from the exact same optimizer state.
        """
        opt = getattr(model, 'opt', None)
        if opt is None:
            self.shadow_adam_state = None
            return

        if not (hasattr(opt, 'get_adam_state') and hasattr(opt, 'betas') and hasattr(opt, 'adam_eps')):
            self.shadow_adam_state = None
            logger.info("[Shadow] Optimizer is SGD, no Adam state needed")
            return

        self.shadow_adam_state = self._clone_shadow_adam_state(model)
        logger.info(
            f"[Shadow] Initialized Adam state: betas={self.shadow_adam_state['betas']}, "
            f"eps={self.shadow_adam_state['adam_eps']}, t={self.shadow_adam_state['t']}"
        )

    def _start_shadow_process(self):
        """Start shadow model as a separate process."""
        if self.shadow_process is not None and self.shadow_process.is_alive():
            return

        _thread_snapshot("Train BEFORE_SHADOW_SPAWN")
        use_pipeline = os.environ.get('SHADOW_PIPELINE', '0') == '1'
        P = int(os.environ.get('SHADOW_PIPELINE_WORKERS', '2'))
        simulate_perturbation = os.environ.get('LOG_BASED_SIMULATE_PERTURBATION', '1') == '1'

        # Detect rng_device
        rng_device = 'native'
        if self.trainer is not None and hasattr(self.trainer, 'model'):
            _opt = getattr(self.trainer.model, 'opt', None)
            if _opt is not None:
                rng_device = getattr(_opt, 'rng_device', 'native')

        # Detect default zo_eps
        default_zo_eps = 0.0
        if self.trainer and hasattr(self.trainer, 'model') and hasattr(self.trainer.model, 'opt'):
            default_zo_eps = getattr(self.trainer.model.opt, 'zo_eps', 0.0)

        # Adam config (picklable dict, not the full state)
        adam_config = None
        if self.shadow_adam_state is not None:
            adam_config = {
                'betas': self.shadow_adam_state['betas'],
                'adam_eps': self.shadow_adam_state['adam_eps'],
            }

        param_names = self._trainable_param_names or list(self.base_checkpoint_state.keys())

        ctx = mp.get_context('spawn')
        self.update_queue = ctx.Queue()
        self.shadow_step_val = ctx.Value('i', int(self.shadow_step), lock=False)
        self.shadow_ready_event = ctx.Event()

        try:
            total_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            total_cores = os.cpu_count() or 64
        n_reserve = int(os.environ.get('SHADOW_RESERVE_THREADS', '1'))
        if use_pipeline and rng_device == "zo_rng":
            n_cons = int(os.environ.get('SHADOW_CONSUMER_THREADS', str(total_cores // 2)))
            c_prod = max(1, total_cores - n_reserve - n_cons)
            aten_threads = max(1, n_cons)
        else:
            # Serial mode: both pools alternate, use all cores minus reserve
            aten_threads = max(1, total_cores - n_reserve)
            n_cons = aten_threads
            c_prod = aten_threads  # serial: same pool, alternating
        _thread_env = {
            'OMP_NUM_THREADS': str(aten_threads),
            'OMP_WAIT_POLICY': 'passive',
            'GOMP_SPINCOUNT': '0',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'KMP_BLOCKTIME': '0',
        }
        _thread_env_keys = list(_thread_env.keys())
        _old_env = {k: os.environ.get(k) for k in _thread_env_keys}
        for k, v in _thread_env.items():
            os.environ[k] = v

        spawn_initial_state = None if self.use_shadow_flat_commit else self.base_checkpoint_state
        t_spawn = time.perf_counter()
        self.shadow_process = ctx.Process(
            target=_shadow_process_main,
            args=(
                self.update_queue,
                spawn_initial_state,
                self.shadow_base_step,
                self.shadow_step,
                self.shadow_step_val,
                param_names,
                getattr(self, "_tied_weight_groups", []),
                rng_device,
                simulate_perturbation,
                default_zo_eps,
                adam_config,
                use_pipeline,
                P,
                self.shadow_replica_path,
                self.shadow_commit_interval,
                self.shadow_flat_storage,
                self.shadow_ready_event,
            ),
            daemon=True,
        )

        self.shadow_process.start()
        spawn_elapsed_s = time.perf_counter() - t_spawn

        # Restore parent env (training process keeps its own settings)
        for k in _thread_env_keys:
            if _old_env[k] is not None:
                os.environ[k] = _old_env[k]
            else:
                os.environ.pop(k, None)

        if rng_device == "zo_rng":
            _train_zo = os.environ.get('ZO_RNG_TRAIN_THREADS')
            if _train_zo is not None:
                _train_zo = int(_train_zo)
                try:
                    import zo_rng as _zr_parent
                    _zr_parent.set_num_threads(_train_zo)
                    logger.info(f"[LogBased] Training zo_rng threads reduced to {_train_zo}")
                except ImportError:
                    pass

        logger.info(f"[LogBased] Started shadow process (PID={self.shadow_process.pid}, "
                    f"pipeline={use_pipeline}, P={P}, rng={rng_device}, "
                    f"child OMP_NUM_THREADS={aten_threads}, spawn_mode="
                    f"{'flat-metadata' if self.use_shadow_flat_commit else 'pickle-state'}, "
                    f"spawn={spawn_elapsed_s:.3f}s)")
        wait_shadow_ready = os.environ.get("SHADOW_WAIT_READY", "1") == "1"
        ready_timeout_s = float(os.environ.get("SHADOW_READY_TIMEOUT", "60"))
        if wait_shadow_ready:
            t_wait_ready = time.perf_counter()
            ready = self.shadow_ready_event.wait(timeout=ready_timeout_s)
            wait_ready_elapsed_s = time.perf_counter() - t_wait_ready
            if not ready:
                alive = self.shadow_process.is_alive()
                raise RuntimeError(
                    f"shadow process did not become ready within {ready_timeout_s:.1f}s "
                    f"(alive={alive}, waited={wait_ready_elapsed_s:.3f}s)"
                )
            if time_log_enabled():
                logger.info(
                    f"[LogBased] Shadow reported ready_for_updates in {wait_ready_elapsed_s:.3f}s "
                    f"(timeout={ready_timeout_s:.1f}s)"
                )
        _thread_snapshot("Train AFTER_SHADOW_SPAWN")

    def _zo_update_hook(self, model, inputs, loss):
        """Hook called after ZO training step.

        NOTE: GPU failure check has been moved to on_step_begin() so that
        recovery happens BEFORE the ZO forward, preserving the data batch
        for bitwise-identical continuation.
        """
        if hasattr(model, 'opt'):
            opt = model.opt
            seed = getattr(opt, 'zo_random_seed', 0)
            # _applied_update_grad: the grad actually used in this step's zo_update
            # (i.e. the PREVIOUS step's projected_grad, captured in zo_forward before inner_zo_forward)
            applied_grad = getattr(opt, '_applied_update_grad', 0)
            # projected_grad: the NEW grad just computed in this step (to be applied NEXT step)
            new_grad = getattr(opt, 'projected_grad', 0)
            lr = getattr(opt, 'lr', 0)
            wd = getattr(opt, 'weight_decay', 0)
            zo_eps = getattr(opt, 'zo_eps', 0.0)

            with self.update_lock:
                # Use current_step + 1 since this hook is called BEFORE global_step is incremented
                actual_step = self.current_step + 1

                # Save the pending grad (newly computed, not yet applied) for checkpoint/resume.
                # On resume, this will be restored to opt.projected_grad so the first step applies it.
                self._pending_grad = float(new_grad) if not isinstance(new_grad, float) else new_grad
                # Save the seed from this step — needed by ZO2 to reconstruct last_rstate on resume.
                self._pending_seed = int(seed) if seed is not None else 0

                # Always record entries, including grad=0 (step 0 perturbation-only).
                # This captures fp16 rounding from the perturbation-restore cycle AND
                # provides the seed chain needed for ZO2 replay (entry[i-1]'s seed is
                # used for entry[i]'s gradient update).
                update = {
                    'step': actual_step,
                    'seed': int(seed) if seed is not None else 0,
                    'grad': float(applied_grad) if not isinstance(applied_grad, float) else applied_grad,
                    'lr': float(lr),
                    'wd': float(wd),
                    'zo_eps': float(zo_eps),
                }
                self.update_history.append(update)
                # Non-blocking send to shadow process (no lock, no GIL contention)
                if self.update_queue is not None:
                    try:
                        t_shadow_send_start = time.perf_counter()
                        self.update_queue.put_nowait({"cmd": "update", "update": update})
                        shadow_send_s = time.perf_counter() - t_shadow_send_start
                        self.timing_stats['shadow_send_times'].append(shadow_send_s)
                        if time_log_enabled():
                            logger.info(
                                f"[ShadowSend] step={actual_step} enqueue={shadow_send_s * 1000.0:.3f}ms"
                            )
                    except Exception as e:
                        if not getattr(self, '_queue_error_logged', False):
                            logger.warning(f"[LogBased] Failed to send update to shadow: {e}")
                            self._queue_error_logged = True
                # Thread diagnostics
                if _step_diag_enabled() and actual_step == 1:
                    _thread_snapshot("Train step=1 AFTER_ZO_FORWARD", detail=True)
                if _step_diag_enabled():
                    _cksum = sum(p.data.float().sum().item() for p in model.parameters())
                    logger.info(f"[CKSUM] step={actual_step} checksum={_cksum:.10e}")
                if (_step_diag_enabled() or _step_exact_enabled()) and hasattr(model, "opt") and hasattr(model.opt, "get_adam_state"):
                    live_adam = model.opt.get_adam_state()
                    if _step_diag_enabled():
                        _log_adam_brief(f"train_live step={actual_step}", live_adam)
                        _log_adam_checksums(f"train_live step={actual_step}", live_adam)
                        if _step_exact_enabled():
                            _log_adam_exact_fingerprint(f"train_live step={actual_step}", live_adam)
                    elif _step_exact_enabled():
                        _log_adam_exact_fingerprint(f"train_live step={actual_step}", live_adam)
                if _step_diag_enabled():
                    live_state = OrderedDict(
                        (name, tensor.detach())
                        for name, tensor in model.state_dict().items()
                        if torch.is_tensor(tensor)
                    )
                    _log_state_checksums(f"train_live step={actual_step}", live_state)
                    if _step_exact_enabled():
                        _log_state_exact_fingerprint(f"train_live step={actual_step}", live_state)
                if (
                    self.enable_shadow
                    and self.shadow_commit_interval > 0
                    and actual_step % self.shadow_commit_interval == 0
                ):
                    live_state = OrderedDict(
                        (name, tensor.detach())
                        for name, tensor in model.state_dict().items()
                        if torch.is_tensor(tensor)
                    )
                    _log_state_checksums(f"train_durable_ref step={actual_step}", live_state)
                    _log_state_exact_fingerprint(f"train_durable_ref step={actual_step}", live_state)
                    if hasattr(model, "opt") and hasattr(model.opt, "get_adam_state"):
                        live_adam = model.opt.get_adam_state()
                        _log_adam_checksums(f"train_durable_ref step={actual_step}", live_adam)
                        _log_adam_exact_fingerprint(f"train_durable_ref step={actual_step}", live_adam)
                if _step_diag_enabled() and actual_step == 1:
                    _thread_snapshot("Train step=1 AFTER_CKSUM", detail=True)
                if _step_diag_enabled() and (actual_step <= 20 or actual_step % 50 == 0):
                    _thread_snapshot(f"Train step={actual_step}")
                if _step_diag_enabled() and applied_grad != 0:
                    logger.info(f"[HOOK] step={actual_step}, UPDATE RECORDED: seed={update['seed']}, "
                                f"applied_grad={update['grad']:.6e}, new_grad={new_grad:.6e}, lr={lr}, wd={wd}")
                    # Verification tag: grep "[VERIFY]" to cross-check train vs replay
                    logger.info(f"[VERIFY] step={actual_step} total_updates={len(self.update_history)} "
                                f"pending_grad={self._pending_grad:.6e}")
                elif _step_diag_enabled():
                    logger.info(f"[HOOK] step={actual_step}, PERTURBATION-ONLY (grad=0): "
                                f"seed={update['seed']}, new_grad={new_grad:.6e} (will be applied next step)")

        return model, inputs, loss

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        """Check for GPU failure — triggers SIGKILL if configured.

        SIGKILL kills the process immediately. Shell script detects exit 137
        and auto-resumes from the latest checkpoint.
        """
        if model is None:
            return

        self._check_anchor_publisher_health()

        if self.failure_simulator.check_and_fail(self.current_step, model):
            self.failure_simulator.trigger_failure(model)
            # Unreachable — SIGKILL kills process

    def on_step_end(self, args, state, control, **kwargs):
        """Called at end of each step"""
        self.current_step = state.global_step

        async_anchor = getattr(self, "_async_anchor", None) or getattr(self.trainer, "_async_anchor", None)
        if async_anchor is not None:
            published = async_anchor.consume_latest_published_snapshot(self.active_base_step)
            if published is not None:
                published_step, published_payload = published
                if isinstance(published_payload, tuple):
                    published_state, published_adam_state = published_payload
                else:
                    published_state, published_adam_state = published_payload, None
                self._activate_base_state(published_state, published_step, adam_state=published_adam_state)
                if self.enable_shadow and published_step > self._last_shadow_rebased_anchor_step:
                    if hasattr(self.trainer.model, 'opt') and hasattr(self.trainer.model.opt, 'betas') and published_adam_state is None:
                        raise RuntimeError(
                            f"async anchor step {published_step} published model state without Adam state"
                        )
                    path = self._publish_anchor_latest(
                        published_state,
                        published_step,
                        adam_state=published_adam_state,
                    )
                    self._queue_shadow_rebase(
                        published_step,
                        path=path,
                    )
                    self._last_shadow_rebased_anchor_step = published_step

        if self.current_step % 10 == 0:
            num_updates = len(self.update_history)
            if resource_log_enabled():
                cpu_pct, mem_gb, mem_total, gpu_alloc, gpu_rsv = _system_stats()
                if self.enable_shadow:
                    shadow_step = self.shadow_step_val.value if self.shadow_step_val else self.shadow_step
                    # Health check: detect dead shadow process (log once)
                    if self.shadow_process is not None and not self.shadow_process.is_alive():
                        if not getattr(self, '_shadow_death_logged', False):
                            logger.error(f"[LogBased] Shadow process DEAD (exitcode={self.shadow_process.exitcode})")
                            self._shadow_death_logged = True
                    logger.info(f"[LogBased] step={self.current_step} shadow={shadow_step} "
                               f"updates={num_updates} "
                               f"| CPU={cpu_pct:.0f}% MEM={mem_gb:.0f}/{mem_total:.0f}GB "
                               f"| GPU alloc={gpu_alloc:.0f}MB rsv={gpu_rsv:.0f}MB")
                else:
                    logger.info(f"[LogBased] step={self.current_step} updates={num_updates} "
                               f"| CPU={cpu_pct:.0f}% MEM={mem_gb:.0f}/{mem_total:.0f}GB "
                               f"| GPU alloc={gpu_alloc:.0f}MB rsv={gpu_rsv:.0f}MB")

    def recover_from_shadow(self) -> OrderedDict:
        """Recover from the latest committed shadow replica in the configured tmpfs directory."""
        t_start = time.time()

        if self.enable_shadow and self._shadow_storage_available():
            try:
                if self.use_shadow_flat_commit and self.shadow_flat_storage is not None:
                    recovered, recovered_adam, base_step, shadow_step = _load_shadow_bundle_flat(
                        self.shadow_flat_storage,
                        tied_groups=getattr(self, "_tied_weight_groups", []),
                    )
                    if recovered_adam is not None:
                        self.shadow_adam_state = recovered_adam
                        _log_adam_checksums(f"recover_from_shadow step={shadow_step}", recovered_adam)
                        _log_adam_exact_fingerprint(f"recover_from_shadow step={shadow_step}", recovered_adam)
                else:
                    recovered, base_step, shadow_step = _load_shadow_replica(
                        self.shadow_replica_path,
                        tied_groups=getattr(self, "_tied_weight_groups", []),
                    )
            except RuntimeError as e:
                logger.error(
                    f"[Recovery] Shadow snapshot is incomplete or corrupted under single-buffer mode: {e}. "
                    f"Treating as hard failure."
                )
                raise
            self.shadow_base_step = base_step
            self.shadow_step = shadow_step

            t_elapsed = time.time() - t_start
            if time_log_enabled():
                logger.info(f"[Recovery] Recovered from shadow model at step {shadow_step} in {t_elapsed:.3f}s")
            logger.info(f"[Recovery] GPU was at step {self.current_step}, shadow was at step {shadow_step}")
            logger.info(f"[Recovery] Lost {self.current_step - shadow_step} steps (will be replayed)")

            self.timing_stats['recoveries'].append({
                'type': 'shadow',
                'time': t_elapsed,
                'shadow_step': shadow_step,
                'gpu_step': self.current_step
            })

            return recovered
        else:
            return self._reconstruct_on_demand()

    def _reconstruct_on_demand(self) -> OrderedDict:
        """
        On-demand reconstruction: replay update_history from base_checkpoint_state
        """
        if self.base_checkpoint_state is None:
            logger.error("[Recovery] No base checkpoint cached!")
            return None

        t_start = time.time()

        base_step = int(self.active_base_step)
        with self.update_lock:
            updates = [u for u in self.update_history if int(u.get("step", -1)) > base_step]

        num_updates = len(updates)
        if time_log_enabled():
            logger.info(
                f"[Recovery] Reconstructing on-demand from active_base_step={base_step}: "
                f"replaying {num_updates} updates..."
            )

        # Copy base checkpoint state
        reconstructed = OrderedDict()
        for key, value in self.base_checkpoint_state.items():
            reconstructed[key] = value.clone()

        # Tie weights before replay so updates accumulate correctly (like during training)
        if hasattr(self, '_tied_weight_groups') and self._tied_weight_groups:
            _tie_state_dict_inplace(reconstructed, self._tied_weight_groups)

        # Get zo_eps for old update records that don't include it
        fallback_zo_eps = 0.0
        rng_device = "native"
        zo2_mode = False
        if self.trainer and hasattr(self.trainer, 'model') and hasattr(self.trainer.model, 'opt'):
            fallback_zo_eps = getattr(self.trainer.model.opt, 'zo_eps', 0.0)
            rng_device = getattr(self.trainer.model.opt, 'rng_device', 'native')
            zo2_mode = hasattr(self.trainer.model.opt, 'rstate_queue')

        # Replay all updates using only trainable param names (matching zo_update iteration)
        _replay_updates_on_state(
            reconstructed, updates,
            trainable_param_names=self._trainable_param_names,
            default_zo_eps=fallback_zo_eps,
            rng_device=rng_device,
            zo2_mode=zo2_mode,
            initial_prev_seed=self._active_base_pending_seed,
        )
        if num_updates > 0 and time_log_enabled():
            logger.info(f"[Recovery] Replayed {num_updates} updates")

        t_elapsed = time.time() - t_start
        if time_log_enabled():
            logger.info(f"[Recovery] Reconstruction completed in {t_elapsed:.3f}s ({num_updates} updates)")

        self.timing_stats['recoveries'].append({
            'type': 'on_demand',
            'time': t_elapsed,
            'num_updates': num_updates
        })

        return reconstructed

    def get_recovery_status(self) -> dict:
        """Get current recovery status"""
        if self.enable_shadow:
            shadow_available = self._shadow_storage_available()
            shadow_step = self.shadow_step_val.value if self.shadow_step_val else self.shadow_step
        else:
            shadow_step = -1
            shadow_available = False

        total_updates = len(self.update_history)

        can_recover = (self.enable_shadow and shadow_available and shadow_step > 0) or \
                      (not self.enable_shadow and self.base_checkpoint_state is not None)

        return {
            'gpu_step': self.current_step,
            'shadow_step': shadow_step,
            'shadow_available': shadow_available,
            'enable_shadow': self.enable_shadow,
            'total_updates': total_updates,
            'shadow_lag': self.current_step - shadow_step if self.enable_shadow else -1,
            'can_recover': can_recover,
            'batch_size': self.batch_size
        }

    def on_save(self, args, state, control, model=None, **kwargs):
        """Called when checkpoint is saved.
        All saving is handled by Trainer._save_checkpoint. This callback only
        updates internal state (shadow model, base checkpoint) if needed."""
        if model is None or self.batch_size < 0:
            return

        self.save_count += 1

        if self.batch_size == 0:
            # batch_size=0: base is always initial model, never changes. Nothing to do.
            self.last_saved_step = state.global_step
            return

        # batch_size >= 1: update base_checkpoint_state only on full steps
        # (trainer already set base_checkpoint_step = global_step on full steps)
        is_full_step = (self.base_checkpoint_step == state.global_step)
        if is_full_step:
            # Full step: history was cleared, base must be updated to current model.
            # Shadow (if enabled) is refreshed from new base.
            self._update_base_and_shadow(model, state.global_step)
        # Log steps: shadow worker catches up asynchronously, no action needed.

        self.last_saved_step = state.global_step
        self.is_first_save = False

    def _update_base_and_shadow(self, model, step):
        """Update base_checkpoint_state from current model (full step only for batch_size>=1).
        GPU → CPU clone, then publish anchor latest / notify shadow to rebase."""
        self._check_anchor_publisher_health()
        t0_total = time.time()
        t0 = time.time()
        base_checkpoint_state = _clone_state_dict_to_cpu(model.state_dict())
        clone_model_s = time.time() - t0

        clone_adam_s = 0.0
        shadow_adam_state = None
        live_adam_state = None
        if self.enable_shadow:
            t0 = time.time()
            opt = getattr(model, "opt", None)
            if opt is not None and hasattr(opt, "get_adam_state"):
                live_adam_state = opt.get_adam_state()
            shadow_adam_state = self._clone_shadow_adam_state(model)
            clone_adam_s = time.time() - t0
        self.base_checkpoint_state = base_checkpoint_state
        self.shadow_adam_state = shadow_adam_state
        self.active_base_step = int(step)
        self.base_checkpoint_step = int(step)
        self._active_base_pending_seed = self._pending_seed
        self._base_pending_seed = self._pending_seed

        submit_anchor_s = 0.0
        publish_anchor_s = 0.0
        queue_rebase_s = 0.0
        if self.batch_size >= 1 and self.enable_shadow:
            if self._use_async_anchor_publisher():
                t0 = time.time()
                self._submit_anchor_publish_task(
                    AnchorPublishTask(
                        step=int(step),
                        base_checkpoint_state=base_checkpoint_state,
                        adam_state=shadow_adam_state,
                        base_pending_seed=int(self._base_pending_seed),
                        created_at=time.time(),
                    )
                )
                submit_anchor_s = time.time() - t0
            else:
                t0 = time.time()
                anchor_path = self._publish_anchor_latest(self.base_checkpoint_state, step)
                publish_anchor_s = time.time() - t0
                if anchor_path is not None:
                    t0 = time.time()
                    self._queue_shadow_rebase(step, path=anchor_path)
                    queue_rebase_s = time.time() - t0
                    self._last_shadow_rebased_anchor_step = int(step)

        if time_log_enabled():
            logger.info(
                f"[LogBased FullCkpt] step={step} "
                f"clone_model={clone_model_s:.3f}s clone_adam={clone_adam_s:.3f}s "
                f"submit_anchor={submit_anchor_s:.3f}s "
                f"publish_anchor={publish_anchor_s:.3f}s queue_rebase={queue_rebase_s:.3f}s "
                f"total={time.time() - t0_total:.3f}s"
            )
        if _step_diag_enabled() or _step_exact_enabled():
            if live_adam_state is not None:
                _log_adam_checksums(f"train_snapshot step={step}", live_adam_state)
                if _step_exact_enabled():
                    _log_adam_exact_fingerprint(f"train_snapshot step={step}", live_adam_state)
            if shadow_adam_state is not None:
                _log_adam_checksums(f"shadow_snapshot step={step}", shadow_adam_state)
                if _step_exact_enabled():
                    _log_adam_exact_fingerprint(f"shadow_snapshot step={step}", shadow_adam_state)
            if _step_exact_enabled() and live_adam_state is not None and shadow_adam_state is not None:
                _log_adam_exact_compare(
                    f"train_vs_shadow_snapshot step={step}",
                    live_adam_state,
                    shadow_adam_state,
                )

    def on_async_anchor_persisted(self, step, checkpoint_path):
        self.base_checkpoint_path = checkpoint_path
        self.base_checkpoint_step = int(step)
        if self.shadow_adam_state is not None:
            logger.info(
                f"[AsyncAnchor] Base updated to step {step} with Adam base "
                f"(t={int(self.shadow_adam_state.get('t', 0))})"
            )
        with self.update_lock:
            for u in reversed(self.update_history):
                if u['step'] <= step:
                    self._base_pending_seed = u['seed']
                    break
            self.update_history = [u for u in self.update_history if u['step'] > step]

    def _get_memory_size(self):
        if self.base_checkpoint_state is None:
            return 0
        total = sum(v.numel() * v.element_size() for v in self.base_checkpoint_state.values())
        return total / (1024 * 1024)

    def _log_memory_status(self):
        if not resource_log_enabled():
            return
        cache_mb = self._get_memory_size()
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)

        if self.enable_shadow:
            shadow_step = self.shadow_step_val.value if self.shadow_step_val else self.shadow_step
            logger.info(f"[LogBased] Memory: cache={cache_mb:.1f}MB, "
                       f"system={used_gb:.1f}/{total_gb:.1f}GB, "
                       f"shadow_step={shadow_step}")
        else:
            logger.info(f"[LogBased] Memory: cache={cache_mb:.1f}MB, "
                       f"system={used_gb:.1f}/{total_gb:.1f}GB (on-demand mode)")

    def on_train_end(self, args, state, control, **kwargs):
        """Called at training end"""
        logger.info("[LogBased] Training ended, cleaning up...")

        # Shutdown async anchor checkpointer (wait for last persist to finish)
        async_anchor = getattr(self, '_async_anchor', None)
        if async_anchor is not None:
            logger.info("[AsyncAnchor] Waiting for last anchor persist to complete...")
            async_anchor.shutdown()
            completed_step = async_anchor.get_latest_completed_anchor_step()
            if completed_step > self.base_checkpoint_step:
                self.on_async_anchor_persisted(
                    completed_step,
                    async_anchor.get_latest_completed_anchor_path(),
                )
            async_stats = async_anchor.stats
            if time_log_enabled():
                logger.info(
                    "[AsyncAnchor] Summary: "
                    f"enqueue_cpu count={async_stats['enqueue_cpu_count']} avg={async_stats['avg_enqueue_cpu_time']:.3f}s | "
                    f"d2h count={async_stats['d2h_count']} avg={async_stats['avg_d2h_time']:.3f}s | "
                    f"cpu_total count={async_stats['cpu_persist_total_count']} "
                    f"avg={async_stats['avg_cpu_persist_total_time']:.3f}s"
                )

        self._stop_anchor_publisher()

        if self.enable_shadow:
            if self.update_queue is not None:
                try:
                    self.update_queue.put_nowait({'cmd': 'stop'})
                except Exception:
                    pass
            if self.shadow_process is not None and self.shadow_process.is_alive():
                self.shadow_process.join(timeout=5.0)
                if self.shadow_process.is_alive():
                    self.shadow_process.terminate()
                    logger.warning("[LogBased] Shadow process force-terminated")

        shadow_send_times = self.timing_stats.get('shadow_send_times', [])
        if shadow_send_times:
            avg_shadow_send_ms = (sum(shadow_send_times) / len(shadow_send_times)) * 1000.0
            if time_log_enabled():
                logger.info(
                    f"[ShadowSend] Summary: count={len(shadow_send_times)} avg={avg_shadow_send_ms:.3f}ms"
                )

        checkpoint_total_times = self.timing_stats.get('checkpoint_total_times', [])
        if checkpoint_total_times:
            avg_checkpoint_s = sum(checkpoint_total_times) / len(checkpoint_total_times)
            if time_log_enabled():
                logger.info(
                    f"[ZOTrainer] Full checkpoint summary: count={len(checkpoint_total_times)} avg={avg_checkpoint_s:.3f}s"
                )

        status = self.get_recovery_status()
        logger.info(f"[LogBased] Final status: {status}")

        if self.base_checkpoint_state:
            del self.base_checkpoint_state
            self.base_checkpoint_state = None

        cleanup_paths = [self.shadow_replica_path, self.anchor_latest_path]
        cleanup_paths.extend(self.shadow_flat_buffer_paths)
        cleanup_paths.extend(self.shadow_flat_adam_m_buffer_paths)
        cleanup_paths.extend(self.shadow_flat_adam_v_buffer_paths)
        if self.shadow_flat_header_path:
            cleanup_paths.append(self.shadow_flat_header_path)
            meta_paths = _shadow_flat_meta_paths(self.shadow_flat_header_path)
            cleanup_paths.append(meta_paths["state_meta_path"])
            cleanup_paths.append(meta_paths["adam_meta_path"])

        if self.rebase_payload_dir:
            for payload_path in list(self._staged_rebase_payloads):
                _cleanup_rebase_payload_flat(payload_path)
            for payload_path in glob.glob(os.path.join(self.rebase_payload_dir, "*.header.json")):
                _cleanup_rebase_payload_flat(payload_path)
            if os.path.isdir(self.rebase_payload_dir):
                try:
                    os.rmdir(self.rebase_payload_dir)
                except OSError:
                    pass

        for path in cleanup_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    logger.warning(f"[Shadow] Cleanup error for {path}: {e}")

        logger.info(
            f"[LogBased] Done. callback_saves={self.save_count}, "
            f"disk_log_saves={self.disk_log_save_count}, full_anchors={self.full_anchor_save_count}"
        )


# Compatibility aliases
ZOReplayCheckpointCallback = LogBasedCheckpointCallback
load_zo_replay_checkpoint = load_log_based_checkpoint
