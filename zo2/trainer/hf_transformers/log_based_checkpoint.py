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
  - SHADOW_COMMIT_INTERVAL=N: Shadow commits `/dev/shm` replica every N updates
      (default: 1). Rebase always forces an immediate commit.
"""

import hashlib
import logging
import math
import multiprocessing as mp
import os
import re
import threading
import time
from collections import OrderedDict

import psutil
import torch
from transformers import TrainerCallback

from .log_based_failure_injection import GPUFailureSimulator
from .log_based_resume import load_log_based_checkpoint, resume_from_log_based
from .log_based_tuning import (
    _benchmark_curves_worker,
    _interp_curve,
    calibrate_producer_consumer,
    optimize_thread_allocation,
)
from .log_based_replay import (
    _apply_single_update_with_pregenerated_z,
    _apply_single_update,
    _generate_z_for_one_step,
    _get_and_clear_replay_adam_state,
    _load_adam_state_from_base,
    _replay_updates_on_state,
    _set_replay_adam_state,
)
from .log_based_shadow import _load_shadow_replica, _shadow_process_main
from .log_based_utils import (
    _DTYPE_MAP,
    _atomic_save_state_dict_safetensors,
    _clone_state_dict_to_cpu,
    _detect_tied_weights,
    _fsync_directory,
    _fsync_file,
    _log_memory,
    _restore_tied_weights,
    _system_stats,
    _thread_snapshot,
    _tie_state_dict_inplace,
)

logger = logging.getLogger(__name__)


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
        self.base_checkpoint_path: str = None
        self.base_checkpoint_step: int = 0
        self.is_first_save = True
        self.save_count = 0

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

        # Shadow model (real-time tracking)
        self.shadow_step = 0
        self.shadow_base_step = 0
        self.shadow_adam_state = None  # Adam m/v/t on CPU (None = SGD mode)

        # Multiprocessing shadow (zero contention with training)
        self.shadow_process = None       # mp.Process
        self.update_queue = None         # mp.Queue: training → shadow
        self.shadow_step_val = None      # mp.Value('i', lock=False): latest committed step
        self.shadow_replica_path = None  # /dev/shm committed shadow replica
        self.anchor_latest_path = None   # /dev/shm async/sync anchor latest
        self.shadow_commit_interval = 1
        self._last_shadow_rebased_anchor_step = -1

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
        return f"/dev/shm/zo_shadow_latest_{self._run_hash()}.safetensors"

    def _anchor_latest_path(self):
        return f"/dev/shm/zo_anchor_latest_{self._run_hash()}.safetensors"

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

    def _publish_anchor_latest(self, state_dict, step):
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
            self.update_queue.put_nowait(
                {
                    "cmd": "rebase",
                    "step": int(step),
                    "path": path or self.anchor_latest_path,
                }
            )
        except Exception as e:
            if not getattr(self, "_queue_error_logged", False):
                logger.warning(f"[LogBased] Failed to send rebase to shadow: {e}")
                self._queue_error_logged = True

    def _refresh_shadow_from_base(self, *, step=None, commit_now=True):
        if self.base_checkpoint_state is None:
            return
        step = int(self.base_checkpoint_step if step is None else step)
        self.shadow_base_step = step
        self.shadow_step = step
        if commit_now and self.enable_shadow and self.shadow_replica_path:
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

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Called at training start"""
        self.output_dir = args.output_dir
        self.shadow_replica_path = self._shadow_replica_path() if self.enable_shadow else None
        self.anchor_latest_path = self._anchor_latest_path() if self.batch_size >= 1 else None
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
            # Fresh training: cache initial pretrained model
            self._cache_initial_model(model)

        # Sync current_step with trainer state so the hook reports correct step numbers
        self.current_step = state.global_step

        # Initialize shadow Adam state if optimizer is MeZO-Adam
        if self.enable_shadow and model is not None:
            self._init_shadow_adam_state(model)

        # Start shadow model as separate process (zero contention with training)
        if self.enable_shadow:
            self._start_shadow_process()
            if self.instant_recover:
                logger.info("[LogBased] Mode: L3 (Instant Recovery) - shadow tracking + instant recovery")
            else:
                logger.info("[LogBased] Mode: L2 (CPU Shadow) - real-time shadow tracking")
        else:
            mode_desc = self._get_mode_description()
            logger.info(f"[LogBased] Mode: L1 ({mode_desc}) - on-demand reconstruction")

        effective_gpu_fail_step = self.failure_simulator.fail_at_step
        if effective_gpu_fail_step is None:
            effective_gpu_fail_step = -1
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
            f"  GPU_FAIL_STEP={effective_gpu_fail_step}\n"
            f"  ASYNC_ANCHOR={effective_async_anchor}\n"
            f"  ZO_RNG_DEVICE={effective_rng_device}\n"
            f"  SHADOW_PIPELINE={os.environ.get('SHADOW_PIPELINE', '0')}\n"
            f"  SHADOW_PIPELINE_WORKERS={os.environ.get('SHADOW_PIPELINE_WORKERS', '2')}\n"
            f"  SHADOW_COMMIT_INTERVAL={self.shadow_commit_interval}\n"
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

    def _cache_initial_model(self, model):
        """Cache initial model to CPU memory and disk for fresh training."""
        logger.info("[LogBased] Caching initial model to CPU memory and disk...")
        t_start = time.time()

        # Detect tied weights (e.g. embed_tokens.weight <-> lm_head.weight)
        self._tied_weight_groups = _detect_tied_weights(model)
        if self._tied_weight_groups:
            logger.info(f"[LogBased] Detected tied weight groups: {self._tied_weight_groups}")

        # Capture trainable parameter names in named_parameters() order.
        # named_parameters() deduplicates tied weights and excludes buffers,
        # and we further filter by requires_grad — this exactly matches zo_update iteration.
        self._trainable_param_names = [
            name for name, p in model.named_parameters() if p.requires_grad
        ]
        total_params = sum(1 for _ in model.named_parameters())
        logger.info(f"[LogBased] Trainable params: {len(self._trainable_param_names)} / {total_params}")

        self.base_checkpoint_state = _clone_state_dict_to_cpu(model.state_dict())
        self.base_checkpoint_step = 0
        self.base_checkpoint_path = "__initial__"  # Special marker for initial state
        self.shadow_base_step = 0
        self.shadow_step = 0

        if self.enable_shadow:
            self._refresh_shadow_from_base(step=0, commit_now=True)
        if self.anchor_latest_path:
            self._publish_anchor_latest(self.base_checkpoint_state, 0)

        # Save initial model to output_dir for recovery (avoids HuggingFace re-download)
        # Strip tied weight duplicates to match HuggingFace save_pretrained() convention.
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
                save_file(save_state, save_path)
                _fsync_file(save_path)
                logger.info(f"[LogBased] Initial model saved to {initial_model_dir} (safetensors, excluded {len(excluded)} tied keys)")
            except ImportError:
                save_path = os.path.join(initial_model_dir, "pytorch_model.bin")
                torch.save(save_state, save_path)
                _fsync_file(save_path)
                logger.info(f"[LogBased] Initial model saved to {initial_model_dir} (pytorch_model.bin, excluded {len(excluded)} tied keys)")

        mem_mb = self._get_memory_size()
        t_elapsed = time.time() - t_start
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

        # Load optimizer.pt once for all metadata (pending_grad + update_history + base info)
        opt_state = None
        opt_path = os.path.join(log_based_resume, "optimizer.pt")
        if os.path.exists(opt_path):
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

        self.base_checkpoint_state = _clone_state_dict_to_cpu(model.state_dict())
        self.shadow_base_step = int(state.global_step)
        self.shadow_step = int(state.global_step)
        if self.enable_shadow:
            self._refresh_shadow_from_base(step=state.global_step, commit_now=True)
        if self.anchor_latest_path and self.batch_size >= 1:
            self._publish_anchor_latest(self.base_checkpoint_state, state.global_step)

        self.is_first_save = False
        mem_mb = self._get_memory_size()
        t_elapsed = time.time() - t_start
        logger.info(f"[LogBased Resume] Initialized ({mem_mb:.1f} MB) in {t_elapsed:.3f}s")

    def _init_shadow_adam_state(self, model):
        """Initialize shadow_adam_state if the optimizer is MeZO-Adam.

        Detects whether the model's optimizer is Adam-based and creates a fresh
        Adam state (m={}, v={}, t=0) on CPU. For resume, the Adam state is loaded
        from the base checkpoint via _load_adam_state_from_base().
        """
        opt = getattr(model, 'opt', None)
        if opt is None:
            self.shadow_adam_state = None
            return

        # Check if optimizer is Adam variant (has adam_betas attribute)
        betas = getattr(opt, 'adam_betas', None)
        if betas is None:
            self.shadow_adam_state = None
            logger.info("[Shadow] Optimizer is SGD, no Adam state needed")
            return

        adam_eps = getattr(opt, 'adam_eps', 1e-8)
        self.shadow_adam_state = {
            'm': {},
            'v': {},
            't': 0,
            'betas': betas,
            'adam_eps': adam_eps,
        }
        logger.info(f"[Shadow] Initialized Adam state: betas={betas}, eps={adam_eps}")

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
        logger.info(f"[Shadow Thread Config] total_cores={total_cores} "
                    f"reserve={n_reserve} consumer(ATen)={aten_threads} "
                    f"producer(zo_rng)={c_prod}")
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

        self.shadow_process = ctx.Process(
            target=_shadow_process_main,
            args=(
                self.update_queue,
                self.base_checkpoint_state,
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
            ),
            daemon=True,
        )

        self.shadow_process.start()

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
                    f"child OMP_NUM_THREADS={aten_threads})")
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
                        logger.info(
                            f"[ShadowSend] step={actual_step} enqueue={shadow_send_s * 1000.0:.3f}ms"
                        )
                    except Exception as e:
                        if not getattr(self, '_queue_error_logged', False):
                            logger.warning(f"[LogBased] Failed to send update to shadow: {e}")
                            self._queue_error_logged = True
                # Thread diagnostics
                if actual_step == 1:
                    _thread_snapshot("Train step=1 AFTER_ZO_FORWARD", detail=True)
                # DEBUG: model checksum after each step
                _cksum = sum(p.data.float().sum().item() for p in model.parameters())
                logger.info(f"[CKSUM] step={actual_step} checksum={_cksum:.10e}")
                if actual_step == 1:
                    _thread_snapshot("Train step=1 AFTER_CKSUM", detail=True)
                if actual_step <= 20 or actual_step % 50 == 0:
                    _thread_snapshot(f"Train step={actual_step}")
                if applied_grad != 0:
                    logger.info(f"[HOOK] step={actual_step}, UPDATE RECORDED: seed={update['seed']}, "
                                f"applied_grad={update['grad']:.6e}, new_grad={new_grad:.6e}, lr={lr}, wd={wd}")
                    # Verification tag: grep "[VERIFY]" to cross-check train vs replay
                    logger.info(f"[VERIFY] step={actual_step} total_updates={len(self.update_history)} "
                                f"pending_grad={self._pending_grad:.6e}")
                else:
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

        if self.failure_simulator.check_and_fail(self.current_step, model):
            self.failure_simulator.trigger_failure(model)
            # Unreachable — SIGKILL kills process

    def on_step_end(self, args, state, control, **kwargs):
        """Called at end of each step"""
        self.current_step = state.global_step

        if self.enable_shadow:
            async_anchor = getattr(self, "_async_anchor", None) or getattr(self.trainer, "_async_anchor", None)
            if async_anchor is not None:
                published_step = async_anchor.get_latest_published_anchor_step()
                if published_step > self._last_shadow_rebased_anchor_step:
                    self._queue_shadow_rebase(published_step, self.anchor_latest_path)
                    self._last_shadow_rebased_anchor_step = published_step

        if self.current_step % 10 == 0:
            num_updates = len(self.update_history)
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
        """Recover from the latest committed shadow replica on /dev/shm."""
        t_start = time.time()

        if self.enable_shadow and self.shadow_replica_path and os.path.exists(self.shadow_replica_path):
            recovered, base_step, shadow_step = _load_shadow_replica(
                self.shadow_replica_path,
                tied_groups=getattr(self, "_tied_weight_groups", []),
            )
            self.shadow_base_step = base_step
            self.shadow_step = shadow_step

            t_elapsed = time.time() - t_start
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

        with self.update_lock:
            updates = self.update_history.copy()

        num_updates = len(updates)
        logger.info(f"[Recovery] Reconstructing on-demand: replaying {num_updates} updates...")

        # Copy base checkpoint state
        reconstructed = OrderedDict()
        for key, value in self.base_checkpoint_state.items():
            reconstructed[key] = value.clone()

        # Tie weights before replay so updates accumulate correctly (like during training)
        if hasattr(self, '_tied_weight_groups') and self._tied_weight_groups:
            _tie_state_dict_inplace(reconstructed, self._tied_weight_groups)

        # Get zo_eps for old update records that don't include it
        fallback_zo_eps = 0.0
        if self.trainer and hasattr(self.trainer, 'model') and hasattr(self.trainer.model, 'opt'):
            fallback_zo_eps = getattr(self.trainer.model.opt, 'zo_eps', 0.0)

        # Replay all updates using only trainable param names (matching zo_update iteration)
        _replay_updates_on_state(
            reconstructed, updates,
            trainable_param_names=self._trainable_param_names,
            default_zo_eps=fallback_zo_eps
        )
        if num_updates > 0:
            logger.info(f"[Recovery] Replayed {num_updates} updates")

        t_elapsed = time.time() - t_start
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
            shadow_available = bool(self.shadow_replica_path and os.path.exists(self.shadow_replica_path))
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
        self.base_checkpoint_state = _clone_state_dict_to_cpu(model.state_dict())
        self.base_checkpoint_step = int(step)
        if self.batch_size >= 1:
            anchor_path = self._publish_anchor_latest(self.base_checkpoint_state, step)
            if self.enable_shadow and anchor_path is not None:
                self._queue_shadow_rebase(step, anchor_path)
                self._last_shadow_rebased_anchor_step = int(step)

    def on_async_anchor_persisted(self, step, checkpoint_path):
        self.base_checkpoint_path = checkpoint_path
        self.base_checkpoint_step = int(step)
        with self.update_lock:
            for u in reversed(self.update_history):
                if u['step'] <= step:
                    self._base_pending_seed = u['seed']
                    break
            self.update_history = [u for u in self.update_history if u['step'] > step]
        if self.enable_shadow and self.anchor_latest_path and os.path.exists(self.anchor_latest_path):
            self._queue_shadow_rebase(step, self.anchor_latest_path)
            self._last_shadow_rebased_anchor_step = max(self._last_shadow_rebased_anchor_step, int(step))

    def _get_memory_size(self):
        if self.base_checkpoint_state is None:
            return 0
        total = sum(v.numel() * v.element_size() for v in self.base_checkpoint_state.values())
        return total / (1024 * 1024)

    def _log_memory_status(self):
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
            logger.info(
                "[AsyncAnchor] Summary: "
                f"enqueue_cpu count={async_stats['enqueue_cpu_count']} avg={async_stats['avg_enqueue_cpu_time']:.3f}s | "
                f"d2h count={async_stats['d2h_count']} avg={async_stats['avg_d2h_time']:.3f}s | "
                f"cpu_total count={async_stats['cpu_persist_total_count']} "
                f"avg={async_stats['avg_cpu_persist_total_time']:.3f}s"
            )

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
            logger.info(
                f"[ShadowSend] Summary: count={len(shadow_send_times)} avg={avg_shadow_send_ms:.3f}ms"
            )

        checkpoint_total_times = self.timing_stats.get('checkpoint_total_times', [])
        if checkpoint_total_times and not self.enable_shadow:
            avg_checkpoint_s = sum(checkpoint_total_times) / len(checkpoint_total_times)
            logger.info(
                f"[ZOTrainer] Checkpoint Summary: count={len(checkpoint_total_times)} avg={avg_checkpoint_s:.3f}s"
            )

        status = self.get_recovery_status()
        logger.info(f"[LogBased] Final status: {status}")

        if self.base_checkpoint_state:
            del self.base_checkpoint_state
            self.base_checkpoint_state = None

        for path in [self.shadow_replica_path, self.anchor_latest_path]:
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
