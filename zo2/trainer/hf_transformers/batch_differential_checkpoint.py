"""
Batch Differential Checkpoint for ZO Training:
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
"""

import os
import re
import threading
import multiprocessing as mp
import queue as queue_module   # for queue.Empty in shadow process
import time
import math
import torch
from transformers import TrainerCallback
from collections import OrderedDict
import logging
import json
import psutil

logger = logging.getLogger(__name__)


def _thread_snapshot(label, _logger=None, detail=False):
    """Print a snapshot of all thread pools in the current process."""
    pid = os.getpid()
    try:
        os_thr = len(os.listdir(f'/proc/{pid}/task'))
    except Exception:
        os_thr = -1
    aten = torch.get_num_threads()
    interop = torch.get_num_interop_threads()
    zo_thr = -1
    try:
        import zo_rng as _zr
        zo_thr = _zr.get_num_threads()
    except ImportError:
        pass
    msg = (f"[ThreadSnap] {label}: os_thr={os_thr} "
           f"aten={aten} zo_rng={zo_thr} interop={interop}")
    if detail:
        from collections import Counter
        names = Counter()
        try:
            for tid in os.listdir(f'/proc/{pid}/task'):
                try:
                    with open(f'/proc/{pid}/task/{tid}/comm') as f:
                        names[f.read().strip()] += 1
                except Exception:
                    names['<unknown>'] += 1
        except Exception:
            pass
        if names:
            parts = [f"{name}={cnt}" for name, cnt in names.most_common()]
            msg += f"\n  [ThreadDetail] {' '.join(parts)}"
        # Also list Python-level threading threads to identify the "python" ones
        import threading
        py_threads = [(t.name, t.ident, t.daemon) for t in threading.enumerate()]
        msg += f"\n  [PyThreads] count={len(py_threads)}"
        for tname, tid, daemon in py_threads:
            msg += f"\n    {tname} (tid={tid}, daemon={daemon})"
    _log = _logger or logger
    _log.info(msg)
    return os_thr


def _fsync_file(path):
    """Flush file to disk if output_dir is on a local filesystem (e.g. /tmp).
    Triggered by the FORCE_FSYNC=1 environment variable."""
    if os.environ.get('FORCE_FSYNC', '0') == '1':
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _fsync_directory(dirpath):
    """Fsync all files in a checkpoint directory."""
    if os.environ.get('FORCE_FSYNC', '0') != '1':
        return
    for fname in os.listdir(dirpath):
        fpath = os.path.join(dirpath, fname)
        if os.path.isfile(fpath):
            _fsync_file(fpath)

_DTYPE_MAP = {
    'torch.float16': torch.float16,
    'torch.bfloat16': torch.bfloat16,
    'torch.float32': torch.float32,
}


def _detect_tied_weights(model) -> list:
    """
    Detect groups of tied (shared) parameters in a model by checking data_ptr equality.

    Uses named_modules + _parameters to bypass PyTorch's named_parameters() deduplication,
    which would hide tied weights (e.g. embed_tokens.weight and lm_head.weight).

    Returns:
        List of lists, where each inner list contains parameter names that share
        the same underlying tensor. Only groups with 2+ parameters are returned.
    """
    ptr_to_names = {}
    for module_name, module in model.named_modules():
        for param_name, param in module._parameters.items():
            if param is None:
                continue
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            ptr = param.data_ptr()
            if ptr not in ptr_to_names:
                ptr_to_names[ptr] = []
            ptr_to_names[ptr].append(full_name)

    return [names for names in ptr_to_names.values() if len(names) > 1]


def _tie_state_dict_inplace(state: OrderedDict, tied_groups: list) -> None:
    """
    Make tied parameter groups share the same tensor in a state dict.

    During ZO training, tied weights (e.g. embed_tokens.weight and lm_head.weight)
    are the same tensor, so updates to both accumulate. When replaying from a cloned
    state dict, they become separate tensors and updates don't accumulate correctly.
    This function re-ties them so replay matches training behavior.

    Args:
        state: State dict to modify in-place
        tied_groups: List of lists of parameter names that should share the same tensor
    """
    for group in tied_groups:
        # Find the primary key (first one present in state)
        primary = None
        for name in group:
            if name in state:
                primary = name
                break

        if primary is None:
            continue

        # Make all other keys in the group reference the primary tensor.
        # Also adds missing keys (e.g. lm_head.weight stripped during save).
        for name in group:
            if name != primary:
                state[name] = state[primary]


class GPUFailureSimulator:
    """GPU failure simulator"""

    def __init__(self):
        self.fail_at_step = None
        self.has_failed = False
        self.callback = None  # BatchDiffCheckpointCallback reference

    def set_fail_step(self, step: int):
        """Set step to simulate GPU failure"""
        self.fail_at_step = step
        self.has_failed = False
        logger.info(f"[GPU Failure] Will simulate failure at step {step}")

    def check_and_fail(self, current_step: int, model):
        """Check if should trigger failure"""
        if self.fail_at_step is not None and current_step >= self.fail_at_step and not self.has_failed:
            self.has_failed = True
            logger.warning(f"[GPU Failure] Simulating GPU failure at step {current_step}!")
            return True
        return False

    def trigger_failure(self, model):
        """Trigger GPU failure (clear GPU memory)"""
        logger.warning("[GPU Failure] Clearing GPU memory...")

        if hasattr(model, 'to'):
            model.to('cpu')

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"[GPU Failure] GPU memory after clear: {allocated:.2f} GB")

        return True


class BatchDiffCheckpointCallback(TrainerCallback):
    """
    Batch Differential Checkpoint Callback:

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
        self.shadow_model: OrderedDict = None
        self.shadow_step = 0
        self.shadow_lock = threading.Lock()
        self.shadow_thread = None
        self.shadow_running = False
        self.shadow_adam_state = None  # Adam m/v/t on CPU (None = SGD mode)

        # Multiprocessing shadow (zero contention with training)
        self.shadow_process = None       # mp.Process
        self.update_queue = None         # mp.Queue: training → shadow
        self.shadow_step_val = None      # mp.Value('i', lock=False): shared step counter
        self.shadow_shared = None        # OrderedDict of shared-memory tensors
        self.recovery_req = None         # mp.Event: training requests recovery pause
        self.recovery_ready = None       # mp.Event: shadow signals "paused, safe to read"
        self.recovery_done = None        # mp.Event: training signals "done reading, resume"
        self.shadow_stop_event = None    # mp.Event: immediate stop signal for shadow process

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
            'recoveries': []
        }

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Called at training start"""
        self.output_dir = args.output_dir

        if 'trainer' in kwargs:
            self.trainer = kwargs['trainer']

        # batch_size=-1: disabled, use default Trainer checkpoint — skip all setup
        if self.batch_size < 0:
            logger.info("[BatchDiff] batch_size=-1, disabled (using default Trainer checkpoint)")
            logger.info(f"[BatchDiff Config] BATCHDIFF_CKPT={self.batch_size}")
            return

        if self.trainer and hasattr(self.trainer, 'zo') and self.trainer.zo:
            if not self._hook_registered:
                self.trainer.register_zo2_training_step_post_hook(self._zo_update_hook)
                self._hook_registered = True
                logger.info("[BatchDiff] Registered post-hook")

        # Detect model dtype for replay consistency
        if model is not None and self.model_dtype is None:
            for p in model.parameters():
                self.model_dtype = str(p.dtype)  # e.g. "torch.float16"
                break
            logger.info(f"[BatchDiff] Detected model dtype: {self.model_dtype}")

        batchdiff_resume = getattr(args, 'batchdiff_resume', '')

        if batchdiff_resume and model is not None:
            # Resume: model already reconstructed via replay — don't cache from scratch
            self._init_for_resume(model, state, batchdiff_resume)
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
                logger.info("[BatchDiff] Mode: L3 (Instant Recovery) - shadow tracking + instant recovery")
            else:
                logger.info("[BatchDiff] Mode: L2 (CPU Shadow) - real-time shadow tracking")
        else:
            mode_desc = self._get_mode_description()
            logger.info(f"[BatchDiff] Mode: L1 ({mode_desc}) - on-demand reconstruction")

        # Log all configuration env vars
        logger.info(
            f"[BatchDiff Config]\n"
            f"  BATCHDIFF_CKPT={self.batch_size}\n"
            f"  ENABLE_SHADOW={self.enable_shadow}\n"
            f"  INSTANT_RECOVER={self.instant_recover}\n"
            f"  SHADOW_PIPELINE={os.environ.get('SHADOW_PIPELINE', '0')}\n"
            f"  SHADOW_PIPELINE_WORKERS={os.environ.get('SHADOW_PIPELINE_WORKERS', '2')}\n"
            f"  SHADOW_RESERVE_THREADS={os.environ.get('SHADOW_RESERVE_THREADS', '1')}\n"
            f"  SHADOW_CONSUMER_THREADS={os.environ.get('SHADOW_CONSUMER_THREADS', 'auto')}\n"
            f"  ZO_RNG_DEVICE={os.environ.get('ZO_RNG_DEVICE', 'native')}\n"
            f"  BATCHDIFF_SIMULATE_PERTURBATION={os.environ.get('BATCHDIFF_SIMULATE_PERTURBATION', '1')}\n"
            f"  ASYNC_ANCHOR={os.environ.get('ASYNC_ANCHOR', '0')}\n"
            f"  FORCE_FSYNC={os.environ.get('FORCE_FSYNC', '0')}\n"
            f"  GPU_FAIL_STEP={os.environ.get('GPU_FAIL_STEP', '-1')}\n"
            f"  BATCHDIFF_REPLAY_DEVICE={os.environ.get('BATCHDIFF_REPLAY_DEVICE', 'cuda')}\n"
            f"  BATCHDIFF_REPLAY_FP32={os.environ.get('BATCHDIFF_REPLAY_FP32', '0')}\n"
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
        logger.info("[BatchDiff] Caching initial model to CPU memory and disk...")
        t_start = time.time()

        # Detect tied weights (e.g. embed_tokens.weight <-> lm_head.weight)
        self._tied_weight_groups = _detect_tied_weights(model)
        if self._tied_weight_groups:
            logger.info(f"[BatchDiff] Detected tied weight groups: {self._tied_weight_groups}")

        # Capture trainable parameter names in named_parameters() order.
        # named_parameters() deduplicates tied weights and excludes buffers,
        # and we further filter by requires_grad — this exactly matches zo_update iteration.
        self._trainable_param_names = [
            name for name, p in model.named_parameters() if p.requires_grad
        ]
        total_params = sum(1 for _ in model.named_parameters())
        logger.info(f"[BatchDiff] Trainable params: {len(self._trainable_param_names)} / {total_params}")

        state_dict = model.state_dict()

        self.base_checkpoint_state = OrderedDict()
        for key, value in state_dict.items():
            t = value.detach().cpu().clone()
            if self.enable_shadow:
                t.share_memory_()  # POSIX shm: shadow process reads via DMA
            self.base_checkpoint_state[key] = t

        # Initialize shadow model
        if self.enable_shadow:
            self._refresh_shadow_from_base()
            self.shadow_step = 0  # Override: no updates yet at init

        self.base_checkpoint_step = 0
        self.base_checkpoint_path = "__initial__"  # Special marker for initial state

        # Save initial model to output_dir for recovery (avoids HuggingFace re-download)
        # Strip tied weight duplicates to match HuggingFace save_pretrained() convention.
        if self.output_dir:
            initial_model_dir = os.path.join(self.output_dir, "initial_model")
            os.makedirs(initial_model_dir, exist_ok=True)
            # Compute excluded keys from tied groups (keep first, exclude rest)
            excluded = set()
            for group in self._tied_weight_groups:
                for name in group[1:]:
                    excluded.add(name)
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
                logger.info(f"[BatchDiff] Initial model saved to {initial_model_dir} (safetensors, excluded {len(excluded)} tied keys)")
            except ImportError:
                save_path = os.path.join(initial_model_dir, "pytorch_model.bin")
                torch.save(save_state, save_path)
                _fsync_file(save_path)
                logger.info(f"[BatchDiff] Initial model saved to {initial_model_dir} (pytorch_model.bin, excluded {len(excluded)} tied keys)")

        mem_mb = self._get_memory_size()
        t_elapsed = time.time() - t_start
        logger.info(f"[BatchDiff] Initial model cached ({mem_mb:.1f} MB) in {t_elapsed:.3f}s")
        self._log_memory_status()

    def _init_for_resume(self, model, state, batchdiff_resume):
        """Initialize callback state from an already-reconstructed (resumed) model.
        Unlike _cache_initial_model (for fresh training), this:
        - Loads update_history and base info from the checkpoint's optimizer.pt
        - Sets base_checkpoint_step correctly for batch_size>=1
        - Sets shadow_step = len(update_history) so shadow starts caught up
        """
        logger.info("[BatchDiff Resume] Initializing from resumed model...")
        t_start = time.time()

        # Detect tied weights
        self._tied_weight_groups = _detect_tied_weights(model)
        if self._tied_weight_groups:
            logger.info(f"[BatchDiff] Detected tied weight groups: {self._tied_weight_groups}")

        # Cache trainable param names
        self._trainable_param_names = [
            name for name, p in model.named_parameters() if p.requires_grad
        ]

        # Load optimizer.pt once for all metadata (pending_grad + update_history + base info)
        opt_state = None
        opt_path = os.path.join(batchdiff_resume, "optimizer.pt")
        if os.path.exists(opt_path):
            try:
                opt_state = torch.load(opt_path, map_location='cpu', weights_only=False)
                if not isinstance(opt_state, dict) or 'zo_update_history' not in opt_state:
                    opt_state = None
            except Exception as e:
                logger.warning(f"[BatchDiff Resume] Failed to load optimizer.pt: {e}")

        # Restore pending_grad
        if hasattr(model, 'opt') and opt_state is not None:
            pg = opt_state.get('pending_grad', None)
            if pg is not None:
                model.opt.projected_grad = float(pg)
                logger.info(f"[BatchDiff Resume] Restored pending_grad={float(pg):.6e}")

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
                            logger.info(f"[BatchDiff Resume] No pending_seed in checkpoint, "
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
                    logger.info(f"[BatchDiff Resume] Reconstructed last_rstate from seed={ps}")
            else:
                logger.warning("[BatchDiff Resume] No pending_grad found, "
                               "first step will skip zo_update (projected_grad=0)")

        # Restore Adam state (only when optimizer is MeZOAdam)
        if hasattr(model, 'opt') and hasattr(model.opt, 'm'):
            # Priority: replayed adam state (log checkpoint) > checkpoint adam state (full checkpoint)
            replay_adam = _get_and_clear_replay_adam_state()
            if replay_adam:
                model.opt.restore_adam_state(replay_adam)
                logger.info(f"[BatchDiff Resume] Restored replayed Adam state: t={replay_adam.get('t', 0)}")
            elif opt_state and 'adam_state' in opt_state:
                model.opt.restore_adam_state(opt_state['adam_state'])
                logger.info(f"[BatchDiff Resume] Restored Adam state from checkpoint: "
                            f"t={opt_state['adam_state'].get('t', 0)}")

        # Load update history and base info
        if opt_state is not None:
            is_full_ckpt = opt_state.get('is_full_checkpoint', False)
            self._base_pending_seed = opt_state.get('base_pending_seed', 0)

            if self.batch_size >= 1:
                if is_full_ckpt:
                    # Full checkpoint: this IS the new base, history starts empty
                    self.base_checkpoint_path = batchdiff_resume
                    self.base_checkpoint_step = state.global_step
                    self.update_history = []
                    logger.info(f"[BatchDiff Resume] Full checkpoint → new base at step {state.global_step}")
                else:
                    # Log checkpoint: inherit base and history from metadata
                    self.update_history = list(opt_state['zo_update_history'])
                    self.base_checkpoint_path = opt_state.get('base_checkpoint', '__initial__')
                    match = re.search(r'checkpoint-(\d+)', str(self.base_checkpoint_path))
                    self.base_checkpoint_step = int(match.group(1)) if match else 0
                    logger.info(f"[BatchDiff Resume] Log checkpoint → base={self.base_checkpoint_path} "
                               f"(step {self.base_checkpoint_step}), {len(self.update_history)} updates")
            else:
                # batch_size=0: always accumulate all updates from initial model
                self.update_history = list(opt_state['zo_update_history'])
                logger.info(f"[BatchDiff Resume] Loaded {len(self.update_history)} updates (accumulative)")

        # batch_size=0: base is always __initial__
        if self.batch_size == 0:
            self.base_checkpoint_path = "__initial__"
            self.base_checkpoint_step = 0

        # Cache current model state to CPU (for shadow model tracking)
        state_dict = model.state_dict()
        self.base_checkpoint_state = OrderedDict()
        for key, value in state_dict.items():
            self.base_checkpoint_state[key] = value.detach().cpu().clone()

        # Initialize shadow (shadow_step = len(update_history) → starts caught up)
        if self.enable_shadow:
            self._refresh_shadow_from_base()

        self.is_first_save = False
        mem_mb = self._get_memory_size()
        t_elapsed = time.time() - t_start
        logger.info(f"[BatchDiff Resume] Initialized ({mem_mb:.1f} MB) in {t_elapsed:.3f}s")

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

    def _refresh_shadow_from_base(self):
        """Refresh shadow model from base_checkpoint_state.
        Creates shared-memory tensors for multiprocessing shadow."""
        self.shadow_shared = OrderedDict()
        for key, value in self.base_checkpoint_state.items():
            t = value.clone()
            t.share_memory_()  # POSIX shm: shadow process writes in-place
            self.shadow_shared[key] = t
        if hasattr(self, '_tied_weight_groups') and self._tied_weight_groups:
            _tie_state_dict_inplace(self.shadow_shared, self._tied_weight_groups)
        # Keep shadow_model as alias for compatibility (status/logging/recovery)
        self.shadow_model = self.shadow_shared
        self.shadow_step = len(self.update_history)

    def _start_shadow_process(self):
        """Start shadow model as a separate process (zero GIL/lock contention).

        The shadow process runs independently with its own GIL. Communication:
        - update_queue (mp.Queue): training → shadow, update dicts
        - shadow_step_val (mp.Value): shared atomic counter, lock-free
        - shadow_shared (POSIX shm tensors): shadow process writes in-place
        - recovery Events: pause/resume protocol for safe cloning
        """
        if self.shadow_process is not None and self.shadow_process.is_alive():
            return

        _thread_snapshot("Train BEFORE_SHADOW_SPAWN")
        use_pipeline = os.environ.get('SHADOW_PIPELINE', '0') == '1'
        P = int(os.environ.get('SHADOW_PIPELINE_WORKERS', '2'))
        simulate_perturbation = os.environ.get('BATCHDIFF_SIMULATE_PERTURBATION', '1') == '1'

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

        param_names = self._trainable_param_names or list(self.shadow_shared.keys())

        # Use spawn context: creates a *fresh* Python process (no inherited
        # corrupted ATen/OpenMP thread pool from fork).  One-time import cost
        # (~15-30s) but full multi-threaded tensor ops throughout training.
        ctx = mp.get_context('spawn')
        self.update_queue = ctx.Queue()
        self.shadow_step_val = ctx.Value('i', 0, lock=False)  # lock-free atomic
        self.recovery_req = ctx.Event()
        self.recovery_ready = ctx.Event()
        self.recovery_done = ctx.Event()
        self.shadow_stop_event = ctx.Event()

        # Pre-set OMP_NUM_THREADS so the spawned child's `import torch` creates
        # exactly the right number of libgomp threads (spawn inherits parent env).
        # If SHADOW_NUMA_NODE is set, use that NUMA node's core count instead of
        # all cores, so libgomp doesn't over-provision threads outside the pinned set.
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
        # Set thread-pool env vars for the spawned child.
        # OMP_NUM_THREADS = aten_threads (ATen element-wise ops need this).
        # MKL/BLAS threads = 1: shadow does no matmul/BLAS, so MKL's libiomp5
        # pool is pure waste — eliminates ~64 rogue threads.
        # KMP_BLOCKTIME=0: any Intel OMP threads sleep immediately (no spin).
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
            args=(self.update_queue, self.base_checkpoint_state, self.shadow_shared,
                  self.shadow_step_val,
                  self.recovery_req, self.recovery_ready, self.recovery_done,
                  self.shadow_stop_event,
                  param_names, rng_device, simulate_perturbation,
                  default_zo_eps, adam_config, use_pipeline, P),
            daemon=True,
        )

        self.shadow_process.start()

        # Restore parent env (training process keeps its own settings)
        for k in _thread_env_keys:
            if _old_env[k] is not None:
                os.environ[k] = _old_env[k]
            else:
                os.environ.pop(k, None)

        # Optionally reduce training process zo_rng threads to avoid bandwidth
        # competition with shadow.  Default pool = hardware_concurrency (128),
        # but training is mostly GPU-bound, so fewer CPU threads suffice.
        # Set ZO_RNG_TRAIN_THREADS=16 (or similar) to enable.
        if rng_device == "zo_rng":
            _train_zo = os.environ.get('ZO_RNG_TRAIN_THREADS')
            if _train_zo is not None:
                _train_zo = int(_train_zo)
                try:
                    import zo_rng as _zr_parent
                    _zr_parent.set_num_threads(_train_zo)
                    logger.info(f"[BatchDiff] Training zo_rng threads reduced to {_train_zo}")
                except ImportError:
                    pass

        logger.info(f"[BatchDiff] Started shadow process (PID={self.shadow_process.pid}, "
                    f"pipeline={use_pipeline}, P={P}, rng={rng_device}, "
                    f"child OMP_NUM_THREADS={aten_threads})")
        _thread_snapshot("Train AFTER_SHADOW_SPAWN")

    def _apply_update_to_shadow(self, update):
        """Apply one update to shadow model (always on CPU).
        Note: When rng_device="native" and training runs on CUDA, the shadow model will use
        CPU RNG which produces different z sequences than CUDA RNG. This is a known limitation —
        the shadow is approximate in this case. Use rng_device="cpu" for exact shadow consistency.
        """
        with self.shadow_lock:
            param_names = self._trainable_param_names if self._trainable_param_names else list(self.shadow_model.keys())
            # Detect rng_device from optimizer
            _rng_device = 'native'
            if self.trainer is not None and hasattr(self.trainer, 'model'):
                _opt = getattr(self.trainer.model, 'opt', None)
                if _opt is not None:
                    _rng_device = getattr(_opt, 'rng_device', 'native')
            # Shadow stays on CPU — never move to CUDA to avoid RNG race with main thread
            _apply_single_update(self.shadow_model, update, param_names,
                                 rng_device=_rng_device,
                                 adam_state=self.shadow_adam_state)

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
                        self.update_queue.put_nowait(update)
                    except Exception as e:
                        if not getattr(self, '_queue_error_logged', False):
                            logger.warning(f"[BatchDiff] Failed to send update to shadow: {e}")
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
        """Check for GPU failure BEFORE ZO forward (so batch isn't consumed).

        By doing recovery here (before zo2_training_step), the data batch is
        already fetched but not consumed.  After replay the model is at the
        pre-failure step, and the ZO forward runs normally on the correct
        batch → bitwise-identical continuation.
        """
        if model is None:
            return

        if self.failure_simulator.check_and_fail(self.current_step, model):
            self.failure_simulator.trigger_failure(model)

            if self.instant_recover:
                if not self.enable_shadow:
                    logger.warning("[Recovery] instant_recover requires enable_shadow=True!")
                    return
                self._instant_recover_with_replay(model)
            else:
                logger.info("[GPU Failure] Failure injected but instant_recover=False, no recovery performed")

    def on_step_end(self, args, state, control, **kwargs):
        """Called at end of each step"""
        self.current_step = state.global_step

        if self.current_step % 10 == 0:
            num_updates = len(self.update_history)
            if self.enable_shadow:
                shadow_step = self.shadow_step_val.value if self.shadow_step_val else self.shadow_step
                # Health check: detect dead shadow process (log once)
                if self.shadow_process is not None and not self.shadow_process.is_alive():
                    if not getattr(self, '_shadow_death_logged', False):
                        logger.error(f"[BatchDiff] Shadow process DEAD (exitcode={self.shadow_process.exitcode})")
                        self._shadow_death_logged = True
                logger.info(f"[BatchDiff] GPU step {self.current_step}, Shadow step {shadow_step}, "
                           f"Updates: {num_updates}")
            else:
                logger.info(f"[BatchDiff] GPU step {self.current_step}, Updates: {num_updates} (on-demand mode)")

    def recover_from_shadow(self) -> OrderedDict:
        """
        Recover from shadow model (shared memory) or update_history.
        Uses Event protocol to pause shadow process for safe cloning.
        Returns state_dict that can be directly load_state_dict.
        """
        t_start = time.time()

        if self.enable_shadow and self.shadow_shared is not None:
            # Request shadow process to pause
            if self.shadow_process is not None and self.shadow_process.is_alive():
                self.recovery_req.set()
                if not self.recovery_ready.wait(timeout=10.0):
                    logger.warning("[Recovery] Shadow process did not respond to pause request")
            elif self.shadow_model is None:
                logger.error("[Recovery] No shadow model available!")
                return None

            shadow_step = self.shadow_step_val.value if self.shadow_step_val else self.shadow_step

            # Clone shared memory tensors (shadow process is paused, safe to read)
            recovered = OrderedDict()
            for key, value in self.shadow_shared.items():
                recovered[key] = value.clone()

            # Resume shadow process
            if self.shadow_process is not None and self.shadow_process.is_alive():
                self.recovery_done.set()

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

    def _instant_recover_with_replay(self, model):
        """Full instant recovery: shadow clone → GPU replay → opt restore.

        Called from on_step_begin (BEFORE ZO forward) so the current data
        batch is not consumed.  After this method the model is at the
        pre-failure step with correct opt state; the training loop's next
        zo2_training_step will execute the correct step on the correct batch.
        """
        t_total_start = time.time()

        # ---- Phase 1: Clone shadow weights ----
        recovered_state = self.recover_from_shadow()
        if recovered_state is None:
            logger.error("[Recovery] recover_from_shadow returned None, aborting")
            return
        shadow_step = self.shadow_step_val.value if self.shadow_step_val else self.shadow_step
        t_clone = time.time() - t_total_start

        # ---- DEBUG: checksum of shadow clone BEFORE replay ----
        # Use _trainable_param_names to match model.parameters() (deduplicates tied weights)
        _pn = self._trainable_param_names or list(recovered_state.keys())
        _shadow_cksum = sum(recovered_state[n].float().sum().item() for n in _pn if n in recovered_state)
        logger.info(f"[Recovery DEBUG] Shadow clone checksum (step {shadow_step}, {len(_pn)} params): {_shadow_cksum:.10e}")

        # ---- Phase 2: GPU replay ----
        replay_updates = self.update_history[shadow_step:]
        num_replay = len(replay_updates)
        logger.info(f"[Recovery] Shadow at step {shadow_step}, "
                     f"replaying {num_replay} updates to step {self.current_step}")

        t_replay_start = time.time()

        # Detect replay parameters (same logic as _start_shadow_process)
        rng_device = 'native'
        default_zo_eps = 0.0
        if self.trainer and hasattr(self.trainer, 'model'):
            _opt = getattr(self.trainer.model, 'opt', None)
            if _opt:
                rng_device = getattr(_opt, 'rng_device', 'native')
                default_zo_eps = getattr(_opt, 'zo_eps', 0.0)
        simulate_perturbation = os.environ.get('BATCHDIFF_SIMULATE_PERTURBATION', '1') == '1'

        # Tie weights for correct replay
        if hasattr(self, '_tied_weight_groups') and self._tied_weight_groups:
            _tie_state_dict_inplace(recovered_state, self._tied_weight_groups)

        param_names = self._trainable_param_names or list(recovered_state.keys())

        if num_replay > 0:
            _replay_updates_on_state(
                recovered_state, replay_updates,
                device='cuda', move_to_device=True,
                trainable_param_names=param_names,
                default_zo_eps=default_zo_eps,
                simulate_perturbation=simulate_perturbation,
                rng_device=rng_device,
                zo2_mode=False,
            )
        t_replay = time.time() - t_replay_start

        # ---- DEBUG: checksum after replay ----
        _cksum = sum(recovered_state[n].float().sum().item() for n in param_names if n in recovered_state)
        logger.info(f"[Recovery DEBUG] Checksum after replay (step {self.current_step}, {len(param_names)} params): {_cksum:.10e}")

        # ---- Phase 3: Load into model ----
        t_load_start = time.time()
        model.load_state_dict(recovered_state)
        if next(model.parameters()).device.type != 'cuda':
            model.to('cuda')
        t_load = time.time() - t_load_start
        logger.info("[Recovery] Model recovered and loaded to GPU!")

        # ---- Phase 4: Restore opt state ----
        if hasattr(model, 'opt'):
            model.opt.projected_grad = self._pending_grad
            logger.info(f"[Recovery] Restored projected_grad={self._pending_grad:.6e}")

            # ZO2: reconstruct last_rstate (same as _init_for_resume)
            if self._pending_seed and hasattr(model.opt, 'rstate_queue'):
                torch.cuda.manual_seed(self._pending_seed)
                model.opt.last_rstate = torch.cuda.get_rng_state()
                from collections import deque
                model.opt.rstate_queue = deque([model.opt.last_rstate.clone()])
                logger.info(f"[Recovery] Reconstructed last_rstate from seed={self._pending_seed}")

        # ---- Summary ----
        t_total = time.time() - t_total_start
        logger.info(f"[Recovery] === Instant Recovery Summary ===")
        logger.info(f"[Recovery]   Shadow clone:  {t_clone:.3f}s")
        logger.info(f"[Recovery]   GPU replay:    {t_replay:.3f}s ({num_replay} updates)")
        logger.info(f"[Recovery]   Model load:    {t_load:.3f}s")
        logger.info(f"[Recovery]   Total:         {t_total:.3f}s")
        logger.info(f"[Recovery]   Replayed steps {shadow_step + 1}-{self.current_step}")
        logger.info(f"[Recovery]   Model at step {self.current_step}, "
                     f"next ZO forward = step {self.current_step + 1}")

        self.timing_stats['recoveries'].append({
            'type': 'instant_replay',
            'shadow_step': shadow_step,
            'gpu_step': self.current_step,
            'replay_updates': num_replay,
            'replay_time': t_replay,
            'load_time': t_load,
            'total_time': t_total,
        })

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

    def recover_to_gpu(self, model) -> int:
        """
        Complete recovery process: recover from shadow and load to GPU
        Returns: recovered step count, -1 if failed
        """
        logger.info("[Recovery] Starting recovery process...")

        recovered = self.recover_from_shadow()
        if recovered is None:
            return -1

        model.load_state_dict(recovered)
        model.to('cuda')
        torch.cuda.synchronize()

        recovered_step = self.shadow_step_val.value if self.shadow_step_val else self.shadow_step

        logger.info(f"[Recovery] Successfully recovered to step {recovered_step}")
        return recovered_step

    def get_recovery_status(self) -> dict:
        """Get current recovery status"""
        if self.enable_shadow:
            shadow_step = self.shadow_step_val.value if self.shadow_step_val else self.shadow_step
            shadow_available = self.shadow_shared is not None or self.shadow_model is not None
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
        GPU → CPU shared memory DMA, then notify shadow process to refresh."""
        state_dict = model.state_dict()

        if self.enable_shadow and self.shadow_shared is not None:
            # GPU → POSIX shared memory DMA (in-place copy to existing shm tensors)
            for key, value in state_dict.items():
                self.base_checkpoint_state[key].copy_(value)
            # Notify shadow process: base updated, please refresh
            if self.update_queue is not None:
                self.update_queue.put_nowait({
                    'cmd': 'refresh',
                    'new_step': len(self.update_history),
                })
        else:
            # Non-shadow path: allocate new tensors (original behavior)
            self.base_checkpoint_state = OrderedDict()
            for key, value in state_dict.items():
                self.base_checkpoint_state[key] = value.detach().cpu().clone()

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
            shadow_step = self.shadow_step_val.value if self.shadow_step_val else (self.shadow_step if self.shadow_model else -1)
            logger.info(f"[BatchDiff] Memory: cache={cache_mb:.1f}MB, "
                       f"system={used_gb:.1f}/{total_gb:.1f}GB, "
                       f"shadow_step={shadow_step}")
        else:
            logger.info(f"[BatchDiff] Memory: cache={cache_mb:.1f}MB, "
                       f"system={used_gb:.1f}/{total_gb:.1f}GB (on-demand mode)")

    def on_train_end(self, args, state, control, **kwargs):
        """Called at training end"""
        logger.info("[BatchDiff] Training ended, cleaning up...")

        # Shutdown async anchor checkpointer (wait for last persist to finish)
        async_anchor = getattr(self, '_async_anchor', None)
        if async_anchor is not None:
            logger.info("[AsyncAnchor] Waiting for last anchor persist to complete...")
            async_anchor.shutdown()
            # Final base update
            completed_step = async_anchor.get_latest_completed_anchor_step()
            if completed_step > self.base_checkpoint_step:
                self.base_checkpoint_path = async_anchor.get_latest_completed_anchor_path()
                self.base_checkpoint_step = completed_step
                with self.update_lock:
                    # Capture base_pending_seed before trimming
                    for u in reversed(self.update_history):
                        if u['step'] <= completed_step:
                            self._base_pending_seed = u['seed']
                            break
                    self.update_history = [
                        u for u in self.update_history
                        if u['step'] > completed_step
                    ]
            logger.info(f"[AsyncAnchor] Final stats: {async_anchor.stats}")

        if self.enable_shadow:
            # 1. Immediate stop signal (shadow checks this every iteration, ~50ns)
            if self.shadow_stop_event is not None:
                self.shadow_stop_event.set()
            # 2. Unblock any recovery_done.wait() deadlock
            if self.recovery_done is not None:
                self.recovery_done.set()
            # 3. Queue stop command as fallback
            if self.update_queue is not None:
                try:
                    self.update_queue.put_nowait({'cmd': 'stop'})
                except Exception:
                    pass
            # 4. Wait for shadow to exit (should be near-instant)
            if self.shadow_process is not None and self.shadow_process.is_alive():
                self.shadow_process.join(timeout=5.0)
                if self.shadow_process.is_alive():
                    self.shadow_process.terminate()
                    logger.warning("[BatchDiff] Shadow process force-terminated")
            # Legacy thread cleanup
            self.shadow_running = False
            if self.shadow_thread and self.shadow_thread.is_alive():
                self.shadow_thread.join(timeout=5.0)

        status = self.get_recovery_status()
        logger.info(f"[BatchDiff] Final status: {status}")

        if self.base_checkpoint_state:
            del self.base_checkpoint_state
            self.base_checkpoint_state = None
        if self.shadow_shared:
            del self.shadow_shared
            self.shadow_shared = None
        if self.shadow_model:
            del self.shadow_model
            self.shadow_model = None

        logger.info(f"[BatchDiff] Done. {self.save_count} checkpoints saved")


# Backward compatibility aliases
ZOReplayCheckpointCallback = BatchDiffCheckpointCallback
IncrementalCheckpointCallback = BatchDiffCheckpointCallback


# ============================================================================
# Shadow Process (multiprocessing, zero contention with training)
# ============================================================================

def _shadow_process_main(update_queue, base_shared, shadow_shared, shadow_step_val,
                          recovery_req, recovery_ready, recovery_done,
                          stop_event,
                          param_names, rng_device, simulate_perturbation,
                          default_zo_eps, adam_config, use_pipeline, P):
    """Shadow process entry point. Runs in a separate process with independent GIL.

    All heavy computation (z generation, tensor updates) happens here without
    any lock/GIL contention with the training process.

    Args:
        update_queue: mp.Queue — receives update dicts and commands from training
        base_shared: OrderedDict of shared-memory tensors (training writes new anchors)
        shadow_shared: OrderedDict of shared-memory tensors (this process writes updates)
        shadow_step_val: mp.Value('i', lock=False) — shared atomic step counter
        recovery_req/ready/done: mp.Event — pause/resume protocol for safe recovery
        stop_event: mp.Event — immediate stop signal from on_train_end
        param_names: list of trainable parameter names
        rng_device: 'native', 'cpu', or 'zo_rng'
        simulate_perturbation: whether to simulate [+1,-2,+1] perturbation
        default_zo_eps: default perturbation epsilon
        adam_config: dict with 'betas' and 'adam_eps' (or None for SGD)
        use_pipeline: whether to use pipelined P-producer pipeline
        P: number of producer threads (if pipeline)
    """
    # Spawn process starts with a fresh Python — set up logging (no inherited handlers)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    _logger = logging.getLogger(__name__ + '.shadow_process')

    # Kill inter-op thread pool — shadow never uses parallel op dispatch.
    # Must be called before any torch computation in this process.
    torch.set_num_interop_threads(1)

    # Thread allocation: use SHADOW_RESERVE_THREADS + SHADOW_CONSUMER_THREADS env vars.
    # Parent already logged the config; here we just apply it.
    try:
        n_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cores = os.cpu_count() or 64

    n_reserve = int(os.environ.get('SHADOW_RESERVE_THREADS', '1'))
    if use_pipeline and rng_device == "zo_rng":
        n_cons = int(os.environ.get('SHADOW_CONSUMER_THREADS', str(n_cores // 2)))
        c_prod = max(1, n_cores - n_reserve - n_cons)
        aten_threads = max(1, n_cons)
        import zo_rng as _zo_rng
        _zo_rng.set_num_threads(c_prod)
        torch.set_num_threads(aten_threads)
        _logger.info(f"[Shadow] threads: zo_rng={c_prod} + ATen={aten_threads} "
                     f"= {c_prod+aten_threads} (n_cores={n_cores}, reserve={n_reserve})")
    elif use_pipeline:
        threads_per_op = max(1, n_cores // (P + 1))
        torch.set_num_threads(threads_per_op)
    else:
        # Serial mode: z_gen and apply alternate (never simultaneous), so both
        # pools can safely use ALL available cores minus reserve.
        serial_threads = max(1, n_cores - n_reserve)
        torch.set_num_threads(serial_threads)
        if rng_device == "zo_rng":
            import zo_rng as _zo_rng
            _zo_rng.set_num_threads(serial_threads)
        _logger.info(f"[Shadow] serial zo_rng={serial_threads} ATen={serial_threads} "
                     f"(alternating, n_cores={n_cores}, reserve={n_reserve})")

    # ---- Probe A: comprehensive boot diagnostic ----
    try:
        _os_threads = len(os.listdir(f'/proc/{os.getpid()}/task'))
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
    _model_bytes = sum(shadow_shared[nm].numel() * shadow_shared[nm].element_size()
                       for nm in param_names)
    _interop_thr = torch.get_num_interop_threads()
    _logger.info(
        f"[Shadow Boot] pid={os.getpid()}\n"
        f"  affinity={{{','.join(str(c) for c in _affinity[:5])},...}} ({len(_affinity)} CPUs)\n"
        f"  aten={torch.get_num_threads()}  zo_rng={_zo_thr}  interop={_interop_thr}  "
        f"OS_threads={_os_threads}\n"
        f"  model_bytes={_model_bytes/1e9:.2f}GB\n"
        f"  pipeline={use_pipeline}  P={P}  rng={rng_device}  "
        f"simulate_perturbation={simulate_perturbation}")
    _logger.info(
        f"[Thread Env — Shadow Process]\n"
        f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}"
        f" → torch.get_num_threads()={torch.get_num_threads()}\n"
        f"  OMP_WAIT_POLICY={os.environ.get('OMP_WAIT_POLICY', 'unset')}\n"
        f"  MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', 'unset')}\n"
        f"  KMP_BLOCKTIME={os.environ.get('KMP_BLOCKTIME', 'unset')}\n"
        f"  GOMP_SPINCOUNT={os.environ.get('GOMP_SPINCOUNT', 'unset')}\n"
        f"  ZO_RNG_NUM_THREADS={os.environ.get('ZO_RNG_NUM_THREADS', 'unset')}"
        f" → zo_rng actual={_zo_thr}\n"
        f"  producer Python threads={P}")
    _thread_snapshot("Shadow BOOT", _logger, detail=True)

    try:
        # Initialize Adam state locally (not shared — only this process uses it)
        adam_state = None
        if adam_config is not None:
            adam_state = {
                'm': {}, 'v': {}, 't': 0,
                'betas': adam_config['betas'],
                'adam_eps': adam_config['adam_eps'],
            }
            _logger.info(f"[Shadow Process] Adam state initialized: betas={adam_config['betas']}")

        if use_pipeline:
            _shadow_process_pipelined(
                update_queue, base_shared, shadow_shared, shadow_step_val,
                recovery_req, recovery_ready, recovery_done, stop_event,
                param_names, rng_device, simulate_perturbation,
                default_zo_eps, adam_state, P, _logger)
        else:
            _shadow_process_serial(
                update_queue, base_shared, shadow_shared, shadow_step_val,
                recovery_req, recovery_ready, recovery_done, stop_event,
                param_names, rng_device, simulate_perturbation,
                default_zo_eps, adam_state, _logger)
    except Exception:
        import traceback
        _logger.error(f"[Shadow Process] CRASHED:\n{traceback.format_exc()}")


def _shadow_process_serial(update_queue, base_shared, shadow_shared, shadow_step_val,
                            recovery_req, recovery_ready, recovery_done, stop_event,
                            param_names, rng_device, simulate_perturbation,
                            default_zo_eps, adam_state, _logger):
    """Serial shadow: get update from queue, generate z + apply, repeat."""
    _logger.info(f"[Shadow Process] Running in serial mode (params={len(param_names)}, rng={rng_device})")

    # Clone shared-memory tensors → local heap memory for fast computation.
    # share_memory_() uses /dev/shm (tmpfs, 4KB pages, no THP) → massive TLB misses.
    # Local heap gets Transparent Huge Pages (2MB) → ~2x faster element-wise ops.
    # After each step, copy local → shared (for training's recovery reads).
    _t0_clone = time.time()
    shadow_local = OrderedDict()
    for name in param_names:
        shadow_local[name] = shadow_shared[name].clone()
    _clone_ms = (time.time() - _t0_clone) * 1000
    _logger.info(f"[Shadow] Cloned shared→local heap: {_clone_ms:.0f}ms "
                 f"({sum(t.numel()*t.element_size() for t in shadow_local.values())/1e9:.2f}GB)")

    while True:
        if stop_event.is_set():
            break

        # Check recovery request (pause for safe cloning by training process)
        if recovery_req.is_set():
            # Flush local → shared before signaling ready
            for key in param_names:
                shadow_shared[key].copy_(shadow_local[key])
            recovery_ready.set()
            while not recovery_done.is_set():
                if stop_event.is_set():
                    break
                recovery_done.wait(timeout=0.1)
            recovery_done.clear()
            recovery_req.clear()
            recovery_ready.clear()
            if stop_event.is_set():
                break
            continue

        try:
            cmd = update_queue.get(timeout=0.05)
        except queue_module.Empty:
            continue

        if isinstance(cmd, dict) and cmd.get('cmd') == 'stop':
            # Flush local → shared before exit
            for key in param_names:
                shadow_shared[key].copy_(shadow_local[key])
            break
        elif isinstance(cmd, dict) and cmd.get('cmd') == 'refresh':
            # Base was updated via DMA — copy base_shared → local (and shared)
            for key in param_names:
                if key in base_shared:
                    shadow_local[key].copy_(base_shared[key])
                    shadow_shared[key].copy_(base_shared[key])
            shadow_step_val.value = cmd['new_step']
            if adam_state is not None:
                adam_state['m'].clear()
                adam_state['v'].clear()
                adam_state['t'] = 0
            _logger.info(f"[Shadow Process] Refreshed from base at step {cmd['new_step']}")
            continue

        # Normal update — pre-generate z once, reuse for perturbation + gradient.
        update = cmd
        seed = update['seed']

        _t_start = time.time()
        z_dict = _generate_z_for_one_step(seed, param_names, shadow_local, rng_device)
        _t_zgen = time.time() - _t_start

        _t0 = time.time()
        _apply_single_update_with_pregenerated_z(
            shadow_local, update, param_names, z_dict,
            default_zo_eps=default_zo_eps,
            simulate_perturbation=simulate_perturbation,
            adam_state=adam_state)
        _t_apply = time.time() - _t0

        # Copy local → shared (so training can read for recovery)
        _t0_sync = time.time()
        for key in param_names:
            shadow_shared[key].copy_(shadow_local[key])
        _t_sync = time.time() - _t0_sync
        _t_total = time.time() - _t_start

        del z_dict

        shadow_step_val.value += 1
        step = shadow_step_val.value

        # Probe C: per-step timing (first 20, then every 10)
        if step <= 20 or step % 10 == 0:
            try:
                _os_thr = len(os.listdir(f'/proc/{os.getpid()}/task'))
            except Exception:
                _os_thr = -1
            _logger.info(
                f"[Shadow Serial] step={step} "
                f"zgen={_t_zgen*1000:.0f}ms "
                f"apply={_t_apply*1000:.0f}ms "
                f"sync={_t_sync*1000:.0f}ms "
                f"total={_t_total*1000:.0f}ms "
                f"os_thr={_os_thr}")

    _logger.info("[Shadow Process] Stopped (serial)")


def _shadow_process_pipelined(update_queue, base_shared, shadow_shared, shadow_step_val,
                               recovery_req, recovery_ready, recovery_done, stop_event,
                               param_names, rng_device, simulate_perturbation,
                               default_zo_eps, adam_state, P, _logger):
    """Pipelined shadow: P producer threads pre-generate z, consumer applies updates.

    All producers share a single Queue(maxsize=1) — at most one result waits
    at any time (consumer dequeue is instant; apply happens after dequeue).
    Runs entirely within this process — threads share this process's GIL
    (not the training process's GIL), so zero training contention.
    """
    shadow_bytes = sum(shadow_shared[nm].numel() * shadow_shared[nm].element_size()
                       for nm in param_names)
    _logger.info(f"[Shadow Pipeline] P={P} producers, "
                 f"shadow_copy={shadow_bytes/1e9:.2f}GB")

    # ---- DEBUG: verify tied weights after spawn/pickle ----
    _seen_ids = {}
    _tied_in_shadow = []
    for nm in shadow_shared:
        _tid = id(shadow_shared[nm])
        if _tid in _seen_ids:
            _tied_in_shadow.append(f"{nm} -> {_seen_ids[_tid]}")
        else:
            _seen_ids[_tid] = nm
    if _tied_in_shadow:
        _logger.info(f"[Shadow Pipeline] Tied weights in shadow_shared after spawn: {_tied_in_shadow}")
    else:
        _logger.warning("[Shadow Pipeline] WARNING: No tied weights in shadow_shared after spawn! "
                        "Tied params will be updated INDEPENDENTLY → divergence from training!")

    # Shared result queue: all producers put here, consumer gets
    result_queue = queue_module.Queue(maxsize=1)

    producer_stop = threading.Event()
    producer_error = [None]

    # Internal queue: main loop puts updates, producers consume
    internal_updates = {}  # step_idx → update dict
    internal_lock = threading.Lock()
    update_available_event = threading.Event()

    # Step assignment
    next_step_to_assign = [shadow_step_val.value]
    next_enqueue_step = [shadow_step_val.value]  # monotonic counter for incoming updates
    assign_lock = threading.Lock()

    # Shared timing for diagnostics (producer writes, consumer reads)
    producer_timing = {'t0': 0.0, 't1': 0.0, 'duration_ms': 0.0}

    def producer():
        """Producer: get update, generate z, put result into shared queue."""
        try:
            while not producer_stop.is_set():
                with assign_lock:
                    step_idx = next_step_to_assign[0]
                    next_step_to_assign[0] += 1

                # Wait for update data
                _t0_pwait = time.monotonic()
                update = None
                while not producer_stop.is_set():
                    with internal_lock:
                        if step_idx in internal_updates:
                            update = internal_updates.pop(step_idx)
                            break
                    update_available_event.wait(timeout=0.05)
                    update_available_event.clear()
                _pwait_ms = (time.monotonic() - _t0_pwait) * 1000

                if producer_stop.is_set():
                    break

                # Pre-generate z (releases GIL via zo_rng/PyTorch C++)
                t0_zgen = time.monotonic()
                z = _generate_z_for_one_step(update['seed'], param_names,
                                              shadow_shared, rng_device)
                t1_zgen = time.monotonic()
                _pzgen_ms = (t1_zgen - t0_zgen) * 1000
                producer_timing['t0'] = t0_zgen
                producer_timing['t1'] = t1_zgen
                producer_timing['duration_ms'] = _pzgen_ms
                producer_timing['wait_ms'] = _pwait_ms

                # Put blocks if queue full (maxsize=1) — naturally throttles
                while not producer_stop.is_set():
                    try:
                        result_queue.put((step_idx, z, update), timeout=0.05)
                        break
                    except queue_module.Full:
                        continue

        except Exception as e:
            _logger.error(f"[Shadow Pipeline] Producer CRASHED: {e}")
            producer_error[0] = e

    # Launch producer threads
    threads = []
    for i in range(P):
        t = threading.Thread(target=producer, daemon=True)
        t.start()
        threads.append(t)

    # Consumer loop (this thread)
    consumer_step = shadow_step_val.value
    _initial_consumer_step = consumer_step
    pending_results = {}

    while True:
        # Immediate exit check (~50ns, no lock)
        if stop_event.is_set():
            break

        # Check recovery
        if recovery_req.is_set():
            recovery_ready.set()
            while not recovery_done.is_set():
                if stop_event.is_set():
                    break
                recovery_done.wait(timeout=0.1)
            recovery_done.clear()
            recovery_req.clear()
            recovery_ready.clear()
            if stop_event.is_set():
                break
            continue

        # Read from external queue (non-blocking)
        try:
            while True:
                cmd = update_queue.get_nowait()
                if isinstance(cmd, dict) and cmd.get('cmd') == 'stop':
                    producer_stop.set()
                    for t in threads:
                        t.join(timeout=2.0)
                    _logger.info("[Shadow Pipeline] Stopped")
                    return
                elif isinstance(cmd, dict) and cmd.get('cmd') == 'refresh':
                    # --- Stop producers cleanly ---
                    producer_stop.set()
                    update_available_event.set()
                    for t in threads:
                        t.join(timeout=2.0)

                    # --- Reset shadow from new base ---
                    for key in param_names:
                        if key in shadow_shared and key in base_shared:
                            shadow_shared[key].copy_(base_shared[key])
                    new_step = cmd['new_step']
                    shadow_step_val.value = new_step
                    consumer_step = new_step

                    # --- Reset all pipeline state ---
                    next_step_to_assign[0] = new_step
                    next_enqueue_step[0] = new_step
                    internal_updates.clear()
                    pending_results.clear()
                    # Drain result queue
                    while not result_queue.empty():
                        try:
                            result_queue.get_nowait()
                        except queue_module.Empty:
                            break
                    update_available_event.clear()
                    if adam_state is not None:
                        adam_state['m'].clear()
                        adam_state['v'].clear()
                        adam_state['t'] = 0

                    # --- Restart producers ---
                    producer_stop.clear()
                    producer_error[0] = None
                    threads.clear()
                    for i in range(P):
                        t = threading.Thread(target=producer, daemon=True)
                        t.start()
                        threads.append(t)
                    _logger.info(f"[Shadow Pipeline] Refreshed at step {new_step}")
                else:
                    # Normal update — store for producers with monotonic index
                    with internal_lock:
                        internal_updates[next_enqueue_step[0]] = cmd
                        next_enqueue_step[0] += 1
                    update_available_event.set()
        except queue_module.Empty:
            pass

        if producer_error[0] is not None:
            _logger.error(f"[Shadow Pipeline] Producer error: {producer_error[0]}")
            break

        # Get next result from shared queue
        _t0_q_wait = time.monotonic()
        try:
            step_idx, z_dict, update = result_queue.get(timeout=0.05)
        except queue_module.Empty:
            continue
        _q_wait_ms = (time.monotonic() - _t0_q_wait) * 1000

        if producer_error[0] is not None:
            break

        pending_results[step_idx] = (z_dict, update)

        # Process consecutive ready steps
        while consumer_step in pending_results:
            z_dict, update = pending_results.pop(consumer_step)

            t0_apply = time.monotonic()
            _apply_single_update_with_pregenerated_z(
                shadow_shared, update, param_names, z_dict,
                default_zo_eps=default_zo_eps,
                simulate_perturbation=simulate_perturbation,
                zo2_mode=False,
                adam_state=adam_state,
                _diag_first_call=(consumer_step == _initial_consumer_step),
                _diag_logger=_logger,
            )
            t1_apply = time.monotonic()
            apply_ms = (t1_apply - t0_apply) * 1000

            # Check overlap with most recent producer z-gen
            p_t0, p_t1 = producer_timing['t0'], producer_timing['t1']
            overlap = (p_t0 < t1_apply and p_t1 > t0_apply)

            del z_dict
            shadow_step_val.value += 1
            consumer_step += 1

            if consumer_step <= 50 or consumer_step % 10 == 0:
                try:
                    _os_thr = len(os.listdir(f'/proc/{os.getpid()}/task'))
                except Exception:
                    _os_thr = -1
                _logger.info(
                    f"[Diag] step={consumer_step} "
                    f"apply={apply_ms:.0f}ms "
                    f"zgen={producer_timing['duration_ms']:.0f}ms "
                    f"overlap={'YES' if overlap else 'no'} "
                    f"threads={torch.get_num_threads()} "
                    f"p_wait={producer_timing.get('wait_ms',0):.0f}ms "
                    f"q_wait={_q_wait_ms:.0f}ms "
                    f"os_thr={_os_thr}"
                )

    # Cleanup
    producer_stop.set()
    for t in threads:
        t.join(timeout=2.0)
    _logger.info("[Shadow Pipeline] Stopped (pipelined)")


def _generate_z_for_replay(param, rng_device="native", zo_gen=None):
    """Generate z noise for replay, respecting rng_device setting.

    Args:
        param: The parameter tensor (determines size, dtype, target device)
        rng_device: "native" (use param's device), "cpu" (always CPU, cross-GPU portable),
                    or "zo_rng" (cross-device deterministic via zo_rng library)
        zo_gen: zo_rng.Generator instance (required when rng_device="zo_rng")
    """
    if rng_device == "zo_rng":
        return zo_gen.randn(param.shape, dtype=param.dtype, device=param.device)
    if rng_device == "cpu" and param.device.type != "cpu":
        z = torch.normal(mean=0, std=1, size=param.size(), dtype=torch.float32, device='cpu')
        return z.to(dtype=param.dtype, device=param.device)
    else:
        return torch.normal(mean=0, std=1, size=param.size(), dtype=param.dtype, device=param.device)


def _is_wd_param(name):
    """Return True if this parameter receives weight decay (not bias/layernorm)."""
    return ('bias' not in name and 'layer_norm' not in name
            and 'layernorm' not in name and 'ln' not in name)


def _generate_z_for_one_step(seed, param_names, state, rng_device, replay_dtype=None):
    """Generate z for all params for a single step (thread-safe).

    Uses a per-call torch.Generator to avoid global RNG state contamination.

    Args:
        seed: RNG seed for this step
        param_names: ordered list of param names (must match training order)
        state: state dict (read-only, used for shape/dtype/device info)
        rng_device: "native", "cpu", or "zo_rng"
        replay_dtype: if set, generate z in this dtype (e.g. fp32 for replay_in_fp32)

    Returns:
        dict of {param_name: z_tensor}
    """
    z_dict = {}

    if rng_device == "zo_rng":
        import zo_rng
        zo_gen = zo_rng.Generator(seed)
        for name in param_names:
            param = state[name]
            dtype = replay_dtype if replay_dtype is not None else param.dtype
            z_dict[name] = zo_gen.randn(param.shape, dtype=dtype, device=param.device)
    else:
        # Determine generator device
        sample_param = state[param_names[0]]
        if rng_device == "cpu" or sample_param.device.type == "cpu":
            gen = torch.Generator(device='cpu')
        else:
            gen = torch.Generator(device=sample_param.device)
        gen.manual_seed(seed)

        for name in param_names:
            param = state[name]
            dtype = replay_dtype if replay_dtype is not None else param.dtype

            if rng_device == "cpu" and param.device.type != "cpu":
                z = torch.normal(mean=0, std=1, size=param.size(),
                                 dtype=torch.float32, device='cpu', generator=gen)
                z_dict[name] = z.to(dtype=dtype, device=param.device)
            else:
                z_dict[name] = torch.normal(mean=0, std=1, size=param.size(),
                                            dtype=dtype, device=param.device, generator=gen)

    return z_dict


def _pipelined_replay_cpu(state, updates, param_names, rng_device,
                          num_producers, default_zo_eps, simulate_perturbation,
                          zo2_mode, seeds_info, replay_dtype):
    """Pipelined replay: P CPU producer threads + ring buffer + main-thread consumer.

    Each producer i owns slot i in a ring buffer of size P, generating z for
    steps {i, i+P, i+2P, ...}. The consumer (main thread) reads slots in order
    0, 1, ..., P-1, 0, 1, ... applying updates sequentially.

    Args:
        state: state dict to modify in-place
        updates: list of update dicts
        param_names: ordered param names
        rng_device: "native", "cpu", or "zo_rng"
        num_producers: P = number of producer threads = ring buffer size
        default_zo_eps: fallback zo_eps
        simulate_perturbation: whether to simulate [+1, -2, +1] perturbation
        zo2_mode: ZO2 replay order
        seeds_info: list of (grad_seed, perturb_seed) per step
        replay_dtype: override dtype for z generation (e.g. fp32)
    """
    P = num_producers
    n = len(updates)

    # Ring buffer: P slots, each holds (z_dict, z_perturb_dict)
    buffer = [None] * P
    ready = [threading.Event() for _ in range(P)]
    free = [threading.Event() for _ in range(P)]
    for e in free:
        e.set()  # all slots start free

    error_holder = [None]

    def producer(slot_id):
        try:
            step = slot_id
            while step < n:
                free[slot_id].wait()
                free[slot_id].clear()
                grad_seed, perturb_seed = seeds_info[step]
                z = _generate_z_for_one_step(grad_seed, param_names, state,
                                             rng_device, replay_dtype)
                z_perturb = None
                if zo2_mode and perturb_seed != grad_seed:
                    z_perturb = _generate_z_for_one_step(perturb_seed, param_names,
                                                         state, rng_device, replay_dtype)
                buffer[slot_id] = (z, z_perturb)
                ready[slot_id].set()
                step += P
        except Exception as e:
            error_holder[0] = e
            ready[slot_id].set()  # unblock consumer

    # Launch P producer threads
    threads = []
    for i in range(min(P, n)):
        t = threading.Thread(target=producer, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    # Consumer (main thread): read slots in order
    for step in range(n):
        slot = step % P
        ready[slot].wait()
        ready[slot].clear()
        if error_holder[0] is not None:
            raise error_holder[0]

        z, z_perturb = buffer[slot]
        _apply_single_update_with_pregenerated_z(
            state, updates[step], param_names, z,
            z_perturb_dict=z_perturb,
            default_zo_eps=default_zo_eps,
            simulate_perturbation=simulate_perturbation,
            zo2_mode=zo2_mode,
        )
        buffer[slot] = None  # release z memory
        free[slot].set()

        # Logging
        if step < 3 or step == n - 1:
            logger.info(f"[PipelinedReplay] update {step}: step={updates[step].get('step','?')}, "
                        f"seed={updates[step]['seed']}, grad={updates[step]['grad']:.6e}, "
                        f"lr={updates[step]['lr']}, wd={updates[step].get('wd', 0.0)}")
        elif step == 3:
            logger.info(f"[PipelinedReplay] ... ({n - 4} more updates) ...")

    # Wait for all producers to finish
    for t in threads:
        t.join()
    if error_holder[0] is not None:
        raise error_holder[0]


def _pipelined_replay_gpu(state, updates, param_names, rng_device,
                          num_producers, default_zo_eps, simulate_perturbation,
                          zo2_mode, seeds_info, replay_dtype):
    """Pipelined replay: P CUDA streams + events, single Python thread.

    Each slot i uses a dedicated CUDA stream for z generation. The consumer
    runs on the default stream. CUDA events synchronize between producer
    streams and the default stream — no Python threads needed.

    Args: same as _pipelined_replay_cpu
    """
    P = num_producers
    n = len(updates)

    streams = [torch.cuda.Stream() for _ in range(P)]
    ready_events = [torch.cuda.Event() for _ in range(P)]
    free_events = [torch.cuda.Event() for _ in range(P)]
    buffer = [None] * P

    # Pre-fill: schedule first P z-generations on separate streams
    for i in range(min(P, n)):
        grad_seed, perturb_seed = seeds_info[i]
        with torch.cuda.stream(streams[i]):
            z = _generate_z_for_one_step(grad_seed, param_names, state,
                                         rng_device, replay_dtype)
            z_perturb = None
            if zo2_mode and perturb_seed != grad_seed:
                z_perturb = _generate_z_for_one_step(perturb_seed, param_names,
                                                     state, rng_device, replay_dtype)
            buffer[i] = (z, z_perturb)
        ready_events[i].record(streams[i])

    # Main loop: consumer on default stream
    default_stream = torch.cuda.current_stream()
    for step in range(n):
        slot = step % P

        # Wait for producer stream to finish this slot's z
        default_stream.wait_event(ready_events[slot])

        z, z_perturb = buffer[slot]
        _apply_single_update_with_pregenerated_z(
            state, updates[step], param_names, z,
            z_perturb_dict=z_perturb,
            default_zo_eps=default_zo_eps,
            simulate_perturbation=simulate_perturbation,
            zo2_mode=zo2_mode,
        )

        # Schedule next z generation for this slot
        next_step = step + P
        if next_step < n:
            free_events[slot].record(default_stream)
            grad_seed, perturb_seed = seeds_info[next_step]
            with torch.cuda.stream(streams[slot]):
                streams[slot].wait_event(free_events[slot])
                z = _generate_z_for_one_step(grad_seed, param_names, state,
                                             rng_device, replay_dtype)
                z_perturb = None
                if zo2_mode and perturb_seed != grad_seed:
                    z_perturb = _generate_z_for_one_step(perturb_seed, param_names,
                                                         state, rng_device, replay_dtype)
                buffer[slot] = (z, z_perturb)
            ready_events[slot].record(streams[slot])
        else:
            buffer[slot] = None  # release z memory

        # Logging
        if step < 3 or step == n - 1:
            logger.info(f"[PipelinedReplay] update {step}: step={updates[step].get('step','?')}, "
                        f"seed={updates[step]['seed']}, grad={updates[step]['grad']:.6e}, "
                        f"lr={updates[step]['lr']}, wd={updates[step].get('wd', 0.0)}")
        elif step == 3:
            logger.info(f"[PipelinedReplay] ... ({n - 4} more updates) ...")

    torch.cuda.synchronize()


def _apply_single_update_with_pregenerated_z(state, update, param_names, z_dict,
                                              z_perturb_dict=None,
                                              default_zo_eps=0.0,
                                              simulate_perturbation=True,
                                              zo2_mode=False,
                                              adam_state=None,
                                              _diag_first_call=False,
                                              _diag_logger=None):
    """Apply one ZO update using pre-generated z tensors.

    Logic matches _apply_single_update exactly, but uses pre-generated z
    instead of calling torch.manual_seed + torch.normal.

    Args:
        state: state dict to modify in-place
        update: update dict with seed, grad, lr, wd, zo_eps
        param_names: ordered param names
        z_dict: pre-generated z for gradient update (keyed by param name)
        z_perturb_dict: pre-generated z for perturbation (ZO2 mode, different seed).
                        If None and zo2_mode, falls back to z_dict for perturbation.
        default_zo_eps: fallback zo_eps
        simulate_perturbation: whether to simulate [+1, -2, +1] perturbation
        zo2_mode: ZO2 replay order
        adam_state: If not None, use Adam update rule instead of SGD. Dict with keys:
            m (dict), v (dict), t (int), betas (tuple), adam_eps (float).
            Modified in-place (m/v/t updated). None = SGD (original path, unchanged).
    """
    grad = update['grad']
    lr = update['lr']
    wd = update.get('wd', 0.0)
    zo_eps = update.get('zo_eps', default_zo_eps)

    # Determine which z to use for perturbation
    z_for_perturb = z_perturb_dict if z_perturb_dict is not None else z_dict

    # Fused element-wise ops: use alpha= to eliminate temporary tensors.
    # Bitwise identical for perturbation (sf ∈ {1,-2,1}: integer mul is exact)
    # and gradient (lr*grad computed in fp64 then cast to fp32 in both paths).
    _lr_grad = float(lr * grad)  # pre-compute scalar (fp64)

    if adam_state is not None:
        # ====== Adam path (regular ZO order only, no ZO2) ======
        # 1) Perturbation simulation [+1, -2, +1]
        if simulate_perturbation and zo_eps > 0:
            for scaling_factor in [1, -2, 1]:
                _alpha = float(scaling_factor * zo_eps)
                for name in param_names:
                    state[name].data.add_(z_dict[name], alpha=_alpha)

        # 2) Adam update (always runs — even for grad=0, m/v decay and t increments,
        #    matching training where zo_update is called unconditionally)
        beta1, beta2 = adam_state['betas']
        a_eps = adam_state['adam_eps']
        adam_state['t'] += 1
        t = adam_state['t']
        bc1, bc2 = 1 - beta1 ** t, 1 - beta2 ** t
        step_size = lr / bc1

        for name in param_names:
            param = state[name]
            z = z_dict[name]
            g = (grad * z).float()  # fp32 to prevent underflow

            m, v = adam_state['m'], adam_state['v']
            if name not in m:
                m[name] = torch.zeros_like(param, dtype=torch.float32)
                v[name] = torch.zeros_like(param, dtype=torch.float32)

            m[name].mul_(beta1).add_(g, alpha=1 - beta1)
            v[name].mul_(beta2).addcmul_(g, g, value=1 - beta2)

            denom = (v[name] / bc2).sqrt_().add_(a_eps)
            upd = m[name].div(denom).mul_(step_size)

            # AdamW weight decay (skip bias/layernorm)
            if all(x not in name for x in ['bias', 'layer_norm', 'layernorm', 'ln']):
                upd.add_(param.float(), alpha=lr * wd)
            param.sub_(upd.to(param.dtype))
        return

    if zo2_mode and grad != 0:
        # ZO2 order: gradient FIRST (using prev step's z), then perturbation (current z)
        _lr_wd = float(lr * wd)
        for name in param_names:
            param = state[name]
            z = z_dict[name]
            if wd == 0.0:
                # When wd=0, use single FMA for ALL params (matches training's zo_update)
                param.sub_(z, alpha=_lr_grad)
            elif _is_wd_param(name):
                # param -= lr*(grad*z + wd*param)  →  1 temp instead of 4
                tmp = z.mul(grad)
                tmp.add_(param, alpha=wd)
                param.sub_(tmp, alpha=lr)
            else:
                param.sub_(z, alpha=_lr_grad)

        if simulate_perturbation and zo_eps > 0:
            for scaling_factor in [1, -2, 1]:
                _alpha = float(scaling_factor * zo_eps)
                for name in param_names:
                    state[name].data.add_(z_for_perturb[name], alpha=_alpha)
    else:
        # Regular ZO order: perturbation first, then gradient
        if simulate_perturbation and zo_eps > 0:
            if _diag_first_call:
                _thread_snapshot("Shadow BEFORE perturbation add_", _diag_logger, detail=True)
            for scaling_factor in [1, -2, 1]:
                _alpha = float(scaling_factor * zo_eps)
                for name in param_names:
                    state[name].data.add_(z_dict[name], alpha=_alpha)
            if _diag_first_call:
                _thread_snapshot("Shadow AFTER perturbation add_", _diag_logger, detail=True)

        if grad != 0:
            if _diag_first_call:
                _thread_snapshot("Shadow BEFORE gradient sub_", _diag_logger)
            for name in param_names:
                param = state[name]
                z = z_dict[name]
                if wd == 0.0:
                    param.sub_(z, alpha=_lr_grad)
                elif _is_wd_param(name):
                    tmp = z.mul(grad)
                    tmp.add_(param, alpha=wd)
                    param.sub_(tmp, alpha=lr)
                else:
                    param.sub_(z, alpha=_lr_grad)
            if _diag_first_call:
                _thread_snapshot("Shadow AFTER gradient sub_", _diag_logger)


def _parallel_replay_updates_on_state(
    state: OrderedDict,
    updates: list,
    device: str = 'cpu',
    move_to_device: bool = True,
    trainable_param_names: list = None,
    default_zo_eps: float = 0.0,
    simulate_perturbation: bool = True,
    replay_in_fp32: bool = False,
    rng_device: str = "native",
    zo2_mode: bool = False,
    initial_prev_seed: int = None,
) -> OrderedDict:
    """Replay ZO updates with pipelined producer-consumer architecture.

    P producer threads/streams generate z concurrently while 1 consumer
    applies updates serially via a ring buffer of size P.
    Result is bitwise-exact with sequential replay.

    CPU mode (threads): when device='cpu' or rng_device in {'cpu', 'zo_rng'}.
    GPU mode (CUDA streams): when device='cuda' and rng_device='native'.

    Args: same as _replay_updates_on_state
    """
    if not updates:
        return state

    P = 1  # default: 1 producer (no overlap, same as sequential)
    env_workers = os.environ.get('PARALLEL_RECOVERY_WORKERS', None)
    if env_workers is not None:
        P = max(1, int(env_workers))

    logger.info(f"[PipelinedReplay] {len(updates)} updates, P={P}, "
                f"rng_device={rng_device}, zo2_mode={zo2_mode}, "
                f"simulate_perturbation={simulate_perturbation}")

    # ---- Device handling (same as sequential) ----
    actual_device = 'cpu'
    if move_to_device and device == 'cuda' and torch.cuda.is_available():
        for key in state:
            if state[key].device.type != 'cuda':
                state[key] = state[key].cuda()
        actual_device = 'cuda'
    elif len(state) > 0:
        # Use .type to normalize 'cuda:0' -> 'cuda'
        actual_device = next(iter(state.values())).device.type

    # ---- fp32 upcast (same as sequential) ----
    original_dtype = None
    replay_dtype = None
    if replay_in_fp32 and actual_device == 'cpu':
        sample = next(iter(state.values()))
        if sample.dtype in (torch.float16, torch.bfloat16):
            original_dtype = sample.dtype
            replay_dtype = torch.float32
            for key in state:
                state[key] = state[key].float()
            logger.info(f"[PipelinedReplay] Upcast {original_dtype} -> fp32 for CPU replay")

    # ---- CPU/CUDA RNG warning ----
    if actual_device == 'cpu' and torch.cuda.is_available() and rng_device != "zo_rng":
        logger.warning("[PipelinedReplay] WARNING: Replaying on CPU but CUDA is available. "
                       "Use device='cuda' or ZO_RNG_DEVICE=zo_rng for exact reconstruction.")

    param_names = trainable_param_names if trainable_param_names is not None else list(state.keys())

    # ---- Memory warning ----
    n = len(updates)
    model_bytes = sum(state[nm].numel() * state[nm].element_size() for nm in param_names)
    z_sets_per_step = 2 if zo2_mode else 1
    buffer_bytes = model_bytes * P * z_sets_per_step
    available_bytes = psutil.virtual_memory().available
    if buffer_bytes > available_bytes * 0.5:
        logger.warning(f"[PipelinedReplay] Ring buffer ~{buffer_bytes / 1e9:.1f} GB "
                       f"but only {available_bytes / 1e9:.1f} GB available. "
                       f"Consider reducing PARALLEL_RECOVERY_WORKERS.")

    # ---- Pre-compute seeds_info ----
    seeds_info = []
    for i, update in enumerate(updates):
        if zo2_mode:
            if i == 0:
                prev_seed = initial_prev_seed if initial_prev_seed is not None else update['seed']
            else:
                prev_seed = updates[i - 1]['seed']
            seeds_info.append((prev_seed, update['seed']))
        else:
            seeds_info.append((update['seed'], update['seed']))

    # ---- Dispatch to CPU or GPU pipeline ----
    t_start = time.time()
    _pip_proc = psutil.Process(os.getpid())
    _pip_cpu0, _pip_gpu0 = _log_memory("pipelined start", _pip_proc, actual_device)

    use_gpu_pipeline = (actual_device == 'cuda' and rng_device == 'native')
    if use_gpu_pipeline:
        logger.info(f"[PipelinedReplay] Using GPU mode (CUDA streams)")
        _pipelined_replay_gpu(
            state, updates, param_names, rng_device,
            num_producers=P, default_zo_eps=default_zo_eps,
            simulate_perturbation=simulate_perturbation,
            zo2_mode=zo2_mode, seeds_info=seeds_info,
            replay_dtype=replay_dtype,
        )
    else:
        logger.info(f"[PipelinedReplay] Using CPU mode (threads)")
        _pipelined_replay_cpu(
            state, updates, param_names, rng_device,
            num_producers=P, default_zo_eps=default_zo_eps,
            simulate_perturbation=simulate_perturbation,
            zo2_mode=zo2_mode, seeds_info=seeds_info,
            replay_dtype=replay_dtype,
        )

    _log_memory("pipelined done", _pip_proc, actual_device, _pip_cpu0, _pip_gpu0)

    # ---- fp32 downcast ----
    if original_dtype is not None:
        for key in state:
            state[key] = state[key].to(original_dtype)
        logger.info(f"[PipelinedReplay] Downcast fp32 -> {original_dtype}")

    t_elapsed = time.time() - t_start
    mode_str = "GPU/CUDA-streams" if use_gpu_pipeline else "CPU/threads"
    logger.info(f"[PipelinedReplay] Completed: {n} updates in {t_elapsed:.3f}s "
                f"(P={P}, mode={mode_str}, device={actual_device})")

    return state


# ============================================================================
# Memory logging helper
# ============================================================================

def _log_memory(tag, proc, device_type, baseline_cpu=None, baseline_gpu=None):
    """Log CPU RSS and GPU memory at a labeled checkpoint.

    Returns (cpu_rss, gpu_alloc_or_None) for use as baseline in later calls.
    """
    cpu_rss = proc.memory_info().rss
    parts = [f"CPU RSS={cpu_rss / 1e9:.2f} GB"]
    if baseline_cpu is not None:
        parts.append(f"(delta={(cpu_rss - baseline_cpu) / 1e9:+.2f} GB)")
    gpu_alloc = None
    if device_type == 'cuda' and torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated()
        gpu_peak = torch.cuda.max_memory_allocated()
        parts.append(f"GPU alloc={gpu_alloc / 1e9:.2f} GB")
        parts.append(f"GPU peak={gpu_peak / 1e9:.2f} GB")
        if baseline_gpu is not None:
            parts.append(f"(delta={(gpu_alloc - baseline_gpu) / 1e9:+.2f} GB)")
    logger.info(f"[Memory] {tag}: {', '.join(parts)}")
    return cpu_rss, gpu_alloc


# ============================================================================
# Closed-form parallel replay
# ============================================================================

def _closedform_cpu(state, param_names, terms, rng_device, num_workers,
                    accum_dtype, replay_dtype):
    """CPU backend for closed-form replay using threads.

    Workers generate z in parallel, scale in-place, then lock-accumulate
    into a single shared buffer.  No per-worker partial-sum buffers.
    Worker k processes terms[k], terms[k+W], terms[k+2W], ...

    Args:
        state: state dict (read-only for shape/dtype/device info)
        param_names: ordered param names
        terms: list of (step_idx, coeff_wd, coeff_nowd, grad_seed)
        rng_device: RNG mode
        num_workers: W
        accum_dtype: dtype for partial sum accumulation
        replay_dtype: dtype override for z generation (or None)

    Returns:
        dict mapping param_name -> total sum tensor
    """
    W = min(num_workers, len(terms))
    if W == 0:
        return {}

    # Single shared accumulation buffer (instead of W separate partials)
    total_sum = {}
    for name in param_names:
        param = state[name]
        total_sum[name] = torch.zeros(param.shape, dtype=accum_dtype,
                                      device=param.device)

    lock = threading.Lock()
    error_holder = [None]

    def worker_fn(worker_id):
        try:
            idx = worker_id
            while idx < len(terms):
                _step_idx, coeff_wd, coeff_nowd, grad_seed = terms[idx]
                z_dict = _generate_z_for_one_step(
                    grad_seed, param_names, state, rng_device, replay_dtype)
                # Scale z in-place (reuse z buffer as result buffer)
                for name in param_names:
                    z = z_dict[name]
                    c = coeff_wd if _is_wd_param(name) else coeff_nowd
                    if accum_dtype != z.dtype:
                        z = z.to(accum_dtype)
                        z_dict[name] = z
                    z.mul_(c)
                # Lock and accumulate into shared buffer
                with lock:
                    for name in param_names:
                        total_sum[name].add_(z_dict[name])
                del z_dict
                idx += W
        except Exception as e:
            error_holder[0] = e

    threads = []
    for i in range(W):
        t = threading.Thread(target=worker_fn, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if error_holder[0] is not None:
        raise error_holder[0]

    return total_sum


def _closedform_gpu(state, param_names, terms, rng_device, num_workers,
                    accum_dtype, replay_dtype):
    """GPU backend for closed-form replay using CUDA streams.

    Processes terms in batches of W: generate z in W parallel streams,
    synchronize, then accumulate into a single shared buffer on the
    default stream.  No per-worker partial-sum buffers.

    Args: same as _closedform_cpu
    Returns: dict mapping param_name -> total sum tensor
    """
    W = min(num_workers, len(terms))
    if W == 0:
        return {}

    streams = [torch.cuda.Stream() for _ in range(W)]

    # Single shared accumulation buffer
    total_sum = {}
    for name in param_names:
        param = state[name]
        total_sum[name] = torch.zeros(param.shape, dtype=accum_dtype,
                                      device=param.device)

    # Process terms in batches of W
    for batch_start in range(0, len(terms), W):
        batch_end = min(batch_start + W, len(terms))
        batch_size = batch_end - batch_start
        z_dicts = [None] * batch_size

        # Phase 1: generate z and scale in-place across W parallel streams
        for i in range(batch_size):
            _step_idx, coeff_wd, coeff_nowd, grad_seed = terms[batch_start + i]
            with torch.cuda.stream(streams[i]):
                z_dicts[i] = _generate_z_for_one_step(
                    grad_seed, param_names, state, rng_device, replay_dtype)
                for name in param_names:
                    z = z_dicts[i][name]
                    c = coeff_wd if _is_wd_param(name) else coeff_nowd
                    if accum_dtype != z.dtype:
                        z = z.to(accum_dtype)
                        z_dicts[i][name] = z
                    z.mul_(c)

        # Phase 2: sync, then accumulate on default stream
        torch.cuda.synchronize()
        for i in range(batch_size):
            for name in param_names:
                total_sum[name].add_(z_dicts[i][name])
        del z_dicts

    return total_sum


def _closedform_replay_on_state(
    state: OrderedDict,
    updates: list,
    device: str = 'cpu',
    move_to_device: bool = True,
    trainable_param_names: list = None,
    rng_device: str = "native",
    zo2_mode: bool = False,
    initial_prev_seed: int = None,
    num_workers: int = 1,
    precision: str = "mixed",
) -> OrderedDict:
    """Closed-form parallel replay of ZO-SGD updates.

    Unrolls the recurrence p_t = (1 - lr_t*wd_t)*p_{t-1} - lr_t*grad_t*z_t
    into a closed-form sum of independent terms:
        p_n = sp[0]*p_0 - Σ_{t=0}^{n-1} sp[t+1] * lr_t * grad_t * z_t
    where sp[i] = Π_{j=i}^{n-1} (1 - lr_j*wd_j) is the suffix product.

    Each term in the sum is independent, so the sum is split across W workers.
    No perturbation simulation — this implements the pure mathematical model.

    Args:
        state: state dict to modify (initial parameters p_0)
        updates: list of update dicts with keys: seed, grad, lr, wd
        device: 'cpu' or 'cuda'
        move_to_device: if True and device='cuda', move state to GPU first
        trainable_param_names: ordered param names (must match training order)
        rng_device: "native", "cpu", or "zo_rng"
        zo2_mode: if True, gradient uses prev step's seed
        initial_prev_seed: seed from base checkpoint for ZO2 mode
        num_workers: W, number of parallel workers (default 1 = serial closed-form)
        precision: "fp32" (all fp32), "fp16" (keep original dtype),
                   "mixed" (accumulate in fp32, keep params in original dtype)

    Returns:
        Modified state dict with recovered parameters
    """
    if not updates:
        return state

    n = len(updates)
    W = num_workers

    logger.info(f"[ClosedForm] {n} updates, W={W}, precision={precision}, "
                f"rng_device={rng_device}, zo2_mode={zo2_mode}")

    # ---- Device handling ----
    actual_device = 'cpu'
    if move_to_device and device == 'cuda' and torch.cuda.is_available():
        for key in state:
            if state[key].device.type != 'cuda':
                state[key] = state[key].cuda()
        actual_device = 'cuda'
    elif len(state) > 0:
        actual_device = next(iter(state.values())).device.type

    # ---- Precision setup ----
    sample = next(iter(state.values()))
    original_dtype = sample.dtype

    if precision == "fp32":
        accum_dtype = torch.float32
        target_dtype = torch.float32
        replay_dtype = torch.float32
        # Upcast state to fp32
        if original_dtype != torch.float32:
            for key in state:
                state[key] = state[key].float()
            logger.info(f"[ClosedForm] Upcast {original_dtype} -> fp32")
    elif precision == "fp16":
        accum_dtype = original_dtype
        target_dtype = original_dtype
        # On CPU, torch.normal(dtype=fp16) is ~22x slower than fp32.
        # Generate z in fp32 and cast during accumulation for speed.
        replay_dtype = torch.float32 if (actual_device == 'cpu' and original_dtype != torch.float32) else None
    elif precision == "mixed":
        accum_dtype = torch.float32
        target_dtype = original_dtype
        # On CPU, generate z in fp32 directly (avoids slow fp16 torch.normal + later cast)
        replay_dtype = torch.float32 if (actual_device == 'cpu' and original_dtype != torch.float32) else None
    else:
        raise ValueError(f"Unknown precision mode: {precision}")

    # ---- CPU/CUDA RNG warning ----
    if actual_device == 'cpu' and torch.cuda.is_available() and rng_device != "zo_rng":
        logger.warning("[ClosedForm] WARNING: Replaying on CPU but CUDA is available. "
                       "Use device='cuda' or ZO_RNG_DEVICE=zo_rng for exact reconstruction.")

    param_names = trainable_param_names if trainable_param_names is not None else list(state.keys())

    # ---- Memory warning ----
    total_numel = sum(state[nm].numel() for nm in param_names)
    accum_elem_size = torch.tensor([], dtype=accum_dtype).element_size()
    replay_es = torch.tensor([], dtype=replay_dtype if replay_dtype is not None else original_dtype).element_size()
    accum_bytes = total_numel * accum_elem_size          # 1 shared buffer
    z_bytes = total_numel * replay_es * W                # W concurrent z buffers
    total_buffer = accum_bytes + z_bytes
    available_bytes = psutil.virtual_memory().available
    if total_buffer > available_bytes * 0.5:
        logger.warning(f"[ClosedForm] Worker buffers ~{total_buffer / 1e9:.1f} GB "
                       f"but only {available_bytes / 1e9:.1f} GB available.")

    # ---- Precompute suffix product sp[i] = Π_{j=i}^{n-1} (1 - lr_j * wd_j) ----
    has_any_wd = any(u.get('wd', 0.0) != 0 for u in updates)
    sp = [1.0] * (n + 1)
    if has_any_wd:
        for i in range(n - 1, -1, -1):
            sp[i] = sp[i + 1] * (1.0 - updates[i]['lr'] * updates[i].get('wd', 0.0))

    sp_0 = sp[0]

    # ---- Precompute seeds_info (same as pipelined) ----
    seeds_info = []
    for i, update in enumerate(updates):
        if zo2_mode:
            if i == 0:
                prev_seed = initial_prev_seed if initial_prev_seed is not None else update['seed']
            else:
                prev_seed = updates[i - 1]['seed']
            seeds_info.append((prev_seed, update['seed']))
        else:
            seeds_info.append((update['seed'], update['seed']))

    # ---- Build term list, filter grad=0 ----
    terms = []
    for t in range(n):
        grad = updates[t]['grad']
        if grad == 0:
            continue
        lr = updates[t]['lr']
        coeff_wd = sp[t + 1] * lr * grad
        coeff_nowd = lr * grad
        terms.append((t, coeff_wd, coeff_nowd, seeds_info[t][0]))

    logger.info(f"[ClosedForm] {len(terms)} non-zero terms out of {n} updates"
                f" (sp[0]={sp_0:.10f})")

    # ---- Dispatch to backend ----
    t_start = time.time()
    _cf_proc = psutil.Process(os.getpid())
    _cf_cpu0, _cf_gpu0 = _log_memory("closedform start", _cf_proc, actual_device)

    use_gpu = (actual_device == 'cuda' and rng_device == 'native')
    if len(terms) == 0:
        total_sum = {}
    elif use_gpu:
        logger.info(f"[ClosedForm] Using GPU mode (CUDA streams)")
        total_sum = _closedform_gpu(
            state, param_names, terms, rng_device, W, accum_dtype, replay_dtype)
    else:
        logger.info(f"[ClosedForm] Using CPU mode (threads)")
        total_sum = _closedform_cpu(
            state, param_names, terms, rng_device, W, accum_dtype, replay_dtype)

    _log_memory("closedform after accumulation", _cf_proc, actual_device, _cf_cpu0, _cf_gpu0)

    # If no terms, fill zeros
    if not total_sum:
        total_sum = {name: torch.zeros(state[name].shape, dtype=accum_dtype,
                                       device=state[name].device)
                     for name in param_names}

    # ---- Finalize: p_n = sp[0] * p_0 - total_sum ----
    for name in param_names:
        p0 = state[name]
        ts = total_sum[name]
        if _is_wd_param(name) and has_any_wd:
            result = sp_0 * p0.to(accum_dtype) - ts
        else:
            result = p0.to(accum_dtype) - ts
        state[name] = result.to(target_dtype)

    t_elapsed = time.time() - t_start
    mode_str = "GPU/CUDA-streams" if use_gpu else "CPU/threads"
    logger.info(f"[ClosedForm] Completed: {n} updates in {t_elapsed:.3f}s "
                f"(W={W}, precision={precision}, mode={mode_str}, device={actual_device})")
    _log_memory("closedform done", _cf_proc, actual_device, _cf_cpu0, _cf_gpu0)

    return state


def validate_closedform_replay(
    state: OrderedDict,
    updates: list,
    trainable_param_names: list = None,
    rng_device: str = "native",
    zo2_mode: bool = False,
    initial_prev_seed: int = None,
    num_workers: int = 1,
    device: str = 'cpu',
) -> dict:
    """Validate closed-form replay against serial replay.

    Runs serial replay (simulate_perturbation=False) as ground truth, then
    runs closed-form replay in all 3 precision modes (fp32, mixed, fp16).
    Reports max absolute error and relative error per parameter per mode.

    Args:
        state: initial state dict (will be cloned, not modified)
        updates: list of update dicts
        trainable_param_names: ordered param names
        rng_device: RNG mode
        zo2_mode: ZO2 mode flag
        initial_prev_seed: for ZO2 mode
        num_workers: W for closed-form
        device: computation device

    Returns:
        dict with structure:
        {
            "fp32": {"param_name": {"max_abs": float, "rel": float}, ...},
            "mixed": {...},
            "fp16": {...},
        }
    """
    def _clone(s):
        return OrderedDict((k, v.clone()) for k, v in s.items())

    param_names = trainable_param_names if trainable_param_names is not None else list(state.keys())

    # Ground truth: serial replay without perturbation
    state_serial = _clone(state)
    _replay_updates_on_state(
        state_serial, updates, device=device, move_to_device=True,
        trainable_param_names=param_names,
        simulate_perturbation=False,
        rng_device=rng_device, zo2_mode=zo2_mode,
        initial_prev_seed=initial_prev_seed,
    )

    results = {}
    for prec in ["fp32", "mixed", "fp16"]:
        state_cf = _clone(state)
        _closedform_replay_on_state(
            state_cf, updates, device=device, move_to_device=True,
            trainable_param_names=param_names,
            rng_device=rng_device, zo2_mode=zo2_mode,
            initial_prev_seed=initial_prev_seed,
            num_workers=num_workers,
            precision=prec,
        )

        prec_results = {}
        for name in param_names:
            serial_p = state_serial[name].float()
            cf_p = state_cf[name].float()
            diff = (serial_p - cf_p).abs()
            max_abs = diff.max().item()
            denom = serial_p.abs().max().item()
            rel = max_abs / max(denom, 1e-10)
            prec_results[name] = {"max_abs": max_abs, "rel": rel}

        results[prec] = prec_results

    # Log summary
    logger.info(f"[ClosedForm Validation] {len(updates)} updates, W={num_workers}")
    for prec in ["fp32", "mixed", "fp16"]:
        max_abs_all = max(v["max_abs"] for v in results[prec].values())
        max_rel_all = max(v["rel"] for v in results[prec].values())
        logger.info(f"  {prec:6s}: max_abs={max_abs_all:.2e}, max_rel={max_rel_all:.2e}")

    return results




# ============================================================
# Producer-Consumer Thread Allocation Optimization
# ============================================================


def _benchmark_curves_worker(shared_tensors, param_names, rng_device, C,
                              n_warmup, n_measure, zo_eps, adam_state,
                              core_points, result_dict):
    """Measure t_gen(c) and t_update(n) for specified core counts in a spawned child.

    t_gen(c): time to generate z for all params using c zo_rng + c aten threads.
    t_update(n): time to apply one update (perturbation simulation + gradient + weight decay)
                 using n aten threads with pre-generated z.

    Args:
        core_points: list of core counts to benchmark (e.g. [1,20,40,...,126]).
    """
    import torch, time, json
    from collections import OrderedDict

    torch.set_num_interop_threads(1)

    try:
        n_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cores = os.cpu_count() or 64

    _zo_rng = None
    if rng_device == "zo_rng":
        import zo_rng as _zo_rng

    # Clone shared → local heap
    state = OrderedDict()
    for name in param_names:
        state[name] = shared_tensors[name].clone()

    def _median(xs):
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    C_max = min(C, n_cores)
    # Filter core_points to valid range [1, C_max)
    points = sorted(set(c for c in core_points if 1 <= c < C_max))
    print(f"[BenchCurves] pid={os.getpid()} cores={n_cores} C={C_max} "
          f"points={len(points)} range=[{points[0]}..{points[-1]}] "
          f"warmup={n_warmup} measure={n_measure}", flush=True)

    # --- Measure t_gen(c) ---
    t_gen_curve = {}
    for idx, c in enumerate(points):
        if _zo_rng is not None:
            _zo_rng.set_num_threads(c)
        else:
            torch.set_num_threads(c)

        times = []
        for i in range(n_warmup + n_measure):
            seed = 1000000 + i
            t0 = time.monotonic()
            z = _generate_z_for_one_step(seed, param_names, state, rng_device)
            t1 = time.monotonic()
            if i >= n_warmup:
                times.append(t1 - t0)
            del z

        t_gen_curve[c] = _median(times)
        print(f"[BenchCurves] t_gen(c={c}) = {t_gen_curve[c]*1000:.1f}ms  "
              f"[{idx+1}/{len(points)}]", flush=True)

    # --- Measure t_update(n) ---
    # Pre-generate z once using max threads for speed.
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
        for i in range(n_warmup + n_measure):
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

        t_update_curve[n] = _median(times)
        print(f"[BenchCurves] t_update(n={n}) = {t_update_curve[n]*1000:.1f}ms  "
              f"[{idx+1}/{len(points)}]", flush=True)

    del z_pregenerated

    # --- Find t_update plateau [n_low, n_high] ---
    t_update_min = min(t_update_curve.values())
    plateau_threshold = 1.10  # within 10% of min
    plateau_points = sorted(
        n for n in t_update_curve
        if t_update_curve[n] < t_update_min * plateau_threshold
    )
    n_low, n_high = plateau_points[0], plateau_points[-1]
    print(f"[BenchCurves] t_update plateau: n=[{n_low}, {n_high}], "
          f"t_min={t_update_min*1000:.1f}ms (±10%)", flush=True)

    # Serialize to result_dict
    result_dict['t_gen_json'] = json.dumps({str(k): v for k, v in t_gen_curve.items()})
    result_dict['t_update_json'] = json.dumps({str(k): v for k, v in t_update_curve.items()})
    result_dict['n_cores'] = n_cores
    result_dict['C_max'] = C_max
    result_dict['n_low'] = n_low
    result_dict['n_high'] = n_high


def _interp_curve(curve_dict, x):
    """Linear interpolation on a sparse {int: float} curve. Clamps at boundaries."""
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


def optimize_thread_allocation(t_gen_curve, t_update_curve, C, t_train,
                                P_max=8, n_sat_range=None):
    """P-first search for optimal (c, P) minimizing pipeline step time.

    Outer loop: P from 1 upward (fewer producers preferred).
    Inner loop: c (threads per producer), using interpolation on sparse curves.

    Returns Pareto table (best config per P) + recommended config
    (smallest P within 5% of global optimum).

    Args:
        t_gen_curve: dict {c: seconds}
        t_update_curve: dict {n: seconds}
        C: total available CPU threads
        t_train: GPU training step time in seconds
        P_max: maximum number of producers to consider (default 8)
        n_sat_range: (n_low, n_high) tuple constraining consumer threads
                     to the t_update plateau range. If None, search [1, C).
    """
    pareto = []
    all_configs = []

    for P in range(1, P_max + 1):
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
            t_step = max(t_gen_P, t_upd, t_train)

            components = {'t_gen/P': t_gen_P, 't_update': t_upd, 't_train': t_train}
            bottleneck = max(components, key=components.get)

            cfg = {
                'c': c, 'P': P, 'c_cons': c_cons,
                't_step': t_step, 'bottleneck': bottleneck,
                'B': 1, 'lag_frac': 0.0,
                't_gen_P': t_gen_P, 't_update_val': t_upd,
            }
            all_configs.append(cfg)

            if t_step < best_t:
                best_t = t_step
                best_cfg = cfg

        if best_cfg:
            pareto.append(best_cfg)

    if not pareto:
        raise ValueError("No valid (c, P) configuration found")

    # Recommend: smallest P within 5% of global best
    global_best = min(pareto, key=lambda x: x['t_step'])
    threshold = global_best['t_step'] * 1.05
    recommended = next(p for p in pareto if p['t_step'] <= threshold)

    return {
        'pareto': pareto,
        'recommended': recommended,
        'all_configs': all_configs,
        'C': C,
        't_train': t_train,
        # Backward compat keys (from recommended)
        'best_c': recommended['c'],
        'best_P': recommended['P'],
        'best_c_cons': recommended['c_cons'],
        'best_t_step': recommended['t_step'],
        'best_bottleneck': recommended['bottleneck'],
        'best_B': recommended['B'],
        'best_lag_frac': recommended['lag_frac'],
        'best_t_gen_P': recommended['t_gen_P'],
        'best_t_update_val': recommended['t_update_val'],
    }


def calibrate_producer_consumer(state, param_names, rng_device="zo_rng",
                                 C=None, t_train=None, dataloader_num_workers=0,
                                 n_warmup=5, n_measure=8,
                                 zo_eps=1e-3, adam_state=None,
                                 core_start=1, core_stop=None, core_step=1):
    """Benchmark t_gen/t_update curves and find optimal (c, P, c_cons).

    Spawns a child process with shadow-identical env to measure:
      t_gen(c): z generation time with c threads (zo_rng + aten)
      t_update(n): update apply time with n aten threads (perturbation + grad + wd)

    Then searches for optimal (c, P) minimizing:
      t_step = max(t_gen(c)/P, t_update(C-P*c), t_train)

    Args:
        state: state dict on CPU (will be cloned, not modified)
        param_names: ordered list of trainable param names
        rng_device: "zo_rng" (default), "native", or "cpu"
        C: total available CPU threads. If None: sched_getaffinity() - 1 - dataloader_num_workers
        t_train: GPU training step time in seconds (required)
        dataloader_num_workers: HF Trainer dataloader workers (default 0)
        n_warmup: warmup iterations per measurement point
        n_measure: measurement iterations per point (median taken)
        zo_eps: perturbation epsilon
        adam_state: if not None, measure Adam update path
        core_start: first core count to benchmark (default 1)
        core_stop: last core count to benchmark (default C-1, inclusive)
        core_step: step size between core counts (default 1)

    Returns:
        dict with keys: t_gen_curve, t_update_curve, C, t_train,
                        best_c, best_P, best_c_cons, best_t_step,
                        best_B, best_lag_frac, best_bottleneck,
                        all_configs, per_slot_bytes
    """
    if t_train is None:
        raise ValueError("t_train (GPU step time in seconds) is required")

    # Ensure state on CPU
    for k in state:
        if state[k].device.type != 'cpu':
            state[k] = state[k].cpu()

    try:
        n_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cores = os.cpu_count() or 64

    if C is None:
        C = n_cores - 1 - dataloader_num_workers
    C = max(3, C)  # need at least c=1, P=1, c_cons=1

    # Generate core_points from start/stop/step
    _stop = core_stop if core_stop is not None else C - 1
    core_points = list(range(core_start, _stop + 1, core_step))
    # Always include endpoint if not already there
    if _stop not in core_points and _stop >= core_start:
        core_points.append(_stop)
    logger.info(f"[CalibratePC] core_points: {len(core_points)} points, "
                f"range=[{core_points[0]}..{core_points[-1]}], step={core_step}")

    # Shared memory for child process
    shared_state = OrderedDict()
    for name in param_names:
        shared_state[name] = state[name].clone().share_memory_()

    # Shadow-identical environment
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
    manager = ctx.Manager()  # keep reference to prevent GC
    result_dict = manager.dict()

    # Estimate timeout: ~60s per point × 2 curves + 120s overhead
    n_points = len(core_points)
    timeout_s = max(300, n_points * 120 + 120)

    p = ctx.Process(
        target=_benchmark_curves_worker,
        args=(shared_state, param_names, rng_device, C,
              n_warmup, n_measure, zo_eps, adam_state,
              core_points, result_dict),
        daemon=True,
    )
    logger.info(f"[CalibratePC] Spawning benchmark worker: C={C}, t_train={t_train*1000:.0f}ms, "
                f"points={n_points}, timeout={timeout_s}s, "
                f"n_warmup={n_warmup}, n_measure={n_measure}")
    p.start()
    p.join(timeout=timeout_s)

    # Restore env
    for k in _thread_env:
        if _old_env[k] is not None:
            os.environ[k] = _old_env[k]
        else:
            os.environ.pop(k, None)

    # Check child process result
    if p.is_alive():
        logger.error("[CalibratePC] Worker timed out, killing...")
        p.kill()
        p.join(timeout=10)
        manager.shutdown()
        raise RuntimeError(f"calibrate_producer_consumer: worker timed out after {timeout_s}s")
    if p.exitcode != 0:
        manager.shutdown()
        raise RuntimeError(f"calibrate_producer_consumer: worker crashed with exitcode={p.exitcode}")

    # Parse curves
    import json
    if 't_gen_json' not in result_dict:
        manager.shutdown()
        raise RuntimeError("calibrate_producer_consumer: worker produced no results")
    t_gen_curve = {int(k): v for k, v in json.loads(result_dict['t_gen_json']).items()}
    t_update_curve = {int(k): v for k, v in json.loads(result_dict['t_update_json']).items()}
    n_low = int(result_dict.get('n_low', 1))
    n_high = int(result_dict.get('n_high', C - 1))
    manager.shutdown()

    logger.info(f"[CalibratePC] t_update plateau: n=[{n_low}, {n_high}]")

    # Optimize (constrained to plateau range)
    opt = optimize_thread_allocation(t_gen_curve, t_update_curve, C, t_train,
                                      n_sat_range=(n_low, n_high))

    # Memory analysis
    per_slot_bytes = sum(state[nm].numel() * state[nm].element_size() for nm in param_names)
    adam_extra = sum(state[nm].numel() * 4 * 2 for nm in param_names) if adam_state is not None else 0
    rec = opt['recommended']
    total_mem = per_slot_bytes + 1 * per_slot_bytes + adam_extra

    # Print Pareto table
    print(f"\n{'='*65}")
    print(f"Producer-Consumer Optimization (C={C}, t_train={t_train*1000:.0f}ms)")
    print(f"{'='*65}")
    print(f"{'P':>3} {'c':>5} {'c_cons':>6} {'t_step':>8} {'bottleneck':>12}")
    print(f"{'-'*45}")
    for row in opt['pareto']:
        marker = ' <--' if row is rec else ''
        print(f"{row['P']:>3} {row['c']:>5} {row['c_cons']:>6} "
              f"{row['t_step']*1000:>7.0f}ms {row['bottleneck']:>12}{marker}")
    print(f"\n  t_update plateau: n=[{n_low}, {n_high}]")
    print(f"  Recommended: P={rec['P']}, c={rec['c']}, c_cons={rec['c_cons']} "
          f"-> t_step={rec['t_step']*1000:.0f}ms")
    print(f"  Memory: shadow={per_slot_bytes/1e9:.2f}GB + "
          f"z_buf=1x{per_slot_bytes/1e9:.2f}GB = {total_mem/1e9:.2f}GB")
    print(f"\n  Env vars:")
    print(f"    SHADOW_PIPELINE_WORKERS={rec['P']}")
    print(f"    SHADOW_CONSUMER_THREADS={rec['c_cons']}")
    print(f"    SHADOW_RESERVE_THREADS=1")
    print(f"{'='*65}\n")

    return {
        't_gen_curve': t_gen_curve,
        't_update_curve': t_update_curve,
        'n_low': n_low,
        'n_high': n_high,
        'per_slot_bytes': per_slot_bytes,
        'adam_extra_bytes': adam_extra,
        'total_bytes': total_mem,
        **opt,
    }


# ============================================================
# Adam replay helpers: module-level cache for passing adam state
# between resume_from_batch_diff() and _init_for_resume()
# ============================================================
_replay_adam_state_cache = {}


def _set_replay_adam_state(adam_state):
    global _replay_adam_state_cache
    _replay_adam_state_cache = adam_state or {}


def _get_and_clear_replay_adam_state():
    global _replay_adam_state_cache
    result = _replay_adam_state_cache
    _replay_adam_state_cache = {}
    return result if result else None


def _load_adam_state_from_base(base_checkpoint_ref, fallback_optimizer_state=None):
    """Load Adam state from base checkpoint. Returns fresh m/v/t=0 for __initial__."""
    if base_checkpoint_ref == '__initial__':
        betas = fallback_optimizer_state.get('adam_betas', (0.9, 0.999)) if fallback_optimizer_state else (0.9, 0.999)
        adam_eps = fallback_optimizer_state.get('adam_eps_value', 1e-8) if fallback_optimizer_state else 1e-8
        return {'m': {}, 'v': {}, 't': 0, 'betas': betas, 'adam_eps': adam_eps}

    opt_path = os.path.join(base_checkpoint_ref, "optimizer.pt")
    if os.path.exists(opt_path):
        opt = torch.load(opt_path, map_location='cpu', weights_only=False)
        adam_state = opt.get('adam_state', None)
        if adam_state:
            return adam_state

    # Fallback: no adam_state in base checkpoint (shouldn't happen in practice)
    betas = fallback_optimizer_state.get('adam_betas', (0.9, 0.999)) if fallback_optimizer_state else (0.9, 0.999)
    adam_eps = fallback_optimizer_state.get('adam_eps_value', 1e-8) if fallback_optimizer_state else 1e-8
    logger.warning(f"[Adam Replay] No adam_state found in base {base_checkpoint_ref}, starting from t=0")
    return {'m': {}, 'v': {}, 't': 0, 'betas': betas, 'adam_eps': adam_eps}


def _apply_single_update(state, update, param_names, default_zo_eps=0.0,
                         simulate_perturbation=True, rng_device="native",
                         zo2_mode=False, prev_seed=None,
                         adam_state=None):
    """Apply one ZO update (perturbation + parameter update) to a state dict in-place.

    Simulates the full perturbation sequence [+1, -2, +1] * eps * z from zo_forward
    to match fp16 rounding residuals, then applies the actual parameter update.

    Args:
        simulate_perturbation: If True (default), simulate the [+1, -2, +1] perturbation
            loop to reproduce fp16 rounding residuals for bitwise-exact replay. If False,
            skip the perturbation loop (~4x faster) at the cost of ~1e-6 level parameter
            differences. Safe to disable for most use cases.
        rng_device: "native" (use param's device), "cpu" (always CPU, cross-GPU portable),
            or "zo_rng" (cross-device deterministic — CPU replay matches GPU training).
            Must match the rng_device used during training for bitwise-exact replay.
        zo2_mode: If True, use ZO2 replay order: gradient FIRST (using prev step's seed),
            then perturbation (using current seed). This matches ZO2 training where the
            delayed update from step N is applied before step N+1's perturbation.
        prev_seed: Seed from the previous update entry. In ZO2 mode, the gradient update
            uses z generated from the previous step's seed (via last_rstate).
        adam_state: If not None, use Adam update rule instead of SGD. Dict with keys:
            m (dict), v (dict), t (int), betas (tuple), adam_eps (float).
            Modified in-place (m/v/t updated). None = SGD (original path, unchanged).
    """
    seed = update['seed']
    grad = update['grad']
    lr = update['lr']
    wd = update.get('wd', 0.0)
    zo_eps = update.get('zo_eps', default_zo_eps)

    def _reset_rng(rng_seed=None):
        """Reset RNG; returns zo_gen for zo_rng mode, None otherwise."""
        s = rng_seed if rng_seed is not None else seed
        if rng_device == "zo_rng":
            import zo_rng
            return zo_rng.Generator(s)
        torch.manual_seed(s)
        return None

    t_start = time.time()
    t_z = 0.0
    t_update = 0.0

    if adam_state is not None:
        # ====== Adam path (regular ZO order only, no ZO2) ======
        # 1) Perturbation simulation [+1, -2, +1]
        if simulate_perturbation and zo_eps > 0:
            for scaling_factor in [1, -2, 1]:
                zo_gen = _reset_rng()
                for name in param_names:
                    param = state[name]
                    _t0 = time.time()
                    z = _generate_z_for_replay(param, rng_device, zo_gen)
                    t_z += time.time() - _t0
                    _t0 = time.time()
                    param.data.add_(z, alpha=float(scaling_factor * zo_eps))
                    t_update += time.time() - _t0

        # 2) Adam update (always runs — even for grad=0, m/v decay and t increments,
        #    matching training where zo_update is called unconditionally)
        beta1, beta2 = adam_state['betas']
        a_eps = adam_state['adam_eps']
        adam_state['t'] += 1
        t = adam_state['t']
        bc1, bc2 = 1 - beta1 ** t, 1 - beta2 ** t
        step_size = lr / bc1

        zo_gen = _reset_rng()
        for name in param_names:
            param = state[name]
            _t0 = time.time()
            z = _generate_z_for_replay(param, rng_device, zo_gen)
            t_z += time.time() - _t0
            _t0 = time.time()
            g = (grad * z).float()  # fp32 to prevent underflow

            m, v = adam_state['m'], adam_state['v']
            if name not in m:
                m[name] = torch.zeros_like(param, dtype=torch.float32)
                v[name] = torch.zeros_like(param, dtype=torch.float32)

            m[name].mul_(beta1).add_(g, alpha=1 - beta1)
            v[name].mul_(beta2).addcmul_(g, g, value=1 - beta2)

            denom = (v[name] / bc2).sqrt_().add_(a_eps)
            upd = m[name].div(denom).mul_(step_size)

            # AdamW weight decay (skip bias/layernorm)
            if all(x not in name for x in ['bias', 'layer_norm', 'layernorm', 'ln']):
                upd.add_(param.float(), alpha=lr * wd)
            param.sub_(upd.to(param.dtype))
            t_update += time.time() - _t0

        return {'total': time.time() - t_start, 'z_gen': t_z, 'update': t_update}

    # Pre-compute scalar in fp64 (matches pipeline path)
    _lr_grad = float(lr * grad)

    if zo2_mode and grad != 0:
        # ZO2 order: gradient FIRST (using prev step's seed), then perturbation.
        # In ZO2 training, module_dual_forward does:
        #   1. zo_update(module) — applies delayed grad using last_rstate (prev seed's RNG)
        #   2. zo_perturb [+1, -2, +1] — uses current seed's RNG
        grad_seed = prev_seed if prev_seed is not None else seed
        zo_gen = _reset_rng(grad_seed)
        for name in param_names:
            param = state[name]
            _t0 = time.time()
            z = _generate_z_for_replay(param, rng_device, zo_gen)
            t_z += time.time() - _t0
            _t0 = time.time()
            if wd == 0.0:
                param.sub_(z, alpha=_lr_grad)
            elif _is_wd_param(name):
                tmp = z.mul(grad)
                tmp.add_(param, alpha=wd)
                param.sub_(tmp, alpha=lr)
            else:
                param.sub_(z, alpha=_lr_grad)
            t_update += time.time() - _t0

        if simulate_perturbation and zo_eps > 0:
            for scaling_factor in [1, -2, 1]:
                zo_gen = _reset_rng(seed)
                for name in param_names:
                    param = state[name]
                    _t0 = time.time()
                    z = _generate_z_for_replay(param, rng_device, zo_gen)
                    t_z += time.time() - _t0
                    _t0 = time.time()
                    param.data.add_(z, alpha=float(scaling_factor * zo_eps))
                    t_update += time.time() - _t0
    else:
        # Regular ZO order (or grad=0 perturbation-only): perturbation first, then gradient.
        if simulate_perturbation and zo_eps > 0:
            for scaling_factor in [1, -2, 1]:
                zo_gen = _reset_rng()
                for name in param_names:
                    param = state[name]
                    _t0 = time.time()
                    z = _generate_z_for_replay(param, rng_device, zo_gen)
                    t_z += time.time() - _t0
                    _t0 = time.time()
                    param.data.add_(z, alpha=float(scaling_factor * zo_eps))
                    t_update += time.time() - _t0

        if grad != 0:
            zo_gen = _reset_rng()
            for name in param_names:
                param = state[name]
                _t0 = time.time()
                z = _generate_z_for_replay(param, rng_device, zo_gen)
                t_z += time.time() - _t0
                _t0 = time.time()
                if wd == 0.0:
                    param.sub_(z, alpha=_lr_grad)
                elif _is_wd_param(name):
                    tmp = z.mul(grad)
                    tmp.add_(param, alpha=wd)
                    param.sub_(tmp, alpha=lr)
                else:
                    param.sub_(z, alpha=_lr_grad)
                t_update += time.time() - _t0

    return {'total': time.time() - t_start, 'z_gen': t_z, 'update': t_update}


def _replay_updates_on_state(
    state: OrderedDict,
    updates: list,
    device: str = 'cpu',
    move_to_device: bool = True,
    trainable_param_names: list = None,
    default_zo_eps: float = 0.0,
    simulate_perturbation: bool = True,
    replay_in_fp32: bool = False,
    rng_device: str = "native",
    zo2_mode: bool = False,
    initial_prev_seed: int = None,
    adam_state: dict = None,
) -> OrderedDict:
    """
    Replay ZO updates on a state dict.

    Args:
        state: The state dict to modify (will be modified in-place)
        updates: List of update dicts with keys: seed, grad, lr, wd, zo_eps (optional)
        device: Device to perform computation on ('cpu' or 'cuda')
        move_to_device: If True and device='cuda', move state to GPU before replay.
                        If False, assume state is already on the correct device.
        trainable_param_names: Ordered list of parameter names that were updated during training
                               (from model.named_parameters() filtered by requires_grad=True).
                               This ensures replay matches training's RNG sequence exactly:
                               - excludes buffers (not in named_parameters)
                               - excludes frozen params (requires_grad=False)
                               - deduplicates tied weights (named_parameters deduplicates)
                               If None, falls back to iterating all state dict keys (legacy behavior).
        default_zo_eps: Fallback zo_eps value for old checkpoints that don't have zo_eps
                        in their update records. Set to the training zo_eps (e.g. 0.001)
                        to enable fp16 perturbation residual simulation for old checkpoints.
        simulate_perturbation: If True (default), simulate the [+1, -2, +1] perturbation
                        loop for bitwise-exact fp16 replay. If False, skip it (~4x faster).
        replay_in_fp32: If True, upcast fp16 state to fp32 before replay and downcast back
                        after. This avoids the ~7x penalty of torch.normal(dtype=fp16) on CPU.
                        Only affects CPU replay with fp16 models. Default False.
        initial_prev_seed: Seed from the base checkpoint step. In ZO2 mode, the first
                        replay entry (i=0) needs this as prev_seed for gradient z generation.
                        Without it, the first entry would use its own seed (wrong).

    Returns:
        The modified state dict (stays on the device where computation was done)
    """
    # ---- Parallel / Closed-form dispatch (SGD only, not Adam) ----
    if adam_state is not None:
        if os.environ.get('PARALLEL_RECOVERY') == '1' or os.environ.get('CLOSEDFORM_RECOVERY') == '1':
            logger.warning("[Replay] Adam mode only supports serial replay, ignoring PARALLEL/CLOSEDFORM")
    else:
        if os.environ.get('PARALLEL_RECOVERY', '0') == '1':
            return _parallel_replay_updates_on_state(
                state, updates, device=device, move_to_device=move_to_device,
                trainable_param_names=trainable_param_names,
                default_zo_eps=default_zo_eps,
                simulate_perturbation=simulate_perturbation,
                replay_in_fp32=replay_in_fp32,
                rng_device=rng_device,
                zo2_mode=zo2_mode,
                initial_prev_seed=initial_prev_seed,
            )

        if os.environ.get('CLOSEDFORM_RECOVERY', '0') == '1':
            return _closedform_replay_on_state(
                state, updates, device=device, move_to_device=move_to_device,
                trainable_param_names=trainable_param_names,
                rng_device=rng_device,
                zo2_mode=zo2_mode,
                initial_prev_seed=initial_prev_seed,
                num_workers=int(os.environ.get('CLOSEDFORM_WORKERS', '1')),
                precision=os.environ.get('CLOSEDFORM_PRECISION', 'mixed'),
            )

    if not updates:
        return state

    # Move to target device if needed, preserving tied weight references.
    # Without this, _tie_state_dict_inplace's aliasing is broken: each
    # state[key] = state[key].cuda() creates an independent GPU tensor,
    # so tied weights (e.g. embed_tokens & lm_head sharing storage) get
    # separate copies and updates no longer accumulate correctly.
    actual_device = 'cpu'
    if move_to_device and device == 'cuda' and torch.cuda.is_available():
        _moved = {}  # id(cpu_tensor) → gpu_tensor
        for key in state:
            _cpu_id = id(state[key])
            if _cpu_id in _moved:
                state[key] = _moved[_cpu_id]
            elif state[key].device.type != 'cuda':
                _gpu_t = state[key].cuda()
                _moved[_cpu_id] = _gpu_t
                state[key] = _gpu_t
        actual_device = 'cuda'
    elif len(state) > 0:
        # Use .type to normalize 'cuda:0' -> 'cuda'
        actual_device = next(iter(state.values())).device.type

    # Move Adam m/v to replay device (loaded from CPU via _load_adam_state_from_base)
    if adam_state is not None:
        for mv_key in ('m', 'v'):
            for name in adam_state.get(mv_key, {}):
                t_mv = adam_state[mv_key][name]
                if t_mv.device.type != actual_device:
                    adam_state[mv_key][name] = t_mv.to(actual_device)

    # Upcast fp16 → fp32 on CPU to avoid slow torch.normal(dtype=fp16) on CPU
    original_dtype = None
    if replay_in_fp32 and actual_device == 'cpu':
        sample = next(iter(state.values()))
        if sample.dtype == torch.float16 or sample.dtype == torch.bfloat16:
            original_dtype = sample.dtype
            for key in state:
                state[key] = state[key].float()
            logger.info(f"[Replay] Upcast {original_dtype} → fp32 for CPU replay")

    logger.info(f"[Replay] Replaying {len(updates)} updates on device={actual_device}"
                f" (simulate_perturbation={simulate_perturbation}, replay_in_fp32={replay_in_fp32},"
                f" rng_device={rng_device})")
    if actual_device == 'cpu' and torch.cuda.is_available() and rng_device != "zo_rng":
        logger.warning("[Replay] WARNING: Replaying on CPU but CUDA is available. "
                       "CPU and CUDA RNG produce different z for the same seed. "
                       "Use device='cuda' for exact reconstruction, or use ZO_RNG_DEVICE=zo_rng "
                       "for cross-device deterministic replay.")

    # Use trainable_param_names if available to match training's iteration order;
    # otherwise fall back to all state dict keys (backward compatible with old checkpoints)
    param_names = trainable_param_names if trainable_param_names is not None else list(state.keys())

    _seq_proc = psutil.Process(os.getpid())
    _seq_cpu0, _seq_gpu0 = _log_memory("sequential start", _seq_proc, actual_device)
    _seq_quarter = max(1, len(updates) // 4)

    timings = []
    for i, update in enumerate(updates):
        prev_seed = (initial_prev_seed if i == 0 else updates[i - 1]['seed']) if zo2_mode else None
        timing = _apply_single_update(state, update, param_names, default_zo_eps=default_zo_eps,
                             simulate_perturbation=simulate_perturbation, rng_device=rng_device,
                             zo2_mode=zo2_mode, prev_seed=prev_seed,
                             adam_state=adam_state)
        timings.append(timing)

        if i > 0 and i % _seq_quarter == 0:
            _log_memory(f"sequential step {i}/{len(updates)}", _seq_proc, actual_device, _seq_cpu0, _seq_gpu0)

        if i < 3 or i == len(updates) - 1:
            logger.info(f"[Replay] update {i}: step={update.get('step','?')}, seed={update['seed']}, "
                        f"grad={update['grad']:.6e}, lr={update['lr']}, wd={update.get('wd', 0.0)}, "
                        f"zo_eps={update.get('zo_eps', default_zo_eps)}, "
                        f"time={timing['total']:.4f}s (z_gen={timing['z_gen']:.4f}s, update={timing['update']:.4f}s)")
        elif i == 3:
            logger.info(f"[Replay] ... ({len(updates) - 4} more updates) ...")

    if timings:
        avg_total = sum(t['total'] for t in timings) / len(timings)
        avg_z = sum(t['z_gen'] for t in timings) / len(timings)
        avg_upd = sum(t['update'] for t in timings) / len(timings)
        total_z = sum(t['z_gen'] for t in timings)
        total_upd = sum(t['update'] for t in timings)
        logger.info(f"[Replay Timing] avg per step: total={avg_total:.4f}s, z_gen={avg_z:.4f}s, update={avg_upd:.4f}s")
        logger.info(f"[Replay Timing] total: z_gen={total_z:.3f}s, update={total_upd:.3f}s")

    # Downcast fp32 → original dtype
    if original_dtype is not None:
        for key in state:
            state[key] = state[key].to(original_dtype)
        logger.info(f"[Replay] Downcast fp32 → {original_dtype}")

    # Don't move back to CPU - keep on GPU for training
    return state


def _restore_tied_weights(state_dict, checkpoint_dir):
    """Restore tied weights that were deduplicated during saving.

    HuggingFace saves only one copy for tied weights (e.g., lm_head.weight
    and model.embed_tokens.weight). This function restores the missing key
    by reading tie_word_embeddings from config.json in the checkpoint dir.
    """
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path, 'r') as f:
        config = json.load(f)
    if config.get('tie_word_embeddings', False):
        if 'model.embed_tokens.weight' in state_dict and 'lm_head.weight' not in state_dict:
            state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight']


def load_batch_diff_checkpoint(checkpoint_dir, base_checkpoint_dir=None, device='cpu',
                               simulate_perturbation=True, replay_in_fp32=False):
    """
    Load batch differential checkpoint.

    Args:
        checkpoint_dir: Path to the checkpoint directory
        base_checkpoint_dir: Optional. Path to base model (required for "__initial__" mode)
        device: Device for replay computation ('cpu' or 'cuda'). Default 'cpu'.
        simulate_perturbation: If True (default), simulate fp16 perturbation loop during replay.
                If False, skip it for ~4x faster replay.

    Returns:
        Reconstructed state_dict (on CPU)
    """
    # Try loading model files directly (safetensors, pytorch_model.bin)
    safe_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(safe_path):
        from safetensors.torch import load_file
        state_dict = load_file(safe_path)
        _restore_tied_weights(state_dict, checkpoint_dir)
        return state_dict

    model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        _restore_tied_weights(state_dict, checkpoint_dir)
        return state_dict

    full_model_path = os.path.join(checkpoint_dir, "pytorch_model_full.bin")
    if os.path.exists(full_model_path):
        return torch.load(full_model_path, map_location='cpu', weights_only=True)

    # No model files found — this is a log-only checkpoint
    # Caller should use resume_from_batch_diff() which reads optimizer.pt for replay
    return None


def _load_base_state(base_checkpoint_ref, pretrained_model_name, tied_groups, model_dtype,
                     output_dir=None):
    """Load the base model state dict for replay.

    Returns:
        (base_state, tied_groups) where tied_groups may be updated if detected from model.
    """
    if base_checkpoint_ref == "__initial__":
        # Try loading from cached initial_model in output_dir first
        if output_dir is not None:
            initial_model_dir = os.path.join(output_dir, "initial_model")
            safe_path = os.path.join(initial_model_dir, "model.safetensors")
            bin_path = os.path.join(initial_model_dir, "pytorch_model.bin")
            if os.path.exists(safe_path):
                from safetensors.torch import load_file
                logger.info(f"[Resume] Loading initial model from cached {safe_path}")
                base_state = load_file(safe_path)
                return base_state, tied_groups
            elif os.path.exists(bin_path):
                logger.info(f"[Resume] Loading initial model from cached {bin_path}")
                base_state = torch.load(bin_path, map_location='cpu', weights_only=True)
                return base_state, tied_groups

        # Fallback: load from pretrained model
        if pretrained_model_name is None:
            raise ValueError(
                "This checkpoint uses differential mode from initial model. "
                "You must provide pretrained_model_name to load it."
            )
        try:
            from transformers import AutoModelForCausalLM
            dtype_kwargs = {'torch_dtype': model_dtype} if model_dtype is not None else {}
            logger.info(f"[Resume] Loading base model from HuggingFace: {pretrained_model_name} (dtype={model_dtype})")
            base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, **dtype_kwargs)
            if not tied_groups:
                tied_groups = _detect_tied_weights(base_model)
                if tied_groups:
                    logger.info(f"[Resume] Detected tied weight groups from model: {tied_groups}")
            base_state = base_model.state_dict()
            del base_model
        except Exception as e:
            raise FileNotFoundError(f"Cannot load pretrained model {pretrained_model_name}: {e}")
    else:
        base_state = load_batch_diff_checkpoint(base_checkpoint_ref, base_checkpoint_dir=pretrained_model_name)
        if base_state is None:
            raise FileNotFoundError(f"Cannot load base checkpoint from {base_checkpoint_ref}")

    return base_state, tied_groups


def resume_from_batch_diff(
    checkpoint_path: str,
    output_dir: str = None,
    pretrained_model_name: str = None,
    device: str = 'cpu',
    simulate_perturbation: bool = True,
    replay_in_fp32: bool = False,
    base_state_dict: OrderedDict = None,
    cached_optimizer_state: dict = None,
    rng_device: str = "native",
    zo2_mode: bool = False,
) -> OrderedDict:
    """
    Resume from batch differential checkpoints.

    Args:
        checkpoint_path: Path to the checkpoint to resume from (will find latest if directory given)
        output_dir: Optional. Directory containing checkpoints. If None, uses parent of checkpoint_path
        pretrained_model_name: Optional. Name of pretrained model for differential mode (e.g., "Qwen/Qwen2.5-1.5B")
                               Required if checkpoints use "__initial__" as base
        device: Device for replay computation ('cpu' or 'cuda'). Default 'cpu'.
                Use 'cuda' for faster replay if GPU memory is available.
        simulate_perturbation: If True (default), simulate the [+1, -2, +1] fp16 perturbation
                loop during replay for bitwise-exact reconstruction. If False, skip it for
                ~4x faster replay with negligible (~1e-6) parameter differences.
        replay_in_fp32: If True, upcast fp16 state to fp32 during CPU replay to avoid
                the ~7x penalty of torch.normal(dtype=fp16) on CPU. Default False.
        base_state_dict: Optional. Pre-loaded base model state_dict (e.g., from model already
                in memory). If provided, skips _load_base_state to avoid redundant disk I/O.
        cached_optimizer_state: Optional. Pre-loaded optimizer.pt content. If provided,
                skips loading optimizer.pt again (avoids double loading).
        zo2_mode: If True, use ZO2 replay order (gradient first with prev step's seed,
                then perturbation). Auto-detected from checkpoint metadata if not specified.

    Returns:
        Reconstructed state_dict at the checkpoint (on CPU)

    Replay strategies based on batch_size:
    - batch_size=0 (Log-based): Each checkpoint contains ALL updates from base.
                                Just load the latest checkpoint and replay its updates.
    - batch_size>=1 (Full + Log): Full checkpoint every N steps, log checkpoints in between.
                                  Each log checkpoint contains all updates from base.
    """
    # Determine checkpoint directory
    ckpt_dir = os.path.dirname(checkpoint_path) if os.path.isfile(checkpoint_path) else checkpoint_path

    # Get the output directory (parent of checkpoint directories)
    if output_dir is None:
        output_dir = os.path.dirname(ckpt_dir)

    # Extract step number from checkpoint
    match = re.search(r'checkpoint-(\d+)', ckpt_dir)
    if not match:
        raise ValueError(f"Cannot extract step from checkpoint path: {ckpt_dir}")
    target_step = int(match.group(1))

    logger.info(f"[Resume] Target checkpoint: {ckpt_dir} (step {target_step})")
    logger.info(f"[Resume] Replay device: {device}")

    # Load update history from optimizer.pt
    optimizer_path = os.path.join(ckpt_dir, "optimizer.pt")

    # Use cached optimizer state if provided (avoids double loading from disk)
    if cached_optimizer_state is not None and isinstance(cached_optimizer_state, dict) and 'zo_update_history' in cached_optimizer_state:
        optimizer_state = cached_optimizer_state
        logger.info(f"[Resume] Using cached optimizer state (skipped disk I/O)")
    elif os.path.exists(optimizer_path):
        optimizer_state = torch.load(optimizer_path, map_location='cpu', weights_only=False)
        if not isinstance(optimizer_state, dict) or 'zo_update_history' not in optimizer_state:
            # Regular optimizer.pt without update history — not a log-based checkpoint
            logger.info(f"[Resume] optimizer.pt has no zo_update_history, loading as regular checkpoint")
            return load_batch_diff_checkpoint(ckpt_dir, base_checkpoint_dir=pretrained_model_name)
        logger.info(f"[Resume] Found log-based checkpoint (optimizer.pt with zo_update_history)")
    else:
        logger.info(f"[Resume] No optimizer.pt found, loading as regular checkpoint")
        return load_batch_diff_checkpoint(ckpt_dir, base_checkpoint_dir=pretrained_model_name)

    # Extract metadata from optimizer state
    batch_size = optimizer_state.get('batch_size', 0)
    base_checkpoint_ref = optimizer_state.get('base_checkpoint', '__initial__')
    updates = optimizer_state['zo_update_history']
    tied_groups = optimizer_state.get('tied_weights', [])
    trainable_param_names = optimizer_state.get('trainable_param_names', None)
    default_zo_eps = optimizer_state.get('zo_eps', 0.0)
    model_dtype_str = optimizer_state.get('model_dtype', None)
    pending_grad = optimizer_state.get('pending_grad', None)
    base_pending_seed = optimizer_state.get('base_pending_seed', None)
    is_full_checkpoint = optimizer_state.get('is_full_checkpoint', False)

    # Auto-detect rng_device from checkpoint if caller didn't specify
    ckpt_rng_device = optimizer_state.get('rng_device', 'native')
    if rng_device == "native" and ckpt_rng_device != "native":
        rng_device = ckpt_rng_device
        logger.info(f"[Resume] Auto-detected rng_device={rng_device} from checkpoint")

    # Auto-detect zo2_mode from checkpoint if caller didn't specify
    if not zo2_mode:
        ckpt_zo2 = optimizer_state.get('zo2', False)
        if ckpt_zo2:
            zo2_mode = True
            logger.info(f"[Resume] Auto-detected zo2_mode=True from checkpoint")
    if zo2_mode:
        logger.info(f"[Resume] ZO2 mode: will use prev-step seed for gradient, current seed for perturbation")

    model_dtype = _DTYPE_MAP.get(model_dtype_str, None)

    logger.info(f"[Resume] Checkpoint mode: batch_size={batch_size}, base_checkpoint={base_checkpoint_ref}")
    if model_dtype is not None:
        logger.info(f"[Resume] Model dtype from checkpoint: {model_dtype}")
    if tied_groups:
        logger.info(f"[Resume] Tied weight groups from checkpoint: {tied_groups}")

    # batch_size=0 always replays from initial pretrained model
    if batch_size == 0 and base_checkpoint_ref != "__initial__":
        logger.warning(
            f"[Resume] batch_size=0 but base_checkpoint={base_checkpoint_ref}, "
            f"overriding to __initial__ (batch_size=0 always replays from initial model)"
        )
        base_checkpoint_ref = "__initial__"

    # Detect Adam optimizer
    zo_method = optimizer_state.get('zo_method', 'mezo-sgd')
    is_adam = (zo_method == 'mezo-adam')
    adam_state = None

    # For full checkpoints (batch_size >= 1), load model directly — no replay needed
    if is_full_checkpoint:
        logger.info(f"[Resume] Target is a full checkpoint, loading directly")
        if is_adam:
            # Full checkpoint: adam state in optimizer.pt, _init_for_resume will restore it
            _set_replay_adam_state(None)
        return load_batch_diff_checkpoint(ckpt_dir, base_checkpoint_dir=pretrained_model_name)

    # For log checkpoints with Adam: load adam state from base, replay will update it
    if is_adam:
        adam_state = _load_adam_state_from_base(base_checkpoint_ref, optimizer_state)
        logger.info(f"[Resume] Adam mode: loaded base adam state (t={adam_state.get('t', 0)}, "
                    f"betas={adam_state.get('betas')}, {len(adam_state.get('m', {}))} m/v entries)")

    # Load base model (common to all modes)
    if base_state_dict is not None and base_checkpoint_ref == "__initial__":
        logger.info(f"[Resume] Using pre-loaded base state dict in-place (no clone)")
        reconstructed = base_state_dict
    else:
        base_state, tied_groups = _load_base_state(
            base_checkpoint_ref, pretrained_model_name, tied_groups, model_dtype,
            output_dir=output_dir
        )

        reconstructed = base_state  # load_file returns fresh tensors, no clone needed

    # ========== REPLAY UPDATES ==========
    logger.info(f"[Resume] Replaying {len(updates)} updates (default_zo_eps={default_zo_eps})")
    if pending_grad is not None:
        logger.info(f"[Resume] pending_grad={pending_grad} (will be restored to opt.projected_grad)")

    # Move to GPU before replay so timing measures pure computation
    if device == 'cuda' and torch.cuda.is_available():
        for key in reconstructed:
            if reconstructed[key].device.type != 'cuda':
                reconstructed[key] = reconstructed[key].cuda()
        torch.cuda.synchronize()

    # Tie weights AFTER GPU move: .cuda() creates new tensors per key, breaking
    # any prior tie (same-object identity). Tying here ensures tied params share
    # one GPU tensor, so in-place replay updates both correctly.
    if tied_groups:
        _tie_state_dict_inplace(reconstructed, tied_groups)

    # ---- Memory snapshot: before replay ----
    _mem_proc = psutil.Process(os.getpid())
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _mem_cpu0, _mem_gpu0 = _log_memory("before replay", _mem_proc, device)

    t_replay_start = time.time()
    _replay_updates_on_state(reconstructed, updates, device=device,
                             move_to_device=False,
                             trainable_param_names=trainable_param_names,
                             default_zo_eps=default_zo_eps,
                             simulate_perturbation=simulate_perturbation,
                             replay_in_fp32=replay_in_fp32,
                             rng_device=rng_device,
                             zo2_mode=zo2_mode,
                             initial_prev_seed=base_pending_seed,
                             adam_state=adam_state)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    t_replay = time.time() - t_replay_start
    logger.info(f"[Resume Replay] {len(updates)} updates replayed in {t_replay:.3f}s (device={device})")

    # Cache replayed Adam state for _init_for_resume to pick up
    if is_adam and adam_state is not None:
        _set_replay_adam_state(adam_state)
        logger.info(f"[Resume] Cached replayed Adam state: t={adam_state.get('t', 0)}")

    # ---- Memory snapshot: after replay ----
    _log_memory("after replay", _mem_proc, device, _mem_cpu0, _mem_gpu0)

    # Verification logging
    if updates:
        last = updates[-1]
        logger.info(f"[VERIFY-RESUME] Last replayed update: step={last.get('step','?')}, "
                    f"seed={last['seed']}, grad={last['grad']:.6e}")
    if pending_grad is not None:
        logger.info(f"[VERIFY-RESUME] pending_grad={pending_grad} => first resumed step should apply this grad")

    logger.info(f"[Resume] Completed! Recovered to step {target_step}")

    return reconstructed


# Backward compatibility
load_zo_replay_checkpoint = load_batch_diff_checkpoint
load_incremental_checkpoint = load_batch_diff_checkpoint
