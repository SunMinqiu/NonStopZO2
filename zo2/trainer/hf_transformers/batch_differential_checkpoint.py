"""
Batch Differential Checkpoint for ZO Training:
- Supports different checkpoint modes controlled by batch_size parameter:
  - batch_size = -1: Disabled (L0 baseline, uses default Trainer checkpoint)
  - batch_size = 0: Log-based (accumulate all updates from initial model)
  - batch_size >= 1: Full + Log (full checkpoint every N steps, log checkpoints in between)
    - Optional: enable_shadow for real-time shadow model on CPU (instant recovery)
- Controllable failure injection for testing
"""

import os
import re
import threading
import time
import torch
from transformers import TrainerCallback
from collections import OrderedDict
import logging
import json
import psutil

logger = logging.getLogger(__name__)


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
            enable_shadow: Enable real-time shadow model on CPU (only effective for batch_size>=1)
            instant_recover: Instantly recover from shadow on GPU failure (requires enable_shadow)
        """
        self.batch_size = batch_size
        # Shadow only makes sense for batch_size>=1 (bounded replay cost).
        # For batch_size<=0: -1 is disabled, 0 replays from initial (no base to shadow).
        self.enable_shadow = enable_shadow if batch_size >= 1 else False
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

        # Start shadow model thread
        if self.enable_shadow:
            self._start_shadow_thread()
            if self.instant_recover:
                logger.info("[BatchDiff] Mode: L3 (Instant Recovery) - shadow tracking + instant recovery")
            else:
                logger.info("[BatchDiff] Mode: L2 (CPU Shadow) - real-time shadow tracking")
        else:
            mode_desc = self._get_mode_description()
            logger.info(f"[BatchDiff] Mode: L1 ({mode_desc}) - on-demand reconstruction")

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
            self.base_checkpoint_state[key] = value.detach().cpu().clone()

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

    def _refresh_shadow_from_base(self):
        """Refresh shadow model from base_checkpoint_state."""
        with self.shadow_lock:
            self.shadow_model = OrderedDict()
            for key, value in self.base_checkpoint_state.items():
                self.shadow_model[key] = value.clone()
            if hasattr(self, '_tied_weight_groups') and self._tied_weight_groups:
                _tie_state_dict_inplace(self.shadow_model, self._tied_weight_groups)
            self.shadow_step = len(self.update_history)

    def _start_shadow_thread(self):
        """Start shadow model tracking thread"""
        if self.shadow_thread is not None and self.shadow_thread.is_alive():
            return

        self.shadow_running = True
        self.shadow_thread = threading.Thread(target=self._shadow_worker, daemon=True)
        self.shadow_thread.start()
        logger.info("[BatchDiff] Started shadow model thread")

    def _shadow_worker(self):
        """Background thread: continuously track GPU updates"""
        logger.info("[Shadow] Worker started")

        while self.shadow_running:
            try:
                with self.update_lock:
                    pending = len(self.update_history) - self.shadow_step

                if pending > 0 and self.shadow_model is not None:
                    with self.update_lock:
                        if self.shadow_step < len(self.update_history):
                            update = self.update_history[self.shadow_step].copy()
                        else:
                            continue

                    self._apply_update_to_shadow(update)

                    with self.shadow_lock:
                        self.shadow_step += 1

                    if self.shadow_step % 10 == 0:
                        logger.info(f"[Shadow] Caught up to step {self.shadow_step}")
                else:
                    time.sleep(0.001)

            except Exception as e:
                logger.error(f"[Shadow] Error: {e}")
                time.sleep(0.1)

        logger.info("[Shadow] Worker stopped")

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
            _apply_single_update(self.shadow_model, update, param_names, rng_device=_rng_device)

    def _zo_update_hook(self, model, inputs, loss):
        """Hook called after ZO training step"""
        # Check if should simulate GPU failure
        if self.failure_simulator.check_and_fail(self.current_step, model):
            self.failure_simulator.trigger_failure(model)

            if self.instant_recover:
                if not self.enable_shadow:
                    logger.warning("[Recovery] instant_recover requires enable_shadow=True!")
                else:
                    recovered_model = self.recover_from_shadow()
                    if recovered_model is not None:
                        model.load_state_dict(recovered_model)
                        model.to('cuda')
                        logger.info("[Recovery] Model recovered and loaded to GPU!")
            else:
                logger.info("[GPU Failure] Failure injected but instant_recover=False, no recovery performed")

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

    def on_step_end(self, args, state, control, **kwargs):
        """Called at end of each step"""
        self.current_step = state.global_step

        if self.current_step % 10 == 0:
            with self.update_lock:
                num_updates = len(self.update_history)
            if self.enable_shadow:
                with self.shadow_lock:
                    shadow_step = self.shadow_step
                logger.info(f"[BatchDiff] GPU step {self.current_step}, Shadow step {shadow_step}, "
                           f"Updates: {num_updates}")
            else:
                logger.info(f"[BatchDiff] GPU step {self.current_step}, Updates: {num_updates} (on-demand mode)")

    def recover_from_shadow(self) -> OrderedDict:
        """
        Recover from shadow model or update_history.
        Returns state_dict that can be directly load_state_dict.
        """
        t_start = time.time()

        if self.enable_shadow:
            with self.shadow_lock:
                if self.shadow_model is None:
                    logger.error("[Recovery] No shadow model available!")
                    return None

                shadow_step = self.shadow_step

                recovered = OrderedDict()
                for key, value in self.shadow_model.items():
                    recovered[key] = value.clone()

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

        with self.shadow_lock:
            recovered_step = self.shadow_step

        logger.info(f"[Recovery] Successfully recovered to step {recovered_step}")
        return recovered_step

    def get_recovery_status(self) -> dict:
        """Get current recovery status"""
        if self.enable_shadow:
            with self.shadow_lock:
                shadow_step = self.shadow_step
                shadow_available = self.shadow_model is not None
        else:
            shadow_step = -1
            shadow_available = False

        with self.update_lock:
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
        Also refreshes shadow model if enabled."""
        state_dict = model.state_dict()

        self.base_checkpoint_state = OrderedDict()
        for key, value in state_dict.items():
            self.base_checkpoint_state[key] = value.detach().cpu().clone()

        if self.enable_shadow:
            self._refresh_shadow_from_base()

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
            with self.shadow_lock:
                shadow_step = self.shadow_step if self.shadow_model else -1
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
            self.shadow_running = False
            if self.shadow_thread and self.shadow_thread.is_alive():
                self.shadow_thread.join(timeout=5.0)

        status = self.get_recovery_status()
        logger.info(f"[BatchDiff] Final status: {status}")

        if self.base_checkpoint_state:
            del self.base_checkpoint_state
            self.base_checkpoint_state = None
        if self.shadow_model:
            del self.shadow_model
            self.shadow_model = None

        logger.info(f"[BatchDiff] Done. {self.save_count} checkpoints saved")


# Backward compatibility aliases
ZOReplayCheckpointCallback = BatchDiffCheckpointCallback
IncrementalCheckpointCallback = BatchDiffCheckpointCallback


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


def _apply_single_update(state, update, param_names, default_zo_eps=0.0,
                         simulate_perturbation=True, rng_device="native",
                         zo2_mode=False, prev_seed=None):
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

    if zo2_mode and grad != 0:
        # ZO2 order: gradient FIRST (using prev step's seed), then perturbation.
        # In ZO2 training, module_dual_forward does:
        #   1. zo_update(module) — applies delayed grad using last_rstate (prev seed's RNG)
        #   2. zo_perturb [+1, -2, +1] — uses current seed's RNG
        grad_seed = prev_seed if prev_seed is not None else seed
        zo_gen = _reset_rng(grad_seed)
        for name in param_names:
            param = state[name]
            z = _generate_z_for_replay(param, rng_device, zo_gen)
            if 'bias' not in name and 'layer_norm' not in name and 'layernorm' not in name and 'ln' not in name:
                param.sub_(lr * (grad * z + wd * param))
            else:
                param.sub_(lr * grad * z)

        if simulate_perturbation and zo_eps > 0:
            for scaling_factor in [1, -2, 1]:
                zo_gen = _reset_rng(seed)
                for name in param_names:
                    param = state[name]
                    z = _generate_z_for_replay(param, rng_device, zo_gen)
                    param.data.add_(scaling_factor * z * zo_eps)
    else:
        # Regular ZO order (or grad=0 perturbation-only): perturbation first, then gradient.
        if simulate_perturbation and zo_eps > 0:
            for scaling_factor in [1, -2, 1]:
                zo_gen = _reset_rng()
                for name in param_names:
                    param = state[name]
                    z = _generate_z_for_replay(param, rng_device, zo_gen)
                    param.data.add_(scaling_factor * z * zo_eps)

        if grad != 0:
            zo_gen = _reset_rng()
            for name in param_names:
                param = state[name]
                z = _generate_z_for_replay(param, rng_device, zo_gen)
                if 'bias' not in name and 'layer_norm' not in name and 'layernorm' not in name and 'ln' not in name:
                    param.sub_(lr * (grad * z + wd * param))
                else:
                    param.sub_(lr * grad * z)


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
    if not updates:
        return state

    # Move to target device if needed
    actual_device = 'cpu'
    if move_to_device and device == 'cuda' and torch.cuda.is_available():
        for key in state:
            if state[key].device.type != 'cuda':
                state[key] = state[key].cuda()
        actual_device = 'cuda'
    elif len(state) > 0:
        actual_device = str(next(iter(state.values())).device)

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

    for i, update in enumerate(updates):
        prev_seed = (initial_prev_seed if i == 0 else updates[i - 1]['seed']) if zo2_mode else None
        _apply_single_update(state, update, param_names, default_zo_eps=default_zo_eps,
                             simulate_perturbation=simulate_perturbation, rng_device=rng_device,
                             zo2_mode=zo2_mode, prev_seed=prev_seed)

        if i < 3 or i == len(updates) - 1:
            logger.info(f"[Replay] update {i}: step={update.get('step','?')}, seed={update['seed']}, "
                        f"grad={update['grad']:.6e}, lr={update['lr']}, wd={update.get('wd', 0.0)}, "
                        f"zo_eps={update.get('zo_eps', default_zo_eps)}")
        elif i == 3:
            logger.info(f"[Replay] ... ({len(updates) - 4} more updates) ...")

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

    # For full checkpoints (batch_size >= 1), load model directly — no replay needed
    if is_full_checkpoint:
        logger.info(f"[Resume] Target is a full checkpoint, loading directly")
        return load_batch_diff_checkpoint(ckpt_dir, base_checkpoint_dir=pretrained_model_name)

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

    t_replay_start = time.time()
    _replay_updates_on_state(reconstructed, updates, device=device,
                             move_to_device=False,
                             trainable_param_names=trainable_param_names,
                             default_zo_eps=default_zo_eps,
                             simulate_perturbation=simulate_perturbation,
                             replay_in_fp32=replay_in_fp32,
                             rng_device=rng_device,
                             zo2_mode=zo2_mode,
                             initial_prev_seed=base_pending_seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    t_replay = time.time() - t_replay_start
    logger.info(f"[Resume Replay] {len(updates)} updates replayed in {t_replay:.3f}s (device={device})")

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
