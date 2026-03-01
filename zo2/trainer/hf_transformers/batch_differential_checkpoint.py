"""
Batch Differential Checkpoint for ZO Training:
- Supports different checkpoint modes controlled by batch_size parameter:
  - batch_size = -1: Disabled (L0 baseline, uses default Trainer checkpoint)
  - batch_size = 0: Every step accumulative (incremental, all updates from base)
  - batch_size = 1: Pure differential (only current step's update)
  - batch_size >= 2: Batch differential (every N steps, clear history and save new full checkpoint)
- Real-time shadow model on CPU for instant recovery
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
import glob

logger = logging.getLogger(__name__)

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

        # Make all other keys in the group reference the primary tensor
        for name in group:
            if name != primary and name in state:
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
    - batch_size = 0: Incremental mode - accumulate all updates from first checkpoint
    - batch_size = 1: Pure differential - only save current step's update
    - batch_size >= 2: Batch differential - every N steps, save new full checkpoint and clear history

    Additional features:
    - L2 CPU Shadow (enable_shadow=True): Real-time shadow model on CPU
    - L3 Instant Recovery (instant_recover=True): Recover from shadow on GPU failure
    """

    def __init__(self, batch_size=0, enable_shadow=True, instant_recover=False, save_full_model=False):
        """
        Args:
            batch_size: Checkpoint mode
                -1: Disabled
                0: Incremental (accumulate all updates)
                1: Pure differential (only current step)
                >=2: Batch differential (new full checkpoint every N steps)
            enable_shadow: L2 - Enable real-time shadow model on CPU
            instant_recover: L3 - Instantly recover from shadow on GPU failure
            save_full_model: For batch_size >= 2, whether to save physical full model at periodic checkpoints
                           (calls Trainer._save_checkpoint to save standard checkpoint)
        """
        self.batch_size = batch_size
        self.enable_shadow = enable_shadow
        self.instant_recover = instant_recover
        self.save_full_model = save_full_model

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

        # Flag to prevent recursion when calling Trainer._save_checkpoint
        self._saving_full_via_trainer = False

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

        # Cache initial model
        if self.base_checkpoint_state is None and model is not None:
            self._cache_initial_model(model)

        # Restore pending_grad from checkpoint on resume.
        # This is the last computed grad that wasn't applied yet; it needs to be
        # set as projected_grad so the first resumed step's zo_update applies it.
        batchdiff_resume = getattr(args, 'batchdiff_resume', '')
        if batchdiff_resume and model is not None and hasattr(model, 'opt'):
            pending_grad = self._load_pending_grad(batchdiff_resume)
            if pending_grad is not None:
                model.opt.projected_grad = pending_grad
                logger.info(f"[BatchDiff Resume] Restored pending_grad={pending_grad:.6e} "
                            f"to model.opt.projected_grad (will be applied in first step's zo_update)")
            else:
                logger.warning("[BatchDiff Resume] No pending_grad found in checkpoint, "
                               "first step will skip zo_update (projected_grad=0)")

        # Sync current_step with trainer state so the hook reports correct step numbers.
        # Without this, the first hook call after resume reports step=1 instead of step=101
        # because self.current_step was initialized to 0 in __init__.
        self.current_step = state.global_step

        # On resume, load the previous update history so that new checkpoints
        # include ALL updates from base (required for batch_size=0 mode).
        if batchdiff_resume and self.batch_size == 0:
            previous_updates = self._load_previous_update_history(batchdiff_resume)
            if previous_updates is not None:
                self.update_history = previous_updates
                logger.info(f"[BatchDiff Resume] Loaded {len(self.update_history)} previous updates "
                            f"from checkpoint (batch_size=0 accumulative mode)")
            else:
                logger.warning("[BatchDiff Resume] No previous update history found in checkpoint, "
                               "starting fresh update_history")

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

    def _load_pending_grad(self, checkpoint_path: str):
        """Load pending_grad from a checkpoint directory.
        Checks zo_replay_history.json first, then optimizer.pt (log-based mode)."""
        # Try zo_replay_history.json
        history_path = os.path.join(checkpoint_path, "zo_replay_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            pg = history.get('pending_grad', None)
            if pg is not None:
                return float(pg)

        # Try optimizer.pt (batch_size=0 log-based mode)
        opt_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(opt_path):
            try:
                opt_state = torch.load(opt_path, map_location='cpu', weights_only=False)
                if isinstance(opt_state, dict):
                    pg = opt_state.get('pending_grad', None)
                    if pg is not None:
                        return float(pg)
            except Exception as e:
                logger.warning(f"[BatchDiff] Failed to load pending_grad from optimizer.pt: {e}")

        return None

    def _load_previous_update_history(self, checkpoint_path: str):
        """Load previous update history from a checkpoint directory for batch_size=0 mode.
        Returns list of update dicts, or None if not found."""
        # Try optimizer.pt (batch_size=0 log-based mode)
        opt_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(opt_path):
            try:
                opt_state = torch.load(opt_path, map_location='cpu', weights_only=False)
                if isinstance(opt_state, dict) and 'zo_update_history' in opt_state:
                    updates = opt_state['zo_update_history']
                    if isinstance(updates, list):
                        return list(updates)
            except Exception as e:
                logger.warning(f"[BatchDiff] Failed to load update history from optimizer.pt: {e}")

        # Try zo_replay_history.json (differential mode)
        history_path = os.path.join(checkpoint_path, "zo_replay_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                updates = history.get('updates', None)
                if isinstance(updates, list):
                    return list(updates)
            except Exception as e:
                logger.warning(f"[BatchDiff] Failed to load update history from zo_replay_history.json: {e}")

        return None

    def _get_mode_description(self) -> str:
        """Get mode description string"""
        if self.batch_size == -1:
            return "Disabled"
        elif self.batch_size == 0:
            return "Incremental (accumulate all)"
        elif self.batch_size == 1:
            return "Pure Differential (current step only)"
        else:
            return f"Batch Differential (every {self.batch_size} steps)"

    def _cache_initial_model(self, model):
        """Cache initial model to CPU memory (no disk save for pure differential mode)"""
        logger.info("[BatchDiff] Caching initial model to CPU memory...")
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
            self.shadow_model = OrderedDict()
            for key, value in self.base_checkpoint_state.items():
                self.shadow_model[key] = value.clone()
            # Tie weights in shadow model so ZO updates accumulate correctly
            if self._tied_weight_groups:
                _tie_state_dict_inplace(self.shadow_model, self._tied_weight_groups)
            self.shadow_step = 0

        self.base_checkpoint_step = 0
        # For differential modes (batch_size >= 0), we don't save initial model to disk
        # Recovery will use the original pretrained model + replay updates
        self.base_checkpoint_path = "__initial__"  # Special marker for initial state

        mem_mb = self._get_memory_size()
        t_elapsed = time.time() - t_start
        logger.info(f"[BatchDiff] Initial model cached ({mem_mb:.1f} MB) in {t_elapsed:.3f}s (no disk save)")
        self._log_memory_status()

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
        """Apply one update to shadow model."""
        with self.shadow_lock:
            param_names = self._trainable_param_names if self._trainable_param_names else list(self.shadow_model.keys())
            _apply_single_update(self.shadow_model, update, param_names)

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

                # Only record if an actual update was applied in this step.
                # Step 0 has applied_grad=0 (projected_grad initialized to 0), so it's skipped.
                if applied_grad != 0:
                    update = {
                        'step': actual_step,
                        'seed': int(seed) if seed is not None else 0,
                        'grad': float(applied_grad) if not isinstance(applied_grad, float) else applied_grad,
                        'lr': float(lr),
                        'wd': float(wd),
                        'zo_eps': float(zo_eps),
                    }
                    self.update_history.append(update)
                    logger.info(f"[HOOK] step={actual_step}, UPDATE RECORDED: seed={update['seed']}, "
                                f"applied_grad={update['grad']:.6e}, new_grad={new_grad:.6e}, lr={lr}, wd={wd}")
                    # Verification tag: grep "[VERIFY]" to cross-check train vs replay
                    logger.info(f"[VERIFY] step={actual_step} total_updates={len(self.update_history)} "
                                f"pending_grad={self._pending_grad:.6e}")
                else:
                    logger.info(f"[HOOK] step={actual_step}, NO UPDATE (applied_grad=0), "
                                f"new_grad={new_grad:.6e} (will be applied next step)")

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

        # Fix tied weights that may have diverged during replay
        _tie_state_dict_inplace(reconstructed, self._tied_weight_groups)

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
        """Called when checkpoint is saved"""
        if model is None:
            return

        # Skip if we're being called recursively from Trainer._save_checkpoint
        if self._saving_full_via_trainer:
            logger.info(f"[BatchDiff] Skipping on_save (called from Trainer._save_checkpoint)")
            return

        self.save_count += 1
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        # batch_size=0 (log-based): full checkpoint already saved by Trainer._save_checkpoint.
        # Only update internal state if shadow model is enabled (needed for recover_from_shadow).
        if self.batch_size == 0:
            if self.enable_shadow:
                self._update_base_and_shadow(model, checkpoint_dir, state.global_step)
            return

        # Determine if we need to save a full checkpoint
        should_save_full = self._should_save_full_checkpoint(state.global_step)

        if should_save_full:
            self._save_full_checkpoint(model, checkpoint_dir, state.global_step)
        else:
            self._save_diff_checkpoint(checkpoint_dir, state.global_step)

    def _update_base_and_shadow(self, model, checkpoint_dir, step):
        """Update internal state (base checkpoint + shadow model) without saving to disk.
        Used by batch_size=0 mode where full checkpoint is already saved by Trainer."""
        state_dict = model.state_dict()

        self.base_checkpoint_state = OrderedDict()
        for key, value in state_dict.items():
            self.base_checkpoint_state[key] = value.detach().cpu().clone()

        if self.enable_shadow:
            with self.shadow_lock:
                self.shadow_model = OrderedDict()
                for key, value in self.base_checkpoint_state.items():
                    self.shadow_model[key] = value.clone()
                if hasattr(self, '_tied_weight_groups') and self._tied_weight_groups:
                    _tie_state_dict_inplace(self.shadow_model, self._tied_weight_groups)
                self.shadow_step = len(self.update_history)

        # Don't update base_checkpoint_path for batch_size=0 — it always replays from __initial__
        self.last_saved_step = step
        self.is_first_save = False

    def _should_save_full_checkpoint(self, step: int) -> bool:
        """Determine if we should save a full checkpoint.
        Note: batch_size=0 never reaches here (on_save returns early)."""
        if self.batch_size == -1:
            return True
        elif self.batch_size == 1:
            # Pure differential - NEVER save full (initial model already cached)
            return False
        else:
            # Batch differential - full checkpoint every batch_size steps since last full
            steps_since_base = step - self.base_checkpoint_step
            return steps_since_base >= self.batch_size

    def _save_full_checkpoint(self, model, checkpoint_dir, step):
        """Save full checkpoint (update base state and optionally call Trainer._save_checkpoint)"""
        logger.info(f"[BatchDiff] Saving full checkpoint at step {step}")

        os.makedirs(checkpoint_dir, exist_ok=True)

        state_dict = model.state_dict()

        self.base_checkpoint_state = OrderedDict()
        for key, value in state_dict.items():
            self.base_checkpoint_state[key] = value.detach().cpu().clone()

        # Step 3: Update shadow model to match current state
        if self.enable_shadow:
            with self.shadow_lock:
                self.shadow_model = OrderedDict()
                for key, value in self.base_checkpoint_state.items():
                    self.shadow_model[key] = value.clone()
                # Re-tie weights in shadow model for correct update accumulation
                if hasattr(self, '_tied_weight_groups') and self._tied_weight_groups:
                    _tie_state_dict_inplace(self.shadow_model, self._tied_weight_groups)
                self.shadow_step = len(self.update_history)  # Shadow is now caught up

        # Conditionally save to disk via Trainer (only if save_full_model=True)
        if self.save_full_model and self.trainer is not None:
            # Set flag to prevent recursion
            self._saving_full_via_trainer = True

            # Call original Trainer._save_checkpoint (which will skip our on_save hook)
            # This saves model, optimizer, scheduler, trainer_state, etc.
            from transformers import Trainer
            Trainer._save_checkpoint(self.trainer, model, trial=None, metrics=None)

            # Reset flag
            self._saving_full_via_trainer = False

        self.base_checkpoint_path = checkpoint_dir
        self.base_checkpoint_step = step
        self.last_saved_step = step
        self.is_first_save = False

        # Clear update history for batch differential mode
        if self.batch_size >= 1:
            with self.update_lock:
                self.update_history = []
            logger.info(f"[BatchDiff] Cleared update history (batch_size={self.batch_size})")

        mem_mb = self._get_memory_size()
        logger.info(f"[BatchDiff] Full checkpoint cached ({mem_mb:.1f} MB)")

        # Save metadata in JSON format (merged with history format)
        # For full checkpoint, we save an empty updates array
        history = {
            "is_batch_diff": True,
            "is_full_checkpoint": True,
            "base_checkpoint": checkpoint_dir,  # This checkpoint is now the base
            "base_step": step,
            "current_step": step,
            "batch_size": self.batch_size,
            "save_full_model": self.save_full_model,
            "model_dtype": self.model_dtype,
            "trainable_param_names": self._trainable_param_names,
            "pending_grad": self._pending_grad,
            "num_updates": 0,
            "updates": []
        }
        history_path = os.path.join(checkpoint_dir, "zo_replay_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"[BatchDiff] Full checkpoint pending_grad={self._pending_grad:.6e}")

    def _save_diff_checkpoint(self, checkpoint_dir, step):
        """Save differential checkpoint"""
        logger.info(f"[BatchDiff] Saving differential checkpoint at step {step}...")

        os.makedirs(checkpoint_dir, exist_ok=True)

        with self.update_lock:
            if self.batch_size == 1:
                # Pure differential: only save the updates since last checkpoint
                # update['step'] now correctly matches the checkpoint step (1-indexed)
                updates_to_save = [u for u in self.update_history if u['step'] > self.last_saved_step]
            else:
                # Incremental (batch_size=0): save all updates from base
                updates_to_save = self.update_history.copy()

            history = {
                'is_batch_diff': True,
                'is_full_checkpoint': False,
                'base_checkpoint': self.base_checkpoint_path,
                'base_step': self.base_checkpoint_step,
                'current_step': step,
                'batch_size': self.batch_size,
                'save_full_model': self.save_full_model,
                'model_dtype': self.model_dtype,
                'trainable_param_names': self._trainable_param_names,
                'num_updates': len(updates_to_save),
                'tied_weights': getattr(self, '_tied_weight_groups', []),
                'pending_grad': self._pending_grad,
                'updates': updates_to_save
            }

        logger.info(f"[BatchDiff] Checkpoint pending_grad={self._pending_grad:.6e} "
                     f"(will be applied on resume's first step)")

        history_path = os.path.join(checkpoint_dir, "zo_replay_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        history_size = os.path.getsize(history_path) / 1024
        logger.info(f"[BatchDiff] Saved {len(history['updates'])} updates ({history_size:.1f} KB)")

        # Delete large files that may have been created by Trainer
        for fname in ["optimizer.pt", "model.safetensors", "pytorch_model.bin"]:
            fpath = os.path.join(checkpoint_dir, fname)
            if os.path.exists(fpath):
                fsize = os.path.getsize(fpath) / (1024 * 1024)
                os.remove(fpath)
                logger.info(f"[BatchDiff] Deleted {fname} ({fsize:.1f} MB)")

        self.last_saved_step = step

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


def _apply_single_update(state, update, param_names, default_zo_eps=0.0,
                         simulate_perturbation=True):
    """Apply one ZO update (perturbation + parameter update) to a state dict in-place.

    Simulates the full perturbation sequence [+1, -2, +1] * eps * z from zo_forward
    to match fp16 rounding residuals, then applies the actual parameter update.

    Args:
        simulate_perturbation: If True (default), simulate the [+1, -2, +1] perturbation
            loop to reproduce fp16 rounding residuals for bitwise-exact replay. If False,
            skip the perturbation loop (~4x faster) at the cost of ~1e-6 level parameter
            differences. Safe to disable for most use cases.
    """
    seed = update['seed']
    grad = update['grad']
    lr = update['lr']
    wd = update.get('wd', 0.0)
    zo_eps = update.get('zo_eps', default_zo_eps)

    if simulate_perturbation and zo_eps > 0:
        for scaling_factor in [1, -2, 1]:
            torch.manual_seed(seed)
            for name in param_names:
                param = state[name]
                z = torch.normal(mean=0, std=1, size=param.size(), dtype=param.dtype, device=param.device)
                param.data.add_(scaling_factor * z * zo_eps)

    torch.manual_seed(seed)
    for name in param_names:
        param = state[name]
        z = torch.normal(mean=0, std=1, size=param.size(), dtype=param.dtype, device=param.device)
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

    logger.info(f"[Replay] Replaying {len(updates)} updates on device={actual_device}"
                f" (simulate_perturbation={simulate_perturbation})")
    if actual_device == 'cpu' and torch.cuda.is_available():
        logger.warning("[Replay] WARNING: Replaying on CPU but CUDA is available. "
                       "CPU and CUDA RNG produce different z for the same seed. "
                       "Use device='cuda' for exact reconstruction of CUDA-trained models.")

    # Use trainable_param_names if available to match training's iteration order;
    # otherwise fall back to all state dict keys (backward compatible with old checkpoints)
    param_names = trainable_param_names if trainable_param_names is not None else list(state.keys())

    for i, update in enumerate(updates):
        _apply_single_update(state, update, param_names, default_zo_eps=default_zo_eps,
                             simulate_perturbation=simulate_perturbation)

        if i < 3 or i == len(updates) - 1:
            logger.info(f"[Replay] update {i}: step={update.get('step','?')}, seed={update['seed']}, "
                        f"grad={update['grad']:.6e}, lr={update['lr']}, wd={update.get('wd', 0.0)}, "
                        f"zo_eps={update.get('zo_eps', default_zo_eps)}")
        elif i == 3:
            logger.info(f"[Replay] ... ({len(updates) - 4} more updates) ...")

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
                               simulate_perturbation=True):
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
    # Check for full model first
    full_model_path = os.path.join(checkpoint_dir, "pytorch_model_full.bin")
    if os.path.exists(full_model_path):
        return torch.load(full_model_path, map_location='cpu', weights_only=True)

    history_path = os.path.join(checkpoint_dir, "zo_replay_history.json")

    # Check if this is a batch diff checkpoint (has zo_replay_history.json)
    if not os.path.exists(history_path):
        # Not a batch diff checkpoint, try loading as regular checkpoint
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            _restore_tied_weights(state_dict, checkpoint_dir)
            return state_dict
        safe_path = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.exists(safe_path):
            from safetensors.torch import load_file
            state_dict = load_file(safe_path)
            _restore_tied_weights(state_dict, checkpoint_dir)
            return state_dict
        return None

    # Load metadata from JSON
    with open(history_path, 'r') as f:
        history = json.load(f)

    # If this is a full checkpoint, load directly
    if history.get('is_full_checkpoint', False):
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

        # If save_full_model=False, there's no physical model file
        # This is expected for full checkpoints that only update base state
        logger.warning(f"Full checkpoint at {checkpoint_dir} has no model file (save_full_model=False)")
        return None

    # Load differential checkpoint - history already loaded above

    # Extract model dtype for replay consistency
    model_dtype_str = history.get('model_dtype', None)
    model_dtype = _DTYPE_MAP.get(model_dtype_str, None)
    if model_dtype is not None:
        logger.info(f"[load_batch_diff_checkpoint] Model dtype from metadata: {model_dtype}")

    base_checkpoint_ref = history.get("base_checkpoint")

    # If base_checkpoint is "__initial__", user must provide the pretrained model path
    if base_checkpoint_ref == "__initial__":
        if base_checkpoint_dir is None:
            raise ValueError(
                "This checkpoint uses differential mode from initial model. "
                "You must provide base_checkpoint_dir (path to original pretrained model) to load it. "
                "Example: load_batch_diff_checkpoint(checkpoint_dir, base_checkpoint_dir='Qwen/Qwen2.5-1.5B')"
            )
        base_dir = base_checkpoint_dir
    else:
        base_dir = base_checkpoint_dir or base_checkpoint_ref

    base_path = os.path.join(base_dir, "model.safetensors")

    if os.path.exists(base_path):
        from safetensors.torch import load_file
        base_state = load_file(base_path)
    else:
        base_path = os.path.join(base_dir, "pytorch_model.bin")
        if os.path.exists(base_path):
            base_state = torch.load(base_path, map_location='cpu', weights_only=True)
        else:
            # Try loading from HuggingFace model hub
            try:
                from transformers import AutoModelForCausalLM
                _dtype_kwargs = {'torch_dtype': model_dtype} if model_dtype is not None else {}
                logger.info(f"Trying to load base model from HuggingFace: {base_dir} (dtype={model_dtype})")
                base_model = AutoModelForCausalLM.from_pretrained(base_dir, **_dtype_kwargs)
                base_state = base_model.state_dict()
                del base_model
            except Exception as e:
                raise FileNotFoundError(
                    f"Cannot find base checkpoint at {base_dir}. "
                    f"Tried local files and HuggingFace hub. Error: {e}"
                )

    # Get tied weight groups and trainable param names from checkpoint metadata
    tied_groups = history.get('tied_weights', [])
    trainable_param_names = history.get('trainable_param_names', None)
    default_zo_eps_val = history.get('zo_eps', 0.0)

    updates = history.get('updates', [])
    logger.info(f"Replaying {len(updates)} ZO updates on {device}..."
                f" (trainable_params: {len(trainable_param_names) if trainable_param_names else 'all'})")

    reconstructed = OrderedDict()
    for key, value in base_state.items():
        reconstructed[key] = value.clone()
    del base_state

    # Tie weights before replay so updates accumulate correctly (like during training)
    if tied_groups:
        _tie_state_dict_inplace(reconstructed, tied_groups)

    # Use the shared replay function with device support
    _replay_updates_on_state(reconstructed, updates, device=device,
                             trainable_param_names=trainable_param_names,
                             default_zo_eps=default_zo_eps_val,
                             simulate_perturbation=simulate_perturbation)

    # Fix tied weights that may have diverged during replay
    _tie_state_dict_inplace(reconstructed, tied_groups)

    return reconstructed


def _load_base_state(base_checkpoint_ref, pretrained_model_name, tied_groups, model_dtype):
    """Load the base model state dict for replay.

    Returns:
        (base_state, tied_groups) where tied_groups may be updated if detected from model.
    """
    if base_checkpoint_ref == "__initial__":
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

    Returns:
        Reconstructed state_dict at the checkpoint (on CPU)

    Replay strategies based on batch_size:
    - batch_size=0 (Incremental): Each checkpoint contains ALL updates from base.
                                  Just load the latest checkpoint and replay its updates.
    - batch_size=1 (Pure Differential): Each checkpoint only contains updates since last checkpoint.
                                        Must traverse ALL checkpoints from base to target.
    - batch_size>=2 (Batch Differential): Similar to incremental within each batch.
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

    # Load the target checkpoint's history to determine batch_size mode
    # Check optimizer.pt first (log-based mode), then zo_replay_history.json (differential mode)
    optimizer_path = os.path.join(ckpt_dir, "optimizer.pt")
    history_path = os.path.join(ckpt_dir, "zo_replay_history.json")

    if os.path.exists(optimizer_path):
        optimizer_state = torch.load(optimizer_path, map_location='cpu', weights_only=False)
        if isinstance(optimizer_state, dict) and 'zo_update_history' in optimizer_state:
            # Log-based checkpoint: replay from optimizer.pt
            logger.info(f"[Resume] Found log-based checkpoint (optimizer.pt with zo_update_history)")
            target_history = {
                'batch_size': optimizer_state.get('batch_size', 0),
                'base_checkpoint': optimizer_state.get('base_checkpoint', '__initial__'),
                'updates': optimizer_state['zo_update_history'],
                'tied_weights': optimizer_state.get('tied_weights', []),
                'model_dtype': optimizer_state.get('model_dtype', None),
                'zo_eps': optimizer_state.get('zo_eps', 0.0),
            }
        else:
            optimizer_state = None  # Regular optimizer.pt, not log-based

    if 'target_history' not in locals():
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                target_history = json.load(f)
        else:
            # Not a batch diff checkpoint, try loading directly
            logger.info(f"[Resume] No optimizer.pt or zo_replay_history.json found, loading as regular checkpoint")
            return load_batch_diff_checkpoint(ckpt_dir, base_checkpoint_dir=pretrained_model_name)

    batch_size = target_history.get('batch_size', 0)
    base_checkpoint_ref = target_history.get('base_checkpoint')
    tied_groups = target_history.get('tied_weights', [])
    trainable_param_names = target_history.get('trainable_param_names', None)
    default_zo_eps = target_history.get('zo_eps', 0.0)
    model_dtype_str = target_history.get('model_dtype', None)

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

    # For batch_size>=2 full checkpoints, load directly
    if batch_size >= 2 and target_history.get('is_full_checkpoint', False):
        logger.info(f"[Resume] Target is a full checkpoint, loading directly")
        return load_batch_diff_checkpoint(ckpt_dir, base_checkpoint_dir=pretrained_model_name)

    # Load base model (common to all modes)
    base_state, tied_groups = _load_base_state(
        base_checkpoint_ref, pretrained_model_name, tied_groups, model_dtype
    )

    # Copy base state and prepare for replay
    reconstructed = OrderedDict()
    for key, value in base_state.items():
        reconstructed[key] = value.clone()
    del base_state

    if tied_groups:
        _tie_state_dict_inplace(reconstructed, tied_groups)

    # ========== PURE DIFFERENTIAL MODE (batch_size=1) ==========
    # Each checkpoint only has updates since the PREVIOUS checkpoint
    # Must traverse all checkpoints from base to target
    if batch_size == 1:
        logger.info(f"[Resume] Pure differential mode: traversing all checkpoints")

        # Find base step
        if base_checkpoint_ref == "__initial__":
            base_step = 0
        else:
            match = re.search(r'checkpoint-(\d+)', base_checkpoint_ref)
            base_step = int(match.group(1)) if match else 0

        # Find all checkpoints between base and target
        diff_output_dir = os.path.dirname(ckpt_dir)
        checkpoint_pattern = os.path.join(diff_output_dir, "checkpoint-*")
        all_checkpoints = glob.glob(checkpoint_pattern)
        logger.info(f"[Resume] Searching for checkpoints in: {diff_output_dir}")

        checkpoint_steps = []
        for ckpt in all_checkpoints:
            match = re.search(r'checkpoint-(\d+)', ckpt)
            if match:
                step = int(match.group(1))
                if base_step < step <= target_step:
                    checkpoint_steps.append((step, ckpt))

        checkpoint_steps.sort(key=lambda x: x[0])
        logger.info(f"[Resume] Found {len(checkpoint_steps)} checkpoints to traverse (steps {base_step+1} to {target_step})")

        # Move to GPU once before traversing all checkpoints
        if device == 'cuda' and torch.cuda.is_available():
            logger.info(f"[Resume] Moving state to GPU for replay...")
            for key in reconstructed:
                reconstructed[key] = reconstructed[key].cuda()
            torch.cuda.synchronize()

        # Traverse and replay each checkpoint's updates (already on GPU)
        t_replay_start = time.time()
        total_updates = 0

        for step, ckpt in checkpoint_steps:
            hist_path = os.path.join(ckpt, "zo_replay_history.json")
            if not os.path.exists(hist_path):
                logger.warning(f"[Resume] Step {step}: No history file found, skipping")
                continue

            with open(hist_path, 'r') as f:
                hist = json.load(f)

            if hist.get('is_full_checkpoint', False):
                logger.info(f"[Resume] Step {step}: Found full checkpoint, loading as new base")
                new_base = load_batch_diff_checkpoint(ckpt)
                if new_base is not None:
                    reconstructed = OrderedDict()
                    for key, value in new_base.items():
                        if device == 'cuda' and torch.cuda.is_available():
                            reconstructed[key] = value.cuda()
                        else:
                            reconstructed[key] = value.clone()
                    if tied_groups:
                        _tie_state_dict_inplace(reconstructed, tied_groups)
                continue

            sub_tpn = hist.get('trainable_param_names', trainable_param_names)
            updates = hist.get('updates', [])
            if updates:
                logger.info(f"[Resume] Step {step}: Replaying {len(updates)} updates")
                _replay_updates_on_state(reconstructed, updates, device=device,
                                         move_to_device=False,
                                         trainable_param_names=sub_tpn,
                                         default_zo_eps=default_zo_eps,
                                         simulate_perturbation=simulate_perturbation)
                total_updates += len(updates)

        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_replay = time.time() - t_replay_start
        logger.info(f"[Resume Replay] {total_updates} updates replayed in {t_replay:.3f}s (device={device})")

    # ========== SINGLE CHECKPOINT REPLAY (batch_size=0 or >=2) ==========
    else:
        updates = target_history.get('updates', [])
        pending_grad = target_history.get('pending_grad', None)
        logger.info(f"[Resume] Replaying {len(updates)} updates (default_zo_eps={default_zo_eps})")
        if pending_grad is not None:
            logger.info(f"[Resume] pending_grad={pending_grad} (will be restored to opt.projected_grad)")

        # Move to GPU before timing so replay measures pure computation
        if device == 'cuda' and torch.cuda.is_available():
            for key in reconstructed:
                if reconstructed[key].device.type != 'cuda':
                    reconstructed[key] = reconstructed[key].cuda()
            torch.cuda.synchronize()

        t_replay_start = time.time()
        _replay_updates_on_state(reconstructed, updates, device=device,
                                 move_to_device=False,
                                 trainable_param_names=trainable_param_names,
                                 default_zo_eps=default_zo_eps,
                                 simulate_perturbation=simulate_perturbation)
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

    # Fix tied weights that may have diverged during replay
    _tie_state_dict_inplace(reconstructed, tied_groups)

    return reconstructed


# Backward compatibility
load_zo_replay_checkpoint = load_batch_diff_checkpoint
load_incremental_checkpoint = load_batch_diff_checkpoint
