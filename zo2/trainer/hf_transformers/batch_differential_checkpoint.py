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
import queue
import time
import torch
import torch.nn as nn
from transformers import TrainerCallback
from collections import OrderedDict
import logging
import json
import psutil
import signal
import glob

logger = logging.getLogger(__name__)


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
        self.last_persist_step = 0

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

        # Timing statistics
        self.timing_stats = {
            'checkpoint_saves': [],
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

        # Cache initial model
        if self.base_checkpoint_state is None and model is not None:
            self._cache_initial_model(model)

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
            return "Incremental (accumulate all)"
        elif self.batch_size == 1:
            return "Pure Differential (current step only)"
        else:
            return f"Batch Differential (every {self.batch_size} steps)"

    def _cache_initial_model(self, model):
        """Cache initial model to CPU memory (no disk save for pure differential mode)"""
        logger.info("[BatchDiff] Caching initial model to CPU memory...")
        t_start = time.time()

        state_dict = model.state_dict()

        self.base_checkpoint_state = OrderedDict()
        for key, value in state_dict.items():
            self.base_checkpoint_state[key] = value.detach().cpu().clone()

        # Initialize shadow model
        if self.enable_shadow:
            self.shadow_model = OrderedDict()
            for key, value in self.base_checkpoint_state.items():
                self.shadow_model[key] = value.clone()
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
        """Apply one update to shadow model"""
        seed = update['seed']
        grad = update['grad']
        lr = update['lr']
        wd = update.get('wd', 0.0)

        torch.manual_seed(seed)

        with self.shadow_lock:
            for name, param in self.shadow_model.items():
                z = torch.normal(mean=0, std=1, size=param.size(), dtype=param.dtype)
                if 'bias' not in name and 'layer_norm' not in name and 'layernorm' not in name and 'ln' not in name:
                    param.sub_(lr * (grad * z + wd * param))
                else:
                    param.sub_(lr * grad * z)

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
            grad = getattr(opt, 'projected_grad', 0)
            lr = getattr(opt, 'lr', 0)
            wd = getattr(opt, 'weight_decay', 0)

            with self.update_lock:
                # Use current_step + 1 since this hook is called BEFORE global_step is incremented
                # This ensures update step matches the checkpoint step it belongs to
                actual_step = self.current_step + 1
                update = {
                    'step': actual_step,
                    'seed': int(seed) if seed is not None else 0,
                    'grad': float(grad) if not isinstance(grad, float) else grad,
                    'lr': float(lr),
                    'wd': float(wd)
                }
                self.update_history.append(update)

        return model, inputs, loss

    def on_step_end(self, args, state, control, **kwargs):
        """Called at end of each step"""
        self.current_step = state.global_step

        # Persist history periodically
        if self.current_step - self.last_persist_step >= 1:
            self._persist_history()
            self.last_persist_step = self.current_step

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

        # Replay all updates
        for i, update in enumerate(updates):
            seed = update['seed']
            grad = update['grad']
            lr = update['lr']
            wd = update.get('wd', 0.0)

            torch.manual_seed(seed)
            for name, param in reconstructed.items():
                z = torch.normal(mean=0, std=1, size=param.size(), dtype=param.dtype)
                if 'bias' not in name and 'layer_norm' not in name and 'layernorm' not in name and 'ln' not in name:
                    param.sub_(lr * (grad * z + wd * param))
                else:
                    param.sub_(lr * grad * z)

            if (i + 1) % 100 == 0:
                logger.info(f"[Recovery] Replayed {i + 1}/{num_updates} updates...")

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

    def _persist_history(self):
        """Persist update history to disk"""
        if not self.output_dir:
            return

        with self.update_lock:
            if not self.update_history:
                return
            history_copy = self.update_history.copy()

        os.makedirs(self.output_dir, exist_ok=True)
        history_path = os.path.join(self.output_dir, "zo_replay_history_latest.json")

        if self.enable_shadow:
            with self.shadow_lock:
                shadow_step = self.shadow_step
        else:
            shadow_step = -1

        history = {
            'base_step': self.base_checkpoint_step,
            'current_step': self.current_step,
            'shadow_step': shadow_step,
            'enable_shadow': self.enable_shadow,
            'batch_size': self.batch_size,
            'num_updates': len(history_copy),
            'updates': history_copy
        }

        tmp_path = history_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(history, f)
        os.replace(tmp_path, history_path)

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

        t_start = time.time()

        # Determine if we need to save a full checkpoint
        should_save_full = self._should_save_full_checkpoint(state.global_step)

        if should_save_full:
            self._save_full_checkpoint(model, checkpoint_dir, state.global_step)
            checkpoint_type = "Full"
        else:
            self._save_diff_checkpoint(checkpoint_dir, state.global_step)
            checkpoint_type = "Differential"

        t_elapsed = time.time() - t_start
        logger.info(f"[Checkpoint Timing] {checkpoint_type} checkpoint save took {t_elapsed:.4f}s at step {state.global_step}")

        self.timing_stats['checkpoint_saves'].append({
            'type': checkpoint_type.lower(),
            'step': state.global_step,
            'time': t_elapsed
        })

    def _should_save_full_checkpoint(self, step: int) -> bool:
        """Determine if we should save a full checkpoint"""
        if self.batch_size == -1:
            # Disabled mode - always full (but this callback shouldn't be used)
            return True
        elif self.batch_size == 0:
            # Incremental mode - NEVER save full (initial model already cached)
            return False
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

        t_start = time.time()
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Step 1: model.state_dict()
        t_state_dict_start = time.time()
        state_dict = model.state_dict()
        t_state_dict = time.time() - t_state_dict_start
        logger.info(f"[Checkpoint Timing] model.state_dict() took {t_state_dict:.3f}s")

        # Step 2: Copy to CPU memory (for later replay)
        t_copy = 0.0
        t_copy_start = time.time()
        self.base_checkpoint_state = OrderedDict()
        for key, value in state_dict.items():
            self.base_checkpoint_state[key] = value.detach().cpu().clone()
        t_copy = time.time() - t_copy_start
        logger.info(f"[Checkpoint Timing] Copy to CPU memory took {t_copy:.3f}s")

        # Step 3: Update shadow model to match current state
        if self.enable_shadow:
            with self.shadow_lock:
                self.shadow_model = OrderedDict()
                for key, value in self.base_checkpoint_state.items():
                    self.shadow_model[key] = value.clone()
                self.shadow_step = len(self.update_history)  # Shadow is now caught up

        # Step 4: Conditionally save to disk via Trainer (only if save_full_model=True)
        t_disk = 0.0
        if self.save_full_model and self.trainer is not None:
            logger.info(f"[BatchDiff] Calling Trainer._save_checkpoint to save full model...")
            t_disk_start = time.time()

            # Set flag to prevent recursion
            self._saving_full_via_trainer = True

            # Call original Trainer._save_checkpoint (which will skip our on_save hook)
            # This saves model, optimizer, scheduler, trainer_state, etc.
            from transformers import Trainer
            Trainer._save_checkpoint(self.trainer, model, trial=None, metrics=None)

            # Reset flag
            self._saving_full_via_trainer = False

            t_disk = time.time() - t_disk_start
            logger.info(f"[Checkpoint Timing] Trainer._save_checkpoint took {t_disk:.3f}s")
        else:
            if not self.save_full_model:
                logger.info(f"[Checkpoint Timing] Skipped writing full model to disk (save_full_model=False)")
            else:
                logger.warning(f"[Checkpoint Timing] Trainer not available, cannot save full model")

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
        t_total = time.time() - t_start
        logger.info(f"[BatchDiff] Full checkpoint cached ({mem_mb:.1f} MB)")
        logger.info(f"[Checkpoint Timing] Full checkpoint total: state_dict={t_state_dict:.3f}s, "
                   f"copy_to_cpu={t_copy:.3f}s, trainer_save={t_disk:.3f}s, total={t_total:.3f}s")

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
            "num_updates": 0,
            "updates": []
        }
        history_path = os.path.join(checkpoint_dir, "zo_replay_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    def _save_diff_checkpoint(self, checkpoint_dir, step):
        """Save differential checkpoint"""
        logger.info(f"[BatchDiff] Saving differential checkpoint at step {step}...")

        t_start = time.time()

        t_mkdir_start = time.time()
        os.makedirs(checkpoint_dir, exist_ok=True)
        t_mkdir = time.time() - t_mkdir_start

        t_copy_start = time.time()
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
                'num_updates': len(updates_to_save),
                'updates': updates_to_save
            }
        t_copy = time.time() - t_copy_start

        t_json_start = time.time()
        history_path = os.path.join(checkpoint_dir, "zo_replay_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        t_json = time.time() - t_json_start

        history_size = os.path.getsize(history_path) / 1024
        logger.info(f"[BatchDiff] Saved {len(history['updates'])} updates ({history_size:.1f} KB)")
        logger.info(f"[Checkpoint Timing] History copy took {t_copy:.3f}s, JSON write took {t_json:.3f}s")

        # Delete large files that may have been created by Trainer
        t_delete_start = time.time()
        for fname in ["optimizer.pt", "model.safetensors", "pytorch_model.bin"]:
            fpath = os.path.join(checkpoint_dir, fname)
            if os.path.exists(fpath):
                fsize = os.path.getsize(fpath) / (1024 * 1024)
                os.remove(fpath)
                logger.info(f"[BatchDiff] Deleted {fname} ({fsize:.1f} MB)")
        t_delete = time.time() - t_delete_start

        self.last_saved_step = step

        t_total = time.time() - t_start
        logger.info(f"[Checkpoint Timing] Differential checkpoint internal: mkdir={t_mkdir:.3f}s, copy={t_copy:.3f}s, "
                   f"json={t_json:.3f}s, delete={t_delete:.3f}s, total={t_total:.3f}s")

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

        self._persist_history()

        status = self.get_recovery_status()
        logger.info(f"[BatchDiff] Final status: {status}")

        # Print timing statistics
        if self.timing_stats['checkpoint_saves']:
            avg_full = sum(s['time'] for s in self.timing_stats['checkpoint_saves'] if s['type'] == 'full') / max(1, len([s for s in self.timing_stats['checkpoint_saves'] if s['type'] == 'full']))
            avg_diff = sum(s['time'] for s in self.timing_stats['checkpoint_saves'] if s['type'] == 'differential') / max(1, len([s for s in self.timing_stats['checkpoint_saves'] if s['type'] == 'differential']))
            logger.info(f"[Timing Stats] Avg full checkpoint: {avg_full:.3f}s, Avg differential: {avg_diff:.3f}s")

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


def _replay_updates_on_state(state: OrderedDict, updates: list, device: str = 'cpu', move_to_device: bool = True) -> OrderedDict:
    """
    Replay ZO updates on a state dict.

    Args:
        state: The state dict to modify (will be modified in-place)
        updates: List of update dicts with keys: seed, grad, lr, wd
        device: Device to perform computation on ('cpu' or 'cuda')
        move_to_device: If True and device='cuda', move state to GPU before replay.
                        If False, assume state is already on the correct device.

    Returns:
        The modified state dict (stays on the device where computation was done)
    """
    if not updates:
        return state

    # Move to target device if needed
    if move_to_device and device == 'cuda' and torch.cuda.is_available():
        for key in state:
            if state[key].device.type != 'cuda':
                state[key] = state[key].cuda()

    for update in updates:
        seed = update['seed']
        grad = update['grad']
        lr = update['lr']
        wd = update.get('wd', 0.0)

        torch.manual_seed(seed)
        for name, param in state.items():
            z = torch.normal(mean=0, std=1, size=param.size(), dtype=param.dtype, device=param.device)
            if 'bias' not in name and 'layer_norm' not in name and 'layernorm' not in name and 'ln' not in name:
                param.sub_(lr * (grad * z + wd * param))
            else:
                param.sub_(lr * grad * z)

    # Don't move back to CPU - keep on GPU for training
    return state


def load_batch_diff_checkpoint(checkpoint_dir, base_checkpoint_dir=None, device='cpu'):
    """
    Load batch differential checkpoint.

    Args:
        checkpoint_dir: Path to the checkpoint directory
        base_checkpoint_dir: Optional. Path to base model (required for "__initial__" mode)
        device: Device for replay computation ('cpu' or 'cuda'). Default 'cpu'.

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
            return torch.load(model_path, map_location='cpu', weights_only=True)
        safe_path = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.exists(safe_path):
            from safetensors.torch import load_file
            return load_file(safe_path)
        return None

    # Load metadata from JSON
    with open(history_path, 'r') as f:
        history = json.load(f)

    # If this is a full checkpoint, load directly
    if history.get('is_full_checkpoint', False):
        safe_path = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.exists(safe_path):
            from safetensors.torch import load_file
            return load_file(safe_path)
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            return torch.load(model_path, map_location='cpu', weights_only=True)

        # If save_full_model=False, there's no physical model file
        # This is expected for full checkpoints that only update base state
        logger.warning(f"Full checkpoint at {checkpoint_dir} has no model file (save_full_model=False)")
        return None

    # Load differential checkpoint - history already loaded above

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
                logger.info(f"Trying to load base model from HuggingFace: {base_dir}")
                base_model = AutoModelForCausalLM.from_pretrained(base_dir)
                base_state = base_model.state_dict()
                del base_model
            except Exception as e:
                raise FileNotFoundError(
                    f"Cannot find base checkpoint at {base_dir}. "
                    f"Tried local files and HuggingFace hub. Error: {e}"
                )

    updates = history.get('updates', [])
    logger.info(f"Replaying {len(updates)} ZO updates on {device}...")

    reconstructed = OrderedDict()
    for key, value in base_state.items():
        reconstructed[key] = value.clone()
    del base_state

    # Use the shared replay function with device support
    _replay_updates_on_state(reconstructed, updates, device=device)

    return reconstructed


def resume_from_batch_diff(
    checkpoint_path: str,
    output_dir: str = None,
    pretrained_model_name: str = None,
    device: str = 'cpu'
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

    Returns:
        Reconstructed state_dict at the checkpoint (on CPU)

    Replay strategies based on batch_size:
    - batch_size=0 (Incremental): Each checkpoint contains ALL updates from base.
                                  Just load the latest checkpoint and replay its updates.
    - batch_size=1 (Pure Differential): Each checkpoint only contains updates since last checkpoint.
                                        Must traverse ALL checkpoints from base to target.
    - batch_size>=2 (Batch Differential): Similar to incremental within each batch.
    """
    t_start = time.time()

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
    history_path = os.path.join(ckpt_dir, "zo_replay_history.json")
    if not os.path.exists(history_path):
        # Not a batch diff checkpoint, try loading directly
        logger.info(f"[Resume] No zo_replay_history.json found, loading as regular checkpoint")
        return load_batch_diff_checkpoint(ckpt_dir, base_checkpoint_dir=pretrained_model_name)

    with open(history_path, 'r') as f:
        target_history = json.load(f)

    batch_size = target_history.get('batch_size', 0)
    base_checkpoint_ref = target_history.get('base_checkpoint')

    logger.info(f"[Resume] Checkpoint mode: batch_size={batch_size}, base_checkpoint={base_checkpoint_ref}")

    # ========== INCREMENTAL MODE (batch_size=0) ==========
    # Each checkpoint has ALL updates from base, just load this one checkpoint
    if batch_size == 0:
        logger.info(f"[Resume] Incremental mode: loading single checkpoint with all updates")

        # Load base model
        t_load_base_start = time.time()
        if base_checkpoint_ref == "__initial__":
            if pretrained_model_name is None:
                raise ValueError(
                    "This checkpoint uses differential mode from initial model. "
                    "You must provide pretrained_model_name to load it."
                )
            # Load from HuggingFace
            try:
                from transformers import AutoModelForCausalLM
                logger.info(f"[Resume] Loading base model from HuggingFace: {pretrained_model_name}")
                base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
                base_state = base_model.state_dict()
                del base_model
            except Exception as e:
                raise FileNotFoundError(f"Cannot load pretrained model {pretrained_model_name}: {e}")
        else:
            base_state = load_batch_diff_checkpoint(base_checkpoint_ref, base_checkpoint_dir=pretrained_model_name)
            if base_state is None:
                raise FileNotFoundError(f"Cannot load base checkpoint from {base_checkpoint_ref}")

        t_load_base = time.time() - t_load_base_start
        logger.info(f"[Resume] Loaded base model in {t_load_base:.3f}s")

        # Copy and replay
        reconstructed = OrderedDict()
        for key, value in base_state.items():
            reconstructed[key] = value.clone()
        del base_state

        updates = target_history.get('updates', [])
        logger.info(f"[Resume] Replaying {len(updates)} updates from single checkpoint")

        t_replay_start = time.time()
        _replay_updates_on_state(reconstructed, updates, device=device)
        t_replay = time.time() - t_replay_start

        t_total = time.time() - t_start
        logger.info(f"[Resume] Completed! Recovered to step {target_step}")
        logger.info(f"[Resume Timing] Load base: {t_load_base:.4f}s, Replay {len(updates)} updates: {t_replay:.4f}s, "
                   f"Total: {t_total:.4f}s")
        return reconstructed

    # ========== PURE DIFFERENTIAL MODE (batch_size=1) ==========
    # Each checkpoint only has updates since the PREVIOUS checkpoint
    # Must traverse all checkpoints from base to target
    elif batch_size == 1:
        logger.info(f"[Resume] Pure differential mode: traversing all checkpoints")

        # Find base step
        if base_checkpoint_ref == "__initial__":
            base_step = 0
        else:
            match = re.search(r'checkpoint-(\d+)', base_checkpoint_ref)
            base_step = int(match.group(1)) if match else 0

        # Load base model
        t_load_base_start = time.time()
        if base_checkpoint_ref == "__initial__":
            if pretrained_model_name is None:
                raise ValueError(
                    "This checkpoint uses differential mode from initial model. "
                    "You must provide pretrained_model_name to load it."
                )
            try:
                from transformers import AutoModelForCausalLM
                logger.info(f"[Resume] Loading base model from HuggingFace: {pretrained_model_name}")
                base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
                base_state = base_model.state_dict()
                del base_model
            except Exception as e:
                raise FileNotFoundError(f"Cannot load pretrained model {pretrained_model_name}: {e}")
        else:
            base_state = load_batch_diff_checkpoint(base_checkpoint_ref, base_checkpoint_dir=pretrained_model_name)
            if base_state is None:
                raise FileNotFoundError(f"Cannot load base checkpoint from {base_checkpoint_ref}")

        t_load_base = time.time() - t_load_base_start
        logger.info(f"[Resume] Loaded base model in {t_load_base:.3f}s")

        # Copy base state
        reconstructed = OrderedDict()
        for key, value in base_state.items():
            reconstructed[key] = value.clone()
        del base_state

        # Find all checkpoints between base and target (inclusive of target)
        # For pure differential mode, always use parent of ckpt_dir to find sibling checkpoints
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
        t_replay_start = time.time()
        if device == 'cuda' and torch.cuda.is_available():
            logger.info(f"[Resume] Moving state to GPU for replay...")
            for key in reconstructed:
                reconstructed[key] = reconstructed[key].cuda()

        # Traverse and replay each checkpoint's updates (already on GPU)
        total_updates = 0

        for step, ckpt in checkpoint_steps:
            hist_path = os.path.join(ckpt, "zo_replay_history.json")
            if not os.path.exists(hist_path):
                logger.warning(f"[Resume] Step {step}: No history file found, skipping")
                continue

            with open(hist_path, 'r') as f:
                hist = json.load(f)

            # Check if this is a full checkpoint (shouldn't happen in pure diff mode, but handle it)
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
                continue

            updates = hist.get('updates', [])
            if updates:
                logger.info(f"[Resume] Step {step}: Replaying {len(updates)} updates")
                # Don't move to device again, already on GPU
                _replay_updates_on_state(reconstructed, updates, device=device, move_to_device=False)
                total_updates += len(updates)

        t_replay = time.time() - t_replay_start
        t_total = time.time() - t_start

        logger.info(f"[Resume] Completed! Recovered to step {target_step}")
        logger.info(f"[Resume Timing] Load base: {t_load_base:.3f}s, Replay {total_updates} updates: {t_replay:.3f}s, "
                   f"Total: {t_total:.3f}s")
        return reconstructed

    # ========== BATCH DIFFERENTIAL MODE (batch_size>=2) ==========
    # Similar to incremental, but may have intermediate full checkpoints
    else:
        logger.info(f"[Resume] Batch differential mode (batch_size={batch_size})")

        # If this is a full checkpoint, load directly
        if target_history.get('is_full_checkpoint', False):
            logger.info(f"[Resume] Target is a full checkpoint, loading directly")
            return load_batch_diff_checkpoint(ckpt_dir, base_checkpoint_dir=pretrained_model_name)

        # Otherwise load base and replay this checkpoint's updates
        t_load_base_start = time.time()
        if base_checkpoint_ref == "__initial__":
            if pretrained_model_name is None:
                raise ValueError(
                    "This checkpoint uses differential mode from initial model. "
                    "You must provide pretrained_model_name to load it."
                )
            try:
                from transformers import AutoModelForCausalLM
                logger.info(f"[Resume] Loading base model from HuggingFace: {pretrained_model_name}")
                base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
                base_state = base_model.state_dict()
                del base_model
            except Exception as e:
                raise FileNotFoundError(f"Cannot load pretrained model {pretrained_model_name}: {e}")
        else:
            base_state = load_batch_diff_checkpoint(base_checkpoint_ref, base_checkpoint_dir=pretrained_model_name)
            if base_state is None:
                raise FileNotFoundError(f"Cannot load base checkpoint from {base_checkpoint_ref}")

        t_load_base = time.time() - t_load_base_start
        logger.info(f"[Resume] Loaded base model in {t_load_base:.3f}s")

        # Copy and replay
        reconstructed = OrderedDict()
        for key, value in base_state.items():
            reconstructed[key] = value.clone()
        del base_state

        updates = target_history.get('updates', [])
        logger.info(f"[Resume] Replaying {len(updates)} updates")

        t_replay_start = time.time()
        _replay_updates_on_state(reconstructed, updates, device=device)
        t_replay = time.time() - t_replay_start

        t_total = time.time() - t_start
        logger.info(f"[Resume] Completed! Recovered to step {target_step}")
        logger.info(f"[Resume Timing] Load base: {t_load_base:.3f}s, Replay {len(updates)} updates: {t_replay:.3f}s, "
                   f"Total: {t_total:.3f}s")
        return reconstructed


# Backward compatibility
load_zo_replay_checkpoint = load_batch_diff_checkpoint
load_incremental_checkpoint = load_batch_diff_checkpoint
