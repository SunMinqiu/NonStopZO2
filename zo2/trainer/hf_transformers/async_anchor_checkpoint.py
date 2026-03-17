"""
Async Anchor Checkpoint for ZO Training.

Two-phase async pipeline for full model checkpoint saving:
  Phase 1 (GPU→CPU): Dedicated CUDA stream async-copies model params to pinned CPU buffer
  Phase 2 (CPU→disk): Background daemon thread clones buffer, forks subprocess to write

At most one checkpoint persist is in progress at any time. When a new checkpoint
triggers, the training thread waits for the previous persist to finish before
proceeding. This guarantees that failure rolls back at most one checkpoint.

Usage:
    anchor = AsyncAnchorCheckpointer(model, checkpoint_dir)
    # In training loop, called by _save_checkpoint when is_full_step:
    anchor.try_save_full_checkpoint(step, model, output_dir)
    # At end of training:
    anchor.shutdown()
"""

import os
import time
import queue
import threading
import logging

import torch
from collections import OrderedDict

logger = logging.getLogger(__name__)


def _subprocess_write_checkpoint(state_dict, save_path):
    """Write checkpoint in forked subprocess (independent GIL).

    Called only in the child process after os.fork(). Must NOT touch any CUDA
    resources. Uses os._exit() to avoid running the parent's atexit handlers
    or CUDA cleanup.
    """
    try:
        from safetensors.torch import save_file
        save_file(state_dict, save_path)
    except Exception:
        import torch
        fallback_path = save_path.replace('.safetensors', '.bin')
        torch.save(state_dict, fallback_path)
    os._exit(0)


class _PersistJob:
    """A pending CPU→disk persist job."""
    __slots__ = ['step', 'output_dir', 'copy_done_event']

    def __init__(self, step, output_dir, copy_done_event):
        self.step = step
        self.output_dir = output_dir
        self.copy_done_event = copy_done_event


class AsyncAnchorCheckpointer:
    """Async full-model checkpoint writer for ZO training.

    Pre-allocates a single CPU pinned buffer matching the model size.
    At anchor steps, async-copies GPU params to the pinned buffer via a
    dedicated CUDA stream, then a background thread persists the buffer
    to disk as model.safetensors.

    Args:
        model: nn.Module (used to determine param shapes/dtypes)
        checkpoint_dir: root output directory
        tied_groups: list of tied weight groups from _detect_tied_weights().
            Secondary keys are excluded from the pinned buffer and saved checkpoint,
            matching HuggingFace's save_pretrained() convention.
    """

    def __init__(self, model, checkpoint_dir, tied_groups=None):
        self._checkpoint_dir = checkpoint_dir

        # Compute excluded keys from tied weight groups (keep first, exclude rest)
        self._excluded_keys = set()
        if tied_groups:
            for group in tied_groups:
                for name in group[1:]:
                    self._excluded_keys.add(name)
            logger.info(f"[AsyncAnchor] Excluding tied keys from checkpoint: {self._excluded_keys}")

        # CUDA stream dedicated to checkpoint copies
        self._ckpt_stream = torch.cuda.Stream()

        # Pre-allocate single pinned buffer (excluding tied weight duplicates)
        self._pinned_buffer = OrderedDict()
        self._copy_done_event = torch.cuda.Event()

        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            if name in self._excluded_keys:
                continue
            self._pinned_buffer[name] = torch.empty(
                tensor.shape, dtype=tensor.dtype,
                device='cpu', pin_memory=True,
            )

        # Buffer availability (protected by Condition for wait/notify)
        self._buffer_free = True
        self._lock = threading.Lock()
        self._buffer_cond = threading.Condition(self._lock)

        # Persist completion: at most 1 child process writing at a time.
        # Training thread waits on this before starting a new checkpoint.
        self._persist_done = threading.Event()
        self._persist_done.set()  # Initially: no persist in progress

        # Completion tracking (protected by _lock)
        self._latest_completed_step = -1
        self._latest_completed_path = None

        # Background persist thread
        self._persist_queue = queue.Queue()
        self._persist_thread = threading.Thread(
            target=self._persist_worker, daemon=True, name="async-anchor-persist"
        )
        self._persist_thread.start()

        # Stats
        self._anchors_saved = 0
        self._copy_times = []
        self._persist_times = []

        total_bytes = sum(
            t.numel() * t.element_size() for t in self._pinned_buffer.values()
        )
        logger.info(
            f"[AsyncAnchor] Init: {total_bytes / 1e9:.2f} GB pinned buffer"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def try_save_full_checkpoint(self, step, model, output_dir) -> bool:
        """Start async full checkpoint save.

        Waits for the previous persist to finish (at most 1 ongoing),
        then starts the async GPU→CPU copy. Never skips.
        """
        t_lock_start = time.time()
        # Ensure previous persist is done before starting a new one
        self._persist_done.wait()
        with self._buffer_cond:
            while not self._buffer_free:
                logger.info(
                    f"[AsyncAnchor] Waiting for buffer at step {step}..."
                )
                self._buffer_cond.wait()
            self._buffer_free = False
        t_lock = time.time() - t_lock_start

        t_sd_start = time.time()
        state_dict = model.state_dict()
        t_sd = time.time() - t_sd_start

        event = self._copy_done_event
        pinned = self._pinned_buffer

        # Phase 1: async copy on dedicated CUDA stream
        t_copy_start = time.time()
        has_cuda = any(t.is_cuda for t in state_dict.values())
        if has_cuda:
            with torch.cuda.stream(self._ckpt_stream):
                for name in pinned:
                    pinned[name].copy_(state_dict[name], non_blocking=True)
                event.record()
            # GPU-side barrier: default stream waits for copy to finish
            # before next training step modifies the parameters.
            # This is a GPU-side dependency only — the CPU thread continues
            # immediately, so training never stalls on the CPU side.
            torch.cuda.current_stream().wait_stream(self._ckpt_stream)
        else:
            # All tensors on CPU (e.g. ZO2 offloading) — direct copy
            for name in pinned:
                pinned[name].copy_(state_dict[name])
            event.record()
        t_copy = time.time() - t_copy_start

        # Queue Phase 2 (CPU→disk) for background thread
        self._persist_queue.put(
            _PersistJob(step, output_dir, event)
        )
        self._anchors_saved += 1
        t_total = time.time() - t_lock_start
        self._copy_times.append(t_total)

        logger.info(
            f"[AsyncAnchor] Queued anchor step {step} "
            f"(lock={t_lock:.3f}s, state_dict={t_sd:.3f}s, "
            f"copy={t_copy:.3f}s, total_enqueue={t_total:.3f}s)"
        )
        return True

    def get_latest_completed_anchor_step(self) -> int:
        """Step number of the most recent fully-persisted anchor (-1 if none)."""
        with self._lock:
            return self._latest_completed_step

    def get_latest_completed_anchor_path(self):
        """Path of the most recent fully-persisted anchor (None if none)."""
        with self._lock:
            return self._latest_completed_path

    def wait_for_completion(self):
        """Block until all pending persists finish."""
        done = threading.Event()
        self._persist_queue.put(done)
        done.wait()

    def shutdown(self):
        """Wait for pending work and stop the persist thread."""
        self.wait_for_completion()
        self._persist_queue.put(None)  # sentinel to exit worker loop
        self._persist_thread.join(timeout=60)
        logger.info(f"[AsyncAnchor] Shutdown complete. Stats: {self.stats}")

    @property
    def stats(self) -> dict:
        """Checkpoint statistics."""
        return {
            'anchors_saved': self._anchors_saved,
            'avg_copy_time': (
                sum(self._copy_times) / len(self._copy_times)
                if self._copy_times else 0
            ),
            'avg_persist_time': (
                sum(self._persist_times) / len(self._persist_times)
                if self._persist_times else 0
            ),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _persist_worker(self):
        """Background thread: clone buffer, fork subprocess, blocking wait.

        At most one child process writes to disk at a time. The training
        thread gates new checkpoints via _persist_done.wait().
        """
        force_fsync = os.environ.get('FORCE_FSYNC', '0') == '1'

        while True:
            item = self._persist_queue.get()

            if item is None:
                break

            if isinstance(item, threading.Event):
                item.set()
                continue

            job = item
            # Block here until GPU→CPU DMA completes — this is the ONLY
            # place that calls synchronize, never in the training thread.
            job.copy_done_event.synchronize()

            # Phase 2a: Clone pinned buffer → regular CPU memory.
            t_clone_start = time.time()
            snapshot = {name: tensor.clone() for name, tensor in self._pinned_buffer.items()}
            t_clone = time.time() - t_clone_start

            # Free pinned buffer IMMEDIATELY and wake up waiting try_save.
            with self._buffer_cond:
                self._buffer_free = True
                self._buffer_cond.notify()

            # Phase 2b: Fork subprocess for disk I/O.
            os.makedirs(job.output_dir, exist_ok=True)
            save_path = os.path.join(job.output_dir, "model.safetensors")

            self._persist_done.clear()  # Mark persist as in-progress
            t0 = time.time()
            pid = os.fork()
            if pid == 0:
                _subprocess_write_checkpoint(snapshot, save_path)
                os._exit(1)  # Safety net
            else:
                del snapshot  # Parent doesn't need it; child has CoW copy

            logger.info(
                f"[AsyncAnchor] Forked persist for step {job.step} "
                f"(clone={t_clone:.3f}s)"
            )

            # Blocking wait — at most 1 child at a time.
            _, status = os.waitpid(pid, 0)
            t_total = time.time() - t0
            child_ok = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
            if child_ok:
                if force_fsync:
                    fd = os.open(save_path, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                with self._lock:
                    if job.step > self._latest_completed_step:
                        self._latest_completed_step = job.step
                        self._latest_completed_path = job.output_dir
                logger.info(
                    f"[AsyncAnchor] Persisted step {job.step} "
                    f"(clone={t_clone:.3f}s, total={t_total:.3f}s) "
                    f"→ {save_path}"
                )
            else:
                if os.WIFSIGNALED(status):
                    logger.error(
                        f"[AsyncAnchor] Subprocess killed by signal "
                        f"{os.WTERMSIG(status)} for step {job.step}"
                    )
                else:
                    logger.error(
                        f"[AsyncAnchor] Subprocess exited with code "
                        f"{os.WEXITSTATUS(status) if os.WIFEXITED(status) else '?'} "
                        f"for step {job.step}"
                    )
            self._persist_times.append(t_total)
            self._persist_done.set()  # Allow next checkpoint
