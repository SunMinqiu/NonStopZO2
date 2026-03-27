import logging
import os

logger = logging.getLogger(__name__)


class GPUFailureSimulator:
    """GPU failure simulator."""

    def __init__(self):
        self.fail_at_step = None
        self.has_failed = False
        self.callback = None

    def set_fail_step(self, step: int):
        self.fail_at_step = step
        self.has_failed = False
        logger.info(f"[GPU Failure] Will simulate failure at step {step}")

    def check_and_fail(self, current_step: int, model):
        if self.fail_at_step is not None and current_step >= self.fail_at_step and not self.has_failed:
            self.has_failed = True
            logger.warning(f"[GPU Failure] Simulating GPU failure at step {current_step}!")
            return True
        return False

    def trigger_failure(self, model):
        """Simulate soft failure via SIGKILL."""
        import signal

        cb = self.callback
        if cb is not None:
            sp = getattr(cb, 'shadow_process', None)
            if sp is not None and sp.is_alive():
                logger.warning("[GPU Failure] Killing shadow process (PID=%d)", sp.pid)
                os.kill(sp.pid, signal.SIGKILL)
                sp.join(timeout=2.0)

        logger.warning("[GPU Failure] SIGKILL -> pid=%d", os.getpid())
        for handler in logging.root.handlers:
            handler.flush()
        os.kill(os.getpid(), signal.SIGKILL)
