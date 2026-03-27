import logging
import os

logger = logging.getLogger(__name__)


class GPUFailureSimulator:
    """GPU failure simulator."""

    def __init__(self):
        self.fail_at_step = None
        self.has_failed = False
        self.callback = None

    def _failure_type(self) -> str:
        failure_type = os.environ.get("FAILURE_TYPE", "soft").strip().lower()
        if failure_type not in {"soft", "hard"}:
            logger.warning(
                "[GPU Failure] Unknown FAILURE_TYPE=%r, falling back to soft",
                failure_type,
            )
            return "soft"
        return failure_type

    def _stop_shadow(self):
        cb = self.callback
        if cb is None:
            return
        sp = getattr(cb, "shadow_process", None)
        if sp is not None and sp.is_alive():
            import signal

            logger.warning("[GPU Failure] Killing shadow process (PID=%d)", sp.pid)
            os.kill(sp.pid, signal.SIGKILL)
            sp.join(timeout=2.0)

    def set_fail_step(self, step: int):
        self.fail_at_step = step
        self.has_failed = False
        logger.info(
            "[GPU Failure] Will simulate %s failure at step %s",
            self._failure_type(),
            step,
        )

    def check_and_fail(self, current_step: int, model):
        if self.fail_at_step is not None and current_step >= self.fail_at_step and not self.has_failed:
            self.has_failed = True
            logger.warning(
                "[GPU Failure] Simulating %s failure at step %s!",
                self._failure_type(),
                current_step,
            )
            return True
        return False

    def trigger_failure(self, model):
        """Simulate either soft or hard failure based on FAILURE_TYPE."""
        import signal

        failure_type = self._failure_type()
        self._stop_shadow()

        if failure_type == "hard":
            logger.error("[GPU Failure] HARD failure -> raising RuntimeError")
            for handler in logging.root.handlers:
                handler.flush()
            raise RuntimeError("Simulated hard failure triggered by FAILURE_TYPE=hard")

        logger.warning("[GPU Failure] SIGKILL -> pid=%d", os.getpid())
        for handler in logging.root.handlers:
            handler.flush()
        os.kill(os.getpid(), signal.SIGKILL)
