import logging
import os
from typing import Iterable, List, Optional, Union

logger = logging.getLogger(__name__)


def parse_gpu_fail_steps(spec: Union[str, int, Iterable[Union[str, int]], None]) -> List[int]:
    """Parse GPU_FAIL_STEP into a sorted list of positive global steps."""
    if spec is None:
        return []

    if isinstance(spec, (list, tuple, set)):
        raw_parts = list(spec)
    else:
        text = str(spec).strip()
        if text in {"", "-1", "none", "None"}:
            return []
        raw_parts = text.split(",")

    steps = []
    for raw_part in raw_parts:
        part = str(raw_part).strip()
        if not part:
            continue
        step = int(part)
        if step == -1 and len(raw_parts) == 1:
            return []
        if step <= 0:
            raise ValueError(f"GPU_FAIL_STEP must contain only positive integers or -1, got {spec!r}")
        steps.append(step)

    return sorted(set(steps))


def format_gpu_fail_steps(steps: Iterable[int]) -> str:
    steps = list(steps)
    if not steps:
        return "-1"
    return ",".join(str(step) for step in steps)


class GPUFailureSimulator:
    """GPU failure simulator."""

    def __init__(self):
        self.fail_steps: List[int] = []
        self.next_fail_idx = 0
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
        self.set_fail_steps(step)

    def set_fail_steps(self, steps: Union[str, int, Iterable[Union[str, int]], None]):
        self.fail_steps = parse_gpu_fail_steps(steps)
        self.next_fail_idx = 0
        logger.info(
            "[GPU Failure] Will simulate %s failure at step(s) %s",
            self._failure_type(),
            format_gpu_fail_steps(self.fail_steps),
        )

    def get_remaining_fail_steps(self) -> List[int]:
        return list(self.fail_steps[self.next_fail_idx:])

    def get_next_fail_step(self) -> Optional[int]:
        remaining = self.get_remaining_fail_steps()
        return remaining[0] if remaining else None

    def advance_past(self, current_step: int):
        while self.next_fail_idx < len(self.fail_steps) and self.fail_steps[self.next_fail_idx] <= current_step:
            self.next_fail_idx += 1

    @property
    def fail_at_step(self):
        return self.get_next_fail_step()

    def check_and_fail(self, current_step: int, model):
        next_fail = self.get_next_fail_step()
        if next_fail is not None and current_step >= next_fail:
            self.next_fail_idx += 1
            logger.warning(
                "[GPU Failure] Simulating %s failure at step %s!",
                self._failure_type(),
                next_fail,
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
