"""
DEPRECATED: Async Anchor Checkpoint has been moved to legacy_functions.py.

This module re-exports the classes for backward compatibility only.
New code should NOT import from here.
"""
import warnings as _warnings
_warnings.warn(
    "async_anchor_checkpoint is deprecated; code moved to legacy_functions.py",
    DeprecationWarning,
    stacklevel=2,
)

from .legacy_functions import (  # noqa: F401
    _subprocess_write_checkpoint,
    _PersistJob,
    AsyncAnchorCheckpointer,
    ADAM_STATE_NAME,
)
