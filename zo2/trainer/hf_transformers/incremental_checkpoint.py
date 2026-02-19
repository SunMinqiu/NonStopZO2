"""
DEPRECATED: This module is deprecated. Use batch_differential_checkpoint.py instead.

This file now re-exports all symbols from batch_differential_checkpoint for backward compatibility.
"""

import warnings
warnings.warn(
    "incremental_checkpoint module is deprecated. Use batch_differential_checkpoint instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from new module for backward compatibility
from .batch_differential_checkpoint import (
    GPUFailureSimulator,
    BatchDiffCheckpointCallback,
    BatchDiffCheckpointCallback as ZOReplayCheckpointCallback,
    BatchDiffCheckpointCallback as IncrementalCheckpointCallback,
    load_batch_diff_checkpoint,
    load_batch_diff_checkpoint as load_zo_replay_checkpoint,
    load_batch_diff_checkpoint as load_incremental_checkpoint,
    resume_from_batch_diff,
)

# Also export for star imports
__all__ = [
    'GPUFailureSimulator',
    'BatchDiffCheckpointCallback',
    'ZOReplayCheckpointCallback',
    'IncrementalCheckpointCallback',
    'load_batch_diff_checkpoint',
    'load_zo_replay_checkpoint',
    'load_incremental_checkpoint',
    'resume_from_batch_diff',
]
