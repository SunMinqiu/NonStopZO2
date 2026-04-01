# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import os


def time_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_TIME", "0") == "1"


def resource_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_RESOURCE", "0") == "1"


def consistency_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_CONSISTENCY", "0") == "1"


def shadow_send_time_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_SHADOW_SEND_TIME", "0") == "1"


def replay_step_time_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_REPLAY_STEP_TIME", "0") == "1"


def shadow_step_time_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_SHADOW_STEP_TIME", "0") == "1"


def shadow_step_resource_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_SHADOW_STEP_RESOURCE", "0") == "1"


def train_step_resource_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_TRAIN_STEP_RESOURCE", "0") == "1"


def memory_debug_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_MEMORY_DEBUG", "0") == "1"


def thread_snapshot_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_THREAD_SNAPSHOT", "0") == "1"


def batch_debug_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_BATCH_DEBUG", "0") == "1"


def state_diag_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_STATE_DIAG", "0") == "1"


def state_exact_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_STATE_EXACT", "0") == "1"


def rng_diag_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_RNG_DIAG", "0") == "1"


def z_diag_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_Z_DIAG", "0") == "1"


def z_exact_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_Z_EXACT", "0") == "1"


def opt_step_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_OPT_STEP", "0") == "1"


def loading_phase_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_LOADING_PHASE", "0") == "1"
