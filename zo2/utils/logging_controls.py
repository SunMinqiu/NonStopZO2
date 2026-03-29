# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import os


def time_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_TIME", "0") == "1"


def resource_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_RESOURCE", "0") == "1"


def consistency_log_enabled() -> bool:
    return os.environ.get("ZO_LOG_CONSISTENCY", "0") == "1"
