# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

from .mezo_sgd import MeZOSGDConfig
from .mezo_adam import MeZOAdamConfig


def ZOConfig(method: str = "mezo-sgd", **kwargs):
    match method:
        case "mezo-sgd":
            return MeZOSGDConfig(**kwargs)
        case "mezo-adam":
            return MeZOAdamConfig(**kwargs)
        case _:
            raise ValueError(f"Unsupported method {method}")
