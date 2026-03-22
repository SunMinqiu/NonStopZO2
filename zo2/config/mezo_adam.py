# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

from dataclasses import dataclass
from typing import Tuple
from .mezo_sgd import MeZOSGDConfig

@dataclass
class MeZOAdamConfig(MeZOSGDConfig):
    zo_method: str = "mezo-adam"

    # Adam-specific parameters
    betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8  # distinct from zo perturbation eps
