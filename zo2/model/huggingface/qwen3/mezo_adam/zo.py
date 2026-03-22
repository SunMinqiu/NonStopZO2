# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

from ..mezo_sgd.zo import (
    Qwen3ForCausalLM as SGDQwen3ForCausalLM,
    OptimizerQwen3ForCausalLM as SGDOptimizerQwen3ForCausalLM,
)
from .....optimizer.mezo_adam.zo import MeZOAdam


# --- Optimizer class: MeZOAdam first in MRO for zo_update ---

class OptimizerQwen3ForCausalLM(MeZOAdam, SGDOptimizerQwen3ForCausalLM):
    """Adam variant. MeZOAdam provides zo_update; SGD class provides inner_zo_forward."""
    pass


# --- Model class: reuse SGD model class, override zo_init ---

class Qwen3ForCausalLM(SGDQwen3ForCausalLM):
    def zo_init(self, zo_config):
        self.opt = OptimizerQwen3ForCausalLM(model=self, config=zo_config)
