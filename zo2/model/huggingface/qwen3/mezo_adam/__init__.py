# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

from . import zo
from .....config.mezo_adam import MeZOAdamConfig

def get_qwen3_for_causalLM_mezo_adam(config: MeZOAdamConfig):
    return zo.Qwen3ForCausalLM
