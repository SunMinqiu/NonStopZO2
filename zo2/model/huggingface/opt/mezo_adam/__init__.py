# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

from . import zo
from .....config.mezo_adam import MeZOAdamConfig

def get_opt_for_causalLM_mezo_adam(config: MeZOAdamConfig):
    return zo.OPTForCausalLM

def get_opt_for_sequence_classification_mezo_adam(config: MeZOAdamConfig):
    return zo.OPTForSequenceClassification

def get_opt_for_question_answering_mezo_adam(config: MeZOAdamConfig):
    return zo.OPTForQuestionAnswering
