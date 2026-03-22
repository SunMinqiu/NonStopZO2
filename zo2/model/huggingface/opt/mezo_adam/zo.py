# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

from ..mezo_sgd.zo import (
    OPTForCausalLM as SGDOPTForCausalLM,
    OPTForSequenceClassification as SGDOPTForSequenceClassification,
    OPTForQuestionAnswering as SGDOPTForQuestionAnswering,
    OptimizerOPTForCausalLM as SGDOptimizerOPTForCausalLM,
    OptimizerOPTForSequenceClassification as SGDOptimizerOPTForSequenceClassification,
    OptimizerOPTForQuestionAnswering as SGDOptimizerOPTForQuestionAnswering,
)
from .....optimizer.mezo_adam.zo import MeZOAdam


# --- Optimizer classes: MeZOAdam first in MRO for zo_update ---

class OptimizerOPTForCausalLM(MeZOAdam, SGDOptimizerOPTForCausalLM):
    """Adam variant. MeZOAdam provides zo_update; SGD class provides inner_zo_forward."""
    pass

class OptimizerOPTForSequenceClassification(MeZOAdam, SGDOptimizerOPTForSequenceClassification):
    pass

class OptimizerOPTForQuestionAnswering(MeZOAdam, SGDOptimizerOPTForQuestionAnswering):
    pass


# --- Model classes: reuse SGD model classes, override zo_init ---

class OPTForCausalLM(SGDOPTForCausalLM):
    def zo_init(self, zo_config):
        self.opt = OptimizerOPTForCausalLM(model=self, config=zo_config)

class OPTForSequenceClassification(SGDOPTForSequenceClassification):
    def zo_init(self, zo_config):
        self.opt = OptimizerOPTForSequenceClassification(model=self, config=zo_config)

class OPTForQuestionAnswering(SGDOPTForQuestionAnswering):
    def zo_init(self, zo_config):
        self.opt = OptimizerOPTForQuestionAnswering(model=self, config=zo_config)
