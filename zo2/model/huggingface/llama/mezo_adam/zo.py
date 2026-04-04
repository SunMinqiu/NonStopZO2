# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn
from ..mezo_sgd.zo import (
    LlamaForCausalLM as SGDLlamaForCausalLM,
)
from .....optimizer.mezo_adam.zo import MeZOAdam


class OptimizerLlamaForCausalLM(MeZOAdam):
    """Adam variant with same inner_zo_forward as SGD version."""

    @torch.inference_mode()
    def inner_zo_forward(self, input_ids=None, attention_mask=None,
                         position_ids=None, past_key_values=None,
                         inputs_embeds=None, labels=None,
                         use_cache=None, output_attentions=None,
                         output_hidden_states=None, return_dict=None,
                         cache_position=None, **kwargs):
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        logits = self.model.lm_head(outputs[0]).contiguous()

        if self.model.zo_train_loss_fn_pre_hooks:
            for pre_hook_fn in self.model.zo_train_loss_fn_pre_hooks:
                input_ids, logits, labels = pre_hook_fn(self.model, input_ids, logits, labels)

        loss = None
        if self.model.zo_custom_train_loss_fn:
            loss = self.model.zo_custom_train_loss_fn(self.model, input_ids, logits, labels, **kwargs)
        elif labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))

        if self.model.zo_train_loss_fn_post_hooks:
            for post_hook_fn in self.model.zo_train_loss_fn_post_hooks:
                loss, input_ids, logits, labels = post_hook_fn(self.model, loss, input_ids, logits, labels)

        return loss.detach()


class LlamaForCausalLM(SGDLlamaForCausalLM):
    def zo_init(self, zo_config):
        self.opt = OptimizerLlamaForCausalLM(model=self, config=zo_config)
