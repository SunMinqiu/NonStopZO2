# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn
import logging

from ..mezo_sgd.zo import MeZOSGD
from ...config.mezo_adam import MeZOAdamConfig

logger = logging.getLogger(__name__)


class MeZOAdam(MeZOSGD):
    """
    Implements MeZO with Adam (AdamW) update rule instead of SGD.

    Inherits perturbation, gradient estimation, and forward pass from MeZOSGD.
    Only overrides __init__ (Adam state), zo_forward (timestep counter),
    and zo_update (Adam update rule).
    """
    def __init__(self, model: nn.Module, config: MeZOAdamConfig):
        super().__init__(model, config)
        self.betas = config.betas
        self.adam_eps = config.adam_eps

        # Adam state: first moment (m) and second moment (v) per parameter
        # Lazy-initialized on first zo_update (like PyTorch Adam), ensuring
        # tensors are created on the correct device with the correct dtype.
        self.m = {}
        self.v = {}
        self.t = 0

    @torch.inference_mode
    def zo_forward(self, *args, zo_random_seed: int = None, **kwargs):
        """Override to increment Adam timestep once per training step."""
        self.t += 1
        return super().zo_forward(*args, zo_random_seed=zo_random_seed, **kwargs)

    @torch.inference_mode
    def zo_update(self, module, weight_decay=None):
        """
        Adam (AdamW) update rule using ZO gradient estimates.

        All optimizer state (m, v) and update computation use fp32 to prevent
        underflow in fp16/bf16 models, matching PyTorch's AdamW behavior.
        In MeZO, g = projected_grad * z is tiny, so g^2 underflows to 0 in fp16,
        collapsing the denominator to eps and causing explosive updates.
        """
        beta1, beta2 = self.betas
        bias_correction1 = 1 - beta1 ** self.t
        bias_correction2 = 1 - beta2 ** self.t
        step_size = self.lr / bias_correction1

        for name, param in module.named_parameters():
            if param.requires_grad:
                z = self._generate_z(param)
                g = (self.projected_grad * z).float()  # fp32 to prevent underflow

                pid = id(param)
                # Lazy init in fp32 (like PyTorch Adam) for numerical stability
                if pid not in self.m:
                    self.m[pid] = torch.zeros_like(param.data, dtype=torch.float32)
                    self.v[pid] = torch.zeros_like(param.data, dtype=torch.float32)

                # Update biased first and second moment estimates (in-place)
                self.m[pid].mul_(beta1).add_(g, alpha=1 - beta1)
                self.v[pid].mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # Adam step: denom = sqrt(v_hat) + eps, update = step_size * m / denom
                denom = (self.v[pid] / bias_correction2).sqrt_().add_(self.adam_eps)
                update = self.m[pid].div(denom).mul_(step_size)

                # Decoupled weight decay (AdamW), skip bias/layernorm
                if weight_decay is not None:
                    update.add_(param.data, alpha=self.lr * weight_decay)
                else:
                    if all(x not in name for x in ["bias", "layer_norm", "layernorm", "ln"]):
                        update.add_(param.data, alpha=self.lr * self.weight_decay)
                param.data.sub_(update.to(param.data.dtype))

    # ---- Adam state serialization for checkpoint / replay ----

    def get_adam_state(self) -> dict:
        """Export Adam state with param names as keys (id(param) changes across sessions)."""
        adam_m, adam_v = {}, {}
        for name, param in self.model.named_parameters():
            pid = id(param)
            if pid in self.m:
                adam_m[name] = self.m[pid]
                adam_v[name] = self.v[pid]
        return {'m': adam_m, 'v': adam_v, 't': self.t,
                'betas': self.betas, 'adam_eps': self.adam_eps}

    def restore_adam_state(self, adam_state: dict):
        """Restore m/v/t from a dict. Called by _init_for_resume or after replay."""
        self.t = adam_state.get('t', 0)
        for name, param in self.model.named_parameters():
            pid = id(param)
            if name in adam_state.get('m', {}):
                self.m[pid] = adam_state['m'][name].to(device=param.device, dtype=torch.float32)
                self.v[pid] = adam_state['v'][name].to(device=param.device, dtype=torch.float32)

    def state_dict(self):
        d = super().state_dict()
        d['adam_state'] = self.get_adam_state()
        return d

    def load_state_dict(self, state_dict):
        adam_state = state_dict.pop('adam_state', None)
        super().load_state_dict(state_dict)
        if adam_state is not None:
            self.restore_adam_state(adam_state)
