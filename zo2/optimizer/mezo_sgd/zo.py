# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import os
import sys
sys.path.append('./zo2')

import hashlib
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseOptimizer
import numpy as np
import logging

from ...config.mezo_sgd import MeZOSGDConfig
from ...utils.logging_controls import (
    rng_diag_log_enabled,
    state_diag_log_enabled,
    state_exact_log_enabled,
)

logger = logging.getLogger(__name__)


def _step_diag_enabled():
    return state_diag_log_enabled()


def _step_exact_enabled():
    return state_exact_log_enabled()


def _rng_diag_enabled():
    return rng_diag_log_enabled()


class MeZOSGD(BaseOptimizer):
    """
    Implements the [MeZO-SGD](https://arxiv.org/abs/2305.17333) optimization method, 
    particularly suited for scenarios with limited compute resources.
    """
    def __init__(self, model: nn.Module, config: MeZOSGDConfig):
        """
        Initializes the MeZOSGD optimizer which applies zeroth-order optimization techniques to the model parameters.

        Args:
            model (nn.Module): The model whose parameters will be optimized.
            config (MeZOSGDConfig): Configuration object containing optimizer settings.
        """
        self.config = config
        self.model = model
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.zo_eps = config.eps
        self.max_zo_random_seed = config.max_zo_random_seed
        self.debug_mode = config.debug_mode
        self.rng_device = getattr(config, 'rng_device', 'native')  # "native" or "cpu"
        self.use_fma = os.environ.get('ZO_FMA', '1') == '1'
        defaults = dict(
            lr=self.lr,
            weight_decay=self.weight_decay,
            maximize=False,
            foreach=None,
            differentiable=False,
            fused=None,
        )
        super().__init__(model.parameters(), defaults)
        logger.info(f"[MeZOSGD] use_fma={self.use_fma}, rng_device={self.rng_device}, weight_decay={self.weight_decay}")

    def _reset_rng(self, seed):
        """Reset RNG state for z generation. Called before each perturb/update block."""
        if self.rng_device == "zo_rng":
            import zo_rng
            self._zo_gen = zo_rng.Generator(seed)
        else:
            torch.manual_seed(seed)

    def _generate_z(self, param):
        """Generate z noise for a parameter, respecting rng_device setting."""
        if self.debug_mode:
            return torch.ones_like(param.data)
        if self.rng_device == "zo_rng":
            # Cross-device deterministic: bit-exact same output on CPU and GPU
            return self._zo_gen.randn(
                param.data.shape, dtype=param.data.dtype, device=param.data.device)
        if self.rng_device == "cpu" and param.data.device.type != "cpu":
            z = torch.normal(mean=0, std=1, size=param.data.size(), dtype=torch.float32, device='cpu')
            return z.to(dtype=param.data.dtype, device=param.data.device)
        else:
            return torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

    @torch.inference_mode
    def zo_perturb_parameters(self, module: nn.Module, scaling_factor: float=1):
        """
        Applies Gaussian noise to parameters of a module, facilitating zeroth-order optimization.

        Args:
            module (nn.Module): Module whose parameters will be perturbed.
            scaling_factor (float): Scaling factor for the noise applied to the parameters.
        """
        for _, param in module.named_parameters():
            if param.requires_grad:
                z = self._generate_z(param)
                if self.use_fma:
                    param.data.add_(z, alpha=float(scaling_factor * self.zo_eps))
                else:
                    param.data.add_(scaling_factor * z * self.zo_eps)

    @torch.inference_mode
    def zo_update(self, module, weight_decay=None):
        """
        Updates the parameters of a module based on zeroth-order perturbations and optional weight decay.

        Args:
            module (nn.Module): Module whose parameters will be updated.
            weight_decay (float, optional): Weight decay coefficient. If None, it defaults to the configuration.
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                z = self._generate_z(param)
                if self.use_fma:
                    wd = weight_decay if weight_decay is not None else (
                        self.weight_decay if all(x not in name for x in ["bias", "layer_norm", "layernorm", "ln"]) else 0.0
                    )
                    if wd != 0.0:
                        tmp = z.mul(self.projected_grad)
                        tmp.add_(param.data, alpha=wd)
                        param.data.sub_(tmp, alpha=self.lr)
                    else:
                        param.data.sub_(z, alpha=float(self.lr * self.projected_grad))
                else:
                    if weight_decay != None:
                        param.data.sub_(
                            self.lr * (self.projected_grad * z + weight_decay * param.data))
                    else:
                        if all(x not in name for x in ["bias", "layer_norm", "layernorm", "ln"]):
                            param.data.sub_(
                                self.lr * (self.projected_grad * z + self.weight_decay * param.data))
                        else:
                            param.data.sub_(self.lr * self.projected_grad * z)
    
    def zo_perturb_shifts(self, first_perturb_shift=1, stride=2):
        """
        Generates shifts for perturbing parameters in a pattern conducive to zeroth-order optimization.

        Returns:
            list: A list of perturb shifts used during the forward and update passes.
        """
        return [first_perturb_shift, -stride, stride-first_perturb_shift]

    def compute_grad(self, loss1, loss2):
        return ((loss1 - loss2) / (2 * self.zo_eps)).item()

    def _log_step_diag(self, loss1, loss2):
        if not _step_diag_enabled():
            return
        loss1_value = float(loss1.detach().float().item()) if torch.is_tensor(loss1) else float(loss1)
        loss2_value = float(loss2.detach().float().item()) if torch.is_tensor(loss2) else float(loss2)
        model_sum = 0.0
        tracked = {}
        digest = hashlib.sha256()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            cpu = param.detach().float().cpu().contiguous()
            model_sum += cpu.sum().item()
            if name in ("model.embed_tokens.weight", "lm_head.weight"):
                tracked[name] = cpu.sum().item()
            if _step_exact_enabled():
                digest.update(name.encode("utf-8"))
                digest.update(memoryview(cpu.numpy()).tobytes())
        parts = [
            f"class={self.__class__.__name__}",
            f"seed={getattr(self, 'zo_random_seed', None)}",
            f"loss1={loss1_value:.10e}",
            f"loss2={loss2_value:.10e}",
            f"projected_grad={float(getattr(self, 'projected_grad', 0.0)):.10e}",
            f"applied_grad={float(getattr(self, '_applied_update_grad', 0.0)):.10e}",
            f"lr={float(getattr(self, 'lr', 0.0)):.10e}",
            f"wd={float(getattr(self, 'weight_decay', 0.0)):.10e}",
            f"zo_eps={float(getattr(self, 'zo_eps', 0.0)):.10e}",
            f"rng_device={getattr(self, 'rng_device', 'native')}",
            f"model_sum={model_sum:.10e}",
        ]
        if hasattr(self, "t"):
            parts.append(f"adam_t={int(getattr(self, 't', 0))}")
        if hasattr(self, "get_adam_state"):
            adam_state = self.get_adam_state()
            m_sum = sum(t.float().sum().item() for t in adam_state.get("m", {}).values())
            v_sum = sum(t.float().sum().item() for t in adam_state.get("v", {}).values())
            parts.append(f"adam_m_sum={m_sum:.10e}")
            parts.append(f"adam_v_sum={v_sum:.10e}")
        if _step_exact_enabled():
            parts.append(f"model_sha256={digest.hexdigest()}")
        logger.info("[OPT-DIAG] " + " | ".join(parts))
        for name, value in tracked.items():
            logger.info(f"[OPT-DIAG] tracked[{name}]={value:.10e}")

    def _log_rng_diag(self, phase):
        if not _rng_diag_enabled():
            return

        parts = [
            f"phase={phase}",
            f"seed={getattr(self, 'zo_random_seed', None)}",
            f"rng_device={getattr(self, 'rng_device', 'native')}",
        ]

        cpu_state = torch.random.get_rng_state()
        parts.append(f"cpu_sha256={hashlib.sha256(memoryview(cpu_state.cpu().numpy()).tobytes()).hexdigest()}")

        if torch.cuda.is_available():
            try:
                cuda_state = torch.cuda.random.get_rng_state()
                parts.append(
                    f"cuda_sha256={hashlib.sha256(memoryview(cuda_state.cpu().numpy()).tobytes()).hexdigest()}"
                )
            except Exception as exc:
                parts.append(f"cuda_error={type(exc).__name__}")

        try:
            np_state = pickle.dumps(np.random.get_state(), protocol=4)
            parts.append(f"numpy_sha256={hashlib.sha256(np_state).hexdigest()}")
        except Exception as exc:
            parts.append(f"numpy_error={type(exc).__name__}")

        try:
            py_state = pickle.dumps(random.getstate(), protocol=4)
            parts.append(f"python_sha256={hashlib.sha256(py_state).hexdigest()}")
        except Exception as exc:
            parts.append(f"python_error={type(exc).__name__}")

        logger.info("[RNG-DIAG] " + " | ".join(parts))
        
    @torch.inference_mode
    def zo_forward(self, *args, zo_random_seed: int=None, **kwargs):
        """
        Forward pass that applies zeroth-order perturbations to compute the loss, used for gradient estimation.
        Notice that the application of Gaussian perturbations for the parameters during both the perturbation and update phases should be the same.

        Args:
            zo_random_seed (int, optional): Random seed for reproducibility of perturbations.
        """
        self._update_lr()
        self.zo_random_seed = zo_random_seed if zo_random_seed else np.random.randint(self.max_zo_random_seed)
        self._reset_rng(self.zo_random_seed)
        self.zo_perturb_parameters(self.model, scaling_factor=self.zo_perturb_shifts()[0])
        self._log_rng_diag("before_loss1")
        loss1 = self.inner_zo_forward(*args, **kwargs)
        self._reset_rng(self.zo_random_seed)
        self.zo_perturb_parameters(self.model, scaling_factor=self.zo_perturb_shifts()[1])
        self._log_rng_diag("before_loss2")
        loss2 = self.inner_zo_forward(*args, **kwargs)
        self.projected_grad = self.compute_grad(loss1, loss2)
        # In base ZO (synchronous), the update uses the grad just computed in the SAME step.
        # Record it so the checkpoint hook can pair (seed_N, grad_N) correctly.
        self._applied_update_grad = self.projected_grad
        self._reset_rng(self.zo_random_seed)
        self.zo_perturb_parameters(self.model, scaling_factor=self.zo_perturb_shifts()[2])
        self._log_rng_diag("before_update")
        self._reset_rng(self.zo_random_seed)
        self.zo_update(self.model)
        self._log_step_diag(loss1, loss2)
        return loss1

    #*********************** evaluate ***********************#

    @torch.inference_mode()
    def zo_eval_forward(self, *args, **kwargs):
        """
        Forward pass in evaluation mode.
        """
        output = self.inner_zo_eval_forward(*args, **kwargs)
        return output
    
    #*********************** api ***********************#

    @torch.inference_mode
    def inner_zo_forward(self, idx, pos, targets):
        """
        Example of ZO inner_zo_forward:
            Match the same args as the original model forward,
            and replace all 'self' to 'self.model'.
        """
        tok_emb = self.model.transformer.wte(idx)
        pos_emb = self.model.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.model.transformer.h:
            x = block(x)
        x = self.model.transformer.ln_f(x)
        x = self.model.lm_head(x)
        loss = F.cross_entropy(
            x[:, :-1, :].reshape(-1, x.size(-1)), 
            targets[:, 1:].reshape(-1)
        )
        return loss.detach()

    @torch.inference_mode()   
    def inner_zo_eval_forward(self, eval_fn, idx, pos, targets):
        output = eval_fn(idx, pos, targets)
        return output
    
