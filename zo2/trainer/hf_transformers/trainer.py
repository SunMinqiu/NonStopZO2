# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import copy
import functools
import glob
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union


# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    get_reporting_integration_callbacks,
)

# isort: on

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import psutil
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

from transformers import Trainer
from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.optimization import Adafactor, get_scheduler
from transformers.processing_utils import ProcessorMixin
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_2_3,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
    set_rng_state_for_device,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    SaveStrategy,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_apollo_torch_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_ipex_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_compile_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.quantization_config import QuantizationMethod


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        AutocastKwargs,
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("1.3.0"):
        from accelerate.utils import TorchTensorParallelPlugin
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


def _get_fsdp_ckpt_kwargs():
    # TODO: @AjayP13, @younesbelkada replace this check with version check at the next `accelerate` release
    if is_accelerate_available() and "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
        return {"adapter_only": True}
    else:
        return {}


def safe_globals():
    # Starting from version 2.4 PyTorch introduces a check for the objects loaded
    # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
    # a default and requires allowlisting of objects being loaded.
    # See: https://github.com/pytorch/pytorch/pull/137602
    # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
    # See: https://github.com/huggingface/accelerate/pull/3036
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    np_core = np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
    allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
    # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
    # all versions of numpy
    allowlist += [type(np.dtype(np.uint32))]

    return torch.serialization.safe_globals(allowlist)


if TYPE_CHECKING:
    import optuna

    if is_datasets_available():
        import datasets

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
LOG_METADATA_NAME = "log_metadata.pt"
SCALER_NAME = "scaler.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


class ZOTrainer(Trainer):

    # Those are used as methods of the Trainer in examples.
    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, None] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset], "datasets.Dataset"]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, processing_class, model_init, compute_loss_func, compute_metrics, callbacks, optimizers, optimizer_cls_and_kwargs, preprocess_logits_for_metrics)
        
        # ZO2 added: if using ZO2:
        if hasattr(model, "zo_training"):
            print("ZO training mode is enabled.")
            self.zo = True
        else:
            self.zo = False

        # ZO2 added: currently unsupported conditions
        if self.zo:
            self._zo2_unsupported_conditions(args)
        
        # ZO2 added: init hooks buffer
        if self.zo:
            self.zo2_training_step_pre_hooks = []
            self.zo2_training_step_post_hooks = []

        # ZO2 added: log-based checkpoint flag
        self.use_log_based_checkpoint = False
        self._log_based_resume_state = None
        self._log_based_resume_checkpoint = None
        self._memory_debug_every = 0
        self._memory_debug_sync = os.environ.get("ZO_MEMORY_DEBUG_SYNC", "1") != "0"
        self._memory_debug_process = None

        if self.zo and torch.cuda.is_available():
            env_every = os.environ.get("ZO_MEMORY_DEBUG_EVERY")
            if env_every is not None:
                try:
                    self._memory_debug_every = max(0, int(env_every))
                except ValueError:
                    logger.warning(f"[MemDebug] Invalid ZO_MEMORY_DEBUG_EVERY={env_every!r}, disabling memory debug")
                    self._memory_debug_every = 0
            elif os.environ.get("ZO_MEMORY_DEBUG", "0") == "1":
                self._memory_debug_every = 1
            elif isinstance(getattr(args, "logging_steps", 0), int) and args.logging_steps > 0:
                self._memory_debug_every = args.logging_steps

            if self._memory_debug_every > 0:
                self._memory_debug_process = psutil.Process(os.getpid())
                logger.info(
                    f"[MemDebug] Enabled: every={self._memory_debug_every} steps, "
                    f"sync={self._memory_debug_sync}"
                )

    def set_log_based_resume_state(self, *, global_step: int, checkpoint_path: str | None = None):
        self._log_based_resume_state = {
            "global_step": int(global_step),
        }
        self._log_based_resume_checkpoint = checkpoint_path

    def _memory_debug_due(self, step: int) -> bool:
        if not self.zo or self._memory_debug_every <= 0 or not torch.cuda.is_available():
            return False
        return step <= 3 or step % self._memory_debug_every == 0

    def _should_log_batch_debug(self, step: int) -> bool:
        if step <= 3:
            return True
        if os.environ.get("ZO_STEP_DIAG", "0") == "1":
            return True
        if os.environ.get("ZO_BATCH_DEBUG", "0") == "1":
            return True
        debug_start = int(os.environ.get("ZO_BATCH_DEBUG_STEP_START", "0"))
        debug_end = int(os.environ.get("ZO_BATCH_DEBUG_STEP_END", "0"))
        if debug_start > 0:
            if debug_end > 0:
                if debug_start <= step <= debug_end:
                    return True
            elif step >= debug_start:
                return True
        resume_steps = int(os.environ.get("ZO_BATCH_DEBUG_RESUME_STEPS", "12"))
        if self._log_based_resume_state is not None:
            resume_start = int(self._log_based_resume_state.get("global_step", 0))
            if resume_start < step <= resume_start + resume_steps:
                return True
        batch_debug_every = int(os.environ.get("ZO_BATCH_DEBUG_EVERY", "0"))
        if batch_debug_every > 0 and step % batch_debug_every == 0:
            return True
        return False

    def _batch_tensor_hash(self, tensor: torch.Tensor) -> str:
        cpu = tensor.detach().cpu().contiguous()
        return hashlib.sha256(memoryview(cpu.numpy()).tobytes()).hexdigest()[:16]

    def _log_batch_debug(self, step: int, inputs: dict) -> None:
        if not self._should_log_batch_debug(step):
            return
        parts = []
        for key in ("input_ids", "attention_mask", "labels", "option_len", "num_options"):
            value = inputs.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            item = f"{key}:shape={tuple(value.shape)} sha={self._batch_tensor_hash(value)}"
            if key == "input_ids" and value.ndim >= 2 and value.size(0) > 0:
                head = value[0, : min(8, value.size(1))].detach().cpu().tolist()
                item += f" head0={head}"
            elif key == "labels" and value.ndim >= 2 and value.size(0) > 0:
                head = value[0, : min(8, value.size(1))].detach().cpu().tolist()
                item += f" head0={head}"
            elif key == "labels" and value.ndim == 1 and value.numel() > 0:
                item += f" head0={value[: min(8, value.numel())].detach().cpu().tolist()}"
            elif key == "attention_mask" and value.ndim >= 2:
                lengths = value.sum(dim=-1)[: min(4, value.size(0))].detach().cpu().tolist()
                item += f" lens={lengths}"
            parts.append(item)
        logger.info(f"[BATCH] step={step} " + " | ".join(parts))

    def _log_memory_debug(self, tag: str, step: int, *, reset_peak: bool = False) -> None:
        if not torch.cuda.is_available():
            return
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        if self._memory_debug_sync:
            torch.cuda.synchronize()

        alloc_mb = torch.cuda.memory_allocated() / 1024**2
        reserved_mb = torch.cuda.memory_reserved() / 1024**2
        peak_alloc_mb = torch.cuda.max_memory_allocated() / 1024**2
        peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
        cpu_rss_mb = 0.0
        if self._memory_debug_process is not None:
            cpu_rss_mb = self._memory_debug_process.memory_info().rss / 1024**2

        logger.info(
            f"[MemDebug] step={step} {tag} | "
            f"GPU alloc={alloc_mb:.0f}MB rsv={reserved_mb:.0f}MB "
            f"peak_alloc={peak_alloc_mb:.0f}MB peak_rsv={peak_reserved_mb:.0f}MB "
            f"| CPU RSS={cpu_rss_mb:.0f}MB"
        )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """
        Override: skip model weight loading when log-based resume is active,
        since the model weights have already been recovered via replay.
        For standard resume (RESUME_CKPT), add timing measurement.
        """
        if getattr(self.args, "log_based_resume", ""):
            logger.info("[LogBased Resume] Skipping _load_from_checkpoint (model already recovered via replay)")
            return

        # Detect log-based checkpoint mismatch: checkpoint has no model files but has
        # ZO replay metadata, meaning it was saved with batch_size>=0.
        # If the current run uses batch_size=-1 (disabled), this is a configuration error.
        if os.path.isdir(resume_from_checkpoint):
            has_model = (
                os.path.isfile(os.path.join(resume_from_checkpoint, "model.safetensors"))
                or os.path.isfile(os.path.join(resume_from_checkpoint, "model.safetensors.index.json"))
                or os.path.isfile(os.path.join(resume_from_checkpoint, "pytorch_model.bin"))
                or os.path.isfile(os.path.join(resume_from_checkpoint, "pytorch_model.bin.index.json"))
            )
            if not has_model:
                opt_path = os.path.join(resume_from_checkpoint, "optimizer.pt")
                if os.path.isfile(opt_path):
                    raise ValueError(
                        f"[Resume Error] The checkpoint at {resume_from_checkpoint} appears to be "
                        f"a log-based checkpoint (no model files, but has ZO replay metadata). "
                        f"This means the previous run used log-based checkpointing (batch_size>=0), "
                        f"but the current run has batch_size=-1 (disabled). To fix this:\n"
                        f"  1. Use the same batch_size as the previous run (e.g. batch_size=0), OR\n"
                        f"  2. Use --log_based_resume={resume_from_checkpoint} for explicit replay-based recovery, OR\n"
                        f"  3. Use --overwrite_output_dir to start fresh training."
                    )

        import time as time_module
        t_start = time_module.time()
        result = super()._load_from_checkpoint(resume_from_checkpoint, model)
        t_elapsed = time_module.time() - t_start
        logger.info(f"[Resume] Model loaded from checkpoint in {t_elapsed:.3f}s")
        return result

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        """Override to measure total checkpoint time (including on_save callback)."""
        import time as time_module
        step = self.state.global_step

        # Let parent handle log and evaluate, but intercept the save block
        # --- Log ---
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            from transformers.integrations.tpu import is_torch_xla_available
            if is_torch_xla_available():
                import torch_xla.core.xla_model as xm
                xm.mark_step()

            logs: dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs, start_time)

        # --- Evaluate ---
        metrics = None
        if self.control.should_evaluate:
            if self._memory_debug_due(step):
                self._log_memory_debug("before_evaluate", step, reset_peak=True)
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            if self._memory_debug_due(step):
                self._log_memory_debug("after_evaluate", step)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)
            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        # --- Save (with timing) ---
        if self.control.should_save:
            if self._memory_debug_due(step):
                self._log_memory_debug("before_save", step, reset_peak=True)
            t_ckpt_start = time_module.time()
            t_save_impl_start = time_module.time()
            self._save_checkpoint(model, trial)
            t_save_impl_elapsed = time_module.time() - t_save_impl_start
            t_on_save_start = time_module.time()
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            t_on_save_elapsed = time_module.time() - t_on_save_start
            t_ckpt_elapsed = time_module.time() - t_ckpt_start
            log_based_callback = None
            from .log_based_checkpoint import LogBasedCheckpointCallback

            for callback in self.callback_handler.callbacks:
                if isinstance(callback, LogBasedCheckpointCallback):
                    log_based_callback = callback
                    break
            if self._memory_debug_due(step):
                self._log_memory_debug("after_save", step)
            if log_based_callback is not None and getattr(log_based_callback, "_last_save_is_full", False):
                log_based_callback.timing_stats['checkpoint_total_times'].append(t_ckpt_elapsed)
                logger.info(
                    f"[ZOTrainer SavePath] step={self.state.global_step} "
                    f"save_impl={t_save_impl_elapsed:.3f}s on_save={t_on_save_elapsed:.3f}s "
                    f"total={t_ckpt_elapsed:.3f}s"
                )
                logger.info(
                    f"[ZOTrainer] Full checkpoint at step {self.state.global_step} "
                    f"took {t_ckpt_elapsed:.3f}s"
                )
            elif log_based_callback is None or not log_based_callback.enable_shadow:
                logger.info(f"[ZOTrainer] Checkpoint at step {self.state.global_step} took {t_ckpt_elapsed:.3f}s")

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Override: for batch_size >= 0, save log-based checkpoint (optimizer.pt with
        zo_update_history). On full checkpoint steps, also save model files via Trainer.
        For batch_size = -1 (disabled), falls through to default Trainer save.
        """
        from .log_based_checkpoint import LogBasedCheckpointCallback

        # Check if the log-based checkpoint callback is registered
        log_based_callback = None
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, LogBasedCheckpointCallback):
                log_based_callback = callback
                self.use_log_based_checkpoint = True
                break

        if self.use_log_based_checkpoint and log_based_callback is not None and log_based_callback.batch_size >= 0:
            # Unified log-based save for all batch_size >= 0 modes
            batch_size = log_based_callback.batch_size
            log_based_callback._last_save_is_full = False
            t_save_breakdown_start = time.time()

            # Async anchor: check if a background persist completed and update base
            async_anchor = getattr(self, '_async_anchor', None)
            if async_anchor is not None and batch_size >= 1:
                completed_step = async_anchor.get_latest_completed_anchor_step()
                if completed_step > log_based_callback.base_checkpoint_step:
                    completed_path = async_anchor.get_latest_completed_anchor_path()
                    log_based_callback.on_async_anchor_persisted(completed_step, completed_path)
                    logger.info(f"[AsyncAnchor] Base updated to step {completed_step}, "
                               f"trimmed to {len(log_based_callback.update_history)} entries")

            # Determine if this is a full checkpoint step (batch_size >= 1 only)
            is_full_step = False
            if batch_size >= 1:
                if async_anchor is not None:
                    # Use last-attempted step for scheduling to avoid repeated triggers
                    effective_base = getattr(log_based_callback, '_async_anchor_last_attempted',
                                             log_based_callback.base_checkpoint_step)
                    steps_since_base = self.state.global_step - effective_base
                else:
                    steps_since_base = self.state.global_step - log_based_callback.base_checkpoint_step
                is_full_step = (steps_since_base >= batch_size)
            log_based_callback._last_save_is_full = bool(is_full_step)

            # With shadow enabled, redo logs stay in memory/IPC only.
            # Without shadow, per-step log checkpoints are still persisted and may use a separate log dir.
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            log_output_dir = getattr(self, '_log_output_dir', None)
            shadow_keeps_log_only = async_anchor is not None and log_based_callback.enable_shadow
            persist_lightweight_redo = bool(
                shadow_keeps_log_only and getattr(log_based_callback, "instant_recover", False)
            )
            if persist_lightweight_redo:
                run_dir = self._get_output_dir(trial=trial)
            elif log_output_dir and not shadow_keeps_log_only and (not is_full_step or async_anchor is not None):
                run_dir = log_output_dir
            else:
                run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            if (not shadow_keeps_log_only) or is_full_step or persist_lightweight_redo:
                os.makedirs(output_dir, exist_ok=True)

            sync_full_uses_sidecar = bool(is_full_step and async_anchor is None)
            log_metadata = None
            with log_based_callback.update_lock:
                updates = log_based_callback.update_history.copy()
            t_state_json = 0.0
            t_scheduler = 0.0
            t_rng = 0.0
            t_opt_build = 0.0
            t_opt_save = 0.0
            t_meta_save = 0.0
            if not shadow_keeps_log_only and not sync_full_uses_sidecar:
                # Save trainer state
                t0 = time.time()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
                t_state_json = time.time() - t0

                # Save scheduler state
                if self.lr_scheduler is not None:
                    t0 = time.time()
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                    t_scheduler = time.time() - t0

                # Save RNG state for deterministic resume
                t0 = time.time()
                self._save_rng_state(output_dir)
                t_rng = time.time() - t0

                # Detect model dtype for replay consistency
                model_dtype = None
                for p in model.parameters():
                    model_dtype = str(p.dtype)  # e.g. "torch.float16"
                    break

                # Full checkpoints keep the full optimizer state for direct resume.
                # Pure log checkpoints only need the optimizer skeleton plus replay
                # metadata; storing MeZO-Adam's full adam_state defeats the purpose
                # of log-based checkpointing.
                t0 = time.time()
                if is_full_step:
                    optimizer_state = self.optimizer.state_dict()
                else:
                    optimizer_state = torch.optim.Optimizer.state_dict(self.optimizer)

                # Add zo_update_history and metadata for replay
                optimizer_state['zo_update_history'] = updates
                if batch_size == 0:
                    optimizer_state['base_checkpoint'] = '__initial__'
                else:
                    optimizer_state['base_checkpoint'] = log_based_callback.base_checkpoint_path
                optimizer_state['current_step'] = self.state.global_step
                optimizer_state['batch_size'] = batch_size
                optimizer_state['num_updates'] = len(updates)
                optimizer_state['tied_weights'] = getattr(log_based_callback, '_tied_weight_groups', [])
                optimizer_state['model_dtype'] = model_dtype
                optimizer_state['pending_grad'] = getattr(log_based_callback, '_pending_grad', 0.0)
                optimizer_state['pending_seed'] = getattr(log_based_callback, '_pending_seed', 0)
                optimizer_state['base_pending_seed'] = getattr(log_based_callback, '_base_pending_seed', 0)
                optimizer_state['zo2'] = hasattr(model, 'opt') and hasattr(model.opt, 'rstate_queue')
                optimizer_state['trainable_param_names'] = getattr(log_based_callback, '_trainable_param_names', None)
                # With async anchor, is_full_checkpoint is always False because
                # model.safetensors may not be on disk yet when optimizer.pt is written.
                # Recovery will use the replay path (base + redo log) which is always correct.
                if async_anchor is not None:
                    optimizer_state['is_full_checkpoint'] = False
                else:
                    optimizer_state['is_full_checkpoint'] = is_full_step

                # Save zo_eps for replay: needed to simulate fp16 perturbation residuals
                zo_eps = getattr(model.opt, 'zo_eps', 0.0) if hasattr(model, 'opt') else 0.0
                optimizer_state['zo_eps'] = zo_eps

                # Save rng_device for replay: ensures replay uses same RNG device as training
                rng_device = getattr(model.opt, 'rng_device', 'native') if hasattr(model, 'opt') else 'native'
                optimizer_state['rng_device'] = rng_device

                # Save zo_method for Adam detection during replay
                zo_method = 'mezo-sgd'
                if hasattr(model, 'opt') and hasattr(model.opt, 'betas'):
                    zo_method = 'mezo-adam'
                    optimizer_state['adam_betas'] = model.opt.betas
                    optimizer_state['adam_eps_value'] = model.opt.adam_eps
                optimizer_state['zo_method'] = zo_method
                log_metadata = {
                    key: value
                    for key, value in optimizer_state.items()
                    if key not in {"state", "param_groups", "adam_state"}
                }
                t_opt_build = time.time() - t0

                opt_path = os.path.join(output_dir, OPTIMIZER_NAME)
                t0 = time.time()
                torch.save(optimizer_state, opt_path)
                from .log_based_utils import _fsync_file
                _fsync_file(opt_path)
                t_opt_save = time.time() - t0
                log_based_callback.disk_log_save_count += 1
                logger.info(
                    f"[ZOTrainer] Saved {len(updates)} updates to disk, "
                    f"pending_grad={optimizer_state['pending_grad']:.6e}, is_full_step={is_full_step}"
                )
            elif not shadow_keeps_log_only and sync_full_uses_sidecar:
                model_dtype = None
                for p in model.parameters():
                    model_dtype = str(p.dtype)
                    break
                log_metadata = {
                    'zo_update_history': [],
                    'base_checkpoint': output_dir,
                    'current_step': self.state.global_step,
                    'batch_size': batch_size,
                    'num_updates': 0,
                    'tied_weights': getattr(log_based_callback, '_tied_weight_groups', []),
                    'model_dtype': model_dtype,
                    'pending_grad': getattr(log_based_callback, '_pending_grad', 0.0),
                    'pending_seed': getattr(log_based_callback, '_pending_seed', 0),
                    'base_pending_seed': getattr(log_based_callback, '_base_pending_seed', 0),
                    'zo2': hasattr(model, 'opt') and hasattr(model.opt, 'rstate_queue'),
                    'trainable_param_names': getattr(log_based_callback, '_trainable_param_names', None),
                    'is_full_checkpoint': True,
                    'zo_eps': getattr(model.opt, 'zo_eps', 0.0) if hasattr(model, 'opt') else 0.0,
                    'rng_device': getattr(model.opt, 'rng_device', 'native') if hasattr(model, 'opt') else 'native',
                    'zo_method': 'mezo-sgd',
                }
                if hasattr(model, 'opt') and hasattr(model.opt, 'betas'):
                    log_metadata['zo_method'] = 'mezo-adam'
                    log_metadata['adam_betas'] = model.opt.betas
                    log_metadata['adam_eps_value'] = model.opt.adam_eps
            elif persist_lightweight_redo:
                if self.lr_scheduler is not None:
                    t0 = time.time()
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                    t_scheduler = time.time() - t0

                t0 = time.time()
                self._save_rng_state(output_dir)
                t_rng = time.time() - t0

                model_dtype = None
                for p in model.parameters():
                    model_dtype = str(p.dtype)
                    break

                t0 = time.time()
                zo_method = 'mezo-sgd'
                log_metadata = {
                    'zo_update_history': updates,
                    'base_checkpoint': '__initial__' if batch_size == 0 else log_based_callback.base_checkpoint_path,
                    'current_step': self.state.global_step,
                    'batch_size': batch_size,
                    'num_updates': len(updates),
                    'tied_weights': getattr(log_based_callback, '_tied_weight_groups', []),
                    'model_dtype': model_dtype,
                    'pending_grad': getattr(log_based_callback, '_pending_grad', 0.0),
                    'pending_seed': getattr(log_based_callback, '_pending_seed', 0),
                    'base_pending_seed': getattr(log_based_callback, '_base_pending_seed', 0),
                    'zo2': hasattr(model, 'opt') and hasattr(model.opt, 'rstate_queue'),
                    'trainable_param_names': getattr(log_based_callback, '_trainable_param_names', None),
                    'is_full_checkpoint': False,
                    'zo_eps': getattr(model.opt, 'zo_eps', 0.0) if hasattr(model, 'opt') else 0.0,
                    'rng_device': getattr(model.opt, 'rng_device', 'native') if hasattr(model, 'opt') else 'native',
                }
                if hasattr(model, 'opt') and hasattr(model.opt, 'betas'):
                    zo_method = 'mezo-adam'
                    log_metadata['adam_betas'] = model.opt.betas
                    log_metadata['adam_eps_value'] = model.opt.adam_eps
                log_metadata['zo_method'] = zo_method
                t_opt_build = time.time() - t0

                meta_path = os.path.join(output_dir, LOG_METADATA_NAME)
                t0 = time.time()
                torch.save(log_metadata, meta_path)
                from .log_based_utils import _fsync_file
                _fsync_file(meta_path)
                t_meta_save = time.time() - t0
                log_based_callback.disk_log_save_count += 1
                logger.info(
                    f"[ZOTrainer] Saved lightweight scalar redo metadata to {LOG_METADATA_NAME}, "
                    f"updates={len(updates)}, pending_grad={log_metadata['pending_grad']:.6e}, "
                    f"is_full_step={is_full_step}"
                )
            else:
                logger.info(
                    f"[ZOTrainer] Shadow-only log at step {self.state.global_step}; disk log persistence skipped"
                )

            t_super_full = 0.0
            t_post_full = 0.0
            if is_full_step:
                if async_anchor is not None:
                    # Async path: anchor model.safetensors is persisted in the same checkpoint dir.
                    anchor_run_dir = self._get_output_dir(trial=trial)
                    anchor_output_dir = os.path.join(anchor_run_dir, checkpoint_folder)
                    accepted = async_anchor.try_save_full_checkpoint(
                        self.state.global_step, model, anchor_output_dir)
                    # Always update scheduling base — even on rejection — to avoid
                    # triggering is_full_step on every subsequent step.
                    # A rejected anchor simply means we skip this one and try again
                    # batch_size steps later (redo log guarantees recoverability).
                    log_based_callback._async_anchor_last_attempted = self.state.global_step
                    if accepted:
                        log_based_callback.full_anchor_save_count += 1
                        logger.info(f"[AsyncAnchor] Async anchor queued at step {self.state.global_step}")
                    else:
                        logger.info(f"[AsyncAnchor] Anchor skipped at step {self.state.global_step}, "
                                    f"next attempt at step {self.state.global_step + batch_size}")
                    # Don't clear history or update base — wait for persist completion
                else:
                    # Sync path: save model files via Trainer, then clear history
                    t0 = time.time()
                    super()._save_checkpoint(model, trial)
                    t_super_full = time.time() - t0
                    t0 = time.time()
                    log_based_callback.full_anchor_save_count += 1
                    # Update callback state
                    log_based_callback.base_checkpoint_path = output_dir
                    log_based_callback.base_checkpoint_step = self.state.global_step
                    log_based_callback._base_pending_seed = log_based_callback._pending_seed
                    with log_based_callback.update_lock:
                        log_based_callback.update_history = []
                    if log_metadata is not None:
                        log_metadata['base_checkpoint'] = output_dir
                        log_metadata['current_step'] = self.state.global_step
                        log_metadata['base_pending_seed'] = log_based_callback._base_pending_seed
                        log_metadata['is_full_checkpoint'] = True
                        meta_path = os.path.join(output_dir, LOG_METADATA_NAME)
                        t_meta_start = time.time()
                        torch.save(log_metadata, meta_path)
                        from .log_based_utils import _fsync_file
                        _fsync_file(meta_path)
                        t_meta_save = time.time() - t_meta_start
                        logger.info(
                            f"[ZOTrainer] Saved lightweight full-step sidecar metadata to {LOG_METADATA_NAME}, "
                            f"pending_grad={log_metadata['pending_grad']:.6e}, is_full_step=True"
                        )
                    t_post_full = time.time() - t0
                    logger.info(f"[ZOTrainer] Full checkpoint at step {self.state.global_step}, cleared update history")
                logger.info(
                    f"[ZOTrainer SaveBreakdown] step={self.state.global_step} "
                    f"state_json={t_state_json:.3f}s scheduler={t_scheduler:.3f}s "
                    f"rng={t_rng:.3f}s build_optimizer={t_opt_build:.3f}s "
                    f"save_optimizer={t_opt_save:.3f}s save_meta={t_meta_save:.3f}s "
                    f"super_full={t_super_full:.3f}s "
                    f"post_full={t_post_full:.3f}s total_save_impl={time.time() - t_save_breakdown_start:.3f}s"
                )
            # Log checkpoint step: only optimizer.pt + metadata saved above, no model files created
            return

        # Default behavior: save full checkpoint when log-based checkpointing is disabled
        super()._save_checkpoint(model, trial)
        # Fsync full checkpoint directory for L0 baseline
        from .log_based_utils import _fsync_directory
        run_dir = self._get_output_dir(trial=trial)
        ckpt_dir = os.path.join(run_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        _fsync_directory(ckpt_dir)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        effective_resume_checkpoint = resume_from_checkpoint
        if effective_resume_checkpoint is None and self._log_based_resume_checkpoint:
            effective_resume_checkpoint = self._log_based_resume_checkpoint

        rng_state_available = False
        if effective_resume_checkpoint is not None:
            rng_state_available = (
                os.path.isfile(os.path.join(effective_resume_checkpoint, "rng_state.pth"))
                or os.path.isfile(os.path.join(effective_resume_checkpoint, f"rng_state_{self.args.process_index}.pth"))
            )

        # Restore checkpoint RNG before creating the train dataloader. Some sampler
        # implementations derive their internal generator state when the dataloader
        # is constructed, so restoring only before the epoch iterator is too late.
        if rng_state_available:
            self._load_rng_state(effective_resume_checkpoint)
            logger.info(
                "[LogBased Resume] Restored RNG before creating train dataloader "
                f"from {effective_resume_checkpoint}"
            )
        else:
            # Fresh runs still need a deterministic starting point.
            set_seed(self.args.seed)

        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        # ZO2 added ->
        # model = self._wrap_model(self.model_wrapped)
        model = self.model

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        # ZO2 added ->
        use_accelerator_prepare = False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ZO2 added ->
        if delay_optimizer_creation:
            if self.zo:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps, model=model)
            else:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # ZO2 added ->
        # Check if saved optimizer or scheduler states exist
        if self.zo:
            _, model = self._load_optimizer_and_scheduler(effective_resume_checkpoint, model)
        else:
            self._load_optimizer_and_scheduler(effective_resume_checkpoint)

        # # ckpt loading
        # if resume_from_checkpoint is not None:
        #     if self.is_deepspeed_enabled:
        #         deepspeed_load_checkpoint(
        #             self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
        #         )
        #     elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
        #         self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # # Check if saved optimizer or scheduler states exist
        # self._load_optimizer_and_scheduler(resume_from_checkpoint)
        # self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None
        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )
        elif self._log_based_resume_state is not None:
            resumed_global_step = int(self._log_based_resume_state.get("global_step", 0))
            self.state.global_step = resumed_global_step
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % num_update_steps_per_epoch
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0
            logger.info(
                "[LogBased Resume] Injected continuation state "
                f"with global_step={self.state.global_step}, "
                f"epoch={epochs_trained}, steps_to_skip={steps_trained_in_current_epoch}"
            )
            if not rng_state_available:
                logger.info(
                    "[LogBased Resume] No RNG state file found in checkpoint; "
                    "continuing without HF RNG-state restore"
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        # Rebuild the dataloader sampler state at the start of the resumed epoch
        # before we skip intra-epoch batches. Without this, mid-epoch resume can
        # regenerate the wrong shuffle order and continue on different samples.
        if not args.ignore_data_skip:
            is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                train_dataloader.sampler, RandomSampler
            )
            torch_pre_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")
            for epoch in range(epochs_trained):
                if hasattr(train_dataloader, "set_epoch"):
                    train_dataloader.set_epoch(epoch)
                elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                    train_dataloader.dataset.set_epoch(epoch)

                if torch_pre_1_11 or not is_random_sampler:
                    # Older PyTorch (and non-random samplers) only advance their
                    # internal state when iteration actually begins.
                    for _ in train_dataloader:
                        break
                else:
                    # RandomSampler consumes extra RNG near the end of iteration,
                    # so exhaust it to land on the same next-epoch shuffle state.
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            resume_epoch_rng_restored = False
            if epoch == epochs_trained and effective_resume_checkpoint is not None and rng_state_available:
                # Restore RNG before building the iterator so samplers/batch samplers
                # for the resumed epoch consume the correct RNG stream.
                self._load_rng_state(effective_resume_checkpoint)
                resume_epoch_rng_restored = True
                logger.info(
                    f"[LogBased Resume] Restored RNG before building epoch iterator "
                    f"(epoch={epoch}, steps_to_skip={steps_trained_in_current_epoch})"
                )

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = not resume_epoch_rng_restored

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            if args.gradient_accumulation_steps == 1:
                total_updates -= 1
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()
                    if rng_to_sync and rng_state_available:
                        self._load_rng_state(effective_resume_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0 and rng_state_available:
                            self._load_rng_state(effective_resume_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    expected_step = self.state.global_step + 1
                    self._log_batch_debug(expected_step, inputs)

                    # Log full resume time on first real training step
                    if hasattr(self, '_t_full_resume_start'):
                        now = time.time()
                        t_full_resume = now - self._t_full_resume_start
                        logger.info(f"[Full Resume] Total checkpoint resume time: {t_full_resume:.3f}s")
                        del self._t_full_resume_start
                        if hasattr(self, '_t_program_start'):
                            t_from_program = now - self._t_program_start
                            t_setup = self._t_program_start  # will be subtracted below
                            logger.info(f"[Full Resume] Total time from program start to first step: {t_from_program:.3f}s")
                            del self._t_program_start
                    elif hasattr(self, '_t_program_start'):
                        t_from_program = time.time() - self._t_program_start
                        logger.info(f"[No Resume] Total time from program start to first step: {t_from_program:.3f}s")
                        del self._t_program_start

                    # ZO2 added -> estimate gradient and updates
                    if self.zo:
                        tr_loss_step = self.zo2_training_step(model, inputs)
                    else:
                        # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                        context = (
                            functools.partial(self.accelerator.no_sync, model=model)
                            if i != len(batch_samples) - 1
                            and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                            else contextlib.nullcontext
                        )
                        with context():
                            tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # ZO2 added -> ignore parameter update since it is fuesd with model forward
                        if self.zo:
                            pass
                        else:
                            # Since we perform prefetching, we need to manually set sync_gradients to True
                            self.accelerator.gradient_state._set_sync_gradients(True)

                            # Gradient clipping
                            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                                if is_sagemaker_mp_enabled() and args.fp16:
                                    _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                                elif self.use_apex:
                                    # Revert to normal clipping otherwise, handling Apex or full precision
                                    _grad_norm = nn.utils.clip_grad_norm_(
                                        amp.master_params(self.optimizer),
                                        args.max_grad_norm,
                                    )
                                else:
                                    _grad_norm = self.accelerator.clip_grad_norm_(
                                        model.parameters(),
                                        args.max_grad_norm,
                                    )

                                if (
                                    is_accelerate_available()
                                    and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                                ):
                                    grad_norm = model.get_global_grad_norm()
                                    # In some cases the grad norm may not return a float
                                    if hasattr(grad_norm, "item"):
                                        grad_norm = grad_norm.item()
                                else:
                                    grad_norm = _grad_norm

                            self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                            self.optimizer.step()

                            self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                            # get leaning rate before update
                            learning_rate = self._get_learning_rate()

                            if not self.accelerator.optimizer_step_was_skipped:
                                # Delay optimizer scheduling until metrics are generated
                                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    self.lr_scheduler.step()

                            model.zero_grad()

                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                            learning_rate=learning_rate,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)


    def _load_optimizer_and_scheduler(self, checkpoint, model=None):
        """
        Override: handle optimizer.pt that contains zo_update_history (log-based checkpoint).
        Load the optimizer state normally, but strip out the zo metadata first.
        """
        if checkpoint is not None:
            optimizer_path = os.path.join(checkpoint, OPTIMIZER_NAME)
            metadata_path = os.path.join(checkpoint, LOG_METADATA_NAME)
            if os.path.exists(optimizer_path):
                try:
                    # Use cached optimizer state if available (avoids double loading from disk)
                    if hasattr(self, '_cached_optimizer_state') and self._cached_optimizer_state is not None:
                        opt_state = self._cached_optimizer_state
                        del self._cached_optimizer_state
                        logger.info("[ZOTrainer] Using cached optimizer state (skipped disk I/O)")
                    else:
                        opt_state = torch.load(optimizer_path, map_location='cpu', weights_only=False)
                    if isinstance(opt_state, dict) and 'zo_update_history' in opt_state:
                        logger.info("[ZOTrainer] Detected log-based checkpoint with zo_update_history")

                        # Extract and remove zo metadata, keep only standard optimizer state
                        zo_metadata_keys = ['zo_update_history', 'base_checkpoint', 'current_step',
                                           'batch_size', 'num_updates', 'tied_weights', 'model_dtype',
                                           'pending_grad', 'pending_seed', 'trainable_param_names',
                                           'zo_eps', 'rng_device', 'is_full_checkpoint', 'zo2',
                                           'zo_method', 'adam_betas', 'adam_eps_value',
                                           'base_pending_seed']
                        # Note: 'adam_state' is NOT stripped — it flows through to
                        # MeZOAdam.load_state_dict() which extracts it
                        for key in zo_metadata_keys:
                            opt_state.pop(key, None)

                        # Load the cleaned optimizer state
                        if self.optimizer is not None:
                            self.optimizer.load_state_dict(opt_state)
                            logger.info("[ZOTrainer] Loaded optimizer state from checkpoint (zo metadata stripped)")

                        # Load scheduler
                        scheduler_path = os.path.join(checkpoint, SCHEDULER_NAME)
                        if os.path.exists(scheduler_path) and self.lr_scheduler is not None:
                            self.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location='cpu'))
                            logger.info("[ZOTrainer] Loaded scheduler state from checkpoint")

                        if self.zo and model is not None:
                            model.opt = self.optimizer
                            return None, model
                        return None
                except Exception as e:
                    logger.warning(f"[ZOTrainer] Failed to load optimizer state: {e}, falling through to default")
                    pass  # Fall through to default loading
            elif os.path.exists(metadata_path):
                try:
                    scheduler_path = os.path.join(checkpoint, SCHEDULER_NAME)
                    if os.path.exists(scheduler_path) and self.lr_scheduler is not None:
                        self.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location='cpu'))
                        logger.info("[ZOTrainer] Loaded scheduler state from lightweight redo checkpoint")
                    if self.zo and model is not None:
                        model.opt = self.optimizer
                        return None, model
                    return None
                except Exception as e:
                    logger.warning(
                        f"[ZOTrainer] Failed to load scheduler from lightweight redo checkpoint: {e}, "
                        "falling through to default"
                    )

        output = super()._load_optimizer_and_scheduler(checkpoint)
        if self.zo and model is not None:
            model.opt = self.optimizer
            return output, model
        return output

    def create_optimizer_and_scheduler(self, num_training_steps: int, model: nn.Module=None):
        """
        disable the optimizer but leave the learning rate scheduler.
        """
        if not self.zo:
            self.create_optimizer()
            if IS_SAGEMAKER_MP_POST_1_10 and smp.state.cfg.fp16:
                # If smp >= 1.10 and fp16 is enabled, we unwrap the optimizer
                optimizer = self.optimizer.optimizer
            else:
                optimizer = self.optimizer
        else:
            if model is None:
                optimizer = self.optimizer = self.model.opt
            else:
                optimizer = self.optimizer = model.opt
        self.create_scheduler(num_training_steps, optimizer)

    def _move_model_to_device(self, model, device):
        pass

    #*********************** zo2 functions ***********************#

    def _zo2_unsupported_conditions(self, args):
        if args.gradient_accumulation_steps > 1:
            raise NotImplementedError
        if args.n_gpu > 1:
            raise NotImplementedError("Currently ZO2 only support one working device")
        if args.deepspeed:
            raise NotImplementedError
        if is_sagemaker_mp_enabled():
            raise NotImplementedError
        if args.torch_compile:
            raise NotImplementedError

    def register_zo2_training_step_pre_hook(self, hook_fn):
        """
        example:
            def print_zo_info(model, inputs):
                tqdm.write("projected grad: {}".format(model.opt.projected_grad))
                return model, inputs
            trainer = ZOTrainer(...)
            trainer.register_zo2_training_step_pre_hook(print_zo_info)
        """
        self.zo2_training_step_pre_hooks.append(hook_fn)

    def register_zo2_training_step_post_hook(self, hook_fn):
        """
        example:
            def drop_invalid_data(model, inputs, loss):
                # Extract projected_grad, handle both tensor and scalar cases
                projected_grad = model.opt.projected_grad
                if isinstance(projected_grad, torch.Tensor):
                    projected_grad_is_nan = torch.isnan(projected_grad).any()
                else:
                    projected_grad_is_nan = projected_grad != projected_grad  # Check for NaN in scalars
                if torch.isnan(loss) or projected_grad_is_nan:
                    tqdm.write("'loss': {} or 'projected_grad': {} is nan. Drop this step.".format(
                        loss, model.opt.projected_grad
                    ))
                    model.opt.projected_grad = 0  # Reset projected_grad to prevent parameter updates
                return model, inputs, loss
            trainer = ZOTrainer(...)
            trainer.register_zo2_training_step_post_hook(drop_invalid_data)
        """
        self.zo2_training_step_post_hooks.append(hook_fn)

    def zo2_training_step(self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        step = self.state.global_step + 1
        mem_debug = self._memory_debug_due(step)
        if mem_debug:
            self._log_memory_debug("step_start", step, reset_peak=True)
        if self.zo2_training_step_pre_hooks != []:
            for pre_hook_fn in self.zo2_training_step_pre_hooks:
                model, inputs = pre_hook_fn(model, inputs)
        if mem_debug:
            self._log_memory_debug("after_pre_hooks", step)
        model.zo_train()
        inputs = self._prepare_inputs(inputs)
        if mem_debug:
            self._log_memory_debug("after_prepare_inputs", step)
        loss = model(**inputs)
        if mem_debug:
            self._log_memory_debug("after_model_forward", step)
        model.zo_eval()
        if self.zo2_training_step_post_hooks != []:
            for post_hook_fn in self.zo2_training_step_post_hooks:
                model, inputs, loss = post_hook_fn(model, inputs, loss)
        if mem_debug:
            self._log_memory_debug("after_post_hooks", step)
        return loss
    
