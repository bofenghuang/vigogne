#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import contextlib
import logging
import os
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from peft import set_peft_model_state_dict
from peft.utils import WEIGHTS_NAME as PEFT_WEIGHTS_NAME
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import WEIGHTS_NAME as TRANSFORMERS_WEIGHTS_NAME

from vigogne.constants import CONFIG_FILE_ARG

logger = logging.getLogger(__name__)


def parse_kwargs(**kwargs):
    config_file = kwargs.pop(CONFIG_FILE_ARG, None)

    parsed_kwargs = {}
    if config_file is not None and os.path.exists(config_file):
        parsed_kwargs = yaml.safe_load(Path(config_file).read_text())
        logger.info(f"Config has been loaded from {config_file}")

    # override file args with sys.argv
    parsed_kwargs.update(kwargs)

    return parsed_kwargs


# Modified from Peft
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    param_by_dtype = defaultdict(int)

    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

        param_by_dtype[param.dtype] += num_params

    logger.info(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    )
    for k, v in param_by_dtype.items():
        logger.info(f"dtype: {k} || num: {v:,d} || percentage: {100 * v / all_param:.4f}%")


# See https://github.com/tloen/alpaca-lora/pull/359
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        # save adapter model
        kwargs["model"].save_pretrained(checkpoint_folder)

        # todo
        # pytorch_model.bin is same to adapter_model.bin
        pytorch_model_path = os.path.join(checkpoint_folder, TRANSFORMERS_WEIGHTS_NAME)
        # if os.path.exists(pytorch_model_path):
        with contextlib.suppress(FileNotFoundError):
            os.remove(pytorch_model_path)

        return control


# Copied from https://github.com/bigcode-project/starcoder/blob/main/finetune/finetune.py#L35
class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        logger.info(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_adapter_model_path = os.path.join(state.best_model_checkpoint, PEFT_WEIGHTS_NAME)
        adapters_weights = torch.load(best_adapter_model_path)
        set_peft_model_state_dict(kwargs["model"], adapters_weights)
        return control
