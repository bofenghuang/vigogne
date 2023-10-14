# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Training callbacks."""

import contextlib
import logging
import os

import torch
from peft import set_peft_model_state_dict
from peft.utils import WEIGHTS_NAME as PEFT_WEIGHTS_NAME
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import WEIGHTS_NAME as TRANSFORMERS_WEIGHTS_NAME

logger = logging.getLogger(__name__)


# Deprecated
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


# Copied and modified from https://github.com/bigcode-project/starcoder/blob/main/finetune/finetune.py#L35
class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.best_model_checkpoint is None:
            logger.error("Failed to load the best peft model")
            return control
        logger.info(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_adapter_model_path = os.path.join(state.best_model_checkpoint, PEFT_WEIGHTS_NAME)
        adapters_weights = torch.load(best_adapter_model_path)
        set_peft_model_state_dict(kwargs["model"], adapters_weights)
        return control
