# coding=utf-8
# Copyright 2023  Bofeng Huang

# NB
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.
# Need to call this before importing transformers.
# from vigogne.model.llama_flash_attn_monkey_patch import replace_attn_with_flash_attn

# replace_attn_with_flash_attn()

import logging
import os
from typing import Any

import torch
from transformers.trainer_utils import get_last_checkpoint, set_seed

from vigogne.utils import configure_logging, load_model, load_tokenizer, merge_lora, prepare_datasets, setup_trainer

logger = logging.getLogger(__name__)


def train(cfg: Any):
    # Setup logging
    configure_logging(cfg)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(cfg.output_dir) and cfg.do_train and not cfg.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(cfg.output_dir)
        if last_checkpoint is None and len(os.listdir(cfg.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({cfg.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and cfg.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model
    set_seed(cfg.seed)

    # Load tokenizer
    tokenizer = load_tokenizer(cfg)

    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(cfg, tokenizer)

    # Load model
    model = load_model(cfg, tokenizer)

    # Setup trainer
    trainer = setup_trainer(cfg, model, tokenizer, train_dataset, eval_dataset)

    checkpoint = None
    if cfg.resume_from_checkpoint is not None:
        checkpoint = cfg.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model()
    # model.save_pretrained(cfg.output_dir)
    # tokenizer.save_pretrained(cfg.output_dir)

    if cfg.do_merge_lora and cfg.adapter in ["lora", "qlora"] and trainer.is_world_process_zero():
        # clear memory
        del model
        torch.cuda.empty_cache()

        # merge lora weights
        merge_lora(cfg)
