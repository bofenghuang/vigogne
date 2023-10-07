#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

# NB
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.
# Need to call this before importing transformers.
# from vigogne.model.llama_flash_attn_monkey_patch import replace_attn_with_flash_attn

# replace_attn_with_flash_attn()

import logging
import os
import sys

import datasets
import torch
import transformers
from transformers import DataCollatorForSeq2Seq, HfArgumentParser, Trainer
from transformers.trainer_utils import get_last_checkpoint, set_seed

import vigogne
from vigogne.data_utils import IGNORE_INDEX
from vigogne.utils import VigogneTrainingArguments, load_model, load_tokenizer, merge_lora, prepare_datasets

logger = logging.getLogger(__name__)


def train():
    # def train(args):
    # HF parser
    parser = HfArgumentParser(VigogneTrainingArguments)
    (training_args,) = parser.parse_args_into_dataclasses()
    # debug
    # (training_args,) = parser.parse_args_into_dataclasses(args=args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # logger.setLevel(log_level)
    vigogne.utils.set_verbosity(log_level)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model
    set_seed(training_args.seed)

    # Load tokenizer
    tokenizer = load_tokenizer(training_args)

    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(training_args, tokenizer)

    # Load model
    model = load_model(training_args)

    # define collator
    # A100 is best at 64, while others at 8
    data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE_INDEX, pad_to_multiple_of=64)

    # Init trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model()
    # model.save_pretrained(training_args.output_dir)
    # tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_merge_lora and training_args.adapter in ["lora", "qlora"] and trainer.is_world_process_zero():
        # clear memory
        del model
        torch.cuda.empty_cache()
        # merge lora weights
        merge_lora(training_args)


if __name__ == "__main__":
    train()
