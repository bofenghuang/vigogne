#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import logging
from typing import List, Optional

import fire
from transformers import HfArgumentParser

from vigogne.data_utils import DECODER, SUPPORTED_MODEL_TYPES
from vigogne.train_sft import train
from vigogne.utils import VigogneSeq2SeqTrainingArguments, VigogneTrainingArguments

logger = logging.getLogger(__name__)


def main(model_type: str = DECODER):
    assert model_type in SUPPORTED_MODEL_TYPES, f"Specified model_type {model_type} doesn't exist in {SUPPORTED_MODEL_TYPES}"
    TrainingArgs = VigogneTrainingArguments if model_type == DECODER else VigogneSeq2SeqTrainingArguments

    # Parse args
    parser = HfArgumentParser(TrainingArgs)
    (training_args,) = parser.parse_args_into_dataclasses()

    training_args.model_type = model_type

    train(training_args)


def debug(model_type: str = DECODER, args: Optional[List[str]] = None):
    assert model_type in SUPPORTED_MODEL_TYPES, f"Specified model_type {model_type} doesn't exist in {SUPPORTED_MODEL_TYPES}"
    TrainingArgs = VigogneTrainingArguments if model_type == DECODER else VigogneSeq2SeqTrainingArguments

    # Parse args
    parser = HfArgumentParser(TrainingArgs)
    (training_args,) = parser.parse_args_into_dataclasses(args=args)

    training_args.model_type = model_type

    train(training_args)


if __name__ == "__main__":
    fire.Fire(main)
