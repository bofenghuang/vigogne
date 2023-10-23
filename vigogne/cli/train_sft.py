#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import logging
from typing import List, Optional

from transformers import HfArgumentParser

from vigogne.data_utils import SEQ2SEQ
from vigogne.train_sft import train
from vigogne.utils import VigogneSeq2SeqTrainingArguments, VigogneTrainingArguments

logger = logging.getLogger(__name__)


def main():
    # Parse args
    parser = HfArgumentParser(VigogneTrainingArguments)
    (training_args,) = parser.parse_args_into_dataclasses()

    # todo: better handle conflict between hf argparse and fire
    if training_args.model_type == SEQ2SEQ:
        parser = HfArgumentParser(VigogneSeq2SeqTrainingArguments)
        (training_args,) = parser.parse_args_into_dataclasses()

    train(training_args)


def debug(args: Optional[List[str]] = None):
    # Parse args
    parser = HfArgumentParser(VigogneTrainingArguments)
    (training_args,) = parser.parse_args_into_dataclasses(args=args)

    # todo: better handle conflict between hf argparse and fire
    if training_args.model_type == SEQ2SEQ:
        parser = HfArgumentParser(VigogneSeq2SeqTrainingArguments)
        (training_args,) = parser.parse_args_into_dataclasses(args=args)

    train(training_args)


if __name__ == "__main__":
    main()
