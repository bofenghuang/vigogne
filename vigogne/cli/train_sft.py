#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import logging

from transformers import HfArgumentParser

from vigogne.train_sft import train
from vigogne.utils import VigogneTrainingArguments

logger = logging.getLogger(__name__)


def main():
    # def train(args):
    # HF parser
    parser = HfArgumentParser(VigogneTrainingArguments)
    (training_args,) = parser.parse_args_into_dataclasses()
    # debug
    # (training_args,) = parser.parse_args_into_dataclasses(args=args)

    train(training_args)


if __name__ == "__main__":
    main()
