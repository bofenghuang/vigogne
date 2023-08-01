#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

"""Convert alpaca instruct-following data to conversation format."""

from functools import partial

import fire

from vigogne.data_utils import Conversation, Role, Utterance
from vigogne.file_utils import jsonl_dump, jsonl_load
from vigogne.preprocess import merge_instruction_and_input


def convert_to_chat(example):
    conversation = Conversation(
        messages=[
            Utterance(role=Role.user, content=merge_instruction_and_input(example["instruction"], example["input"])),
            Utterance(role=Role.assistant, content=example["output"])
        ]
    )
    return conversation.fully_model_dump()


def main(input_file, output_file):
    data = jsonl_load(input_file)
    reformatted_data = list(map(convert_to_chat, data))
    jsonl_dump(reformatted_data, output_file, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
