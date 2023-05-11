#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

from functools import partial

import fire

from vigogne.constants import ASSISTANT, CONTENT, CONVERSATION, ID, ROLE, USER
from vigogne.data.utils import jsonl_dump, jsonl_load
from vigogne.preprocess import merge_instruction_and_input


# todo: prepend alpaca prompt ?
def convert_to_chat(example):
    return {
        # ID: f"{task_id_prefix}-{example_idx:08d}",
        CONVERSATION: [
            {
                ROLE: USER,
                CONTENT: merge_instruction_and_input(example["instruction"], example["input"]),
            },
            {ROLE: ASSISTANT, CONTENT: example["output"]},
        ],
    }


def main(input_file, output_file):
    data = jsonl_load(input_file)
    reformatted_data = list(map(convert_to_chat, data))
    jsonl_dump(reformatted_data, output_file, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
