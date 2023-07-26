#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

import re

import fire

from vigogne.constants import ASSISTANT, CONTENT, CONVERSATION, ID, ROLE, USER
from vigogne.data.utils import jload, jsonl_dump

ROLES = [USER, ASSISTANT]


def convert_to_chat(example):
    splitted = re.split(r"\s+\[\|(?:Human|AI)\|\]\s+", example["input"])
    if splitted[0] == "The conversation between human and AI assistant.":
        del splitted[0]
    if not splitted[-1]:
        del splitted[-1]

    assert len(splitted) % 2 == 0, f"Failed to parse the example: {splitted}"
    # print(*splitted, sep="\n")

    return {
        # ID: f"{task_id_prefix}-{example_idx:08d}",
        CONVERSATION: [{ROLE: ROLES[idx % 2], CONTENT: sentence} for idx, sentence in enumerate(splitted)]
    }


def main(input_file, output_file):
    data = jload(input_file)
    # print(f"Load {len(data)} examples from {input_file}")
    # print(convert_to_chat(data[0]))
    reformatted_data = list(map(convert_to_chat, data))
    jsonl_dump(reformatted_data, output_file, mode="w")
    print(f"{len(reformatted_data)} reformatted examples are saved to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
