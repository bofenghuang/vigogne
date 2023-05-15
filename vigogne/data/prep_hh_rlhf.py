#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import re

import fire
from datasets import load_dataset

from vigogne.constants import ASSISTANT, CONTENT, CONVERSATION, ROLE, USER

role_mappings = {
    "Human": USER,
    "Assistant": ASSISTANT,
}


def process_function(example):
    splitted_data = re.split(r"\s+(Human|Assistant):\s+", example["chosen"], flags=re.IGNORECASE)
    splitted_data = splitted_data[1:] if not splitted_data[0] else splitted_data
    assert len(splitted_data) % 2 == 0, print(splitted_data)

    example[CONVERSATION] = [
        {ROLE: role_mappings[splitted_data[i]], CONTENT: splitted_data[i + 1]} for i in range(0, len(splitted_data), 2)
    ]
    return example


def main(output_file):
    raw_dataset = load_dataset("Anthropic/hh-rlhf")["train"]
    print(raw_dataset)

    # debug
    raw_dataset = raw_dataset.select(range(10))

    processed_dataset = raw_dataset.map(process_function, num_proc=8, remove_columns=raw_dataset.column_names)
    print(processed_dataset)

    processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"The processed data is saved into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
