#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import fire
from datasets import load_dataset

from vigogne.constants import ASSISTANT, CONTENT, CONVERSATION, ROLE, USER


def main(output_file):
    raw_dataset = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k")["train"]
    print(raw_dataset)

    # processed_dataset = raw_dataset.filter(lambda example: len(example["conversations"]) == 2)
    # print(processed_dataset)

    def process_function(example):
        return {
            "instruction": example["conversations"][0]["value"],
            "input": "",
            "output": example["conversations"][1]["value"],
        }

    # debug
    # raw_dataset = raw_dataset.select(range(10))

    processed_dataset = raw_dataset.map(process_function, num_proc=16, remove_columns=raw_dataset.column_names)
    print(processed_dataset)

    processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"The processed data is saved into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
