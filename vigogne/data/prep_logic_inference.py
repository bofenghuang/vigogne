#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import fire
from datasets import load_dataset

from vigogne.constants import ASSISTANT, CONTENT, CONVERSATION, ROLE, USER


def main(output_file):
    # raw_dataset = load_dataset("KK04/LogicInference_OA")["train"]
    raw_dataset = load_dataset("qwedsacf/grade-school-math-instructions")["train"]
    print(raw_dataset)

    # debug
    # raw_dataset = raw_dataset.select(range(10))

    processed_dataset = raw_dataset.rename_columns({"INSTRUCTION": "instruction", "RESPONSE": "output"})
    processed_dataset = processed_dataset.map(lambda _: {"input": ""}, num_proc=16)
    print(processed_dataset)

    processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"The processed data is saved into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
