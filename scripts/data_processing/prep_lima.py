#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import fire
from datasets import load_dataset

from vigogne.data_utils import Instruct
from vigogne.preprocess import INSTRUCT_SYSTEM_MESSAGE_EN


def main(output_file):
    raw_dataset = load_dataset("GAIR/lima")["train"]
    print(raw_dataset)

    # debug
    # raw_dataset = raw_dataset.select(range(10))

    # 1000 of 1030 are single turn
    processed_dataset = raw_dataset.filter(lambda example: len(example["conversations"]) == 2)
    print(processed_dataset)

    def process_function(example):
        # return {
        #     "instruction": example["conversations"][0],
        #     "input": "",
        #     "output": example["conversations"][1],
        # }
        return Instruct(
            system=INSTRUCT_SYSTEM_MESSAGE_EN,
            instruction=example["conversations"][0],
            input="",
            output=example["conversations"][1],
        ).model_dump()

    processed_dataset = processed_dataset.map(process_function, num_proc=4, remove_columns=processed_dataset.column_names)
    print(processed_dataset)

    processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"The processed data is saved into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
