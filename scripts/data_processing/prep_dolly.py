#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

import fire
from datasets import load_dataset

from vigogne.preprocess import merge_instruction_and_input


def main(output_file, preprocessing_num_workers: int = 4):
    # raw_dataset = load_dataset("argilla/databricks-dolly-15k-multilingual", split="fr")
    raw_dataset = load_dataset("MBZUAI/Bactrian-X", "fr")["train"]
    # keep only dolly part
    processed_dataset = raw_dataset.filter(lambda x: "dolly" in x["id"])
    # processed_dataset = raw_dataset.filter(lambda x: "alpaca" in x["id"])
    # tmp fix
    processed_dataset = processed_dataset.filter(lambda x: x["output"] is not None)
    print(processed_dataset)

    def process_function(example):
        example["instruction"] = merge_instruction_and_input(example["instruction"], example["input"])
        return example

    processed_dataset = processed_dataset.map(process_function, num_proc=preprocessing_num_workers)
    processed_dataset = processed_dataset.remove_columns("input")
    print(processed_dataset)

    # save
    processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"The processed data is saved into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
