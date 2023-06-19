#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import fire
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial

from vigogne.preprocess import get_chat_example_length, get_instruct_example_length


def main(input_file, model_name_or_path, mode, example_max_length, output_file):
    raw_dataset = load_dataset("json", data_files=input_file)["train"]
    print(raw_dataset)

    # debug
    # raw_dataset = raw_dataset.select(range(10))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
        use_fast=False,
    )

    if mode == "instruct":
        get_example_length_function = get_instruct_example_length
    elif mode == "chat":
        get_example_length_function = get_chat_example_length
    else:
        raise ValueError(f"Invalid value for mode: {mode}")

    get_example_length_function_p = partial(get_example_length_function, tokenizer=tokenizer)

    processed_dataset = raw_dataset.map(
        get_example_length_function_p,
        num_proc=16,
        # remove_columns=raw_datasets.column_names,
        # load_from_cache_file=True,
        desc="get example lengths",
    )

    processed_dataset = processed_dataset.filter(lambda example: example["example_length"] <= example_max_length)
    processed_dataset = processed_dataset.remove_columns("example_length")
    print(processed_dataset)

    processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"The processed data is saved into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
