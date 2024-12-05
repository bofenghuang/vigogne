#! /usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

import fire
from typing import Optional

# import random
from datasets import load_dataset

# from vigogne.file_utils import jsonl_dump, jsonl_load


def main(
    input_file: str,
    output_train_file: str,
    output_test_file: Optional[str] = None,
    num_train_samples: Optional[str] = None,
    num_test_samples: Optional[str] = None,
    train_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
):
    # Filter out None values and check the count of non-None parameters
    non_none_parameters = list(filter(None.__ne__, [num_train_samples, num_test_samples, train_ratio, test_ratio]))
    # Assert that only one parameter is not None
    assert (
        len(non_none_parameters) == 1
    ), "Exactly one of num_train_samples, num_test_samples, train_ratio, or test_ratio must be set"

    dataset = load_dataset("json", data_files=input_file)["train"]
    print(f"Loaded {dataset.num_rows:,d} examples")

    # convert to ClassLabel column
    # dataset = dataset.class_encode_column(stratifed_column)

    if num_train_samples is not None:
        train_ratio = num_train_samples / dataset.num_rows
    elif num_test_samples is not None:
        test_ratio = num_test_samples / dataset.num_rows

    processed_dataset = dataset.train_test_split(
        train_size=train_ratio,
        test_size=test_ratio,
        # stratify_by_column=stratifed_column,
        shuffle=True,
    )

    processed_dataset["train"].to_json(output_train_file, orient="records", lines=True, force_ascii=False)
    print(f'Saved {processed_dataset["train"].num_rows:,d} examples into {output_train_file}')

    if output_test_file is not None:
        processed_dataset["test"].to_json(output_test_file, orient="records", lines=True, force_ascii=False)
        print(f'Saved {processed_dataset["test"].num_rows:,d} examples into {output_test_file}')


if __name__ == "__main__":
    fire.Fire(main)
