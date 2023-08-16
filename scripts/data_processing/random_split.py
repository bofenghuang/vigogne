#! /usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

import fire

# import random
from datasets import load_dataset

# from vigogne.file_utils import jsonl_dump, jsonl_load


def main(input_file, sampled_output_file, remaining_output_file=None, num_samples=5_000):
    # data = jsonl_load(input_file)
    # random.shuffle(data)
    # jsonl_dump(data[:num_samples], sampled_output_file, mode="w")
    # jsonl_dump(data[num_samples:], remaining_output_file, mode="w")
    # print(f"Saved {len(data[:num_samples])} examples into {sampled_output_file}")
    # print(f"Saved {len(data[num_samples:])} examples into {remaining_output_file}")

    raw_dataset = load_dataset("json", data_files=input_file)["train"]
    print(f"Loaded {raw_dataset.num_rows:,d} examples")

    processed_dataset = raw_dataset.train_test_split(train_size=num_samples / raw_dataset.num_rows, shuffle=True)

    processed_dataset["train"].to_json(sampled_output_file, orient="records", lines=True, force_ascii=False)
    print(f'Saved {processed_dataset["train"].num_rows:,d} examples into {sampled_output_file}')

    if remaining_output_file is not None:
        processed_dataset["test"].to_json(remaining_output_file, orient="records", lines=True, force_ascii=False)
        print(f'Saved {processed_dataset["test"].num_rows:,d} examples into {remaining_output_file}')


if __name__ == "__main__":
    fire.Fire(main)
