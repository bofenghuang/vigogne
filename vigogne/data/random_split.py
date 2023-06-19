#! /usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

import random

import fire

from vigogne.data.utils import jsonl_dump, jsonl_load


def main(input_file, output_file_train, output_file_test, num_test_samples=5_000):
    data = jsonl_load(input_file)

    random.shuffle(data)

    jsonl_dump(data[:num_test_samples], output_file_test, mode="w")
    jsonl_dump(data[num_test_samples:], output_file_train, mode="w")

    print(f"Saved {len(data[num_test_samples:])} examples into {output_file_train}")
    print(f"Saved {len(data[:num_test_samples])} examples into {output_file_test}")


if __name__ == "__main__":
    fire.Fire(main)
