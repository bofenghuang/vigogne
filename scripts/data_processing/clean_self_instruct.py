#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

"""Filter unwanted examples from self-instruct datasets."""

import re

import fire
from datasets import load_dataset


def filter_function(s):
    # remove all summarization task: only titles or web page or short articles are given
    if re.search(r"résumé|résume|article", s, flags=re.IGNORECASE):
        return False
    if re.search(r"\bhttps\b", s):
        return False
    if re.search(r"préféré", s, flags=re.IGNORECASE):
        return False
    if re.search(r"donner|recommande|recommandation", s, flags=re.IGNORECASE) and re.search(
        r"restaurant|resto", s, flags=re.IGNORECASE
    ):
        return False

    return True


def main(input_file, output_file, instruction_field="instruction"):
    dataset = load_dataset("json", data_files=input_file, split="train")
    print(f"Loaded {dataset.num_rows:,d} examples from {input_file}")

    processed_dataset = dataset.filter(filter_function, input_columns=instruction_field, num_proc=8)
    print(f"Filtered to {processed_dataset.num_rows:,d} examples")

    # export
    processed_dataset = processed_dataset.shuffle(10)
    processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"Saved data into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
