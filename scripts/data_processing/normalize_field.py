#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

"""Normalize translated text"""

import re

import fire
from tqdm import tqdm

from vigogne.file_utils import jsonl_dump, jsonl_load


def normalize_text(s):
    s = re.sub(r"[’´′ʼ‘ʻ`]", "'", s)  # replace special quote
    s = re.sub(r"[−‐]", "-", s)  # replace special dash

    # mt model added space before $
    s = re.sub(r"(?<=\d)\s+(?=\$)", "", s)

    return s


def main(input_file, output_file, field_names):
    data = jsonl_load(input_file)
    print(f"Loaded {len(data)} examples from {input_file}")

    for item in tqdm(data):
        for field_name in field_names:
            item[field_name] = normalize_text(item[field_name])

    jsonl_dump(data, output_file, mode="w")
    print(f"Saved {len(data)} examples into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
