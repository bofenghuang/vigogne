#!/usr/bin/env python
# coding=utf-8
# Copyright  2023 Bofeng Huang


"""
Modified from: https://github.com/lm-sys/FastChat/blob/main/fastchat/data/optional_clean.py

1. Download sharegpt 90k dataset and combine
wget https://huggingface.co/datasets/RyokoAI/ShareGPT52K/resolve/main/sg_90k_part1.json
wget https://huggingface.co/datasets/RyokoAI/ShareGPT52K/resolve/main/sg_90k_part2.json

2. Clean
# Requirement: pip install markdownify==0.11.6 bs4
python -m fastchat.data.clean_sharegpt --in data/sg_90k_all.json --out data/sg_90k_all_cleaned.json

3. Split long conversations
python -m fastchat.data.split_long_conversation --in data/sg_90k_all_cleaned.json --out data/sg_90k_all_cleaned_splitted.json --model-name-or-path huggyllama/llama-7b

4. Filter by language and other staffs (this script)
# Requirement: pip install polyglot icu pyicu pycld2
python vigogne/data/prep_sharegpt.py data/sg_90k_all_cleaned_splitted.json

"""

import collections
import re
from collections import Counter
from pathlib import Path

import fire
import polyglot
import pycld2
from datasets import Dataset, load_dataset
from polyglot.detect import Detector
from polyglot.detect.base import logger as polyglot_logger
from tqdm import tqdm

from vigogne.constants import ASSISTANT, CONTENT, CONVERSATION, ROLE, USER
from vigogne.data.utils import jdump, jload, jsonl_dump

polyglot_logger.setLevel("ERROR")

role_mappings = {
    "human": USER,
    # "user": USER,
    "gpt": ASSISTANT,
    # "chatgpt": ASSISTANT,
    # "bing": ASSISTANT,
    # "assistant": ASSISTANT,
    # "bard": ASSISTANT,
}


def detect_language_simple(s):
    try:
        return Detector(s).language.code
    except (pycld2.error, polyglot.detect.base.UnknownLanguage):
        return "unknown"


def detect_language(example):
    concatenated_text = " ".join([turn[CONTENT] for turn in example[CONVERSATION]])
    example["lang"] = detect_language_simple(concatenated_text)
    return example


def convert_format(example):
    example[CONVERSATION] = [
        {ROLE: role_mappings[turn["from"]], CONTENT: turn["value"]}
        for turn in example.pop("conversations")
        if turn["from"] not in ["system"]
    ]
    return example


def process_function(example):
    try:
        example = convert_format(example)
        example = detect_language(example)
        return example
    except Exception as e:
        print(str(e))


def filter_function(example, validated_languages):
    if example is None:
        return False

    if example["lang"] not in validated_languages:
        return False

    # Remove repetitive numbers
    if any(bool(re.search(r"(\d)\1{8}", turn[CONTENT])) for turn in example[CONVERSATION]):
        return False

    if example[CONVERSATION][0][ROLE] != USER:
        return False

    return True


def main(input_file, validated_languages=["en", "fr"]):
    # raw_dataset = load_dataset("RyokoAI/ShareGPT52K")
    # raw_dataset = load_dataset("json", data_files=f"{input_dir}/sg_90k_part*.json")
    # print(raw_dataset)
    # data = []
    # for p in Path(input_dir).glob("sg_90k_part*.json"):
    #     data.extend(jload(p.as_posix()))
    data = jload(input_file)
    print(f"Loaded {len(data)} examples")
    # jdump(data, f"{input_dir}/sg_90k_all.json")
    # quit()

    # debug
    # data = data[:10]

    processed_data = list(map(process_function, tqdm(data, desc="process data")))
    processed_data = [
        example for example in tqdm(processed_data, desc="filter data") if filter_function(example, validated_languages)
    ]
    print(f"Filtered to {len(processed_data)} examples")

    processed_data_by_lang = collections.defaultdict(list)
    for example in processed_data:
        processed_data_by_lang[example["lang"]].append(example)

    for lang_, data_ in processed_data_by_lang.items():
        # output_file = f"{output_dir}/sharegpt90k_{lang_}.jsonl"
        output_file = f"{input_file.rsplit('.', 1)[0]}_{lang_}.jsonl"
        jsonl_dump(data_, output_file, mode="w")
        # output_file = f"{input_file.rsplit('.', 1)[0]}_{lang_}.json"
        # jdump(data_, output_file, mode="w")
        print(f"Saved {len(data_)} examples into {output_file}")

    # print(Counter([example["lang"] for example in processed_data]))
    # jsonl_dump(processed_data, output_file, mode="w")
    # print(f"Saved {len(processed_data)} examples into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
