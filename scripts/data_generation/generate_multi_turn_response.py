#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang


# import os
# import re
# import sys
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import fire
from datasets import load_dataset

# from tqdm import tqdm

# from vigogne.file_utils import jsonl_load, thread_safe_jsonl_dump


def process_item(
    item: Dict,
    id_field: str = "id",
    messages_field: str = "messages",
    model: str = "gpt-4",
    **kwargs,
):

    return {
        "custom_id": item[id_field],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "messages": item[messages_field],
            "model": model,
            # "max_tokens": kwargs.get("max_tokens", 1024),
            # "temperature": kwargs.get("temperature", 0.7),
            **kwargs,
        },
    }


def main(
    input_file: str,
    output_file: str,
    id_field: str = "id",
    messages_field: str = "messages",
    max_samples: Optional[int] = None,
    model: str = "gpt-4",
    # max_parallel_requests: int = 4,
    **kwargs,
):
    dataset = load_dataset("json", data_files=input_file, split="train")
    print(f"Loaded {dataset.num_rows:,d} examples from {input_file}")

    if max_samples is not None:
        dataset = dataset.select(range(max_samples))
        print(f"Sampled the first {dataset.num_rows:,d} examples")

    # debug
    # dataset = dataset.select(range(10))

    # if id_field not in dataset.column_names:
    #     dataset = dataset.map(lambda _, idx: {id_field: f"{idx:09d}"}, with_indices=True, num_proc=8)

    dataset = dataset.map(
        process_item,
        fn_kwargs={
            "id_field": id_field,
            "messages_field": messages_field,
            "model": model,
            **kwargs,
        },
        num_proc=8,
        remove_columns=dataset.column_names,
        # remove_columns=[c for c in dataset.column_names if c not in [id_field]],
    )

    dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
