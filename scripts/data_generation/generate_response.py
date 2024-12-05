#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Invoke OpenAI's API to generate responses based on provided instructions.
With the option to customize the system message in the style of Orca.

Usage:
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

python scripts/data_generation/generate_response.py \
    --input_file data/instruct/openorca_gpt4_1m_translated.jsonl \
    --output_file data/instruct/openorca_gpt4_1m_translated_completed.jsonl \
    --system_field system_prompt \
    --instruction_field translated_question \
    --response_field response_on_translated_question \
    --model gpt-4 \
    --max_parallel_requests 1 \
    --max_samples 1
"""

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import fire
from datasets import load_dataset
from tqdm import tqdm

from vigogne.file_utils import jsonl_load, thread_safe_jsonl_dump
# from vigogne.data.get_api_answer import set_global_api
from get_api_answer import set_global_api

# generate_api_messages, call_endpoint, process_api_response = None, None, None


def process_item(
    item: Dict,
    output_file: str,
    id_field: str = "id",
    system_field: str = "system",
    instruction_field: str = "instruction",
    response_field: str = "output",
    model: str = "gpt-4",
    **kwargs,
):
    gen_kwargs = {}
    # if (system_message := item.get(system_field)) is not None:
    # None or empty
    if (system_message := item.get(system_field, "")):
        gen_kwargs["system_message"] = system_message

    request_messages = generate_api_messages(item[instruction_field], **gen_kwargs)

    return {
        "custom_id": item[id_field],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "messages": request_messages,
            "model": model,
            # "max_tokens": kwargs.get("max_tokens", 1024),
            # "temperature": kwargs.get("temperature", 0.7),
            **kwargs,
        },
    }

    response = call_endpoint(request_messages, model, **kwargs)
    result = process_api_response(response)

    result[response_field] = result.pop("output")
    item.update(result)

    thread_safe_jsonl_dump(item, output_file, mode="a")

    return item


def main(
    input_file: str,
    output_file: str,
    id_field: str = "id",
    system_field: str = "system",
    instruction_field: str = "instruction",
    response_field: str = "output",
    max_samples: Optional[int] = None,
    model: str = "gpt-4",
    max_parallel_requests: int = 4,
    **kwargs,
):
    global generate_api_messages
    global call_endpoint
    global process_api_response

    # data = jsonl_load(input_file)
    dataset = load_dataset("json", data_files=input_file, split="train")
    print(f"Loaded {dataset.num_rows:,d} examples from {input_file}")

    if id_field not in dataset.column_names:
        dataset = dataset.map(lambda _, idx: {id_field: f"{idx:09d}"}, with_indices=True, num_proc=8)

    # dedup by instruction
    # seen = set()
    # data = [example for example in data if not (example[instruction_field] in seen or seen.add(example[instruction_field]))]
    # print(f"Deduped to {len(data):,d} instructions")

    if max_samples is not None:
        # data = data[:max_samples]
        dataset = dataset.select(range(max_samples))
        print(f"Sampled the first {dataset.num_rows:,d} examples")

    # debug
    # data = data[:10]
    # dataset = dataset.select(range(10))

    if os.path.exists(output_file):
        # existing_data = jsonl_load(output_file)
        # existing_instructions = {existing_example[instruction_field] for existing_example in existing_data}
        # print(f"Found {len(existing_instructions):,d} existing examples in {output_file}")

        # data = [example for example in data if example[instruction_field] not in existing_instructions]
        # print(f"Filtered to {len(data):,d} examples")
        existing_dataset = load_dataset("json", data_files=output_file, split="train")
        existing_values = existing_dataset.unique(instruction_field)
        existing_values = set(existing_values)
        print(f"Found {len(existing_values):,d} existing examples in {output_file}")

        dataset = dataset.filter(lambda x: x not in existing_values, input_columns=instruction_field, num_proc=4)
        print(f"Filtered to {dataset.num_rows:,d} examples")

    # set up api
    generate_api_messages, call_endpoint, process_api_response = set_global_api(model)

    dataset = dataset.map(
        process_item,
        fn_kwargs={
            "output_file": output_file,
            "id_field": id_field,
            "system_field": system_field,
            "instruction_field": instruction_field,
            "response_field": response_field,
            "model": model,
            **kwargs,
        },
        num_proc=max_parallel_requests,
        remove_columns=dataset.column_names,
        # remove_columns=[c for c in dataset.column_names if c not in [id_field]],
    )
    dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    quit()

    start_time = time.perf_counter()

    translated_data = []
    # with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
    #     futures = {
    #         executor.submit(
    #             process_item,
    #             item,
    #             output_file,
    #             system_field,
    #             instruction_field,
    #             response_field,
    #             model,
    #             **kwargs,
    #         ): item
    #         for item in data
    #     }

    #     for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
    #         translated_data.append(future.result())

    with tqdm(total=dataset.num_rows, desc="Genrating") as pbar:
        with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
            futures = {
                executor.submit(
                    process_item,
                    item,
                    output_file,
                    id_field,
                    system_field,
                    instruction_field,
                    response_field,
                    model,
                    **kwargs,
                ): item
                for item in dataset
            }
            for future in as_completed(futures):
                translated_data.append(future.result())
                pbar.update(1)

    # translated_data = [x for x in translated_data if x is not None]

    # Save the translated data to a new JSON file named 'translated_data.json'
    # with open(output_file, "w") as f:
    #     json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(
        f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}. The generated"
        f" data is saved in {output_file}"
    )


if __name__ == "__main__":
    fire.Fire(main)
