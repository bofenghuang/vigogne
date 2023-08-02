#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Generate Orca's style instruction-following examples.

Usage:
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

python scripts/data_generation/generate_responses.py \
    --input_json_file data/instruct/openorca_gpt4_1m_translated.jsonl \
    --output_json_file data/instruct/openorca_gpt4_1m_translated_completed.jsonl \
    --system_field system_prompt \
    --instruction_field translated_question \
    --response_field fr_response \
    --model gpt-4 \
    --max_parallel_requests 1 \
    --max_samples 1
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import fire
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from vigogne.file_utils import jsonl_load, thread_safe_jsonl_dump

# Replace 'your_api_key' with your actual API key
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.organization = os.getenv("OPENAI_ORG")


def generate_messages(prompt: str, system_message: str = "You are a helpful assistant."):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]


# Add exponential backoff to mitigate openai.error.RateLimitError
# See: https://platform.openai.com/docs/guides/rate-limits/error-mitigation
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_endpoint(messages: List[Dict], model: str = "gpt-4"):
    # todo: customize gen args
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=1024,  # todo
        # temperature=0,  # greedy
        # logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
    )
    # print(response)
    return response.choices[0]["message"]["content"]


def post_process(output_str):
    # todo
    pass


def process_item(
    item: Dict,
    output_file: str,
    existing_instructions: List[str] = [],
    system_field: str = "system",
    instruction_field: str = "instruction",
    response_field: str = "output",
    model: str = "gpt-4",
):
    input_text = item[instruction_field]
    if input_text in existing_instructions:
        return

    generated_output_str = call_endpoint(generate_messages(input_text, system_message=item[system_field]), model)
    # print(generate_messages(input_text, system_message=item[system_field]))
    # print(generated_output_str)
    # quit()

    # generated_output_str = post_process(generated_output_str)
    if generated_output_str is None:
        return

    item[response_field] = generated_output_str
    # print(item)

    thread_safe_jsonl_dump(item, output_file, mode="a")

    return item


def process_data(
    input_json_file: str,
    output_json_file: str,
    system_field: str = "system",
    instruction_field: str = "instruction",
    response_field: str = "output",
    max_samples: Optional[int] = None,
    model: str = "gpt-4",
    max_parallel_requests: int = 4,
):
    data = jsonl_load(input_json_file, "r")
    print(f"Loaded {len(data)} instructions from {input_json_file}")

    # dedup by instruction
    seen = set()
    data = [
        example for example in data if not (example[instruction_field] in seen or seen.add(example[instruction_field]))
    ]
    print(f"Deduped to {len(data)} instructions")

    if max_samples is not None:
        data = data[:max_samples]
        print(f"Sampled the first {max_samples} instructions")

    # debug
    # start = 0
    # end = 10_000
    # data = data[start:end]
    # data = data[end:]

    existing_instructions = []
    if os.path.exists(output_json_file):
        existing_data = jsonl_load(output_json_file, "r")
        existing_instructions = [existing_example[instruction_field] for existing_example in existing_data]
        print(f"Found {len(existing_instructions)} existing instruction-following examples in {output_json_file}")

    start_time = time.perf_counter()

    translated_data = []
    with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
        futures = {
            executor.submit(
                process_item,
                item,
                output_json_file,
                existing_instructions,
                system_field,
                instruction_field,
                response_field,
                model,
            ): item
            for item in data
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            translated_data.append(future.result())

    # translated_data = [x for x in translated_data if x is not None]

    # Save the translated data to a new JSON file named 'translated_data.json'
    # with open(output_json_file, "w") as f:
    #     json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(
        f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}. The generated data is saved in {output_json_file}"
    )


if __name__ == "__main__":
    fire.Fire(process_data)
