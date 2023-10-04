#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Generate Orca's style instruction-following examples.

Usage:
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

python scripts/data_generation/generate_responses.py \
    --input_file data/instruct/openorca_gpt4_1m_translated.jsonl \
    --output_file data/instruct/openorca_gpt4_1m_translated_completed.jsonl \
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
from typing import Any, Dict, List, Optional

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
def call_endpoint(
    messages: List[Dict],
    model: str = "gpt-4",
    max_tokens: int = 1024,
    temperature: float = 0.7,
):
    # print(locals())
    # quit()

    return openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        # logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
    )


def post_process_response(item: Dict, response: Any, response_field: str = "output"):
    parsed_response = {
        response_field: response.choices[0]["message"]["content"],
        # "created": response["created"],
        "model": response["model"],
        "finish_reason": response.choices[0]["finish_reason"],
        "prompt_tokens": response["usage"]["prompt_tokens"],
        "completion_tokens": response["usage"]["completion_tokens"],
        "total_tokens": response["usage"]["total_tokens"],
    }

    item.update(parsed_response)

    if parsed_response["finish_reason"] == "length":
        print("max_tokens reached")

    return item


def process_item(
    item: Dict,
    output_file: str,
    system_field: str = "system",
    instruction_field: str = "instruction",
    response_field: str = "output",
    model: str = "gpt-4",
    **kwargs,
):
    request_messages = generate_messages(item[instruction_field], system_message=item[system_field])
    response = call_endpoint(request_messages, model, **kwargs)
    final_item = post_process_response(item, response, response_field)

    thread_safe_jsonl_dump(final_item, output_file, mode="a")

    return final_item


def process_data(
    input_file: str,
    output_file: str,
    system_field: str = "system",
    instruction_field: str = "instruction",
    response_field: str = "output",
    max_samples: Optional[int] = None,
    model: str = "gpt-4",
    max_parallel_requests: int = 4,
    **kwargs,
):
    data = jsonl_load(input_file)
    print(f"Loaded {len(data):,d} instructions from {input_file}")

    # dedup by instruction
    # seen = set()
    # data = [example for example in data if not (example[instruction_field] in seen or seen.add(example[instruction_field]))]
    # print(f"Deduped to {len(data):,d} instructions")

    if max_samples is not None:
        data = data[:max_samples]
        print(f"Sampled the first {max_samples:,d} instructions")

    # debug
    # data = data[:10]

    if os.path.exists(output_file):
        existing_data = jsonl_load(output_file)
        existing_instructions = {existing_example[instruction_field] for existing_example in existing_data}
        print(f"Found {len(existing_instructions):,d} existing examples in {output_file}")

        data = [example for example in data if example[instruction_field] not in existing_instructions]
        print(f"Filtered to {len(data):,d} examples")

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

    with tqdm(total=len(data), desc="Translating") as pbar:
        with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
            futures = {
                executor.submit(
                    process_item,
                    item,
                    output_file,
                    system_field,
                    instruction_field,
                    response_field,
                    model,
                    **kwargs,
                ): item
                for item in data
            }
            for future in as_completed(futures):
                translated_data.append(future.result())
                pbar.update(1)

    # translated_data = [x for x in translated_data if x is not None]

    # Save the translated data to a new JSON file named 'translated_data.json'
    # with open(output_file, "w") as f:
    #     json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(
        f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}. The generated data is saved in {output_file}"
    )


if __name__ == "__main__":
    fire.Fire(process_data)
