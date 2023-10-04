#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Translate instructions using OpenAI's API.

Usage:
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

python scripts/data_generation/translate_alpaca.py \
    --input_file data/alpaca_data_cleaned.jsonl \
    --output_file data/alpaca_data_cleaned_fr.jsonl \
    --model gpt-3.5-turbo \
    --max_parallel_requests 16
"""

import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import fire
import openai
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from vigogne.file_utils import jsonl_load, thread_safe_jsonl_dump

# Replace 'your_api_key' with your actual API key
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.organization = os.getenv("OPENAI_ORG")


TRANSLATION_TEMPLATES = [
    'Translate the following text into French:\n\n"{input}"\n\nTranslate the text above into French, ensuring a faithful translation while preserving the original format, without providing any explanations. Please translate the imperative sentence using the informal subject "tu".',
    'Translate the following text into French:\n\n"{input}"\n\nTranslate the text above into French, ensuring a faithful translation while preserving the original format, without providing any explanations.',
]


def generate_prompt(input_str):
    return random.choice(TRANSLATION_TEMPLATES).format(input=input_str)


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
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 1024,
    temperature: float = 0.7,
):
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
        # "model": response["model"],
        "finish_reason": response.choices[0]["finish_reason"],
        # "prompt_tokens": response["usage"]["prompt_tokens"],
        # "completion_tokens": response["usage"]["completion_tokens"],
        # "total_tokens": response["usage"]["total_tokens"],
    }

    item.update(parsed_response)

    if parsed_response["finish_reason"] == "length":
        print("max_tokens reached")

    return item


def process_item(
    item: Dict,
    output_file: str,
    column_name: str,
    model: str = "gpt-3.5-turbo",
    **kwargs,
):
    request_messages = generate_messages(generate_prompt(item[column_name]))
    response = call_endpoint(request_messages, model, **kwargs)
    final_item = post_process_response(item, response, f"translated_{column_name}")

    thread_safe_jsonl_dump(final_item, output_file, mode="a")

    return final_item


def process_data(
    input_file: str,
    output_file: str,
    column_name: str,
    max_samples: Optional[int] = None,
    model: str = "gpt-3.5-turbo",
    max_parallel_requests: int = 16,
    **kwargs,
):
    # dataset = jsonl_load(input_file)
    dataset = load_dataset("json", data_files=input_file, split="train")
    print(f"Loaded {dataset.num_rows:,d} examples from {input_file}")

    if max_samples is not None:
        # dataset = dataset[:max_samples]
        dataset = dataset.select(range(max_samples))
        print(f"Sampled the first {dataset.num_rows:,d} examples")

    # debug
    # dataset = dataset[:10]
    # dataset = dataset.select(range(10))

    if os.path.exists(output_file):
        # existing_data = jsonl_load(output_file)
        # existing_values = {existing_example[column_name] for existing_example in existing_data}
        existing_dataset = load_dataset("json", data_files=output_file, split="train")
        existing_values = existing_dataset.unique(column_name)
        existing_values = set(existing_values)
        print(f"Found {len(existing_values):,d} existing examples in {output_file}")

        dataset = dataset.filter(lambda x: x not in existing_values, input_columns=column_name, num_proc=4)
        print(f"Filtered to {dataset.num_rows:,d} examples")

    start_time = time.perf_counter()

    translated_data = []
    # with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
    #     futures = {
    #         executor.submit(
    #             process_item,
    #             item,
    #             output_file,
    #             column_name,
    #             model,
    #             **kwargs,
    #         ): item
    #         for item in dataset
    #     }

    #     for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
    #         translated_data.append(future.result())

    with tqdm(total=dataset.num_rows, desc="Translating") as pbar:
        with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
            futures = {
                executor.submit(
                    process_item,
                    item,
                    output_file,
                    column_name,
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
        f"Translation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}. The translated data is saved into {output_file}"
    )


if __name__ == "__main__":
    fire.Fire(process_data)
