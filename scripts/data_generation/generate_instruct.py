#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Generate instructions.

Usage:
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

"""

import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import fire
from generate_response import call_endpoint, generate_messages
from tqdm import tqdm

from vigogne.file_utils import thread_safe_jsonl_dump


def encode_prompt(
    prompt: str,
    num_examples: int = 5,
    min_batch_size: int = 1,
    max_batch_size: int = 4,
    generated_examples: List[str] = [],
):
    # todo
    splitted_data = re.split(r"</?EXAMPLES>", prompt)
    assert len(splitted_data) == 3, f"num_splits != 3\n\n{prompt}"

    examples = re.split(r"\s*Exemple[\s\d]+:\s*", splitted_data[1].strip())

    if not examples[0]:
        del examples[0]

    examples += generated_examples

    sampled_examples = random.sample(examples, num_examples)

    examples_text = "\n".join([f"Exemple {idx+1} : {example}" for idx, example in enumerate(sampled_examples)])

    splitted_data[1] = examples_text
    processed_prompt = "".join(splitted_data)

    batch_size = random.randint(min_batch_size, max_batch_size)
    processed_prompt = processed_prompt.format(batch_size=batch_size)

    return processed_prompt


def post_process_response(response: Any, prefix: str = "Prompt"):
    response_text = response.choices[0]["message"]["content"]

    splitted_text = re.split(rf"\s*{prefix}[\s\d]+:\s*", response_text)
    # print(response_text)
    # print(splitted_text)

    if not splitted_text[0]:
        del splitted_text[0]

    if response.choices[0]["finish_reason"] == "length":
        del splitted_text[-1]
        print("max_tokens reached")

    return splitted_text


def process_item(
    prompt: str,
    num_examples: int = 5,
    min_batch_size: int = 1,
    max_batch_size: int = 4,
    model: str = "gpt-3.5-turbo",
    system_message: str = "You are a helpful assistant.",
    generated_examples: List[str] = [],
    **kwargs,
):
    processed_prompt = encode_prompt(
        prompt,
        num_examples=num_examples,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        generated_examples=generated_examples,
    )
    request_messages = generate_messages(processed_prompt, system_message=system_message)
    # print(processed_prompt)
    # print(request_messages)
    # quit()
    response = call_endpoint(request_messages, model=model, **kwargs)
    processed_response = post_process_response(response)

    return processed_response


def process_data(
    input_prompt_file: str,
    output_file: str,
    output_column_name: str = "instruction",
    num_instructions_to_generate: int = 100,
    model: str = "gpt-3.5-turbo",
    **kwargs,
):
    with open(input_prompt_file) as f:
        prompt = f.read()
    # print(prompt)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    start_time = time.perf_counter()

    # now let's generate new instructions!
    progress_bar = tqdm(total=num_instructions_to_generate)

    generated_instructions = []
    while len(generated_instructions) < num_instructions_to_generate:
        new_generated_instructions = process_item(prompt, model=model, generated_examples=generated_instructions, **kwargs)

        thread_safe_jsonl_dump([{output_column_name: x} for x in new_generated_instructions], output_file, mode="a")

        generated_instructions += new_generated_instructions
        progress_bar.update(len(new_generated_instructions))

    print(
        f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}. The generated"
        f" data is saved in {output_file}"
    )


def main(max_parallel_requests: int = 1, **kwargs):
    with ThreadPoolExecutor() as executor:
        # future = executor.submit(process_data, **kwargs)
        _ = [executor.submit(process_data, **kwargs) for _ in range(max_parallel_requests)]


if __name__ == "__main__":
    fire.Fire(main)
