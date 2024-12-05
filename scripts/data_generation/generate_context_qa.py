#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Invoke OpenAI's API to generate question and response based on provided context.

Usage:
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

python scripts/data_generation/generate_context_qa.py \
    --input_document_file /path/to/context/jsonl/file \
    --input_prompt_file data/generation/context_qa_direct.txt \
    --output_file /path/to/output/jsonl/file \
    --id_field id \
    --max_samples 5000 \
    --model gpt-4-0125-preview \
    --max_tokens 4096 \
    --max_parallel_requests 2
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import fire
from datasets import Dataset, load_dataset
from get_api_answer import set_global_api
from tqdm import tqdm

from vigogne.file_utils import thread_safe_jsonl_dump

# generate_api_messages, call_endpoint, process_api_response = None, None, None


def deduplicate_dataset(ds: Dataset, field_name: str, shuffle: bool = True):
    def get_hash(example):
        """Get hash of content field."""
        return {"hash": hash(example[field_name])}  # can use any hashing function here

    def check_uniques(example, uniques):
        """Check if current hash is still in set of unique hashes and remove if true."""
        # if example["hash"] in uniques:
        #     uniques.remove(example["hash"])
        if example[field_name] in uniques:
            uniques.remove(example[field_name])
            return True
        else:
            return False

    if shuffle:
        ds = ds.shuffle(seed=10)
    # ds = ds.map(get_hash, num_proc=16)
    # uniques = set(ds.unique("hash"))
    uniques = set(ds.unique(field_name))
    filtered_ds = ds.filter(check_uniques, fn_kwargs={"uniques": uniques})
    # filtered_ds = filtered_ds.remove_columns("hash")
    return filtered_ds


def preprocess(documents):
    _template = "Document: {idx}\nTitle: {title}\nText: {text}"
    # random.shuffle(documents)
    return "\n\n".join([_template.format(idx=i + 1, **doc) for i, doc in enumerate(documents)])


def process_item(
    item: Dict,
    prompt_template: str,
    output_file: str,
    id_field: str = "id",
    context_field: str = "text",
    response_field: str = "output",
    model: str = "gpt-4",
    **kwargs,
):
    # todo
    # -- contextqa
    instruction = prompt_template.format(context=item[context_field])
    # instruction = prompt_template.format(context=preprocess(item[context_field]))
    # instruction = prompt_template.format(input=item[context_field])
    # -- grade prompt
    # instruction = prompt_template.format(question=item[context_field])
    # instruction = re.sub(r"\{input\}", item[context_field], prompt_template)
    # instruction = re.sub(r"\{input\}", item[context_field].encode("latin1").decode("unicode-escape"), prompt_template)

    request_messages = generate_api_messages(instruction)
    # print(request_messages)
    # print(request_messages[-1]["content"])
    # quit()

    # return {
    #     "custom_id": item[id_field],
    #     "method": "POST",
    #     "url": "/v1/chat/completions",
    #     "body": {
    #         "messages": request_messages,
    #         "model": model,
    #         # "max_tokens": kwargs.get("max_tokens", 1024),
    #         # "temperature": kwargs.get("temperature", 0.7),
    #         **kwargs,
    #     },
    # }

    response = call_endpoint(request_messages, model, **kwargs)
    result = process_api_response(response)

    result[response_field] = result.pop("output")
    item.update(result)

    thread_safe_jsonl_dump(item, output_file, mode="a")

    return item


def main(
    input_document_file: str,
    input_prompt_file: str,
    output_file: str,
    id_field: str = "id",
    context_field: str = "text",
    response_field: str = "output",
    max_samples: Optional[int] = None,
    model: str = "gpt-4",
    max_parallel_requests: int = 4,
    **kwargs,
):
    global generate_api_messages
    global call_endpoint
    global process_api_response

    with open(input_prompt_file) as f:
        prompt = f.read()

    dataset = load_dataset("json", data_files=input_document_file, split="train")
    print(f"Loaded {dataset.num_rows:,d} contexts from {input_document_file}")

    # dedup by wikipedia id
    dataset = deduplicate_dataset(dataset, id_field)
    print(f"Dedup to {dataset.num_rows:,d} contexts")

    # tmp filter
    """
    data_files = [
        "/projects/bhuang/corpus/text/llm/documents/wikipedia_fr/tmp_wikipedia_fr_chunked_256_2048_contextqa_direct_responded_gpt4turbo.jsonl",
        "/projects/bhuang/corpus/text/llm/documents/wikipedia_fr/tmp_wikipedia_fr_chunked_256_2048_contextqa_reasoning_responded_gpt4turbo.jsonl",
        "/projects/bhuang/corpus/text/llm/documents/wikipedia_fr/tmp_wikipedia_fr_chunked_256_2048_contextqa_limited_responded_gpt4turbo.jsonl",
    ]
    existing_dataset = load_dataset("json", data_files=data_files, split="train")
    existing_values = existing_dataset.unique(id_field)
    existing_values = set(existing_values)
    # print(f"Found {len(existing_values):,d} existing examples in {output_file}")

    dataset = dataset.filter(lambda x: x not in existing_values, input_columns=id_field, num_proc=4)
    print(f"Filtered to {dataset.num_rows:,d} examples")
    """

    if max_samples is not None:
        # data = data[:max_samples]
        dataset = dataset.select(range(max_samples))
        print(f"Sampled the first {dataset.num_rows:,d} contexts")

    # debug
    # data = data[:10]
    # dataset = dataset.select(range(10))

    if os.path.exists(output_file):
        existing_dataset = load_dataset("json", data_files=output_file, split="train")
        existing_values = existing_dataset.unique(id_field)
        existing_values = set(existing_values)
        print(f"Found {len(existing_values):,d} existing examples in {output_file}")

        dataset = dataset.filter(lambda x: x not in existing_values, input_columns=id_field, num_proc=4)
        print(f"Filtered to {dataset.num_rows:,d} examples")

    # set up api
    generate_api_messages, call_endpoint, process_api_response = set_global_api(model)

    # dataset = dataset.map(
    #     process_item,
    #     fn_kwargs={
    #         "prompt_template": prompt,
    #         "output_file": output_file,
    #         "id_field": id_field,
    #         "context_field": context_field,
    #         "response_field": response_field,
    #         "model": model,
    #         **kwargs,
    #     },
    #     num_proc=max_parallel_requests,
    #     remove_columns=dataset.column_names,
    #     # remove_columns=[c for c in dataset.column_names if c not in [id_field]],
    # )
    # dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    # quit()

    start_time = time.perf_counter()

    translated_data = []
    with tqdm(total=dataset.num_rows, desc="Generating") as pbar:
        with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
            futures = {
                executor.submit(
                    process_item,
                    item,
                    prompt,
                    output_file,
                    id_field,
                    context_field,
                    response_field,
                    model,
                    **kwargs,
                ): item
                for item in dataset
            }
            for future in as_completed(futures):
                translated_data.append(future.result())
                pbar.update(1)

    print(
        f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}. The generated"
        f" data is saved in {output_file}"
    )


if __name__ == "__main__":
    fire.Fire(main)
