#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from time import sleep

import requests
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Response Generation Manager.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="We will support more models in the future.",
    )
    parser.add_argument("--model_configs_file", type=str, default=None, help="")
    parser.add_argument("--prompt_file", type=str, default=None, help="")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--id_column_name", type=str, default="id", help="")
    parser.add_argument("--instruct_column_name", type=str, default="instruction", help="")
    parser.add_argument("--output_column_name", type=str, default="output", help="")
    parser.add_argument("--max_samples", type=int, default=None, help="")
    parser.add_argument("--output_file", type=str, default=None, help="")

    parser.add_argument("--api_url", type=str, default="https://api.together.xyz/v1/chat/completions", help="API URL")
    parser.add_argument("--api_key", type=str, default=None, help="Together API Key")

    # Generation Parameters
    parser.add_argument("--engine", default="vllm", type=str, choices=["vllm", "hf", "together"])
    # parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism. Only used for Llama 70B models.",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Number of samples per batch")
    parser.add_argument("--max_parallel_requests", type=int, default=4, help="")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--tokenizer_template", type=bool, default=False, help="Use tokenizer template for generating the response."
    )
    parser.add_argument("--use_tokenizer_template", action="store_true", dest="tokenizer_template")

    return parser.parse_args()

# API Setups
# if args.engine == "together":
#     # Change name for API (Together Naming Convention)
#     if MODEL_NAME == "meta-llama/Meta-Llama-3-8B-Instruct":
#         api_model_name = "meta-llama/Llama-3-8b-chat-hf"
#     elif MODEL_NAME == "meta-llama/Meta-Llama-3-70B-Instruct":
#         api_model_name = "meta-llama/Llama-3-70b-chat-hf"
#     else:
#         api_model_name = MODEL_NAME

#     # Constants for the API
#     API_ENDPOINT = args.api_url
#     API_HEADERS = {
#         "Authorization": args.api_key,
#     }
#     API_PARAMS = {
#         "model": api_model_name,
#         "max_tokens": args.max_tokens,
#         "temperature": args.temperature,
#         "top_p": args.top_p,
#         "repetition_penalty": args.repetition_penalty,
#         "stop": stop_tokens,
#     }


def write_dataset_to_json(dataset, output_file, mode="w", encoding="utf-8", default=str, ensure_ascii=False):
    with open(output_file, mode, encoding=encoding) as fo:
        for sample in dataset:
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")


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


# Process a batch of data using the API
# def process_batch_with_api(batch, args=None):
#     # with concurrent.futures.ThreadPoolExecutor() as executor:
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         future_to_item = {
#             executor.submit(
#                 make_api_request_with_retry,
#                 [{"content": item["instruction"], "role": "user"}],
#                 API_PARAMS,
#                 API_ENDPOINT,
#                 API_HEADERS,
#             ): item
#             for item in batch
#         }

#         for future in concurrent.futures.as_completed(future_to_item):
#             item = future_to_item[future]
#             try:
#                 api_response = future.result()
#                 item["response"] = api_response.strip()
#                 item["gen_response_configs"] = {
#                     "temperature": args.temperature,
#                     "top_p": args.top_p,
#                     "repetition_penalty": args.repetition_penalty,
#                     "max_tokens": args.max_tokens,
#                     "stop_tokens": stop_tokens,
#                     "output_generator": MODEL_NAME,
#                     "engine": api_model_name,
#                 }
#             except Exception as e:
#                 print(f"Failed to process item: {item} with error: {str(e)}")
#                 item["response"] = ""

#     return batch


# Process a batch of data using local vllm engine
def process_batch(batch, llm, params, tokenizer=None, args=None):
    user_instructions = [item[args.instruct_column_name] for item in batch]

    # tmp: format instruction for extraction/grading/..
    # wrap instruction by higher-level prompts
    if args.prompt_file is not None:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt_template = f.read()

        user_instructions = [prompt_template.format(text=user_inst) for user_inst in user_instructions]

    prompts = []
    for instruction in user_instructions:
        # if not args.tokenizer_template:
        #     conv = get_conversation_template(MODEL_NAME)
        #     conv.append_message(conv.roles[0], instruction)
        #     conv.append_message(conv.roles[1], None)
        #     template = conv.get_prompt()
        # else:
        #     chat = [{"role": "user", "content": instruction}]
        #     template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        chat = [{"role": "user", "content": instruction}]
        template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(template)
    if args.engine == "vllm":
        outputs = llm.generate(prompts, params)
    elif args.engine == "hf":
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(torch.cuda.current_device())
        gen_do_sample = False if args.temperature == 0 else True
        outputs = llm.generate(
            **inputs,
            tokenizer=tokenizer,
            do_sample=gen_do_sample,
            temperature=(
                args.temperature if gen_do_sample else None
            ),  # To avoid temperature` (=0) has to be a strictly positive float
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_length=args.max_tokens,
        )
        outputs = tokenizer.batch_decode(outputs[i][len(inputs[i]):] for i in range(len(outputs)))
        # Setting stop tokens seems not working for Gemma, so we manually truncate the outputs
        for i, completion in enumerate(outputs):
            for stop_token in args.stop_tokens:
                if stop_token in completion:
                    outputs[i] = completion[: completion.index(stop_token)]

    for i, item in enumerate(batch):
        if args.engine == "vllm":
            item[args.output_column_name] = outputs[i].outputs[0].text.strip()
        elif args.engine == "hf":
            item[args.output_column_name] = outputs[i].strip()
        item["gen_configs"] = {
            "prompt": prompts[i],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_tokens": args.max_tokens,
            "stop_tokens": args.stop_tokens,
            "output_generator": args.model_name_or_path,
            "engine": args.engine,
        }
    return batch


# Generate outputs, update dataset in batches, and overwrite checkpoint
def generate_and_update(dataset, llm=None, params=None, tokenizer=None, args=None):
    # Initialize tokenizer
    if tokenizer is not None:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if "gemma-2" in args.model_name_or_path.lower():
            tokenizer.padding_side = "right"

    # Calculate total number of batches
    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

    for i in tqdm(range(num_batches)):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(dataset))
        # batch = dataset[start_idx: end_idx]
        batch = [dataset[idx] for idx in range(start_idx, end_idx)]
        # if args.engine == "together":
        #     batch = process_batch_with_api(batch)
        # else:
        #     batch = process_batch(batch, llm, params, tokenizer)
        batch = process_batch(batch, llm, params, tokenizer, args=args)

        write_dataset_to_json(batch, args.output_file, mode="a")

    return dataset


# Main function to control workflow
def main():
    args = get_args()
    print(f"Response Generation Manager. Arguments: {args}")  # For logging

    if args.input_file is None:
        raise ValueError("Please specify the input file path.")

    # load dataset
    dataset = load_dataset("json", data_files=args.input_file, split="train")
    print(f"Loaded {dataset.num_rows:,d} examples from {args.input_file}")

    # dedup
    dataset = deduplicate_dataset(dataset, args.id_column_name)
    print(f"Deduplicated to {dataset.num_rows:,d} examples")

    # take max samples
    if args.max_samples is not None:
        # data = data[:args.max_samples]
        dataset = dataset.select(range(args.max_samples))
        print(f"Sampled the first {dataset.num_rows:,d} examples")

    # remove existing samples
    if os.path.exists(args.output_file):
        existing_dataset = load_dataset("json", data_files=args.output_file, split="train")
        existing_values = existing_dataset.unique(args.id_column_name)
        existing_values = set(existing_values)
        print(f"Found {len(existing_values):,d} existing examples in {args.output_file}")

        dataset = dataset.filter(lambda x: x not in existing_values, input_columns=args.id_column_name, num_proc=8)
        print(f"Filtered to {dataset.num_rows:,d} examples")

    # load llm

    # Obtain config from configs/model_configs.json
    with open(args.model_configs_file, "r") as f:
        model_configs = json.load(f)
        # model_config = model_configs[args.model_name_or_path]
        for model_pretty_name in model_configs:
            if re.search(model_pretty_name, args.model_name_or_path, flags=re.I):
                model_config = model_configs[model_pretty_name]
                break
        print(f"Loaded {model_pretty_name} config")
        args.stop_tokens = model_config["stop_tokens"]
        args.stop_token_ids = model_config["stop_token_ids"]

    if args.engine == "together":
        print("Start together API engine...")
        llm = None
        params = None
        tokenizer = None
    elif args.engine == "vllm":
        # Set the device
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        print("Start Local vllm engine...")
        llm = LLM(
            model=args.model_name_or_path,
            dtype=args.dtype,
            trust_remote_code=True,
            max_model_len=args.max_model_len,  # limited by kv-cache
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_token_ids=args.stop_token_ids,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif args.engine == "hf":
        print("Start Hugging Face engine...")
        params = None
        # Load the model and tokenizer
        llm = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map={"": torch.cuda.current_device()},
            torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError("Invalid engine type.")

    start_time = time.perf_counter()

    updated_dataset = generate_and_update(dataset, llm, params, tokenizer, args=args)

    print(
    f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}. The generated"
    f" data is saved in {args.output_file}"
)

# Run the main function
if __name__ == "__main__":
    main()
