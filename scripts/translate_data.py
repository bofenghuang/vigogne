#! /usr/bin/env python
# coding=utf-8

"""
Translate Stanford Alpaca data by the API of OpenAI.

Modified from: https://github.com/22-hours/cabrita/blob/main/scripts/translate_data.py
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import fire
import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

# Replace 'your_api_key' with your actual API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )


def generate_messages(value):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "user", "content": f"Translate the following text from English to French: '{value}'"},
        {
            "role": "user",
            "content": f"""Please translate the following text from English into French without providing any explanation, while maintaining the format and ensuring faithful translation. It is crucial that the translation work is done accurately.
This is the text:
> {value}""",
        },
    ]


# Add exponential backoff to mitigate openai.error.RateLimitError
# See: https://platform.openai.com/docs/guides/rate-limits/error-mitigation
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_endpoint(messages, model="gpt-3.5-turbo"):
    # todo: customize gen args
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        temperature=0,  # greedy
        logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
    )
    return response.choices[0]["message"]["content"].strip()


def process_item(item, model="gpt-3.5-turbo"):
    # Remove when <nooutput> in instructed output
    if "nooutput" in item.get("output", "").lower():
        return None

    translated_item = {}
    for key, value in item.items():
        if value:
            translated_value = call_endpoint(generate_messages(value), model)
            translated_item[key] = translated_value
        else:
            translated_item[key] = ""
    return translated_item


def estimate_num_tokens_per_item(item, model):
    # Remove when <nooutput> in instructed output
    if "nooutput" in item.get("output", "").lower():
        return 0

    total_tokens = 0
    for _, value in item.items():
        if value:
            total_tokens += num_tokens_from_messages(generate_messages(value), model)
    return total_tokens


def process_data(input_json_file: str, output_json_file: str, model="gpt-3.5-turbo", max_parallel_requests: int = 32):
    # Assuming the input JSON is in a file named 'input.json'
    with open(input_json_file, "r") as f:
        data = json.load(f)

    # debug
    # start = 0
    # end = 10_000
    # data = data[start:end]
    # data = data[end:]

    translated_data = []
    with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
        futures = {executor.submit(process_item, item, model): item for item in data}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
            translated_data.append(future.result())

    translated_data = [x for x in translated_data if x is not None]

    # Save the translated data to a new JSON file named 'translated_data.json'
    with open(output_json_file, "w") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(f"Translation complete. The translated data is saved in {output_json_file}")


def estimate_price(input_json_file: str, ratio_output_input=1.0, model="gpt-3.5-turbo-0301", price_per_thousand_tokens=0.002):
    # Assuming the input JSON is in a file named 'input.json'
    with open(input_json_file, "r") as f:
        data = json.load(f)

    # debug
    # start = 0
    # end = 10_000
    # data = data[start:end]

    prompt_tokens = [estimate_num_tokens_per_item(item, model) for item in tqdm(data, desc="Counting")]
    prompt_tokens = sum(prompt_tokens)
    total_tokens = int(prompt_tokens * (1 + ratio_output_input))
    print(f"prompt tokens: {prompt_tokens:,}")
    print(f"Estimated total tokens: {total_tokens:,}")
    print(f"Estimated price: ${total_tokens / 1000 * price_per_thousand_tokens:,.2f}")


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
