#! /usr/bin/env python
# coding=utf-8

"""
Translate Stanford Alpaca data by the API of OpenAI.

Usage:
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

python scripts/translate_alpaca.py \
    --input_json_file data/alpaca_data_cleaned.json \
    --output_json_file data/vigogne_data_cleaned.json \
    --failed_output_json_file data/vigogne_data_cleaned_failed.json \
    --model gpt-3.5-turbo \
    --max_parallel_requests 16
"""

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import fire
import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

# Replace 'your_api_key' with your actual API key
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.organization = os.getenv("OPENAI_ORG")

lck = threading.Lock()


def safe_write(item, output_file, mode="a", ensure_ascii=False):
    with open(output_file, mode) as outf:
        with lck:
            outf.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")


TRANSLATION_PROMPT = """Please translate the following text from English into French without providing any explanation, while maintaining the format and ensuring faithful translation. It is crucial that the translation work is done accurately. The imperative sentence should be translated using the informal address.
Following is the text to translate:
"""

TASK_TRANSLATION_PROMPT = """You are asked to translate the following example task from English into French without providing any explanation.

Here are the requirements:
1. Translate the instruction text, the output text, and the input text if it exists.
2. Ensure faithful translation, and keep the correctness of the example. Translate the instruction using the informal address.
3. Maintain the format, keep the "instruction", "output" and the "input" keywords if they exist in the example.
4. If the task is to correct grammar mistakes or spelling mistakes in the input, you have to generate another input in French that has a similar mistake and then generate a corrected output.
5. If the task is to translate some text from one to another language, you only translate the instruction into French and keep the text and the output in their original language.
6. Don't translate the code, including its syntax, and variable names.

"""

TASK_TRANSLATION_PROMPT_DICT = {
    "prompt_no_input": TASK_TRANSLATION_PROMPT + 'instruction: "{instruction}"\n\noutput:"{output}"',
    "prompt_input": TASK_TRANSLATION_PROMPT + 'instruction: "{instruction}"\n\ninput: "{input}"\n\noutput: "{output}"',
}

INFERENCE_PROMPT = """Soit vous recevrez une instruction avec une entrée, soit seulement une instruction, et vous devrez fournir la réponse appropriée en sortie sans explication.

"""

INFERENCE_PROMPT_DICT = {
    "prompt_no_input": INFERENCE_PROMPT + 'instruction: "{instruction}"',
    "prompt_input": INFERENCE_PROMPT + 'instruction: "{instruction}"\n\nentrée:"{input}"',
}

# todo
translation_blacklist = [
    # generate words that rhyme with others
    "rhyme",
    "rhymes",
    # reverse string order
    "reverse the",
    "reverse it",
    "reverse each",
    "reverse this",
    "reverse these",
    "reverse order",
    "its reverse",
]


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


def generate_prompt(prompt_template, item):
    return (
        prompt_template["prompt_input"].format_map(item)
        if item["input"]
        else prompt_template["prompt_no_input"].format_map(item)
    )


def generate_messages(prompt):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "user", "content": f"Translate the following text from English to French: '{value}'"},
        # {"role": "user", "content": f"Please translate the following text from English into French without providing any explanation, while maintaining the format and ensuring faithful translation. It is crucial that the translation work is done accurately.\nThis is the text:\n> {value}"},
        {"role": "user", "content": prompt},
    ]


# Add exponential backoff to mitigate openai.error.RateLimitError
# See: https://platform.openai.com/docs/guides/rate-limits/error-mitigation
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_endpoint(messages, model="gpt-3.5-turbo"):
    # print(messages)
    # todo: customize gen args
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        temperature=0,  # greedy
        logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
    )
    return response.choices[0]["message"]["content"].strip()


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def post_process_response(output_str):
    # splitted_output = re.split("(?:instruction|input|output)\s*:\s*", output_str, flags=re.IGNORECASE)
    # splitted_output = re.split(r"(?:(?<=^)|(?<=\n))(?:instruction|input|output)\s*:\s*", output_str, flags=re.IGNORECASE)
    splitted_output = re.split(r"(?:(?<=^)|(?<=\n))(?:instruction|input|output)\s*:\s*(?=\")", output_str, flags=re.IGNORECASE)
    # print(output_str)
    # print(splitted_output)

    if len(splitted_output) not in [3, 4]:
        # skip
        print(output_str)
        return {"output_str": output_str}

    inst = splitted_output[1].strip(' \n"')
    out = splitted_output[-1].strip(' \n"')
    inp = ""
    if len(splitted_output) == 4:
        inp = splitted_output[2].strip(' \n"')

    return {"instruction": inst, "input": inp, "output": out}


def process_item(item, output_file, failed_output_file=None, model="gpt-3.5-turbo"):

    do_translation = not any(find_word_in_string(word, item["instruction"]) for word in translation_blacklist)

    if do_translation:
        request_messages = generate_messages(generate_prompt(TASK_TRANSLATION_PROMPT_DICT, item))
        translated_item_str = call_endpoint(request_messages, model)
        translated_item = post_process_response(translated_item_str)

        # skip this
        # save into another file for all failed processed examples
        if "instruction" not in translated_item:
            failed_item = {"id": item["id"], "original_instruction": item["instruction"], **translated_item}
            if failed_output_file is not None:
                safe_write(failed_item, failed_output_file)
            return failed_item
    else:
        translated_item = {}

        inst = call_endpoint(generate_messages(TRANSLATION_PROMPT + item["instruction"]))
        translated_item["instruction"] = inst.strip(' \n"')

        inp = call_endpoint(generate_messages(TRANSLATION_PROMPT + item["input"])) if item["input"] else ""
        translated_item["input"] = inp.strip(' \n"')

        request_messages = generate_messages(generate_prompt(INFERENCE_PROMPT_DICT, translated_item))
        out = call_endpoint(request_messages, model)
        translated_item["output"] = re.sub(r"^(?:sortie|traitement)\s*:\s*", "", out, flags=re.IGNORECASE).strip(' \n"')

    final_item = {"id": item["id"], "original_instruction": item["instruction"]}
    final_item.update(translated_item)

    safe_write(final_item, output_file)

    return final_item


def process_data(
    input_json_file: str,
    output_json_file: str,
    failed_output_json_file: Optional[str] = None,
    model="gpt-3.5-turbo",
    max_parallel_requests: int = 16,
):
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
        futures = {
            executor.submit(process_item, item, output_json_file, failed_output_json_file, model): item for item in data
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
            translated_data.append(future.result())

    # translated_data = [x for x in translated_data if x is not None]

    # Save the translated data to a new JSON file named 'translated_data.json'
    # with open(output_json_file, "w") as f:
    #     json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(f"Translation complete. The translated data is saved in {output_json_file}")


if __name__ == "__main__":
    fire.Fire(process_data)
