#!/usr/bin/env python
# coding=utf-8

"""
Modified from: https://github.com/tatsu-lab/stanford_alpaca/blob/main/generate_instruction.py

Usage:
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

# num_instructions_to_generate is by worker
python vigogne/data/generate_instruction.py \
    --seed_tasks_path data/instruct/seed_tasks_vigogne.jsonl \
    --prompt_path data/instruct/prompt_vigogne.txt \
    --output_file data/instruct/self_instruct_data.jsonl \
    --num_instructions_to_generate 1 \
    --n_workers 1
"""

import json
import os
import random
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor

import fire
import tqdm

import vigogne.data.generation_utils as generation_utils
from vigogne.data.utils import thread_safe_jsonl_dump


french_chars_lower = "a-zàâäéèêëîïôöùûüÿçñ"
french_chars_upper = "A-ZÀÂÄÇÈÉÊËÎÏÔÖÙÛÜŸ"
number_chars = "0-9"
allowed_chars = french_chars_lower + french_chars_upper + number_chars
pattern_allowed_chars = re.compile(rf"[{allowed_chars}]")


def encode_prompt(prompt_instructions, prompt_path):
    """Encode multiple prompt instructions into a single string."""
    prompt = open(prompt_path).read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<vide>" if input.lower() == "" else input
        prompt += "###\n"
        prompt += f"{idx + 1}. Instruction : {instruction}\n"
        prompt += f"{idx + 1}. Entrée :\n{input}\n"
        prompt += f"{idx + 1}. Réponse :\n{output}\n"
    prompt += "###\n"
    prompt += f"{idx + 2}. Instruction :"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response, api):
    if response is None:
        return []

    if api == "chat":
        raw_instructions = response["message"]["content"]
        # debug
        print("\nAPI Output\n" + "=" * 50)
        print(raw_instructions)
        print("=" * 50 + "\n")
        # if 'Instruction:' not in raw_instructions[0: 10]:
        raw_instructions = f"{num_prompt_instructions+1}. Instruction :" + raw_instructions
    elif api == "completion":
        raw_instructions = f"{num_prompt_instructions+1}. Instruction :" + response["text"]

    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue

        idx += num_prompt_instructions + 1

        splitted_data = re.split(f"{idx}\.\s+(Instruction|Entrée|Réponse)\s*:", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<vide>" else input
            output = splitted_data[6].strip()

        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue

        # filter based on keywords that are not suitable for language models.
        # todo
        blacklist = [
            "image",
            "images",
            "graphique",
            "graphiques",
            "fichier",
            "fichiers",
            "carte",
            "cartes",
            "dessiner",
            "tracer",
            "aller à",
            "vidéo",
            "audio",
            "musique",
            "organigramme",
            "diagramme",
            # other
            "http",
            "https",
            "OpenAI",
            "chatgpt",
            "gpt-3",
            "gpt-3.5",
            "gpt-4",
        ]
        # blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            print(f"Found invalid word in: {inst}")
            continue

        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue

        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue

        # filter those starting with non-english character
        # if not inst[0].isascii():
        if not bool(pattern_allowed_chars.match(inst[0])):
            print(f"First character is invalid: {inst}")
            continue

        instructions.append({"instruction": inst, "input": input, "output": output})

    # print(json.dumps(instructions, indent=4, ensure_ascii=False))
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    seed_tasks_path: str = "./data/seed_tasks_vigogne.jsonl",
    prompt_path: str = "./data/prompt_vigogne.txt",
    output_file: str = "./data/machine_generated_data.jsonl",
    num_instructions_to_generate: int = 100,
    num_prompt_instructions: int = 3,
    # api: str = "completion",
    # model_name: str = "text-davinci-003",
    # request_batch_size: int= 5,
    api: str = "chat",
    model_name: str = "gpt-3.5-turbo",
    request_batch_size: int = 1,  # chatgpt only support 1
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)

    n_machine_instruction_data = 0
    while n_machine_instruction_data < num_instructions_to_generate:
        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions, prompt_path)
            batch_inputs.append(prompt)

        decoding_args = generation_utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=2048,
            # max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=["\n20", "20.", "20."],
        )

        # request_start = time.time()
        results = generation_utils.openai_completion(
            prompts=batch_inputs,
            api=api,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        # request_duration = time.time() - request_start

        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result, api)
            instruction_data += new_instructions

        thread_safe_jsonl_dump(instruction_data, output_file)
        n_machine_instruction_data += len(instruction_data)
        progress_bar.update(len(instruction_data))


def main(n_workers, **kwargs):
    with ThreadPoolExecutor() as executor:
        # future = executor.submit(generate_instruction_following_data, **kwargs)
        _ = [executor.submit(generate_instruction_following_data, **kwargs) for _ in range(n_workers)]


if __name__ == "__main__":
    fire.Fire(main)
