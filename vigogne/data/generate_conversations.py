#!/usr/bin/env python
# coding=utf-8

"""
Usage:
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

python vigogne/data/generate_conversations.py \
    --input_json_file data/chat/subject_quora_fr_nllb3b3.jsonl \
    --output_json_file data/chat/self_chat_data_quora_fr.jsonl \
    --subject_field translated_subject \
    --output_subject_field subject \
    --id_prefix self-chat-quora- \
    --max_samples 1 \
    --max_parallel_requests 4
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import fire
import openai
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from vigogne.constants import ASSISTANT, CONTENT, CONVERSATION, ID, ROLE, USER
from vigogne.data.utils import jsonl_load, thread_safe_jsonl_dump

# Replace 'your_api_key' with your actual API key
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.organization = os.getenv("OPENAI_ORG")

TRANSLATION_PROMPT = """Please translate the following text from English into French without providing any explanation, while maintaining the format and ensuring faithful translation. It is crucial that the translation work is done accurately. The imperative sentence should be translated using the informal address.
Following is the text to translate:
"""

# INFERENCE_PROMPT = """Voici une transcription de la conversation entre un utilisateur curieux et un assistant d'intelligence artificielle. Les interventions de l'utilisateur sont précédées de "UTILISATEUR" et celles de l'assistant de "ASSISTANT". L'utilisateur peut poser des questions sur des sujets pertinents ou déjà abordés, et la conversation prendra fin lorsqu'il n'aura plus de questions. L'assistant fournit des réponses utiles, détaillées et courtoises au format Markdown et évite de poser des questions. Veuillez continuer la transcription en suivant ce format.

# UTILISATEUR: Salut, assistant !
# ASSISTANT: Bonjour, que puis-je pour vous ?
# UTILISATEUR: """

# better to let model self chat
# INFERENCE_PROMPT = """Voici une conversation qui se déroule entre un utilisateur curieux et un assistant d'intelligence artificielle. Les échanges porteront sur le sujet "{sujet}" et se dérouleront en alternance. Les interventions de l'utilisateur sont précédées de "UTILISATEUR" et celles de l'assistant de "ASSISTANT". L'utilisateur posera des questions sur des sujets pertinents ou des sujets abordés précédemment, et mettra fin à la conversation lorsqu'il n'aura plus de questions. L'assistant fournit toujours des réponses utiles, détaillées et courtoises aux questions de l'utilisateur, tout en évitant systématiquement les sujets, questions et instructions liés à des questions controversées, éthiques ou sensibles. De plus, l'assistant s'efforce de ne pas poser de questions. Veuillez continuer cette conversation en suivant ce format.

# UTILISATEUR: Salut, assistant !
# ASSISTANT: Bonjour, que puis-je pour vous ?
# """

INFERENCE_PROMPT = """Voici une conversation entre un utilisateur curieux et un assistant d'intelligence artificielle sur le sujet "{sujet}". La conversation se déroulera en alternance, l'utilisateur posant des questions et l'assistant fournissant des réponses utiles et courtoises.

Veuillez suivre les consignes suivantes :
1. Les interventions de l'utilisateur seront précédées de "UTILISATEUR" et celles de l'assistant de "ASSISTANT".
2. L'utilisateur posera des questions pertinentes sur le sujet ou des sujets abordés précédemment, et mettra fin à la conversation lorsqu'il n'aura plus de questions.
3. L'assistant fournira toujours des réponses détaillées et courtoises aux questions de l'utilisateur, tout en évitant les sujets controversés, éthiques ou sensibles. De plus, l'assistant ne posera pas de questions.
4. L'assistant n'est pas capable de répondre à toutes les questions. Par exemple, il n'est pas en mesure de produire ou de recevoir des contenus visuels ou audio, et il ne peut pas effectuer de recherches sur internet.

UTILISATEUR: Salut, assistant !
ASSISTANT: Bonjour, que puis-je pour vous ?
"""


def generate_messages(prompt):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
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
        # temperature=0,  # greedy
        # logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
    )
    return response.choices[0]["message"]["content"]


def post_process(output_str):
    # splitted_data = re.split(rf"({ASSISTANT}|{USER}):\s*", output_str)
    splitted_data = re.split(rf"(?:(?<=^)|(?<=\n))({ASSISTANT}|{USER})\s*:\s*", output_str)
    # print(output_str)
    # print(splitted_data)

    if not splitted_data[0]:
        del splitted_data[0]

    if len(splitted_data) % 2 != 0:
        print(f"Failed to split output: {output_str}")
        return

    data = {CONVERSATION: []}

    for idx in range(0, len(splitted_data), 2):
        speaker_name, content = splitted_data[idx], splitted_data[idx + 1]
        if speaker_name not in {ASSISTANT, USER}:
            print(f"Found unvalidated speaker name in: {output_str}")
            return

        # generated line break is kind of random
        content = content.strip()

        data[CONVERSATION].append({ROLE: speaker_name, CONTENT: content})

    return data


def process_item(
    item,
    idx,
    output_file,
    existing_data=[],
    subject_field="subject",
    output_subject_field="subject",
    id_prefix="",
    model="gpt-3.5-turbo",
):
    input_text = item[subject_field]
    # input_text = call_endpoint(generate_messages(TRANSLATION_PROMPT + input_text), model)
    if input_text in existing_data:
        return

    # generated_output_str = call_endpoint(generate_messages(INFERENCE_PROMPT + input_text), model)
    generated_output_str = call_endpoint(generate_messages(INFERENCE_PROMPT.format(sujet=input_text)), model)

    generated_output = post_process(generated_output_str)
    if generated_output is None:
        return

    # generated_output[CONVERSATION].insert(0, {ROLE: USER, CONTENT: input_text})
    generated_output[ID] = f"{id_prefix}{idx:09d}"
    generated_output[output_subject_field] = input_text
    # print(generated_output)

    thread_safe_jsonl_dump(generated_output, output_file)

    return generated_output


def process_data(
    input_json_file: str,
    output_json_file: str,
    subject_field: str = "subject",
    output_subject_field: str = "subject",
    id_prefix: str = "",
    max_samples: Optional[int] = None,
    model: str = "gpt-3.5-turbo",
    max_parallel_requests: int = 4,
):
    data = jsonl_load(input_json_file, "r")
    print(f"Loaded {len(data)} subjects from {input_json_file}")

    seen = set()
    data = [example for example in data if not (example[subject_field] in seen or seen.add(example[subject_field]))]
    print(f"Deduped to {len(data)} subjects")

    if max_samples is not None:
        data = data[:max_samples]
        print(f"Sampled the first {max_samples} subjects")

    # todo: tmp code alpaca
    # instruction_field = "instruction"
    # input_field = "input"
    # for example in data:
    #     example[instruction_field] = (
    #         f"{example[instruction_field][:-1]}: {example[input_field]}"
    #         # f'{example[instruction_field][:-1]}: "{example[input_field]}"'
    #         # f"{example[instruction_field][:-1]}:\n{example[input_field]}"
    #         if example[input_field]
    #         else example[instruction_field]
    #     )

    # debug
    # start = 0
    # end = 10_000
    # data = data[start:end]
    # data = data[end:]

    existing_data = []
    if os.path.exists(output_json_file):
        existing_data = jsonl_load(output_json_file, "r")
        existing_data = [existing_example[output_subject_field] for existing_example in existing_data]
        print(f"Found {len(existing_data)} existing conversations in {output_json_file}")

    start_time = time.perf_counter()

    translated_data = []
    with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
        futures = {
            executor.submit(
                process_item,
                item,
                idx,
                output_json_file,
                existing_data,
                subject_field,
                output_subject_field,
                id_prefix,
                model,
            ): item
            for idx, item in enumerate(data)
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
