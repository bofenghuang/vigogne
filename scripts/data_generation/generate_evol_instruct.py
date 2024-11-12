#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Copied and modified from WizarLM.

Usage:
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

"""

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import fire
from datasets import load_dataset
from tqdm import tqdm

from vigogne.file_utils import thread_safe_jsonl_dump
from generate_response import call_endpoint, generate_messages, post_process_response
from common import generate_api_messages_mistral, chat_completion_mistral, process_api_response_mistral

# Original English version
DEPTH_INSTRUCTION_TEMPLATE = """I want you act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.
You SHOULD complicate the given prompt using the following method:
{method}
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#.
#The Given Prompt#:
{prompt}
#Rewritten Prompt#:
# """

# DEPTH_CONSTRAINTS_INSTRUCTION = "Please add one more constraints/requirements into #The Given Prompt#'"
# DEPTH_DEEPEN_INSTRUCTION = (
#     "If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased."
# )
# DEPTH_CONCRETIZING_INSTRUCTION = "Please replace general concepts with more specific concepts."
# DEPTH_REASONING_INSTRUCTION = (
#     "If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request"
#     " multiple-step reasoning."
# )

# BREADTH_INSTRUCTION_TEMPLATE = """I want you act as a Prompt Creator.
# Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.
# This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.
# The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.
# The #Created Prompt# must be reasonable and must be understood and responded by humans.
# '#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#.
# #Given Prompt#:
# {prompt}
# #Created Prompt#:
# """

# French version
DEPTH_INSTRUCTION_TEMPLATE = """Votre mission consiste à agir comme un Réécrivain de Prompt.
Votre objectif est de réécrire un prompt donné en une version plus complexe pour rendre ces célèbres systèmes d'IA (par exemple, chatgpt et GPT4) un peu plus difficiles à gérer.
Cependant, le prompt réécrit doit rester raisonnable et compréhensible pour les humains, qui doivent pouvoir y répondre.
Votre réécriture ne peut pas omettre les parties non textuelles telles que le tableau et le code dans #Prompt Donné#:. De plus, veuillez ne pas omettre l'entrée dans #Prompt Donné#.
Vous DEVEZ complexifier le prompt donné en utilisant la méthode suivante :
{method}
Vous devez faire de votre mieux pour ne pas rendre #Prompt Réécrit# verbeuse, #Prompt Réécrit# ne peut ajouter que 10 à 20 mots dans #Prompt Donné#.
'#Prompt Donné#', '#Prompt Réécrit#', 'prompt donné' et 'prompt réécrit' ne doivent pas apparaître dans #Prompt Réécrit#.
#Prompt Donné# :
{prompt}
#Prompt Réécrit# :
"""

DEPTH_CONSTRAINTS_INSTRUCTION = "Veuillez ajouter une contrainte/exigence supplémentaire dans le #Prompt Donné#"
DEPTH_DEEPEN_INSTRUCTION = (
    "Si le #Prompt Donné# contient des questions sur certains problèmes, la profondeur et l'étendue de l'enquête peuvent être"
    " augmentées."
)
DEPTH_CONCRETIZING_INSTRUCTION = "Veuillez remplacer les concepts généraux par des concepts plus spécifiques."
DEPTH_REASONING_INSTRUCTION = (
    "Si le #Prompt Donné# peut être résolu avec quelques processus de réflexion simples, vous pouvez le reformuler pour"
    " demander explicitement un raisonnement en plusieurs étapes."
)

BREADTH_INSTRUCTION_TEMPLATE = """Votre mission consiste à agir comme un Créateur de Prompt.
Votre objectif est de vous inspirer de la #Prompt Donné# pour créer un tout nouveau prompt.
Ce nouveau prompt doit appartenir au même domaine que la #Prompt Donné# mais être encore plus rare.
La LONGUEUR et la complexité de la #Prompt Créé# doivent être similaires à celles de la #Prompt Donné#.
La #Prompt Créé# doit être raisonnable et compréhensible, et devra être répondu par des humains.
'#Prompt Donné#', '#Prompt Créé#', 'prompt donné' et 'prompt créé' ne doivent pas apparaître dans la #Prompt Créé#.
#Prompt Donné# :
{prompt}
#Prompt Créé# :
"""


class EvolInstructor:
    def __init__(self, weights: Optional[List[float]] = None):
        self.depth_instruction_template = DEPTH_INSTRUCTION_TEMPLATE
        self.breadth_instruction_template = BREADTH_INSTRUCTION_TEMPLATE

        self.depth_constraints_instruction = DEPTH_CONSTRAINTS_INSTRUCTION
        self.depth_deepen_instruction = DEPTH_DEEPEN_INSTRUCTION
        self.depth_concretizing_instruction = DEPTH_CONCRETIZING_INSTRUCTION
        self.depth_reasoning_instruction = DEPTH_REASONING_INSTRUCTION

        self.process_functions = [
            (None, None),
            ("create_constraints_prompt", self.create_constraints_prompt),
            ("create_deepen_prompt", self.create_deepen_prompt),
            ("create_concretizing_prompt", self.create_concretizing_prompt),
            ("create_reasoning_prompt", self.create_reasoning_prompt),
            ("create_breadth_prompt", self.create_breadth_prompt),
        ]

        self.weights = weights

    def __call__(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        **kwargs,
    ):
        sampling_args = {"weights": self.weights} if self.weights is not None else {}

        process_function_name, process_function = random.choices(self.process_functions, k=1, **sampling_args)[0]

        if process_function_name is not None:
            processed_prompt = process_function(prompt)

            # request_messages = generate_messages(processed_prompt)
            # response = call_endpoint(request_messages, model, **kwargs)
            # result = post_process_response({}, response)

            request_messages = generate_api_messages_mistral(processed_prompt)
            response = chat_completion_mistral(request_messages, model, **kwargs)
            result = process_api_response_mistral(response)
        else:
            # todo
            # result = post_process_response({}, {})
            result = process_api_response_mistral({})
            result["output"] = prompt

        result["evol_function"] = process_function_name

        return result

    def create_constraints_prompt(self, prompt: str):
        return self.depth_instruction_template.format(method=self.depth_constraints_instruction, prompt=prompt)

    def create_deepen_prompt(self, prompt: str):
        return self.depth_instruction_template.format(method=self.depth_deepen_instruction, prompt=prompt)

    def create_concretizing_prompt(self, prompt: str):
        return self.depth_instruction_template.format(method=self.depth_concretizing_instruction, prompt=prompt)

    def create_reasoning_prompt(self, prompt: str):
        return self.depth_instruction_template.format(method=self.depth_reasoning_instruction, prompt=prompt)

    def create_breadth_prompt(self, prompt: str):
        return self.breadth_instruction_template.format(prompt=prompt)


def process_item(
    item: Dict,
    output_file: str,
    evol_instructor: EvolInstructor,
    column_name: str = "instruction",
    model: str = "gpt-3.5-turbo",
    **kwargs,
):
    result = evol_instructor(item[column_name], model=model, **kwargs)
    result[f"evolved_{column_name}"] = result.pop("output")
    item.update(result)

    thread_safe_jsonl_dump(item, output_file, mode="a")

    return item


def main(
    input_file: str,
    output_file: str,
    column_name: str = "instruction",
    max_samples: Optional[int] = None,
    model: str = "gpt-3.5-turbo",
    max_parallel_requests: int = 1,
    **kwargs,
):
    dataset = load_dataset("json", data_files=input_file, split="train")
    # dataset = dataset.shuffle(seed=10)
    print(f"Loaded {dataset.num_rows:,d} examples from {input_file}")

    if max_samples is not None:
        dataset = dataset.select(range(max_samples))
        print(f"Sampled the first {dataset.num_rows:,d} examples")

    # debug
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

    evol_instructor = EvolInstructor()

    start_time = time.perf_counter()

    translated_data = []
    with tqdm(total=dataset.num_rows, desc="Generating") as pbar:
        with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
            futures = {
                executor.submit(
                    process_item,
                    item,
                    output_file,
                    evol_instructor,
                    column_name,
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
