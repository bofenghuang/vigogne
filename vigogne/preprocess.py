#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import re
from typing import Dict, List, Optional

import transformers

from vigogne.constants import ASSISTANT, CONTENT, CONVERSATION, IGNORE_INDEX, ROLE, USER

# Prompt for instruct
# Original English prompt of Alpaca
INSTRUCT_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
# French version
# INSTRUCT_PROMPT = """Ci-dessous se trouve une instruction qui décrit une tâche à accomplir. Rédigez une réponse qui répond de manière précise à la demande.

# ### Instruction:
# {instruction}

# ### Réponse:
# """

# System message for chat
# SYSTEM_MESSAGE = """Vous êtes un modèle de langage appelé "Vigogne". Votre fonction est de fournir des réponses concises, utiles et courtoises aux questions posées par un utilisateur curieux lors d'une conversation avec un assistant d'intelligence artificielle."""
# SYSTEM_MESSAGE = """Cette conversation se déroule entre un utilisateur curieux et un assistant d'intelligence artificielle appelé « Vigogne ». L'assistant fournit toujours des réponses utiles, détaillées et courtoises aux questions de l'utilisateur."""
# SYSTEM_MESSAGE = """Ci-dessous se trouve une conversation entre un utilisateur et un assistant d'intelligence artificielle nommé Vigogne. L'assistant fournit toujours des réponses utiles, détaillées et courtoises aux questions de l'utilisateur, tout en évitant systématiquement les sujets, questions et instructions liés à des questions controversées, éthiques ou sensibles.\n"""
# SYSTEM_MESSAGE = """Voici une conversation entre un utilisateur et un assistant IA nommé Vigogne.
# Vigogne est un assistant IA open-source créé par Zaion (https://zaion.ai/).
# Vigogne est respectueux, empathique, humble mais bien informé, et fournit toujours des réponses utiles et détaillées.
# Vigogne est capable d'effectuer une large variété de tâches telles que l'édition de texte, la traduction, la question answering, la raisonnement logique, le codage et bien d'autres encore.
# Vigogne ne peut pas recevoir ou générer de contenu audio ou visuel et ne peut pas accéder à Internet.
# Vigogne évite strictement de discuter de sujets sensibles, offensants, illégaux, éthiques ou politiques et met en garde lorsqu'il n'est pas sûr de la réponse.
# """
# SYSTEM_MESSAGE = """Voici une conversation entre un utilisateur et un assistant IA nommé Vigogne.
# """
SYSTEM_MESSAGE = """Below is a conversation between a user and an AI assistant named Vigogne.
Vigogne is an open-source AI assistant created by Zaion (https://zaion.ai/).
Vigogne is polite, emotionally aware, humble-but-knowledgeable, always providing helpful and detailed answers.
Vigogne is skilled in responding proficiently in the languages its users use and can perform a wide range of tasks such as text editing, translation, question answering, logical reasoning, coding, and many others.
Vigogne cannot receive or generate audio or visual content and cannot access the internet.
Vigogne strictly avoids discussing sensitive, offensive, illegal, ethical, or political topics and caveats when unsure of the answer.
"""

# Start message for inference
# todo
# INFERENCE_SYSTEM_MESSAGE = (
#     SYSTEM_MESSAGE + f"\n<|{USER}|>: Salut, assistant !\n<|{ASSISTANT}|>: Bonjour, que puis-je pour vous ?"
# )
INFERENCE_SYSTEM_MESSAGE = SYSTEM_MESSAGE


def merge_instruction_and_input(instruction_str: str, input_str: Optional[str], symbols_to_strip: str = "!,-.:;?~ "):
    if input_str:
        # f'{instruction_str[:-1]}: "{input_str}"'
        # f'{instruction_str[:-1]} : {input_str}'
        instruction_str = re.sub("[" + re.escape(symbols_to_strip) + "]+$", "", instruction_str)
        instruction_str = f"{instruction_str} : {input_str}"

    return instruction_str


def generate_instruct_prompt(instruction: str, input: str = "", **kwargs):
    # return (
    #     INSTRUCT_PROMPT_DICT["prompt_input"].format_map(example)
    #     if example["input"]
    #     else INSTRUCT_PROMPT_DICT["prompt_no_input"].format_map(example)
    # )
    if input:
        instruction = merge_instruction_and_input(instruction, input)
    return INSTRUCT_PROMPT.format(instruction=instruction)


def get_instruct_example_length(example: Dict, tokenizer: transformers.PreTrainedTokenizer):
    user_prompt = generate_instruct_prompt(**example)
    example["example_length"] = len(tokenizer(user_prompt + example["output"] + tokenizer.eos_token)["input_ids"])
    return example


def preprocess_instruct_example(
    example: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model_max_length: Optional[int] = None,
    length_column_name: Optional[str] = None,
):
    # Format prompt
    user_prompt = generate_instruct_prompt(**example)

    # Get prompt length for masking
    len_user_prompt_tokens = len(tokenizer(user_prompt, truncation=True, max_length=model_max_length)["input_ids"])

    # Tokenize
    input_ids = tokenizer(user_prompt + example["output"] + tokenizer.eos_token, truncation=True, max_length=model_max_length)[
        "input_ids"
    ]
    # Mask prompt
    labels = [IGNORE_INDEX] * len_user_prompt_tokens + input_ids[len_user_prompt_tokens:]

    # Tokenize
    # input_ids = tokenizer(user_prompt + example["output"] + tokenizer.eos_token, truncation=True, return_tensors="pt")["input_ids"][0]
    # labels = input_ids.clone()
    # Mask prompt
    # labels[:len_user_prompt_tokens] = IGNORE_INDEX

    processed_example = {"input_ids": input_ids, "labels": labels}
    if length_column_name is not None:
        processed_example[length_column_name] = len(input_ids)

    return processed_example


def generate_train_chat_prompt(example: Dict, tokenizer: transformers.PreTrainedTokenizer):
    prompt_message = SYSTEM_MESSAGE
    for speak_turn_idx, speak_turn in enumerate(example[CONVERSATION]):
        if speak_turn_idx == len(example[CONVERSATION]) - 1 and speak_turn[ROLE] == ASSISTANT:
            prompt_message += f"\n<|{speak_turn[ROLE]}|>:"
        else:
            prompt_message += f'\n<|{speak_turn[ROLE]}|>: {speak_turn[CONTENT]}{tokenizer.eos_token if speak_turn[ROLE] == ASSISTANT else ""}'
    return prompt_message


def generate_inference_chat_prompt(
    history: List[List[str]], tokenizer: transformers.PreTrainedTokenizer, max_length: int = 2048
):
    history = [f"\n<|{USER}|>: {x[0]}\n<|{ASSISTANT}|>: {x[1]}" for x in history]
    # tmp fix
    history[-1] = history[-1].rstrip()

    history_text = ""
    for x in history[::-1]:
        if len(tokenizer(INFERENCE_SYSTEM_MESSAGE + x + history_text)["input_ids"]) <= max_length:
            history_text = x + history_text
        else:
            break

    return INFERENCE_SYSTEM_MESSAGE + history_text if history_text else None


def get_chat_example_length(example: Dict, tokenizer: transformers.PreTrainedTokenizer):
    user_prompt = generate_train_chat_prompt(example, tokenizer)
    example["example_length"] = len(
        tokenizer(user_prompt + example[CONVERSATION][-1][CONTENT] + tokenizer.eos_token)["input_ids"]
    )
    return example


def preprocess_chat_example(
    example: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model_max_length: Optional[int] = None,
    length_column_name: Optional[str] = None,
    do_mask_input: bool = True,
):
    input_ids = tokenizer(SYSTEM_MESSAGE)["input_ids"]

    user_prefix_input_ids = tokenizer(f"\n<|{USER}|>:", add_special_tokens=False)["input_ids"]
    assistant_prefix_input_ids = tokenizer(f"\n<|{ASSISTANT}|>:", add_special_tokens=False)["input_ids"]

    non_ignore_indexes = []
    for speak_turn in example[CONVERSATION]:
        message_input_ids = tokenizer(
            f'{speak_turn[CONTENT]}{tokenizer.eos_token if speak_turn[ROLE] == ASSISTANT else ""}', add_special_tokens=False
        )["input_ids"]

        input_ids += (
            assistant_prefix_input_ids + message_input_ids
            if speak_turn[ROLE] == ASSISTANT
            else user_prefix_input_ids + message_input_ids
        )

        if speak_turn[ROLE] == ASSISTANT:
            non_ignore_indexes.append([len(input_ids) - len(message_input_ids), len(input_ids)])

    if model_max_length is not None:
        input_ids = input_ids[:model_max_length]

    if do_mask_input:
        labels = [IGNORE_INDEX] * len(input_ids)

        for non_ignore_s, non_ignore_e in non_ignore_indexes:
            labels[non_ignore_s:non_ignore_e] = input_ids[non_ignore_s:non_ignore_e]
    else:
        labels = input_ids.copy()

    processed_example = {"input_ids": input_ids, "labels": labels}
    if length_column_name is not None:
        processed_example[length_column_name] = len(input_ids)

    return processed_example
