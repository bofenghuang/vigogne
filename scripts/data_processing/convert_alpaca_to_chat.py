#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

"""Convert alpaca instruct-following data to conversation format."""

from functools import partial
from typing import Optional

import fire
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from vigogne.data_utils import Conversation, Role, Utterance
from vigogne.file_utils import jload, jsonl_dump, jsonl_load
from vigogne.preprocess import (
    CONVERSATION_SYSTEM_MESSAGE_EN,
    CONVERSATION_SYSTEM_MESSAGE_FR,
    INSTRUCT_SYSTEM_MESSAGE_EN,
    INSTRUCT_SYSTEM_MESSAGE_FR,
    merge_instruction_and_input,
)
from vigogne.train.utils.process_data import SUPPORTED_PROCESSOR_TEMPLATES

instruct_processor = SUPPORTED_PROCESSOR_TEMPLATES["instruct"]
chat_processor = SUPPORTED_PROCESSOR_TEMPLATES["chat"]

def main(
    input_file: str,
    output_file: str,
    # model_name_or_path: str = "meta-llama/Llama-2-7b-hf",
    # max_length: Optional[int] = None,
    preprocessing_num_workers: int = 4,
):
    # data = jsonl_load(input_file)
    # reformatted_data = list(map(convert_to_chat, data))
    # jsonl_dump(reformatted_data, output_file, mode="w")

    raw_dataset = load_dataset("json", data_files=input_file)["train"]
    print(raw_dataset)

    # raw_dataset = Dataset.from_list(jload(input_file))
    # print(raw_dataset)
    # raw_dataset = raw_dataset.remove_columns(["most_similar_instructions", "avg_similarity_score"])

    # raw_dataset = raw_dataset.rename_columns({"context": "input", "response": "output"})

    # def process_function(example):
    #     example["instruction"] = merge_instruction_and_input(example["instruction"], example["input"])
    #     example["system"] = INSTRUCT_SYSTEM_MESSAGE_FR
    #     # example["system"] = INSTRUCT_SYSTEM_MESSAGE_EN

    #     return example

    def process_function(example):
        conversation = Conversation(
            id=example["id"],
            system=example["system"],
            # system=CONVERSATION_SYSTEM_MESSAGE_FR,
            # system=CONVERSATION_SYSTEM_MESSAGE_EN,
            messages=[
                # Utterance(role=Role.user, content=merge_instruction_and_input(example["instruction"], example["input"])),
                Utterance(role=Role.user, content=example["instruction"]),
                Utterance(role=Role.assistant, content=example["output"]),
            ]
        )
        return conversation.fully_model_dump()

    processed_dataset = raw_dataset.map(
        process_function,
        num_proc=preprocessing_num_workers,
        remove_columns=raw_dataset.column_names,
    )
    # processed_dataset = processed_dataset.remove_columns("input")
    # processed_dataset = processed_dataset.remove_columns("original_instruction")
    print(processed_dataset)

    # if max_length is not None:

    #     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    #     processed_dataset = processed_dataset.map(
    #         # lambda example: instruct_processor.get_example_length(example, tokenizer),
    #         lambda example: chat_processor.get_example_length(example, tokenizer),
    #         num_proc=preprocessing_num_workers,
    #     )

    #     processed_dataset = processed_dataset.filter(
    #         lambda x: x <= max_length, input_columns="example_length", num_proc=preprocessing_num_workers
    #     )

    #     processed_dataset = processed_dataset.remove_columns("example_length")
    #     print(processed_dataset)

    processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"The processed data is saved into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
