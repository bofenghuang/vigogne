#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

from typing import Optional

import fire
from datasets import load_dataset
from transformers import AutoTokenizer

from vigogne.data_utils import Conversation, Instruct, Role, Utterance
from vigogne.preprocess import CONVERSATION_SYSTEM_MESSAGE_EN, INSTRUCT_SYSTEM_MESSAGE_EN
from vigogne.train.utils.process_data import SUPPORTED_PROCESSOR_TEMPLATES

instruct_processor = SUPPORTED_PROCESSOR_TEMPLATES["instruct"]
chat_processor = SUPPORTED_PROCESSOR_TEMPLATES["chat"]

ROLES = [Role.user, Role.assistant]


def main(
    output_file,
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf",
    max_length: Optional[int] = None,
    preprocessing_num_workers: int = 4,
):
    raw_dataset = load_dataset("GAIR/lima")["train"]
    print(raw_dataset)

    # debug
    # raw_dataset = raw_dataset.select(range(10))

    # 1000 of 1030 are single turn
    # processed_dataset = raw_dataset.filter(lambda example: len(example["conversations"]) == 2)
    # print(processed_dataset)

    # def process_function(example):
    #     return {
    #         "system": INSTRUCT_SYSTEM_MESSAGE_EN,
    #         "instruction": example["conversations"][0],
    #         # "input": "",
    #         "output": example["conversations"][1],
    #     }
    #     # return Instruct(
    #     #     system=INSTRUCT_SYSTEM_MESSAGE_EN,
    #     #     instruction=example["conversations"][0],
    #     #     # input="",
    #     #     output=example["conversations"][1],
    #     # ).dict()

    # processed_dataset = processed_dataset.map(
    #     process_function, num_proc=preprocessing_num_workers, remove_columns=processed_dataset.column_names
    # )
    # print(processed_dataset)

    def process_function(example):
        conversation = Conversation(
            # id=example["id"],
            # system=example["system"],
            system=CONVERSATION_SYSTEM_MESSAGE_EN,
            messages=[
                Utterance(role=ROLES[i % 2], content=content) for i, content in enumerate(example["conversations"])
            ],
        )
        return conversation.fully_model_dump()

    processed_dataset = raw_dataset.map(
        process_function, num_proc=preprocessing_num_workers, remove_columns=raw_dataset.column_names
    )
    print(processed_dataset)

    if max_length is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        processed_dataset = processed_dataset.map(
            # lambda example: instruct_processor.get_example_length(example, tokenizer),
            lambda example: chat_processor.get_example_length(example, tokenizer),
            num_proc=preprocessing_num_workers,
        )

        processed_dataset = processed_dataset.filter(
            lambda x: x <= max_length, input_columns="example_length", num_proc=preprocessing_num_workers
        )

        processed_dataset = processed_dataset.remove_columns("example_length")
        print(processed_dataset)

    processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"The processed data is saved into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
