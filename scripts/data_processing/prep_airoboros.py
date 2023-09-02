#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Convert to the Vigogne's format and remove examples with prompt longer than max_length."""

from typing import Optional

import fire
from datasets import load_dataset
from transformers import AutoTokenizer

from vigogne.data_utils import Instruct
from vigogne.preprocess import INSTRUCT_SYSTEM_MESSAGE_EN
from vigogne.train.utils.process_data import SUPPORTED_PROCESSOR_TEMPLATES

instruct_processor = SUPPORTED_PROCESSOR_TEMPLATES["instruct"]


def main(
    input_file: str,
    output_file: str,
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf",
    max_length: Optional[int] = None,
    preprocessing_num_workers: int = 4,
):
    raw_dataset = load_dataset("json", data_files=input_file)["train"]
    print(raw_dataset)

    # stat
    print(raw_dataset.to_pandas()["category"].value_counts())

    processed_dataset = raw_dataset.remove_columns("question_id")
    processed_dataset = processed_dataset.rename_columns({"response": "output"})
    processed_dataset = processed_dataset.add_column("system", [INSTRUCT_SYSTEM_MESSAGE_EN] * processed_dataset.num_rows)

    if max_length is not None:

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        processed_dataset = processed_dataset.map(
            lambda example: instruct_processor.get_example_length(example, tokenizer),
            num_proc=preprocessing_num_workers,
        )

        processed_dataset = processed_dataset.filter(
            lambda x: x <= max_length, input_columns="example_length", num_proc=preprocessing_num_workers
        )

        processed_dataset = processed_dataset.remove_columns("example_length")
        print(processed_dataset)

        # stat
        print(processed_dataset.to_pandas()["category"].value_counts())

    processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"The processed data is saved into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
