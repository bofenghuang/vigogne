#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

"""Filter unwanted examples from self-instruct datasets."""

import fire
import re

from vigogne.file_utils import jsonl_load, jsonl_dump


def main(input_file, valid_output_file, invalid_output_file, instruction_field="instruction"):
    data = jsonl_load(input_file)

    validated_data, unvalidated_data = [], []
    for example in data:
        # summarize web page
        # if re.search(r"résumez|résume|résumes|résumer", example[instruction_field], flags=re.IGNORECASE):
        #     if re.search(r"\bhttps\b", example[instruction_field]):
        #         unvalidated_data.append(example)
        # remove all summarization task: only titles are given
        if re.search(r"résumez|résume|résumes|résumer", example[instruction_field], flags=re.IGNORECASE):
            unvalidated_data.append(example)
        if re.search(r"préféré", example[instruction_field], flags=re.IGNORECASE):
            unvalidated_data.append(example)
        else:
            validated_data.append(example)

    jsonl_dump(validated_data, valid_output_file, mode="w")
    jsonl_dump(unvalidated_data, invalid_output_file, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
