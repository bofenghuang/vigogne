#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

import argparse

from datasets import load_dataset

# from vigogne.file_utils import jload, jsonl_dump


def main(input_files, output_file):
    # jsonl_dump(jload(input_file), output_file, mode="w")

    dataset = load_dataset("json", data_files=input_files, split="train")
    print(dataset)
    dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert multiple JSON files into a single JSONL dataset file")
    parser.add_argument("-i", "--input_files", nargs="+", required=True, help="Input JSON files to process (space-separated)")
    parser.add_argument("-o", "--output_file", required=True, help="Output JSONL file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input_files, args.output_file)
