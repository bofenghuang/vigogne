#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

import fire

from vigogne.data.utils import jload, jsonl_dump


def main(input_file, output_file):
    jsonl_dump(jload(input_file), output_file, mode="w")


if __name__ == "__main__":
    fire.Fire(main)
