#!/usr/bin/env python
# coding=utf-8

"""
Modified from: https://github.com/tatsu-lab/stanford_alpaca/blob/main/generate_instruction.py

Usage:
python vigogne/data/deduplicate_instructions.py \
    --input_file ./data/machine_generated_data.jsonl \
    --output_file ./data/machine_generated_data_deduped.jsonl \
    --n_workers 64
"""

import os
from functools import partial
from multiprocessing import Pool

import fire
import numpy as np
from nltk import RegexpTokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm

from vigogne.data.utils import jsonl_dump, jsonl_load


class CustomizedTokenizer(RegexpTokenizer):
    def __init__(self):
        super().__init__(r"\w'|\w+|[^\w\s]")

    def tokenize(self, s):
        s = s.lower()
        return super().tokenize(s)


def deduplicate_instructions(
    input_file: str = "./data/machine_generated_data.jsonl",
    output_file: str = "./data/machine_generated_data_deduped.jsonl",
    n_workers: int = 16,
):
    machine_instruction_data = jsonl_load(input_file)
    print(f"Loaded {len(machine_instruction_data)} raw machine generated instructions")
    # print(machine_instruction_data[0])

    all_instruction_data = []
    if os.path.exists(output_file):
        all_instruction_data = jsonl_load(output_file)
        print(f"Found {len(all_instruction_data)} existing clean instructions in {output_file}")

    # NB: default tokenizer removes accent
    # scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    # todo
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True, tokenizer=CustomizedTokenizer())

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [example["instruction"] for example in all_instruction_data]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    keep = 0
    for machine_instruction_example in tqdm(machine_instruction_data, desc="process machine generated example"):
        # computing similarity with the pre-tokenzied instructions
        new_instruction_tokens = scorer._tokenizer.tokenize(machine_instruction_example["instruction"])
        with Pool(n_workers) as p:
            rouge_scores = p.map(
                partial(rouge_scorer._score_lcs, new_instruction_tokens),
                all_instruction_tokens,
            )
        rouge_scores = [score.fmeasure for score in rouge_scores]
        most_similar_instructions = {all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]}
        if max(rouge_scores) > 0.7:
            continue
        else:
            keep += 1
        machine_instruction_example["most_similar_instructions"] = most_similar_instructions
        machine_instruction_example["avg_similarity_score"] = float(np.mean(rouge_scores))
        all_instruction_data.append(machine_instruction_example)
        all_instructions.append(machine_instruction_example["instruction"])
        all_instruction_tokens.append(new_instruction_tokens)

    print(f"Loaded {len(machine_instruction_data)} instructions, kept {keep} instructions")
    # Override output_file
    jsonl_dump(all_instruction_data, output_file, mode="w")


if __name__ == "__main__":
    fire.Fire(deduplicate_instructions)
