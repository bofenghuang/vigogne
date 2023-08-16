#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

"""Filter reponses with OpenAI disclaimers and refusals."""

import re

import fire
from datasets import Dataset, load_dataset

# # Regexes used to filter responses, mostly common words and phrases used in refusals.
# Adapted from https://github.com/jondurbin/airoboros/blob/3d42fb0aff543182bd75707ff298fdbe591dc3b9/example-config.yaml#L27C1-L42C13
filter_words = [
    # English
    "my programming",
    "openai",
    "language model",
    "large language",
    "as an? (ai|generative language|gpt|bot)",
    "illegal and dangerous",
    "i do(n't| not) (possess|have|exhibit) (personal|consciousness|subjective)",
    "personal (feelings|thoughts|emotions|desires|experiences|goals|objective|belief)",
    "(can('t| ?not)|w(on't|will not)|unable.?) (\\w+\\s)+(with (that|your)|your \\w+|provide)",
    "my limitations",
    "the limitations of my",
    "my abilities",
    "violates my",
    "i (can('t| ?not)|w(on't|will not)|am (not |un)able.?).{0,30}(you are|you're|your )",
    "flesch",
    # French
    "je suis désolé",
    "en tant qu('|e )(ia|assistant|modèle|gpt|bot)",
    "Je suis un(e)? (ia|assistant|modèle|gpt|bot)",
    "je( ne)? suis pas (capable|programmé|en mesure)",
    "j(e n)?'ai pas (l'|le |la |d'|de )?(accès|capacité|sentiment|personnalité)",
]

filter_patterns = [re.compile(x, re.I) for x in filter_words]

def filter_function(s):
    for pattern in filter_patterns:
        if pattern.search(s):
            return False
    return True

def main(
    input_file: str,
    output_file: str,
    response_field: str = "output",
    preprocessing_num_workers: int = 4,
):
    raw_dataset = load_dataset("json", data_files=input_file)["train"]
    print(raw_dataset)

    processed_dataset = raw_dataset.filter(filter_function, input_columns=response_field, num_proc=preprocessing_num_workers)
    print(processed_dataset)

    processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"The processed data is saved into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
