#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

"Filter reponses with OpenAI disclaimers and refusals."

import json
import re
from collections import defaultdict

import fire
from datasets import load_dataset

# Regexes used to filter responses, mostly common words and phrases used in refusals.
# Adapted from https://github.com/jondurbin/airoboros/blob/3d42fb0aff543182bd75707ff298fdbe591dc3b9/example-config.yaml#L27C1-L42C13
# https://huggingface.co/datasets/cognitivecomputations/WizardLM_alpaca_evol_instruct_70k_unfiltered/blob/main/wizardlm_clean.py
# https://erichartford.com/uncensored-models
# todo
filter_words = [
    # English
    "my programming",
    "language model",
    "large language",
    "as( an)? (ai|generative language|gpt|bot)",
    "illegal and dangerous",
    "i do(n't| not) (possess|have|exhibit) (personal|consciousness|subjective)",
    "personal (feelings|thoughts|emotions|desires|experiences|goals|objective|belief)",
    "(can('t| ?not)|w(on't|will not)|unable.?) (\\w+\\s)+(with (that|your)|your \\w+|provide)",
    "my limitations",
    "the limitations of my",
    "my abilities",
    "violates my",
    "i (can('t| ?not)|w(on't|will not)|am (not |un)able.?).{0,30}(you are|you're|your )",
    r"\bflesch\b",
    "not answerable",  # always problem with instruction
    # Knowledge cutoff
    "my knowledge cut(\s)?off",
    # "September 2021",
    # Branding
    # r"\bopenai\b",
    # r"\b(?:openai|mistral|meta|facebook|llama|anthropic|claude|gemini)\b",
    # r"\b(chat)?[\s-]?gpt\b",
    # r"gpt[\s-]?[\d\.]+[\s-]?(turbo)?",
    # r"\bgpt[\s-]?[\d\.]+[\s-]?(turbo|o)?(-[\d-]+)?\b",
    # French
    r"en tant qu('|e )(ia|assistant|modèle|gpt|bot)\b",
    r"(Je suis|Étant|etant) un(e)? (ia|assistant|modèle|gpt|bot)\b",
    r"je( ne)? suis pas (capable|programmé|en mesure)",
    r"j(e n)?'ai pas (l'|le |la |d'|de )?(accès|capacité|sentiment|personnalité)",
    r"je( ne)? peux pas",
    # "je suis désolé",
    r"(pas de|sans|informations insuffisantes pour fournir une) réponse",
    "non répondu",
]

# Compile patterns
# case insensitive
# filter_patterns = [re.compile(x, re.I) for x in filter_words]
filter_patterns = {x: re.compile(x, re.I) for x in filter_words}

filtered_counter = defaultdict(int)


def _filter(s):
    # for pattern in filter_patterns:
    for pattern_str, pattern in filter_patterns.items():
        if pattern.search(s):
            filtered_counter[pattern_str] += 1
            return False
    return True


def filter_function(example):
    assistant_text = "\n".join([x["content"] for x in example["messages"] if x["role"] == "assistant"])
    return _filter(assistant_text)


def main(
    input_file: str,
    output_file: str,
    # response_field: str = "output",
    # preprocessing_num_workers: int = 4,
):
    # load
    dataset = load_dataset("json", data_files=input_file, split="train")
    print(f"Loaded {dataset.num_rows:,d} examples from {input_file}")

    # filter
    processed_dataset = dataset.filter(
        filter_function,
        # lambda *x, **y: not filter_function(*x, **y),  # debug
        # input_columns=response_field,
        # num_proc=preprocessing_num_workers,
        num_proc=1,
    )
    print(json.dumps(filtered_counter, ensure_ascii=False, indent=4))
    print(f"Filtered to {processed_dataset.num_rows:,d} examples")

    # export
    processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"Saved into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
