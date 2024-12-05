#! /usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

import fire
from dataclasses import dataclass

# import random
from datasets import load_dataset


@dataclass
class Judge:
    name: str
    type: str
    category: str
    system_prompt: str
    prompt_template: str
    output_format: str
    description: str

default
# todo: single, single w/ ref, multi-turn

def run_judge_single(question, answer, judge):
    user_prompt = judge.prompt_template["prompt_template"].format(
        question=question["turns"][0],
        answer=answer["choices"][0]["turns"][0],
        **kwargs,
    )

    rating = -1

    if model in ["gpt-3.5-turbo", "gpt-4"]:
        judgment = chat_compeletion_openai(model, conv, temperature=0, max_tokens=2048)
    elif model in ANTHROPIC_MODEL_LIST:
        judgment = chat_compeletion_anthropic(
            model, conv, temperature=0, max_tokens=1024
        )
    elif model in MISTRAL_MODEL_LIST:
        judgment = chat_compeletion_mistral(
            model, conv, temperature=0, max_tokens=1024
        )
    else:
        raise ValueError(f"Invalid judge model name: {model}")

    if judge.prompt_template["output_format"] == "[[rating]]":
        match = re.search(one_score_pattern, judgment)
        if not match:
            match = re.search(one_score_pattern_backup, judgment)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1
    else:
        raise ValueError(
            f"invalid output format: {judge.prompt_template['output_format']}"
        )

    return rating, user_prompt, judgment


def main(input_file, sampled_output_file, remaining_output_file=None, num_samples=5_000):
    # data = jsonl_load(input_file)
    # random.shuffle(data)
    # jsonl_dump(data[:num_samples], sampled_output_file, mode="w")
    # jsonl_dump(data[num_samples:], remaining_output_file, mode="w")
    # print(f"Saved {len(data[:num_samples])} examples into {sampled_output_file}")
    # print(f"Saved {len(data[num_samples:])} examples into {remaining_output_file}")

    raw_dataset = load_dataset("json", data_files=input_file)["train"]
    print(f"Loaded {raw_dataset.num_rows:,d} examples")

    processed_dataset = raw_dataset.train_test_split(train_size=num_samples / raw_dataset.num_rows, shuffle=True)

    processed_dataset["train"].to_json(sampled_output_file, orient="records", lines=True, force_ascii=False)
    print(f'Saved {processed_dataset["train"].num_rows:,d} examples into {sampled_output_file}')

    if remaining_output_file is not None:
        processed_dataset["test"].to_json(remaining_output_file, orient="records", lines=True, force_ascii=False)
        print(f'Saved {processed_dataset["test"].num_rows:,d} examples into {remaining_output_file}')


if __name__ == "__main__":
    fire.Fire(main)
