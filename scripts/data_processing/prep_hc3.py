#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import fire
from datasets import load_dataset

from vigogne.constants import ASSISTANT, CONTENT, CONVERSATION, ROLE, USER


def main(output_file):
    raw_dataset = load_dataset("Hello-SimpleAI/HC3", "all")["train"]
    print(raw_dataset)

    # debug
    # raw_dataset = raw_dataset.select(range(10))

    data_df = raw_dataset.to_pandas()
    # data_df["answer"] = data_df.apply(lambda row: row["human_answers"] + row["chatgpt_answers"], axis=1)
    data_df["answer"] = data_df.apply(lambda row: [*row["human_answers"], *row["chatgpt_answers"]], axis=1)
    data_df = data_df.explode("answer")
    # data_df = data_df[["question", "answer", "source"]]
    # print(data_df.head())
    # print(data_df.info())
    print(data_df.shape[0])

    # dedup by idx
    data_df = data_df.sample(frac=1)
    data_df.drop_duplicates(subset="id", keep="first", inplace=True)
    print(data_df.shape[0])

    # instruct format
    data_df.rename(columns={"question": "instruction", "answer": "output"}, inplace=True)
    data_df["input"] = ""
    data_df = data_df[["instruction", "input", "output"]]
    print(data_df.head())

    # conversation format
    # data_df[CONVERSATION] = data_df.apply(
    #     lambda row: [
    #         {ROLE: USER, CONTENT: row["question"]},
    #         {ROLE: ASSISTANT, CONTENT: row["answer"]},
    #     ],
    #     axis=1,
    # )
    # data_df = data_df[["source", CONVERSATION]]
    # print(data_df.head())

    data_df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"The processed data is saved into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
