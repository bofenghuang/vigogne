#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Translate dataset using open source models.

Usage:
python scripts/data_processing/translate_dataset.py \
    --dataset_name qwedsacf/grade-school-math-instructions \
    --output_file data/chat/grade_school_math_instructions_fr_nllb3b3.jsonl \
    --field_names '["INSTRUCTION", "RESPONSE"]' \
    --model_name_or_path facebook/nllb-200-3.3B
"""

import json
import os
import re
import time
from itertools import chain
from typing import List, Optional

import fire
import torch
from datasets import Dataset, load_dataset
from nltk import tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# or use simple regexes (.!?)
def split_by_sentence(s, language="english"):
    return tokenize.sent_tokenize(s, language=language)


def split_by_paragraph(s):
    return s.split("\n")


def split_by_sentence_and_keep_space(s, language="english"):
    sentences = tokenize.sent_tokenize(s, language=language)

    splitted = []
    for sentence in sentences:
        start = s.find(sentence)
        if start != 0:
            splitted.append(s[:start])
        splitted.append(sentence)
        s = s[start + len(sentence) :]
    if s:
        splitted.append(s)

    return splitted


def split_by_paragraph_and_sentence(s, language="english"):
    paragraphs = re.split("(\n+)", s)

    splitted = []
    for paragraph in paragraphs:
        if re.search("\S", paragraph):
            splitted.extend(split_by_sentence_and_keep_space(paragraph, language=language))
        else:
            splitted.append(paragraph)

    return splitted


def split_function(example, field_name):
    # example[f"{field_name}_splitted"] = split_by_sentence(example[field_name])
    # example[f"{field_name}_splitted"] = split_by_paragraph(example[field_name])
    example[f"{field_name}_splitted"] = split_by_paragraph_and_sentence(example[field_name])
    example[f"{field_name}_length"] = len(example[f"{field_name}_splitted"])
    return example


# def translate_sentence(s):
#     gen_args = {
#         # "num_beams": 5,
#         # contrastive search
#         "top_k": 6,
#         "penalty_alpha": 0.6,
#     }
#         return pipe(s, clean_up_tokenization_spaces=True, **gen_args)[0]["translation_text"]


def main(
    output_file: str,
    field_names: List[str],
    dataset_name: Optional[str] = None,
    dataset_config_name: Optional[str] = None,
    dataset_split_name: Optional[str] = None,
    dataset_parquet_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    model_name_or_path: str = "facebook/m2m100_1.2B",
    batch_size: int = 64,
    fp16: bool = False,
    translate_after_split: bool = False,
):
    # model_name_or_path = "facebook/m2m100_418M"
    # model_name_or_path = "facebook/m2m100_1.2B"
    # model_name_or_path = "facebook/nllb-200-distilled-600M"
    # model_name_or_path = "facebook/nllb-200-1.3B"
    # model_name_or_path = "facebook/nllb-200-3.3B"

    print(f"Running argument: {json.dumps(locals(), indent=4, ensure_ascii=False)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model_args = {}
    if fp16:
        model_args["torch_dtype"] = torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **model_args).eval().cuda()
    # pipe = pipeline("translation", model=model, tokenizer=tokenizer, num_workers=1, batch_size=8, device=0)

    def translate_sentences(input_sentences):
        # NB
        input_ids = tokenizer(input_sentences, padding=True, truncation=True, max_length=1024, return_tensors="pt")["input_ids"].cuda()

        # NB
        gen_args = {
            # "num_beams": 5,
            # "top_k": 6,
            # "penalty_alpha": 0.6,
            # "length_penalty": 2,
            "max_length": 2048,
        }
        if "m2m100" in model_name_or_path:
            gen_args["forced_bos_token_id"] = tokenizer.get_lang_id("fr")
        if "nllb" in model_name_or_path:
            gen_args["forced_bos_token_id"] = tokenizer.lang_code_to_id["fra_Latn"]

        generated_tokens = model.generate(input_ids, **gen_args)
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # def translate_sentences(s):
    #     sentences = split_by_sentence(s)
    #     return " ".join([translate_sentence(s_) for s_ in sentences])

    def translate_field(raw_ds, field_name, batch_size=64):
        final_ds = raw_ds.map(
            lambda examples: {f"translated_{field_name}": translate_sentences(examples[field_name])},
            batched=True,
            batch_size=batch_size,
            desc="translating dataset",
        )

        return final_ds

    def translate_field_after_split(raw_ds, field_name, batch_size=64):
        # split by sentence
        processed_ds = raw_ds.map(lambda example: split_function(example, field_name), desc="splitting")
        flattened_input_texts = list(chain.from_iterable(processed_ds[f"{field_name}_splitted"]))
        flattened_input_lengths = processed_ds[f"{field_name}_length"]

        # find non empty text and keep their indexes
        no_empty_indexes = []
        no_empty_texts = []
        for idx, input_text in enumerate(flattened_input_texts):
            if re.search("\S", input_text):
                no_empty_indexes.append(idx)
                no_empty_texts.append(input_text)

        # only translate non empty text
        temp_ds = Dataset.from_dict({"text": no_empty_texts})
        translated_no_empty_texts = temp_ds.map(
            lambda examples: {"translated_text": translate_sentences(examples["text"])},
            batched=True,
            batch_size=batch_size,
            desc="translating dataset",
        )["translated_text"]
        # print(processed_dataset["text"])
        # print(processed_dataset["translated_text"])
        # quit()

        # refill translated non empty text into all text
        flattened_translated_texts = flattened_input_texts
        for idx, no_empty_index in enumerate(no_empty_indexes):
            flattened_translated_texts[no_empty_index] = translated_no_empty_texts[idx]

        # NB: if splitted input is empty, then also set translated to empty
        # flattened_translated_texts = [
        #     # s if flattened_input_texts[i] else "" for i, s in enumerate(flattened_translated_texts)
        #     s if re.search("\S", flattened_input_texts[i]) else flattened_input_texts[i] for i, s in enumerate(flattened_translated_texts)
        # ]

        # gather translated sentences
        generated_texts = []
        current_idx = 0
        for flattened_input_length in flattened_input_lengths:
            # NB
            # generated_texts.append(" ".join(flattened_translated_texts[current_idx : current_idx + flattened_input_length]))
            # generated_texts.append("\n".join(flattened_translated_texts[current_idx : current_idx + flattened_input_length]))
            generated_texts.append("".join(flattened_translated_texts[current_idx : current_idx + flattened_input_length]))
            current_idx += flattened_input_length
        # print(generated_texts)

        final_ds = raw_ds.map(lambda _, idx: {f"translated_{field_name}": generated_texts[idx]}, with_indices=True)

        return final_ds

    if dataset_name is not None:
        raw_dataset = load_dataset(dataset_name, dataset_config_name, split=dataset_split_name)
    elif dataset_parquet_file is not None:
        raw_dataset = load_dataset("parquet", data_files=dataset_parquet_file, split="train")
    else:
        raise ValueError("No dataset has been specified.")
    print(raw_dataset)

    # tmp ops
    if dataset_name == "quora":
        raw_dataset = raw_dataset.map(
            lambda x: {"subject": x["questions"]["text"][0]}, remove_columns=raw_dataset.column_names, num_proc=4
        )
    if dataset_name == "pacovaldez/stackoverflow-questions":
        raw_dataset = raw_dataset.rename_column("title", "subject")
        raw_dataset = raw_dataset.remove_columns(["body"])
    if dataset_name == "OpenAssistant/oasst1":
        raw_dataset = raw_dataset.filter(lambda x: x["lang"] in {"en", "es"})

    # debug
    # raw_dataset = raw_dataset.select(range(10))

    if max_samples is not None:
        raw_dataset = raw_dataset.shuffle().select(range(max_samples))
        print(raw_dataset)

    start_time = time.perf_counter()

    translate_func = translate_field_after_split if translate_after_split else translate_field

    # translate by column
    processed_dataset = raw_dataset
    for field_name in field_names:
        processed_dataset = translate_func(processed_dataset, field_name=field_name, batch_size=batch_size)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # processed_dataset.to_json(output_file, orient="records", lines=False, indent=4, force_ascii=False)
    # save to json lines file
    processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)

    print(
        f"Translation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}. The translated data is saved into {output_file}"
    )


if __name__ == "__main__":
    fire.Fire(main)
