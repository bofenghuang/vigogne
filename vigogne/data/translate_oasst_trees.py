#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Usage:
python scripts/translate_oasst_trees.py \
    --input_file_path data/2023-04-12_oasst_ready.trees.jsonl \
    --output_file_path data/2023-04-12_oasst_ready_fr_nllb3B3.trees.jsonl \
    --model_name_or_path facebook/nllb-200-3.3B
"""

import os
import time
from typing import List, Optional
from tqdm import tqdm

import fire
import torch
from nltk import tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from vigogne.data.utils import jsonl_load, jsonl_dump

LANG_MAPPINGS = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}


def split_by_sentence(s, language="english"):
    return tokenize.sent_tokenize(s, language=language)


def item_generator(json_input, lookup_key):
    if isinstance(json_input, dict):
        for k, v in json_input.items():
            if k == lookup_key:
                # yield v
                yield json_input
            else:
                yield from item_generator(v, lookup_key)
    elif isinstance(json_input, list):
        for item in json_input:
            yield from item_generator(item, lookup_key)


def main(
    input_file_path: str,
    output_file_path: str,
    langs: List[str] = ["en", "es"],
    max_samples: Optional[int] = None,
    model_name_or_path: str = "facebook/nllb-200-1.3B",
):
    # model_name_or_path = "facebook/m2m100_418M"
    # model_name_or_path = "facebook/m2m100_1.2B"
    # model_name_or_path = "facebook/nllb-200-distilled-600M"
    # model_name_or_path = "facebook/nllb-200-1.3B"
    # model_name_or_path = "facebook/nllb-200-3.3B"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16).eval().cuda()
    # model.config.forced_bos_token_id = tokenizer.lang_code_to_id["fra_Latn"]
    # pipe = pipeline("translation", model=model, tokenizer=tokenizer, num_workers=1, batch_size=8, device=0)

    def translate_sentences(input_sentences):
        # print(input_sentences)
        input_ids = tokenizer(input_sentences, padding=True, return_tensors="pt")["input_ids"].cuda()

        # NB
        gen_args = {
            # "num_beams": 5,
            # "top_k": 6, "penalty_alpha": 0.6,
            # "length_penalty": 2,
            "max_length": 512,
            # "forced_bos_token_id": tokenizer.get_lang_id("fr"),  # m2m
            "forced_bos_token_id": tokenizer.lang_code_to_id["fra_Latn"],  # nllb
        }

        generated_tokens = model.generate(input_ids, **gen_args)
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def translate_by_sentence(s, language="english"):
        return " ".join(translate_sentences(split_by_sentence(s, language=language)))

    def translate_function(input_sentences, language="english", split_symbol="\n"):
        # todo: still loss format
        # re.split(r"(\n+(?:\s|#|`)*)", raw_sentence)
        return split_symbol.join(
            [
                translate_by_sentence(paragraph, language=language) if paragraph else paragraph
                for paragraph in input_sentences.split(split_symbol)
            ]
        )

    tree_data = jsonl_load(input_file_path)
    print(f"Loaded {len(tree_data)} trees")

    tree_data = [x for x in tree_data if x["prompt"]["lang"] in langs]
    print(f"Got {len(tree_data)} trees in {langs}")

    if max_samples is not None:
        tree_data = tree_data[:max_samples]
        print(f"Sampled the first {max_samples} trees")

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    start_time = time.perf_counter()

    for tree_example in tqdm(tree_data):
        for item in item_generator(tree_example, "text"):
            # can't change dict size during iteration
            # item["original_text"] = item["text"]
            item["text"] = translate_function(item["text"], language=LANG_MAPPINGS.get(item.get("lang", "en"), "english"))

        jsonl_dump(tree_example, output_file_path)

    print(
        f"Translation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}. The translated data is saved in {output_file_path}"
    )


if __name__ == "__main__":
    fire.Fire(main)
