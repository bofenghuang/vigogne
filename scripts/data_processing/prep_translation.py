#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

"""
Usage:
python scripts/data_processing/prep_translation.py \
    --instruction_file_path scripts/data_processing/prompt_translation_en_fr.txt \
    --output_file path/to/translation_task_wmt14_en_fr_cleaned.jsonl \
    --embedding_model_name_or_path paraphrase-multilingual-mpnet-base-v2 \
    --embedding_batch_size 256
"""

import logging
import os
import random
import urllib.request
from typing import Optional

import fasttext
import fire
import pandas as pd
import regex as re
from datasets import Dataset, load_dataset
from scipy.spatial import distance

from vigogne.data_utils import Conversation, Role, Utterance
from vigogne.preprocess import CONVERSATION_SYSTEM_MESSAGE_EN, CONVERSATION_SYSTEM_MESSAGE_FR

logger = logging.getLogger(__name__)

fmt = "%(asctime)s [%(levelname)s] [%(name)s] [%(funcName)s] %(message)s"
datefmt = "%Y-%m-%dT%H:%M:%SZ"
logging.basicConfig(format=fmt, datefmt=datefmt, level=logging.INFO)

# language code
language_mappings = {
    "en": {"en": "English", "fr": "French"},
}

fasttext_loc = f"{os.path.dirname(os.path.abspath(__file__))}/lid.176.ftz"

if not os.path.exists(fasttext_loc):
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/" + "fasttext/supervised-models/lid.176.ftz",
        fasttext_loc,
    )
lang_id_func = fasttext.load_model(fasttext_loc)


def deduplicate_dataset(ds, field_name, num_proc=4):
    def _get_hash(example):
        """Get hash of content field."""
        return {"hash": hash(example[field_name])}

    def _check_uniques(example, uniques):
        """Check if current hash is still in set of unique hashes and remove if true."""
        if example["hash"] in uniques:
            uniques.remove(example["hash"])
            return True
        else:
            return False

    processed_ds = ds.map(_get_hash, num_proc=num_proc, desc="get hash")
    uniques = set(processed_ds.unique("hash"))
    filtered_ds = processed_ds.filter(_check_uniques, fn_kwargs={"uniques": uniques}, desc="dedup data")
    filtered_ds = filtered_ds.remove_columns("hash")
    return filtered_ds


# Adapted from https://github.com/OpenNMT/OpenNMT-py/blob/ca3f3e254a27c0dea92cabab37727ceeeb1abeb4/onmt/transforms/clean.py#L11
# Shoutout to LinxiaoZeng for his valuable advice!
def filter_function(
    src_text,
    tgt_text,
    src_embedding=None,
    tgt_embedding=None,
    scripts_ok=["Latin", "Common"],
    scripts_nok=[],
    min_tok=15,
    max_tok=50,
    avg_tok_min=3,
    avg_tok_max=20,
    tok_max=20,
    src_tgt_ratio=2,
    allowed_src_lang_ids=["en"],
    allowed_tgt_lang_ids=["fr"],
    min_similariry=0.8,
):
    n_src_words = len(src_text.split())
    n_tgt_words = len(tgt_text.split())

    ok_regex = "[^" + "".join(r"\p{%s}" % sc for sc in scripts_ok) + "]"
    nok_regex = "[" + "".join(r"\p{%s}" % sc for sc in scripts_nok) + "]"

    def _lang_id(string):
        res = lang_id_func.predict(string, k=1)
        res = res[0][0].replace("__label__", "")
        return res

    if not src_text or not tgt_text:
        logger.debug("src or tgt empty")
        return False
    if src_text == tgt_text:
        logger.debug("src = tgt")
        return False
    if re.search(r"([^0-9])\1{3}", src_text) or re.search(r"([^0-9])\1{3}", tgt_text):
        logger.debug("too many same char in src or tgt")
        return False
    if re.search(r"(\ .*|.*\ )\1{2}", src_text) or re.search(r"(\ .*|.*\ )\1{2}", tgt_text):
        logger.debug("too many same word in src or tgt")
        return False
    if n_src_words < min_tok or n_tgt_words < min_tok:
        logger.debug(f"num of words < {min_tok}")
        return False
    if n_src_words > max_tok or n_tgt_words > max_tok:
        logger.debug(f"num of words > {max_tok}")
        return False
    if len(src_text) / n_src_words < avg_tok_min or len(tgt_text) / n_tgt_words < avg_tok_min:
        logger.debug(f"avg token < {avg_tok_min}")
        return False
    if len(src_text) / n_src_words > avg_tok_max or len(tgt_text) / n_tgt_words > avg_tok_max:
        logger.debug(f"avg token > {avg_tok_max}")
        return False
    if max(len(x) for x in src_text.split()) > tok_max or max(len(x) for x in tgt_text.split()) > tok_max:
        logger.debug(f"max length of word > {tok_max}")
        return False
    if re.search(ok_regex, src_text) or re.search(ok_regex, tgt_text):
        logger.debug("text does not fully belong to wanted script")
        return False
    # if re.search(nok_regex, src_text) or re.search(nok_regex, tgt_text):
    #     logger.debug("some text belong to unwanted scripts")
    #     return False
    if _lang_id(src_text) not in allowed_src_lang_ids or _lang_id(tgt_text) not in allowed_tgt_lang_ids:
        logger.debug("lang ids don't match")
        return False
    if n_src_words / n_tgt_words > src_tgt_ratio or n_src_words / n_tgt_words < 1 / src_tgt_ratio:
        logger.debug("src / tgt ratio ", n_src_words / n_tgt_words)
        return False
    if (
        src_embedding is not None
        and tgt_embedding is not None
        and 1 - distance.cosine(src_embedding, tgt_embedding) < min_similariry
    ):
        logger.debug(f"semantic similarity < {min_similariry}")
        return False

    return True


def read_instruct(file_path, src="en", tgt="fr"):
    source, target = language_mappings[src][src], language_mappings[src][tgt]
    with open(file_path) as f:
        ins_list = [line.strip().replace("[SRC]", source).replace("[TGT]", target) for line in f]
    return ins_list


def main(
    instruction_file_path: str,
    output_file: str,
    preprocessing_num_workers: int = 4,
    embedding_model_name_or_path: Optional[str] = None,
    embedding_batch_size: int = 64,
):
    if embedding_model_name_or_path is not None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            raise ModuleNotFoundError(
                "Please install sentence-transformers first, e.g., with `pip install -U sentence-transformers`"
            )

    # Load from parallel files
    # source_lang_dataset = load_dataset("text", data_files=source_lang_file)["train"]
    # target_lang_dataset = load_dataset("text", data_files=target_lang_file)["train"]
    # assert (
    #     source_lang_dataset.num_rows == target_lang_dataset.num_rows
    # ), f"Num of sentences are unmatched: {source_lang_dataset.num_rows} vs {target_lang_dataset.num_rows}"
    # print(f"Loaded {source_lang_dataset.num_rows} sentence pairs")

    # source_lang_df = source_lang_dataset.to_pandas()
    # target_lang_df = target_lang_dataset.to_pandas()
    # data_df = pd.concat([source_lang_df, target_lang_df], axis=1)
    # data_df.columns = ["src_lang_text", "tgt_lang_text"]
    # raw_dataset = Dataset.from_pandas(data_df)
    # print(raw_dataset)

    raw_dataset = load_dataset("wmt14", "fr-en", split="train")
    print(f"Loaded {raw_dataset.num_rows:,d} examples")

    # debug
    # raw_dataset = raw_dataset.select(range(10))

    # tmp
    # raw_dataset = raw_dataset.shuffle(seed=10).select(range(1_000_000))
    # raw_dataset = raw_dataset.shuffle(seed=10).select(range(1_000_000, 2_000_000))
    print(f"Sampled {raw_dataset.num_rows:,d} examples")

    # reformat
    processed_dataset = raw_dataset.map(
        lambda x: {"en_text": x["translation"]["en"], "fr_text": x["translation"]["fr"]},
        remove_columns=raw_dataset.column_names,
        num_proc=preprocessing_num_workers,
        desc="reformat data",
    )

    # dedup by en_text
    processed_dataset = deduplicate_dataset(processed_dataset, "en_text", num_proc=preprocessing_num_workers)
    print(f"Deduped to {processed_dataset.num_rows:,d}")

    # clean mt data
    if embedding_model_name_or_path is not None:
        # embedding model
        model = SentenceTransformer(embedding_model_name_or_path)

        processed_dataset = processed_dataset.map(
            lambda example: {
                "en_embedding": model.encode(
                    example["en_text"], batch_size=embedding_batch_size, show_progress_bar=False
                ),
                "fr_embedding": model.encode(
                    example["fr_text"], batch_size=embedding_batch_size, show_progress_bar=False
                ),
            },
            batched=True,
            batch_size=embedding_batch_size,
            desc="compute embedding",
        )

    input_columns = ["en_text", "fr_text"]
    if embedding_model_name_or_path is not None:
        input_columns += ["en_embedding", "fr_embedding"]
    processed_dataset = processed_dataset.filter(
        filter_function,
        input_columns=input_columns,
        # fn_kwargs=,
        num_proc=preprocessing_num_workers,
        desc="filter data",
    )

    if embedding_model_name_or_path is not None:
        processed_dataset = processed_dataset.remove_columns(["en_embedding", "fr_embedding"])

    print(f"Cleaned to {processed_dataset.num_rows:,d}")

    # format to chat
    # todo
    instructions = read_instruct(instruction_file_path)

    def process_function(example):
        # en -> fr
        instruction = random.choice(instructions) + "\n\n" + example["en_text"]
        conversation = Conversation(
            # id=example["id"],
            # system=CONVERSATION_SYSTEM_MESSAGE_EN,
            system=CONVERSATION_SYSTEM_MESSAGE_FR,
            messages=[
                Utterance(role=Role.user, content=instruction),
                Utterance(role=Role.assistant, content=example["fr_text"]),
            ],
        )

        # fr -> en
        # instruction = random.choice(instructions) + "\n\n" + example["fr_text"]
        # conversation = Conversation(
        #     # id=example["id"],
        #     system=CONVERSATION_SYSTEM_MESSAGE_FR,
        #     messages=[
        #         Utterance(role=Role.user, content=instruction),
        #         Utterance(role=Role.assistant, content=example["en_text"]),
        #     ],
        # )

        return conversation.fully_model_dump()

    processed_dataset = processed_dataset.map(
        process_function,
        remove_columns=processed_dataset.column_names,
        num_proc=preprocessing_num_workers,
        desc="format data",
    )
    print(processed_dataset)

    # save
    processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"The processed data is saved into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
