#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

"""
Post-process translated data.

Usage:
python scripts/data_processing/process_translated_data.py \
    --input_file data/alpaca_data_cleaned.jsonl \
    --output_file data/alpaca_data_cleaned_fr.jsonl \
    --src_column instruction \
    --tgt_column translated_instruction \
    --embedding_model_name_or_path paraphrase-multilingual-mpnet-base-v2 \
    --embedding_batch_size 64
"""

import logging
import re
import zlib
from collections import defaultdict
from typing import Any, Optional

import fire
import tiktoken
from datasets import load_dataset
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# logger = logging.getLogger(__name__)

# fmt = "%(asctime)s [%(levelname)s] [%(name)s] [%(funcName)s] %(message)s"
# datefmt = "%Y-%m-%dT%H:%M:%SZ"
# logging.basicConfig(format=fmt, datefmt=datefmt, level=logging.INFO)


# Copied from https://github.com/openai/whisper/blob/0a60fcaa9b86748389a656aa013c416030287d47/whisper/utils.py#L45
def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def filter_function(
    src_text: str,
    tgt_text: str,
    src_length: Optional[int] = None,
    tgt_length: Optional[int] = None,
    src_embedding: Any = None,
    tgt_embedding: Any = None,
    tgt_min_tokens: int = 4,
    tgt_max_tokens: int = 1024,
    src_tgt_tokens_ratio: int = 2,
    tgt_max_compression_ratio: float = 2.4,
    src_min_compression_ratio_to_ignore_tgt_compression_rate: float = 2,
    min_cos_similariry: float = 0.8,
    filtered_counter: Any = None,
    **kwargs,
):
    """
    Heuristic for filtering out poorly translated text.
    To add:
    - Remove tgt_text by lang
    """

    if src_length is None or tgt_length is None:
        src_length = len(src_text.split())
        tgt_length = len(tgt_text.split())

    if not tgt_text:
        filtered_counter["empty_tgt"] += 1
        # logger.debug("Empty tgt_text")
        return False
    if src_text == tgt_text:
        filtered_counter["same_text"] += 1
        # logger.debug("src_text == tgt_text")
        return False
    if tgt_length < tgt_min_tokens:
        filtered_counter["short_tgt"] += 1
        # logger.debug(f"tgt_length is {tgt_length} and smaller than tgt_min_tokens {tgt_min_tokens}")
        return False
    if tgt_length > tgt_max_tokens:
        filtered_counter["long_tgt"] += 1
        # logger.debug(f"tgt_length is {tgt_length} and bigger than tgt_max_tokens {tgt_max_tokens}")
        return False
    if src_length / tgt_length > src_tgt_tokens_ratio or src_length / tgt_length < 1 / src_tgt_tokens_ratio:
        filtered_counter["diff_length"] += 1
        # logger.debug(f"src_length / tgt_length is {src_length / tgt_length}")
        return False
    if (
        compression_ratio(src_text) < src_min_compression_ratio_to_ignore_tgt_compression_rate
        and compression_ratio(tgt_text) > tgt_max_compression_ratio
    ):
        filtered_counter["hight_compression_rate"] += 1
        # logger.debug(
        #     f"src_compression_rate is {compression_ratio(src_text)} and tgt_compression_rate is {compression_ratio(tgt_text)}"
        # )
        return False
    if (
        src_embedding is not None
        and tgt_embedding is not None
        and 1 - distance.cosine(src_embedding, tgt_embedding) < min_cos_similariry
    ):
        filtered_counter["low_cos_sim"] += 1
        # logger.debug("Sim")
        return False

    # dataset specific ops
    if re.search(r"Traduis le texte suivant en français", tgt_text):
        filtered_counter["bad_translation"] += 1
        return False
    # meta-math
    # if re.search(r"\[asy\]", tgt_text):
    #     return False

    return True


def normalize_text(s):
    # sometimes translated text got " at start and end
    if s.startswith('"'):
        s = s.strip('"')

    return s


def main(
    input_file: str,
    output_file: str,
    src_column: str = "instruction",
    tgt_column: str = "translated_instruction",
    tokenizer_name_or_path: str = "gpt-3.5-turbo",
    embedding_model_name_or_path: Optional[str] = None,
    embedding_batch_size: int = 64,
    preprocessing_num_workers: int = 32,
    **kwargs,
):
    dataset = load_dataset("json", data_files=input_file, split="train")
    print(f"Loaded {dataset.num_rows:,d} examples from {input_file}")

    column_names = dataset.column_names

    # debug
    # dataset = dataset.select(range(10))

    # NB: normalize translated text
    dataset = dataset.map(
        lambda x: {tgt_column: normalize_text(x[tgt_column])},
        num_proc=preprocessing_num_workers,
        desc="normalize text",
    )

    print(dataset.to_pandas()["finish_reason"].value_counts().to_dict())
    dataset = dataset.filter(
        lambda x: x["finish_reason"] == "stop",
        num_proc=preprocessing_num_workers,
        desc="filter by finish_reason",
    )
    print(f"Filtered to {dataset.num_rows:,d} examples")
    dataset = dataset.remove_columns("finish_reason")

    # hf tokenizer
    # tokenizer_name_or_path = "meta-llama/Llama-2-7b-hf"
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    # openai tokenizer
    # tokenizer_name_or_path = "gpt-3.5-turbo"
    tokenizer = tiktoken.encoding_for_model(tokenizer_name_or_path)

    def get_num_tokens(s):
        # return len(tokenizer(s)["input_ids"])
        return len(tokenizer.encode(s))

    dataset = dataset.map(
        lambda example: {
            f"{src_column}_length": get_num_tokens(example[src_column]),
            f"{tgt_column}_length": get_num_tokens(example[tgt_column]),
        },
        # num_proc=preprocessing_num_workers,
        num_proc=1,  # todo: bug?
        desc="get length",
    )

    if embedding_model_name_or_path is not None:
        st_model = SentenceTransformer(embedding_model_name_or_path, device="cuda")
        # todo: longer
        st_model.max_seq_length = 512

        dataset = dataset.map(
            lambda example: {
                f"{src_column}_embedding": st_model.encode(
                    example[src_column], batch_size=embedding_batch_size, show_progress_bar=False
                ),
                f"{tgt_column}_embedding": st_model.encode(
                    example[tgt_column], batch_size=embedding_batch_size, show_progress_bar=False
                ),
            },
            batched=True,
            batch_size=embedding_batch_size,
            desc="compute embedding",
        )

    input_columns = [src_column, tgt_column, f"{src_column}_length", f"{tgt_column}_length"]
    if embedding_model_name_or_path is not None:
        input_columns += [f"{src_column}_embedding", f"{tgt_column}_embedding"]

    filtered_counter = defaultdict(int)

    dataset = dataset.filter(
        filter_function,
        # lambda *x, **y: not filter_function(*x, **y),  # debug
        input_columns=input_columns,
        fn_kwargs={"filtered_counter": filtered_counter, **kwargs},
        load_from_cache_file=False,
        desc="filter data",
    )
    print(f"Filtered to {dataset.num_rows:,d} examples")
    print(f"Filtered out counts: {filtered_counter}")

    dataset = dataset.remove_columns(
        [column_name for column_name in dataset.column_names if column_name not in column_names]
    )

    # export
    dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"Saved data into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
