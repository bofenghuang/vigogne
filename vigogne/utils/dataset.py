# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Load and process datasets"""

import logging
import random
import sys
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import transformers
from datasets import Dataset, DatasetDict, load_dataset

from ..processors import SUPPORTED_PROCESSORS
from .packing import ModerateConcatenator

logger = logging.getLogger(__name__)


def prepare_datasets(cfg: Any, tokenizer: transformers.PreTrainedTokenizerBase):
    # load datasets
    dataset = load_datasets(cfg)
    # process datasets
    processed_dataset = process_datasets(cfg, dataset, tokenizer)
    # filter datasets
    filtered_dataset = filter_datasets(cfg, processed_dataset, tokenizer)
    # Count tokens
    final_dataset = count_total_tokens(cfg, filtered_dataset)

    train_dataset = final_dataset["train"]
    if cfg.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), cfg.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    # todo
    # pack (group) examples
    # only pack training set
    if cfg.pack_into_block:
        block_size = min(cfg.block_size, tokenizer.model_max_length)
        with cfg.main_process_first(desc="packing samples together"):
            # shuffle examples before packing
            train_dataset = train_dataset.shuffle(seed=cfg.seed).map(
                ModerateConcatenator(block_size=block_size),
                batched=True,
                load_from_cache_file=not cfg.overwrite_cache,
                desc=f"packing texts in blocks of {block_size}",
            )

    eval_dataset = final_dataset.get("eval")
    if eval_dataset is not None and cfg.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), cfg.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    if cfg.preprocessing_only:
        logger.info(f"Data processing finished. Files cached at {final_dataset.cache_files}")
        sys.exit()

    return train_dataset, eval_dataset


def get_ds_type(file_path: str):
    # extension = file_path.rsplit(".", 1)[-1]
    extension = Path(file_path).suffix
    if ".json" in extension or ".jsonl" in extension:
        return "json"
    elif ".parquet" in extension:
        return "parquet"
    elif ".arrow" in extension:
        return "arrow"
    elif ".csv" in extension:
        return "csv"
    elif ".txt" in extension:
        return "text"
    raise ValueError(f"Cannot handle file extension {extension}")


def load_datasets(cfg: Any):
    # todo: add ds from hf hub
    dataset = DatasetDict()
    if cfg.train_file is not None:
        ds_type = get_ds_type(cfg.train_file)
        dataset["train"] = load_dataset(ds_type, data_files=cfg.train_file)["train"]
    else:
        raise ValueError("You have not specified any train file")

    if cfg.eval_file is not None:
        ds_type = get_ds_type(cfg.eval_file)
        dataset["eval"] = load_dataset(ds_type, data_files=cfg.eval_file)["train"]

    logger.info(f"Raw dataset: {dataset}")

    # Log a few random samples from the training set
    for index in random.sample(range(len(dataset["train"])), 3):
        logger.info(f'Sample {index} of the training set: {dataset["train"][index]}.')

    return dataset


def process_datasets(cfg: Any, dataset: Union[Dataset, DatasetDict], tokenizer: transformers.PreTrainedTokenizerBase):
    process_function = SUPPORTED_PROCESSORS.get(cfg.processor_style).process_example

    with cfg.main_process_first():
        processed_dataset = dataset.map(
            process_function,
            fn_kwargs={"tokenizer": tokenizer, "length_column_name": cfg.length_column_name},
            num_proc=cfg.preprocessing_num_workers,
            remove_columns=next(iter(dataset.values())).column_names,
            load_from_cache_file=not cfg.overwrite_cache,
            desc="process dataset",
        )

    return processed_dataset


def filter_datasets(cfg: Any, dataset: Union[Dataset, DatasetDict], tokenizer: transformers.PreTrainedTokenizerBase):
    """Filter data that is shorter than model_min_length or longer than model_max_length."""

    def _in_length_range(length):
        # return length > data_args.model_min_length and length < data_args.model_max_length
        is_in_range = True
        if cfg.model_min_length is not None:
            is_in_range &= length > cfg.model_min_length
        if cfg.model_max_length is not None:
            is_in_range &= length < cfg.model_max_length
        return is_in_range

    if (cfg.model_min_length is None and cfg.model_max_length is None) or (
        cfg.length_column_name not in next(iter(dataset.values())).column_names
    ):
        return dataset

    with cfg.main_process_first():
        processed_dataset = dataset.filter(
            _in_length_range,
            num_proc=cfg.preprocessing_num_workers,
            input_columns=[cfg.length_column_name],
            load_from_cache_file=not cfg.overwrite_cache,
            desc="filter dataset",
        )

        logger.info(f"Filtered dataset: {processed_dataset}")

    return processed_dataset


def count_total_tokens(cfg: Any, dataset: Union[Dataset, DatasetDict]):
    training_num_tokens = np.sum(dataset["train"][cfg.length_column_name])
    logger.info(f"Total tokens in training set: {training_num_tokens:,d}")

    if (eval_dataset := dataset.get("eval")) is not None:
        eval_num_tokens = np.sum(eval_dataset[cfg.length_column_name])
        logger.info(f"Total tokens in eval set: {eval_num_tokens:,d}")

    # Remove length column
    processed_dataset = dataset.remove_columns(cfg.length_column_name)

    return processed_dataset
