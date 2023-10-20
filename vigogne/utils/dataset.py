# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Load and process datasets."""

import logging
import random
import sys
from pathlib import Path
from typing import Any, Union

import numpy as np
import transformers
from datasets import Dataset, DatasetDict, load_dataset

from ..data_utils import IGNORE_INDEX
from ..processors import SUPPORTED_PROCESSORS
from .packing import ModerateConcatenator

logger = logging.getLogger(__name__)


def prepare_datasets(cfg: Any, tokenizer: transformers.PreTrainedTokenizerBase):
    # load datasets
    dataset = load_datasets(cfg)
    # process datasets
    processed_dataset = process_datasets(cfg, dataset, tokenizer)
    # filter datasets
    filtered_dataset = filter_datasets(cfg, processed_dataset)
    # Count tokens
    final_dataset = get_num_tokens(cfg, filtered_dataset)

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


def _get_ds_type(file_path: str):
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
    """Load datasets from local files."""

    # todo: add ds from hf hub

    logger.info("Loading datasets...")

    with cfg.main_process_first():
        dataset = DatasetDict()
        if cfg.train_file is not None:
            ds_type = _get_ds_type(cfg.train_file)
            dataset["train"] = load_dataset(ds_type, data_files=cfg.train_file)["train"]
        else:
            raise ValueError("You have not specified any train file")

        if cfg.eval_file is not None:
            ds_type = _get_ds_type(cfg.eval_file)
            dataset["eval"] = load_dataset(ds_type, data_files=cfg.eval_file)["train"]

        elif cfg.eval_split_ratio is not None:
            logger.info(f"Splitting train/eval set with a eval_split_ratio of {cfg.eval_split_ratio}")
            dataset = dataset["train"].train_test_split(test_size=cfg.eval_split_ratio, shuffle=True, seed=cfg.seed)
            dataset["eval"] = dataset.pop("test")

    logger.info(
        f'Training set -> num_rows: {dataset["train"].num_rows:,d}, features: {", ".join(dataset["train"].column_names)}'
    )
    if "eval" in dataset:
        logger.info(
            f'Evaluation set -> num_rows: {dataset["eval"].num_rows:,d}, features: {", ".join(dataset["eval"].column_names)}'
        )

    # Log a few random samples from the training set
    for index in random.sample(range(len(dataset["train"])), 1):
        logger.info(f'Sample {index} of the training set: {dataset["train"][index]}.')

    return dataset


def process_datasets(cfg: Any, dataset: Union[Dataset, DatasetDict], tokenizer: transformers.PreTrainedTokenizerBase):
    """Process datasets by processor."""

    logger.info(f"Processing datasets with {cfg.processor_style} prompter...")

    processor = SUPPORTED_PROCESSORS.get(cfg.processor_style)

    with cfg.main_process_first():
        processed_dataset = dataset.map(
            processor.process_example,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=cfg.preprocessing_num_workers,
            remove_columns=next(iter(dataset.values())).column_names,
            load_from_cache_file=not cfg.overwrite_cache,
            desc="process dataset",
        )

    # assign for saving
    if (default_chat_template := getattr(processor, "default_chat_template", None)) is not None:
        tokenizer.chat_template = default_chat_template()

    return processed_dataset


def get_example_length_in_datasets(cfg: Any, dataset: Union[Dataset, DatasetDict]):
    """Get examples' lengths in datasets."""

    logger.info("Get example lengths in datasets...")

    with cfg.main_process_first():
        processed_dataset = dataset.map(
            lambda example: {cfg.length_column_name: len(example["input_ids"])},
            num_proc=cfg.preprocessing_num_workers,
            load_from_cache_file=not cfg.overwrite_cache,
            desc="get example length",
        )

    return processed_dataset


def filter_datasets(cfg: Any, dataset: Union[Dataset, DatasetDict]):
    """Filter data that is shorter than model_min_length or longer than model_max_length."""

    def _in_length_range(length):
        # return length > data_args.model_min_length and length < data_args.model_max_length
        is_in_range = True
        if cfg.model_min_length is not None:
            is_in_range &= length > cfg.model_min_length
        if cfg.model_max_length is not None:
            is_in_range &= length < cfg.model_max_length
        return is_in_range

    if cfg.model_min_length is None and cfg.model_max_length is None:
        return dataset

    dataset = get_example_length_in_datasets(cfg, dataset)

    logger.info("Filtering datasets by length...")

    with cfg.main_process_first():
        processed_dataset = dataset.filter(
            _in_length_range,
            num_proc=cfg.preprocessing_num_workers,
            input_columns=[cfg.length_column_name],
            load_from_cache_file=not cfg.overwrite_cache,
            desc="filter dataset",
        )

        # Remove length column
        processed_dataset = processed_dataset.remove_columns(cfg.length_column_name)

    logger.info(f'Filtered training set to {processed_dataset["train"].num_rows:,d} rows')
    if "eval" in processed_dataset:
        logger.info(f'Filtered evaluation set to {processed_dataset["eval"].num_rows:,d} rows')

    return processed_dataset


def get_num_tokens(cfg: Any, dataset: Union[Dataset, DatasetDict]):
    """Counting number of tokens and supervised tokens."""

    def _process_function(example):
        return {
            "num_tokens": len(example["input_ids"]),
            "num_supervised_tokens": sum(np.array(example["labels"]) != IGNORE_INDEX),
        }

    logger.info("Counting total number of tokens...")

    with cfg.main_process_first():
        length_data = dataset.map(
            _process_function,
            num_proc=cfg.preprocessing_num_workers,
            remove_columns=next(iter(dataset.values())).column_names,
            load_from_cache_file=not cfg.overwrite_cache,
            desc="count tokens",
        )

    cfg.num_training_tokens = np.sum(length_data["train"]["num_tokens"])
    cfg.num_training_supervised_tokens = np.sum(length_data["train"]["num_supervised_tokens"])

    logger.info(
        f"Training set -> num_tokens: {cfg.num_training_tokens:,d}, num_supervised_tokens:"
        f" {cfg.num_training_supervised_tokens:,d}"
    )
    if "eval" in dataset:
        cfg.num_eval_tokens = np.sum(length_data["eval"]["num_tokens"])
        cfg.num_eval_supervised_tokens = np.sum(length_data["eval"]["num_supervised_tokens"])
        logger.info(
            f"Evaluation set -> num_tokens: {cfg.num_eval_tokens:,d}, num_supervised_tokens: {cfg.num_eval_supervised_tokens:,d}"
        )

    return dataset
