#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

# NB
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.
# Need to call this before importing transformers.
# from vigogne.model.llama_flash_attn_monkey_patch import replace_attn_with_flash_attn

# replace_attn_with_flash_attn()

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional

import bitsandbytes as bnb
import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint, set_seed

import vigogne
from vigogne.constants import CHAT, INSTRUCT, VALID_MODES
from vigogne.train.utils import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_UNK_TOKEN,
    SUPPORTED_PROCESSOR_TEMPLATES,
    DataCollatorForSupervisedDataset,
    LoadBestPeftModelCallback,
    ModerateConcatenator,
    SavePeftModelCallback,
    print_trainable_parameters,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Base model parameters."""

    model_name_or_path: Optional[str] = field(default=None)
    compute_dtype: str = field(
        default="float16",
        metadata={
            "help": "Load the model under a specific `dtype`. Use `bfloat16` if you have NVIDIA architecture higher than Ampere to accelerate."
        },
    )
    load_in_8bit: bool = field(
        default=False, metadata={"help": "Whether to convert the loaded model into mixed-8bit quantized model."}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "Whether to convert the loaded model into mixed-4bit quantized model."}
    )
    use_nested_quant: bool = field(
        default=False,
        metadata={
            "help": "Nested quantization where the quantization constants from the first quantization are quantized again. Only used 4bit base models."
        },
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type in the bnb.nn.Linear4Bit layers, which are specified by `fp4` or `nf4`. Only used 4bit base models."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                " should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                " execute code present on the Hub on your local machine."
            )
        },
    )

    def __post_init__(self):
        assert not (
            self.load_in_8bit and self.load_in_4bit
        ), "You can't pass both `load_in_8bit=True` and `load_in_4bit=True`"


@dataclass
class LoraArguments:
    """LoRA parameters."""

    # todo: add modules_to_save
    lora_r: int = field(default=8, metadata={"help": "Lora rank."})
    lora_alpha: int = field(default=16, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout."})
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"], metadata={"help": "Names of the modules to apply Lora to."}
    )


@dataclass
class DataArguments:
    """Data parameters."""

    train_file: Optional[str] = field(default=None, metadata={"help": "Path to the training file."})
    eval_file: Optional[str] = field(default=None, metadata={"help": "Path to the evaluation file."})
    model_min_length: Optional[int] = field(
        default=None, metadata={"help": "Filter examples that have fewer than `model_min_length` tokens"}
    )
    model_max_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # model_max_length_percentile: Optional[int] = field(
    #     default=95, metadata={"help": "Percentile of the example length. Used to determin `model_max_length`."}
    # )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    mode: str = field(default=INSTRUCT, metadata={"help": "The mode to preprocess and format the data."})
    pack_into_block: bool = field(
        default=False,
        metadata={
            "help": "Whether to pack examples into blocks for faster training. Note this might affect performance."
        },
    )
    block_size: int = field(
        default=1024, metadata={"help": "Block size for packed examples. Only used when `pack_into_block` is True."}
    )

    def __post_init__(self):
        assert self.mode in VALID_MODES, f"`mode` should be chosen in {VALID_MODES}"


@dataclass
class VigogneTrainingArguments(TrainingArguments):
    # todo: remove when adding bf16
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training."},
    )
    length_column_name: Optional[str] = field(
        default="input_length",
        metadata={"help": "Column name with precomputed lengths to use when grouping by length."},
    )


# Copied from https://github.com/tatsu-lab/stanford_alpaca/blob/eb5b171d9b103a12a8e14e0edca9cbc45fe1d512/train.py#L75-L95
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        logger.info(f"Added {num_new_tokens} new special tokens to the model")


def train():
    # debug
    # def train(args):
    # HF parser
    parser = HfArgumentParser((ModelArguments, LoraArguments, DataArguments, VigogneTrainingArguments))
    model_args, lora_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # debug
    # model_args, lora_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    vigogne.train.utils.set_verbosity(log_level)
    logger.setLevel(log_level)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Model parameters {model_args}")
    logger.info(f"LoRA parameters {lora_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model
    set_seed(training_args.seed)

    # Compute type
    compute_dtype = getattr(torch, model_args.compute_dtype)

    # BitsAndBytesConfig support
    quantization_config = None
    if model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    if model_args.load_in_4bit:
        # todo: customize 4bit config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=compute_dtype,
        device_map={"": Accelerator().process_index},
        quantization_config=quantization_config,
        trust_remote_code=model_args.trust_remote_code,
        use_cache=not training_args.gradient_checkpointing,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
    # Some special tokens can be "" or None depending on releases
    special_tokens_dict = dict()
    if tokenizer.pad_token is None or not tokenizer.pad_token:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None or not tokenizer.eos_token:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None or not tokenizer.bos_token:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None or not tokenizer.unk_token:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if model_args.load_in_8bit or model_args.load_in_4bit:
        # todo
        # model.gradient_checkpointing_enable()
        # model = prepare_model_for_kbit_training(model)

        # Cast the small parameters (e.g. layernorm) to fp32 for stability
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    elif training_args.gradient_checkpointing:
        # For backward compatibility
        # See https://github.com/huggingface/peft/issues/137
        model.enable_input_require_grads()

    # Load LoRA
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # Enable MP, now will be force-set
    # See: https://github.com/tloen/alpaca-lora/pull/131
    # See: https://github.com/huggingface/transformers/pull/22628
    # if not ddp and torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    # Load data
    raw_datasets = DatasetDict()
    if data_args.train_file is not None:
        ext = data_args.train_file.rsplit(".", 1)[-1]
        ext = "json" if ext == "jsonl" else ext
        raw_datasets["train"] = load_dataset(ext, data_files=data_args.train_file)["train"]
    else:
        raise ValueError("You have not specified any train file")
    if data_args.eval_file is not None:
        ext = data_args.eval_file.rsplit(".", 1)[-1]
        ext = "json" if ext == "jsonl" else ext
        raw_datasets["eval"] = load_dataset(ext, data_files=data_args.eval_file)["train"]
    logger.info(f"Raw dataset: {raw_datasets}")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f'Sample {index} of the training set: {raw_datasets["train"][index]}.')

    # Determine model_max_length for truncation
    # model_max_length = data_args.model_max_length
    # get_example_length_function = get_chat_example_length if data_args.mode == CHAT else get_instruct_example_length
    # get_example_length_function_p = partial(get_example_length_function, tokenizer=tokenizer)

    # if model_max_length is None:
    #     with training_args.main_process_first(desc="dataset map tokenization"):
    #         train_example_lengths = raw_datasets["train"].map(
    #             get_example_length_function_p,
    #             num_proc=data_args.preprocessing_num_workers,
    #             remove_columns=next(iter(raw_datasets.values())).column_names,
    #             load_from_cache_file=not data_args.overwrite_cache,
    #             desc="get example lengths",
    #         )["example_length"]
    #     # Take percentile of max length
    #     model_max_length = math.ceil(np.percentile(train_example_lengths, data_args.model_max_length_percentile))
    #     logger.info(
    #         f"`model_max_length` has been set to the {data_args.model_max_length_percentile}th percentile of training example lengths: {model_max_length}"
    #     )

    # Tokenize data
    process_function = SUPPORTED_PROCESSOR_TEMPLATES[data_args.mode].process_example
    # process_function_p = partial(process_function, tokenizer=tokenizer, model_max_length=model_max_length)
    process_function_p = partial(
        process_function, tokenizer=tokenizer, length_column_name=training_args.length_column_name
    )

    with training_args.main_process_first(desc="dataset map tokenization"):
        processed_dataset = raw_datasets.map(
            process_function_p,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="process dataset",
        )

        # Remove long examples
        def is_input_in_length_range(length):
            # return length > data_args.model_min_length and length < data_args.model_max_length
            is_in_range = True
            if data_args.model_min_length is not None:
                is_in_range &= length > data_args.model_min_length
            if data_args.model_max_length is not None:
                is_in_range &= length < data_args.model_max_length
            return is_in_range

        if (
            data_args.model_min_length is not None or data_args.model_max_length is not None
        ) and training_args.length_column_name in next(iter(processed_dataset.values())).column_names:
            # filter data that is shorter than model_min_length or longer than model_max_length
            processed_dataset = processed_dataset.filter(
                is_input_in_length_range,
                num_proc=data_args.preprocessing_num_workers,
                input_columns=[training_args.length_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="filter dataset by input length",
            )

            # Remove length column
            if training_args.length_column_name in next(iter(processed_dataset.values())).column_names:
                processed_dataset = processed_dataset.remove_columns(training_args.length_column_name)

            logger.info(f"Filtered dataset: {processed_dataset}")

    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {processed_dataset.cache_files}")
        return

    train_dataset = processed_dataset["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    # pack (group) examples
    # NB: here we only pack training set, but can also pack eval set depending on use case
    if data_args.pack_into_block:
        block_size = min(data_args.block_size, tokenizer.model_max_length)
        with training_args.main_process_first(desc="packing samples together"):
            train_dataset = train_dataset.map(
                ModerateConcatenator(block_size=block_size),
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"packing texts in blocks of {block_size}",
            )

    if data_args.eval_file is not None:
        eval_dataset = processed_dataset["eval"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Init trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if data_args.eval_file is not None else None,
        args=training_args,
        data_collator=DataCollatorForSupervisedDataset(
            tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        ),
        callbacks=[SavePeftModelCallback, LoadBestPeftModelCallback],
    )

    # Comment to stay with new version of PEFT
    # old_state_dict = model.state_dict
    # model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        # train_result = trainer.train()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # Saves the tokenizer too for easy upload
        # trainer.save_model()
        model.save_pretrained(training_args.output_dir)

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        # trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card or push to hf hub
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    train()
