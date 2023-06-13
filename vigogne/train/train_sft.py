#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Sequence

import bitsandbytes as bnb
import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model, 
    get_peft_model_state_dict, 
    prepare_model_for_kbit_training
)
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments

from vigogne.constants import (
    CHAT,
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_UNK_TOKEN,
    IGNORE_INDEX,
    INSTRUCT,
    VALID_MODES,
)
from vigogne.preprocess import (
    get_chat_example_length,
    get_instruct_example_length,
    preprocess_chat_example,
    preprocess_instruct_example,
)
from vigogne.train.training_utils import LoadBestPeftModelCallback, SavePeftModelCallback, print_trainable_parameters

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Base model parameters."""

    model_name_or_path: Optional[str] = field(default=None)
    load_in_8bit: bool = field(
        default=False, metadata={"help": "Whether to convert the loaded model into mixed-8bit quantized model."}
    )


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
    model_max_length: Optional[int] = field(
        default=None, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}
    )
    model_max_length_percentile: Optional[int] = field(
        default=95, metadata={"help": "Percentile of the example length. Used to determin `model_max_length`."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this " "value if set."
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
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    mode: str = field(default=INSTRUCT, metadata={"help": "The mode to preprocess and format the data."})

    def __post_init__(self):
        assert self.mode in VALID_MODES, f"`mode` should be chosen in {VALID_MODES}"


@dataclass
class VigogneTrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer to use."})
    fp16: bool = field(
        default=True, metadata={"help": "Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training."}
    )


# Modified from: https://github.com/bofenghuang/stanford_alpaca/blob/eb5b171d9b103a12a8e14e0edca9cbc45fe1d512/train.py#L166-L182
# Almost same to transformers.DataCollatorForSeq2Seq
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # dtype = torch.long
        # input_ids, labels = tuple([torch.LongTensor(instance[key]) for instance in instances] for key in ("input_ids", "labels"))
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        if self.pad_to_multiple_of is not None:
            max_length_index, max_length = max(enumerate([len(input_ids_) for input_ids_ in input_ids]), key=lambda x: x[1])
            # n_padding = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of - max_length
            n_padding = math.ceil(max_length / self.pad_to_multiple_of) * self.pad_to_multiple_of - max_length
            # Pad the longest example to pad_to_multiple_of * N
            input_ids[max_length_index].extend([self.tokenizer.pad_token_id] * n_padding)
            labels[max_length_index].extend([IGNORE_INDEX] * n_padding)

        input_ids = [torch.LongTensor(input_ids_) for input_ids_ in input_ids]
        labels = [torch.LongTensor(labels_) for labels_ in labels]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# Copied from https://github.com/bofenghuang/stanford_alpaca/blob/eb5b171d9b103a12a8e14e0edca9cbc45fe1d512/train.py#L75-L95
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


def train():
    # HF parser
    parser = HfArgumentParser((ModelArguments, LoraArguments, DataArguments, VigogneTrainingArguments))
    model_args, lora_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # debug
    # model_args, lora_args, data_args, training_args = parser.parse_args_into_dataclasses(args)

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
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

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

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        load_in_8bit=model_args.load_in_8bit,
        device_map={"": Accelerator().process_index},
        use_cache=not training_args.gradient_checkpointing,
    )

    # fast version llama for "with you.</s>"
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
    # llama has no pad token, like gpt2
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

    if model_args.load_in_8bit:
        # Cast the small parameters (e.g. layernorm) to fp32 for stability
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    elif training_args.gradient_checkpointing:
        # For backward compatibility
        # See https://github.com/huggingface/peft/issues/137
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
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
    # logger.info(raw_datasets)

    # Determine model_max_length for truncation
    model_max_length = data_args.model_max_length
    get_example_length_function = get_chat_example_length if data_args.mode == CHAT else get_instruct_example_length
    get_example_length_function_p = partial(get_example_length_function, tokenizer=tokenizer)

    if model_max_length is None:
        with training_args.main_process_first(desc="dataset map tokenization"):
            train_example_lengths = raw_datasets["train"].map(
                get_example_length_function_p,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=next(iter(raw_datasets.values())).column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="get example lengths",
            )["example_length"]
        # Take percentile of max length
        model_max_length = math.ceil(np.percentile(train_example_lengths, data_args.model_max_length_percentile))
        logger.info(
            f"`model_max_length` has been set to the {data_args.model_max_length_percentile}th percentile of training example lengths: {model_max_length}"
        )

    # Tokenize data
    preprocess_function = preprocess_chat_example if data_args.mode == CHAT else preprocess_instruct_example
    preprocess_function_p = partial(preprocess_function, tokenizer=tokenizer, model_max_length=data_args.model_max_length)

    with training_args.main_process_first(desc="dataset map tokenization"):
        preprocessed_dataset = raw_datasets.map(
            preprocess_function_p,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="preprocess dataset",
        )

    train_dataset = preprocessed_dataset["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    if data_args.eval_file is not None:
        eval_dataset = preprocessed_dataset["eval"]
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

        train_result = trainer.train()

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
