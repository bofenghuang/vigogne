#! /usr/bin/env python
# coding=utf-8

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import bitsandbytes as bnb
import fire
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

# Original English instruct format
# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:\n"
#     ),
# }
# French instruct translated by chatgpt
PROMPT_DICT = {
    "prompt_input": (
        "Ci-dessous se trouve une instruction qui décrit une tâche, associée à une entrée qui fournit un contexte supplémentaire. Écrivez une réponse qui complète correctement la demande.\n\n"
        "### Instruction:\n{instruction}\n\n### Entrée:\n{input}\n\n### Réponse:\n"
    ),
    "prompt_no_input": (
        "Ci-dessous se trouve une instruction qui décrit une tâche. Écrivez une réponse qui complète correctement la demande.\n\n"
        "### Instruction:\n{instruction}\n\n### Réponse:\n"
    ),
}


def generate_prompt(data_point):
    return (
        PROMPT_DICT["prompt_input"].format_map(data_point)
        if data_point["input"]
        else PROMPT_DICT["prompt_no_input"].format_map(data_point)
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
            # int(math.ceil
            n_padding = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of - max_length
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


def train(
    model_name_or_path: str,
    output_dir: str,
    data_path: str,
    val_set_size: int = 2000,
    model_max_length: int = 256,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: List[str] = ["q_proj", "v_proj"],
    num_train_epochs: int = 3,
    learning_rate: float = 3e-4,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 32,
    **kwargs,
):

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_8bit=True,
        device_map=device_map,
    )

    # todo: better handle
    tokenizer_class = LlamaTokenizer if "llama" in model_name_or_path else AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        model_name_or_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    # Freeze the model parameters
    # Cast the small parameters (e.g. layernorm) to fp32 for stability
    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Load data
    data = load_dataset("json", data_files=data_path)

    def generate_and_tokenize_prompt(data_point):
        # Format prompt
        user_prompt = generate_prompt(data_point)

        # Get prompt length for masking
        len_user_prompt_tokens = len(tokenizer(user_prompt, truncation=True)["input_ids"])

        # Tokenize
        # todo: need eos?
        input_ids = tokenizer(user_prompt + data_point["output"] + tokenizer.eos_token, truncation=True)["input_ids"]
        # Mask prompt
        labels = [IGNORE_INDEX] * len_user_prompt_tokens + input_ids[len_user_prompt_tokens:]

        # Tokenize
        # input_ids = tokenizer(user_prompt + data_point["output"] + tokenizer.eos_token, truncation=True, return_tensors="pt")["input_ids"][0]
        # labels = input_ids.clone()
        # Mask prompt
        # labels[:len_user_prompt_tokens] = IGNORE_INDEX

        return {"input_ids": input_ids, "labels": labels}

    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
        val_data = train_val["test"].map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=True,
            output_dir=output_dir,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            **kwargs,
        ),
        data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer, pad_to_multiple_of=8),
    )
    print(trainer.args)

    # Silence the warnings. Please re-enable for inference!
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)

    # debug
    # train(
    #     model_name_or_path="decapoda-research/llama-7b-hf",
    #     data_path="data/tmp_vigogne_data_cleaned_head10.json",
    #     val_set_size=2,
    #     model_max_length=256,
    #     output_dir="outputs/tmp",
    #     lora_r=8,
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     target_modules=["q_proj", "v_proj"],
    #     num_train_epochs=3,
    #     optim="adamw_torch",
    #     learning_rate=3e-4,
    #     per_device_train_batch_size=4,
    #     per_device_eval_batch_size=2,
    #     gradient_accumulation_steps=1,
    #     warmup_steps=100,
    #     logging_steps=1,
    #     save_total_limit=3,
    #     save_strategy="steps",
    #     save_steps=200,
    #     evaluation_strategy="steps",
    #     eval_steps=200
    # )
