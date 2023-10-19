# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Load models."""

import logging
import os
import shutil
from collections import defaultdict
from typing import Any, Dict

import bitsandbytes as bnb
import torch
import transformers
from peft import (
    AutoPeftModelForCausalLM,
    AutoPeftModelForSeq2SeqLM,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import QuantLinear
from peft.utils import CONFIG_NAME as PEFT_CONFIG_NAME
from peft.utils import SAFETENSORS_WEIGHTS_NAME as PEFT_SAFETENSORS_WEIGHTS_NAME
from peft.utils import WEIGHTS_NAME as PEFT_WEIGHTS_NAME
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig

from vigogne.data_utils import DECODER

PEFT_FOLDER_NAME = "adapter"

logger = logging.getLogger(__name__)


def load_model(cfg: Any, tokenizer: transformers.PreTrainedTokenizerBase):
    """Load Transformer models."""

    logger.info("Loading model...")

    model_kwargs = {
        "revision": cfg.model_revision,
    }
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    # todo: load config then patch
    # todo: add GTPQ

    torch_dtype = getattr(torch, cfg.torch_dtype)

    if torch_dtype == torch.float16 and torch.cuda.get_device_capability()[0] >= 8:
        logger.warning("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")

    # BitsAndBytesConfig support
    # todo: fast peft without kaiming init
    # See https://twitter.com/jeremyphoward/status/1709601456620515638
    if cfg.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            # llm_int8_threshold=6.0,
            # llm_int8_has_fp16_weight=False,
        )
    if cfg.adapter == "qlora" and cfg.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )

    # Load model
    model_cls = AutoModelForCausalLM if cfg.model_type == DECODER else AutoModelForSeq2SeqLM
    model = model_cls.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.trust_remote_code,
        use_flash_attention_2=cfg.use_flash_attention_2,
        **model_kwargs,
    )

    # Resize token_embeddings
    # todo: to 32x
    # todo: model.tie_weights()
    if (num_new_tokens := len(tokenizer) - model.get_input_embeddings().num_embeddings) > 0:
        model.resize_token_embeddings(len(tokenizer))

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        logger.info(f"Resized model embedding layers to {len(tokenizer)}")

    if cfg.load_in_8bit or (cfg.adapter == "qlora" and cfg.load_in_4bit):
        # Freeze base model
        # Cast all non INT8 parameters (e.g. layernorm) to fp32 for stability
        # Make embedding layer's output require grads for backward compatibility
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)

    # Post cast
    # LlamaRMSNorm layers are in fp32 after prepare_model_for_kbit_training() or full finetune,
    # so we need to convert them back to fp16/bf16 for flash-attn compatibility
    # if needs_fa2_dtype or (cfg.flash_attention and cfg.is_llama_derived_model):
    #     for name, module in model.named_modules():
    #         if "norm" in name:
    #             module.to(torch_dtype)
    #         if "lm_head" in name or "embed_tokens" in name:
    #             if hasattr(module, "weight"):
    #                 module.to(torch_dtype)

    # Load adapter
    if cfg.adapter:
        model = load_adapter(model, cfg)

    # Disable HF Trainer's DataParallel for multigpu
    # See https://github.com/johnsmith0031/alpaca_lora_4bit/blob/08b3fca4a4a9e0d3945be1bab4529f100a428636/finetune.py#L130-L133
    # and https://github.com/tloen/alpaca-lora/pull/131
    # and https://github.com/huggingface/transformers/pull/22628
    if torch.cuda.device_count() > 1 and int(os.getenv("WORLD_SIZE", "1")) > 1 and cfg.load_in_4bit:
        model.is_parallelizable = True
        model.model_parallel = True

    # Disable kv cache
    model.config.use_cache = False

    return model


def load_adapter(model: transformers.PreTrainedModel, cfg: Any, inference: bool = False):
    """Load adapter."""

    if cfg.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        # When using PEFT + gradient checkpointing + Trainer we need to make sure the input has requires_grad=True
        # When training with PEFT, only LoRA layers will have requires grad set to True, but the output of frozen layers need to propagate
        # the gradients to make sure the gradient flows.
        # This might have been already done in prepare_model_for_kbit_training()
        # See https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
        # and https://github.com/huggingface/transformers/blob/ca7912d191cad41f3a212ce491736d9dc4cb812b/src/transformers/modeling_utils.py#L1830-L1835
        # and https://github.com/huggingface/peft/issues/137
        model.enable_input_require_grads()
    if cfg.adapter in ["lora", "qlora"]:
        return load_lora(model, cfg, inference=inference)

    raise NotImplementedError(f"{cfg.adapter} peft adapter is not available")


# Copied and modified from https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model: transformers.PreTrainedModel):
    """Find all linear layers to apply LoRA."""

    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear, QuantLinear)

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls) or "Linear" in module.__class__.__name__:
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


# Adapted from https://github.com/bofenghuang/vigogne/blob/76e1cd0b35fd4f9e360aecdf7130c998459df0ff/vigogne/train/utils/peft.py#L13
def print_trainable_parameters(model: transformers.PreTrainedModel):
    """Returns the number of trainable parameters and number of all parameters in the model."""
    trainable_params = 0
    total_params = 0
    params_by_dtype = defaultdict(int)

    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

        params_by_dtype[param.dtype] += num_params

    logger.info(
        f"trainable params: {trainable_params:,d} || all params: {total_params:,d} || percentage:"
        f" {100 * trainable_params / total_params:.2f}%"
    )
    for k, v in params_by_dtype.items():
        logger.info(f"dtype: {k} || num: {v:,d} || percentage: {100 * v / total_params:.2f}%")


def load_lora(model: transformers.PreTrainedModel, cfg: Any, inference: bool = False):
    """Apply LoRA layers to base model."""

    logger.info("Loading LoRA layers...")

    lora_target_modules = cfg.lora_target_modules or []

    if cfg.lora_target_all_linear_layers:
        linear_target_modules = find_all_linear_names(model)
        lora_target_modules = list(set(lora_target_modules + linear_target_modules))

    if not lora_target_modules:
        raise ValueError("Found empty lora_target_modules and lora_target_all_linear_layers is set False")

    logger.info(f'Apply LoRA on layers: {", ".join(lora_target_modules)}')

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        fan_in_fan_out=cfg.lora_fan_in_fan_out or False,
        modules_to_save=cfg.lora_modules_to_save or None,
        bias="none",
        task_type=TaskType.CAUSAL_LM if cfg.model_type == DECODER else TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)

    # model.print_trainable_parameters()
    print_trainable_parameters(model)

    return model


def move_adapter_files(cfg: Any):
    """Move adapter files."""

    adapter_dir = os.path.join(cfg.output_dir, PEFT_FOLDER_NAME)
    os.makedirs(adapter_dir, exist_ok=True)

    for filename in [PEFT_CONFIG_NAME, PEFT_WEIGHTS_NAME, PEFT_SAFETENSORS_WEIGHTS_NAME]:
        if os.path.exists(file_path := os.path.join(cfg.output_dir, filename)):
            shutil.move(file_path, os.path.join(adapter_dir, filename))


def merge_lora(cfg: Any):
    """Remerge LoRA layers with base model."""

    logger.info("Merging LoRA with base model...")

    model_cls = AutoPeftModelForCausalLM if cfg.model_type == DECODER else AutoPeftModelForSeq2SeqLM
    model = model_cls.from_pretrained(
        cfg.output_dir,
        device_map="auto",  # todo: cpu
        # torch_dtype=torch.bfloat16
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.merge_and_unload()
    model.save_pretrained(cfg.output_dir, safe_serialization=cfg.save_safetensors, max_shard_size=cfg.max_shard_size)

    move_adapter_files(cfg)


# Deprecated
# Copied and modified from https://github.com/tatsu-lab/stanford_alpaca/blob/eb5b171d9b103a12a8e14e0edca9cbc45fe1d512/train.py#L75-L95
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
