# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Load models"""

import logging
import os
from typing import Any, Dict

import bitsandbytes as bnb
import torch
import transformers
from peft import AutoPeftModelForCausalLM, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import QuantLinear
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def load_model(cfg: Any):
    model_kwargs = {
        "revision": cfg.model_revision,
    }
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    # todo: load config then patch
    # todo: add GTPQ

    torch_dtype = getattr(torch, cfg.torch_dtype)

    # BitsAndBytesConfig support
    if cfg.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    if cfg.adapter == "qlora" and cfg.load_in_4bit:
        # todo: customize 4bit config
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            # llm_int8_threshold=6.0,
            # llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.trust_remote_code,
        use_flash_attention_2=cfg.use_flash_attention_2,
        **model_kwargs,
    )

    # embeddings_len = (
    #     math.ceil(len(tokenizer) / 32) * 32
    #     if cfg.resize_token_embeddings_to_32x
    #     else len(tokenizer)
    # )
    # if model.get_input_embeddings().num_embeddings < embeddings_len:
    #     model.resize_token_embeddings(embeddings_len)
    # else:
    #     model.tie_weights()

    if cfg.load_in_8bit or (cfg.adapter == "qlora" and cfg.load_in_4bit):
        # Cast the small parameters (e.g. layernorm) to fp32 for stability
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)

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
    if cfg.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        # When using PEFT + gradient checkpointing + Trainer we need to make sure the input has requires_grad=True
        # When training with PEFT, only LoRA layers will have requires grad set to True, but the output of frozen layers need to propagate
        # the gradients to make sure the gradient flows.
        # See https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
        # and https://github.com/huggingface/transformers/blob/ca7912d191cad41f3a212ce491736d9dc4cb812b/src/transformers/modeling_utils.py#L1830-L1835
        # and https://github.com/huggingface/peft/issues/137
        model.enable_input_require_grads()
    if cfg.adapter in ["lora", "qlora"]:
        return load_lora(model, cfg, inference=inference)

    raise NotImplementedError(f"{cfg.adapter} peft adapter is not available")


# Copied and modified from https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model: transformers.PreTrainedModel):
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear, QuantLinear)

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls) or "Linear" in module.__class__.__name__:
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def load_lora(model: transformers.PreTrainedModel, cfg: Any, inference: bool = False):
    lora_target_modules = cfg.lora_target_modules or []

    if cfg.lora_target_all_linear_layers:
        linear_target_modules = find_all_linear_names(model)
        logger.info(f"Found linear modules: {repr(linear_target_modules)}")
        lora_target_modules = list(set(lora_target_modules + linear_target_modules))

    if not lora_target_modules:
        raise ValueError("Found empty lora_target_modules and lora_target_all_linear_layers is set False")

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        fan_in_fan_out=cfg.lora_fan_in_fan_out or False,
        modules_to_save=cfg.lora_modules_to_save or None,
        bias="none",
        task_type=cfg.lora_task_type,
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def merge_lora(cfg: Any):
    logger.info("Merging LoRA with base model")

    model = AutoPeftModelForCausalLM.from_pretrained(
        cfg.output_dir,
        device_map="auto",
        # torch_dtype=torch.bfloat16
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.merge_and_unload()
    model.save_pretrained(cfg.output_dir, safe_serialization=cfg.save_safetensors, max_shard_size=cfg.max_shard_size)


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
