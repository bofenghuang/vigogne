#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import fire
import torch
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(
    # base_model_name_or_path, lora_model_name_or_path, output_dir, max_shard_size: str = "10GB", safe_serialization: bool = False
    lora_model_name_or_path, output_dir, max_shard_size: str = "10GB", safe_serialization: bool = False
):

    # tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, padding_side="right", use_fast=False)

    # base_model = AutoModelForCausalLM.from_pretrained(
    #     base_model_name_or_path,
    #     torch_dtype=torch.float16,
    #     device_map={"": "cpu"},
    #     #    load_in_8bit=True,
    #     trust_remote_code=True,
    #     low_cpu_mem_usage=True,
    #     #    offload_folder=offload_dir,
    #     #    offload_state_dict=True,
    # )

    # first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    # first_weight_old = first_weight.clone()

    # lora_model = PeftModel.from_pretrained(base_model, lora_model_name_or_path)
    # assert torch.allclose(first_weight_old, first_weight)

    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_model_name_or_path,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
        #    load_in_8bit=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        #    offload_folder=offload_dir,
        #    offload_state_dict=True,
    )

    model = model.merge_and_unload()
    # assert not torch.allclose(first_weight_old, first_weight)

    model.save_pretrained(
        output_dir,
        max_shard_size=max_shard_size,
        safe_serialization=safe_serialization,
    )

    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="right", use_fast=False)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(main)
