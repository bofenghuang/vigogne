#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import fire
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch


def main(
    base_model_name_or_path, lora_model_name_or_path, output_dir, max_shard_size: str = "10GB", safe_serialization: bool = True
):

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, padding_side="right", use_fast=False)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
        #    load_in_8bit=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        #    offload_folder=offload_dir,
        #    offload_state_dict=True,
    )

    # first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    # first_weight_old = first_weight.clone()

    lora_model = PeftModel.from_pretrained(base_model, lora_model_name_or_path)
    # assert torch.allclose(first_weight_old, first_weight)

    base_model = lora_model.merge_and_unload()
    # assert not torch.allclose(first_weight_old, first_weight)

    base_model.save_pretrained(
        output_dir,
        max_shard_size=max_shard_size,
        safe_serialization=safe_serialization,
    )

    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(main)
