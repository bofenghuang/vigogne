#!/usr/bin/env python
# coding=utf-8

"""
Modified from: https://github.com/tloen/alpaca-lora
"""

import json
import os

import fire
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM

CHECKPOINT_PARAMS = {
    "7B": {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": -1},
    "13B": {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-06, "vocab_size": -1},
    "30B": {"dim": 6656, "multiple_of": 256, "n_heads": 52, "n_layers": 60, "norm_eps": 1e-06, "vocab_size": -1},
    "65B": {"dim": 8192, "multiple_of": 256, "n_heads": 64, "n_layers": 80, "norm_eps": 1e-06, "vocab_size": -1},
}


def main(base_model_name_or_path: str, lora_model_name_or_path: str, output_dir: str, base_model_size: str = "7B"):

    # Retrieve the model parameters
    params = CHECKPOINT_PARAMS.get(base_model_size)
    if params is None:
        raise ValueError(
            f"Cannot find the right model parameters for {base_model_size}. Please choose between {list(CHECKPOINT_PARAMS.keys())}."
        )

    # tokenizer = LlamaTokenizer.from_pretrained(base_model_name_or_path)

    base_model = LlamaForCausalLM.from_pretrained(
        base_model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_model_name_or_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    # merge weights
    for layer in lora_model.base_model.model.model.layers:
        if hasattr(layer.self_attn.q_proj, "merge_weights"):
            layer.self_attn.q_proj.merge_weights = True
        if hasattr(layer.self_attn.v_proj, "merge_weights"):
            layer.self_attn.v_proj.merge_weights = True
        if hasattr(layer.self_attn.k_proj, "merge_weights"):
            layer.self_attn.k_proj.merge_weights = True
        if hasattr(layer.self_attn.o_proj, "merge_weights"):
            layer.self_attn.o_proj.merge_weights = True
        if hasattr(layer.mlp.gate_proj, "merge_weights"):
            layer.mlp.gate_proj.merge_weights = True
        if hasattr(layer.mlp.down_proj, "merge_weights"):
            layer.mlp.down_proj.merge_weights = True
        if hasattr(layer.mlp.up_proj, "merge_weights"):
            layer.mlp.up_proj.merge_weights = True

    # todo
    # lora_model = lora_model.merge_and_unload()

    lora_model.train(False)

    lora_model_sd = lora_model.state_dict()

    # params = {
    #     "dim": 4096,
    #     "multiple_of": 256,
    #     "n_heads": 32,
    #     "n_layers": 32,
    #     "norm_eps": 1e-06,
    #     "vocab_size": -1,
    # }
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

    def permute(w):
        return w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)

    def unpermute(w):
        return w.view(n_heads, 2, dim // n_heads // 2, dim).transpose(1, 2).reshape(dim, dim)

    def translate_state_dict_key(k):
        k = k.replace("base_model.model.", "")
        if k == "model.embed_tokens.weight":
            return "tok_embeddings.weight"
        elif k == "model.norm.weight":
            return "norm.weight"
        elif k == "lm_head.weight":
            return "output.weight"
        elif k.startswith("model.layers."):
            layer = k.split(".")[2]
            if k.endswith(".self_attn.q_proj.weight"):
                return f"layers.{layer}.attention.wq.weight"
            elif k.endswith(".self_attn.k_proj.weight"):
                return f"layers.{layer}.attention.wk.weight"
            elif k.endswith(".self_attn.v_proj.weight"):
                return f"layers.{layer}.attention.wv.weight"
            elif k.endswith(".self_attn.o_proj.weight"):
                return f"layers.{layer}.attention.wo.weight"
            elif k.endswith(".mlp.gate_proj.weight"):
                return f"layers.{layer}.feed_forward.w1.weight"
            elif k.endswith(".mlp.down_proj.weight"):
                return f"layers.{layer}.feed_forward.w2.weight"
            elif k.endswith(".mlp.up_proj.weight"):
                return f"layers.{layer}.feed_forward.w3.weight"
            elif k.endswith(".input_layernorm.weight"):
                return f"layers.{layer}.attention_norm.weight"
            elif k.endswith(".post_attention_layernorm.weight"):
                return f"layers.{layer}.ffn_norm.weight"
            elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
                return None
            else:
                print(layer, k)
                raise NotImplementedError
        else:
            print(k)
            raise NotImplementedError

    new_state_dict = {}
    for k, v in lora_model_sd.items():
        new_k = translate_state_dict_key(k)
        if new_k is not None:
            if "wq" in new_k or "wk" in new_k:
                new_state_dict[new_k] = unpermute(v)
            else:
                new_state_dict[new_k] = v

    os.makedirs(output_dir, exist_ok=True)

    torch.save(new_state_dict, output_dir + "/consolidated.00.pth")

    with open(output_dir + "/params.json", "w") as f:
        json.dump(params, f)


if __name__ == "__main__":
    fire.Fire(main)
