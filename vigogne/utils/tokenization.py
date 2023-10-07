# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Load tokenizers"""

from typing import Any, List, Union

import transformers
from transformers import AutoTokenizer

# tokenizer default
DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def tok(self, text: str, add_bos_token: bool = True, add_eos_token: bool = True, **kwargs):
    tokenized_outputs = self(text, **kwargs)

    max_length = kwargs.get("max_length", None) or self.model_max_length

    if (
        add_bos_token
        and len(tokenized_outputs["input_ids"]) > 0
        and tokenized_outputs["input_ids"][0] != self.bos_token_id
    ):
        tokenized_outputs["input_ids"].insert(0, self.bos_token_id)
        tokenized_outputs["attention_mask"].insert(0, 1)

        if len(tokenized_outputs["input_ids"]) > max_length:
            tokenized_outputs["input_ids"] = tokenized_outputs["input_ids"][:max_length]
            tokenized_outputs["attention_mask"] = tokenized_outputs["attention_mask"][:max_length]

    if (
        not add_bos_token
        and len(tokenized_outputs["input_ids"]) > 0
        and tokenized_outputs["input_ids"][0] == self.bos_token_id
    ):
        tokenized_outputs["input_ids"] = tokenized_outputs["input_ids"][1:]
        tokenized_outputs["attention_mask"] = tokenized_outputs["attention_mask"][1:]

    if (
        add_eos_token
        and len(tokenized_outputs["input_ids"]) > 0
        and tokenized_outputs["input_ids"][-1] != self.eos_token_id
        and len(tokenized_outputs["input_ids"]) < max_length
    ):
        tokenized_outputs["input_ids"].append(self.eos_token_id)
        tokenized_outputs["attention_mask"].append(1)

    if (
        not add_eos_token
        and len(tokenized_outputs["input_ids"]) > 0
        and tokenized_outputs["input_ids"][-1] == self.eos_token_id
    ):
        tokenized_outputs["input_ids"] = tokenized_outputs["input_ids"][:-1]
        tokenized_outputs["attention_mask"] = tokenized_outputs["attention_mask"][:-1]

    return tokenized_outputs


def load_tokenizer(cfg: Any):
    tokenizer_kwargs = {
        "revision": cfg.tokenizer_revision,
        # True is the default w/ https://github.com/huggingface/transformers/pull/25224
        "legacy": cfg.tokenizer_legacy,
        "padding_side": cfg.tokenizer_padding_side,
    }
    tokenizer_kwargs = {k: v for k, v in tokenizer_kwargs.items() if v is not None}

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name_or_path,
        use_fast=cfg.tokenizer_use_fast,
        **tokenizer_kwargs,
    )

    if (
        tokenizer.__class__.__name__ in ["LlamaTokenizer", "LlamaTokenizerFast", "CodeLlamaTokenizer"]
        and hasattr(tokenizer, "pad_token")
        and not tokenizer.pad_token
    ):
        # use eos_token as pad_token
        tokenizer.pad_token = tokenizer.eos_token

    # deprecated
    # Some special tokens can be "" or None depending on releases
    # special_tokens_dict = dict()
    # if tokenizer.pad_token is None or not tokenizer.pad_token:
    #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # if tokenizer.eos_token is None or not tokenizer.eos_token:
    #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    # if tokenizer.bos_token is None or not tokenizer.bos_token:
    #     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    # if tokenizer.unk_token is None or not tokenizer.unk_token:
    #     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=special_tokens_dict,
    #     tokenizer=tokenizer,
    #     model=model,
    # )

    # todo: better handle
    # Replace the instance method
    # cannot override __call__ of instance
    # tokenizer.__call__ = tok.__get__(tokenizer, tokenizer.__class__)
    tokenizer.tok = tok.__get__(tokenizer, tokenizer.__class__)
    # or monkey patching

    return tokenizer
