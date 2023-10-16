# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Load tokenizers."""

import logging
from typing import Any

# import transformers
from transformers import AddedToken, AutoTokenizer

logger = logging.getLogger(__name__)


# tokenizer default
# DEFAULT_PAD_TOKEN = "[PAD]"
# # DEFAULT_PAD_TOKEN = "<pad>"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "<s>"
# DEFAULT_UNK_TOKEN = "<unk>"


def tok(self, text: str, add_bos_token: bool = True, add_eos_token: bool = True, **kwargs):
    tokenized_outputs = self(text, **kwargs)

    max_length = kwargs.get("max_length", None) or self.model_max_length

    if add_bos_token and len(tokenized_outputs["input_ids"]) > 0 and tokenized_outputs["input_ids"][0] != self.bos_token_id:
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
    logger.info("Loading tokenizer...")

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

    # Add tokens
    if cfg.add_tokens:
        tokenizer.add_tokens([AddedToken(token, rstrip=False, lstrip=False, normalized=False) for token in cfg.add_tokens])
        logger.info(f"Added tokens: {cfg.add_tokens}")
    if cfg.add_special_tokens:
        for k, v in cfg.add_special_tokens.items():
            tokenizer.add_special_tokens({k: AddedToken(v, rstrip=False, lstrip=False, normalized=False)})
        logger.info(f"Added special tokens: {cfg.add_special_tokens}")

    logger.info(f"bos_token: {tokenizer.bos_token} / bos_token_id: {tokenizer.bos_token_id}")
    logger.info(f"eos_token: {tokenizer.eos_token} / eos_token_id: {tokenizer.eos_token_id}")
    logger.info(f"pad_token: {tokenizer.pad_token} / pad_token_id: {tokenizer.pad_token_id}")
    logger.info(f"unk_token: {tokenizer.unk_token} / unk_token_id: {tokenizer.unk_token_id}")

    # todo: better handle
    # Override instance method
    # Cannot override special method (e.g., __call__) of instance
    # See https://stackoverflow.com/questions/60062100/is-it-possible-to-override-a-class-call-method
    # tokenizer.__call__ = tok.__get__(tokenizer, tokenizer.__class__)
    tokenizer.tok = tok.__get__(tokenizer, tokenizer.__class__)
    # or monkey patching

    return tokenizer
