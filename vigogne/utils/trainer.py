# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Trainer utilities."""

from typing import Any, Optional

import torch
import transformers
from transformers import DataCollatorForSeq2Seq, Trainer

from ..data_utils import IGNORE_INDEX


def setup_trainer(
    cfg: Any,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: Optional[torch.utils.data.Dataset] = None,
):
    # define collator
    # A100 is best at 64, while others at 8
    data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE_INDEX, pad_to_multiple_of=64)

    # Init trainer
    trainer = Trainer(
        model=model,
        args=cfg,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    return trainer
