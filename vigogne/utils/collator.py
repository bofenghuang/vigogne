# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Training collators."""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import transformers

from ..data_utils import IGNORE_INDEX


# Deprecated
# Almost same to transformers.DataCollatorForSeq2Seq
# Copied and modified from https://github.com/tatsu-lab/stanford_alpaca/blob/eb5b171d9b103a12a8e14e0edca9cbc45fe1d512/train.py#L166-L182
@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # dtype = torch.long
        # input_ids, labels = tuple([torch.LongTensor(instance[key]) for instance in instances] for key in ("input_ids", "labels"))
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        if self.pad_to_multiple_of is not None:
            max_length_index, max_length = max(
                enumerate([len(input_ids_) for input_ids_ in input_ids]), key=lambda x: x[1]
            )
            # n_padding = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of - max_length
            n_padding = math.ceil(max_length / self.pad_to_multiple_of) * self.pad_to_multiple_of - max_length
            # Pad the longest example to pad_to_multiple_of * N
            input_ids[max_length_index].extend([self.tokenizer.pad_token_id] * n_padding)
            labels[max_length_index].extend([IGNORE_INDEX] * n_padding)

        input_ids = [torch.LongTensor(input_ids_) for input_ids_ in input_ids]
        labels = [torch.LongTensor(labels_) for labels_ in labels]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# Copied and modified from https://github.com/tatsu-lab/stanford_alpaca/blob/eb5b171d9b103a12a8e14e0edca9cbc45fe1d512/train.py#L166-L182
# Almost same to transformers.DataCollatorForSeq2Seq
@dataclass
class Seq2SeqDataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # dtype = torch.long
        # input_ids, labels = tuple([torch.LongTensor(instance[key]) for instance in instances] for key in ("input_ids", "labels"))
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        if self.pad_to_multiple_of is not None:
            max_input_length_index, max_input_length = max(
                enumerate([len(input_ids_) for input_ids_ in input_ids]), key=lambda x: x[1]
            )
            # n_input_padding = ((max_input_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of - max_input_length
            n_input_padding = (
                math.ceil(max_input_length / self.pad_to_multiple_of) * self.pad_to_multiple_of - max_input_length
            )
            # Pad the longest example to pad_to_multiple_of * N
            input_ids[max_input_length_index].extend([self.tokenizer.pad_token_id] * n_input_padding)

            max_label_length_index, max_label_length = max(
                enumerate([len(labels_) for labels_ in labels]), key=lambda x: x[1]
            )
            # n_label_padding = ((max_label_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of - max_label_length
            n_label_padding = (
                math.ceil(max_label_length / self.pad_to_multiple_of) * self.pad_to_multiple_of - max_label_length
            )
            # Pad the longest example to pad_to_multiple_of * N
            labels[max_label_length_index].extend([IGNORE_INDEX] * n_label_padding)

        input_ids = [torch.LongTensor(input_ids_) for input_ids_ in input_ids]
        labels = [torch.LongTensor(labels_) for labels_ in labels]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
