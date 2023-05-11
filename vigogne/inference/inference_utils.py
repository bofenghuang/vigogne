#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import transformers
from transformers import StoppingCriteria
from typing import List
import re
import torch


# class StopWordsCriteria(StoppingCriteria):
#     def __init__(self, stop_words: List[str], tokenizer: transformers.PreTrainedTokenizer):
#         self.stop_token_ids = [tokenizer(stop_word, add_special_tokens=False)["input_ids"] for stop_word in stop_words]
#         # print(self.stop_token_ids)

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         return any(input_ids.tolist()[0][-len(stop_token_id) :] == stop_token_id for stop_token_id in self.stop_token_ids)


class StopWordsCriteria(StoppingCriteria):
    def __init__(self, stop_words: List[str], tokenizer: transformers.PreTrainedTokenizer):
        self.stop_words = stop_words
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # return any(generated_text.endswith(stop_word) for stop_word in self.stop_words)
        return bool(re.search(rf'(?:{"|".join([re.escape(stop_word) for stop_word in self.stop_words])})\W*$', generated_text))
