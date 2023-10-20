# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Legacy utils."""

from typing import List, Optional

import transformers

from vigogne.data_utils import Conversation, Instruct, Role, Utterance
from vigogne.processors import alpaca_template, vigogne_chat_v2_template


# legacy
def generate_instruct_prompt(instruction: str, system: Optional[str] = None):
    return alpaca_template.build_inference_prompt(Instruct(instruction=instruction, system=system))


# legacy
def generate_inference_chat_prompt(
    history: List[List[str]],
    tokenizer: transformers.PreTrainedTokenizer,
    system_message: Optional[str] = None,
    max_length: int = 2048,
):
    conversation = Conversation(system=system_message, messages=[])
    for x in history:
        conversation.messages.append(Utterance(role=Role.user, content=x[0]))
        conversation.messages.append(Utterance(role=Role.assistant, content=x[1]))
    return vigogne_chat_v2_template.build_inference_prompt(conversation, tokenizer, max_length=max_length)
