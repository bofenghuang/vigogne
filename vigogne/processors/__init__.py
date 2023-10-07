# coding=utf-8
# Copyright 2023  Bofeng Huang

from typing import Any, Dict, List, Optional, Union

import transformers

from ..data_utils import Conversation, Instruct, Role, Utterance
from .alpaca import AlpacaProcessor, AlpacaTemplate
from .vigogne_chat_v2 import VigogneChatV2Processor, VigogneChatV2Template
from .vigogne_chat_v3 import VigogneChatV3Processor, VigogneChatV3Template

# Template and processor for vigogne instruct models which follow the Alpaca's style
alpaca_template = AlpacaTemplate()
alpaca_processor = AlpacaProcessor(**alpaca_template.to_dict())

# Template and processor for vigogne chat v2 models
vigogne_chat_v2_template = VigogneChatV2Template()
vigogne_chat_v2_processor = VigogneChatV2Processor(**vigogne_chat_v2_template.to_dict())

# Template and processor for Vigogne Chat V3 models, styled based on modifications from Llama-2
vigogne_chat_v3_template = VigogneChatV3Template()
vigogne_chat_v3_processor = VigogneChatV3Processor(**vigogne_chat_v3_template.to_dict())


SUPPORTED_PROCESSORS = {
    "alpaca": alpaca_processor,
    "vigogne_chat_v2": vigogne_chat_v2_processor,
    "vigogne_chat_v3": vigogne_chat_v3_processor,
}


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
