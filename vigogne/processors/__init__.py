# coding=utf-8
# Copyright 2023  Bofeng Huang

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
