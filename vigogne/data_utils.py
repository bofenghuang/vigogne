# coding=utf-8
# Copyright 2023  Bofeng Huang

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# model types
SUPPORTED_MODEL_TYPES = [DECODER := "decoder-only", SEQ2SEQ := "seq2seq"]

# ignore index in loss
IGNORE_INDEX = -100


# instruction-following example's format
class Instruct(BaseModel):
    instruction: str
    id: Optional[str] = None
    system: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None


# role in chat
class Role(str, Enum):
    user = "user"
    assistant = "assistant"


# utterance in chat
class Utterance(BaseModel):
    role: Role
    content: str


# chat example's format
class Conversation(BaseModel):
    messages: List[Utterance]
    id: Optional[str] = None
    system: Optional[str] = None

    def fully_model_dump(self, **kwargs) -> Dict[str, Any]:
        dumped_dict = super().model_dump(**kwargs)
        for utterance in dumped_dict["messages"]:
            utterance["role"] = utterance["role"].value
        return dumped_dict
