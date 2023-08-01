# coding=utf-8
# Copyright 2023  Bofeng Huang

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel


# todo
class SFTMode(str, Enum):
    instruct = "instruct"
    chat = "chat"


class Instruct(BaseModel):
    instruction: str
    system: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None


class Role(str, Enum):
    user = "USER"
    assistant = "ASSISTANT"
    system = "SYSTEM"


class Utterance(BaseModel):
    role: Role
    content: str


class Conversation(BaseModel):
    messages: List[Utterance]
    id: Optional[str] = None
    system: Optional[str] = None

    def fully_model_dump(self, **kwargs) -> dict[str, Any]:
        dumped_dict = super().model_dump(**kwargs)
        for utterance in dumped_dict["messages"]:
            utterance["role"] = utterance["role"].value
        return dumped_dict
