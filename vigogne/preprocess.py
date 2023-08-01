# coding=utf-8
# Copyright 2023  Bofeng Huang


import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import transformers

from vigogne.data_utils import Conversation, Instruct, Role, SFTMode, Utterance

# instruct system message
INSTRUCT_SYSTEM_MESSAGE_EN = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
INSTRUCT_SYSTEM_MESSAGE_FR = "Ci-dessous se trouve une instruction qui décrit une tâche à accomplir. Rédigez une réponse qui répond de manière précise à la demande."
DEFAULT_INSTRUCT_SYSTEM_MESSAGE = INSTRUCT_SYSTEM_MESSAGE_EN

# conversation system message
CONVERSATION_SYSTEM_MESSAGE_EN = """Below is a conversation between a user and an AI assistant named Vigogne.
Vigogne is an open-source AI assistant created by Zaion (https://zaion.ai/).
Vigogne is polite, emotionally aware, humble-but-knowledgeable, always providing helpful and detailed answers.
Vigogne is skilled in responding proficiently in the languages its users use and can perform a wide range of tasks such as text editing, translation, question answering, logical reasoning, coding, and many others.
Vigogne cannot receive or generate audio or visual content and cannot access the internet.
Vigogne strictly avoids discussing sensitive, offensive, illegal, ethical, or political topics and caveats when unsure of the answer."""
CONVERSATION_SYSTEM_MESSAGE_FR = """Voici une conversation entre un utilisateur et un assistant IA nommé Vigogne.
Vigogne est un assistant IA open-source créé par Zaion (https://zaion.ai/).
Vigogne est respectueux, empathique, humble mais bien informé, et fournit toujours des réponses utiles et détaillées.
Vigogne est capable d'effectuer une large variété de tâches telles que l'édition de texte, la traduction, la question answering, la raisonnement logique, le codage et bien d'autres encore.
Vigogne ne peut pas recevoir ou générer de contenu audio ou visuel et ne peut pas accéder à Internet.
Vigogne évite strictement de discuter de sujets sensibles, offensants, illégaux, éthiques ou politiques et met en garde lorsqu'il n'est pas sûr de la réponse."""
CONVERSATION_SYSTEM_MESSAGE_EN_SHORT = "Below is a conversation between a user and an AI assistant named Vigogne."
CONVERSATION_SYSTEM_MESSAGE_FR_SHORT = "Voici une conversation entre un utilisateur et un assistant IA nommé Vigogne."
DEFAULT_CHAT_SYSTEM_MESSAGE = CONVERSATION_SYSTEM_MESSAGE_EN_SHORT


def merge_instruction_and_input(instruction_str: str, input_str: Optional[str], symbols_to_strip: str = "!,-.:;?~ "):
    if input_str:
        instruction_str = re.sub("[" + re.escape(symbols_to_strip) + "]+$", "", instruction_str)
        instruction_str = f"{instruction_str} : {input_str}"

    return instruction_str


@dataclass
class InstructTemplate:
    system_prefix: str
    instruction_prefix: str
    output_prefix: str
    default_system_message: str = DEFAULT_INSTRUCT_SYSTEM_MESSAGE

    def _ensure_type(self, instuct: Union[Instruct, Dict]) -> Instruct:
        return Instruct(**instuct) if not isinstance(instuct, Instruct) else instuct

    def get_training_prompt(
        self,
        instuct: Union[Instruct, Dict],
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> str:
        instuct = self._ensure_type(instuct)

        prompt_message = self.get_inference_prompt(instuct)
        prompt_message += instuct.output + tokenizer.eos_token

        return prompt_message

    def get_inference_prompt(self, instuct: Union[Instruct, Dict]) -> str:
        instuct = self._ensure_type(instuct)

        instruction = (
            merge_instruction_and_input(instuct.instruction, instuct.input) if instuct.input else instuct.instruction
        )

        system_message = self.default_system_message if instuct.system is None else instuct.system

        prompt_message = self.system_prefix + ":" + "\n" + system_message
        prompt_message += "\n\n" + self.instruction_prefix + ":" + "\n" + instruction
        prompt_message += "\n\n" + self.output_prefix + ":" + "\n"

        return prompt_message

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}


@dataclass
class ConversationTemplate:
    system_prefix: str
    user_prefix: str
    assistant_prefix: str
    default_system_message: str = DEFAULT_CHAT_SYSTEM_MESSAGE

    def _ensure_type(self, conversation: Union[Conversation, Dict]) -> Conversation:
        return Conversation(**conversation) if not isinstance(conversation, Conversation) else conversation

    def get_training_prompt(self, conversation: Union[Conversation, Dict], tokenizer: transformers.PreTrainedTokenizer) -> str:
        conversation = self._ensure_type(conversation)

        system_message = self.default_system_message if conversation.system is None else conversation.system
        prompt_message = f"{self.system_prefix}: {system_message}{tokenizer.eos_token}"

        for utterance in conversation.messages:
            if utterance.role == Role.assistant:
                # Add eos token after system message / user utterance / assistant utterance
                prompt_message += "\n" + f"{self.assistant_prefix}: {utterance.content}{tokenizer.eos_token}"
            else:
                prompt_message += "\n" + f"{self.user_prefix}: {utterance.content}{tokenizer.eos_token}"
        return prompt_message

    def get_inference_prompt(
        self, conversation: Union[Conversation, Dict], tokenizer: transformers.PreTrainedTokenizer, max_length: int = 2048
    ) -> str:
        conversation = self._ensure_type(conversation)

        messages_by_round = []
        current_round_message = ""
        for utterance in conversation.messages:
            if utterance.role == Role.user:
                # output a round if not empty and have assistant message
                # one round starts from user and has at least one assistant message
                if current_round_message and self.assistant_prefix in current_round_message:
                    messages_by_round.append(current_round_message)
                    current_round_message = ""

                current_round_message += "\n" + f"{self.user_prefix}: {utterance.content}{tokenizer.eos_token}"
            else:
                current_round_message += (
                    "\n" + f"{self.assistant_prefix}: {utterance.content}{tokenizer.eos_token}"
                )

        if current_round_message:
            messages_by_round.append(current_round_message)

        # debug
        # print(messages_by_round)

        system_message = self.default_system_message if conversation.system is None else conversation.system
        system_header_text = f"{self.system_prefix}: {system_message}{tokenizer.eos_token}"

        # prefix for response
        prompt_message = "\n" + self.assistant_prefix + ":"
        for x in messages_by_round[::-1]:
            if len(tokenizer(system_header_text + x + prompt_message)["input_ids"]) <= max_length:
                prompt_message = x + prompt_message
            else:
                break

        return system_header_text + prompt_message if prompt_message else None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}


instruct_template = InstructTemplate(
    system_prefix="### System",
    instruction_prefix="### Instruction",
    output_prefix="### Response",
)

conversation_template = ConversationTemplate(
    system_prefix=f"<|{Role.system.value}|>",
    user_prefix=f"<|{Role.user.value}|>",
    assistant_prefix=f"<|{Role.assistant.value}|>",
)


SUPPORTED_DATA_TEMPLATES = {
    SFTMode.instruct.value: instruct_template,
    SFTMode.chat.value: conversation_template,
}


# todo
# legacy
def generate_instruct_prompt(instruction: str, system: Optional[str] = None):
    return instruct_template.get_inference_prompt(Instruct(instruction=instruction, system=system))


# legacy
def generate_inference_chat_prompt(
    history: List[List[str]], tokenizer: transformers.PreTrainedTokenizer, max_length: int = 2048
):
    conversation = Conversation(messages=[])
    for x in history:
        conversation.messages.append(Utterance(role=Role.user, content=x[0]))
        conversation.messages.append(Utterance(role=Role.assistant, content=x[1]))
    # tmp fix
    del conversation.messages[-1]
    return conversation_template.get_inference_prompt(conversation, tokenizer, max_length=max_length)
