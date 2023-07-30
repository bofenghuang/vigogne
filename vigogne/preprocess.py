# coding=utf-8
# Copyright 2023  Bofeng Huang


import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import transformers

from vigogne.constants import ASSISTANT, CHAT, CONTENT, INSTRUCT, ROLE, USER


def merge_instruction_and_input(instruction_str: str, input_str: Optional[str], symbols_to_strip: str = "!,-.:;?~ "):
    if input_str:
        instruction_str = re.sub("[" + re.escape(symbols_to_strip) + "]+$", "", instruction_str)
        instruction_str = f"{instruction_str} : {input_str}"

    return instruction_str


@dataclass
class InstructTemplate:
    # system_prefix: str
    system_message: str
    instruction_prefix: str
    output_prefix: str

    def get_training_prompt(self, instruction: str, input: str = "", output: str = "", **kwargs) -> str:
        if input:
            instruction = merge_instruction_and_input(instruction, input)

        prompt_message = self.system_message
        prompt_message += "\n\n" + self.instruction_prefix + ":" + "\n" + instruction
        prompt_message += "\n\n" + self.output_prefix + ":" + "\n" + output

        return prompt_message

    def get_inference_prompt(self, instruction: str, input: str = "", **kwargs) -> str:
        return self.get_training_prompt(instruction, input=input)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}


@dataclass
class ConversationTemplate:
    # system_prefix: str
    system_message: str
    user_prefix: str
    assistant_prefix: str

    def get_training_prompt(self, messages: List[Dict[str, str]], tokenizer: transformers.PreTrainedTokenizer) -> str:
        prompt_message = self.system_message + "\n"
        for speaking_turn in messages:
            if speaking_turn[ROLE] == USER:
                prompt_message += "\n" + f"{self.user_prefix}: {speaking_turn[CONTENT]}"
            else:
                prompt_message += "\n" + f"{self.assistant_prefix}: {speaking_turn[CONTENT]}" + tokenizer.eos_token
        return prompt_message

    def get_inference_prompt(
        self, messages: List[Dict[str, str]], tokenizer: transformers.PreTrainedTokenizer, max_length: int = 2048
    ) -> str:
        messages_by_round = []
        current_round_message = ""
        for speaking_turn in messages:
            if speaking_turn[ROLE] == USER:
                # output a round if not empty and have assistant message
                # one round starts from user and has at least one assistant message
                if current_round_message and self.assistant_prefix in current_round_message:
                    messages_by_round.append(current_round_message)
                    current_round_message = ""

                current_round_message += "\n" + f"{self.user_prefix}: {speaking_turn[CONTENT]}"
            else:
                current_round_message += "\n" + f"{self.assistant_prefix}: {speaking_turn[CONTENT]}"

        if current_round_message:
            messages_by_round.append(current_round_message)

        # debug
        # print(messages_by_round)

        prompt_message = "\n" + self.assistant_prefix + ":"
        for x in messages_by_round[::-1]:
            if len(tokenizer(self.system_message + "\n" + x + prompt_message)["input_ids"]) <= max_length:
                prompt_message = x + prompt_message
            else:
                break

        return self.system_message + "\n" + prompt_message if prompt_message else None

    def get_conversation(self, messages: List[Dict[str, str]]) -> str:
        return "".join(
            [
                "\n"
                + f"{self.user_prefix if speaking_turn[ROLE] == USER else self.assistant_prefix}: {speaking_turn[CONTENT]}"
                for speaking_turn in messages
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}

# instruct system message
INSTRUCT_SYSTEM_MESSAGE_EN = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
INSTRUCT_SYSTEM_MESSAGE_FR = "Ci-dessous se trouve une instruction qui décrit une tâche à accomplir. Rédigez une réponse qui répond de manière précise à la demande."

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
CONVERSATION_SYSTEM_MESSAGE_FR_SIMPLE = "Voici une conversation entre un utilisateur et un assistant IA nommé Vigogne."


instruct_template_en = InstructTemplate(
    system_message=INSTRUCT_SYSTEM_MESSAGE_EN,
    instruction_prefix="### Instruction",
    output_prefix="### Response",
)

conversation_template_en = ConversationTemplate(
    system_message=CONVERSATION_SYSTEM_MESSAGE_EN,
    user_prefix=f"<|{USER}|>",
    assistant_prefix=f"<|{ASSISTANT}|>",
)


SUPPORTED_DATA_TEMPLATES = {
    INSTRUCT: instruct_template_en,
    CHAT: conversation_template_en,
}


# legacy
def generate_instruct_prompt(instruction: str, input: str = ""):
    return SUPPORTED_DATA_TEMPLATES[INSTRUCT].get_inference_prompt(instruction, input=input)


# legacy
def generate_inference_chat_prompt(
    history: List[List[str]], tokenizer: transformers.PreTrainedTokenizer, max_length: int = 2048
):
    messages = []
    for x in history:
        messages.append({ROLE: USER, CONTENT: x[0]})
        messages.append({ROLE: ASSISTANT, CONTENT: x[1]})
    # tmp fix
    del messages[-1]
    return SUPPORTED_DATA_TEMPLATES[CHAT].get_inference_prompt(messages, tokenizer, max_length=max_length)
