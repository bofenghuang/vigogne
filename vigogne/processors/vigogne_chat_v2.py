# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Template and processor for Vigogne Chat V2 models."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import transformers

from vigogne.data_utils import IGNORE_INDEX, Conversation, Role

# conversation system message v1
SYSTEM_MESSAGE_EN_V1 = """Below is a conversation between a user and an AI assistant named Vigogne.
Vigogne is an open-source AI assistant created by Zaion (https://zaion.ai/).
Vigogne is polite, emotionally aware, humble-but-knowledgeable, always providing helpful and detailed answers.
Vigogne is skilled in responding proficiently in the languages its users use and can perform a wide range of tasks such as text editing, translation, question answering, logical reasoning, coding, and many others.
Vigogne cannot receive or generate audio or visual content and cannot access the internet.
Vigogne strictly avoids discussing sensitive, offensive, illegal, ethical, or political topics and caveats when unsure of the answer."""
SYSTEM_MESSAGE_FR_V1 = """Voici une conversation entre un utilisateur et un assistant IA nommé Vigogne.
Vigogne est un assistant IA open-source créé par Zaion (https://zaion.ai/).
Vigogne est respectueux, empathique, humble mais bien informé, et fournit toujours des réponses utiles et détaillées.
Vigogne est capable d'effectuer une large variété de tâches telles que l'édition de texte, la traduction, la question answering, la raisonnement logique, le codage et bien d'autres encore.
Vigogne ne peut pas recevoir ou générer de contenu audio ou visuel et ne peut pas accéder à Internet.
Vigogne évite strictement de discuter de sujets sensibles, offensants, illégaux, éthiques ou politiques et met en garde lorsqu'il n'est pas sûr de la réponse."""
# conversation system message v2
SYSTEM_MESSAGE_EN_V2 = "You are an AI assistant that follows instructions extremely well. Help as much as you can."
SYSTEM_MESSAGE_FR_V2 = "Vous êtes un assistant IA qui suit extrêmement bien les instructions. Aidez autant que vous le pouvez."
DEFAULT_SYSTEM_MESSAGE = SYSTEM_MESSAGE_FR_V2
SYSTEM_MESSAGE_GEN_EN_V2 = (
    "You are Vigogne, an AI assistant created by Zaion Lab. You follow instructions extremely well. Help as much as you can."
)
SYSTEM_MESSAGE_GEN_FR_V2 = (
    "Vous êtes Vigogne, un assistant IA créé par Zaion Lab. Vous suivez extrêmement bien les instructions. Aidez autant que"
    " vous le pouvez."
)
DEFAULT_SYSTEM_MESSAGE_GEN = SYSTEM_MESSAGE_GEN_FR_V2


@dataclass
class VigogneChatV2Template:
    system_prefix: str = "<|system|>"
    user_prefix: str = "<|user|>"
    assistant_prefix: str = "<|assistant|>"
    default_train_system_message: str = DEFAULT_SYSTEM_MESSAGE
    default_inference_system_message: str = DEFAULT_SYSTEM_MESSAGE_GEN

    def _ensure_type(self, conversation: Union[Conversation, Dict]) -> Conversation:
        conversation = Conversation(**conversation) if not isinstance(conversation, Conversation) else conversation
        assert all([utterance.role == Role.user for utterance in conversation.messages[::2]]) and all(
            [utterance.role == Role.assistant for utterance in conversation.messages[1::2]]
        ), "Conversation should start with User, then Assistant, and alter User/Assistant/User/Assistant/User...)"
        return conversation

    def _embed_system_message(self, conversation: Union[Conversation, Dict], use_train_system_message: bool = True) -> str:
        system_message = (
            conversation.system
            if conversation.system is not None
            else (self.default_train_system_message if use_train_system_message else self.default_inference_system_message)
        )
        system_header_text = f"{self.system_prefix}: {system_message}"
        # Remove whole line if empty system_message
        # system_header_text = f"{self.system_prefix}: {system_message}" if system_message else ""
        return system_header_text

    def _build_prompt_by_round(self, conversation: Union[Conversation, Dict], tokenizer: transformers.PreTrainedTokenizer):
        # Add eos_token after assistant utterance
        # NB: eos_token might be splitted when concatenating with other characters, depending on tokenizer
        # See https://huggingface.co/mistralai/Mistral-7B-v0.1/discussions/26
        # Can add a space before eos_toke, as per in llama-2 template, but will add a space token after split (["hello", "_", eos_token])
        # Also note the behaviour of fast tokenizer and normal might be different
        # Didn't add strip here, user_utterance.content.strip()
        prompts_by_round = [
            f"\n{self.user_prefix}: {user_utterance.content}\n{self.assistant_prefix}:"
            f" {assitant_utterance.content}{tokenizer.eos_token}"
            for user_utterance, assitant_utterance in zip(conversation.messages[::2], conversation.messages[1::2])
        ]
        return prompts_by_round

    def build_training_prompt(
        self, conversation: Union[Conversation, Dict], tokenizer: transformers.PreTrainedTokenizer
    ) -> str:
        conversation = self._ensure_type(conversation)
        system_header_text = self._embed_system_message(conversation)
        prompt_message = "".join(self._build_prompt_by_round(conversation, tokenizer))

        return system_header_text + prompt_message if prompt_message else None

    def build_inference_prompt(
        self,
        conversation: Union[Conversation, Dict],
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int = 2048,
    ) -> str:
        conversation = self._ensure_type(conversation)
        system_header_text = self._embed_system_message(conversation, use_train_system_message=False)

        # strip last assistant utterance
        if conversation.messages[-1].role == Role.assistant:
            del conversation.messages[-1]

        # build previous prompts
        prompts_by_round = self._build_prompt_by_round(conversation, tokenizer)

        # last user utterance
        last_user_utterance_text = f"\n{self.user_prefix}: {conversation.messages[-1].content}\n{self.assistant_prefix}:"

        prompt_message = last_user_utterance_text
        # reverse
        for x in prompts_by_round[::-1]:
            if (
                len(
                    tokenizer.tok(system_header_text + x + prompt_message, add_bos_token=True, add_eos_token=False)[
                        "input_ids"
                    ]
                )
                < max_length
            ):
                prompt_message = x + prompt_message
            else:
                break

        return system_header_text + prompt_message if prompt_message else None

    def default_chat_template(
        self,
        default_system_message: Optional[str] = None,
        use_default_system_prompt: bool = True,
        use_train_system_prompt: bool = False,
    ):
        default_system_message = (
            default_system_message
            if default_system_message is not None
            else (self.default_train_system_message if use_train_system_prompt else self.default_inference_system_message)
        )

        template = (
            "{{ bos_token }}"
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
            "{% set system_message = messages[0]['content'] %}"
            "{% elif USE_DEFAULT_PROMPT == true %}"
            "{% set loop_messages = messages %}"  # Or use the default system message if the flag is set
            "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = false %}"
            "{% endif %}"
            "{% if system_message != false %}"
            "{{ '<|system|>: ' + system_message + '\\n' }}"
            "{% endif %}"
            "{% for message in loop_messages %}"  # Loop over all non-system messages
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|user|>: ' + message['content'].strip() + '\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|assistant|>: ' + message['content'].strip() + eos_token + '\\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|assistant|>:' }}"  # Add generation prompt
            "{% endif %}"
        )
        template = template.replace("USE_DEFAULT_PROMPT", "true" if use_default_system_prompt else "false")
        default_message = default_system_message.replace("\n", "\\n").replace("'", "\\'")
        template = template.replace("DEFAULT_SYSTEM_MESSAGE", default_message)
        return template

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}


@dataclass
class VigogneChatV2Processor(VigogneChatV2Template):
    def process_example(
        self,
        example: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model_max_length: Optional[int] = None,
        do_mask_input: bool = True,
    ):
        conversation = self._ensure_type(example)

        # system header tokens
        system_header_text = self._embed_system_message(conversation)
        # w/ bos_token, w/o eos_token
        input_ids = tokenizer.tok(system_header_text, add_bos_token=True, add_eos_token=False)["input_ids"]

        # user/assistant prefix tokens
        # w/o bos_token or eos_token
        user_prefix_input_ids = tokenizer.tok(f"\n{self.user_prefix}:", add_bos_token=False, add_eos_token=False)["input_ids"]
        assistant_prefix_input_ids = tokenizer.tok(f"\n{self.assistant_prefix}:", add_bos_token=False, add_eos_token=False)[
            "input_ids"
        ]

        # NB: might be incorrect for other tokenizers than llama depending on config
        # tmp fix for llama-2
        # tokenizer.tokenize("hello\n<user>:") -> ['▁hello', '<0x0A>', '<', 'user', '>:']
        # tokenizer.tokenize("\n<user>:") -> ['▁', '<0x0A>', '<', 'user', '>:']
        # Remove '▁' token to ensure consistant behaviour when tokenizing the entire prompt
        if tokenizer.__class__.__name__ in ["LlamaTokenizer", "LlamaTokenizerFast", "CodeLlamaTokenizer"]:
            user_prefix_input_ids = user_prefix_input_ids[1:]
            assistant_prefix_input_ids = assistant_prefix_input_ids[1:]

        non_ignore_indexes = []
        for utterance in conversation.messages:
            # w/o bos_token
            # w/ eos_token for assistant, w/o eos_token for user
            utterance_input_ids = tokenizer.tok(
                utterance.content, add_bos_token=False, add_eos_token=utterance.role == Role.assistant
            )["input_ids"]

            prefix_input_ids = assistant_prefix_input_ids if utterance.role == Role.assistant else user_prefix_input_ids
            input_ids += prefix_input_ids + utterance_input_ids

            # note token indexes for reponse
            if utterance.role == Role.assistant:
                non_ignore_indexes.append([len(input_ids) - len(utterance_input_ids), len(input_ids)])

        if model_max_length is not None:
            input_ids = input_ids[:model_max_length]

        # mask system message, user prompt, and all prefix tokens
        # todo: efficient
        if do_mask_input:
            labels = [IGNORE_INDEX] * len(input_ids)

            for non_ignore_s, non_ignore_e in non_ignore_indexes:
                labels[non_ignore_s:non_ignore_e] = input_ids[non_ignore_s:non_ignore_e]
        else:
            labels = input_ids.copy()

        processed_example = {"input_ids": input_ids, "labels": labels}

        return processed_example
