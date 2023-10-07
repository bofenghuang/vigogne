# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Template and processor for Vigogne Chat V3 models, styled based on modifications from Llama-2.
Referring to https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L228-L266
"""


from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import transformers

from vigogne.data_utils import IGNORE_INDEX, Conversation, Role, Utterance

DEFAULT_SYSTEM_MESSAGE = (
    "Vous êtes un assistant IA qui suit extrêmement bien les instructions. Aidez autant que vous le pouvez."
)
DEFAULT_SYSTEM_MESSAGE_GEN = (
    "Vous êtes l'assistant IA nommé Vigogne, créé par Zaion Lab (https://zaion.ai). Vous suivez extrêmement bien les"
    " instructions. Aidez autant que vous le pouvez."
)


@dataclass
class VigogneChatV3Template:
    b_inst = "[INST]"
    e_inst = "[/INST]"
    b_sys = "<<SYS>>\n"
    e_sys = "\n<</SYS>>\n\n"
    default_train_system_message: str = DEFAULT_SYSTEM_MESSAGE
    default_inference_system_message: str = DEFAULT_SYSTEM_MESSAGE_GEN

    def _ensure_type(self, conversation: Union[Conversation, Dict]) -> Conversation:
        conversation = Conversation(**conversation) if not isinstance(conversation, Conversation) else conversation
        assert all([utterance.role == Role.user for utterance in conversation.messages[::2]]) and all(
            [utterance.role == Role.assistant for utterance in conversation.messages[1::2]]
        ), "Conversation should start with User, then Assistant, and alter User/Assistant/User/Assistant/User...)"
        return conversation

    def _embed_system_message(self, conversation: Union[Conversation, Dict], use_train_system_message: bool = True):
        system_message = (
            conversation.system
            if conversation.system is not None
            else (
                self.default_train_system_message
                if use_train_system_message
                else self.default_inference_system_message
            )
        )
        if system_message:
            conversation.messages[0].content = (
                self.b_sys + system_message + self.e_sys + conversation.messages[0].content
            )
        return conversation

    def _build_prompt_by_round(
        self, conversation: Union[Conversation, Dict], tokenizer: transformers.PreTrainedTokenizer
    ):
        # Add eos_token after assistant utterance
        # NB: eos_token might be splitted when concatenating with other characters, depending on tokenizer
        # See https://huggingface.co/mistralai/Mistral-7B-v0.1/discussions/26
        # Can add a space before eos_toke, as per in llama-2 template, but will add a space token after split (["hello", "_", eos_token])
        # Also note the behaviour of fast tokenizer and normal might be different
        # Didn't add strip here, user_utterance.content.strip()
        prompts_by_round = [
            f"{self.b_inst} {user_utterance.content} {self.e_inst} {assitant_utterance.content}{tokenizer.eos_token}"
            for user_utterance, assitant_utterance in zip(conversation.messages[::2], conversation.messages[1::2])
        ]
        return prompts_by_round

    def build_training_prompt(
        self, conversation: Union[Conversation, Dict], tokenizer: transformers.PreTrainedTokenizer
    ) -> str:
        conversation = self._ensure_type(conversation)
        conversation = self._embed_system_message(conversation)
        prompt_message = " ".join(self._build_prompt_by_round(conversation, tokenizer))

        return prompt_message or None

    def build_inference_prompt(
        self,
        conversation: Union[Conversation, Dict],
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int = 2048,
    ) -> str:
        conversation = self._ensure_type(conversation)
        conversation = self._embed_system_message(conversation, use_train_system_message=False)

        # strip last assistant utterance
        if conversation.messages[-1].role == Role.assistant:
            del conversation.messages[-1]

        # build previous prompts
        prompts_by_round = self._build_prompt_by_round(conversation, tokenizer)

        # always keep first round as we need system message inside
        first_round_prompt_message = prompts_by_round.pop(0) if prompts_by_round else None

        # last user utterance
        last_user_utterance_text = f"{self.b_inst} {conversation.messages[-1].content} {self.e_inst}"

        prompt_message = last_user_utterance_text
        # reverse
        for x in prompts_by_round[::-1]:
            if (
                len(
                    tokenizer.tok(
                        f"{first_round_prompt_message} {x} {prompt_message}", add_bos_token=True, add_eos_token=False
                    )["input_ids"]
                )
                < max_length
            ):
                prompt_message = f"{x} {prompt_message}"
            else:
                break

        final_prompt_message = (
            f"{first_round_prompt_message} {prompt_message}"
            if first_round_prompt_message is not None
            else prompt_message
        )

        return final_prompt_message or None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}


@dataclass
class VigogneChatV3Processor(VigogneChatV3Template):
    def process_example(
        self,
        example: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model_max_length: Optional[int] = None,
        length_column_name: Optional[str] = None,
        do_mask_input: bool = True,
    ):
        conversation = self._ensure_type(example)
        conversation = self._embed_system_message(conversation)

        input_ids = [tokenizer.bos_token_id]
        non_ignore_indexes = []

        for utterance in conversation.messages:
            utterance_text = (
                f"{self.b_inst} {utterance.content} {self.e_inst}"
                if utterance.role == Role.user
                else utterance.content
            )

            # w/o bos_token
            # w/ eos_token for assistant, w/o eos_token for user
            utterance_input_ids = tokenizer.tok(
                utterance_text, add_bos_token=False, add_eos_token=utterance.role == Role.assistant
            )["input_ids"]

            input_ids += utterance_input_ids

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
        if length_column_name is not None:
            processed_example[length_column_name] = len(input_ids)

        return processed_example
