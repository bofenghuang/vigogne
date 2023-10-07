# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Template and processor for Vigogne Instruct models which follow the Alpaca's style."""

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import transformers

from vigogne.data_utils import IGNORE_INDEX, Instruct

# instruct system message
SYSTEM_MESSAGE_EN = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)
SYSTEM_MESSAGE_FR = (
    "Ci-dessous se trouve une instruction qui décrit une tâche à accomplir. Rédigez une réponse qui répond de manière"
    " précise à la demande."
)
DEFAULT_SYSTEM_MESSAGE = SYSTEM_MESSAGE_EN


def merge_instruction_and_input(instruction_text: str, input_text: Optional[str], symbols_to_strip: str = "!,-.:;?~ "):
    if input_text:
        instruction_text = re.sub("[" + re.escape(symbols_to_strip) + "]+$", "", instruction_text)
        instruction_text = f"{instruction_text} : {input_text}"

    return instruction_text


@dataclass
class AlpacaTemplate:
    system_prefix: str = "### System"
    instruction_prefix: str = "### Instruction"
    output_prefix: str = "### Response"
    default_system_message: str = DEFAULT_SYSTEM_MESSAGE

    def _ensure_type(self, instuct: Union[Instruct, Dict]) -> Instruct:
        return Instruct(**instuct) if not isinstance(instuct, Instruct) else instuct

    def _embed_input(self, instuct: Union[Instruct, Dict]) -> Instruct:
        if instuct.input:
            instuct.instruction = merge_instruction_and_input(instuct.instruction, instuct.input)
        return instuct

    def build_training_prompt(
        self,
        instuct: Union[Instruct, Dict],
    ) -> str:
        """eos_token will be added later by tokenizer."""
        instuct = self._ensure_type(instuct)

        prompt_message = self.build_inference_prompt(instuct)
        prompt_message += instuct.output

        return prompt_message

    def build_inference_prompt(self, instuct: Union[Instruct, Dict]) -> str:
        instuct = self._ensure_type(instuct)
        instuct = self._embed_input(instuct)

        system_message = instuct.system if instuct.system is not None else self.default_system_message

        prompt_message = f"{self.system_prefix}:\n{system_message}"
        prompt_message += "\n\n" + f"{self.instruction_prefix}:\n{instuct.instruction}"
        prompt_message += "\n\n" + f"{self.output_prefix}:\n"

        return prompt_message

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}


@dataclass
class AlpacaProcessor(AlpacaTemplate):
    def process_example(
        self,
        example: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model_max_length: Optional[int] = None,
        length_column_name: Optional[str] = None,
        do_mask_input: bool = True,
    ):
        """
        input_tokens = [tokenizer.bos_token] + prompt_tokens + completion_tokens + [tokenizer.eos_token]
        label_tokens = [tokenizer.bos_token] + [-100] * len(prompt_tokens) + completion_tokens + [tokenizer.eos_token]
        """
        # Format prompt
        full_prompt = self.build_training_prompt(example)

        # Tokenize
        input_ids = tokenizer.tok(full_prompt, add_bos_token=True, add_eos_token=True)["input_ids"]

        if model_max_length is not None:
            input_ids = input_ids[:model_max_length]

        # Mask prompt
        # todo: efficient
        if do_mask_input:
            user_prompt = self.build_inference_prompt(example)

            # Get prompt length for masking
            len_user_prompt_tokens = len(
                tokenizer.tok(user_prompt, add_bos_token=True, add_eos_token=False)["input_ids"]
            )

            labels = [IGNORE_INDEX] * len_user_prompt_tokens + input_ids[len_user_prompt_tokens:]
        else:
            labels = input_ids.copy()

        # Tokenize
        # input_ids = tokenizer(full_prompt, truncation=True, return_tensors="pt")["input_ids"][0]
        # labels = input_ids.clone()
        # Mask prompt
        # labels[:len_user_prompt_tokens] = IGNORE_INDEX

        # attention_mask will be added later by collator
        processed_example = {"input_ids": input_ids, "labels": labels}
        if length_column_name is not None:
            processed_example[length_column_name] = len(input_ids)

        return processed_example
