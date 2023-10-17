# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Test cases for the processor utilities."""

import unittest

from vigogne.data_utils import IGNORE_INDEX
from vigogne.processors import alpaca_seq2seq_processor, alpaca_seq2seq_template
from vigogne.utils import VigogneTrainingArguments, load_tokenizer


class TestInstructSeq2SeqProcessor(unittest.TestCase):
    def setUp(self):
        cfg = VigogneTrainingArguments(
            tokenizer_name_or_path="meta-llama/Llama-2-7b-hf",
            output_dir="",
        )

        self.tokenizer = load_tokenizer(cfg)

    def test_template_build_inference_prompt(self):
        example = {
            "instruction": "Explique pourquoi la fraction suivante est équivalente à 1/4",
            "input": "4/16",
            "output": (
                "La fraction 4/16 est équivalente à 1/4 car les numérateurs et les dénominateurs sont tous deux"
                " divisibles par 4. En divisant les nombres du haut et du bas par 4, on obtient la fraction 1/4."
            ),
        }

        generated_text = alpaca_seq2seq_template.build_inference_prompt(example)
        expected_text = """### System:
Ci-dessous se trouve une instruction qui décrit une tâche à accomplir. Rédigez une réponse qui répond de manière précise à la demande.

### Instruction:
Explique pourquoi la fraction suivante est équivalente à 1/4 : 4/16

### Response:
"""
        self.assertEqual(generated_text, expected_text)

    def test_template_build_inference_prompt_customized_system_message(self):
        example = {
            "system": "",
            "instruction": "Explique pourquoi la fraction suivante est équivalente à 1/4",
            "input": "4/16",
            "output": (
                "La fraction 4/16 est équivalente à 1/4 car les numérateurs et les dénominateurs sont tous deux"
                " divisibles par 4. En divisant les nombres du haut et du bas par 4, on obtient la fraction 1/4."
            ),
        }

        generated_text = alpaca_seq2seq_template.build_inference_prompt(example)
        expected_text = """### System:


### Instruction:
Explique pourquoi la fraction suivante est équivalente à 1/4 : 4/16

### Response:
"""
        self.assertEqual(generated_text, expected_text)

    def test_processor_get_example_length(self):
        example = {
            "instruction": "Explique pourquoi la fraction suivante est équivalente à 1/4",
            "input": "4/16",
            "output": (
                "La fraction 4/16 est équivalente à 1/4 car les numérateurs et les dénominateurs sont tous deux"
                " divisibles par 4. En divisant les nombres du haut et du bas par 4, on obtient la fraction 1/4."
            ),
        }

        example_length = len(alpaca_seq2seq_processor.process_example(example, self.tokenizer)["input_ids"])
        expected_example_length = 81
        self.assertEqual(example_length, expected_example_length)

    def test_chat_template(self):
        example = {
            "messages": [
                {"role": "user", "content": "Bonjour."},
                {"role": "assistant", "content": "Bonjour, tu vas bien ?"},
                {"role": "user", "content": "Non, ça ne va pas."},
            ]
        }

        self.tokenizer.chat_template = alpaca_seq2seq_template.default_chat_template()
        generated_text = self.tokenizer.apply_chat_template(example["messages"], add_generation_prompt=True, tokenize=False)

        expected_text = """<s>### System:
Ci-dessous se trouve une instruction qui décrit une tâche à accomplir. Rédigez une réponse qui répond de manière précise à la demande.

### Instruction:
Bonjour.

### Response:
Bonjour, tu vas bien ?</s>

### Instruction:
Non, ça ne va pas.

### Response:
"""
        self.assertEqual(generated_text, expected_text)
