# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Test cases for the processor utilities."""

import unittest

from vigogne.data_utils import IGNORE_INDEX
from vigogne.preprocess import generate_instruct_prompt
from vigogne.processors import alpaca_processor, alpaca_template
from vigogne.utils import VigogneTrainingArguments, load_tokenizer


class TestInstructProcessor(unittest.TestCase):
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

        generated_text = alpaca_template.build_inference_prompt(example)
        expected_text = """### System:
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Explique pourquoi la fraction suivante est équivalente à 1/4 : 4/16

### Response:
"""
        self.assertEqual(generated_text, expected_text)

    def test_template_build_train_prompt(self):
        example = {
            "instruction": "Explique pourquoi la fraction suivante est équivalente à 1/4",
            "input": "4/16",
            "output": (
                "La fraction 4/16 est équivalente à 1/4 car les numérateurs et les dénominateurs sont tous deux"
                " divisibles par 4. En divisant les nombres du haut et du bas par 4, on obtient la fraction 1/4."
            ),
        }

        generated_text = alpaca_template.build_training_prompt(example)
        expected_text = """### System:
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Explique pourquoi la fraction suivante est équivalente à 1/4 : 4/16

### Response:
La fraction 4/16 est équivalente à 1/4 car les numérateurs et les dénominateurs sont tous deux divisibles par 4. En divisant les nombres du haut et du bas par 4, on obtient la fraction 1/4."""
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

        generated_text = alpaca_template.build_inference_prompt(example)
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

        example_length = len(alpaca_processor.process_example(example, self.tokenizer)["input_ids"])
        expected_example_length = 128
        self.assertEqual(example_length, expected_example_length)

    def test_processor_process_example(self):
        example = {
            "instruction": "Explique pourquoi la fraction suivante est équivalente à 1/4",
            "input": "4/16",
            "output": (
                "La fraction 4/16 est équivalente à 1/4 car les numérateurs et les dénominateurs sont tous deux"
                " divisibles par 4. En divisant les nombres du haut et du bas par 4, on obtient la fraction 1/4."
            ),
        }

        processed_example = alpaca_processor.process_example(example, self.tokenizer)
        input_ids = processed_example["input_ids"]
        labels = processed_example["labels"]

        expected_text = """<s> ### System:
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Explique pourquoi la fraction suivante est équivalente à 1/4 : 4/16

### Response:
La fraction 4/16 est équivalente à 1/4 car les numérateurs et les dénominateurs sont tous deux divisibles par 4. En divisant les nombres du haut et du bas par 4, on obtient la fraction 1/4.</s>"""
        self.assertEqual(self.tokenizer.decode(input_ids), expected_text)

        expected_text = """<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>La fraction 4/16 est équivalente à 1/4 car les numérateurs et les dénominateurs sont tous deux divisibles par 4. En divisant les nombres du haut et du bas par 4, on obtient la fraction 1/4.</s>"""
        self.assertEqual(self.tokenizer.decode([l if l != IGNORE_INDEX else 0 for l in labels]), expected_text)

    def test_compare_example_processing_and_promot_tokenization(self):
        example = {
            "instruction": "Explique pourquoi la fraction suivante est équivalente à 1/4",
            "input": "4/16",
            "output": (
                "La fraction 4/16 est équivalente à 1/4 car les numérateurs et les dénominateurs sont tous deux"
                " divisibles par 4. En divisant les nombres du haut et du bas par 4, on obtient la fraction 1/4."
            ),
        }

        processed_input_ids = alpaca_processor.process_example(example, self.tokenizer)["input_ids"]
        tokenized_input_ids = self.tokenizer.tok(
            alpaca_processor.build_training_prompt(example), add_bos_token=True, add_eos_token=True
        )["input_ids"]
        self.assertEqual(processed_input_ids, tokenized_input_ids)

    def test_lagacy_generate_instruct_prompt(self):
        instruction = "Donne trois conseils pour rester en bonne santé."

        generated_text = generate_instruct_prompt(instruction)
        expected_text = """### System:
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Donne trois conseils pour rester en bonne santé.

### Response:
"""
        self.assertEqual(generated_text, expected_text)
