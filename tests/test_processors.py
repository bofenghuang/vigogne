# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Test cases for the processor utilities."""

import unittest

from transformers import DataCollatorForSeq2Seq

from vigogne.data_utils import IGNORE_INDEX
from vigogne.processors import (
    alpaca_processor,
    alpaca_template,
    generate_inference_chat_prompt,
    generate_instruct_prompt,
    vigogne_chat_v2_processor,
    vigogne_chat_v2_template,
    vigogne_chat_v3_processor,
    vigogne_chat_v3_template,
)
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

        example_length = alpaca_processor.process_example(
            example, self.tokenizer, length_column_name="example_length"
        )["example_length"]
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


class TestConversationV2Processor(unittest.TestCase):
    def setUp(self):
        cfg = VigogneTrainingArguments(
            tokenizer_name_or_path="meta-llama/Llama-2-7b-hf",
            output_dir="",
        )

        self.tokenizer = load_tokenizer(cfg)

    def test_template_build_inference_prompt(self):
        example = {
            "messages": [
                {"role": "User", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "Assistant",
                    "content": (
                        "1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et"
                        " légumes.\n2. Faites de l'exercice régulièrement pour maintenir votre corps actif et"
                        " fort.\n3. Dormez suffisamment et maintenez un horaire de sommeil régulier."
                    ),
                },
            ]
        }

        generated_text = vigogne_chat_v2_template.build_inference_prompt(example, self.tokenizer)
        expected_text = """<|system|>: Vous êtes l'assistant IA nommé Vigogne, créé par Zaion Lab (https://zaion.ai). Vous suivez extrêmement bien les instructions. Aidez autant que vous le pouvez.
<|user|>: Donne trois conseils pour rester en bonne santé.
<|assistant|>:"""
        self.assertEqual(generated_text, expected_text)

    def test_template_build_train_prompt(self):
        example = {
            "messages": [
                {"role": "User", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "Assistant",
                    "content": (
                        "1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et"
                        " légumes.\n2. Faites de l'exercice régulièrement pour maintenir votre corps actif et"
                        " fort.\n3. Dormez suffisamment et maintenez un horaire de sommeil régulier."
                    ),
                },
            ]
        }

        generated_text = vigogne_chat_v2_template.build_training_prompt(example, self.tokenizer)
        expected_text = """<|system|>: Vous êtes un assistant IA qui suit extrêmement bien les instructions. Aidez autant que vous le pouvez.
<|user|>: Donne trois conseils pour rester en bonne santé.
<|assistant|>: 1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et légumes.
2. Faites de l'exercice régulièrement pour maintenir votre corps actif et fort.
3. Dormez suffisamment et maintenez un horaire de sommeil régulier.</s>"""
        self.assertEqual(generated_text, expected_text)

    def test_template_build_inference_prompt_customized_system_message(self):
        example = {
            "system": "",
            "messages": [
                {"role": "User", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "Assistant",
                    "content": (
                        "1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et"
                        " légumes.\n2. Faites de l'exercice régulièrement pour maintenir votre corps actif et"
                        " fort.\n3. Dormez suffisamment et maintenez un horaire de sommeil régulier."
                    ),
                },
            ],
        }

        generated_text = vigogne_chat_v2_template.build_training_prompt(example, self.tokenizer)
        expected_text = """<|system|>: 
<|user|>: Donne trois conseils pour rester en bonne santé.
<|assistant|>: 1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et légumes.
2. Faites de l'exercice régulièrement pour maintenir votre corps actif et fort.
3. Dormez suffisamment et maintenez un horaire de sommeil régulier.</s>"""
        self.assertEqual(generated_text, expected_text)

    def test_processor_get_example_length(self):
        example = {
            "messages": [
                {"role": "User", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "Assistant",
                    "content": (
                        "1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et"
                        " légumes.\n2. Faites de l'exercice régulièrement pour maintenir votre corps actif et"
                        " fort.\n3. Dormez suffisamment et maintenez un horaire de sommeil régulier."
                    ),
                },
            ]
        }

        example_length = vigogne_chat_v2_processor.process_example(
            example, self.tokenizer, length_column_name="example_length"
        )["example_length"]
        expected_example_length = 145
        self.assertEqual(example_length, expected_example_length)

    def test_processor_process_example(self):
        example = {
            "messages": [
                {"role": "User", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "Assistant",
                    "content": (
                        "1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et"
                        " légumes.\n2. Faites de l'exercice régulièrement pour maintenir votre corps actif et"
                        " fort.\n3. Dormez suffisamment et maintenez un horaire de sommeil régulier."
                    ),
                },
            ]
        }

        processed_example = vigogne_chat_v2_processor.process_example(example, self.tokenizer)
        input_ids = processed_example["input_ids"]
        labels = processed_example["labels"]

        expected_text = """<s> <|system|>: Vous êtes un assistant IA qui suit extrêmement bien les instructions. Aidez autant que vous le pouvez.
<|user|>: Donne trois conseils pour rester en bonne santé.
<|assistant|>: 1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et légumes.
2. Faites de l'exercice régulièrement pour maintenir votre corps actif et fort.
3. Dormez suffisamment et maintenez un horaire de sommeil régulier.</s>"""
        self.assertEqual(self.tokenizer.decode(input_ids), expected_text)

        expected_text = """<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> 1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et légumes.
2. Faites de l'exercice régulièrement pour maintenir votre corps actif et fort.
3. Dormez suffisamment et maintenez un horaire de sommeil régulier.</s>"""
        self.assertEqual(self.tokenizer.decode([l if l != IGNORE_INDEX else 0 for l in labels]), expected_text)

    def test_template_build_inference_prompt_multi_turn(self):
        example = {
            "messages": [
                {"role": "User", "content": "Bonjour."},
                {"role": "Assistant", "content": "Bonjour, tu vas bien ?"},
                {"role": "User", "content": "Non, ça ne va pas."},
                {"role": "Assistant", "content": "Qu'est-ce qui s'est passé ?"},
            ]
        }

        generated_text = vigogne_chat_v2_template.build_inference_prompt(example, self.tokenizer)
        expected_text = """<|system|>: Vous êtes l'assistant IA nommé Vigogne, créé par Zaion Lab (https://zaion.ai). Vous suivez extrêmement bien les instructions. Aidez autant que vous le pouvez.
<|user|>: Bonjour.
<|assistant|>: Bonjour, tu vas bien ?</s>
<|user|>: Non, ça ne va pas.
<|assistant|>:"""
        self.assertEqual(generated_text, expected_text)

    def test_template_build_train_prompt_multi_turn(self):
        example = {
            "messages": [
                {"role": "User", "content": "Bonjour."},
                {"role": "Assistant", "content": "Bonjour, tu vas bien ?"},
                {"role": "User", "content": "Non, ça ne va pas."},
                {"role": "Assistant", "content": "Qu'est-ce qui s'est passé ?"},
            ]
        }

        generated_text = vigogne_chat_v2_template.build_training_prompt(example, self.tokenizer)
        expected_text = """<|system|>: Vous êtes un assistant IA qui suit extrêmement bien les instructions. Aidez autant que vous le pouvez.
<|user|>: Bonjour.
<|assistant|>: Bonjour, tu vas bien ?</s>
<|user|>: Non, ça ne va pas.
<|assistant|>: Qu'est-ce qui s'est passé ?</s>"""
        self.assertEqual(generated_text, expected_text)

    def test_processor_process_example_multi_turn(self):
        example = {
            "messages": [
                {"role": "User", "content": "Bonjour."},
                {"role": "Assistant", "content": "Bonjour, tu vas bien ?"},
                {"role": "User", "content": "Non, ça ne va pas."},
                {"role": "Assistant", "content": "Qu'est-ce qui s'est passé ?"},
            ]
        }

        processed_example = vigogne_chat_v2_processor.process_example(example, self.tokenizer)
        input_ids = processed_example["input_ids"]
        labels = processed_example["labels"]

        expected_text = """<s> <|system|>: Vous êtes un assistant IA qui suit extrêmement bien les instructions. Aidez autant que vous le pouvez.
<|user|>: Bonjour.
<|assistant|>: Bonjour, tu vas bien ?</s>
<|user|>: Non, ça ne va pas.
<|assistant|>: Qu'est-ce qui s'est passé ?</s>"""
        self.assertEqual(self.tokenizer.decode(input_ids), expected_text)

        expected_text = """<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> Bonjour, tu vas bien ?</s><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> Qu'est-ce qui s'est passé ?</s>"""
        self.assertEqual(self.tokenizer.decode([l if l != IGNORE_INDEX else 0 for l in labels]), expected_text)

    def test_compare_example_processing_and_promot_tokenization(self):
        example = {
            "messages": [
                {"role": "User", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "Assistant",
                    "content": (
                        "1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et"
                        " légumes.\n2. Faites de l'exercice régulièrement pour maintenir votre corps actif et"
                        " fort.\n3. Dormez suffisamment et maintenez un horaire de sommeil régulier."
                    ),
                },
            ]
        }

        processed_input_ids = vigogne_chat_v2_processor.process_example(example, self.tokenizer)["input_ids"]
        tokenized_input_ids = self.tokenizer.tok(
            vigogne_chat_v2_processor.build_training_prompt(example, self.tokenizer),
            add_bos_token=True,
            add_eos_token=True,
        )["input_ids"]
        self.assertEqual(processed_input_ids, tokenized_input_ids)

    # def test_compare_example_processing_and_promot_tokenization_multi_turn(self):
    #     example = {
    #         "messages": [
    #             {"role": "User", "content": "Bonjour."},
    #             {"role": "Assistant", "content": "Bonjour, tu vas bien ?"},
    #             {"role": "User", "content": "Non, ça ne va pas."},
    #             {"role": "Assistant", "content": "Qu'est-ce qui s'est passé ?"},
    #         ]
    #     }

    #     processed_input_ids = vigogne_chat_v2_processor.process_example(example, self.tokenizer)["input_ids"]
    #     tokenized_input_ids = self.tokenizer.tok(
    #         vigogne_chat_v2_processor.build_training_prompt(example, self.tokenizer), add_bos_token=True, add_eos_token=True
    #     )["input_ids"]
    #     self.assertEqual(processed_input_ids, tokenized_input_ids)

    def test_lagacy_generate_inference_chat_prompt(self):
        history = [["Donne trois conseils pour rester en bonne santé.", ""]]

        generated_text = generate_inference_chat_prompt(history, self.tokenizer)
        expected_text = """<|system|>: Vous êtes l'assistant IA nommé Vigogne, créé par Zaion Lab (https://zaion.ai). Vous suivez extrêmement bien les instructions. Aidez autant que vous le pouvez.
<|user|>: Donne trois conseils pour rester en bonne santé.
<|assistant|>:"""
        self.assertEqual(generated_text, expected_text)


class TestConversationV3Processor(unittest.TestCase):
    def setUp(self):
        cfg = VigogneTrainingArguments(
            tokenizer_name_or_path="meta-llama/Llama-2-7b-hf",
            tokenizer_padding_side="right",
            output_dir="",
        )

        self.tokenizer = load_tokenizer(cfg)

    def test_template_build_inference_prompt(self):
        example = {
            "messages": [
                {"role": "User", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "Assistant",
                    "content": (
                        "1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et"
                        " légumes.\n2. Faites de l'exercice régulièrement pour maintenir votre corps actif et"
                        " fort.\n3. Dormez suffisamment et maintenez un horaire de sommeil régulier."
                    ),
                },
            ]
        }

        generated_text = vigogne_chat_v3_template.build_inference_prompt(example, self.tokenizer)
        expected_text = """[INST] <<SYS>>
Vous êtes l'assistant IA nommé Vigogne, créé par Zaion Lab (https://zaion.ai). Vous suivez extrêmement bien les instructions. Aidez autant que vous le pouvez.
<</SYS>>

Donne trois conseils pour rester en bonne santé. [/INST]"""
        self.assertEqual(generated_text, expected_text)

    def test_template_build_train_prompt(self):
        example = {
            "messages": [
                {"role": "User", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "Assistant",
                    "content": (
                        "1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et"
                        " légumes.\n2. Faites de l'exercice régulièrement pour maintenir votre corps actif et"
                        " fort.\n3. Dormez suffisamment et maintenez un horaire de sommeil régulier."
                    ),
                },
            ]
        }

        generated_text = vigogne_chat_v3_template.build_training_prompt(example, self.tokenizer)
        expected_text = """[INST] <<SYS>>
Vous êtes un assistant IA qui suit extrêmement bien les instructions. Aidez autant que vous le pouvez.
<</SYS>>

Donne trois conseils pour rester en bonne santé. [/INST] 1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et légumes.
2. Faites de l'exercice régulièrement pour maintenir votre corps actif et fort.
3. Dormez suffisamment et maintenez un horaire de sommeil régulier.</s>"""
        self.assertEqual(generated_text, expected_text)

    def test_template_build_inference_prompt_customized_system_message(self):
        example = {
            "system": "",
            "messages": [
                {"role": "User", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "Assistant",
                    "content": (
                        "1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et"
                        " légumes.\n2. Faites de l'exercice régulièrement pour maintenir votre corps actif et"
                        " fort.\n3. Dormez suffisamment et maintenez un horaire de sommeil régulier."
                    ),
                },
            ],
        }

        generated_text = vigogne_chat_v3_template.build_training_prompt(example, self.tokenizer)
        expected_text = """[INST] Donne trois conseils pour rester en bonne santé. [/INST] 1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et légumes.
2. Faites de l'exercice régulièrement pour maintenir votre corps actif et fort.
3. Dormez suffisamment et maintenez un horaire de sommeil régulier.</s>"""
        self.assertEqual(generated_text, expected_text)

    def test_processor_get_example_length(self):
        example = {
            "messages": [
                {"role": "User", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "Assistant",
                    "content": (
                        "1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et"
                        " légumes.\n2. Faites de l'exercice régulièrement pour maintenir votre corps actif et"
                        " fort.\n3. Dormez suffisamment et maintenez un horaire de sommeil régulier."
                    ),
                },
            ]
        }

        example_length = vigogne_chat_v3_processor.process_example(
            example, self.tokenizer, length_column_name="example_length"
        )["example_length"]
        expected_example_length = 147
        self.assertEqual(example_length, expected_example_length)

    def test_processor_process_example(self):
        example = {
            "messages": [
                {"role": "User", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "Assistant",
                    "content": (
                        "1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et"
                        " légumes.\n2. Faites de l'exercice régulièrement pour maintenir votre corps actif et"
                        " fort.\n3. Dormez suffisamment et maintenez un horaire de sommeil régulier."
                    ),
                },
            ]
        }

        processed_example = vigogne_chat_v3_processor.process_example(example, self.tokenizer)
        input_ids = processed_example["input_ids"]
        labels = processed_example["labels"]

        expected_text = """<s> [INST] <<SYS>>
Vous êtes un assistant IA qui suit extrêmement bien les instructions. Aidez autant que vous le pouvez.
<</SYS>>

Donne trois conseils pour rester en bonne santé. [/INST] 1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et légumes.
2. Faites de l'exercice régulièrement pour maintenir votre corps actif et fort.
3. Dormez suffisamment et maintenez un horaire de sommeil régulier.</s>"""
        self.assertEqual(self.tokenizer.decode(input_ids), expected_text)

        expected_text = """<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> 1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et légumes.
2. Faites de l'exercice régulièrement pour maintenir votre corps actif et fort.
3. Dormez suffisamment et maintenez un horaire de sommeil régulier.</s>"""
        self.assertEqual(self.tokenizer.decode([l if l != IGNORE_INDEX else 0 for l in labels]), expected_text)

    def test_template_build_inference_prompt_multi_turn(self):
        example = {
            "messages": [
                {"role": "User", "content": "Bonjour."},
                {"role": "Assistant", "content": "Bonjour, tu vas bien ?"},
                {"role": "User", "content": "Non, ça ne va pas."},
                {"role": "Assistant", "content": "Qu'est-ce qui s'est passé ?"},
            ]
        }

        generated_text = vigogne_chat_v2_template.build_inference_prompt(example, self.tokenizer)
        expected_text = """<|system|>: Vous êtes l'assistant IA nommé Vigogne, créé par Zaion Lab (https://zaion.ai). Vous suivez extrêmement bien les instructions. Aidez autant que vous le pouvez.
<|user|>: Bonjour.
<|assistant|>: Bonjour, tu vas bien ?</s>
<|user|>: Non, ça ne va pas.
<|assistant|>:"""
        self.assertEqual(generated_text, expected_text)

    def test_template_build_train_prompt_multi_turn(self):
        example = {
            "messages": [
                {"role": "User", "content": "Bonjour."},
                {"role": "Assistant", "content": "Bonjour, tu vas bien ?"},
                {"role": "User", "content": "Non, ça ne va pas."},
                {"role": "Assistant", "content": "Qu'est-ce qui s'est passé ?"},
            ]
        }

        generated_text = vigogne_chat_v2_template.build_training_prompt(example, self.tokenizer)
        expected_text = """<|system|>: Vous êtes un assistant IA qui suit extrêmement bien les instructions. Aidez autant que vous le pouvez.
<|user|>: Bonjour.
<|assistant|>: Bonjour, tu vas bien ?</s>
<|user|>: Non, ça ne va pas.
<|assistant|>: Qu'est-ce qui s'est passé ?</s>"""
        self.assertEqual(generated_text, expected_text)

    def test_processor_process_example_multi_turn(self):
        example = {
            "messages": [
                {"role": "User", "content": "Bonjour."},
                {"role": "Assistant", "content": "Bonjour, tu vas bien ?"},
                {"role": "User", "content": "Non, ça ne va pas."},
                {"role": "Assistant", "content": "Qu'est-ce qui s'est passé ?"},
            ]
        }

        processed_example = vigogne_chat_v3_processor.process_example(example, self.tokenizer)
        input_ids = processed_example["input_ids"]
        labels = processed_example["labels"]

        expected_text = """<s> [INST] <<SYS>>
Vous êtes un assistant IA qui suit extrêmement bien les instructions. Aidez autant que vous le pouvez.
<</SYS>>

Bonjour. [/INST] Bonjour, tu vas bien ?</s> [INST] Non, ça ne va pas. [/INST] Qu'est-ce qui s'est passé ?</s>"""
        self.assertEqual(self.tokenizer.decode(input_ids), expected_text)

        expected_text = """<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> Bonjour, tu vas bien ?</s><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> Qu'est-ce qui s'est passé ?</s>"""
        self.assertEqual(self.tokenizer.decode([l if l != IGNORE_INDEX else 0 for l in labels]), expected_text)

    def test_compare_example_processing_and_promot_tokenization(self):
        example = {
            "messages": [
                {"role": "User", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "Assistant",
                    "content": (
                        "1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et"
                        " légumes.\n2. Faites de l'exercice régulièrement pour maintenir votre corps actif et"
                        " fort.\n3. Dormez suffisamment et maintenez un horaire de sommeil régulier."
                    ),
                },
            ]
        }

        processed_input_ids = vigogne_chat_v3_processor.process_example(example, self.tokenizer)["input_ids"]
        tokenized_input_ids = self.tokenizer.tok(
            vigogne_chat_v3_processor.build_training_prompt(example, self.tokenizer),
            add_bos_token=True,
            add_eos_token=True,
        )["input_ids"]
        self.assertEqual(processed_input_ids, tokenized_input_ids)

    # def test_compare_example_processing_and_promot_tokenization_multi_turn(self):
    #     example = {
    #         "messages": [
    #             {"role": "User", "content": "Bonjour."},
    #             {"role": "Assistant", "content": "Bonjour, tu vas bien ?"},
    #             {"role": "User", "content": "Non, ça ne va pas."},
    #             {"role": "Assistant", "content": "Qu'est-ce qui s'est passé ?"},
    #         ]
    #     }

    #     processed_input_ids = vigogne_chat_v3_processor.process_example(example, self.tokenizer)["input_ids"]
    #     tokenized_input_ids = self.tokenizer.tok(
    #         vigogne_chat_v3_processor.build_training_prompt(example, self.tokenizer), add_bos_token=True, add_eos_token=True
    #     )["input_ids"]
    #     self.assertEqual(processed_input_ids, tokenized_input_ids)

    def test_processor_process_example_padding_right(self):
        examples = [
            {
                "system": "",
                "messages": [
                    {"role": "User", "content": "Donne trois conseils pour rester en bonne santé."},
                    {
                        "role": "Assistant",
                        "content": (
                            "1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et"
                            " légumes.\n2. Faites de l'exercice régulièrement pour maintenir votre corps actif et"
                            " fort.\n3. Dormez suffisamment et maintenez un horaire de sommeil régulier."
                        ),
                    },
                ],
            },
            {
                "system": "",
                "messages": [
                    {"role": "User", "content": "Bonjour."},
                    {"role": "Assistant", "content": "Bonjour, tu vas bien ?"},
                    {"role": "User", "content": "Non, ça ne va pas."},
                    {"role": "Assistant", "content": "Qu'est-ce qui s'est passé ?"},
                ],
            },
        ]

        feautres = [vigogne_chat_v3_processor.process_example(example, self.tokenizer) for example in examples]

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, label_pad_token_id=IGNORE_INDEX, pad_to_multiple_of=8)
        collated_features = data_collator(feautres)
        input_ids = collated_features["input_ids"]
        labels = collated_features["labels"]
        attention_mask = collated_features["attention_mask"]

        expected_text = """<s> [INST] Donne trois conseils pour rester en bonne santé. [/INST] 1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et légumes.
2. Faites de l'exercice régulièrement pour maintenir votre corps actif et fort.
3. Dormez suffisamment et maintenez un horaire de sommeil régulier.</s></s></s></s></s></s></s></s>"""
        self.assertEqual(self.tokenizer.decode(input_ids[0]), expected_text)

        expected_text = """<s> [INST] Bonjour. [/INST] Bonjour, tu vas bien ?</s> [INST] Non, ça ne va pas. [/INST] Qu'est-ce qui s'est passé ?</s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s>"""
        self.assertEqual(self.tokenizer.decode(input_ids[1]), expected_text)

        expected_text = """<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> 1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et légumes.
2. Faites de l'exercice régulièrement pour maintenir votre corps actif et fort.
3. Dormez suffisamment et maintenez un horaire de sommeil régulier.</s><unk><unk><unk><unk><unk><unk><unk>"""
        self.assertEqual(self.tokenizer.decode([l if l != IGNORE_INDEX else 0 for l in labels[0]]), expected_text)

        expected_text = """<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> Bonjour, tu vas bien ?</s><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> Qu'est-ce qui s'est passé ?</s><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>"""
        self.assertEqual(self.tokenizer.decode([l if l != IGNORE_INDEX else 0 for l in labels[1]]), expected_text)

        self.assertEqual(attention_mask[0].sum().item(), 105)
        self.assertEqual(attention_mask[1].sum().item(), 49)

    def test_processor_process_example_padding_left(self):
        examples = [
            {
                "system": "",
                "messages": [
                    {"role": "User", "content": "Donne trois conseils pour rester en bonne santé."},
                    {
                        "role": "Assistant",
                        "content": (
                            "1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et"
                            " légumes.\n2. Faites de l'exercice régulièrement pour maintenir votre corps actif et"
                            " fort.\n3. Dormez suffisamment et maintenez un horaire de sommeil régulier."
                        ),
                    },
                ],
            },
            {
                "system": "",
                "messages": [
                    {"role": "User", "content": "Bonjour."},
                    {"role": "Assistant", "content": "Bonjour, tu vas bien ?"},
                    {"role": "User", "content": "Non, ça ne va pas."},
                    {"role": "Assistant", "content": "Qu'est-ce qui s'est passé ?"},
                ],
            },
        ]

        self.tokenizer.padding_side = "left"

        feautres = [vigogne_chat_v3_processor.process_example(example, self.tokenizer) for example in examples]

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, label_pad_token_id=IGNORE_INDEX, pad_to_multiple_of=8)
        collated_features = data_collator(feautres)
        input_ids = collated_features["input_ids"]
        labels = collated_features["labels"]
        attention_mask = collated_features["attention_mask"]

        expected_text = """</s></s></s></s></s></s></s><s> [INST] Donne trois conseils pour rester en bonne santé. [/INST] 1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et légumes.
2. Faites de l'exercice régulièrement pour maintenir votre corps actif et fort.
3. Dormez suffisamment et maintenez un horaire de sommeil régulier.</s>"""
        self.assertEqual(self.tokenizer.decode(input_ids[0]), expected_text)

        expected_text = """</s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s><s> [INST] Bonjour. [/INST] Bonjour, tu vas bien ?</s> [INST] Non, ça ne va pas. [/INST] Qu'est-ce qui s'est passé ?</s>"""
        self.assertEqual(self.tokenizer.decode(input_ids[1]), expected_text)

        expected_text = """<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> 1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et légumes.
2. Faites de l'exercice régulièrement pour maintenir votre corps actif et fort.
3. Dormez suffisamment et maintenez un horaire de sommeil régulier.</s>"""
        self.assertEqual(self.tokenizer.decode([l if l != IGNORE_INDEX else 0 for l in labels[0]]), expected_text)

        expected_text = """<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> Bonjour, tu vas bien ?</s><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> Qu'est-ce qui s'est passé ?</s>"""
        self.assertEqual(self.tokenizer.decode([l if l != IGNORE_INDEX else 0 for l in labels[1]]), expected_text)

        self.tokenizer.padding_side = "right"
