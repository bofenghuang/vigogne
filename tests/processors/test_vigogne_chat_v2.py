# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Test cases for the processor utilities."""

import unittest

from vigogne.data_utils import IGNORE_INDEX
from vigogne.preprocess import generate_inference_chat_prompt
from vigogne.processors import vigogne_chat_v2_processor, vigogne_chat_v2_template
from vigogne.utils import VigogneTrainingArguments, load_tokenizer


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
                {"role": "user", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "assistant",
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
                {"role": "user", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "assistant",
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
                {"role": "user", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "assistant",
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
                {"role": "user", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "assistant",
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
                {"role": "user", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "assistant",
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
                {"role": "user", "content": "Bonjour."},
                {"role": "assistant", "content": "Bonjour, tu vas bien ?"},
                {"role": "user", "content": "Non, ça ne va pas."},
                {"role": "assistant", "content": "Qu'est-ce qui s'est passé ?"},
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
                {"role": "user", "content": "Bonjour."},
                {"role": "assistant", "content": "Bonjour, tu vas bien ?"},
                {"role": "user", "content": "Non, ça ne va pas."},
                {"role": "assistant", "content": "Qu'est-ce qui s'est passé ?"},
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
                {"role": "user", "content": "Bonjour."},
                {"role": "assistant", "content": "Bonjour, tu vas bien ?"},
                {"role": "user", "content": "Non, ça ne va pas."},
                {"role": "assistant", "content": "Qu'est-ce qui s'est passé ?"},
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
                {"role": "user", "content": "Donne trois conseils pour rester en bonne santé."},
                {
                    "role": "assistant",
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

    # todo
    # def test_compare_example_processing_and_chat_template(self):
    #     example = {
    #         "messages": [
    #             {"role": "user", "content": "Bonjour."},
    #             {"role": "assistant", "content": "Bonjour, tu vas bien ?"},
    #             {"role": "user", "content": "Non, ça ne va pas."},
    #             {"role": "assistant", "content": "Qu'est-ce qui s'est passé ?"},
    #         ]
    #     }
    #     # processed_input_ids = vigogne_chat_v2_processor.process_example(example, self.tokenizer)["input_ids"]
    #     processed_input_ids = self.tokenizer.tok(
    #         vigogne_chat_v2_template.build_inference_prompt(example, self.tokenizer),
    #         add_bos_token=True,
    #         add_eos_token=False,
    #     )["input_ids"]

    #     self.tokenizer.chat_template = vigogne_chat_v2_processor.default_chat_template(use_train_system_prompt=True)
    #     tokenized_input_ids = self.tokenizer.apply_chat_template(example["messages"], add_generation_prompt=True)
    #     self.assertEqual(processed_input_ids, tokenized_input_ids)

    # def test_compare_example_processing_and_promot_tokenization_multi_turn(self):
    #     example = {
    #         "messages": [
    #             {"role": "user", "content": "Bonjour."},
    #             {"role": "assistant", "content": "Bonjour, tu vas bien ?"},
    #             {"role": "user", "content": "Non, ça ne va pas."},
    #             {"role": "assistant", "content": "Qu'est-ce qui s'est passé ?"},
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
