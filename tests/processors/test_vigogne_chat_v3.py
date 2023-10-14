# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Test cases for the processor utilities."""

import unittest

from transformers import DataCollatorForSeq2Seq

from vigogne.data_utils import IGNORE_INDEX
from vigogne.processors import vigogne_chat_v3_processor, vigogne_chat_v3_template
from vigogne.utils import VigogneTrainingArguments, load_tokenizer


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

        generated_text = vigogne_chat_v3_template.build_inference_prompt(example, self.tokenizer)
        expected_text = """[INST] <<SYS>>
Vous êtes Vigogne, l'assistant IA créé par Zaion Lab. Vous suivez extrêmement bien les instructions. Aidez autant que vous le pouvez.
<</SYS>>

Donne trois conseils pour rester en bonne santé. [/INST]"""
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

        generated_text = vigogne_chat_v3_template.build_training_prompt(example, self.tokenizer)
        expected_text = """[INST] Donne trois conseils pour rester en bonne santé. [/INST] 1. Mangez une alimentation équilibrée et assurez-vous d'inclure beaucoup de fruits et légumes.
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

        example_length = vigogne_chat_v3_processor.process_example(
            example, self.tokenizer, length_column_name="example_length"
        )["example_length"]
        expected_example_length = 147
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
                {"role": "user", "content": "Bonjour."},
                {"role": "assistant", "content": "Bonjour, tu vas bien ?"},
                {"role": "user", "content": "Non, ça ne va pas."},
                {"role": "assistant", "content": "Qu'est-ce qui s'est passé ?"},
            ]
        }

        generated_text = vigogne_chat_v3_template.build_inference_prompt(example, self.tokenizer)
        expected_text = """[INST] <<SYS>>
Vous êtes Vigogne, l'assistant IA créé par Zaion Lab. Vous suivez extrêmement bien les instructions. Aidez autant que vous le pouvez.
<</SYS>>

Bonjour. [/INST] Bonjour, tu vas bien ?</s> [INST] Non, ça ne va pas. [/INST]"""
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

        generated_text = vigogne_chat_v3_template.build_training_prompt(example, self.tokenizer)
        expected_text = """[INST] <<SYS>>
Vous êtes un assistant IA qui suit extrêmement bien les instructions. Aidez autant que vous le pouvez.
<</SYS>>

Bonjour. [/INST] Bonjour, tu vas bien ?</s> [INST] Non, ça ne va pas. [/INST] Qu'est-ce qui s'est passé ?</s>"""
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
    #             {"role": "user", "content": "Bonjour."},
    #             {"role": "assistant", "content": "Bonjour, tu vas bien ?"},
    #             {"role": "user", "content": "Non, ça ne va pas."},
    #             {"role": "assistant", "content": "Qu'est-ce qui s'est passé ?"},
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
            },
            {
                "system": "",
                "messages": [
                    {"role": "user", "content": "Bonjour."},
                    {"role": "assistant", "content": "Bonjour, tu vas bien ?"},
                    {"role": "user", "content": "Non, ça ne va pas."},
                    {"role": "assistant", "content": "Qu'est-ce qui s'est passé ?"},
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
            },
            {
                "system": "",
                "messages": [
                    {"role": "user", "content": "Bonjour."},
                    {"role": "assistant", "content": "Bonjour, tu vas bien ?"},
                    {"role": "user", "content": "Non, ça ne va pas."},
                    {"role": "assistant", "content": "Qu'est-ce qui s'est passé ?"},
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
