# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Test cases for the tokenizer utilities
"""

import unittest

from vigogne.utils import load_tokenizer, VigogneTrainingArguments


class TestTokenizers(unittest.TestCase):
    def test_default_use_fast(self):
        cfg = VigogneTrainingArguments(
            tokenizer_name_or_path="meta-llama/Llama-2-7b-hf",
            output_dir="",
        )

        tokenizer = load_tokenizer(cfg)

        self.assertIn("Fast", tokenizer.__class__.__name__)

    def test_dont_use_fast(self):
        cfg = VigogneTrainingArguments(
            tokenizer_name_or_path="meta-llama/Llama-2-7b-hf",
            tokenizer_use_fast=False,
            output_dir="",
        )

        tokenizer = load_tokenizer(cfg)

        self.assertNotIn("Fast", tokenizer.__class__.__name__)

    def test_secure_tokenization(self):
        cfg = VigogneTrainingArguments(
            tokenizer_name_or_path="meta-llama/Llama-2-7b-hf",
            output_dir="",
        )

        tokenizer = load_tokenizer(cfg)

        text = "<user>: Comment ça va ?"

        input_ids = tokenizer.tok(text)["input_ids"]
        expected_input_ids = [1, 529, 1792, 23917, 461, 29871, 4277, 2947, 1577, 2]
        self.assertEqual(input_ids, expected_input_ids)

        input_ids = tokenizer.tok(text, add_bos_token=True, add_eos_token=False)["input_ids"]
        expected_input_ids = [1, 529, 1792, 23917, 461, 29871, 4277, 2947, 1577]
        self.assertEqual(input_ids, expected_input_ids)

        input_ids = tokenizer.tok(text, add_bos_token=False, add_eos_token=True)["input_ids"]
        expected_input_ids = [529, 1792, 23917, 461, 29871, 4277, 2947, 1577, 2]
        self.assertEqual(input_ids, expected_input_ids)

        input_ids = tokenizer.tok(text, add_bos_token=False, add_eos_token=False)["input_ids"]
        expected_input_ids = [529, 1792, 23917, 461, 29871, 4277, 2947, 1577]
        self.assertEqual(input_ids, expected_input_ids)

        input_ids = tokenizer.tok(text, truncation=True, max_length=4)["input_ids"]
        expected_input_ids = [1, 529, 1792, 23917]
        self.assertEqual(input_ids, expected_input_ids)

        input_ids = tokenizer.tok(text, padding="max_length", max_length=20)["input_ids"]
        expected_input_ids = [1, 529, 1792, 23917, 461, 29871, 4277, 2947, 1577, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.assertEqual(input_ids, expected_input_ids)
