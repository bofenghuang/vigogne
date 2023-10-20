#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

"""
An example of using vLLM for offline batched inference on a dataset.
See https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html#offline-batched-inference
"""

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


model_name_or_path = "bofenghuang/vigogne-2-7b-chat"

# load tokenizer for applying chat template
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# tensor_parallel_size to load with TP
llm = LLM(model=model_name_or_path)

sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)

conversations = [
    [
        {"role": "user", "content": "Parle-moi de toi-même."},
    ],
    [
        {"role": "user", "content": "Bonjour ! Comment ça va aujourd'hui ?"},
        {
            "role": "assistant",
            "content": (
                "Bonjour ! Je suis une IA, donc je n'ai pas de sentiments, mais je suis prêt à vous aider. Comment puis-je"
                " vous assister aujourd'hui ?"
            ),
        },
        {"role": "user", "content": "Quelle est la hauteur de la Tour Eiffel ?"},
    ],
]

prompts = [
    tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) for conversation in conversations
]

# infer
outputs = llm.generate(prompts, sampling_params)

# Print the outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print(f"Q:\n\n{prompt}\n\nA:\n\n{generated_text}\n\n")
