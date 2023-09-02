#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/examples/openai_chatcompletion_client.py


import openai

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

# List models API
models = openai.Model.list()
print("Models:", models)

# first model
model = models["data"][0]["id"]

# Chat completion API
chat_completion = openai.ChatCompletion.create(
    model=model,
    messages=[
        # {"role": "system", "content": DEFAULT_CHAT_SYSTEM_MESSAGE_GEN},
        {"role": "user", "content": "Parle-moi de toi-même."},
    ],
    max_tokens=1024,
    temperature=0.7,
    # stream=True,
)

print("Chat completion results:")
print(chat_completion)
# for chunk in chat_completion:
#     new_text = chunk["choices"][0]["delta"].get("content")
#     if new_text is not None:
#         print(new_text, end="")
