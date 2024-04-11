#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import os
from typing import Any, Dict, List, Optional

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

client = None


# def set_global_api(api_name: str = "openai"):
def set_global_api(model_name: str = "gpt"):
    # if api_name == "openai":
    if "gpt" in model_name:
        global client
        # check https://github.com/openai/openai-python
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # this is also the default, it can be omitted

        return generate_api_messages_openai, chat_completion_openai, process_api_response_openai
    # elif api_name == "mistral":
    elif "mistral" in model_name:
        return generate_api_messages_mistral, chat_completion_mistral, process_api_response_mistral
    else:
        raise ValueError(f"Invalid model name: {model_name}")

def generate_api_messages_openai(prompt: str, system_message: str = "You are a helpful assistant."):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]


# Add exponential backoff to mitigate openai.error.RateLimitError
# See: https://platform.openai.com/docs/guides/rate-limits/error-mitigation
# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_completion_openai(
    messages: List[Dict],
    model: str = "gpt-4",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    # api_dict=None,
):
    # print(locals())
    # quit()
    # if api_dict is not None:
    #     openai.api_base = api_dict["api_base"]
    #     openai.api_key = api_dict["api_key"]

    return client.chat.completions.create(
        model=model,
        messages=messages,
        n=1,
        max_tokens=max_tokens,
        temperature=temperature,
        # logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
    )


def process_api_response_openai(response: Any, response_field: str = "output"):
    # todo: response.model_dump()
    parsed_response = {
        response_field: response.choices[0].message.content if hasattr(response, "choices") else None,
        "created": getattr(response, "created", None),
        "model": getattr(response, "model", None),
        "finish_reason": response.choices[0].finish_reason if hasattr(response, "choices") else None,
        "prompt_tokens": getattr(getattr(response, "usage", {}), "prompt_tokens", None),
        "completion_tokens": getattr(getattr(response, "usage", {}), "completion_tokens", None),
        "total_tokens": getattr(getattr(response, "usage", {}), "total_tokens", None),
    }

    if parsed_response["finish_reason"] == "length":
        print("max_tokens reached")

    return parsed_response


def generate_api_messages_mistral(prompt: str):
    return [ChatMessage(role="user", content=prompt)]


# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_completion_mistral(
    messages: List[Dict],
    model: str = "mistral-medium",
    max_tokens: int = 1024,
    temperature: float = 0.7,
):
    c = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    return c.chat(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def process_api_response_mistral(response: Any, response_field: str = "output"):
    parsed_response = {
        response_field: response.choices[0].message.content if hasattr(response, "choices") else None,
        "created": getattr(response, "created", None),
        "model": getattr(response, "model", None),
        "finish_reason": response.choices[0].finish_reason.value if hasattr(response, "choices") else None,
        "prompt_tokens": response.usage.prompt_tokens if hasattr(response, "usage") else None,
        "completion_tokens": response.usage.completion_tokens if hasattr(response, "usage") else None,
        "total_tokens": response.usage.total_tokens if hasattr(response, "usage") else None,
    }

    if parsed_response["finish_reason"] == "length":
        print("max_tokens reached")

    return parsed_response
