#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Modified from: https://huggingface.co/spaces/mosaicml/mpt-7b-chat/raw/main/app.py

Usage:
CUDA_VISIBLE_DEVICES=0
python vigogne/demo/demo_chat.py \
    --base_model_name_or_path huggyllama/llama-7b \
    --lora_model_name_or_path bofenghuang/vigogne-chat-7b
"""

# import datetime
import logging
import os
import re
from threading import Event, Thread
from typing import List, Optional


# from uuid import uuid4

import fire
import json
import gradio as gr

# import requests
import torch
import transformers
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from vigogne.constants import ASSISTANT, INFERENCE_SYSTEM_MESSAGE, USER
from vigogne.inference.inference_utils import StopWordsCriteria

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

logger.info(f"Model will be loaded on device `{device}`")


def convert_history_to_text(history: List[List[str]], tokenizer: transformers.PreTrainedTokenizer, max_length: int = 2048):
    history = [f"\n<|{USER}|>: {x[0]}\n<|{ASSISTANT}|>: {x[1]}" for x in history]

    history_text = ""
    for x in history[::-1]:
        if len(tokenizer(INFERENCE_SYSTEM_MESSAGE + x + history_text)["input_ids"]) <= max_length:
            history_text = x + history_text
        else:
            break

    return INFERENCE_SYSTEM_MESSAGE + history_text if history_text else None


# def log_conversation(conversation_id, history, messages, generate_kwargs):
#     logging_url = os.getenv("LOGGING_URL", None)
#     if logging_url is None:
#         return

#     timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

#     data = {
#         "conversation_id": conversation_id,
#         "timestamp": timestamp,
#         "history": history,
#         "messages": messages,
#         "generate_kwargs": generate_kwargs,
#     }

#     try:
#         requests.post(logging_url, json=data)
#     except requests.exceptions.RequestException as e:
#         print(f"Error logging conversation: {e}")


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


# def get_uuid():
#     return str(uuid4())


def main(
    base_model_name_or_path: str = "huggyllama/llama-7b",
    lora_model_name_or_path: str = "bofenghuang/vigogne-chat-7b",
    load_8bit: bool = False,
    server_name: Optional[str] = "0.0.0.0",
    server_port: Optional[str] = None,
    share: bool = True,
):

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, padding_side="right", use_fast=False)

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_model_name_or_path,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_model_name_or_path,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, device_map={"": device}, low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(
            model,
            lora_model_name_or_path,
            device_map={"": device},
        )

    if not load_8bit and device != "cpu":
        model.half()  # seems to fix bugs for some users.

    model.eval()

    stop_words = [f"<|{ASSISTANT}|>", f"<|{USER}|>"]
    stop_words_criteria = StopWordsCriteria(stop_words=stop_words, tokenizer=tokenizer)
    pattern_trailing_stop_words = re.compile(rf'(?:{"|".join([re.escape(stop_word) for stop_word in stop_words])})\W*$')

    def bot(history, max_new_tokens, temperature, top_p, top_k, repetition_penalty, conversation_id=None):
        logger.info(f"History: {json.dumps(history, indent=4, ensure_ascii=False)}")

        # Construct the input message string for the model by concatenating the current system message and conversation history
        messages = convert_history_to_text(history, tokenizer)
        assert messages is not None, "User input is too long!"

        # Tokenize the messages string
        input_ids = tokenizer(messages, return_tensors="pt")["input_ids"].to(device)
        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=input_ids,
            generation_config=GenerationConfig(
                temperature=temperature,
                do_sample=temperature > 0.0,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            ),
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([stop_words_criteria]),
        )

        # stream_complete = Event()

        def generate_and_signal_complete():
            model.generate(**generate_kwargs)
            # stream_complete.set()

        # def log_after_stream_complete():
        #     stream_complete.wait()
        #     log_conversation(
        #         conversation_id,
        #         history,
        #         messages,
        #         {
        #             "top_k": top_k,
        #             "top_p": top_p,
        #             "temperature": temperature,
        #             "repetition_penalty": repetition_penalty,
        #         },
        #     )

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        # t2 = Thread(target=log_after_stream_complete)
        # t2.start()

        # Initialize an empty string to store the generated text
        partial_text = ""
        for new_text in streamer:
            # NB
            new_text = pattern_trailing_stop_words.sub("", new_text)

            partial_text += new_text
            history[-1][1] = partial_text
            yield history

        logger.info(f"Response: {history[-1][1]}")

    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        # conversation_id = gr.State(get_uuid)
        gr.Markdown(
            """<h1><center>🦙 Vigogne Chat</center></h1>

            This demo is of [Vigogne-Chat-7B](https://huggingface.co/bofenghuang/vigogne-chat-7b). It's based on [LLaMA-7B](https://github.com/facebookresearch/llama) finetuned to conduct French 🇫🇷 dialogues between a user and an AI assistant.

            For more information, please visit the [Github repo](https://github.com/bofenghuang/vigogne) of the Vigogne project.
    """
        )
        chatbot = gr.Chatbot().style(height=500)
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(
                    label="Chat Message Box",
                    placeholder="Chat Message Box",
                    show_label=False,
                ).style(container=False)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")
        with gr.Row():
            with gr.Accordion("Advanced Options:", open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            max_new_tokens = gr.Slider(
                                label="Max New Tokens",
                                value=512,
                                minimum=0,
                                maximum=1024,
                                step=1,
                                interactive=True,
                                info="The Max number of new tokens to generate.",
                            )
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                value=0.1,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                interactive=True,
                                info="Higher values produce more diverse outputs.",
                            )
                    with gr.Column():
                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p (nucleus sampling)",
                                value=1.0,
                                minimum=0.0,
                                maximum=1,
                                step=0.01,
                                interactive=True,
                                info=(
                                    "Sample from the smallest possible set of tokens whose cumulative probability "
                                    "exceeds top_p. Set to 1 to disable and sample from all tokens."
                                ),
                            )
                    with gr.Column():
                        with gr.Row():
                            top_k = gr.Slider(
                                label="Top-k",
                                value=0,
                                minimum=0.0,
                                maximum=200,
                                step=1,
                                interactive=True,
                                info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                            )
                    with gr.Column():
                        with gr.Row():
                            repetition_penalty = gr.Slider(
                                label="Repetition Penalty",
                                value=1.1,
                                minimum=1.0,
                                maximum=2.0,
                                step=0.1,
                                interactive=True,
                                info="Penalize repetition — 1.0 to disable.",
                            )
        with gr.Row():
            gr.Markdown(
                "Disclaimer: Vigogne is still under development, and there are many limitations that have to be addressed. Please note that it is possible that the model generates harmful or biased content, incorrect information or generally unhelpful answers.",
                elem_classes=["disclaimer"],
            )
        with gr.Row():
            gr.Markdown(
                "Acknowledgements: This demo is built on top of [MPT-7B-Chat](https://huggingface.co/mosaicml/mpt-7b-chat). Thanks for their contribution!",
                elem_classes=["disclaimer"],
            )

        submit_event = msg.submit(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False,).then(
            fn=bot,
            inputs=[
                chatbot,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                # conversation_id,
            ],
            outputs=chatbot,
            queue=True,
        )
        submit_click_event = submit.click(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False,).then(
            fn=bot,
            inputs=[
                chatbot,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                # conversation_id,
            ],
            outputs=chatbot,
            queue=True,
        )
        stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[submit_event, submit_click_event],
            queue=False,
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue(max_size=128, concurrency_count=2)
    demo.launch(enable_queue=True, share=share, server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    fire.Fire(main)
