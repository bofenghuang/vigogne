#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Usage:
CUDA_VISIBLE_DEVICES=0

python vigogne/demo/demo_instruct.py \
    --base_model_name_or_path name/or/path/to/hf/llama/7b/model \
    --lora_model_name_or_path bofenghuang/vigogne-instruct-7b
"""

import logging
import sys
from threading import Thread
from typing import Optional

import fire
import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextIteratorStreamer

from vigogne.preprocess import generate_instruct_prompt

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


def main(
    base_model_name_or_path: str = "huggyllama/llama-7b",
    lora_model_name_or_path: str = "bofenghuang/vigogne-instruct-7b",
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

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def instruct(
        instruction,
        streaming=True,
        temperature=0.1,
        do_sample=True,
        no_repeat_ngram_size=3,
        max_new_tokens=512,
        **kwargs,
    ):
        prompt = generate_instruct_prompt(instruction=instruction)
        logger.info(f"Prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]

        generation_config = GenerationConfig(
            temperature=temperature,
            do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        if streaming:
            # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
            # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
            streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                generation_config=generation_config,
                # return_dict_in_generate=True,
                # output_scores=True,
            )
            t = Thread(target=model.generate, kwargs=generation_kwargs)
            t.start()

            # Pull the generated text from the streamer, and update the model output.
            output_text = ""
            for new_text in streamer:
                output_text += new_text
                yield output_text
            logger.info(f"Response: {output_text}")
            return output_text

        else:
            generated_outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                # output_scores=True,
            )
            generated_tokens = generated_outputs.sequences[0, input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            logger.info(f"Response: {generated_text}")
            return generated_text

    gr.Interface(
        fn=instruct,
        inputs=[
            gr.inputs.Textbox(label="Instruction", default="Expliquez la différence entre DoS et phishing en français."),
            gr.Checkbox(label="Streaming mode?", value=True),
        ],
        outputs=[gr.Textbox(label="Output", interactive=False)],
        title="🦙 Vigogne Instruction-following",
        description="This demo is of [Vigogne-Instruct-7B](https://huggingface.co/bofenghuang/vigogne-instruct-7b). It's based on [LLaMA-7B](https://github.com/facebookresearch/llama) finetuned finetuned to follow the French 🇫🇷 instructions. For more information, please visit the [Github repo](https://github.com/bofenghuang/vigogne) of the Vigogne project.",
    ).launch(enable_queue=True, share=share, server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    fire.Fire(main)
