#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Usage:
export CUDA_VISIBLE_DEVICES=0

python vigogne/inference/gradio/demo_instruct.py --base_model_name_or_path bofenghuang/vigogne-2-7b-instruct
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

examples = [
    "Répondez à la question suivante : Les pratiques artistiques transforment-elles le monde ?",
    "Expliquez la différence entre DoS et phishing en français.",
    "Écrivez une fonction qui prend une liste de chaînes de caractères et renvoie une liste sans doublons.",
]


def main(
    base_model_name_or_path: str = "bofenghuang/vigogne-2-7b-instruct",
    lora_model_name_or_path: Optional[str] = None,
    load_8bit: bool = False,
    server_name: Optional[str] = "0.0.0.0",
    server_port: Optional[str] = None,
    share: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=load_8bit,
            trust_remote_code=True,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map={"": device},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, device_map={"": device}, low_cpu_mem_usage=True
        )

    if lora_model_name_or_path is not None:
        model = PeftModel.from_pretrained(model, lora_model_name_or_path)

    if not load_8bit and device != "cpu":
        model.half()  # seems to fix bugs for some users.

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def instruct(
        instruction,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        # no_repeat_ngram_size=3,
        streaming=True,
        **kwargs,
    ):
        prompt = generate_instruct_prompt(instruction=instruction)
        logger.info(f"Prompt: {prompt}")

        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        input_length = input_ids.shape[1]

        generation_config = GenerationConfig(
            temperature=temperature,
            do_sample=temperature > 0.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            # no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs,
        )

        if streaming:
            # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
            # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
            streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                input_ids=input_ids,
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
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                # output_scores=True,
            )
            generated_tokens = generated_outputs.sequences[0, input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            logger.info(f"Response: {generated_text}")
            return generated_text

    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        # session_id = gr.State(lambda: str(uuid4()))
        gr.Markdown(
            """<h1><center>🦙 Vigogne Instruction-following</center></h1>

            This demo is of [Vigogne-Instruct](https://huggingface.co/models?search=bofenghuang+vigogne+instruct) models finetuned to follow the French 🇫🇷 instructions.

            For more information, please visit the [Github repo](https://github.com/bofenghuang/vigogne) of the Vigogne project.
    """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    instruction = gr.Textbox(
                        placeholder="Enter your instruction here",
                        label="Question/Instruction",
                        elem_id="q-input",
                    )
                with gr.Accordion("Advanced Options:", open=False):
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                max_new_tokens = gr.Slider(
                                    label="Max New Tokens",
                                    value=1024,
                                    minimum=0,
                                    maximum=2048,
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
                                    value=1.0,
                                    minimum=1.0,
                                    maximum=2.0,
                                    step=0.1,
                                    interactive=True,
                                    info="Penalize repetition — 1.0 to disable.",
                                )
        with gr.Row():
            submit = gr.Button("Submit")
        with gr.Row():
            with gr.Box():
                gr.Markdown("**Response**")
                instruct_output = gr.Markdown()

        with gr.Row():
            gr.Examples(
                examples=examples,
                inputs=[instruction],
                cache_examples=False,
                fn=instruct,
                outputs=instruct_output,
            )
        with gr.Row():
            gr.Markdown(
                "Disclaimer: Vigogne is still under development, and there are many limitations that have to be addressed. Please note that it is possible that the model generates harmful or biased content, incorrect information or generally unhelpful answers.",
                elem_classes=["disclaimer"],
            )
        with gr.Row():
            gr.Markdown(
                "Acknowledgements: This demo's UI follows the style of [MPT-7B-Instruct](https://huggingface.co/spaces/mosaicml/mpt-7b-instruct). Thanks for their contribution!",
                elem_classes=["disclaimer"],
            )

        submit.click(
            instruct,
            inputs=[instruction, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
            outputs=instruct_output,
        )
        instruction.submit(
            instruct,
            inputs=[instruction, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
            outputs=instruct_output,
        )

    demo.queue(max_size=128, concurrency_count=2)
    demo.launch(enable_queue=True, share=share, server_name=server_name, server_port=server_port)

    # gr.Interface(
    #     fn=instruct,
    #     inputs=[
    #         gr.inputs.Textbox(label="Instruction", default="Expliquez la différence entre DoS et phishing en français."),
    #         gr.Checkbox(label="Streaming mode?", value=True),
    #     ],
    #     outputs=[gr.Textbox(label="Output", interactive=False)],
    #     title="🦙 Vigogne Instruction-following",
    #     description="This demo is of [Vigogne-7B-Instruct](https://huggingface.co/bofenghuang/vigogne-7b-instruct). It's based on [LLaMA-7B](https://github.com/facebookresearch/llama) finetuned finetuned to follow the French 🇫🇷 instructions. For more information, please visit the [Github repo](https://github.com/bofenghuang/vigogne) of the Vigogne project.",
    # ).launch(enable_queue=True, share=share, server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    fire.Fire(main)
