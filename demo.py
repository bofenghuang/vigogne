#! /usr/bin/env python
# coding=utf-8

"""
Modified from: https://github.com/tloen/alpaca-lora/blob/main/generate.py
"""

import logging
import sys
from threading import Thread

import fire
import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaTokenizer, TextIteratorStreamer

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

logger.info(f"Model will be loaded on device `{device}`")


PROMPT_DICT = {
    "prompt_input": (
        "Ci-dessous se trouve une instruction qui décrit une tâche, associée à une entrée qui fournit un contexte supplémentaire. Écrivez une réponse qui complète correctement la demande.\n\n"
        "### Instruction:\n{instruction}\n\n### Entrée:\n{input}\n\n### Réponse:\n"
    ),
    "prompt_no_input": (
        "Ci-dessous se trouve une instruction qui décrit une tâche. Écrivez une réponse qui complète correctement la demande.\n\n"
        "### Instruction:\n{instruction}\n\n### Réponse:\n"
    ),
}


def generate_prompt(instruction, input=None):
    return (
        PROMPT_DICT["prompt_input"].format_map({"instruction": instruction, "input": input})
        if input is not None
        else PROMPT_DICT["prompt_no_input"].format_map({"instruction": instruction})
    )


def main(
    base_model_name_or_path: str,
    lora_model_name_or_path: str = "bofenghuang/vigogne-lora-7b",
    load_8bit: bool = False,
):

    tokenizer_class = LlamaTokenizer if "llama" in base_model_name_or_path else AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(base_model_name_or_path)

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

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def instruct(
        instruction,
        input=None,
        streaming=True,
        temperature=0.1,
        no_repeat_ngram_size=3,
        max_new_tokens=512,
        **kwargs,
    ):
        prompt = generate_prompt(instruction, input)
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        generation_config = GenerationConfig(
            temperature=temperature,
            no_repeat_ngram_size=no_repeat_ngram_size,
            **kwargs,
        )

        if streaming:
            # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
            # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
            streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
            generate_kwargs = dict(
                input_ids=input_ids,
                streamer=streamer,
                generation_config=generation_config,
                # return_dict_in_generate=True,
                # output_scores=True,
                max_new_tokens=max_new_tokens,
            )
            t = Thread(target=model.generate, kwargs=generate_kwargs)
            t.start()

            # Pull the generated text from the streamer, and update the model output.
            output_text = ""
            for new_text in streamer:
                output_text += new_text
                yield output_text
            logger.info(output_text)
            return output_text

        else:
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                # return_dict_in_generate=True,
                # output_scores=True,
                max_new_tokens=max_new_tokens,
            )

            output_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
            logger.info(output_text)
            output_text = output_text.rsplit("### Réponse:", 1)[-1].strip()
            return output_text

    gr.Interface(
        fn=instruct,
        inputs=[
            gr.inputs.Textbox(label="Instruction", default="Parlez-moi des vigognes."),
            gr.inputs.Textbox(label="Input"),
            gr.Checkbox(label="Streaming mode?", value=True),
        ],
        outputs=[gr.Textbox(label="Output", interactive=False)],
        title="🦙 Vigogne-LoRA",
        description="Vigogne-LoRA is a 7B-parameter LLaMA model finetuned to follow the French 🇫🇷 instructions. For more information, please visit the [Github repo](https://github.com/bofenghuang/vigogne).",
    ).launch(enable_queue=True, share=True)


if __name__ == "__main__":
    fire.Fire(main)
