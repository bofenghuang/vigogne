from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import torch

print(Accelerator().process_index)

tokenizer = AutoTokenizer.from_pretrained("../models/vigogne2-enno-13b-sft-lora-4bit")
model = AutoModelForCausalLM.from_pretrained("../models/vigogne2-enno-13b-sft-lora-4bit",
                                             load_in_8bit=True,
                                             low_cpu_mem_usage=True,
                                             device_map={"": Accelerator().process_index},
                                             torch_dtype=torch.bfloat16
                                             )


def predict(input, history=[]):
    # tokenize the new input sentence
    new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([torch.LongTensor(history), new_user_input_ids], dim=-1)

    # generate a response
    history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id).tolist()

    # convert the tokens to text, and then split the responses into the right format
    response = tokenizer.decode(history[0]).split("</s>")
    response = [(response[i], response[i + 1]) for i in range(0, len(response) - 1, 2)]  # convert to tuples of list
    return response, history


import gradio as gr

gr.Interface(fn=predict,
             theme="default",
             css=".footer {display:none !important}",
             inputs=["text", "state"],
             outputs=["chatbot", "state"]).launch()
