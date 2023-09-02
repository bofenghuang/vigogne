import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList
import gradio as gr

model_path = "../models/vigogne2-enno-13b-sft-lora-4bit"

if os.getenv('MODEL_PATH'):
    model_path = os.getenv('MODEL_PATH')
else:
    model_path = "../models/vigogne2-enno-13b-sft-lora-4bit"


print('loading tokenizer')
tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
print('loading model')
model = LlamaForCausalLM.from_pretrained(model_path,
                                         device_map="auto",
                                         torch_dtype=torch.float16,
                                         load_in_8bit=True)

model = torch.compile(model)
print('done')


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords: list, tokenizer):
        self.keywords_ids = [tokenizer.encode(w)[0] for w in keywords]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords_ids:
            return True
        return False


# stop_words_criteria = StopWordsCriteria(stop_words=stop_words, tokenizer=tokenizer)
stop_criteria = KeywordsStoppingCriteria(["#", tokenizer.eos_token], tokenizer=tokenizer)


def transformers_generate(prompt: str):
    full_prompt = prompt.strip() + '\n'
    print('^^'+full_prompt+'^^')
    inputs = tokenizer(full_prompt, return_tensors='pt').to('cuda')
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(inputs.input_ids, do_sample=True, temperature=0.5, top_p=50, max_new_tokens=158,
                             return_dict_in_generate=True, output_scores=True,
                             stopping_criteria=StoppingCriteriaList([stop_criteria]))

    generated = outputs.sequences[0, input_length:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    gen_sequences = outputs.sequences[:, inputs.input_ids.shape[-1]:]
    probs = torch.stack(outputs.scores, dim=1).softmax(-1)
    gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
    probs_mean = torch.mean(gen_probs).tolist()
    print(probs_mean)
    # 0.8795413374900818 -> Hallucination : Afficher : Je ne suis pas sûre de ma réponse
    # 0.8764693737030029

    return text


demo = gr.Interface(fn=transformers_generate, inputs='text', outputs='text', title='EnnoData Chat Démo')

demo.launch(show_api=True, inline=True)
