<p align="center" width="100%">
<img src="assets/vigogne_logo.png" alt="Vigogne" style="width: 40%; min-width: 300px; display: block; margin: auto;">
</p>

# Vigogne: French Instruction-following Models

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/bofenghuang/vigogne/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/bofenghuang/vigogne/blob/main/DATA_LICENSE)

*The vigogne (French name for vicuña) is a South American camelid native to the Andes Mountains. It is closely related to the llama, alpaca, and guanaco.*

This repository contains code for reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) in French 🇫🇷 using [low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685) provided by 🤗 Hugging Face's [PEFT](https://github.com/huggingface/peft) library. In addition to the LoRA technique, we also use [LLM.int8()](https://arxiv.org/abs/2208.07339) provided by [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) to quantize pretrained language models (PLMs) to int8. Combining these two techniques allows us to fine-tune PLMs on a single consumer GPU such as RTX 4090.

This project is based on [LLaMA](https://github.com/facebookresearch/llama), [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [**Alpaca-Lora**](https://github.com/tloen/alpaca-lora), [Cabrita](https://github.com/22-hours/cabrita) and [Hugging Face](https://huggingface.co/docs/transformers/main_classes/trainer). In addition, we adapted the [training script](https://github.com/bofenghuang/vigogne/blob/main/finetune.py) to fine-tune on more models such as [BLOOM](https://huggingface.co/bigscience/bloom-7b1) and [mT5](https://huggingface.co/google/mt5-xxl). We also share the [translated dataset](https://github.com/bofenghuang/vigogne/blob/main/data/vigogne_data_cleaned.json) and the trained [vigogne-lora-7b](https://huggingface.co/bofenghuang/vigogne-lora-7b) and [vigogne-lora-bloom-7b1](https://huggingface.co/bofenghuang/vigogne-lora-bloom-7b1) weights.

**Usage and License Notices**: Same as [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), Vigogne is intended and licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

## Play with 🦙 vigogne models

The fine-tuned vigogne models are available on 🤗 Hugging Face:

- Fine-tuned LLaMA-7B model: [bofenghuang/vigogne-lora-7b](https://huggingface.co/bofenghuang/vigogne-lora-7b)
- Fine-tuned LLaMA-13B model: [bofenghuang/vigogne-lora-13b](https://huggingface.co/bofenghuang/vigogne-lora-13b)
- Fine-tuned LLaMA-30B model: [bofenghuang/vigogne-lora-30b](https://huggingface.co/bofenghuang/vigogne-lora-30b)
- Fine-tuned BLOOM-7B1 model: [bofenghuang/vigogne-lora-bloom-7b1](https://huggingface.co/bofenghuang/vigogne-lora-bloom-7b1)

You can infer the fine-tuned vigogne model model by using the following Google Colab Notebook.

<a href="https://colab.research.google.com/github/bofenghuang/vigogne/blob/main/infer.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

You can also run with a Gradio demo using the following command:

```bash
./demo.py \
    --base_model_name_or_path decapoda-research/llama-7b-hf \
    --lora_model_name_or_path bofenghuang/vigogne-lora-7b
```

## Data

We translated the original [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) to French using `gpt-3.5-turbo` by the chat completion API.

You can also translate it to other languages using the [translation script](https://github.com/bofenghuang/vigogne/blob/main/scripts/translate_data.py). Don't forget to modify your [translation prompt](https://github.com/bofenghuang/vigogne/blob/e6ae25fc0569ca85c25529a6d06122b35426aa2d/scripts/translate_data.py#L47-L57).

The translation may have compromised the accuracy of certain tasks, such as generating rhyming words or correcting grammar (discussed [here](https://github.com/tloen/alpaca-lora/pull/127)). We warmly welcome PRs to help clean up this dataset!

The following command shows how to estimate the price for translating the full dataset.

```bash
./scripts/translate_data.py estimate_price \
    --input_json_file data/alpaca_data_cleaned.json \
    --ratio_output_input 1.0 \
    --model gpt-3.5-turbo-0301 \
    --price_per_thousand_tokens 0.002
```

You can translate the dataset using the following command.

```bash
# Specify your OpenAI API key
export OPENAI_API_KEY=xx

./scripts/translate_data.py process_data \
    --input_json_file data/alpaca_data_cleaned.json \
    --output_json_file data/vigogne_data_cleaned.json \
    --model gpt-3.5-turbo \
    --max_parallel_requests 32
```

## Fine-tuning

### Setup

Install dependencies

```bash
pip install -r requirements.txt
```

### Fine-tuning LLaMA-7B model

The following command shows how to fine-tune [LLaMA-7B](https://huggingface.co/decapoda-research/llama-7b-hf) model using a single GPU.

```bash
python finetune.py \
    --model_name_or_path "decapoda-research/llama-7b-hf" \
    --data_path "data/vigogne_data_cleaned.json" \
    --val_set_size 2000 \
    --model_max_length 368 \
    --output_dir "outputs/llama-7b-ft-vigogne-lora" \
    --run_name "llama-7b-ft-vigogne-lora" \
    --overwrite_output_dir \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --target_modules '["q_proj", "v_proj"]' \
    --num_train_epochs 3 \
    --optim "adamw_torch" \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --dataloader_num_workers="1" \
    --logging_steps 25 \
    --save_total_limit 3 \
    --save_strategy "steps" \
    --save_steps 200 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --report_to='["tensorboard", "wandb"]'
```

### Fine-tuning LLaMA-30B model

The following command shows how to fine-tune [LLaMA-30B](https://huggingface.co/decapoda-research/llama-30b-hf) model using multi GPUs.

```bash
WORLD_SIZE=2 torchrun --nproc_per_node=2 --master_port=29001 finetune.py \
    --model_name_or_path "decapoda-research/llama-30b-hf" \
    --data_path "data/vigogne_data_cleaned.json" \
    --val_set_size 2000 \
    --model_max_length 368 \
    --output_dir "outputs/llama-30b-ft-vigogne-lora" \
    --run_name "llama-30b-ft-vigogne-lora" \
    --overwrite_output_dir \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --target_modules '["q_proj", "v_proj"]' \
    --num_train_epochs 3 \
    --optim "adamw_torch" \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --dataloader_num_workers="1" \
    --logging_steps 25 \
    --save_total_limit 3 \
    --save_strategy "steps" \
    --save_steps 200 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --report_to='["tensorboard", "wandb"]'
```

### Fine-tuning BLOOM-7B1 model

The following command shows how to fine-tune [bigscience/bloom-7b1](https://huggingface.co/bigscience/bloom-7b1) model using a single GPU.

```bash
python finetune.py \
    --model_name_or_path "bigscience/bloom-7b1" \
    --data_path "data/vigogne_data_cleaned.json" \
    --val_set_size 2000 \
    --model_max_length 256 \
    --output_dir "outputs/bloom-7b1-ft-vigogne" \
    --run_name "bloom-7b1-ft-vigogne" \
    --overwrite_output_dir \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules '["query_key_value"]' \
    --num_train_epochs 3 \
    --optim "adamw_torch" \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --dataloader_num_workers="1" \
    --logging_steps 25 \
    --save_total_limit 3 \
    --save_strategy "steps" \
    --save_steps 200 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --report_to='["tensorboard", "wandb"]'
```

## Limitations

Vigogne is still under development, and there are many limitations that have to be addressed. Please note that it is possible that the model generates harmful or biased content, incorrect information or generally unhelpful answers.

## Next Steps

- Collect more and cleaner French instruction-following data
