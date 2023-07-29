# Training

## Supervised Fine-tuning

To fine-tune LLMs more efficiently, we employ a technique called [low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685) provided by 🤗 Hugging Face's [PEFT](https://github.com/huggingface/peft) library, which involves freezing the base model's weights and adding a small number of learnable parameters.

In addition, you can further reduce the memory usage during fine-tuning by using [LLM.int8()](https://arxiv.org/abs/2208.07339), which employs a 2-stage quantization method that quantizes part of the computation to int8. This enables efficient training on a single consumer GPU such as the RTX 4090. However, it may be slightly slower than the fp16 version. If your GPUs have enough memory, you can skip this step and train using [DeepSpeed](https://github.com/microsoft/DeepSpeed).

More examples can be found in [examples](https://github.com/bofenghuang/vigogne/blob/main/examples/train).

The following command shows how to fine-tune the LLaMA-7B model on a single GPU using LLM.int8().

```bash
python vigogne/train/train_sft.py \
    --model_name_or_path "name/or/path/to/hf/llama/7b/model" \
    --train_file "data/instruct/alpaca_data_cleaned_fr_52k_validated_train.jsonl" \
    --eval_file "data/instruct/alpaca_data_cleaned_fr_52k_validated_test.jsonl" \
    --output_dir "outputs/llama-7b-ft-instruct-llmint8" \
    --run_name "llama-7b-ft-instruct-llmint8" \
    --overwrite_output_dir \
    --mode "instruct" \
    --model_max_length "512" \
    --preprocessing_num_workers "4" \
    --dataloader_num_workers "1" \
    --load_in_8bit \
    --lora_r "8" \
    --lora_alpha "16" \
    --lora_dropout "0.05" \
    --target_modules "q_proj" "v_proj" "k_proj" "o_proj" "gate_proj" "down_proj" "up_proj" \
    --per_device_train_batch_size "16" \
    --per_device_eval_batch_size "8" \
    --gradient_accumulation_steps "8" \
    --num_train_epochs "3" \
    --learning_rate "3e-4" \
    --warmup_ratio "0.05" \
    --weight_decay "0.01" \
    --gradient_checkpointing \
    --logging_steps "10" \
    --logging_first_step true \
    --save_strategy "steps" \
    --save_steps "100" \
    --save_total_limit "3" \
    --evaluation_strategy "steps" \
    --eval_steps "100" \
    --load_best_model_at_end \
    --report_to "tensorboard" "wandb" \
    --do_train \
    --do_eval
```

The following command shows how to fine-tune the LLaMA-13B model on 4 GPUs using DeepSpeed ZeRO stage 2.

```bash
torchrun \
    --nproc_per_node 4 \
    --master_port 29001 \
    vigogne/train/train_sft.py \
    --deepspeed vigogne/configs/ds_zero2_config.json \
    --model_name_or_path "name/or/path/to/hf/llama/13b/model" \
    --train_file "data/instruct/alpaca_data_cleaned_fr_52k_validated_train.jsonl" \
    --eval_file "data/instruct/alpaca_data_cleaned_fr_52k_validated_test.jsonl" \
    --output_dir "outputs/llama-13b-ft-instruct-ds" \
    --run_name "llama-13b-ft-instruct-ds" \
    --overwrite_output_dir \
    --mode "instruct" \
    --model_max_length "512" \
    --preprocessing_num_workers "4" \
    --dataloader_num_workers "1" \
    --lora_r "8" \
    --lora_alpha "16" \
    --lora_dropout "0.05" \
    --target_modules "q_proj" "v_proj" "k_proj" "o_proj" "gate_proj" "down_proj" "up_proj" \
    --per_device_train_batch_size "16" \
    --per_device_eval_batch_size "8" \
    --gradient_accumulation_steps "2" \
    --num_train_epochs "3" \
    --learning_rate "3e-4" \
    --warmup_ratio "0.05" \
    --weight_decay "0.01" \
    --gradient_checkpointing \
    --ddp_find_unused_parameters false \
    --logging_steps "10" \
    --logging_first_step true \
    --save_strategy "steps" \
    --save_steps "100" \
    --save_total_limit "3" \
    --evaluation_strategy "steps" \
    --eval_steps "100" \
    --load_best_model_at_end \
    --report_to "tensorboard" "wandb" \
    --do_train \
    --do_eval
```
