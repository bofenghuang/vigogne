#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

export WANDB_PROJECT="llm-sft-chat-fr"
export OMP_NUM_THREADS="1"
export TOKENIZERS_PARALLELISM="false"
export BITSANDBYTES_NOWELCOME="1"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

train_file=/path/to/train/chat/file.jsonl

mode=chat
model_max_length=2048

model_name_or_path=meta-llama/Llama-2-7b-hf
output_dir=outputs/llama-2-7b-sft-chat-lora-int8

per_device_train_batch_size=8
gradient_accumulation_steps=4

# Might need to adjust the batch size and other hyperparameters by yourself
torchrun \
    --nproc_per_node 4 \
    vigogne/train/train_sft.py \
    --model_name_or_path $model_name_or_path \
    --train_file $train_file \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --mode $mode \
    --model_max_length $model_max_length \
    --preprocessing_num_workers "8" \
    --dataloader_num_workers "1" \
    --pack_into_block \
    --block_size "2048" \
    --load_in_8bit \
    --lora_r "64" \
    --lora_alpha "16" \
    --lora_dropout "0.05" \
    --target_modules "q_proj" "v_proj" "k_proj" "o_proj" "gate_proj" "up_proj" "down_proj" \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs "3" \
    --learning_rate "1e-4" \
    --warmup_ratio "0.03" \
    --lr_scheduler_type "cosine" \
    --weight_decay "0" \
    --torch_compile \
    --fp16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters false \
    --log_level "info" \
    --logging_steps "10" \
    --logging_first_step true \
    --save_strategy "steps" \
    --save_steps "100" \
    --save_total_limit "3" \
    --report_to "tensorboard" "wandb" \
    --do_train
