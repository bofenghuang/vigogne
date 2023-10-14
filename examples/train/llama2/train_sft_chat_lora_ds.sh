#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# Train chat models using LoRA + DeepSpeed Stage 2

export WANDB_PROJECT="llm-sft-chat"
export OMP_NUM_THREADS="1"
export TOKENIZERS_PARALLELISM="false"
export BITSANDBYTES_NOWELCOME="1"
# export CUDA_VISIBLE_DEVICES="0"

# Model
model_name_or_path=meta-llama/Llama-2-7b-hf

# Dataset
# Customize dataset here
train_file=data/chat/oasst_20230412_fr_top1.jsonl
model_max_length=2048

# Outdir
run_name=llama-2-7b-sft-chat-lora-ds
output_dir=outputs/$run_name

# Might need to adjust the batch size and other hyperparameters by yourself
per_device_train_batch_size=8
gradient_accumulation_steps=8

torchrun \
    vigogne/train_sft.py \
    --deepspeed vigogne/configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $model_name_or_path \
    --tokenizer_use_fast false \
    --tokenizer_padding_side "right" \
    --train_file $train_file \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --run_name $run_name \
    --processor_style "vigogne_chat_v3" \
    --model_max_length $model_max_length \
    --preprocessing_num_workers "8" \
    --dataloader_num_workers "1" \
    --adapter "lora" \
    --lora_r "64" \
    --lora_alpha "16" \
    --lora_dropout "0.05" \
    --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj" "gate_proj" "up_proj" "down_proj" \
    --do_merge_lora \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs "3" \
    --learning_rate "1e-4" \
    --warmup_ratio "0.03" \
    --lr_scheduler_type "cosine" \
    --weight_decay "0" \
    --fp16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters false \
    --log_level "info" \
    --logging_steps "10" \
    --logging_first_step \
    --save_strategy "steps" \
    --save_steps "100" \
    --save_total_limit "3" \
    --report_to "tensorboard" "wandb"
