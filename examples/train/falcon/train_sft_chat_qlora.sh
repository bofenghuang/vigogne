#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# Train chat models using QLoRA (int4)

export WANDB_PROJECT="llm-sft-chat"
export OMP_NUM_THREADS="1"
export TOKENIZERS_PARALLELISM="false"
export BITSANDBYTES_NOWELCOME="1"
# export CUDA_VISIBLE_DEVICES="0"

# Model
model_name_or_path=tiiuae/falcon-7b
# model_name_or_path=tiiuae/falcon-40b

# Dataset
# Customize dataset here
train_file=data/chat/oasst_20230412_fr_top1.jsonl
model_max_length=2048

# Outdir
run_name=falcon-7b-sft-chat-qlora
output_dir=outputs/$run_name

# Might need to adjust the batch size and other hyperparameters by yourself
per_device_train_batch_size=8
per_device_eval_batch_size=4
gradient_accumulation_steps=8

# Further optimization
# DeepSpeed Stage 2
# --deepspeed vigogne/configs/ds_config_zero2_no_offload.json \

torchrun \
    vigogne/cli/train_sft.py \
    --model_name_or_path $model_name_or_path \
    --tokenizer_use_fast false \
    --tokenizer_padding_side "right" \
    --add_special_tokens '{"bos_token":">>ABSTRACT<<","pad_token":"<|endoftext|>"}' \
    --train_file $train_file \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --run_name $run_name \
    --processor_style "vigogne_chat_v3" \
    --model_max_length $model_max_length \
    --eval_split_ratio "0.01" \
    --preprocessing_num_workers "8" \
    --dataloader_num_workers "1" \
    --adapter "qlora" \
    --load_in_4bit \
    --optim "paged_adamw_32bit" \
    --lora_r "64" \
    --lora_alpha "16" \
    --lora_dropout "0.05" \
    --lora_target_all_linear_layers \
    --do_merge_lora \
    --num_train_epochs "3" \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
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
    --evaluation_strategy "steps" \
    --eval_steps "100" \
    --report_to "tensorboard" "wandb"
