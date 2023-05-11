#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

export WANDB_PROJECT=llm-sft-instruct-fr
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Might need to adjust the batch size and other hyperparameters by yourself
torchrun \
    --nproc_per_node 4 \
    --master_port 29001 \
    vigogne/train/train_sft.py \
    --deepspeed vigogne/configs/ds_zero2_config.json \
    --model_name_or_path "name/or/path/to/hf/llama/7b/model" \
    --train_file "data/instruct/alpaca_data_cleaned_fr_52k_train.jsonl" \
    --eval_file "data/instruct/alpaca_data_cleaned_fr_52k_test.jsonl" \
    --output_dir "outputs/llama-7b-ft-instruct-ds" \
    --run_name "llama-7b-ft-instruct-ds" \
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
