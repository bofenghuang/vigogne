#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

export WANDB_PROJECT=llm-sft-instruct-fr
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES=0

# --model_name_or_path "google/mt5-xxl" \

# Might need to adjust the batch size and other hyperparameters by yourself
python vigogne/train/train_sft_seq2seq.py \
    --model_name_or_path "google/flan-t5-xxl" \
    --train_file "data/instruct/alpaca_data_cleaned_fr_52k_train.jsonl" \
    --eval_file "data/instruct/alpaca_data_cleaned_fr_52k_test.jsonl" \
    --output_dir "outputs/flan-t5-ft-instruct-llmint8" \
    --run_name "flan-t5-ft-instruct-llmint8" \
    --overwrite_output_dir \
    --model_max_source_length_percentile "95" \
    --model_max_target_length_percentile "95" \
    --preprocessing_num_workers "4" \
    --dataloader_num_workers "1" \
    --load_in_8bit \
    --lora_r "16" \
    --lora_alpha "32" \
    --lora_dropout "0.05" \
    --target_modules "q" "v" \
    --per_device_train_batch_size "8" \
    --per_device_eval_batch_size "4" \
    --gradient_accumulation_steps "16" \
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
