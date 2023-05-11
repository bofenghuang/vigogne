#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

export WANDB_PROJECT=llm-sft-instruct-fr
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES=0

# Might need to adjust the batch size and other hyperparameters by yourself
python vigogne/train/train_sft.py \
    --model_name_or_path "bigscience/bloom-7b1" \
    --train_file "data/instruct/alpaca_data_cleaned_fr_52k_train.jsonl" \
    --eval_file "data/instruct/alpaca_data_cleaned_fr_52k_test.jsonl" \
    --output_dir "outputs/bloom-7b1-ft-instruct-llmint8" \
    --run_name "bloom-7b1-ft-instruct-llmint8" \
    --overwrite_output_dir \
    --mode "instruct" \
    --model_max_length "512" \
    --preprocessing_num_workers "4" \
    --dataloader_num_workers "1" \
    --load_in_8bit \
    --lora_r "16" \
    --lora_alpha "32" \
    --lora_dropout "0.05" \
    --target_modules "query_key_value" "dense" "dense_h_to_4h" "dense_4h_to_h" \
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
