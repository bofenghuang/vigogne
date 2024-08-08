#!/usr/bin/env bash

# Magpie instruction extraction & grading

set -x

echo "START TIME: $(date)"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
# export TOKENIZERS_PARALLELISM="false"

# cuda
export CUDA_VISIBLE_DEVICES="4,5,6,7"

# tp_size=8
tp_size=4

# input_file="/projects/bhuang/corpus/text/llm/generated/magpie/magpie-inst_l31_405b_fp8.jsonl"
# input_file="/projects/bhuang/corpus/text/llm/generated/magpie/magpie-inst_l31_70b_500k.json"
# input_file="/projects/bhuang/corpus/text/llm/generated/magpie/magpie_inst-Meta-Llama-3.1-405B-Instruct-FP8-1000000-240803.jsonl"
input_file="/projects/bhuang/corpus/text/llm/generated/magpie/magpie_inst-Meta-Llama-3.1-405B-Instruct-FP8-1.6m.jsonl"

output_file="${input_file%.*}_extracted-l31-70b.jsonl"

    # --max_samples 20

python scripts/data_generation/generate_response_b.py \
    --prompt_file data/generation/extract_instruct.txt \
    --input_file $input_file \
    --output_file $output_file \
    --id_column_name raw_instruction \
    --instruct_column_name raw_instruction \
    --output_column_name instruction \
    --model_configs_file data/configs/model_configs.json \
    --model_name_or_path /projects/bhuang/models/llm/pretrained/meta-llama/Meta-Llama-3.1-70B-Instruct \
    --dtype bfloat16 \
    --tensor_parallel_size $tp_size \
    --gpu_memory_utilization 0.95 \
    --batch_size 1024 \
    --max_tokens 2048 \
    --max_model_len 2048 \
    --temperature 0 \
    --top_p 1.0

input_file="$output_file"
output_file="${input_file%.*}_graded-l31-70b.jsonl"

python scripts/data_generation/generate_response_b.py \
    --prompt_file data/generation/grade_instruct/grade_prompt_c.txt \
    --input_file $input_file \
    --output_file $output_file \
    --id_column_name instruction \
    --instruct_column_name instruction \
    --output_column_name instruction_evaluation \
    --model_configs_file data/configs/model_configs.json \
    --model_name_or_path /projects/bhuang/models/llm/pretrained/meta-llama/Meta-Llama-3.1-70B-Instruct \
    --dtype bfloat16 \
    --tensor_parallel_size $tp_size \
    --gpu_memory_utilization 0.95 \
    --batch_size 1024 \
    --max_tokens 2048 \
    --max_model_len 2048 \
    --temperature 0 \
    --top_p 1.0

echo "END TIME: $(date)"
