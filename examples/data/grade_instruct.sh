#!/usr/bin/env bash

# instruction grading

set -x

echo "START TIME: $(date)"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
# export TOKENIZERS_PARALLELISM="false"

# cuda
# export CUDA_VISIBLE_DEVICES="4,5,6,7"

# tp
tp_size=8
# tp_size=4

# model_path="/projects/bhuang/models/llm/pretrained/meta-llama/Meta-Llama-3.1-70B-Instruct"
model_path="/projects/bhuang/models/llm/pretrained/mistralai/Mistral-Large-Instruct-2407"

# input_file="/projects/bhuang/corpus/text/llm/generated/self_instruct/self_instruct_merged_v21_v22_processed_mininstlen8_promptevaluatedllama370b_minscore2.jsonl"
# instruction_field = "evolved_instruction"
# input_file="/projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_merged_v1_v2_processed.jsonl"
# input_file="/projects/bhuang/corpus/text/llm/generated/orca/1m_gpt4_augmented_fr_m2m1b2_normalized_responded_gpt3_5_processed4_filteredsystem_responded_gpt4turbo1106_processed.jsonl"
# instruction_field = "translated_instruction"
input_file=$1

output_file="${input_file%.*}_graded-${model_path##*/}.jsonl"

# grade instruction
python scripts/data_generation/generate_response_b.py \
    --prompt_file data/generation/grade_instruct/grade_prompt_c.txt \
    --input_file $input_file \
    --output_file $output_file \
    --id_column_name instruction \
    --instruct_column_name instruction \
    --output_column_name instruction_evaluation \
    --model_configs_file data/configs/model_configs.json \
    --model_name_or_path $model_path \
    --dtype bfloat16 \
    --tensor_parallel_size $tp_size \
    --gpu_memory_utilization 0.95 \
    --batch_size 1024 \
    --max_tokens 3072 \
    --max_model_len 3072 \
    --temperature 0 \
    --top_p 1.0

echo "END TIME: $(date)"
