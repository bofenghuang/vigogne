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
export CUDA_VISIBLE_DEVICES="4,5,6,7"

# tp
# tp_size=8
tp_size=4

# model_path="/projects/bhuang/models/llm/pretrained/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"
# model_path="/projects/bhuang/models/llm/pretrained/meta-llama/Meta-Llama-3.1-70B-Instruct"
model_path="/projects/bhuang/models/llm/pretrained/mistralai/Mistral-Large-Instruct-2407"

# input_file="/projects/bhuang/corpus/text/llm/collected/wild_chat_1m/wild_chat_1m_french_nontoxic_mininstlen8_maxinstlen512_deduped_uncensored_promptevaluatedmixtral8x7b_processed_minscore4_clustered.jsonl"
# input_file="/projects/bhuang/corpus/text/llm/vigogne-alignment-data/wild_chat_1m/wild_chat_1m_french_graded_filtered_clustered.jsonl"

# output_file="/projects/bhuang/corpus/text/llm/vigogne-alignment-data/wild_chat_1m/wild_chat_1m_french_graded_filtered_clustered_promptgenerated_mistrallargeinstruct2407.jsonl"
# output_file="/projects/bhuang/corpus/text/llm/vigogne-alignment-data/wild_chat_1m/wild_chat_1m_french_graded_filtered_clustered_promptgenerated_outliers_mistrallargeinstruct2407.jsonl"

# output_file="/projects/bhuang/corpus/text/llm/vigogne-alignment-data/wild_chat_1m/wild_chat_1m_french_graded_filtered_clustered_promptgenerated_llama31405b.jsonl"
# output_file="/projects/bhuang/corpus/text/llm/vigogne-alignment-data/wild_chat_1m/wild_chat_1m_french_graded_filtered_clustered_promptgenerated_outliers_llama31405b.jsonl"

# gen instruction
# python scripts/data_generation/generate_instruct_vllm.py \
#     --prompt_file data/generation/gen_instruct/general.txt \
#     --input_file $input_file \
#     --output_file $output_file \
#     --output_column_name instruction \
#     --model_configs_file data/configs/model_configs.json \
#     --model_name_or_path $model_path \
#     --dtype bfloat16 \
#     --tensor_parallel_size $tp_size \
#     --gpu_memory_utilization 0.95 \
#     --batch_size 1024 \
#     --max_tokens 4096 \
#     --max_model_len 4096 \
#     --temperature 0.9 \
#     --top_p 1.0

input_file="/projects/bhuang/corpus/text/llm/vigogne-alignment-data/wild_chat_1m/wild_chat_1m_french_graded_filtered_clustered_promptgenerated_filtered_graded_filtered_part1.jsonl"
output_file="/projects/bhuang/corpus/text/llm/vigogne-alignment-data/wild_chat_1m/wild_chat_1m_french_graded_filtered_clustered_promptgenerated_filtered_graded_filtered_part1_evolved.jsonl"

    # --prompt_file data/generation/evolve_instruct.txt \

# evolve instruction
python scripts/data_generation/generate_instruct_vllm.py \
    --prompt_file data/generation/evolve_instruct_c.txt \
    --input_file $input_file \
    --output_file $output_file \
    --id_column_name instruction \
    --instruct_column_name instruction \
    --output_column_name evolved_instruction \
    --model_configs_file data/configs/model_configs.json \
    --model_name_or_path $model_path \
    --dtype bfloat16 \
    --tensor_parallel_size $tp_size \
    --gpu_memory_utilization 0.95 \
    --batch_size 1024 \
    --max_tokens 3072 \
    --max_model_len 3072 \
    --temperature 0.9 \
    --top_p 1.0

echo "END TIME: $(date)"
