#!/bin/bash

# gen inst
####################################################################################################

input_root=/lustre/fswork/projects/rech/gkb/commun/corpus/text/llm_instruct/magpie-fr

stage=2

# input_file=${input_root}/magpie_inst-Llama-3.1-Nemotron-70B-Instruct-HF-241126.jsonl
# examples/data/magpie/magpie_process_slurm.sh $stage $input_file

# input_file=${input_root}/magpie_inst-Meta-Llama-3.1-70B-Instruct-241126.jsonl
# examples/data/magpie/magpie_process_slurm.sh $stage $input_file

# input_file=${input_root}/magpie_inst-Mistral-Large-Instruct-2411-241126.jsonl
# examples/data/magpie/magpie_process_slurm.sh $stage $input_file

# input_file=${input_root}/magpie_inst-Qwen2.5-72B-Instruct-241126.jsonl
# examples/data/magpie/magpie_process_slurm.sh $stage $input_file

input_file=${input_root}/Llama-3.1-405B-Instruct-FP8/magpie_inst-Llama-3.1-405B-Instruct-FP8-241126.jsonl
examples/data/magpie/magpie_process_slurm.sh $stage $input_file

# gen resp
####################################################################################################

# input_root=/lustre/fswork/projects/rech/gkb/commun/corpus/text/llm_instruct/magpie-fr
# model_root=/lustre/fswork/projects/rech/gkb/commun/models/pretrained

# input_file=${input_root}/Mistral-Large-Instruct-2411/magpie_inst-Mistral-Large-Instruct-2411-241126_extracted-Llama-3.1-70B-Instruct_graded-Llama-3.1-70B-Instruct_processed.jsonl
# model_path=${model_root}/mistralai/Mistral-Large-Instruct-2411

# ./scripts/data_processing/split_file.sh $input_file
# sbatch examples/data/magpie/magpie_gen_response.slurm $input_file $model_path 0.7 1.0 4096
# ./scripts/data_processing/merge_files.sh $input_file "responded-${model_path##*/}"

# input_file=${input_root}/Llama-3.1-70B-Instruct/magpie_inst-Meta-Llama-3.1-70B-Instruct-241126_extracted-Llama-3.1-70B-Instruct_graded-Llama-3.1-70B-Instruct_processed.jsonl
# model_path=${model_root}/meta-llama/Llama-3.1-70B-Instruct

# ./scripts/data_processing/split_file.sh $input_file
# sbatch examples/data/magpie/magpie_gen_response.slurm $input_file $model_path 0.7 1.0 4096
# ./scripts/data_processing/merge_files.sh $input_file "responded-${model_path##*/}"

# input_file=${input_root}/Llama-3.1-Nemotron-70B-Instruct-HF/magpie_inst-Llama-3.1-Nemotron-70B-Instruct-HF-241126_extracted-Llama-3.1-70B-Instruct_graded-Llama-3.1-70B-Instruct_processed.jsonl
# model_path=${model_root}/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF

# ./scripts/data_processing/split_file.sh $input_file
# sbatch examples/data/magpie/magpie_gen_response.slurm $input_file $model_path 0.7 1.0 4096
# ./scripts/data_processing/merge_files.sh $input_file "responded-${model_path##*/}"

# input_file=${input_root}/Qwen2.5-72B-Instruct/magpie_inst-Qwen2.5-72B-Instruct-241126_extracted-Llama-3.1-70B-Instruct_graded-Llama-3.1-70B-Instruct_processed.jsonl
# model_path=${model_root}/Qwen/Qwen2.5-72B-Instruct

# ./scripts/data_processing/split_file.sh $input_file
# sbatch examples/data/magpie/magpie_gen_response.slurm $input_file $model_path 0.7 0.8 4096
# ./scripts/data_processing/merge_files.sh $input_file "responded-${model_path##*/}"

