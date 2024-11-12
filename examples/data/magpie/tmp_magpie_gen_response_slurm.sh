#!/bin/bash

input_file=/lustre/fswork/projects/rech/gkb/commun/corpus/text/llm_instruct/magpie-fr/magpie_inst-Mistral-Large-Instruct-2407-640k-241109_extracted-Meta-Llama-3.1-70B-Instruct_graded-Meta-Llama-3.1-70B-Instruct_processed09.jsonl
model_path=/lustre/fswork/projects/rech/gkb/commun/models/pretrained/mistralai/Mistral-Large-Instruct-2407

./scripts/data_processing/split_file.sh $input_file
sbatch examples/data/magpie/magpie_gen_response.slurm $input_file $model_path 0.7 1.0 4096
./scripts/data_processing/merge_files.sh $input_file "responded-${model_name}"

input_file=/lustre/fswork/projects/rech/gkb/commun/corpus/text/llm_instruct/magpie-fr/magpie_inst-Llama-3.1-Nemotron-70B-Instruct-HF-720k-241109_extracted-Meta-Llama-3.1-70B-Instruct_graded-Meta-Llama-3.1-70B-Instruct_processed09.jsonl
model_path=/lustre/fswork/projects/rech/gkb/commun/models/pretrained/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF

./scripts/data_processing/split_file.sh $input_file
sbatch examples/data/magpie/magpie_gen_response.slurm $input_file $model_path 0.7 1.0 4096
./scripts/data_processing/merge_files.sh $input_file "responded-${model_name}"

input_file=/lustre/fswork/projects/rech/gkb/commun/corpus/text/llm_instruct/magpie-fr/magpie_inst-Qwen2.5-72B-Instruct-1m-241109_extracted-Meta-Llama-3.1-70B-Instruct_graded-Meta-Llama-3.1-70B-Instruct_processed09.jsonl
model_path=/lustre/fswork/projects/rech/gkb/commun/models/pretrained/Qwen/Qwen2.5-72B-Instruct

./scripts/data_processing/split_file.sh $input_file
sbatch examples/data/magpie/magpie_gen_response.slurm $input_file $model_path 0.7 0.8 4096
./scripts/data_processing/merge_files.sh $input_file "responded-${model_name}"
