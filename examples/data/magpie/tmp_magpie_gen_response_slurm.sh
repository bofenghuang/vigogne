#!/bin/bash

input_file=
model_path=/lustre/fswork/projects/rech/gkb/commun/models/pretrained/mistralai/Mistral-Large-Instruct-2407

./scripts/data_processing/split_file.sh $input_file
sbatch examples/data/magpie_gen_response.slurm $input_file $model_path 0.7 1.0 4096
./scripts/data_processing/merge_files.sh $input_file "responded-${model_name}"

input_file=
model_path=/lustre/fswork/projects/rech/gkb/commun/models/pretrained/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF

./scripts/data_processing/split_file.sh $input_file
sbatch examples/data/magpie_gen_response.slurm $input_file $model_path 0.7 1.0 4096
./scripts/data_processing/merge_files.sh $input_file "responded-${model_name}"

input_file=
model_path=/lustre/fswork/projects/rech/gkb/commun/models/pretrained/Qwen/Qwen2.5-72B-Instruct

./scripts/data_processing/split_file.sh $input_file
sbatch examples/data/magpie_gen_response.slurm $input_file $model_path 0.7 0.8 4096
./scripts/data_processing/merge_files.sh $input_file "responded-${model_name}"
