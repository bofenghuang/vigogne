#!/bin/bash

#SBATCH --job-name=magpie-extract
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96           # number of cores per tasks
#SBATCH --gres=gpu:4                 # reserve number of GPUs per node
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=h100
#SBATCH --array=1-8

# Magpie instruction grading

set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
module load cuda/12.2.0
# conda
conda activate llm

# Debugging flags (optional)
# force crashing on nccl issues like hanging broadcast
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1
# export PYTHONFAULTHANDLER=1

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# slurm buffers python output by default, unbuffer python -u
export PYTHONUNBUFFERED="1"

# hf
# export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export HF_EVALUATE_OFFLINE="1"

# cuda
# export CUDA_VISIBLE_DEVICES="4,5,6,7"

# take arg
input_file=$1
model_path=$2

# model
# model_path="/projects/bhuang/models/llm/pretrained/meta-llama/Meta-Llama-3.1-70B-Instruct"

# tp_size=8
tp_size=4

# input_file="/projects/bhuang/corpus/text/llm/generated/magpie/magpie_inst-Meta-Llama-3.1-405B-Instruct-FP8-1.6m.jsonl"

# tmp: get tmp files
tmp_dir=${input_file%/*}/splitted_files
filename=${input_file##*/}
filename=${filename%.*}
# append SLURM_ARRAY_TASK_ID
input_file="${tmp_dir}/${filename}_0${SLURM_ARRAY_TASK_ID}.jsonl"

# output_file="${input_file%.*}_graded-l31-70b.jsonl"
output_file="${input_file%.*}_graded-${model_path##*/}.jsonl"

    # --max_samples 20

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
    --max_tokens 2048 \
    --max_model_len 2048 \
    --temperature 0 \
    --top_p 1.0

echo "END TIME: $(date)"
