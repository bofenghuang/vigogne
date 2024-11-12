#!/bin/bash

#SBATCH --job-name=magpie-resp
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96           # number of cores per tasks
#SBATCH --gres=gpu:4                 # reserve number of GPUs per node
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=h100
#SBATCH --account=gkb@h100           # account to use, with xyz the 3 letter code of your project
#SBATCH --array=1-8

# Magpie response generation

set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
module load arch/h100
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
temp=${3:-0.7}
topp=${4:-1.0}
maxtokens=${5:-4096}
maxseqs=${6:-256}

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

output_file="${input_file%.*}_responded-${model_path##*/}.jsonl"

    # --max_samples 20

# generate response
# todo: hyperparams by model
# todo: max_num_seqs vs max_num_batched_tokens
# preemption error https://docs.vllm.ai/en/latest/models/performance.html
    # l3.1-405b
    # --max_num_seqs 64 \
    # --temperature 0.6 \
    # --top_p 0.9
    # mistral
    # --temperature 0.7 \
    # --top_p 1.0
python scripts/data_generation/generate_response_b.py \
    --input_file $input_file \
    --output_file $output_file \
    --id_column_name instruction \
    --instruct_column_name instruction \
    --output_column_name response \
    --model_configs_file data/configs/model_configs.json \
    --model_name_or_path $model_path \
    --dtype bfloat16 \
    --tensor_parallel_size $tp_size \
    --gpu_memory_utilization 0.95 \
    --batch_size 1024 \
    --max_num_seqs $maxseqs \
    --max_tokens $maxtokens \
    --max_model_len $maxtokens \
    --temperature $temp \
    --top_p $topp

echo "END TIME: $(date)"
