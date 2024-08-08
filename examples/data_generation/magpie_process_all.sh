#!/bin/bash

# pipeline for magpie instruct extraction & grading

stage=${1:-0}

input_file="/lustre/fswork/projects/rech/gkb/commun/corpus/text/llm_instruct/magpie-fr/magpie_inst-Meta-Llama-3.1-405B-Instruct-FP8-1.6m.jsonl"
model_path="/lustre/fswork/projects/rech/gkb/commun/models/pretrained/meta-llama/Meta-Llama-3.1-70B-Instruct"

model_name=${model_path##*/}

# Step 1: split and extract
if [ $stage -eq 0 ]; then
    echo -e "Step 1: split and extract...\n"
    # split
    ./scripts/data_processing/split_file.sh $input_file
    # extract instruct
    sbatch examples/data_generation/magpie_extract_instruct.slurm $input_file $model_path
fi

# Step 2. merge, resplit, and grade
if [ $stage -eq 1 ]; then
    echo -e "Step 2.1: merge...\n"
    # merge
    ./scripts/data_processing/merge_files.sh $input_file "extracted-${model_name}"
fi

# update input_file to "extracted"
input_file="${input_file%.*}_extracted-${model_name}.jsonl"

if [ $stage -eq 1 ]; then
    echo -e "Step 2.2: resplit, and grade...\n"
    # split
    ./scripts/data_processing/split_file.sh $input_file
    # grade instruct
    sbatch examples/data_generation/magpie_grade_instruct.slurm $input_file $model_path
fi

# Step 3. merge
if [ $stage -eq 2 ]; then
    echo -e "Step 3: merge...\n"
    # merge
    ./scripts/data_processing/merge_files.sh $input_file "graded-${model_name}"
fi
