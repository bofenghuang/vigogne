#!/bin/bash

# pipeline for magpie instruct extraction & grading

stage=${1:-0}

# input_file="/lustre/fswork/projects/rech/gkb/commun/corpus/text/llm_instruct/magpie-fr/magpie_inst-Meta-Llama-3.1-405B-Instruct-FP8-1.6m.jsonl"
# input_file="/lustre/fswork/projects/rech/gkb/commun/corpus/text/llm_instruct/magpie-fr/magpie_inst-Meta-Llama-3.1-70B-Instruct-1m.jsonl"
input_file=$2

model_path="/lustre/fswork/projects/rech/gkb/commun/models/pretrained/meta-llama/Meta-Llama-3.1-70B-Instruct"
# model_path="/lustre/fswork/projects/rech/gkb/commun/models/pretrained/mistralai/Mistral-Large-Instruct-2407"

model_name=${model_path##*/}

# Step 1: split and extract
if [ $stage -eq 0 ]; then
    echo -e "Step 1.1: split...\n"
    # split
    ./scripts/data_processing/split_file.sh $input_file
fi

if [ $stage -eq 0 ]; then
    echo -e "Step 1.2: extract...\n"
    # extract instruct
    sbatch examples/data/magpie_extract_instruct.slurm $input_file $model_path
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
    echo -e "Step 2.2: split...\n"
    # split
    ./scripts/data_processing/split_file.sh $input_file
fi

if [ $stage -eq 1 ]; then
    echo -e "Step 2.3: grade...\n"
    # grade instruct
    sbatch examples/data/magpie_grade_instruct.slurm $input_file $model_path
fi

# Step 3. merge
if [ $stage -eq 2 ]; then
    echo -e "Step 3: merge...\n"
    # merge
    ./scripts/data_processing/merge_files.sh $input_file "graded-${model_name}"
fi

# update input_file to "extracted"
input_file="${input_file%.*}_extracted-${model_name}_graded-${model_name}.jsonl"

# Step 4. filter

# if [ $stage -eq 3 ]; then
#     echo -e "Step 5.1: split...\n"
#     # split
#     ./scripts/data_processing/split_file.sh $input_file
# fi

# if [ $stage -eq 3 ]; then
#     echo -e "Step 5.2: respond...\n"
#     # grade instruct
#     sbatch examples/data/magpie_gen_response.slurm $input_file $model_path
# fi

# # Step 3. merge
# if [ $stage -eq 4 ]; then
#     echo -e "Step 6: merge...\n"
#     # merge
#     ./scripts/data_processing/merge_files.sh $input_file "responded-${model_name}"
# fi