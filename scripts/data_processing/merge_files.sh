#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

# merge json lines files

# take args
input_file=$1
suffix=$2

input_dir=${input_file%/*}
tmp_dir=${input_dir}/splitted_files

filename=${input_file##*/}
filename=${filename%.*}

splitted_files=${tmp_dir}/${filename}_*_${suffix}.jsonl
output_file=${input_dir}/${filename}_${suffix}.jsonl

cat $splitted_files > $output_file

# wc -l $input_file
# wc -l $output_file
