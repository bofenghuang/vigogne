#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

# split json lines file

# take args
input_file=$1
N=${2:-8}

tmp_dir=${input_file%/*}/splitted_files

split -n l/$N --numeric-suffixes=1 --additional-suffix=.jsonl $input_file ${input_file%.*}_

[ -d tmp_dir ] || mkdir $tmp_dir

mv ${input_file%.*}_*.jsonl $tmp_dir
