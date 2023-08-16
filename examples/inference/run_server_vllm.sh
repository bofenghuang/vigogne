#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

export BITSANDBYTES_NOWELCOME="1"
export CUDA_VISIBLE_DEVICES="0"

# Install requirements
# pip install vllm

# Launch vllm server with Vigogne instruct models
# python -m vllm.entrypoints.openai.api_server \
#     --model bofenghuang/vigogne-2-7b-instruct \
#     --host "0.0.0.0"

# Launch vllm server with Vigogne chat models
python -m vllm.entrypoints.openai.api_server \
    --model bofenghuang/vigogne-2-7b-chat \
    --host "0.0.0.0"
