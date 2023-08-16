#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

export BITSANDBYTES_NOWELCOME="1"
export CUDA_VISIBLE_DEVICES="0"

# Install requirements
# pip install "fschat[model_worker,webui]"

# Launch vllm server with Vigogne instruct models
# python -m fastchat.serve.cli \
#     --model bofenghuang/vigogne-2-7b-instruct \
#     --temperature 0.1 \
#     --max-new-tokens 1024

# Launch vllm server with Vigogne chat models
python -m fastchat.serve.cli \
    --model bofenghuang/vigogne-2-7b-chat \
    --temperature 0.1 \
    --max-new-tokens 1024
