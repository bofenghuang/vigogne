#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

export BITSANDBYTES_NOWELCOME="1"
export CUDA_VISIBLE_DEVICES="0"

# Launch demo for Vigogne instruct models
# python vigogne/inference/gradio/demo_instruct.py \
#     --base_model_name_or_path bofenghuang/vigogne-2-7b-instruct

# Launch demo for Vigogne chat models
python vigogne/inference/gradio/demo_chat.py \
    --base_model_name_or_path bofenghuang/vigogne-2-7b-chat
