#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# Install requirements
# pip install "fschat[model_worker,webui]"

# Path to quantized model like path/to/vigogne/models/vigogne_2_7b_chat/ggml-model-q4_0.bin
model=$1

# Launch Vigogne instruct models
# ./main -m $model \
#     --color \
#     -f prompts/instruct.txt \
#     -ins \
#     -c 4096 \
#     -n 256 \
#     --temp 0.1 \
#     --repeat_penalty 1.1

# Launch Vigogne chat models
./main -m $model \
    --color \
    -f prompts/chat.txt \
    --reverse-prompt "<|user|>:" \
    --in-prefix " " \
    --in-suffix "<|assistant|>:" \
    --interactive-first \
    -c 4096 \
    -n -1 \
    --temp 0.1
