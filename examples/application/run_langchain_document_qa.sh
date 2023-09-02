#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

export BITSANDBYTES_NOWELCOME="1"
export CUDA_VISIBLE_DEVICES="0"

# Install requirements
# pip install -U langchain sentence_transformers faiss-gpu

python vigogne/application/langchain/langchain_document_qa.py \
    --web_url "https://zaion.ai/en/resources/zaion-lab-blog/zaion-emotion-dataset" \
    --embedding_model_name_or_path "dangvantuan/sentence-camembert-base" \
    --llm_model_name_or_path "bofenghuang/vigogne-2-7b-chat" \
    --initial_question "Donne la définition de la speech emotion diarization."
