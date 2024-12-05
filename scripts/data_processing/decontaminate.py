#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

"Decontaminate against benchmark questions."

# https://github.com/facebookresearch/faiss/blob/main/tutorial/python/4-GPU.py

from typing import Union

import faiss
import fire
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# use a single GPU
res = faiss.StandardGpuResources()

ENCODING_PROMPTS = {"intfloat/multilingual-e5-base": "query: "}


def main(
    question_file: Union[str, list[str]],
    dataset_file: Union[str, list[str]],
    output_dataset_file: str,
    question_column: str = "text",
    embedding_model_name_or_path: str = "OrdalieTech/Solon-embeddings-base-0.1",
    batch_size: int = 64,
    max_cosine_similarity: float = 0.9,
    preprocessing_num_workers: int = 16,
    instruction_column: str = "tmp_instruction",
):
    # load existing questions
    existing_questions = load_dataset("json", data_files=question_file, split="train")[question_column]
    print(f"Loaded {len(existing_questions):,} existing questions")

    # load dataset
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    print(f"Loaded {dataset.num_rows:,} examples")

    # decontaminate by first instruction
    dataset = dataset.map(
        lambda example: {instruction_column: next((x["content"] for x in example["messages"] if x["role"] == "user"), "")},
        num_proc=preprocessing_num_workers,
    )

    # load emb model
    emb_model = SentenceTransformer(embedding_model_name_or_path, device="cuda")

    d = emb_model.get_sentence_embedding_dimension()
    # index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index
    index_flat = faiss.IndexFlatIP(d)  # build the index
    # make it a flat GPU index
    index = faiss.index_cpu_to_gpu(res, 0, index_flat)

    encoding_kwargs = {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "prompt": ENCODING_PROMPTS.get(embedding_model_name_or_path),
        # "precision":
        # "normalize_embeddings":
        "show_progress_bar": True,
    }

    # compute embeddings by batch
    existing_question_embeddings = emb_model.encode(existing_questions, **encoding_kwargs)

    # existing_question_embeddings = existing_question_embeddings.cpu().numpy()
    faiss.normalize_L2(existing_question_embeddings)
    # add vectors to the index
    index.add(existing_question_embeddings)

    embeddings = emb_model.encode(dataset[instruction_column], **encoding_kwargs)

    # embeddings = embeddings.cpu().numpy()
    faiss.normalize_L2(embeddings)
    distances, _ = index.search(embeddings, 1)

    selected_indices = np.where(distances[..., 0] < max_cosine_similarity)[0].tolist()
    processed_dataset = dataset.select(selected_indices)
    print(f"Filtered to {processed_dataset.num_rows:,} examples")

    processed_dataset = processed_dataset.remove_columns(instruction_column)

    # export
    processed_dataset.to_json(output_dataset_file, orient="records", lines=True, force_ascii=False)
    print(f"Saved into {output_dataset_file}")


if __name__ == "__main__":
    fire.Fire(main)
