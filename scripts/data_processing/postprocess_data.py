#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

"""

Usage:
python scripts/data_processing/postprocess_data.py \
    --input_file data/alpaca_data_cleaned.jsonl \
    --output_file data/alpaca_data_cleaned_fr.jsonl \
    --min_response_length 8 \
    --embedding_model_name_or_path dangvantuan/sentence-camembert-base \
    --threshold_similarity 0.95
"""

import re
from typing import Optional
import faiss
import fire
import torch
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer


# todo
def get_utterance_text(example, role):
    return "\n".join([x["content"] for x in example["messages"] if x["role"] == role])


def main(
    input_file: str,
    output_file: str,
    tokenizer_name_or_path: str = "meta-llama/Llama-2-7b-hf",
    min_response_length: Optional[int] = None,
    embedding_model_name_or_path: str = "dangvantuan/sentence-camembert-base",
    batch_size: int = 64,
    threshold_similarity: Optional[float] = 0.95,
    num_proc: int = 32,
):
    dataset = load_dataset("json", data_files=input_file, split="train")
    print(f"Loaded {dataset.num_rows:,d} examples from {input_file}")

    # debug
    # dataset = dataset.select(range(10))

    # filter by response length
    if min_response_length is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        dataset = dataset.map(
            lambda example: {
                "response_length": len(tokenizer(get_utterance_text(example, role="assistant"))["input_ids"])
            },
            num_proc=num_proc,
        )

        dataset = dataset.filter(
            lambda x: x >= min_response_length, input_columns="response_length", num_proc=num_proc
        )
        # debug
        # dataset = dataset.filter(lambda x: x < min_response_length, input_columns="response_length", num_proc=num_proc)
        print(f"Filtered to {dataset.num_rows:,d} examples with min_response_length {min_response_length}")

        # sort by response length
        dataset = dataset.remove_columns("response_length")

    if threshold_similarity is not None:
        st_model = SentenceTransformer(embedding_model_name_or_path, device="cuda")
        # todo: longer
        st_model.max_seq_length = 512

        # Initialize FAISS index on GPU
        res = faiss.StandardGpuResources()
        emb_dim = st_model.get_sentence_embedding_dimension()
        index = faiss.GpuIndexFlatIP(res, emb_dim)

        def process_function(example):
            question_text = get_utterance_text(example, role="user")
            question_text = example["system"] + question_text

            return {"question_text": question_text, "question_length": len(tokenizer(question_text)["input_ids"])}

        dataset = dataset.map(process_function, num_proc=num_proc)
        # keep longest
        dataset = dataset.sort("question_length", reverse=True)

        # compute embeddings by batch
        embeddings = st_model.encode(
            dataset["question_text"], batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True
        )

        # # cosine similarity filter
        # filtered_embeddings = torch.empty((0, st_model[1].word_embedding_dimension), dtype=torch.float32).to(
        #     st_model.device
        # )

        # def filter_function(example, example_idx):
        #     global filtered_embeddings

        #     similarity_scores = util.cos_sim(embeddings[example_idx], filtered_embeddings)

        #     if filtered_embeddings.size()[0] == 0 or similarity_scores.max() < threshold_similarity:
        #         filtered_embeddings = torch.cat((filtered_embeddings, embeddings[example_idx].unsqueeze(0)))
        #         return True
        #     else:
        #         # print(example_idx)
        #         # print(torch.argmax(similarity_scores).item())
        #         return False

        # dataset = dataset.filter(filter_function, with_indices=True)

        # convert to numpy for faiss
        embeddings_np = embeddings.cpu().numpy()

        # Iinitialize filtered indices and first embedding
        filtered_indices = [0]
        index.add(embeddings_np[0:1])

        # Process remaining embeddings
        for i in range(1, len(embeddings_np)):
            # Search current index
            D, _ = index.search(embeddings_np[i:i+1], 1)
            
            if D[0][0] < threshold_similarity:
                filtered_indices.append(i)
                index.add(embeddings_np[i:i+1])

        # Filter dataset
        filtered_dataset = dataset.select(filtered_indices)

        print(f"Filtered to {dataset.num_rows:,d} examples with max_cos_similarity {threshold_similarity}")

        dataset = dataset.remove_columns(["question_text", "question_length"])

    # export
    dataset = dataset.shuffle(10)
    dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"Saved data into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
