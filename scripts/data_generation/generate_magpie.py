#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

"""
Generate magpie-style instructions.

Usage:
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

"""

import os
import random
import time
import torch
from vllm import LLM, SamplingParams
import fire
from tqdm import tqdm
import json

# magpie task categories
booster_str_lst = [
    "L'utilisateur pose des questions qui peuvent être des demandes d'informations spécifiques ou de faits sur divers sujets.",
    "L'utilisateur pose des questions qui peuvent nécessiter une réflexion logique, la résolution de problèmes ou le traitement d'idées complexes.",
    "L'utilisateur pose des questions qui peuvent nécessiter de l'aide pour créer des plans ou des stratégies pour des activités et des projets.",
    "L'utilisateur pose des questions qui peuvent nécessiter de l'aide pour la révision, la reformulation, la relecture ou d'autres tâches liées à la composition de contenu écrit général.",
    "L'utilisateur pose des questions qui peuvent chercher de l'aide pour écrire, examiner ou corriger du code en programmation.",
    "L'utilisateur pose des questions qui peuvent être liées à des concepts mathématiques, des problèmes ou des calculs.",
    "L'utilisateur pose des questions qui peuvent impliquer des scénarios nécessitant que ChatGPT adopte un personnage ou un rôle.",
    "L'utilisateur pose des questions qui peuvent impliquer l'interprétation de données, de statistiques ou l'exécution de tâches analytiques.",
    "L'utilisateur pose des questions qui peuvent nécessiter de l'aide pour rédiger des histoires, des poèmes ou d'autres textes créatifs.",
    "L'utilisateur pose des questions qui peuvent demander des recommandations ou des conseils sur divers sujets personnels ou professionnels.",
    "L'utilisateur pose des questions qui peuvent impliquer la génération d'idées, la pensée créative ou l'exploration de possibilités.",
]


def main(
    model_name_or_path: str,
    output_file: str,
    num_instructions_to_generate: int = 100,
    **kwargs,
):

    llm = LLM(
        model=model_name_or_path,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        max_tokens=512,
        stop_token_ids=[128009, 128001, 128006, 128007],
    )

    prompts = []
    for _ in range(num_instructions_to_generate):
        system_prompt = f"Une conversation entre un utilisateur curieux et un assistant d'IA. {random.choice(booster_str_lst)}"  # introduce randomness
        instruction = f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        prompts.append(instruction)

    start_time = time.perf_counter()

    outputs = llm.generate(prompts, sampling_params)

    print(f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, mode="w", encoding="utf-8") as f:
        for output in tqdm(outputs, desc="Writting to json..."):
            result = {
                "prompt": output.prompt,
                "generated_text": output.outputs[0].text,
            }
            f.write(json.dumps(result, default=str, ensure_ascii=False) + "\n")

    print(f"The generated data is saved in {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
