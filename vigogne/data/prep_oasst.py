#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Modified from https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/oasst_dataset.py

# 1. Download oasst conversation tree file
wget https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz

# 2. Convert to chat
# Requirement: Python 3.10, oasst-data (See https://github.com/LAION-AI/Open-Assistant/tree/main/oasst-data)

python vigogne/data/prep_oasst.py \
    --input_file path/to/2023-04-12_oasst_ready.trees.jsonl.gz \
    --output_file data/oasst_20230412_fr_chat.jsonl \
    --lang fr
"""

from functools import partial
from pathlib import Path
from typing import Iterable, Literal, Optional

import fire
# from model_training.custom_datasets.formatting import DatasetEntrySft, Role, Utterance
from oasst_data import ExportMessageNode, read_dataset_message_trees, read_message_trees, visit_threads_depth_first
from oasst_data.schemas import ExportMessageTree
from torch import Generator
from torch.utils.data import Dataset, random_split

from vigogne.constants import ASSISTANT, CONTENT, CONVERSATION, ID, ROLE, USER
from vigogne.data.utils import jsonl_dump


class ListDataset(Dataset):
    def __init__(self, data: list):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_oasst_export(
    input_file_path: Optional[str | Path] = None,
    hf_dataset_name: Optional[str] = "OpenAssistant/oasst1",
    val_split: float = 0.2,
    lang: str = "en",
    top_k: Optional[int] = None,
    manual_seed: int = 287631038922,
    data_path: str | Path = None,
    mode: Literal["sft", "rm", "rl"] = "sft",
) -> tuple[ListDataset, ListDataset]:
    if mode not in ("sft", "rm", "rl"):
        raise ValueError(f"Unknown dataset mode: {mode}")

    lang_codes: list[str] = lang.split(",")

    generator = Generator()
    generator.manual_seed(manual_seed)

    tree_iter: Iterable[ExportMessageTree] = None
    if input_file_path:
        if not isinstance(input_file_path, Path):
            input_file_path = Path(input_file_path)
        if not input_file_path.is_absolute() and data_path:
            if not isinstance(data_path, Path):
                data_path = Path(data_path)
            input_file_path = data_path / input_file_path
        tree_iter = read_message_trees(input_file_path)
    elif hf_dataset_name:
        tree_iter = read_dataset_message_trees(hf_dataset_name, split="train+validation")
    else:
        raise RuntimeError("Either `input_file_path` or `hf_dataset_name` must be specified.")

    threads_per_tree = []
    for tree in tree_iter:
        if tree.tree_state != "ready_for_export" or not tree.prompt.review_result:
            continue

        if mode in ("sft", "rm"):
            if tree.tree_state != "ready_for_export":
                continue
        elif mode == "rl":
            if tree.tree_state not in ("ready_for_export", "prompt_lottery_waiting"):
                continue

        # extract all threads up to last assistant reply
        threads: list[list[ExportMessageNode]] = []

        def thread_filter(thread: list[ExportMessageNode]) -> bool:
            if any(m.deleted or m.synthetic for m in thread):
                return False

            if top_k is not None:
                for i, m in enumerate(thread):
                    if m.role == "assistant":
                        if m.rank is None:
                            if i > 0 and len(thread[i - 1].replies) > 1:
                                return False
                        elif m.rank >= top_k:
                            return False
            return True

        def leaf_filter(thread: list[ExportMessageNode]) -> bool:
            if mode == "sft":
                # in SFT mode `not thread[-1].replies` finds nodes without children (leaves).
                # We are interested in those which are role='assistant' but some trees don't end on assistant nodes
                # but have prompter leaves .. we want to use those trees too .. e.g. remove the last prompter message(s)
                # so that they end with assistant. The `thread[-2].replies[0] == thread[-1]` check makes sure that only
                # the FIRST prompter reply is added .. e.g. the parent does not appear multiple times and we can use
                # pop() to remove superfluous prompter leaf node later.
                return (
                    len(thread) > 1
                    and not thread[-1].replies
                    and (thread[-1].role == "assistant" or thread[-2].replies[0] == thread[-1])
                    and thread_filter(thread)
                )
            elif mode == "rm":
                # for reward models we use thread-fragments ending on prompter messages as prefix and
                # their (ranked) replies as possible continuations.
                return (
                    thread[-1].role == "prompter"
                    and len([r for r in thread[-1].replies if r.rank is not None]) > 1
                    and thread_filter(thread)
                )
            elif mode == "rl":
                # during rl we are interested in all possible prefixes ending in prompter messages
                return thread[-1].role == "prompter" and not any(m.deleted or m.synthetic for m in thread)

            raise RuntimeError()

        visit_threads_depth_first(tree.prompt, visitor=threads.append, predicate=leaf_filter)
        if mode == "sft":
            for t in threads:
                if t[-1].role == "prompter":
                    t.pop()

        threads_per_tree.append(threads)

    def process_thread(thread: list[ExportMessageNode]):
        if mode == "sft":
            # ensure roles are strictly alternating between prompter and assistant
            assert all(m.role == "prompter" for m in thread[0::2]) and all(m.role == "assistant" for m in thread[1::2])
            return [m.text for m in thread]
            # conversation: list[Utterance] = [
            #     Utterance(
            #         text=m.text,
            #         role=Role.prompter if m.role == "prompter" else Role.assistant,
            #         lang=m.lang,
            #         quality=m.get_label_value("quality"),
            #         humor=m.get_label_value("humor"),
            #         creativity=m.get_label_value("creativity"),
            #     )
            #     for m in thread
            # ]
            # return DatasetEntrySft(conversation=conversation)
        elif mode == "rm":
            prefix = [m.text for m in thread]
            replies = [r for r in thread[-1].replies if r.role == "assistant" and r.rank is not None]
            replies = sorted(replies, key=lambda r: r.rank)
            replies = [r.text for r in replies]
            return (prefix, replies)
        elif mode == "rl":
            return ([m.text for m in thread],)

        raise RuntimeError()

    # split on tree basis, messages from same tree must not end up in different splits
    trees = ListDataset(threads_per_tree)
    splits = random_split(trees, lengths=[1.0 - val_split, val_split], generator=generator)

    def flatten(ds: ListDataset) -> ListDataset:
        return ListDataset([process_thread(thread) for tree_threads in ds for thread in tree_threads])

    train = flatten(splits[0])
    val = flatten(splits[1])

    if input_file_path:
        print(f"OASST JSONL file {str(input_file_path)}: {len(train)=}, {len(val)=}")
    else:
        print(f"OASST HF dataset {hf_dataset_name}: {len(train)=}, {len(val)=}")

    return train, val


def convert_to_chat(example_input, task_id_prefix):
    example_idx, example = example_input

    conversation = []
    for idx in range(0, len(example), 2):
        conversation.append({ROLE: USER, CONTENT: example[idx]})
        conversation.append({ROLE: ASSISTANT, CONTENT: example[idx + 1]})

    return {ID: f"{task_id_prefix}-{example_idx:08d}", CONVERSATION: conversation}


def main(
    input_file, output_file, hf_dataset_name=None, val_split=0, lang="fr", top_k=None, task_id_prefix="oasst-20230412-fr"
):
    train_data, _ = load_oasst_export(
        input_file_path=input_file, hf_dataset_name=hf_dataset_name, val_split=val_split, lang=lang, top_k=top_k, mode="sft"
    )
    convert_to_chat_p = partial(convert_to_chat, task_id_prefix=task_id_prefix)
    reformatted_data = list(map(convert_to_chat_p, enumerate(train_data)))
    jsonl_dump(reformatted_data, output_file, mode="w")
    print(f"Saved {len(reformatted_data)} examples into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
