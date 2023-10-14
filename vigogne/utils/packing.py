# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Pack examples (TODO)."""

from itertools import chain
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm


# Modified from https://github.com/facebookresearch/llama-recipes/blob/905f633dab92688b0a989f8d5cd11d86f882f534/ft_datasets/utils.py#L8-L39
class Concatenator:
    def __init__(self, block_size=2048):
        self.block_size = block_size
        # self.remainder = {"input_ids": [], "attention_mask": [], "labels": []}
        self.remainder = defaultdict(list)

    def __call__(self, batch):
        # Concatenate all values
        # concatenated_samples = {k: v + list(chain(*batch[k])) for k, v in self.remainder.items()}
        concatenated_samples = {k: self.remainder[k] + list(chain(*v)) for k, v in batch.items()}

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.block_size:
            num_blocks = total_length // self.block_size
            result = {
                k: [v[i : i + self.block_size] for i in range(0, num_blocks * self.block_size, self.block_size)]
                for k, v in concatenated_samples.items()
            }
            self.remainder = {k: v[num_blocks * self.block_size :] for k, v in concatenated_samples.items()}
        else:
            result = concatenated_samples
            self.remainder = {k: [] for k in concatenated_samples.keys()}

        return result


class ModerateConcatenator:
    """
    Packing ensures total length within the block_size without truncation or inserting tailing parts into subsequent blocks.
    """

    def __init__(self, block_size=2048):
        self.block_size = block_size

    def __call__(self, batch):
        keys = list(batch.keys())

        result = defaultdict(list)
        current_block = defaultdict(list)

        for input_index in range(len(batch[keys[0]])):
            if len(current_block[keys[0]]) + len(batch[keys[0]][input_index]) < self.block_size:
                for k in keys:
                    current_block[k] += batch[k][input_index]
            else:
                for k in keys:
                    result[k].append(current_block[k])
                    current_block[k] = batch[k][input_index]

        for k in keys:
            result[k].append(current_block[k])

        return result


# Modified from https://github.com/facebookresearch/llama-recipes/blob/905f633dab92688b0a989f8d5cd11d86f882f534/ft_datasets/utils.py#L41-L65
class ConcatDataset(Dataset):
    def __init__(self, dataset, block_size=4096):
        self.dataset = dataset
        self.block_size = block_size

        self.samples = []

        # buffer = {"input_ids": [], "attention_mask": [], "labels": []}
        buffer = defaultdict(list)

        for sample in tqdm(self.dataset, desc="Preprocessing dataset"):
            # buffer = {k: v + sample[k] for k, v in buffer.items()}
            buffer = {k: buffer[k] + v for k, v in sample.items()}

            while len(next(iter(buffer.values()))) > self.block_size:
                self.samples.append({k: v[: self.block_size] for k, v in buffer.items()})
                buffer = {k: v[self.block_size :] for k, v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
