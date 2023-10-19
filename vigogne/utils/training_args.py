# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Training arguments."""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from transformers import Seq2SeqTrainingArguments, TrainingArguments

from ..data_utils import DECODER, SUPPORTED_MODEL_TYPES
from ..processors import SUPPORTED_PROCESSORS

CONFIG_FILENAME = "hparams.json"


@dataclass
class VigogneTrainingArguments(TrainingArguments):
    """Arguments for training vigogne models."""

    # model arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model id of a pretrained model hosted inside a model repo on huggingface.co or the path to a"
                " directory containing model weights saved using `~PreTrainedModel.save_pretrained`."
            )
        },
    )
    model_revision: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model version to use. It can be a branch name, a tag name, or a commit id."},
    )
    model_type: str = field(default=DECODER, metadata={"help": "The type of the model architecture."})
    torch_dtype: str = field(
        default="float16",
        metadata={
            "help": (
                "Load the model under a specific `dtype`. Use `bfloat16` if you have NVIDIA architecture higher than"
                " Ampere to accelerate."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                " should only be set to `True` for repositories you trust and in which you have read the code, as it"
                " will execute code present on the Hub on your local machine."
            )
        },
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the flash_attention implemented in Huggingface's transformer repository."},
    )

    # quantiation arguments
    load_in_8bit: bool = field(
        default=False, metadata={"help": "Whether or not to convert the loaded model into mixed-8bit quantized model."}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "Whether or not to convert the loaded model into mixed-4bit quantized model."}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4", metadata={"help": "The quantization data type in the bnb.nn.Linear4Bit layers. Options are fp4 or nf4."}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to quantize again the quantization constants from the first quantization in nested"
                " quantization."
            )
        },
    )

    # peft arguments
    adapter: Optional[str] = field(default=None, metadata={"help": "Adapter type, can be either 'lora' or 'qlora'."})
    lora_r: int = field(default=8, metadata={"help": "Lora rank, aka the dimension used by the LoRA update matrices."})
    lora_alpha: int = field(default=16, metadata={"help": "Lora alpha, aka the scaling factor."})
    lora_dropout: float = field(default=0.05, metadata={"help": "The dropout probability for Lora layers."})
    lora_target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "The names of the modules to apply Lora to."}
    )
    lora_fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."},
    )
    lora_target_all_linear_layers: bool = field(
        default=False, metadata={"help": "Whether or not to target all linear layers."}
    )
    do_merge_lora: bool = field(
        default=False, metadata={"help": "Whether or not to merge LoRA weights to the base model after training."}
    )

    # model saving arguments
    max_shard_size: str = field(
        default="10GB",
        metadata={
            "help": (
                "The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a"
                ' unit (like `"5MB"`).'
            )
        },
    )

    # tokenizer arguments
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The id of a pretrained tokenizer hosted inside a model repo on huggingface.co or the path to a"
                " directory containing tokenizer saved using `~PreTrainedTokenizer.save_pretrained`."
            )
        },
    )
    tokenizer_revision: Optional[str] = field(
        default=None,
        metadata={"help": "The specific tokenizer version to use. It can be a branch name, a tag name, or a commit id."},
    )
    tokenizer_use_fast: bool = field(
        default=True,
        metadata={"help": "Whether or not to use the fast Rust-based tokenizer."},
    )
    tokenizer_legacy: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether or not the `legacy` behavior of the tokenizer should be used."},
    )
    tokenizer_padding_side: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The side on which the model should have padding applied. Should be selected between ['right',"
                " 'left']. Default value is picked from the class attribute of the same name."
            )
        },
    )
    add_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "The extra tokens to add to tokenier."},
    )
    add_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "The special tokens to add to tokenier."},
    )

    # data arguments
    train_file: Optional[str] = field(default=None, metadata={"help": "The local path to the training file."})
    eval_file: Optional[str] = field(default=None, metadata={"help": "The local path to the evaluation file."})
    eval_split_ratio: Optional[float] = field(default=None, metadata={"help": "The ratio of the evaluation split."})
    model_min_length: Optional[int] = field(
        default=None,
        metadata={"help": "The minimum sequence length. Sequences shorter than this will be filtered out."},
    )
    model_max_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum sequence length. Sequences longer than this will be filtered out."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
            )
        },
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )

    # processor augments
    processor_style: str = field(default="vigogne_chat_v3", metadata={"help": "The processor applied to dataset."})

    # packing arguments
    pack_into_block: bool = field(
        default=False,
        metadata={"help": "Whether to pack examples into blocks for faster training. Note this might affect performance."},
    )
    block_size: int = field(
        default=1024, metadata={"help": "Block size for packed examples. Only used when `pack_into_block` is True."}
    )

    # others
    config_path: Optional[str] = field(default=None, metadata={"help": "The file path to save config."})

    def __post_init__(self):
        super().__post_init__()

        # tokenizer
        self.tokenizer_name_or_path = self.tokenizer_name_or_path or self.model_name_or_path
        self.tokenizer_revision = self.tokenizer_revision or self.model_revision
        # todo: better handle
        self.add_tokens = json.loads(self.add_tokens) if self.add_tokens is not None else None
        self.add_special_tokens = json.loads(self.add_special_tokens) if self.add_special_tokens is not None else None

        # model
        assert not (
            self.load_in_8bit and self.load_in_4bit
        ), "You can't pass `load_in_8bit=True` and `load_in_4bit=True` at the same time"
        assert (
            self.model_type in SUPPORTED_MODEL_TYPES
        ), f"Specified model_type {self.model_type} doesn't exist in {SUPPORTED_MODEL_TYPES}"

        # data
        assert (
            self.processor_style in SUPPORTED_PROCESSORS.keys()
        ), f"Specified processor_style {self.processor_style} doesn't exist in {SUPPORTED_PROCESSORS}"

        # config
        self.config_path = self.config_path if self.config_path is not None else os.path.join(self.output_dir, CONFIG_FILENAME)


@dataclass
class VigogneSeq2SeqTrainingArguments(VigogneTrainingArguments, Seq2SeqTrainingArguments):
    ...
