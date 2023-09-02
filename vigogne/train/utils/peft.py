# coding=utf-8
# Copyright 2023  Bofeng Huang

import logging
from collections import defaultdict

# import bitsandbytes as bnb

logger = logging.getLogger(__name__)


# Modified from Peft
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    param_by_dtype = defaultdict(int)

    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

        param_by_dtype[param.dtype] += num_params

    logger.info(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    )
    for k, v in param_by_dtype.items():
        logger.info(f"dtype: {k} || num: {v:,d} || percentage: {100 * v / all_param:.4f}%")


# todo
# Modified from https://github.com/artidoro/qlora/blob/main/qlora.py
# def find_all_linear_names(model):
#     lora_module_names = set()
#     for name, module in model.named_modules():
#         if isinstance(module, bnb.nn.Linear4bit):
#             names = name.split(".")
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])

#     if "lm_head" in lora_module_names:  # needed for 16-bit
#         lora_module_names.remove("lm_head")
#     return list(lora_module_names)
