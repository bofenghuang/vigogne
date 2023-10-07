# coding=utf-8
# Copyright 2023  Bofeng Huang

"""Copied and modified from https://github.com/huggingface/transformers/blob/main/src/transformers/utils/logging.py"""

import logging


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def set_verbosity(verbosity: int) -> None:
    # _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)
