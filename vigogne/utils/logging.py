# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Logging utilities.

Adapted from https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/logging_config.py
"""

import logging
import os
import sys
from logging import Formatter
from logging.config import dictConfig
from typing import Any, Dict

# import transformers
from colorama import Fore, Style, init


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def set_verbosity(verbosity: int) -> None:
    # _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


class ColorfulFormatter(Formatter):
    """Formatter adding coloring and process rank."""

    colors = {
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        # todo: torch.distributed.get_rank()
        # record.rank = int(os.getenv("LOCAL_RANK", "0"))
        log_message = super().format(record)
        return self.colors.get(record.levelname, "") + log_message + Fore.RESET


LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    # "disable_existing_loggers": True,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            "datefmt": "%m/%d/%Y %H:%M:%S",
        },
        "colorful": {
            "()": ColorfulFormatter,
            # "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [RANK:%(rank)d] %(message)s",
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            "datefmt": "%m/%d/%Y %H:%M:%S",
        },
    },
    "filters": {},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": sys.stdout,
            "filters": [],
        },
        "color_console": {
            "class": "logging.StreamHandler",
            "formatter": "colorful",
            "stream": sys.stdout,
            "filters": [],
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
    "loggers": {
        "vigogne": {
            "handlers": ["color_console"],
            "level": "INFO",
            "propagate": False,
        },
        "transformers": {
            "handlers": ["color_console"],
            "level": "WARNING",
            "propagate": False,
        },
    },
}


def configure_default_logging():
    """Configure default logging"""
    init()  # Initialize colorama
    dictConfig(LOGGING_CONFIG)


def configure_logging(cfg: Any):
    """Configure default logging for process"""
    configure_default_logging()

    log_level = cfg.get_process_log_level()

    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()
    set_verbosity(log_level)
