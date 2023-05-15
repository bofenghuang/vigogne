#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

CONFIG_FILE_ARG = "config"

# ignore index in loss
IGNORE_INDEX = -100

# tokenizer
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# data preprocess mode
CHAT = "chat"
INSTRUCT = "instruct"
VALID_MODES = {CHAT, INSTRUCT}

# keyword for chat json
# Role name
# USER = "UTILISATEUR"
USER = "USER"
ASSISTANT = "ASSISTANT"
# Data field
ID = "id"
CONVERSATION = "conversation"
ROLE = "role"
CONTENT = "content"
