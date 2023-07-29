# coding=utf-8
# Copyright 2023  Bofeng Huang

import json
import io
import os
import threading

lck = threading.Lock()


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def jdump(obj, f, mode="w", indent=4, default=str, ensure_ascii=False):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default, ensure_ascii=ensure_ascii)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jsonl_load(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jlist = [json.loads(l.strip()) for l in f]
    f.close()
    return jlist


def jsonl_dump(obj, f, mode="a", default=str, ensure_ascii=False):
    f = _make_w_io_base(f, mode)
    if isinstance(obj, dict):
        f.write(json.dumps(obj, default=default, ensure_ascii=ensure_ascii) + "\n")
    elif isinstance(obj, list):
        for item in obj:
            f.write(json.dumps(item, default=default, ensure_ascii=ensure_ascii) + "\n")
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


# def thread_safe_jsonl_dump(obj, f, mode="a", default=str, ensure_ascii=False):
#     f = _make_w_io_base(f, mode)
#     with lck:
#         if isinstance(obj, (dict, list)):
#             f.write(json.dumps(obj, default=default, ensure_ascii=ensure_ascii) + "\n")
#         elif isinstance(obj, str):
#             f.write(obj + "\n")
#         else:
#             raise ValueError(f"Unexpected type: {type(obj)}")
#     f.close()


def thread_safe_jsonl_dump(obj, f, **kwargs):
    # acquire the lock
    with lck:
        jsonl_dump(obj, f, **kwargs)
