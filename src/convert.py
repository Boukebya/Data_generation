import argparse
import json
import os
import shutil
from collections import defaultdict
from inspect import signature
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Set, Tuple

import torch

from huggingface_hub import CommitInfo, CommitOperationAdd, Discussion, HfApi, hf_hub_download
from huggingface_hub.file_download import repo_folder_name
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from transformers.pipelines.base import infer_framework_load_model


COMMIT_DESCRIPTION = """
This is an automated PR created with https://huggingface.co/spaces/safetensors/convert

This new file is equivalent to `pytorch_model.bin` but safe in the sense that
no arbitrary code can be put into it.

These files also happen to load much faster than their pytorch counterpart:
https://colab.research.google.com/github/huggingface/notebooks/blob/main/safetensors_doc/en/speed.ipynb

The widgets on your model page will run using this model even if this is not merged
making sure the file actually works.

If you find any issues: please report here: https://huggingface.co/spaces/safetensors/convert/discussions

Feel free to ignore this PR.
"""

ConversionResult = Tuple[List["CommitOperationAdd"], List[Tuple[str, "Exception"]]]


class AlreadyExists(Exception):
    pass


def shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for ptr, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )


def convert_file(
    pt_filename: str,
    sf_filename: str,
):
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared = shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    #os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


