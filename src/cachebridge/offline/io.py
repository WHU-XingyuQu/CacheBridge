"""Explicit paths and immutable input identities for the offline recipe."""

import hashlib
import json
from pathlib import Path

from .recipes import get_recipe

ASSISTANT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def validate_target_model(path, *, recipe="qwen3_14b_32b_trajectory"):
    config = json.loads((Path(path) / "config.json").read_text())
    return get_recipe(recipe).validate_target(config)


def load_protocol(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    get_recipe(value.get("recipe", "qwen3_14b_32b_trajectory")).validate_protocol(value)
    for key in (
        "inputs",
        "input_review",
        "position_manifest",
        "initializer",
        "key_source",
        "current_mapper",
        "base_initializer",
        "target_model",
    ):
        if key in value:
            p = Path(value[key]).expanduser()
            value[key] = str(p if p.is_absolute() else (path.parent / p).resolve())
    value["files"] = {
        str(Path(k) if Path(k).is_absolute() else (path.parent / k).resolve()): v
        for k, v in value.get("files", {}).items()
    }
    return value
