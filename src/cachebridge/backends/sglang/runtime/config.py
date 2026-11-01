"""Process-local bridge from the launcher to explicit runtime configuration."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from ..config import RuntimeConfig
from ....architectures.pairs import get_pair

PAIR = get_pair()
from ....artifacts import file_sha256 as sha256

MAX_RUNNING_REQUESTS = 16
BLOCK_TOKENS = 512
HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=8)
def _load(path):
    return RuntimeConfig.load(path)


def current():
    path = os.environ.get("CACHEBRIDGE_CONFIG")
    if not path:
        raise RuntimeError(
            "Use cachebridge serve --config PATH to configure the SGLang process"
        )
    return _load(str(Path(path).expanduser().resolve()))


def __getattr__(name):
    fields = {
        "SOURCE_MODEL": "source_model",
        "TARGET_MODEL": "target_model",
        "TARGET_TP": "tensor_parallel_size",
        "POOL_TOKENS": "pool_tokens",
        "SOURCE_PORT": "source_port",
        "TARGET_PORT": "target_port",
        "CONFIG_PATH": "config_path",
    }
    if name in fields:
        return getattr(current(), fields[name])
    raise AttributeError(name)


def identity(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def validate_model_pair():
    return current().validate_model_pair()


def artifact_contract():
    return current().artifact_contract(os.environ.get("CB_AGENT_MAPPER_DESCRIPTOR"))


def runtime_config():
    return current().runtime_parameters(os.environ.get("CB_AGENT_MAPPER_DESCRIPTOR"))
