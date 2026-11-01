"""Lazy model-family adapters for structure, layer execution and cache state."""

from importlib import import_module

from .base import (
    ArchitectureSpec,
    LayerSpec,
    config_value,
    head_assignment,
    text_config,
)

_FAMILIES = {
    "qwen3": "qwen3",
    "llama": "llama",
    "ministral": "ministral",
    "ministral3": "ministral",
    "ministral3_text": "ministral",
    "gemma3": "gemma3",
    "gemma3_text": "gemma3",
    "qwen35": "qwen35",
    "qwen3_5": "qwen35",
    "qwen3_5_text": "qwen35",
}


def get_adapter(config_or_family):
    name = (
        config_or_family
        if isinstance(config_or_family, str)
        else config_value(text_config(config_or_family), "model_type")
    )
    if name not in _FAMILIES:
        raise ValueError(f"Unsupported architecture: {name}")
    return import_module(f"{__name__}.{_FAMILIES[name]}")


def describe(config):
    return get_adapter(config).describe(config)


__all__ = [
    "ArchitectureSpec",
    "LayerSpec",
    "describe",
    "get_adapter",
    "head_assignment",
]
