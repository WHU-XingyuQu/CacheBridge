"""Gemma 3 local/global attention layout and native cache installation."""

from dataclasses import dataclass

import torch

from .base import config_value, describe_attention, positive_int, text_config

CAPABILITIES = ("description", "window_materialization", "cache_installation")


def describe(config):
    cfg = text_config(config)
    kinds = config_value(cfg, "layer_types", None)
    if kinds is None:
        depth = positive_int(
            config_value(cfg, "num_hidden_layers"), "num_hidden_layers"
        )
        pattern = positive_int(
            config_value(cfg, "sliding_window_pattern", 6), "sliding_window_pattern"
        )
        kinds = [
            "full_attention" if (i + 1) % pattern == 0 else "sliding_attention"
            for i in range(depth)
        ]
    return describe_attention(
        cfg, "gemma3", ("gemma3_text", "gemma3"), layer_types=kinds
    )


@dataclass(frozen=True)
class MaterializedKV:
    key: torch.Tensor
    value: torch.Tensor
    first_position: int
    prefix_length: int


def materialize_kv(
    content_key, value, *, layer_type, sliding_window, prefix_length=None
):
    """Retain local/global content-space KV before applying target-layer RoPE."""
    if content_key.ndim != 3 or content_key.shape != value.shape:
        raise ValueError("Gemma K/V must have matching [tokens,heads,dim] shapes")
    tokens = content_key.shape[0]
    prefix_length = tokens if prefix_length is None else prefix_length
    if prefix_length != tokens or prefix_length < 1:
        raise ValueError("prefix length must equal the materialized token count")
    if layer_type == "sliding_attention":
        if positive_int(sliding_window, "sliding_window") < 2:
            raise ValueError("Sliding window must leave room for the next token")
        retained = min(tokens, sliding_window - 1)
    elif layer_type == "full_attention":
        retained = tokens
    else:
        raise ValueError(f"Unsupported Gemma layer type: {layer_type}")
    first = tokens - retained
    return MaterializedKV(content_key[first:], value[first:], first, prefix_length)


def install_kv(cache, *, mapped, prefix_length):
    """Install a complete roster of RoPE-applied KV into native cache layers."""
    positive_int(prefix_length, "prefix_length")
    if set(mapped) != set(range(len(cache.layers))):
        raise ValueError("Gemma payload must cover every target layer")
    # Validate all layers before mutating any cache state.
    for index, layer in enumerate(cache.layers):
        key, value = mapped[index]
        if (
            key.ndim != 4
            or key.shape != value.shape
            or key.dtype != value.dtype
            or key.device != value.device
        ):
            raise ValueError(
                "K/V must match [batch,heads,tokens,head_dim], dtype and device"
            )
        if not hasattr(layer, "update") or hasattr(layer, "recurrent_states"):
            raise TypeError("Expected an attention cache layer")
        if getattr(layer, "is_initialized", False):
            raise ValueError(f"Target cache layer {index} is already initialized")
        expected = (
            min(prefix_length, int(layer.sliding_window) - 1)
            if getattr(layer, "is_sliding", False)
            else prefix_length
        )
        if expected < 1 or key.shape[-2] != expected:
            raise ValueError(
                f"Invalid history length for layer {index}: expected {expected}"
            )
    for index, layer in enumerate(cache.layers):
        layer.update(*mapped[index])
        if getattr(layer, "is_sliding", False):
            layer.cumulative_length = prefix_length
