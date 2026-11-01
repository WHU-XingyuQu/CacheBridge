"""Qwen3.5 attention, DeltaNet memory and convolution-state contracts."""

import torch

from .base import (
    ArchitectureSpec,
    LayerSpec,
    config_value,
    describe_attention,
    positive_int,
    text_config,
)

CAPABILITIES = ("description", "recurrent_mapping", "cache_installation")


def describe(config):
    cfg = text_config(config)
    depth = positive_int(config_value(cfg, "num_hidden_layers"), "num_hidden_layers")
    attention = describe_attention(
        cfg,
        "qwen35",
        ("qwen3_5_text", "qwen3_5"),
        layer_types=["full_attention"] * depth,
    )
    kinds = config_value(cfg, "layer_types")
    if len(kinds) != depth:
        raise ValueError("layer_types must cover every layer")
    layers = []
    for i, kind in enumerate(kinds):
        if kind == "full_attention":
            layers.append(attention.layers[i])
        elif kind == "linear_attention":
            layers.append(
                LayerSpec(
                    i,
                    kind,
                    config_value(cfg, "linear_num_key_heads"),
                    config_value(cfg, "linear_num_value_heads"),
                    config_value(cfg, "linear_key_head_dim"),
                    config_value(cfg, "linear_value_head_dim"),
                    conv_kernel_size=config_value(cfg, "linear_conv_kernel_dim"),
                )
            )
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer kind: {kind}")
    return ArchitectureSpec(
        "qwen35", attention.hidden_size, tuple(layers), attention.position_encoding
    )


def transform_delta_state(source_state, key_basis, value_basis):
    """Map recurrent memory as A^T U B, preserving batch/head dimensions."""
    if min(source_state.ndim, key_basis.ndim, value_basis.ndim) < 2:
        raise ValueError("Delta state and basis tensors must be matrix-valued")
    if (
        source_state.shape[-2] != key_basis.shape[-2]
        or source_state.shape[-1] != value_basis.shape[-2]
    ):
        raise ValueError("Source state dimensions do not match the key/value bases")
    return key_basis.transpose(-2, -1) @ source_state @ value_basis


def required_replay(conv_kernel_size):
    """Suffix length that replaces zero convolution history, not recurrent error."""
    return positive_int(conv_kernel_size, "conv_kernel_size") - 1


def install_attention_kv(cache, layer_index, key, value):
    if key.ndim != 4 or key.shape != value.shape or key.shape[-2] < 1:
        raise ValueError("K/V must match nonempty [batch,heads,tokens,head_dim]")
    if key.dtype != value.dtype or key.device != value.device:
        raise ValueError("K/V dtype and device must match")
    layer = cache.layers[layer_index]
    if (
        getattr(layer, "is_sliding", False)
        or hasattr(layer, "recurrent_states")
        or not hasattr(layer, "update")
    ):
        raise TypeError("Expected a full-attention cache layer")
    if getattr(layer, "is_initialized", False):
        raise ValueError("Target cache layer is already initialized")
    layer.update(key, value)


def install_linear_state(
    cache, layer_index, *, recurrent_state, conv_state, conv_kernel_size, spec=None
):
    """Install mapped memory and the receiver convolution tail into a fresh layer."""
    if recurrent_state.ndim != 4 or conv_state.ndim != 3:
        raise ValueError(
            "Expected recurrent [batch,heads,key,value] and convolution [batch,channels,kernel]"
        )
    positive_int(conv_kernel_size, "conv_kernel_size")
    if (
        conv_state.shape[-1] != conv_kernel_size
        or recurrent_state.shape[0] != conv_state.shape[0]
    ):
        raise ValueError("Convolution length or batch geometry mismatch")
    if recurrent_state.device != conv_state.device:
        raise ValueError("Recurrent and convolution states must share a device")
    if not recurrent_state.is_floating_point() or not conv_state.is_floating_point():
        raise ValueError(
            "Recurrent and convolution states must be floating-point tensors"
        )
    if spec is not None:
        if spec.index != layer_index or spec.kind != "linear_attention":
            raise ValueError(
                "Layer specification does not match the recurrent destination"
            )
        if recurrent_state.shape[1:] != (
            spec.value_heads,
            spec.key_dim,
            spec.value_dim,
        ):
            raise ValueError("Recurrent tensor does not match the layer specification")
        if (
            conv_state.shape[1:] != (spec.conv_channels, spec.conv_kernel_size)
            or conv_kernel_size != spec.conv_kernel_size
        ):
            raise ValueError(
                "Convolution tensor does not match the layer specification"
            )
    layer = cache.layers[layer_index]
    required = (
        "update_conv_state",
        "update_recurrent_state",
        "has_previous_state",
        "recurrent_states",
        "conv_states",
    )
    if any(not hasattr(layer, name) for name in required):
        raise TypeError("Target is not a linear-attention cache layer")
    if bool(layer.has_previous_state[0]):
        raise ValueError("Target cache layer already has previous state")
    layer.update_conv_state(conv_state, state_idx=0, conv_kernel_size=conv_kernel_size)
    layer.update_recurrent_state(recurrent_state, state_idx=0)
    if not bool(layer.has_previous_state[0]):
        raise RuntimeError("Linear cache did not enter previous-state mode")
