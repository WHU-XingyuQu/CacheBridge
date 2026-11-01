"""Model structure and continuation-state contracts shared by fitting and serving."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

LayerKind = Literal["full_attention", "sliding_attention", "linear_attention"]
_MISSING = object()


def config_value(config, name, default=_MISSING):
    value = (
        config.get(name, default)
        if isinstance(config, Mapping)
        else getattr(config, name, default)
    )
    if value is _MISSING:
        raise ValueError(f"Missing model configuration field: {name}")
    return value


def text_config(config):
    """Accept either a text configuration or a multimodal configuration wrapper."""
    return config_value(config, "text_config", None) or config


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class LayerSpec:
    index: int
    kind: LayerKind
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    query_heads: int | None = None
    sliding_window: int | None = None
    conv_kernel_size: int | None = None

    def __post_init__(self):
        if type(self.index) is not int or self.index < 0:
            raise ValueError("Layer index must be nonnegative")
        for name in ("key_heads", "value_heads", "key_dim", "value_dim"):
            positive_int(getattr(self, name), name)
        if self.kind == "linear_attention":
            positive_int(self.conv_kernel_size, "conv_kernel_size")
            if self.value_heads % self.key_heads:
                raise ValueError("Recurrent value heads must be divisible by key heads")
            if self.query_heads is not None or self.sliding_window is not None:
                raise ValueError(
                    "Linear attention does not use token-attention geometry"
                )
        elif self.kind in ("full_attention", "sliding_attention"):
            positive_int(self.query_heads, "query_heads")
            if self.key_heads != self.value_heads or self.query_heads % self.key_heads:
                raise ValueError("Invalid grouped-query attention head geometry")
            if self.conv_kernel_size is not None:
                raise ValueError("Token attention has no convolution state")
            if self.kind == "sliding_attention":
                if positive_int(self.sliding_window, "sliding_window") < 2:
                    raise ValueError(
                        "Sliding window must leave room for the next token"
                    )
            elif self.sliding_window is not None:
                raise ValueError("Full attention has no sliding window")
        else:
            raise ValueError(f"Unsupported layer kind: {self.kind}")

    @property
    def state_names(self):
        return (
            ("recurrent_state", "conv_state")
            if self.kind == "linear_attention"
            else ("keys", "values")
        )

    @property
    def conv_channels(self):
        if self.kind != "linear_attention":
            raise ValueError("Convolution channels require a recurrent layer")
        return 2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim


@dataclass(frozen=True)
class ArchitectureSpec:
    family: str
    hidden_size: int
    layers: tuple[LayerSpec, ...]
    position_encoding: Mapping[str, Any]

    def __post_init__(self):
        positive_int(self.hidden_size, "hidden_size")
        if not self.layers or tuple(layer.index for layer in self.layers) != tuple(
            range(len(self.layers))
        ):
            raise ValueError("Layers must form a complete ordered roster")

    @property
    def num_layers(self):
        return len(self.layers)

    def compatible_source_layers(self, target_layer: LayerSpec):
        """Structural candidates; numerical support selection remains offline."""
        return tuple(
            layer.index for layer in self.layers if layer.kind == target_layer.kind
        )


def head_assignment(
    source_heads, target_heads, assignment: Sequence[int] | None = None
):
    """Use identity for equal head counts; require explicit ownership otherwise."""
    positive_int(source_heads, "source_heads")
    positive_int(target_heads, "target_heads")
    if assignment is None:
        if source_heads != target_heads:
            raise ValueError(
                "Unequal KV-head counts require an explicit head assignment"
            )
        return tuple(range(target_heads))
    values = tuple(assignment)
    if len(values) != target_heads or any(
        type(h) is not int or not 0 <= h < source_heads for h in values
    ):
        raise ValueError(
            "Head assignment must identify a valid source head for each target head"
        )
    return values


def describe_attention(config, family, model_types, *, layer_types=None):
    config = text_config(config)
    if config_value(config, "model_type") not in model_types:
        raise ValueError(f"Expected {family} text configuration")
    depth = positive_int(config_value(config, "num_hidden_layers"), "num_hidden_layers")
    queries = positive_int(
        config_value(config, "num_attention_heads"), "num_attention_heads"
    )
    heads = positive_int(
        config_value(config, "num_key_value_heads", queries), "num_key_value_heads"
    )
    hidden = positive_int(config_value(config, "hidden_size"), "hidden_size")
    dim = config_value(config, "head_dim", None)
    if dim is None:
        if hidden % queries:
            raise ValueError(
                "head_dim is required when hidden_size is not divisible by query heads"
            )
        dim = hidden // queries
    positive_int(dim, "head_dim")
    kinds = (
        tuple(layer_types)
        if layer_types is not None
        else tuple(
            config_value(config, "layer_types", None) or ["full_attention"] * depth
        )
    )
    if len(kinds) != depth:
        raise ValueError("layer_types must cover every layer")
    layers = tuple(
        LayerSpec(
            i,
            kind,
            heads,
            heads,
            dim,
            dim,
            queries,
            (
                config_value(config, "sliding_window")
                if kind == "sliding_attention"
                else None
            ),
        )
        for i, kind in enumerate(kinds)
    )
    return ArchitectureSpec(family, hidden, layers, position_encoding(config))


def position_encoding(config):
    """Keep native positional parameters; adapters apply model-specific rotations."""
    return {
        name: config_value(config, name)
        for name in (
            "rope_theta",
            "rope_scaling",
            "rope_parameters",
            "rope_local_base_freq",
            "partial_rotary_factor",
        )
        if config_value(config, name, None) is not None
    }
