"""Ministral 3 dense-attention geometry and layerwise query execution."""

from . import _hf
from .base import config_value, describe_attention, text_config

CAPABILITIES = ("description", "layer_execution")


def describe(config):
    config = text_config(config)
    kinds = config_value(config, "layer_types", None)
    if kinds is None and config_value(config, "sliding_window", None) is not None:
        kinds = ["sliding_attention"] * config_value(config, "num_hidden_layers")
    return describe_attention(
        config, "ministral", ("ministral3", "ministral3_text"), layer_types=kinds
    )


def prepare(model, ids, history):
    return _hf.prepare(model, ids, history, "ministral3")


def step(model, layer, hidden, history_kv, metadata):
    return _hf.step(model, layer, hidden, history_kv, metadata)


def logits(model, hidden):
    return _hf.logits(model, hidden)
