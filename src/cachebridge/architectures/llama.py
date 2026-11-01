"""Llama dense-attention geometry and layerwise query execution."""

from . import _hf
from .base import describe_attention

CAPABILITIES = ("description", "layer_execution")


def describe(config):
    return describe_attention(config, "llama", ("llama",))


def prepare(model, ids, history):
    return _hf.prepare(model, ids, history, "llama")


def step(model, layer, hidden, history_kv, metadata):
    return _hf.step(model, layer, hidden, history_kv, metadata)


def logits(model, hidden):
    return _hf.logits(model, hidden)
