"""Qwen3 dense-attention geometry and layerwise query execution."""

import torch
from .base import describe_attention

CAPABILITIES = ("description", "layer_execution")


def describe(config):
    return describe_attention(config, "qwen3", ("qwen3",))


def prepare(target, ids, history):
    from transformers import DynamicCache
    from transformers.models.qwen3.modeling_qwen3 import create_causal_mask

    device = target.model.embed_tokens.weight.device
    hidden = target.model.embed_tokens(torch.tensor([ids], device=device))
    pos = torch.arange(history, history + len(ids), device=device)
    # Only cache length is needed to construct the same HF causal mask.
    cache = DynamicCache()
    dummy = torch.empty(1, 1, history, 1, device=device, dtype=hidden.dtype)
    cache.update(dummy, dummy, 0)
    mask = create_causal_mask(
        config=target.config,
        input_embeds=hidden,
        attention_mask=torch.ones(
            1, history + len(ids), dtype=torch.long, device=device
        ),
        cache_position=pos,
        past_key_values=cache,
        position_ids=pos[None],
    )
    rope = target.model.rotary_emb(hidden, pos[None])
    return hidden.cpu(), dict(
        pos=pos.cpu(),
        mask=None if mask is None else mask.cpu(),
        rope=tuple(x.cpu() for x in rope),
    )


def step(target, layer, hidden, history_kv, metadata):
    from transformers import DynamicCache

    module = target.model.layers[layer]
    device = next(module.parameters()).device
    cache = DynamicCache()
    k, v = history_kv
    cache.update(k.to(device), v.to(device), layer)
    pos = metadata["pos"].to(device)
    mask = metadata["mask"]
    result = module(
        hidden.to(device),
        attention_mask=None if mask is None else mask.to(device),
        position_ids=pos[None],
        past_key_value=cache,
        use_cache=True,
        cache_position=pos,
        position_embeddings=tuple(x.to(device) for x in metadata["rope"]),
    )
    return result.cpu(), tuple(x.cpu() for x in cache[layer])


def logits(target, hidden):
    norm = target.model.norm
    value = norm(hidden.to(next(norm.parameters()).device))
    return (
        target.lm_head(value[:, -1:].to(target.lm_head.weight.device))[0, -1]
        .float()
        .cpu()
    )
