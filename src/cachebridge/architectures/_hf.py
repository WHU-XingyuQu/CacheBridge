"""Layer execution across the supported Transformers cache interfaces."""

import inspect
from importlib import import_module

import torch


def backbone(model):
    for candidate in (
        model,
        getattr(model, "text_model", None),
        getattr(model, "model", None),
    ):
        if candidate is not None and all(
            hasattr(candidate, name)
            for name in ("layers", "embed_tokens", "rotary_emb", "norm")
        ):
            return candidate
        language = getattr(candidate, "language_model", None)
        if language is not None and hasattr(language, "layers"):
            return language
    raise TypeError("Expected a causal language model or its text backbone")


def new_cache(config):
    from transformers import DynamicCache

    if "config" in inspect.signature(DynamicCache).parameters:
        return DynamicCache(config=config)
    return DynamicCache()


def prepare(model, ids, history, family):
    target = backbone(model)
    if getattr(target.config, "sliding_window", None) is not None or any(
        kind != "full_attention"
        for kind in (getattr(target.config, "layer_types", None) or [])
    ):
        raise ValueError("Layerwise execution requires full-attention history caches")
    module = import_module(f"transformers.models.{family}.modeling_{family}")
    device = target.embed_tokens.weight.device
    hidden = target.embed_tokens(torch.tensor([ids], device=device))
    pos = torch.arange(history, history + len(ids), device=device)
    cache = new_cache(target.config)
    dummy = torch.empty(1, 1, history, 1, device=device, dtype=hidden.dtype)
    cache.update(dummy, dummy, 0)
    mask_parameters = inspect.signature(module.create_causal_mask).parameters
    embedding_name = (
        "inputs_embeds" if "inputs_embeds" in mask_parameters else "input_embeds"
    )
    if embedding_name not in mask_parameters:
        raise TypeError("Causal-mask function exposes no supported embedding argument")
    mask_arguments = dict(
        config=target.config,
        attention_mask=torch.ones(
            1, history + len(ids), dtype=torch.long, device=device
        ),
        past_key_values=cache,
        position_ids=pos[None],
    )
    mask_arguments[embedding_name] = hidden
    if "cache_position" in mask_parameters:
        mask_arguments["cache_position"] = pos
    mask = module.create_causal_mask(**mask_arguments)
    rope = target.rotary_emb(hidden, pos[None])
    return hidden.cpu(), dict(
        pos=pos.cpu(),
        mask=None if mask is None else mask.cpu(),
        rope=tuple(x.cpu() for x in rope),
    )


def step(model, layer, hidden, history_kv, metadata):
    target = backbone(model)
    module = target.layers[layer]
    device = next(module.parameters()).device
    cache = new_cache(target.config)
    key, value = history_kv
    cache.update(key.to(device), value.to(device), layer)
    parameters = inspect.signature(module.forward).parameters
    if "past_key_values" in parameters:
        cache_argument = "past_key_values"
    elif "past_key_value" in parameters:
        cache_argument = "past_key_value"
    else:
        raise TypeError("Decoder layer exposes no supported cache argument")
    pos = metadata["pos"].to(device)
    mask = metadata["mask"]
    arguments = {cache_argument: cache}
    if "cache_position" in parameters:
        arguments["cache_position"] = pos
    result = module(
        hidden.to(device),
        attention_mask=None if mask is None else mask.to(device),
        position_ids=pos[None],
        use_cache=True,
        position_embeddings=tuple(x.to(device) for x in metadata["rope"]),
        **arguments,
    )
    if isinstance(result, tuple):
        result = result[0]
    if hasattr(cache, "layers"):
        item = cache.layers[layer]
        kv = item.keys, item.values
    else:
        kv = cache[layer]
    return result.cpu(), tuple(x.cpu() for x in kv)


def logits(model, hidden):
    norm = backbone(model).norm
    value = norm(hidden.to(next(norm.parameters()).device))
    return (
        model.lm_head(value[:, -1:].to(model.lm_head.weight.device))[0, -1]
        .float()
        .cpu()
    )
