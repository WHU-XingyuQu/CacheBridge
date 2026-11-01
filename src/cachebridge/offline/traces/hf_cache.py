"""Transformers cache construction used by the qualified fitter."""

import transformers
from transformers import DynamicCache


def dynamic_cache_from_data(data, *, config=None) -> DynamicCache:
    """Construct a populated cache across the Transformers 4.x/5.x APIs."""

    major = int(transformers.__version__.split(".", 1)[0])
    if major >= 5:
        return DynamicCache(data, config=config)
    cache = DynamicCache(data)
    # Transformers 4.54 initializes one empty DynamicLayer before appending
    # ddp_cache_data.  Retain only the populated layers so the materialized
    # cache has the same depth as the Transformers 5 reference path.
    cache.layers = [layer for layer in cache.layers if layer.keys is not None]
    return cache
