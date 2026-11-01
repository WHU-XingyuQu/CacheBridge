"""Load trusted, hash-verified, rank-sharded native KV traces."""

import torch


def read_cache(record, role):
    paths = sorted(
        [x for x in record["files"] if x["role"] == role], key=lambda x: x["rank"]
    )
    assert [x["rank"] for x in paths] == list(range(4))
    parts = [
        torch.load(x["path"], map_location="cpu", weights_only=True, mmap=True)
        for x in paths
    ]
    assert all(x.dtype == torch.bfloat16 and x.ndim == 5 for x in parts)
    cache = torch.cat(parts, dim=3)
    assert cache.shape[-2:] == (8, 128)
    return cache
