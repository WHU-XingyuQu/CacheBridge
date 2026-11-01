"""Exact weighted centered ridge initializer; fixed head-local support."""

import torch
from safetensors.torch import load_file, save_file
from ..rope import qwen_rope, remove_rope
from .trace_io import read_cache
from . import io as b


def positions(h, n):
    assert 4 < n <= h
    # Preserve the initially omitted prefix while keeping the total budget.
    tail = (
        4
        + ((torch.arange(n - 4, dtype=torch.float64) + 0.5) * (h - 4) / (n - 4))
        .floor()
        .long()
    )
    result = torch.cat((torch.arange(4), tail))
    assert len(result.unique()) == n and int(result.max()) < h
    return result


@torch.inference_mode()
def fit(source, target, base, weights=None):
    selected = base["selected"].long()
    x = (
        source[selected]
        .permute(2, 1, 0, 3)
        .reshape(8, source.shape[1], -1)
        .to("cuda:0", torch.float64)
    )
    y = target.permute(1, 0, 2).to("cuda:0", torch.float64)
    if weights is None:
        mx, my = x.mean(1), y.mean(1)
        x -= mx[:, None]
        y -= my[:, None]
    else:
        w = weights.to(device="cuda:0", dtype=torch.float64)
        assert w.shape == (source.shape[1],) and bool((w > 0).all())
        mx = (x * w[None, :, None]).sum(1) / w.sum()
        my = (y * w[None, :, None]).sum(1) / w.sum()
        x = (x - mx[:, None]) * w.sqrt()[None, :, None]
        y = (y - my[:, None]) * w.sqrt()[None, :, None]
    gram, cross = x.transpose(1, 2) @ x, x.transpose(1, 2) @ y
    system = gram.clone()
    system.diagonal(dim1=-2, dim2=-1).add_(1.0)
    chol, info = torch.linalg.cholesky_ex(system)
    assert not info.any(), info
    weight = torch.cholesky_solve(cross, chol)
    bias = my - torch.einsum("hf,hfd->hd", mx, weight)
    residual = (
        (system @ weight - cross).square().sum((-1, -2))
        / cross.square().sum((-1, -2)).clamp_min(1e-30)
    ).sqrt()
    assert torch.isfinite(weight).all() and float(residual.max()) < 1e-7
    values = {k: v.clone() for k, v in base.items()}
    values["k_weight"] = weight.to(
        dtype=base["k_weight"].dtype, device="cpu"
    ).contiguous()
    values["k_bias"] = bias.to(dtype=base["k_bias"].dtype, device="cpu").contiguous()
    stats = dict(G=gram.cpu(), C=cross.cpu(), mean_x=mx.cpu(), mean_y=my.cpu())
    return values, stats, float(residual.max())


def error(source, target, values):
    selected = values["selected"].long()
    x = (
        source[selected]
        .permute(2, 1, 0, 3)
        .reshape(8, source.shape[1], -1)
        .to("cuda:0")
    )
    y = target.permute(1, 0, 2).to("cuda:0")
    prediction = (
        x @ values["k_weight"].to("cuda:0").float()
        + values["k_bias"].to("cuda:0").float()[:, None]
    )
    return float((prediction - y).square().sum()), float(y.square().sum())
