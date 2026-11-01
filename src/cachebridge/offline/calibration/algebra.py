"""Conditional attention calibration algebra; no autograd/optimizer."""

import torch


def probabilities(q, k, h, positions):
    # q [1,query_heads,Q,d], k [1,kv_heads,H+Q,d].
    heads, groups = k.shape[1], q.shape[1] // k.shape[1]
    picked = q[0, :, positions].float().reshape(heads, groups, len(positions), -1)
    scores = torch.einsum("hgqd,htd->hgqt", picked, k[0].float()) / q.shape[-1] ** 0.5
    visible = (
        torch.arange(k.shape[2], device=k.device)[None, :]
        <= (h + torch.tensor(positions, device=k.device))[:, None]
    )
    scores = scores.masked_fill(~visible[None, None], float("-inf"))
    return scores.softmax(-1).reshape(heads, groups * len(positions), k.shape[2])


def pool(p, values):
    return torch.bmm(p, values)


def gram(x, y, weight=None):
    if weight is None:
        return (
            x.transpose(-1, -2) @ x / x.shape[-2],
            x.transpose(-1, -2) @ y / x.shape[-2],
        )
    return x.transpose(-1, -2) @ (x * weight[..., None]), x.transpose(-1, -2) @ (
        y * weight[..., None]
    )


def solve(g, c, fraction=0.01):
    g = (g + g.transpose(-1, -2)) * 0.5
    lam = g.diagonal(dim1=-2, dim2=-1).mean(-1).mul(fraction).clamp_min(1e-8)
    reg = g + torch.diag_embed(lam[:, None].expand(-1, g.shape[-1]))
    b = torch.cholesky_solve(c, torch.linalg.cholesky(reg))
    residual = (
        (reg @ b - c).square().sum((-1, -2)) / c.square().sum((-1, -2)).clamp_min(1e-30)
    ).sqrt()
    return b, lam, residual
