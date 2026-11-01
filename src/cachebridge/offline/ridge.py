"""Two-pass centered ridge algebra extracted from the research fitting path.

Call under torch.inference_mode(); ridge, dtype and chunk size are explicit.
"""

import time

import torch


def check_time(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(
            "cumulative process wall cap reached; preserve incomplete output"
        )


def design(source, local_head=None):
    # source [N,selected_layers,source_heads,source_dim]; Dense layer/head/dim order.
    if local_head is None:
        return source.reshape(source.shape[0], -1)
    return source[:, :, local_head, :].reshape(source.shape[0], -1)


def sufficient_stats(x, y, weights, *, device, dtype, chunk_rows, deadline=None):
    """Same two-pass centered arithmetic for every arm; bounded device staging."""
    if len(x) != len(y) or len(x) != len(weights) or not len(x):
        raise ValueError("observation count mismatch")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("invalid weights")
    if weights.sum() <= 0:
        raise ValueError("zero weight mass")
    n, p = x.shape
    q = y.shape[1]
    mx = torch.zeros(p, device=device, dtype=dtype)
    my = torch.zeros(q, device=device, dtype=dtype)
    mass = torch.zeros((), device=device, dtype=dtype)
    for start in range(0, n, chunk_rows):
        check_time(deadline)
        stop = min(start + chunk_rows, n)
        a = x[start:stop].to(device=device, dtype=dtype)
        b = y[start:stop].to(device=device, dtype=dtype)
        w = weights[start:stop].to(device=device, dtype=dtype)
        mx.add_(w @ a)
        my.add_(w @ b)
        mass.add_(w.sum())
    mx.div_(mass)
    my.div_(mass)
    gram = torch.zeros((p, p), device=device, dtype=dtype)
    cross = torch.zeros((p, q), device=device, dtype=dtype)
    for start in range(0, n, chunk_rows):
        check_time(deadline)
        stop = min(start + chunk_rows, n)
        a = x[start:stop].to(device=device, dtype=dtype) - mx
        b = y[start:stop].to(device=device, dtype=dtype) - my
        w = weights[start:stop].to(device=device, dtype=dtype)
        gram.addmm_(a.T, w[:, None] * a)
        cross.addmm_(a.T, w[:, None] * b)
    # Explicit symmetric system for common solver; no lambda-dependent adjustment.
    gram = (gram + gram.T) * 0.5
    if not all(torch.isfinite(t).all() for t in (gram, cross, mx, my)):
        raise ValueError("nonfinite sufficient statistics")
    return dict(gram=gram, cross=cross, mean_x=mx, mean_y=my, mass=mass)


def solve_stats(stats, ridge):
    a = stats["gram"].clone()
    a.diagonal().add_(ridge)
    chol, info = torch.linalg.cholesky_ex(a)
    if int(info.item()) == 0:
        w = torch.cholesky_solve(stats["cross"], chol)
        solver = "cholesky"
    else:
        # Preserve failed Cholesky as a numerical diagnostic; never increase lambda.
        try:
            w = torch.linalg.solve(a, stats["cross"])
        except RuntimeError as exc:
            raise RuntimeError(
                f"ridge={ridge}, p={len(a)}, cholesky_info={int(info.item())}; LU failed"
            ) from exc
        solver = "lu_after_cholesky_failure"
    b = stats["mean_y"] - stats["mean_x"] @ w
    if not torch.isfinite(w).all() or not torch.isfinite(b).all():
        raise ValueError("nonfinite ridge solution")
    residual = (a @ w - stats["cross"]).norm() / stats["cross"].norm().clamp_min(1e-30)
    return (
        w,
        b,
        dict(
            solver=solver,
            cholesky_info=int(info.item()),
            linear_system_relative_residual=float(residual.item()),
        ),
    )
