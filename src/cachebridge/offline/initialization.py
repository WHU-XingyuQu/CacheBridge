"""Fit a single head-local initializer from paired content-space KV traces.

This is the initialization stage, not a complete calibrated CacheBridge mapper.
The caller supplies the source-layer support and ridge; neither is tuned here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from .ridge import sufficient_stats, solve_stats


@torch.inference_mode()
def fit_layer(
    source: Tensor,
    target: Tensor,
    *,
    selected: Sequence[int],
    ridge: float,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    chunk_rows: int = 2048,
) -> tuple[dict[str, Tensor], list[dict]]:
    """Return tensors for one ``layer_NNN.safetensors`` and solve diagnostics.

    ``source`` has shape [2, source_layers, observations, heads, source_dim].
    ``target`` has shape [2, observations, heads, target_dim] for ONE target layer.
    Keys must already have source/target RoPE removed. Observations must be paired
    by input and absolute position. Current head-local fitting requires equal
    source/target KV-head counts and preserves head identity.

    The default FP32, two-pass centered solve follows the extracted initializer.
    Calibration inputs, selection provenance, and TF32 policy belong to the
    calling recipe. CPU tensors may be staged in chunks to the requested device.
    """
    if source.ndim != 5 or source.shape[0] != 2:
        raise ValueError("source must be [2, layers, observations, heads, dim]")
    if target.ndim != 4 or target.shape[0] != 2:
        raise ValueError("target must be [2, observations, heads, dim]")
    if source.shape[2:4] != target.shape[1:3]:
        raise ValueError("paired observation and head counts must match")
    if any(n <= 0 for n in source.shape + target.shape):
        raise ValueError("trace dimensions must be positive")
    if not source.is_floating_point() or not target.is_floating_point():
        raise ValueError("traces must use floating-point tensors")
    if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("traces must be finite")
    selected = tuple(selected)
    if (
        not selected
        or any(type(i) is not int or not 0 <= i < source.shape[1] for i in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("selected must contain unique, valid source-layer indices")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("statistics must use float32 or float64")

    observations, heads = source.shape[2:4]
    weights = torch.ones(observations, dtype=dtype)
    tensors = {
        "selected": torch.tensor(selected, dtype=torch.int64),
        "headwise": torch.tensor(True),
    }
    diagnostics = []
    for component_index, component in enumerate(("k", "v")):
        coefficients, biases = [], []
        for head in range(heads):
            x = torch.cat(
                [source[component_index, layer, :, head] for layer in selected],
                dim=-1,
            )
            y = target[component_index, :, head]
            stats = sufficient_stats(
                x, y, weights, device=device, dtype=dtype, chunk_rows=chunk_rows
            )
            weight, bias, diagnostic = solve_stats(stats, ridge)
            coefficients.append(weight.cpu().float())
            biases.append(bias.cpu().float())
            diagnostics.append(dict(component=component, head=head, **diagnostic))
        tensors[f"{component}_weight"] = torch.stack(coefficients).contiguous()
        tensors[f"{component}_bias"] = torch.stack(biases).contiguous()
    return tensors, diagnostics
