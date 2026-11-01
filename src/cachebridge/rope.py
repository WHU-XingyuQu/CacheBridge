from __future__ import annotations

import math

import torch
from torch import Tensor


def rotate_half(value: Tensor) -> Tensor:
    if value.shape[-1] % 2:
        raise ValueError("RoPE head dimension must be even")
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def qwen_rope(
    positions: Tensor,
    *,
    head_dim: int,
    theta: float,
    scaling: dict | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Return split-half Qwen RoPE values as ``[tokens, head_dim]``."""

    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    frequency_ids = torch.arange(0, head_dim, 2, device=positions.device).float()
    inverse_frequency = 1.0 / (theta ** (frequency_ids / head_dim))
    magnitude = 1.0
    if scaling is not None:
        rope_type = scaling.get("rope_type", scaling.get("type"))
        if rope_type != "yarn":
            raise ValueError(f"unsupported RoPE scaling type: {rope_type!r}")
        factor = float(scaling["factor"])
        original_maximum = int(scaling["original_max_position_embeddings"])
        beta_fast = float(scaling.get("beta_fast", 32))
        beta_slow = float(scaling.get("beta_slow", 1))
        extrapolation = float(scaling.get("extrapolation_factor", 1))
        attention_factor = float(scaling.get("attn_factor", 1))
        if factor <= 1 or original_maximum <= 0:
            raise ValueError(
                "YaRN factor must exceed one and original window must be positive"
            )

        def correction_dimension(rotations: float) -> float:
            return (
                head_dim
                * math.log(original_maximum / (rotations * 2 * math.pi))
                / (2 * math.log(theta))
            )

        low = max(math.floor(correction_dimension(beta_fast)), 0)
        high = min(math.ceil(correction_dimension(beta_slow)), head_dim - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp(
            (torch.arange(head_dim // 2, device=positions.device).float() - low)
            / (high - low),
            0,
            1,
        )
        extrapolation_mask = (1 - ramp) * extrapolation
        interpolated = inverse_frequency / factor
        inverse_frequency = (
            interpolated * (1 - extrapolation_mask)
            + inverse_frequency * extrapolation_mask
        )
        magnitude = (0.1 * math.log(factor) + 1.0) * attention_factor
    phase = positions.float().unsqueeze(-1) * inverse_frequency.unsqueeze(0)
    phase = torch.cat((phase, phase), dim=-1)
    return (phase.cos() * magnitude).to(dtype), (phase.sin() * magnitude).to(dtype)


def apply_rope(value: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rotation_shape = (1, cos.shape[0], 1, cos.shape[1])
    cosine = cos.reshape(rotation_shape).to(value)
    sine = sin.reshape(rotation_shape).to(value)
    return value * cosine + rotate_half(value) * sine


def remove_rope(value: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rotation_shape = (1, cos.shape[0], 1, cos.shape[1])
    cosine = cos.reshape(rotation_shape).to(value)
    sine = sin.reshape(rotation_shape).to(value)
    scale_squared = cosine.square() + sine.square()
    return (value * cosine - rotate_half(value) * sine) / scale_squared.clamp_min(
        torch.finfo(value.dtype).tiny
    )
