from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file
from torch import Tensor

from .rope import apply_rope, qwen_rope, remove_rope


Method = Literal["nv_transfer", "cachebridge"]


@dataclass(frozen=True)
class LayerWeights:
    selected: Tensor
    k_weight: Tensor
    k_bias: Tensor
    v_weight: Tensor
    v_bias: Tensor
    headwise: bool


class PageMapper:
    """Apply existing Dense or CacheBridge mapper files to SGLang pages.

    Input and output use the canonical page/prefix layout
    ``[2, layers, tokens, kv_heads, head_dim]``. SGLang stores RoPE-applied
    keys, while both fitted mapper families consume content-space keys.
    """

    def __init__(
        self,
        mapper_dir: str | Path,
        *,
        method: Method,
        source_rope_theta: float,
        target_rope_theta: float,
        source_rope_scaling: dict | None = None,
        target_rope_scaling: dict | None = None,
        device: str | torch.device = "cpu",
        target_tp_rank: int = 0,
        target_tp_size: int = 1,
    ) -> None:
        self.mapper_dir = Path(mapper_dir)
        self.method = method
        self.source_rope_theta = float(source_rope_theta)
        self.target_rope_theta = float(target_rope_theta)
        self.source_rope_scaling = source_rope_scaling
        self.target_rope_scaling = target_rope_scaling
        self.device = torch.device(device)
        self.target_tp_rank = int(target_tp_rank)
        self.target_tp_size = int(target_tp_size)
        self.target_head_start: int | None = None
        self.target_head_stop: int | None = None
        paths = sorted(self.mapper_dir.glob("layer_*.safetensors"))
        if not paths:
            raise FileNotFoundError(f"no layer_*.safetensors in {self.mapper_dir}")
        self.layers: list[LayerWeights] = []
        for path in paths:
            tensors = load_file(path, device=str(self.device))
            required = {"selected", "k_weight", "k_bias", "v_weight", "v_bias"}
            missing = required.difference(tensors)
            if missing:
                raise ValueError(f"{path} is missing tensors: {sorted(missing)}")
            headwise = "headwise" in tensors
            expected_headwise = method == "cachebridge"
            if headwise != expected_headwise:
                raise ValueError(
                    f"{path} headwise={headwise} does not match method={method}"
                )
            target_heads = int(tensors["k_bias"].shape[0])
            if target_heads % self.target_tp_size:
                raise ValueError(
                    f"target KV heads ({target_heads}) must divide target TP "
                    f"({self.target_tp_size})"
                )
            heads_per_rank = target_heads // self.target_tp_size
            head_start = self.target_tp_rank * heads_per_rank
            head_stop = head_start + heads_per_rank
            if self.target_head_start is None:
                self.target_head_start = head_start
                self.target_head_stop = head_stop
            elif (self.target_head_start, self.target_head_stop) != (
                head_start,
                head_stop,
            ):
                raise ValueError("target-head geometry changes across mapper layers")
            head_slice = slice(head_start, head_stop)
            self.layers.append(
                LayerWeights(
                    selected=tensors["selected"].long().cpu(),
                    k_weight=(
                        tensors["k_weight"][head_slice]
                        if headwise
                        else tensors["k_weight"][:, head_slice]
                    ).contiguous(),
                    k_bias=tensors["k_bias"][head_slice].contiguous(),
                    v_weight=(
                        tensors["v_weight"][head_slice]
                        if headwise
                        else tensors["v_weight"][:, head_slice]
                    ).contiguous(),
                    v_bias=tensors["v_bias"][head_slice].contiguous(),
                    headwise=headwise,
                )
            )

    @property
    def target_layers(self) -> int:
        return len(self.layers)

    @torch.inference_mode()
    def map_page(
        self,
        source_page: Tensor,
        *,
        position_start: int,
        source_head_offset: int = 0,
        output_device: str | torch.device = "cpu",
        output: Tensor | None = None,
    ) -> Tensor:
        if source_page.ndim != 5 or source_page.shape[0] != 2:
            raise ValueError("source page must be [2,layers,tokens,kv_heads,head_dim]")
        source_page = source_page.to(self.device)
        source_dtype = source_page.dtype
        _, source_layers, page_size, source_heads, head_dim = source_page.shape
        positions = torch.arange(
            position_start,
            position_start + page_size,
            device=self.device,
            dtype=torch.long,
        )
        source_cos, source_sin = qwen_rope(
            positions,
            head_dim=head_dim,
            theta=self.source_rope_theta,
            scaling=self.source_rope_scaling,
        )
        source_keys_rotated = source_page[0]
        source_keys = remove_rope(source_keys_rotated.float(), source_cos, source_sin)
        source_values = source_page[1].float()

        mapped_keys: list[Tensor] = []
        mapped_values: list[Tensor] = []
        for layer in self.layers:
            selected = layer.selected.tolist()
            if not selected or max(selected) >= source_layers:
                raise ValueError("mapper selected layer exceeds source page depth")
            if layer.headwise:
                key_features = torch.cat(
                    [source_keys[index] for index in selected], dim=-1
                )
                value_features = torch.cat(
                    [source_values[index] for index in selected], dim=-1
                )
                key_features = key_features[
                    :,
                    self.target_head_start
                    - source_head_offset : self.target_head_stop
                    - source_head_offset,
                ]
                value_features = value_features[
                    :,
                    self.target_head_start
                    - source_head_offset : self.target_head_stop
                    - source_head_offset,
                ]
                if key_features.shape[1] != layer.k_weight.shape[0]:
                    raise ValueError("CacheBridge source-head geometry mismatch")
                mapped_key = (
                    torch.einsum("thf,hfd->thd", key_features, layer.k_weight.float())
                    + layer.k_bias.float()
                )
                mapped_value = (
                    torch.einsum("thf,hfd->thd", value_features, layer.v_weight.float())
                    + layer.v_bias.float()
                )
            else:
                key_features = torch.cat(
                    [source_keys[index].reshape(page_size, -1) for index in selected],
                    dim=-1,
                )
                value_features = torch.cat(
                    [source_values[index].reshape(page_size, -1) for index in selected],
                    dim=-1,
                )
                if key_features.shape[-1] != layer.k_weight.shape[0]:
                    raise ValueError("NV Dense feature-width mismatch")
                target_heads, target_dim = layer.k_bias.shape
                mapped_key = (
                    key_features
                    @ layer.k_weight.float().reshape(key_features.shape[-1], -1)
                    + layer.k_bias.float().reshape(-1)
                ).reshape(page_size, target_heads, target_dim)
                mapped_value = (
                    value_features
                    @ layer.v_weight.float().reshape(value_features.shape[-1], -1)
                    + layer.v_bias.float().reshape(-1)
                ).reshape(page_size, target_heads, target_dim)
            mapped_keys.append(mapped_key)
            mapped_values.append(mapped_value)

        target_keys = torch.stack(mapped_keys, dim=0)
        target_values = torch.stack(mapped_values, dim=0)
        target_cos, target_sin = qwen_rope(
            positions,
            head_dim=target_keys.shape[-1],
            theta=self.target_rope_theta,
            scaling=self.target_rope_scaling,
        )
        target_keys = apply_rope(target_keys, target_cos, target_sin)
        if output is None:
            return torch.stack((target_keys, target_values), dim=0).to(
                device=output_device, dtype=source_dtype
            )
        expected = (
            2,
            self.target_layers,
            page_size,
            target_keys.shape[2],
            target_keys.shape[3],
        )
        if (
            tuple(output.shape) != expected
            or output.dtype != source_dtype
            or output.device != torch.device(output_device)
        ):
            raise ValueError(
                "preallocated mapper output geometry/dtype/device mismatch"
            )
        # Cast directly into caller-owned storage. This is algebraically the same
        # final BF16 conversion as stack(...).to(...), without a second BF16 copy.
        output[0].copy_(target_keys)
        output[1].copy_(target_values)
        return output
