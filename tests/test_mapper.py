from pathlib import Path

from safetensors.torch import save_file
import torch

from cachebridge.mapper import PageMapper
from cachebridge.rope import apply_rope, qwen_rope


def _rotated_page() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    content = torch.randn(1, 3, 2, 4)
    values = torch.randn(1, 3, 2, 4)
    positions = torch.arange(3)
    cos, sin = qwen_rope(positions, head_dim=4, theta=10_000)
    rotated = apply_rope(content, cos, sin)
    page = torch.stack((rotated, values), dim=0)
    return page, torch.stack((rotated, values), dim=0)


def _write_dense(root: Path) -> None:
    weight = torch.zeros(8, 2, 4)
    for head in range(2):
        for dim in range(4):
            weight[head * 4 + dim, head, dim] = 1
    save_file(
        {
            "selected": torch.tensor([0]),
            "k_weight": weight,
            "k_bias": torch.zeros(2, 4),
            "v_weight": weight.clone(),
            "v_bias": torch.zeros(2, 4),
        },
        root / "layer_000.safetensors",
    )


def _write_headwise(root: Path) -> None:
    weight = torch.eye(4).expand(2, 4, 4).contiguous()
    save_file(
        {
            "selected": torch.tensor([0]),
            "k_weight": weight,
            "k_bias": torch.zeros(2, 4),
            "v_weight": weight.clone(),
            "v_bias": torch.zeros(2, 4),
            "headwise": torch.tensor(1, dtype=torch.int8),
        },
        root / "layer_000.safetensors",
    )


def test_dense_identity_page(tmp_path: Path):
    _write_dense(tmp_path)
    page, expected = _rotated_page()
    mapper = PageMapper(
        tmp_path,
        method="nv_transfer",
        source_rope_theta=10_000,
        target_rope_theta=10_000,
    )
    actual = mapper.map_page(page, position_start=0)
    assert torch.allclose(actual, expected, atol=2e-6)


def test_headwise_tp_slice(tmp_path: Path):
    _write_headwise(tmp_path)
    page, expected = _rotated_page()
    mapper = PageMapper(
        tmp_path,
        method="cachebridge",
        source_rope_theta=10_000,
        target_rope_theta=10_000,
        target_tp_rank=1,
        target_tp_size=2,
    )
    actual = mapper.map_page(page, position_start=0)
    assert actual.shape == (2, 1, 3, 1, 4)
    assert torch.allclose(actual, expected[:, :, :, 1:2], atol=2e-6)


def test_headwise_yarn_identity_page(tmp_path: Path):
    _write_headwise(tmp_path)
    scaling = {
        "rope_type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 32,
    }
    torch.manual_seed(11)
    content = torch.randn(1, 3, 2, 4)
    values = torch.randn(1, 3, 2, 4)
    positions = torch.arange(40, 43)
    cos, sin = qwen_rope(positions, head_dim=4, theta=10_000, scaling=scaling)
    rotated = apply_rope(content, cos, sin)
    page = torch.stack((rotated, values), dim=0)
    mapper = PageMapper(
        tmp_path,
        method="cachebridge",
        source_rope_theta=10_000,
        target_rope_theta=10_000,
        source_rope_scaling=scaling,
        target_rope_scaling=scaling,
    )
    actual = mapper.map_page(page, position_start=40)
    assert torch.allclose(actual, page, atol=2e-6)
