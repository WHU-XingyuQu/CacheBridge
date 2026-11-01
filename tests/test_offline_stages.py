import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from cachebridge.architectures.pairs import get_pair
from cachebridge.offline.recipes import get_recipe
from cachebridge.offline.support import load_support


def test_pair_contract_rejects_same_family_with_different_geometry():
    pair = get_pair()
    config = dict(
        model_type="qwen3",
        num_hidden_layers=64,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        hidden_size=5120,
        rope_theta=1e6,
        rope_scaling=None,
        vocab_size=151936,
    )
    assert pair.validate(config, "target").num_layers == 64
    assert get_recipe().validate_target(config).num_layers == 64
    with pytest.raises(ValueError, match="num_hidden_layers"):
        pair.validate(dict(config, num_hidden_layers=40), "target")
    with pytest.raises(ValueError, match="recipe"):
        get_recipe("llama")


def test_support_loader_requires_complete_unique_valid_support(tmp_path):
    for layer in range(2):
        save_file(
            {"selected": torch.tensor([2, 0])},
            str(tmp_path / f"layer_{layer:03d}.safetensors"),
        )
    assert load_support(tmp_path, source_layers=3, target_layers=2, support_size=2) == (
        (2, 0),
        (2, 0),
    )
    for selected in ([2, 2], [3, 0], [0]):
        save_file(
            {"selected": torch.tensor(selected)},
            str(tmp_path / "layer_001.safetensors"),
        )
        with pytest.raises(ValueError, match="support"):
            load_support(tmp_path, source_layers=3, target_layers=2, support_size=2)
    (tmp_path / "layer_001.safetensors").unlink()
    with pytest.raises(ValueError, match="complete"):
        load_support(tmp_path, source_layers=3, target_layers=2)
