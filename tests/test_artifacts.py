"""An incomplete or modified mapper must fail before use."""

import json
import torch
import pytest
from safetensors.torch import save_file
from cachebridge.artifacts import file_sha256, verify_descriptor


def fixture(tmp_path):
    values = dict(
        selected=torch.tensor([0]),
        headwise=torch.tensor(True),
        k_weight=torch.eye(2)[None],
        v_weight=torch.eye(2)[None],
        k_bias=torch.zeros(1, 2),
        v_bias=torch.zeros(1, 2),
    )
    p = tmp_path / "layer_000.safetensors"
    save_file(values, str(p))
    d = dict(
        mapper_id="test",
        mapper_dir=".",
        target_layers=1,
        source_layers=1,
        kv_heads=1,
        head_dim=2,
        logical_parameter_bytes=48,
        file_hashes={p.name: file_sha256(p)},
    )
    manifest = tmp_path / "mapper.json"
    manifest.write_text(json.dumps(d))
    return manifest, p, d


def test_verified_portable_descriptor(tmp_path):
    manifest, _, _ = fixture(tmp_path)
    assert verify_descriptor(manifest)["layers"] == 1


def test_changed_coefficients_rejected(tmp_path):
    manifest, path, _ = fixture(tmp_path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_descriptor(manifest)


def test_missing_layer_rejected(tmp_path):
    manifest, path, _ = fixture(tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match="Missing"):
        verify_descriptor(manifest)


def test_extra_layer_rejected(tmp_path):
    manifest, path, _ = fixture(tmp_path)
    (tmp_path / "layer_001.safetensors").write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="Missing"):
        verify_descriptor(manifest)
