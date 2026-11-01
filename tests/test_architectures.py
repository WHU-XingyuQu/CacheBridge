"""Architecture selection, continuation-state geometry and native cache contracts."""

from types import SimpleNamespace

import pytest
import torch

from cachebridge.architectures import describe, get_adapter, head_assignment
from cachebridge.architectures import gemma3, qwen35


def dense_config(family="qwen3", **changes):
    return dict(
        model_type=family,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        **changes
    )


def hybrid_config():
    return dict(
        dense_config("qwen3_5_text"),
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=3,
        linear_value_head_dim=5,
        linear_conv_kernel_dim=4,
        rope_parameters=dict(rope_theta=10000000, partial_rotary_factor=0.25),
    )


@pytest.mark.parametrize(
    "family,expected",
    [("qwen3", "qwen3"), ("llama", "llama"), ("ministral3", "ministral")],
)
def test_dense_structure_from_wrapped_configuration(family, expected):
    spec = describe(
        SimpleNamespace(text_config=SimpleNamespace(**dense_config(family)))
    )
    assert spec.family == expected and spec.num_layers == 4
    assert all(layer.state_names == ("keys", "values") for layer in spec.layers)
    assert spec.layers[0].key_dim == 8


def test_layer_type_matching_does_not_mix_local_global_and_recurrent():
    cfg = dense_config(
        "gemma3_text",
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"] * 2,
    )
    gemma = describe(cfg)
    hybrid = describe(hybrid_config())
    assert gemma.compatible_source_layers(gemma.layers[0]) == (0, 2)
    assert gemma.compatible_source_layers(gemma.layers[1]) == (1, 3)
    assert gemma.compatible_source_layers(hybrid.layers[0]) == ()
    assert hybrid.layers[0].state_names == ("recurrent_state", "conv_state")
    assert hybrid.layers[0].conv_channels == 16
    assert hybrid.position_encoding["rope_parameters"]["partial_rotary_factor"] == 0.25


def test_gemma_implicit_layer_pattern():
    cfg = dict(dense_config("gemma3_text"), num_hidden_layers=6, sliding_window=4)
    assert [x.kind for x in describe(cfg).layers] == ["sliding_attention"] * 5 + [
        "full_attention"
    ]


def test_head_ownership_requires_explicit_mapping_for_different_counts():
    assert head_assignment(2, 2) == (0, 1)
    with pytest.raises(ValueError, match="explicit"):
        head_assignment(2, 4)
    assert head_assignment(2, 4, [0, 0, 1, 1]) == (0, 0, 1, 1)
    with pytest.raises(ValueError, match="valid source head"):
        head_assignment(2, 4, [0, 0, 1, 2])


@pytest.mark.parametrize(
    "change",
    [
        dict(num_hidden_layers=0),
        dict(num_attention_heads=3),
        dict(layer_types=["full_attention"]),
        dict(layer_types=["unknown"] * 4),
    ],
)
def test_invalid_geometry_rejected(change):
    with pytest.raises(ValueError):
        describe(dict(dense_config(), **change))


def test_unknown_architecture_rejected():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        get_adapter("unknown")


def test_recurrent_transform_preserves_bilinear_readout_for_rectangular_bases():
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(2, 3, 4, generator=generator, dtype=torch.float64)
    a = torch.randn(2, 3, 5, generator=generator, dtype=torch.float64)
    b = torch.randn(2, 4, 6, generator=generator, dtype=torch.float64)
    q = torch.randn(2, 1, 5, generator=generator, dtype=torch.float64)
    mapped = qwen35.transform_delta_state(source, a, b)
    expected = torch.einsum("hri,hji,hjk,hkl->hrl", q, a, source, b)
    torch.testing.assert_close(q @ mapped, expected)
    assert mapped.shape == (2, 5, 6)
    assert qwen35.required_replay(4) == 3


class AttentionLayer:
    def __init__(self, window=None):
        self.is_sliding = window is not None
        self.sliding_window = window
        self.is_initialized = False

    def update(self, key, value):
        self.keys, self.values = key, value
        self.is_initialized = True
        self.cumulative_length = key.shape[-2]


def test_gemma_window_and_absolute_position_survive_installation():
    key = torch.arange(9 * 2 * 3).reshape(9, 2, 3).float()
    local = gemma3.materialize_kv(
        key, key + 1, layer_type="sliding_attention", sliding_window=4
    )
    full = gemma3.materialize_kv(
        key, key + 1, layer_type="full_attention", sliding_window=4
    )
    assert local.first_position == 6 and local.prefix_length == 9
    torch.testing.assert_close(local.key, key[-3:])
    cache = SimpleNamespace(layers=[AttentionLayer(4), AttentionLayer()])
    mapped = {
        i: (x.key.permute(1, 0, 2)[None], x.value.permute(1, 0, 2)[None])
        for i, x in enumerate((local, full))
    }
    gemma3.install_kv(cache, mapped=mapped, prefix_length=9)
    assert (
        cache.layers[0].keys.shape[-2] == 3 and cache.layers[0].cumulative_length == 9
    )
    assert cache.layers[1].keys.shape[-2] == 9


def test_invalid_later_gemma_payload_leaves_all_layers_unmodified():
    cache = SimpleNamespace(layers=[AttentionLayer(4), AttentionLayer()])
    local = torch.zeros(1, 2, 3, 8)
    invalid_full = torch.zeros(1, 2, 8, 8)
    with pytest.raises(ValueError, match="history length"):
        gemma3.install_kv(
            cache,
            mapped={0: (local, local), 1: (invalid_full, invalid_full)},
            prefix_length=9,
        )
    assert all(not layer.is_initialized for layer in cache.layers)


class RecurrentLayer:
    def __init__(self):
        self.has_previous_state = [False]
        self.recurrent_states, self.conv_states = {}, {}

    def update_conv_state(self, value, *, state_idx, conv_kernel_size):
        self.conv_states[state_idx] = value

    def update_recurrent_state(self, value, *, state_idx):
        self.recurrent_states[state_idx] = value
        self.has_previous_state[state_idx] = True


def test_recurrent_install_preserves_fp32_memory_and_bf16_convolution():
    spec = describe(hybrid_config()).layers[0]
    cache = SimpleNamespace(layers=[RecurrentLayer()])
    recurrent = torch.randn(1, 2, 3, 5)
    conv = torch.randn(1, 16, 4, dtype=torch.bfloat16)
    qwen35.install_linear_state(
        cache,
        0,
        recurrent_state=recurrent,
        conv_state=conv,
        conv_kernel_size=4,
        spec=spec,
    )
    assert cache.layers[0].has_previous_state[0]
    assert cache.layers[0].recurrent_states[0] is recurrent
    assert cache.layers[0].conv_states[0] is conv
    with pytest.raises(ValueError, match="previous state"):
        qwen35.install_linear_state(
            cache, 0, recurrent_state=recurrent, conv_state=conv, conv_kernel_size=4
        )


def test_recurrent_install_rejects_geometry_before_mutation():
    spec = describe(hybrid_config()).layers[0]
    cache = SimpleNamespace(layers=[RecurrentLayer()])
    with pytest.raises(ValueError, match="Recurrent tensor"):
        qwen35.install_linear_state(
            cache,
            0,
            recurrent_state=torch.zeros(1, 2, 4, 5),
            conv_state=torch.zeros(1, 16, 4),
            conv_kernel_size=4,
            spec=spec,
        )
    assert not cache.layers[0].conv_states and not cache.layers[0].recurrent_states


def test_attention_install_rejects_recurrent_destination():
    with pytest.raises(TypeError, match="full-attention"):
        qwen35.install_attention_kv(
            SimpleNamespace(layers=[RecurrentLayer()]),
            0,
            torch.zeros(1, 2, 3, 8),
            torch.zeros(1, 2, 3, 8),
        )


def test_qwen_executor_compatibility_imports():
    from cachebridge.architectures import qwen3
    from cachebridge.offline import qwen3_executor

    assert qwen3_executor.prepare is qwen3.prepare
    assert qwen3_executor.step is qwen3.step
    assert qwen3_executor.logits is qwen3.logits
