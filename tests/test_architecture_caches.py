"""CPU cache installation against native Transformers structured-cache classes."""

import inspect

import pytest
import torch

from cachebridge.architectures import describe, gemma3, qwen35


def native_cache_type():
    transformers = pytest.importorskip("transformers")
    if "config" not in inspect.signature(transformers.DynamicCache).parameters:
        pytest.skip("Requires the config-driven structured DynamicCache interface")
    return transformers.DynamicCache


def test_native_gemma_cache_preserves_next_query_window_offset():
    cache_type = native_cache_type()
    module = pytest.importorskip("transformers.models.gemma3.configuration_gemma3")
    cfg = module.Gemma3TextConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
    )
    cache = cache_type(config=cfg)
    mapped = {}
    for layer in describe(cfg).layers:
        length = 3 if layer.kind == "sliding_attention" else 9
        key = torch.randn(1, 2, length, 8)
        mapped[layer.index] = key, torch.randn_like(key)
    gemma3.install_kv(cache, mapped=mapped, prefix_length=9)
    assert cache.layers[0].get_mask_sizes(1) == (4, 6)
    assert cache.layers[1].get_seq_length() == 9
    for index, layer in enumerate(cache.layers):
        torch.testing.assert_close(layer.keys, mapped[index][0])
        torch.testing.assert_close(layer.values, mapped[index][1])


def test_native_qwen35_cache_contains_all_continuation_states():
    cache_type = native_cache_type()
    module = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
    cfg = module.Qwen3_5TextConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
    )
    cache = cache_type(config=cfg)
    spec = describe(cfg)
    state = torch.randn(1, 2, 4, 4)
    conv = torch.randn(1, spec.layers[0].conv_channels, 4, dtype=torch.bfloat16)
    qwen35.install_linear_state(
        cache,
        0,
        recurrent_state=state,
        conv_state=conv,
        conv_kernel_size=4,
        spec=spec.layers[0],
    )
    key, value = torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8)
    qwen35.install_attention_kv(cache, 1, key, value)
    assert cache.layers[0].has_previous_state[0]
    torch.testing.assert_close(cache.layers[0].recurrent_states[0], state)
    torch.testing.assert_close(cache.layers[0].conv_states[0], conv)
    torch.testing.assert_close(cache.layers[1].keys, key)
    torch.testing.assert_close(cache.layers[1].values, value)
