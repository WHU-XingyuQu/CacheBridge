"""Small CPU models compare layerwise execution with native cached forward."""

import importlib

import pytest
import torch

from cachebridge.architectures import get_adapter


@pytest.mark.parametrize(
    "family,config_name,model_name",
    [
        ("qwen3", "Qwen3Config", "Qwen3ForCausalLM"),
        ("llama", "LlamaConfig", "LlamaForCausalLM"),
        ("ministral3", "Ministral3Config", "Ministral3ForCausalLM"),
    ],
)
@torch.inference_mode()
def test_layerwise_execution_matches_native(family, config_name, model_name):
    transformers = pytest.importorskip("transformers")
    if family == "qwen3" and not transformers.__version__.startswith("4.54."):
        pytest.skip("Qwen3 layer executor uses the pinned Transformers 4.54 API")
    configuration = pytest.importorskip(
        f"transformers.models.{family}.configuration_{family}"
    )
    modeling = importlib.import_module(
        f"transformers.models.{family}.modeling_{family}"
    )
    config_type = getattr(configuration, config_name)
    model_type = getattr(modeling, model_name)
    torch.manual_seed(11)
    cfg = config_type(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "sdpa"
    model = model_type(cfg).eval()
    prefix, query = [1, 2, 3, 4], [5, 6, 7]
    cache = model(input_ids=torch.tensor([prefix]), use_cache=True).past_key_values
    histories = (
        [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
        if hasattr(cache, "layers")
        else [(key.clone(), value.clone()) for key, value in cache]
    )
    expected = model(
        input_ids=torch.tensor([query]), past_key_values=cache, use_cache=True
    ).logits[0, -1]
    adapter = get_adapter(family)
    hidden, metadata = adapter.prepare(model, query, len(prefix))
    saved = [(k.clone(), v.clone()) for k, v in histories]
    for layer, kv in enumerate(histories):
        hidden, result_kv = adapter.step(model, layer, hidden, kv, metadata)
        assert result_kv[0].shape[-2] == len(prefix) + len(query)
    actual = adapter.logits(model, hidden)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for before, after in zip(saved, histories):
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
