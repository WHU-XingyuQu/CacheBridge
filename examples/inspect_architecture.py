"""Inspect mixed continuation-state requirements without loading a model."""

from cachebridge.architectures import describe, head_assignment

spec = describe(
    dict(
        model_type="qwen3_5_text",
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
    )
)

for layer in spec.layers:
    print(f"layer {layer.index}: {layer.kind}, state={layer.state_names}")
print("Explicit head ownership:", head_assignment(1, 2, [0, 0]))
