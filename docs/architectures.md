# Model architecture adapters

`cachebridge.architectures` owns model-family structure, layer execution and
continuation-state layout. Offline fitting uses this information to select
compatible source layers and execute target query layers. Online adapters use
it to materialize and install the target's state.

```text
architectures/
  pairs.py       # Explicit contracts for complete model-pair recipes
  base.py        # ArchitectureSpec, LayerSpec and explicit head ownership
  __init__.py    # Lazy model-family registry
  qwen3.py       # Full attention and Qwen3 layer execution
  llama.py       # Full attention and Llama layer execution
  ministral.py   # Ministral 3 layer execution
  gemma3.py      # Local/global KV layout and cache installation
  qwen35.py      # Attention, DeltaNet memory and convolution state
  _hf.py         # Shared Transformers layer-execution compatibility
```

## Describe a model

Pass a resolved Hugging Face configuration or a complete configuration mapping.
Multimodal wrappers are unwrapped through `text_config`. Raw configuration files
that omit default geometry should first be resolved with the corresponding
Transformers configuration class.

```python
from cachebridge.architectures import describe, get_adapter, head_assignment

spec = describe(model.config)
adapter = get_adapter(model.config)
for layer in spec.layers:
    print(layer.index, layer.kind, layer.state_names)
```

Each layer records its type, head counts and dimensions, plus its sliding
window or convolution kernel when applicable. `position_encoding` retains
the model's native RoPE configuration; it does not flatten local/global or
partial-rotary conventions into a single generic rotation.

`source_spec.compatible_source_layers(target_layer)` selects layers with the
same state kind. It supplies structural candidates, not an R² ranking or
fitted coefficients. `head_assignment(source_heads, target_heads)` returns
identity for equal counts. Different counts require an explicit assignment,
for example `head_assignment(2, 4, [0, 0, 1, 1])`.

## Offline layer execution

Qwen3, Llama and Ministral expose the same functions:

```python
hidden, metadata = adapter.prepare(model, query_ids, history_length)
for layer in range(spec.num_layers):
    hidden, query_cache = adapter.step(
        model, layer, hidden, history_kv[layer], metadata
    )
next_token_logits = adapter.logits(model, hidden)
```

`model` is a causal language model with its text backbone and output head.
The query state and per-layer KV returned by these functions reside on CPU;
each layer moves its inputs to that layer's device. Model loading and placement
are the caller's responsibility. The caller should use `torch.inference_mode()`.

Qwen3 uses the Transformers 4.54 interface of the complete fitting recipe.
Llama and Ministral select the supported cache argument and cache layout from
the installed Transformers API. Their layer executors require full-attention
history caches. Ministral requires a Transformers installation
that provides `transformers.models.ministral3`.

## Gemma 3 cache layout

`gemma3.materialize_kv` accepts content-space KV in `[tokens, heads, dim]`.
Global layers retain the full history. Local layers retain at most
`sliding_window - 1` historical tokens at a decode boundary, and return the
absolute `first_position` alongside the original `prefix_length`.

Apply the target layer's native RoPE at those absolute positions, arrange KV
as `[batch, heads, tokens, dim]`, then call
`gemma3.install_kv(cache, mapped=..., prefix_length=...)`. All payloads are
validated before installation. Local layers preserve their absolute consumed
prefix length so the next query receives the correct window mask.

## Qwen3.5 structured state

`qwen35.transform_delta_state(source_state, key_basis, value_basis)` computes
`A.transpose(-2, -1) @ source_state @ B`. Batch/head dimensions may broadcast;
the last two dimensions represent key and value coordinates.

Use `install_linear_state` for the recurrent memory and convolution tail,
and `install_attention_kv` for full-attention layers. Passing the destination
`LayerSpec` to `install_linear_state` also checks recurrent head dimensions
and convolution channels. FP32 recurrent memory and BF16 convolution state
may coexist. Reconstructed coefficients, head assignment and convolution
history are supplied by the fitting and handoff code.

`required_replay(kernel_size)` gives the suffix length needed to replace a
zero convolution history. It does not specify a replay budget for recurrent
approximation error or perform first-layer reconstruction.

## Integration boundaries

| Family | Adapter operations |
|---|---|
| Qwen3 | Structure and layerwise query execution |
| Llama | Structure and layerwise query execution |
| Ministral 3 | Structure and layerwise query execution |
| Gemma 3 | Structure, local/global KV materialization and native cache installation |
| Qwen3.5 | Structure, recurrent-state transformation and native cache installation |

`CAPABILITIES` on each module lists its operations. Family modules load lazily;
structure inspection and tensor/state operations do not import Transformers.
Gemma 3 and Qwen3.5 installers require the native cache-layer APIs used by
Transformers 5.15, including local-window accounting and recurrent-state methods.

These adapters are building blocks. The complete `cachebridge-fit` driver and
SGLang launcher still enforce their Qwen3-14B → Qwen3-32B recipe and geometry.
The generic initializer currently requires equal KV-head counts. Registering
another architecture does not change those entry-point contracts; a new complete
recipe must wire its traces, executor, state mapping and serving backend together.

Complete pair validation is shared by offline recipes and the SGLang configuration
loader through `architectures/pairs.py`. The sequential recipe resolves the target
executor from the family registry and retains its own model-specific attention
capture. Adding a family does not automatically register a complete pair.
