# Mapper artifact interface

Mapper weights are supplied separately and referenced by a portable descriptor.
No weight download occurs during import.

The existing head-local loader reads one `layer_NNN.safetensors` file for each
target layer, in increasing layer order. A complete artifact must contain every
target layer. The low-level layer loader is intentionally small. Use
`cachebridge-verify path/to/mapper.json` to validate the complete layer roster,
file hashes, tensor geometry and finite coefficients before loading. The
concrete serving adapter also checks model geometry, tokenizer equality and
loaded mapper hashes.

| Tensor | Shape / meaning |
| --- | --- |
| `selected` | `[support_layers]`, int64 source-layer indices in feature order |
| `headwise` | Present for head-local artifacts; writer emits boolean true |
| `k_weight`, `v_weight` | `[target_heads, support_layers * source_dim, target_dim]` |
| `k_bias`, `v_bias` | `[target_heads, target_dim]` |

Weights and biases are stored as FP32. Source layer support is shared between
K and V for each target layer. Final V corrections are folded into `v_weight`
and `v_bias`; the online mapper does not need the calibration statistics.

`offline.fitting.fit_layer` returns this tensor dictionary for one target
layer, together with solve diagnostics. Save it using
`safetensors.torch.save_file`. Its output is an **initializer**. The full
sequential V-calibration driver must run before describing the artifact as
the complete method. The CPU example intentionally uses a one-layer fixture.

The trace API uses content-space K. The online page API uses RoPE-applied K:
the mapper removes source RoPE, transforms K/V, then applies target RoPE using
the page's true absolute `position_start`. Values have no RoPE transform.
Compact TP input also needs the correct `source_head_offset`. A preallocated
`output` must match the resulting shape, dtype and device.

The low-level reference mapper also retains its legacy dense-affine mode to
preserve regression checks. That mode is not a strict-original CFM execution
baseline and is not author-provided CFM code.

Before enabling a production serving adapter, bind the tensor files to source
and target model revisions, tokenizer compatibility, RoPE configuration,
fitting stage/recipe, precision, calibration-input fingerprints and file hashes.
The descriptor records these identities and the fitting provenance. Foundation-model revisions are recorded separately from mapper
file hashes; an architecture-compatible fine-tune is not automatically a
compatible replacement for the model used during fitting.
