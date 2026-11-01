# Recipes

`cachebridge recipes` lists the registered complete fitting recipe and its
stages. Choose `--recipe qwen3_14b_32b_trajectory` with `cachebridge fit prepare`
to make that choice explicit; it is also the compatibility default.

- `runtime.example.json`: local model, external mapper, prepared SGLang path,
  TP geometry and runtime settings. Relative paths are relative to this file.
- `tau_train.example.json`: example TRAIN IDs for native trajectory collection.
- `tau_test.json`: official TEST task roster for the workload adapter.
- `qwen3_14b_32b_support.json`: explicit source support configuration.

The [paper pipeline](../docs/paper_pipeline.md) maps each stage to code and
distinguishes generic initialization from the complete trajectory recipe.
Fitted coefficients, downloaded models, traces and outputs belong outside this
directory. Preserve recipe/input identities when resuming a fit.
