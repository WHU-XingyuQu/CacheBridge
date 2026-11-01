# Complete trajectory fitting

The fitting recipe is head-local and closed-form. No gradient training,
backpropagation or new neural modules are used. Support, dimensions, coefficient
dtype and transfer bytes stay fixed.

Model-specific execution lives in `cachebridge.architectures`; the complete
recipe below uses the Qwen3 adapter. Other family executors and structured-cache
operations are described in [model architecture adapters](architectures.md).

The stages are organized under `offline/traces`, `support.py`,
`initialization.py`, `calibration` and `export.py`. `offline/recipes.py` binds the
complete recipe; `cachebridge recipes` lists its stages. The original
`cachebridge-fit` command remains an alias for `cachebridge fit`.

## 1. Collect native TRAIN handoffs

The real benchmark collector uses native target continuations. It preserves
unsuccessful/no-handoff attempts; it does not select examples by reward or read
TRAIN gold actions for fitting.

```bash
cachebridge serve --config /path/to/runtime.json -- \
  python benchmarks/tau_bench/run.py sample \
  --tasks recipes/tau_train.example.json --output /path/to/new-native-bank
```

`attempts.jsonl` retains all attempts. `inputs.jsonl` contains usable native
handoffs and exact rank-sharded KV file references. `handoff_samples.jsonl`
exports the public `HandoffSample` contract. Token IDs, tokenizer identity,
source/target model pair, H/Q boundary and assistant prediction positions must
remain aligned. Re-tokenizing a decoded history can invalidate a handoff.

Sampling defaults to a 2,048-token fitting-query cap and at most 6 target calls.
The `sample` command also accepts `--max-query-tokens 4096
--max-target-calls 24`. These control TRAIN collection; evaluation budgets and
the model router are unchanged.

For a controlled extension of an existing bank, keep the same tasks and use:

```bash
cachebridge serve --config /path/to/runtime.json -- \
  python benchmarks/tau_bench/extend.py \
  --inputs /path/to/native-bank/inputs.jsonl \
  --output /path/to/longer-native-bank \
  --max-query-tokens 4096 --max-target-calls 24
```

This command extends only native continuations stopped by their query/call
budget. It retains terminal and other stopped records, with no reward filtering
or replacement. The reference bank must include each `case_<id>.json`, eight
native KV shards per input, and their SHA256 values as emitted by `sample`.
`record_path` can explicitly locate a reference case. Preserve the reference
bank: unchanged rows in the new `inputs.jsonl` keep their original file paths.
The original attempt ledger is retained when available.

Before extending, the sampler requires exact old token prefixes, generated
texts/actions/observations and all eight native KV tensors. It rechecks the
original tensor prefixes after extension. Every old selected prediction input
is retained; the former final input can become one additional selected position
when its following assistant token is now visible. Longer collection changes
both continuation coverage and V observation count. It is not an equal-budget
quality experiment, and its task-quality benefit must be measured separately.

For another agent environment, implement the environment/router/backend
protocols in `cachebridge.agents`, call `sample_trajectory`, and extract
native TRAIN examples with `cachebridge.offline.trajectory.handoff_samples`.
Provide corresponding source-H and target-(H+Q) native KV snapshots to the
fitter. The supplied concrete snapshot adapter is currently τ-bench/Qwen3.

The complete fitter consumes four BF16 TP shards per role, each with layout
`[layers, 2, tokens, local_kv_heads, 128]`; concatenation is over heads. Source
has 40 layers, Target 64, and total KV heads 8. Hashes are checked before use.
Only `split=train`, `task_answers_read=false`, unique usable cases are accepted.

## 2. Prepare a fixed recipe

Supply a complete initializer with the desired source-layer support. Use
`--freeze-keys` to retain its K coefficients while fitting V on a new bank.

```bash
cachebridge-fit prepare \
  --inputs /path/to/new-native-bank/inputs.jsonl \
  --target-model /path/to/Qwen3-32B \
  --initializer /path/to/initializer/weights \
  --freeze-keys --output /path/to/new-fit
```

Without `--freeze-keys`, K is refit from the supplied TRAIN bank using weighted
centered ridge 1 in FP64 and a total 16,384 observations. The unchanged generic V
initializer must be supplied. `--k-order order.json` fixes task remainder
allocation when exact replay requires a different order than input concatenation.
`--input-order order.json` independently freezes the sequential V accumulation
order. Store both order files with the fitting configuration.
The four prefix anchors are retained; the remaining static and dialogue tokens
receive equal total mass. Every history must support its allocated sample count.
The recipe currently requires at least 16 usable cases.

Increasing case count at fixed 16,384 K observations reduces observations per
case. It changes both sampling and statistics; do not call it a pure V-data
intervention. Freeze K when isolating the effect of additional V trajectories.

For each target layer and KV head, the eight selected source layers form a
1024-dimensional input. After removing RoPE and weighted centering, K solves
`W = (Xᵀ D X + I)⁻¹ Xᵀ D Y`, with `b = mean(Y) - mean(X) W`.
Moments and the solve use FP64; the deployed coefficients are FP32. Source-layer
support is supplied by the initializer and is not reselected on agent tasks.

## 3. Run the complete sequential V calibration

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 cachebridge-fit run \
  --directory /path/to/new-fit --mapper-id my-qwen3-agent-mapper
```

The target model is distributed over four GPUs. Each of 64 layers runs the
current candidate query states, computes the native-history shadow attention
output at assistant prediction positions, accumulates equal-case FP64 moments,
and solves a regularized affine V correction. Ridge is fixed to `.01` times
mean Gram diagonal, with floor `1e-8`. The newly fitted layer is propagated before
fitting the next layer. Reusing old V statistics with a different K initializer
is not a complete recalibration.

Assistant positions refer to query **input** activations predicting assistant
tokens, including the final eligible input at a truncated query boundary. Tool
observations remain in the history/context but are not treated as assistant
next-token targets. The objective is conditional attention-output matching,
not supervised fitting to task answers or a globally optimized task reward.

In each V solve, the feature is the current history attention applied to the
source V features plus an affine constant. The target is the difference between
the native-history shadow output and the current mapped output. Per-case moments
are normalized by the number of selected positions and query groups before
summation. The correction solves `(G + lambda I) delta = C`; the affine constant
is regularized too. The correction is folded into the existing V coefficients,
preserving the mapper's tensor shapes and online parameter footprint.

The original layer executor is checked against the model forward. Complete
folded mapper logits must match the propagated fitted states exactly at the
fixed shortest/longest TRAIN cases. Partial, failed or numerically invalid runs
are preserved and must not be relabeled as complete mappers.

## 4. Verify and evaluate

```bash
cachebridge-verify /path/to/new-fit/candidate.json
```

Use the verified descriptor in the serving config. Compare full agent tasks
against a same-router Native run, using identical task IDs, source intake,
generation budgets, model/runtime versions and seed. Report reward, terminal
status, real mapped/fallback counts, tool validity and paired task uncertainty.
A lower conditional TRAIN loss is not evidence of higher agent task quality.
