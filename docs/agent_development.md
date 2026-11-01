# Agent development guide

CacheBridge provides separate interfaces for model routing, trajectory sampling,
offline fitting and online KV conversion. The `offline` package prepares fitting
data and mapper coefficients. The `online` package handles conversion, cache
lifetimes and model handoffs.

## Integrating an agent and router

Implement the environment, router and model backend protocols defined in
`cachebridge.agents`, then call `sample_trajectory` to collect the complete event
sequence. At each model handoff, record the routing decision, actual cache mode,
exact token prefix and model identity. Give each generation request a distinct
request ID and each episode an explicit cache namespace.

`cachebridge.offline.trajectory.handoff_samples` converts native TRAIN
trajectories into fitting inputs. Save KV snapshots for the source history and
the target history and query. Keep tokens, tokenizer identity, model revisions
and the boundary between history (H) and query (Q) consistent.

## Collecting and extending trajectories

The agentic workload adaptor provides a sampling entry point for τ-bench:

```bash
cachebridge serve --config /path/to/runtime.json -- \
  python benchmarks/tau_bench/run.py sample \
  --tasks recipes/tau_train.example.json --output /path/to/native-bank
```

The sampling directory stores all attempts, usable handoffs and references to
token and KV files. Attempt records retain generation failures, episodes without
a handoff and stops caused by budget limits. Training inputs are selected
independently of task reward.

```bash
cachebridge serve --config /path/to/runtime.json -- \
  python benchmarks/tau_bench/extend.py \
  --inputs /path/to/native-bank/inputs.jsonl \
  --output /path/to/extended-bank \
  --max-query-tokens 4096 --max-target-calls 24
```

Extension preserves the original task order and completed records, and extends
only native continuations that reached their budget. Check the original tokens,
actions, observations and KV prefixes before and after extension, and retain all
original assistant prediction positions. Tool observations remain context;
align the assistant mask with the input positions that predict the next
assistant token.

## Offline fitting

```bash
cachebridge-fit prepare \
  --inputs /path/to/native-bank/inputs.jsonl \
  --target-model /path/to/Qwen3-32B \
  --initializer /path/to/initializer/weights \
  --freeze-keys --output /path/to/fit

CUDA_VISIBLE_DEVICES=0,1,2,3 cachebridge-fit run \
  --directory /path/to/fit --mapper-id my-agent-mapper

cachebridge-verify /path/to/fit/candidate.json
```

`--freeze-keys` preserves the initializer's K mapping. Omit it to refit K using
the input trajectories. V calibration proceeds in target-layer order: each
layer uses the current candidate state to construct attention-output targets,
solves the mapping in closed form, then propagates the updated state to the next
layer. Rerun complete V calibration whenever K or the fitting data changes.
Record source-layer support, numerical precision, data order and sampling
budgets in the fitting configuration.

## Online serving and evaluation

Set the mapper descriptor in the runtime configuration and use the SGLang
integration to run Source TP4 and Target TP4. The online path includes block
conversion, double buffering, GPU peer copies and direct installation into the
target cache. For a new adaptor, verify consistency across the complete history,
check that the source KV remains unchanged, and validate buffer lifetimes.

```bash
python benchmarks/tau_bench/compare.py \
  --native /path/to/native-evaluation \
  --candidate /path/to/mapped-evaluation \
  --tasks recipes/tau_test.json --output /path/to/comparison.json
```

Evaluate with the same router, task roster, model revisions, user simulator,
generation budgets and seed. Compare rewards for complete tasks, and record
handoffs, fallbacks, tool calls and stop reasons. Use
`benchmarks/tau_bench/analyze.py` for comparisons across three evaluation arms.

See the [trajectory API guide](agent_trajectories.md) for interface contracts,
the [fitting guide](fitting.md) for numerical steps, and the
[serving guide](serving.md) for deployment settings.
