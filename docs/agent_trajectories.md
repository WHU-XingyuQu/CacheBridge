# Agent trajectories and model routing

`sample_trajectory` is a synchronous integration interface for an external
agent environment, model router and CacheBridge backend. It does not instantiate
model servers or perform network/tool calls unless your adapters implement them.

```python
from cachebridge.agents import CacheBridgePolicy, sample_trajectory
from cachebridge.trajectory import TaskRef

trajectory = sample_trajectory(
    TaskRef("my-benchmark", "my-domain", "train", "task-001"),
    run_id="experiment-v1/native/rep-0",
    environment=environment,
    router=router,
    backend=backend,
    models=("source-model@revision", "target-model@revision"),
    cache_policy=CacheBridgePolicy({
        ("source-model@revision", "target-model@revision"): "local-mapper-v1"
    }),
    seed=7,
    max_steps=30,
    max_new_tokens=1024,
)
trajectory.save("trajectory-task-001.json")
```

The names `environment`, `router` and `backend` above are application adapters.
For a complete runnable CPU example with toy implementations, run:

```bash
python examples/sample_trajectory.py
```

## Adapter responsibilities

| Adapter | Methods | Responsibility |
| --- | --- | --- |
| Environment | `reset(task, seed=...)`, `step(response)` | Original agent observations, tool execution and episode termination |
| Model router | `choose(context)` | Select the next model using only the currently visible conversation |
| Backend | `begin_episode(id)`, `generate(request)`, `end_episode(id)` | Tokenization, actual model execution, cache ownership, transfer and completion receipts |

The router sees `RouteContext`: current messages, step index, previous model
and available model IDs. It receives no success label, reference answer, future
observation, or fitting data. Task scoring occurs separately after the episode.
The environment must keep gold actions and labels private; the API cannot make
an arbitrary third-party environment label-free by itself.

Use a distinct `run_id` for every arm and repetition. Request namespaces include
it so paired Native/mapped runs cannot accidentally share an episode cache.
Reusing the same run/task/seed deliberately reproduces the same record identity.

`CacheBridgePolicy` acts after model selection. It requests same-model reuse,
directed cross-model mapping when a registered mapper exists, or native
re-prefill. It never changes the router's chosen model. Disable cross-model
mapping with `enabled=False` to run the same router as a native control.

The backend returns actual token IDs, tokenizer identity, execution mode,
reused-token count, mapper identity, finish reason and optional timings.
Unsupported mapping or an evicted cache may fall back to native re-prefill,
but the record must say so explicitly. CPU metadata validation is not proof of
physical GPU transfer; the backend still needs its own full-cache and lifetime
checks. The supplied `benchmarks/tau_bench/live_adapter.py` implements the
concrete Qwen3/SGLang backend on top of `src/cachebridge/backends/sglang/runtime`. It supports
the qualified single intake-to-resolution handoff; arbitrary multi-model or
bidirectional routing requires a separate backend extension and validation.

The backend must report the actual model ID; substitution is rejected. Model
IDs should include revisions. Use a fresh or explicitly reset router and
environment for each trajectory. Cleanup is called even after a partial begin
failure and should be idempotent. Backend cleanup must retain uncertain GPU
ownership when completion cannot be certified. Do not release shared resident
models during episode cleanup.

## Fitting samples

```python
from cachebridge.offline.trajectory import (
    select_training_trajectories, handoff_samples,
)

selected = select_training_trajectories(records, count=32, seed="experiment-v1")
samples = [sample for record in selected
           for sample in handoff_samples(record, max_query_tokens=2048)]
```

Selection uses a deterministic hash of task identity, independent of outcome.
It requires unique TRAIN tasks, rejects dev/test data, and retains failed and
limited trajectories. A trajectory with no usable model switch produces no
handoff sample; keep it in the attempted-task count and never silently replace
it with a successful task.

Collect teachers with native target execution. At a switch the backend must
record the exact candidate `handoff_prefix_ids` even when re-prefill is used.
The extractor checks exact token-prefix continuity across later target calls,
then emits history/query IDs and assistant prediction positions. A position is
the query input token that predicts the following assistant token, not the
assistant token's own index. The final query input may predict an assistant
token outside a truncated window: that input still has a valid Native attention
output for conditional V fitting, so its position is retained. This format does
not supply next-token labels. Truncation is explicit. Mapped continuations cannot be
silently used as native teachers.

The result is tokenized calibration input, not KV tensors or a trained mapper.
Model adapters must collect paired source/target KV and run the complete
head-local initialization and sequential V-calibration recipe. Mapper weights
remain external to the open-source package.

## Quality experiments

Fix the router, task IDs, user simulator, prompts, model/runtime versions and
generation budgets across native, baseline-mapper and expanded-data arms.
Use TRAIN trajectories only for fitting. Compare complete-task successes on
the same evaluation tasks and bootstrap paired task clusters, stratifying by
domain for pooled summaries. Include failures, no-handoff episodes and budget
limits. Report actual route choices, mapping/fallback counts and fitting cost.

In an interactive environment, changed model responses can change later
observations. Hold the routing *policy* fixed and record those consequences;
do not force an oracle route from the future of another arm. Reused historical
test tasks are exposed evaluation, not a fresh blind confirmation.

Cache reconstruction error, attention residuals and CPU examples are numerical
diagnostics. They cannot establish an improvement in agent task quality.

The supplied CPU analysis command replays the official tools and rewards,
checks actual generation token receipts and paired source histories/KV, then
computes the candidate-minus-baseline success difference and retention:

```bash
python benchmarks/tau_bench/analyze.py \
  --native /path/to/native-evaluation \
  --baseline /path/to/baseline-evaluation \
  --candidate /path/to/candidate-evaluation \
  --tasks recipes/tau_test.json --output /path/to/paired-analysis.json
```

It requires all tasks in the supplied roster and three distinct arm names.
Choose distinct `--arm` names when evaluating different mappers. It uses 10,000
paired task bootstrap draws, stratified by domain for the pooled estimate.
If any draw has zero Native successes, the corresponding retention interval is
left undefined; the paired success-rate difference remains available. Intervals
describe the supplied single-seed task panel and are not adjusted for repeated
experiments or confirmation on new tasks. The caller must also freeze the
model/runtime/router and generation budgets; a score replay cannot prove those
external experimental conditions.
