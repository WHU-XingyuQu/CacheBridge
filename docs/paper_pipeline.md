# Paper pipeline and implementation

The main package follows the offline fitting / online serving boundary.
Recipes select objectives and data; architecture adapters describe model
structure; the backend owns engine-specific execution.

| Pipeline stage | Implementation | Entry point |
|---|---|---|
| Native paired traces | `offline/traces`, `offline/trajectory`, benchmark collector | `cachebridge serve ... -- python benchmarks/tau_bench/run.py sample ...` |
| Source support | `offline/support.py` | `load_support`; complete trajectory recipe uses supplied initializer support |
| Centered K/V initialization | `offline/initialization.py`, `offline/ridge.py` | `fit_layer` for paired synthetic or prepared content-space tensors |
| Trajectory K initialization | `offline/keys.py`, `offline/weighted_keys.py` | `cachebridge fit prepare`, then `cachebridge fit run` |
| Sequential conditional V calibration | `offline/calibration/sequential.py` | Value stage of `cachebridge fit run` |
| Fold and export | `offline/export.py`, `artifacts.py` | `cachebridge fit run`, then `cachebridge verify` |
| Source block mapping | `mapper.py`, `online/mapping.py`, backend `source_mapper_adapter.py` | Prepared SGLang source pipeline |
| Buffers, delivery, installation | `online/pipeline.py`, backend peer/lifetime modules | Backend handoff execution |
| All-rank publication | `online/handoff.py`, backend `handoff.py` and `ready_publication.py` | `run_handoff` / backend `transfer` |
| Agent sampling and routing | `agents/sampling.py`, `trajectory.py` | `sample_trajectory` |
| Workload evaluation | `benchmarks/tau_bench` | Benchmark `evaluate`, `compare.py` |

`cachebridge recipes` describes the registered complete recipe. The generic
initializer and trajectory-weighted K solver retain distinct objectives and
precision. The exported sequential calibration currently solves regularized
readout matching. Automatic R² support ranking, energy-constrained calibration,
and a dynamic shared multi-request buffer-pool coordinator are not exposed as
complete public recipes here. An API or directory name does not imply those
method stages are implemented.

The online lifecycle reference modules define ownership contracts. Actual
CUDA streams, peer operations and scheduler publication are in the SGLang
backend; changing a reference state machine alone does not change GPU execution.

Weights, traces and generated outputs are external inputs/artifacts. Recipes
contain configuration and commands, not fitted weights or evaluation outputs.
