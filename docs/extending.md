# Extending CacheBridge

## Add a model family or pair

1. Implement the relevant operations in `architectures`: description, layer
   execution, or structured-state installation. Declare only actual capabilities.
2. Verify small-tensor state behavior and compare the layer executor with native
   model forward in the appropriate dependency environment.
3. Register a complete model-pair contract in `architectures/pairs.py` when its
   geometry, tokenizer relationship, head ownership and state mapping are defined.
4. Add a fitting recipe and connect its trace format, support, initializer,
   calibration and export stages. Reusing Qwen's ChatML positions or SDPA shadow
   capture requires an explicit model-specific justification.
5. Connect the pair to a backend, including TP geometry, absolute positions,
   cache installation and publication. Run the backend functional gate before
   describing the pair as end-to-end supported.

The supplied complete trajectory/serving recipe remains Qwen3-14B → Qwen3-32B.
Family adapter operations and complete recipe support are documented separately
in [architectures](architectures.md).

## Add an agent workload

Implement `AgentEnvironment`, `ModelRouter` and `AgentBackend` from
`cachebridge.agents`. Use `sample_trajectory` and retain exact token IDs and
cache provenance. The environment owns tools and observations; the router owns
model choice; the backend owns inference and cache transfer.

A SGLang-backed workload can reuse
`cachebridge.backends.sglang.handoff.transfer`. Supply an explicit namespace
when it does not use the example client's episode namespace. Keep task parsing,
rewards and workload-specific budgets in the benchmark adapter.

## Add a serving backend

Implement an endpoint for `online.handoff.HandoffEndpoint`, and keep engine
dependencies optional. Start Source and Target before waiting, retain ownership
through asynchronous failures, and provide all-rank completion evidence before
publication. Engine-specific streams, allocator pages and scheduler callbacks
belong under `backends/<engine>`.

Use the common mapper algebra. Add adapters for layouts or state types instead
of copying numerical implementations. Test transaction failures with a fake
endpoint, then use backend-specific GPU functional checks when available.

## Dependency environments

Use `requirements/fitting.txt` for the Qwen recipe and
`requirements/architecture-caches.txt` for the structured-cache adapters in
separate environments. The Transformers major versions are intentionally
distinct. SGLang's CUDA environment is described in [serving](serving.md).
