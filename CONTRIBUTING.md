# Contributing

Use a fresh environment and install `python -m pip install -e '.[dev]'`.
Run `python -m pytest` and `python examples/fit_and_map.py` for the CPU path.
CPU CI is defined in `.github/workflows/cpu.yml`. The concrete GPU integration
is under `src/cachebridge/backends/sglang`; its functional gate runs through the real source
pipeline and target installation.

Choose the module that owns the change:

- Model geometry, layer execution and continuation-state layout belong under
  `cachebridge.architectures`. Keep family-specific Transformers imports lazy.
- Fitting data, source support, sufficient statistics and calibration belong
  under `cachebridge.offline`.
- Conversion scheduling, memory ownership, delivery and installation belong
  under `cachebridge.online`.
- Engine-specific streams, collectives and scheduler attachment belong under
  `src/cachebridge/backends/sglang/runtime`; benchmark policies belong under `benchmarks`.
- Artifact loading and cache representation are shared boundaries. Avoid
  importing a serving engine into offline numerical modules.

A fitting change should state its objective, precision and data boundary.
Retain calibration/evaluation separation. An online change should state which
resource or dependency it changes and verify numerical parity, source
immutability, slot lifetime and publication behavior as applicable. Compare
performance only with the full handoff-to-first-token cost accounted for.

First-party Python is formatted with Black24.10.0; keep external upstream code
unchanged. Apply formatting only to `src`, `tests`, `examples`, the runtime
adapters and benchmark adapters.

Use synthetic inputs for unit tests. Keep fitted mapper weights, model weights,
traces, evaluation outputs, credentials and machine-specific service files out
of the source tree. Paths and hardware selection should be explicit parameters.

When changing numerical code, document the objective and tensor conventions
and add focused checks for the behavior being changed. Keep alternative model
adapters and optional baselines separate from the default execution path.

There is no contribution licensing policy yet; the maintainers will choose the
project license before public release.

Model-pair support is explicit in `architectures/pairs.py`; a family adapter alone
does not register a full fitting or serving recipe. Follow [extension guidance](docs/extending.md).
See [paper pipeline](docs/paper_pipeline.md) for stage ownership and commands.

Keep the SGLang submodule at the pinned commit. Edit the compatibility patch in
`src/cachebridge/backends/sglang/resources`, update its file hashes, and prepare
a new engine directory. Never edit a prepared engine or silently advance upstream.
