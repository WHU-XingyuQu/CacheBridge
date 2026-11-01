# Third-party notices

- **SGLang**: external Git submodule under `third_party/sglang`, from
  https://github.com/sgl-project/sglang. Its upstream license and kernel notices
  remain in the checkout and prepared engine. Base revision, patch and hashes
  are packaged under `src/cachebridge/backends/sglang/resources`. Source
  distributions contain the submodule declaration and preparation tools.
- **τ-bench**: official environment, tools, task definitions and scorer under
  `benchmarks/tau_bench/vendor/tau-bench`. Its upstream `LICENSE` and README
  are retained. The CacheBridge adapter uses a local model-backed user simulator;
  it does not replace the official task reward with a new grader.
  `benchmarks/tau_bench/upstream_provenance.json` records the exact upstream
  revision and hashes of all 102 bundled upstream files.
- **PyTorch, Transformers, Accelerate, safetensors and other runtime dependencies**
  are external dependencies. Their packages retain their respective licenses.
- **Qwen3 foundation-model weights** are not included. Supply your own local
  model installation. Mapper artifacts are distributed separately from the
  source package, with their provenance and pending project license status.
