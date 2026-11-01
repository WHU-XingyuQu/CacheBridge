# Serving

The supported integration uses Qwen3-14B Source TP4 and Qwen3-32B Target TP4 on one
eight-GPU host. It retains source-side conversion, 2048-token blocks, two source
output slots, explicit source read lifetime, asynchronous CUDA peer copies,
direct target-pool installation and all-rank ready publication. The reference
handoff reconstructs the complete history serially for equality checks.

## Runtime dependencies

SGLang 0.4.10.post2 is an external Git submodule at `third_party/sglang`.
Its commit is pinned in Git and in
`src/cachebridge/backends/sglang/resources/engine_provenance.json`; the
compatibility patch lives alongside it. CacheBridge-owned adapters are installed
as `cachebridge.backends.sglang`.

```bash
git submodule update --init third_party/sglang
python -m pip install -e .
cachebridge backend prepare
```

Preparation exports the exact upstream commit, applies the patch in an isolated
directory, and checks file hashes. It keeps the submodule clean. For a source
archive or an installed wheel, `cachebridge backend prepare --fetch` explicitly
fetches the pin. Preparation is CPU-only and does not import SGLang or load models.

For a separate Linux CUDA 12.6 environment:

```bash
python -m pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -c requirements/sglang.txt \
  -e '.cache/cachebridge/sglang/python[srt]' accelerate==1.12.0 safetensors==0.8.0
python -m pip install -c requirements/sglang.txt -e '.[fitting,benchmark,serving]'
```

The prepared SGLang project supplies dependency metadata. The launcher verifies
its source files and selects that engine for spawned workers. Run `--check` and
the complete-history functional gate after provisioning the environment.

CUDA runtime discovery uses the installed `nvidia.cuda_runtime` package or an
explicit `CACHEBRIDGE_CUDART=/path/to/libcudart.so.12`. Compilation caches should
be on local writable storage; `TRITON_CACHE_DIR` can override the local default.

## Explicit configuration

Copy `recipes/runtime.example.json`, then set local model, mapper descriptor and
prepared `engine_path` paths. `--engine` can override the configured engine.
Relative paths resolve against the config file. No model download is attempted.
The default pool is 530000 tokens and context capacity 40960. Source/target ports
must be distinct. The launcher accepts `--devices` with eight distinct physical
indices; the TP layout is Source first four, Target last four.

The launcher adds the prepared external engine and installed CacheBridge package
to child-process import paths. Runtime modules use qualified package imports. Run `--check` before GPU work;
this verifies imports and the CUDA runtime version without loading models.

The benchmark starts both endpoints, checks complete-history mapping parity on
a fixed TRAIN case, then runs sequential full task episodes. It shuts down its
own endpoints when finished. Other GPU processes cause admission failure;
there is no foreign-job cancellation or cluster reservation management.

Native user simulation and agent resolution have distinct cache namespaces.
Every generation receives a unique request ID, while turns share the explicit
episode namespace. Reusing a generation ID across turns can corrupt incremental
detokenization; the adapter verifies actual completion token IDs against text.

## Runtime boundaries

The artifact stores global head-local weights; each TP rank uses its head slice.
Only a complete published mapped prefix may be used by the target. Source pages
stay pinned until all GPU reads and slot acknowledgements complete. Direct
installation removes history-sized target staging; it does not imply constant
total request memory.

An optional dense mapper can be configured with `dense_descriptor`. When
measuring serving performance, account for conversion, transport, installation,
queueing and target generation after handoff arrival, and document resident
memory separately from per-request allocations.

## Configuration API

`RuntimeConfig.load(path)` resolves and validates configuration explicitly.
Importing backend modules does not read model paths or configuration files.
The launcher passes the config path across process boundaries; each worker loads
its process-local configuration when runtime functions first need it. Use separate
processes for independently configured engine instances.
