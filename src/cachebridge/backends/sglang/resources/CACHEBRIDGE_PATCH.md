# CacheBridge compatibility patch

Base: SGLang `v0.4.10.post2`, commit
`8cd344586e09669432d35c678b7cc208c0a6f47e`.

The compatibility patch supports online cross-model KV reuse with CUDA 12:

- add `dynamic` to `--hicache-storage-backend`;
- load the backend class from the JSON file named by
  `SGLANG_HICACHE_DYNAMIC_CONFIG`;
- call optional `batch_set_chained(..., prior_hash=...)` so the backend can
  preserve absolute page positions for RoPE removal/reapplication;
- lower the L3 prefetch threshold from a hard-coded 256 tokens to one page;
- continue `write_through` from L2 to L3 when the host-copy acknowledgement
  arrives, rather than requiring a second identical request;
- wait for dynamic-backend L3 prefetch/mapping completion on every TP rank
  before moving a request from the waiting queue into target prefill;
- constrain `cuda-python` to `<13`.

No attention kernel, model executor, scheduler decision, radix hash function,
or device/host page-transfer kernel is replaced. The patch is stored in
`src/cachebridge/backends/sglang/resources/engine.patch`. Its scheduler hook
imports the installed `cachebridge.backends.sglang.runtime` adapter.

Keep `third_party/sglang` at the pinned upstream commit. `cachebridge backend
prepare` applies the patch in a separate prepared-engine directory and verifies
the packaged hashes. The upstream submodule remains unchanged.
