# External modules

SGLang is a Git submodule at `third_party/sglang`, pinned by the repository
gitlink and the packaged engine provenance. Its upstream URL is recorded in
`.gitmodules`. Upstream licenses remain in the submodule.

```bash
git submodule update --init third_party/sglang
cachebridge backend prepare
```

Preparation checks the pinned commit and clean tracked files, applies the
CacheBridge patch to `.cache/cachebridge/sglang`, and verifies file hashes.
It does not modify the submodule or download models. Repeating preparation
verifies the existing output; modified output must use a new destination.

Source archives and wheels do not embed the SGLang checkout. For an archive
without Git metadata, or an installed wheel, explicitly fetch the same pin:

```bash
cachebridge backend prepare --fetch --source /path/to/sglang \
  --output /path/to/prepared-sglang
```

τ-bench's environment and scorer remain a small pinned snapshot under
`benchmarks/tau_bench/vendor`, with their original licenses and provenance.
