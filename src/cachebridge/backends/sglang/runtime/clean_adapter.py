"""Clean serving path: diagnostics run only in an explicit verification phase.

Only single-request, idle schedulers, full page-aligned prefixes, PP1 / TP4,
BF16 MHA radix caches. No scheduler algorithm or attention kernel changes.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from types import MethodType
import torch
from cachebridge.backends.sglang.runtime.ipc_transport import (
    publish as ipc_publish,
    receive as transport_receive,
)
from safetensors.torch import load_file, save_file
from cachebridge.backends.sglang.runtime.overlap_bridge import (
    model_runner,
    require_drained,
    attach_work,
    attach_status,
)


def identity(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def attach(s):
    if os.environ.get("CACHEBRIDGE_SERVING_HANDOFF_AUDIT") != "1":
        raise RuntimeError("Explicit handoff switch required")
    root = Path(os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROOT"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    role = os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROLE"]
    rank = s.tp_rank
    pool = s.token_to_kv_pool_allocator.get_kvcache()
    assert s.pp_size == 1 and s.tp_size == int(os.environ["R013_TARGET_TP"])
    assert (
        type(s.tree_cache).__name__ == "RadixCache"
    ), "Audit disables asynchronous HiCache storage"
    assert pool.dtype == torch.bfloat16 and pool.store_dtype == pool.dtype
    layers = range(pool.start_layer, pool.start_layer + pool.layer_num)
    execution_device = pool._get_key_buffer(pool.start_layer).device
    log = root / f"{role}-rank{rank}.jsonl"
    detailed_verify = False
    profiler = None
    # No stats formatting/token-usage probes when serving metrics/KV events are
    # disabled. Cache safety checks and forward counters are independent of this.
    assert not getattr(s, "enable_metrics", False)
    assert not getattr(s, "enable_kv_cache_events", False)
    s.current_scheduler_metrics_enabled = lambda: False

    def emit(event, **kw):
        record = dict(
            event=event,
            role=role,
            tp_rank=rank,
            pid=os.getpid(),
            device=str(pool.device),
            cuda_device=torch.cuda.current_device(),
            phase=getattr(s, "audit_phase", "startup"),
            local_monotonic_ns=time.perf_counter_ns(),
            instrumentation="verify-only",
        )
        record.update(kw)
        with log.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def resolve(value):
        p = (root / value).resolve()
        if not p.is_relative_to(root):
            raise ValueError("Artifact path outside audit namespace")
        return p

    def idle():
        if (
            s.waiting_queue
            or not s.running_batch.is_empty()
            or s.chunked_req is not None
        ):
            raise RuntimeError("Handoff requires idle single-request scheduler")
        require_drained(s)
        torch.cuda.synchronize()

    original = s.run_batch

    def traced(self, batch):
        reqs = []
        for req in batch.reqs:
            prefix = len(req.prefix_indices)
            reqs.append(
                dict(
                    rid=req.rid,
                    origin_ids=req.origin_input_ids,
                    prefix_tokens=prefix,
                    extend_input_len=req.extend_input_len,
                    request_pool_row=int(req.req_pool_idx),
                )
            )
        if detailed_verify:
            emit(
                "scheduled_forward_lite",
                mode=str(batch.forward_mode),
                num_tokens=batch.input_ids.numel(),
                requests=reqs,
                actual_GPU_ids_positions_and_page_table="verify only",
            )
        if detailed_verify:
            emit(
                "verify_scheduled_forward",
                input_ids=batch.input_ids.cpu().tolist(),
                requests=[
                    dict(
                        prefix_tokens=len(r.prefix_indices),
                        page_mapping=self.req_to_token_pool.req_to_token[
                            r.req_pool_idx, : len(r.origin_input_ids)
                        ]
                        .cpu()
                        .tolist(),
                    )
                    for r in batch.reqs
                ],
            )
        return original(batch)

    if hasattr(s, "tp_worker"):
        runner = model_runner(s)
        original_forward = runner.forward

        def forward_traced(self, forward_batch, *args, **kwargs):
            if detailed_verify:
                emit(
                    "verify_model_input",
                    input_ids=forward_batch.input_ids.cpu().tolist(),
                    positions=forward_batch.positions.cpu().tolist(),
                )
            return original_forward(forward_batch, *args, **kwargs)

    ipc_receive = transport_receive

    def rpc(self, p):
        nonlocal detailed_verify, profiler
        if p["action"] in ("load_mapper", "prepare_commit", "prepare", "commit"):
            raise RuntimeError(
                "R013 uses only its immutable topology-aware factor path"
            )
        if p["action"] == "export" and (
            role != "source" or p.get("transport") != "file"
        ):
            raise RuntimeError(
                "R013 diagnostics only export source TP4 fixtures to files"
            )
        if p["action"] == "profile_end":
            # Observation only: stop before any audit idle/synchronize probe.
            # A returned generation can precede finished-batch cleanup.
            assert not detailed_verify
            if True:  # Separate diagnostic profiles on all TP ranks.
                assert profiler is not None
                profiler.stop()
                profiler.export_chrome_trace(
                    str(root / f"profile-{p['label']}-{role}-rank{rank}.json")
                )
                profiler = None
            return
        if p["action"] == "status":
            # Diagnostic receipt, requested only outside timing/profile windows.
            state = dict(
                phase=p["phase"],
                tp_rank=rank,
                pid=os.getpid(),
                role=role,
                role_tp_size=self.tp_size,
                target_tp=int(os.environ["R013_TARGET_TP"]),
                idle=not self.waiting_queue
                and self.running_batch.is_empty()
                and self.chunked_req is None,
                forward_ct=self.forward_ct,
                tree_tokens=self.tree_cache.total_size(),
                diagnostic_hooks=detailed_verify
                or self.run_batch != original
                or runner.forward != original_forward,
                runtime_stats_enabled=self.current_scheduler_metrics_enabled(),
                allocated=torch.cuda.memory_allocated(execution_device),
                reserved=torch.cuda.memory_reserved(execution_device),
                peak_allocated=torch.cuda.max_memory_allocated(execution_device),
                peak_reserved=torch.cuda.max_memory_reserved(execution_device),
            )
            if "history_ids" in p:
                state["resident_tokens"] = len(
                    self.tree_cache.match_prefix(p["history_ids"]).device_indices
                )
            (root / f"{role}-status-rank{rank}.json").write_text(json.dumps(state))
            return
        idle()
        action = p["action"]
        self.audit_phase = p.get("phase", action)
        if detailed_verify:
            emit("rpc_start", action=action)
        if action == "diagnostics":
            detailed_verify = bool(p["enabled"])
            self.run_batch = MethodType(traced, self) if detailed_verify else original
            runner.forward = (
                MethodType(forward_traced, runner)
                if detailed_verify
                else original_forward
            )
            return
        if action == "memory_begin":
            torch.cuda.reset_peak_memory_stats(execution_device)
            if detailed_verify:
                emit(
                    "memory_baseline",
                    allocated=torch.cuda.memory_allocated(execution_device),
                    reserved=torch.cuda.memory_reserved(execution_device),
                    execution_device=str(execution_device),
                )
            return
        if action == "memory_end":
            if detailed_verify:
                emit(
                    "memory_peak",
                    allocated=torch.cuda.max_memory_allocated(execution_device),
                    reserved=torch.cuda.max_memory_reserved(execution_device),
                    execution_device=str(execution_device),
                )
            return
        if action == "profile_begin":
            assert not detailed_verify and profiler is None
            if True:  # Separate diagnostic profiles on all TP ranks.
                # CUDA-only trace: no autograd CPU-op, shape, memory, stack or
                # per-layer record_function callbacks. Never use this TTFT as latency evidence.
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                )
                profiler.start()
            return
        if action == "prepare_commit":
            self.audit_transaction_active = True
            error = None
            try:
                try:
                    rpc(self, dict(p, action="prepare"))
                except Exception as exc:
                    error = exc
                ready = torch.tensor([int(error is None)], dtype=torch.int32)
                torch.distributed.all_reduce(
                    ready, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
                )
                if not ready.item():
                    if hasattr(self, "audit_pending"):
                        self.token_to_kv_pool_allocator.free(self.audit_pending[1])
                        del self.audit_pending
                    raise RuntimeError(
                        f"TP prepare failed, allocated pages returned: {error}"
                    )
                if detailed_verify:
                    emit("tp_writes_ready", tp_size=4)
                rpc(self, dict(p, action="commit"))
            finally:
                self.audit_transaction_active = False
            return
        if action in {"prepare", "commit"} and (
            not getattr(self, "audit_transaction_active", False)
        ):
            raise RuntimeError(
                "Use prepare_commit: unowned pages cannot span scheduler idle iterations"
            )
        if action == "phase":
            return
        if action == "flush":
            if hasattr(self, "audit_pending"):
                raise RuntimeError("Cannot flush pending unpublished handoff")
            from cachebridge.backends.sglang.runtime.warm_prefix_reset import (
                reset_prefix_preserving_allocator,
            )

            assert reset_prefix_preserving_allocator(self)
            if detailed_verify:
                emit(
                    "fresh_target_state",
                    resident_prefix_tokens=0,
                    allocator_policy="preserve_reserved",
                    allocated=torch.cuda.memory_allocated(execution_device),
                    reserved=torch.cuda.memory_reserved(execution_device),
                )
            return
        if action == "load_mapper":
            if os.environ.get("CACHEBRIDGE_LOCAL_HEAD_PREP") == "1":
                from cachebridge.online.mapping import PageMapper
            else:
                from cachebridge.mapper import PageMapper
            self.audit_mappers = {}
            for name, folder in [
                ("Dense", p["mapper_dir"]),
                ("CacheBridge", p["headlocal_dir"]),
            ]:
                self.audit_mappers[name] = PageMapper(
                    folder,
                    method="nv_transfer" if name == "Dense" else "cachebridge",
                    source_rope_theta=1000000.0,
                    target_rope_theta=1000000.0,
                    device=str(execution_device),
                    target_tp_rank=rank,
                    target_tp_size=4,
                )
            assert all(
                (
                    torch.equal(a.selected, b.selected)
                    for (a, b) in zip(
                        self.audit_mappers["Dense"].layers,
                        self.audit_mappers["CacheBridge"].layers,
                    )
                )
            )
            if detailed_verify:
                emit(
                    "mapper_loaded",
                    methods=list(self.audit_mappers),
                    mapper_device=str(execution_device),
                    parameter_bytes={
                        name: sum(
                            (
                                w.numel() * w.element_size()
                                for l in obj.layers
                                for w in [l.k_weight, l.k_bias, l.v_weight, l.v_bias]
                            )
                        )
                        for (name, obj) in self.audit_mappers.items()
                    },
                )
            return
        if action == "release_ipc":
            value = self.audit_ipc_exports.pop(p["artifact"])
            size = value.numel() * value.element_size()
            del value
            torch.cuda.ipc_collect()
            if detailed_verify:
                emit(
                    "ipc_export_released",
                    artifact=p["artifact"],
                    bytes=size,
                    reason="controller acknowledged all consumer copies",
                )
            return
        if action == "relay_native":
            path = resolve(p["artifact"])
            meta = json.loads((path / f"rank{rank}.json").read_text())
            assert (
                meta["kind"] == "native_target"
                and meta["history_ids"] == p["history_ids"]
            )
            value = ipc_receive(path, meta, rank, execution_device)
            destination = resolve(p["destination"])
            meta["ipc_descriptors"] = ipc_publish(value, destination, rank, [rank])
            meta["relay_device"] = str(execution_device)
            (destination / f"rank{rank}.json").write_text(
                json.dumps(meta, indent=2) + "\n"
            )
            if not hasattr(self, "audit_ipc_exports"):
                self.audit_ipc_exports = {}
            self.audit_ipc_exports[p["destination"]] = value
            if detailed_verify:
                emit("native_relay_ready", bytes=value.numel() * value.element_size())
            return
        ids = p["history_ids"]
        n = len(ids)
        if n == 0 or n % self.page_size:
            raise ValueError("Only nonempty page-aligned histories supported")
        found = self.tree_cache.match_prefix(ids)
        if action == "export":
            if len(found.device_indices) != n:
                raise RuntimeError(
                    "History not resident in actual source GPU radix cache"
                )
            indices = found.device_indices
            self.tree_cache.inc_lock_ref(found.last_device_node)
            try:
                chunks = []
                use_ipc = p.get("transport") == "ipc"
                for layer in layers:
                    k = pool._get_key_buffer(layer)[indices]
                    v = pool._get_value_buffer(layer)[indices]
                    chunks.append(
                        torch.stack([k, v] if use_ipc else [k.cpu(), v.cpu()])
                    )
                value = torch.stack(chunks, dim=1)
                path = resolve(p["artifact"])
                path.mkdir(parents=True, exist_ok=True, mode=448)
                descriptors = None
                if use_ipc:
                    descriptors = ipc_publish(
                        value,
                        path,
                        rank,
                        (
                            [rank]
                            if p.get("compact") or p["kind"] == "native_target"
                            else range(4)
                        ),
                    )
                    if not hasattr(self, "audit_ipc_exports"):
                        self.audit_ipc_exports = {}
                    self.audit_ipc_exports[p["artifact"]] = value
                else:
                    save_file(
                        {"kv": value.contiguous()},
                        str(path / f"rank{rank}.safetensors"),
                    )
                metadata = dict(
                    kind=p["kind"],
                    role=role,
                    model=str(self.server_args.model_path),
                    history_ids=ids,
                    input_sha256=identity(ids),
                    position_start=0,
                    valid_tokens=n,
                    page_size=self.page_size,
                    tp_size=4,
                    tp_rank=rank,
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    forward_ct=self.forward_ct,
                    tensor_sha256=None,
                    transport="ipc" if use_ipc else "file",
                    ipc_descriptors=descriptors,
                    artifact_protocol="single request namespace; payload hash scan verify-only",
                )
                (path / f"rank{rank}.json").write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )
                if detailed_verify:
                    emit(
                        "pool_export",
                        **metadata,
                        bytes=value.numel() * value.element_size(),
                    )
            finally:
                self.tree_cache.dec_lock_ref(found.last_device_node)
            return
        if action == "prepare":
            if len(found.device_indices) or self.tree_cache.total_size() != 0:
                raise RuntimeError(
                    "Installation requires empty actual target radix cache"
                )
            path = resolve(p["artifact"])
            mode = p["mode"]
            metas = [json.loads((path / f"rank{i}.json").read_text()) for i in range(4)]
            for i, m in enumerate(metas):
                assert (
                    m["history_ids"] == ids
                    and m["valid_tokens"] == n
                    and (m["position_start"] == 0)
                )
                assert (
                    m["tp_rank"] == i
                    and m["tp_size"] == 4
                    and (m["page_size"] == self.page_size)
                )
                assert m["kind"] == (
                    "native_target" if mode == "native" else "source_history"
                )
                assert (
                    m["input_sha256"] == identity(ids)
                    and m["dtype"] == "torch.bfloat16"
                )
            if mode == "native":
                assert all(
                    (m["model"] == str(self.server_args.model_path) for m in metas)
                )
                value = (
                    ipc_receive(path, metas[rank], rank, execution_device)
                    if metas[rank].get("transport") == "ipc"
                    else load_file(str(path / f"rank{rank}.safetensors"))["kv"]
                )
                if detailed_verify and metas[rank].get("transport") == "ipc":
                    reference = load_file(
                        str(
                            resolve(p["reference_artifact"]) / f"rank{rank}.safetensors"
                        )
                    )["kv"]
                    assert torch.equal(reference, value.cpu())
                    if detailed_verify:
                        emit("verify_native_IPC", bitwise_equal=True)
            elif mode in ("dense", "planned"):
                if not hasattr(self, "audit_mappers"):
                    raise RuntimeError("Mapper must be loaded before handoff")
                assert all((m["role"] == "source" for m in metas))
                assert all(
                    (
                        Path(m["model"]).name == "Qwen--Qwen3-14B--40c069824f42"
                        for m in metas
                    )
                )
                assert (
                    Path(self.server_args.model_path).name
                    == "Qwen--Qwen3-32B--9216db5781bf"
                )
                method = p.get("mapper_kind", "Dense")
                delivery = p.get("delivery", "cpu")
                assert method in self.audit_mappers and delivery in (
                    "cpu",
                    "gpu",
                    "support",
                )
                mapper = self.audit_mappers[method]
                sparse = delivery == "support" and method == "CacheBridge"
                ranks = [rank] if sparse else list(range(4))
                use_ipc = metas[0].get("transport") == "ipc"
                shards = [
                    (
                        ipc_receive(path, metas[i], rank, execution_device)
                        if use_ipc
                        else load_file(str(path / f"rank{i}.safetensors"))["kv"]
                    )
                    for i in ranks
                ]
                for x, i in zip(shards, ranks):
                    assert (
                        list(x.shape) == metas[i]["shape"] == [2, 40, n, 2, 128]
                        and x.dtype == pool.dtype
                    )
                full = shards[0] if sparse else torch.cat(shards, dim=3)
                if detailed_verify:
                    emit(
                        "materialized_inputs",
                        copied_shards_bytes=sum(
                            (x.numel() * x.element_size() for x in shards)
                        ),
                        assembled_bytes=full.numel() * full.element_size(),
                        assembled_aliases_shard=sparse,
                        allocated=torch.cuda.memory_allocated(execution_device),
                    )
                if detailed_verify:
                    emit(
                        "dependency_plan",
                        method=method,
                        delivery=delivery,
                        read_source_ranks=ranks,
                        source_head_ids=[h for i in ranks for h in [i * 2, i * 2 + 1]],
                        target_heads=[rank * 2, rank * 2 + 1],
                        input_shape=list(full.shape),
                        logical_input_bytes=full.numel() * full.element_size(),
                        transport="ipc" if use_ipc else "file",
                        file_bytes=(
                            None
                            if use_ipc
                            else sum(
                                (
                                    (path / f"rank{i}.safetensors").stat().st_size
                                    for i in ranks
                                )
                            )
                        ),
                        output_device=(
                            "cpu" if delivery == "cpu" else str(execution_device)
                        ),
                    )
                value = torch.cat(
                    [
                        mapper.map_page(
                            full[:, :, start : start + 512],
                            position_start=start,
                            source_head_offset=rank * 2 if sparse else 0,
                            output_device=(
                                "cpu" if delivery == "cpu" else execution_device
                            ),
                        )
                        for start in range(0, n, 512)
                    ],
                    dim=2,
                )
                if detailed_verify:
                    from cachebridge.mapper import PageMapper as OriginalMapper

                    reference_path = resolve(p.get("reference_artifact", p["artifact"]))
                    reference_input = torch.cat(
                        [
                            load_file(str(reference_path / f"rank{i}.safetensors"))[
                                "kv"
                            ]
                            for i in range(4)
                        ],
                        dim=3,
                    )
                    reference = torch.cat(
                        [
                            OriginalMapper.map_page(
                                mapper,
                                reference_input[:, :, start : start + 512],
                                position_start=start,
                            )
                            for start in range(0, n, 512)
                        ],
                        dim=2,
                    )
                    actual = value.cpu()
                    difference = (reference.float() - actual.float()).abs().max().item()
                    if detailed_verify:
                        emit(
                            "verify_original_mapper",
                            bitwise_equal=torch.equal(reference, actual),
                            max_abs=difference,
                        )
                    assert torch.equal(reference, actual), difference
                if detailed_verify:
                    emit(
                        "dense_converted",
                        method=method,
                        delivery=delivery,
                        shape=list(value.shape),
                        device=str(value.device),
                    )
            else:
                raise ValueError(mode)
            assert value.dtype == pool.dtype and value.shape[1] == pool.layer_num
            indices = self.token_to_kv_pool_allocator.alloc(n)
            if indices is None:
                raise RuntimeError("No target GPU pages available")
            try:
                for i, layer in enumerate(layers):
                    pool.set_kv_buffer(
                        None,
                        indices,
                        value[0, i].to(execution_device),
                        value[1, i].to(execution_device),
                        layer_id_override=layer,
                    )
                torch.cuda.synchronize()
                equal = None
            except Exception:
                self.token_to_kv_pool_allocator.free(indices)
                raise
            self.audit_pending = (ids, indices)
            if detailed_verify:
                equal = all(
                    (
                        torch.equal(
                            pool._get_key_buffer(layer)[indices].cpu(),
                            value[0, i].cpu(),
                        )
                        and torch.equal(
                            pool._get_value_buffer(layer)[indices].cpu(),
                            value[1, i].cpu(),
                        )
                        for (i, layer) in enumerate(layers)
                    )
                )
                assert equal
                if detailed_verify:
                    emit(
                        "verify_installed_pool",
                        bitwise_equal=equal,
                        pool_indices=indices.cpu().tolist(),
                    )
            if detailed_verify:
                emit(
                    "prepare_buffers_scope_end",
                    output_bytes=value.numel() * value.element_size(),
                    allocated=torch.cuda.memory_allocated(execution_device),
                    note="prepare function locals released on return; output already copied to pool",
                )
            if detailed_verify:
                emit(
                    "target_pool_written_unpublished",
                    method=mode,
                    bitwise_equal=equal,
                    valid_tokens=n,
                    bytes=value.numel() * value.element_size(),
                    forward_ct=self.forward_ct,
                )
            return
        if action == "commit":
            (pending_ids, indices) = self.audit_pending
            assert pending_ids == ids
            assert self.tree_cache.insert(ids, indices) == 0
            match = self.tree_cache.match_prefix(ids)
            assert torch.equal(match.device_indices, indices)
            del self.audit_pending
            if detailed_verify:
                emit(
                    "radix_published",
                    valid_tokens=n,
                    input_sha256=identity(ids),
                    forward_ct=self.forward_ct,
                )
            if detailed_verify:
                if detailed_verify:
                    emit(
                        "verify_published_mapping", pool_indices=indices.cpu().tolist()
                    )
            return
        if action == "verify_resident":
            assert len(found.device_indices) == n
            if detailed_verify:
                emit("resident_verified", valid_tokens=n, forward_ct=self.forward_ct)
            return
        raise ValueError(action)

    s.audit_handoff = MethodType(rpc, s)
    if detailed_verify:
        emit(
            "adapter_ready",
            model=str(s.server_args.model_path),
            page_size=s.page_size,
            tp_size=s.tp_size,
            start_layer=pool.start_layer,
            end_layer=pool.end_layer,
            allocated_layers=pool.layer_num,
            pool_type=type(pool).__name__,
            tree_type=type(s.tree_cache).__name__,
        )
