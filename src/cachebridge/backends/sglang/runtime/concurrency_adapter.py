"""Request-scoped, sequential-install concurrency preparation for Q14->32.

Handoffs are deliberately serialized.  The target may retain several disjoint
prefixes, after which the ordinary scheduler can execute their query suffixes
concurrently.  This measures the current architecture and does not claim
cross-request handoff overlap.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
from types import MethodType

import torch

from cachebridge.backends.sglang.runtime.bounded_cachebridge_core import (
    SourceLease,
    pack_source_block,
)
from cachebridge.backends.sglang.runtime.bounded_handoff_channel import (
    Cancelled,
    CudaCodec,
    Receiver,
    Sender,
)
from cachebridge.backends.sglang.runtime import config as _config

LOCAL_KV_HEADS = _config.PAIR.kv_heads // _config.PAIR.tensor_parallel_size
from cachebridge.backends.sglang.runtime.r010_core import map_write_received
from cachebridge.backends.sglang.runtime.resident_session import write
from cachebridge.backends.sglang.runtime.overlap_bridge import (
    model_runner,
    require_drained,
    attach_work,
    attach_status,
)


def _history_identity(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def _contract(p, source_rank, target_rank):
    method = p["mapper_kind"]
    if method not in ("CacheBridge", "Dense"):
        raise ValueError(method)
    if method == "CacheBridge" and source_rank != target_rank:
        raise ValueError("CacheBridge support route must be rank local")
    ids = p["history_ids"]
    return {
        "version": 2,
        "request": p["request_id"],
        "source_rank": source_rank,
        "target_rank": target_rank,
        "method": method,
        # bounded_handoff_channel uses mode=C to derive the per-offer span.
        # This concurrency adapter always streams fixed-size blocks.
        "mode": "C",
        "tokens": len(ids),
        "block_tokens": p.get("reference_block_tokens", _config.BLOCK_TOKENS),
        "layers": 40,
        "heads": LOCAL_KV_HEADS,
        "dim": 128,
        "elements": 2
        * 40
        * min(len(ids), p.get("reference_block_tokens", _config.BLOCK_TOKENS))
        * LOCAL_KV_HEADS
        * 128,
        "history_sha256": _history_identity(ids),
    }


class _Workspace:
    def __init__(self, device, method, block_tokens=_config.BLOCK_TOKENS):
        heads = LOCAL_KV_HEADS if method == "CacheBridge" else 8
        self.method = method
        self.device = device
        self.slots = [
            torch.empty(
                (2, 40, block_tokens, heads, 128),
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in range(2)
        ]
        self.copy_stream = torch.cuda.Stream(device=device)
        self.compute_stream = torch.cuda.Stream(device=device)
        self.done = [torch.cuda.Event(), torch.cuda.Event()]
        self.used = [False, False]
        self.slot = 0

    @property
    def bytes(self):
        return sum(value.numel() * value.element_size() for value in self.slots)

    def begin(self):
        current = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current)
        self.compute_stream.wait_stream(current)

    def select(self, slot):
        self.slot = slot

    def finish(self):
        self.copy_stream.synchronize()
        self.compute_stream.synchronize()


class _SliceCodec(CudaCodec):
    def __init__(self, workspace, head_start):
        super().__init__(workspace.device)
        self.workspace = workspace
        self.head_start = head_start

    def allocate(self, elements):
        return None

    def copy(self, remote, local, elements):
        n = elements // (2 * 40 * LOCAL_KV_HEADS * 128)
        workspace = self.workspace
        slot = workspace.slot
        head_stop = self.head_start + LOCAL_KV_HEADS
        value = workspace.slots[slot][:, :, :n, self.head_start : head_stop]
        with torch.cuda.stream(workspace.copy_stream):
            if workspace.used[slot]:
                workspace.copy_stream.wait_event(workspace.done[slot])
            value.copy_(
                remote[:elements].view(2, 40, n, LOCAL_KV_HEADS, 128), non_blocking=True
            )
        # The ACK certifies remote reads only.  Map/write can remain in flight.
        workspace.copy_stream.synchronize()
        return value

    def synchronize(self):
        self.workspace.copy_stream.synchronize()


def _verify_files(folder, expected):
    folder = Path(folder)
    for name, digest in expected.items():
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"mapper hash mismatch: {folder / name}")


def attach(s):
    if hasattr(s, "concurrency_state"):
        raise RuntimeError("concurrency adapter already attached")
    role = os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROLE"]
    rank = s.tp_rank
    root = Path(os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROOT"])
    pool = s.token_to_kv_pool_allocator.get_kvcache()
    device = pool._get_key_buffer(pool.start_layer).device
    if s.tp_size != _config.TARGET_TP or s.pp_size != 1 or s.page_size != 1:
        raise RuntimeError("frozen concurrency path requires TP4/PP1/page1")
    if pool.layer_num != (40 if role == "source" else 64):
        raise RuntimeError("unexpected source/target layer partition")

    artifacts = _config.artifact_contract()
    mappers = {}
    workspaces = {}
    if role == "target":
        from cachebridge.online.mapping import PageMapper

        for item in artifacts.values():
            _verify_files(item["mapper_dir"], item["file_hashes"])
        mappers["CacheBridge"] = PageMapper(
            artifacts["cachebridge"]["mapper_dir"],
            method="cachebridge",
            source_rope_theta=1_000_000.0,
            target_rope_theta=1_000_000.0,
            device=device,
            target_tp_rank=rank,
            target_tp_size=_config.TARGET_TP,
        )
        if "dense" in artifacts:
            mappers["Dense"] = PageMapper(
                artifacts["dense"]["mapper_dir"],
                method="nv_transfer",
                source_rope_theta=1_000_000.0,
                target_rope_theta=1_000_000.0,
                device=device,
                target_tp_rank=rank,
                target_tp_size=_config.TARGET_TP,
            )
        workspaces = {method: _Workspace(device, method) for method in mappers}
        gc.collect()
        torch.cuda.empty_cache()

    original = s.audit_handoff
    state = {
        "active_request": None,
        "quarantined": False,
        "requests": {},
        "serial_handoffs_only": True,
    }
    codec = CudaCodec(device)
    workers = s.factor_threads if role == "source" else None
    listeners = s.factor_listeners if role == "target" else None

    def idle():
        if (
            s.waiting_queue
            or not s.running_batch.is_empty()
            or s.chunked_req is not None
        ):
            raise RuntimeError("handoff requires an idle scheduler")
        require_drained(s)
        torch.cuda.synchronize(device)

    def vote(ok, safe):
        flags = torch.tensor([int(ok), int(safe)], dtype=torch.int32)
        torch.distributed.all_reduce(
            flags, op=torch.distributed.ReduceOp.MIN, group=s.tp_cpu_group
        )
        return bool(flags[0]), bool(flags[1])

    def parallel(links, operation):
        futures = [workers.submit(operation, link) for link in links]
        errors = []
        for future in futures:
            try:
                future.result()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            if all(isinstance(exc, Cancelled) for exc in errors):
                raise Cancelled("all remote reads certified")
            raise RuntimeError([repr(exc) for exc in errors])

    def source_targets(method):
        return [rank] if method == "CacheBridge" else list(range(_config.TARGET_TP))

    def target_sources(method):
        return [rank] if method == "CacheBridge" else list(range(_config.TARGET_TP))

    def source_run(p):
        block_tokens = p.get("reference_block_tokens", _config.BLOCK_TOKENS)
        ids = p["history_ids"]
        n = len(ids)
        found = s.tree_cache.match_prefix(ids)
        if len(found.device_indices) != n:
            raise RuntimeError("source history is not fully resident")
        targets = source_targets(p["mapper_kind"])
        storage = torch.empty(
            2 * 40 * min(n, block_tokens) * LOCAL_KV_HEADS * 128,
            dtype=pool.dtype,
            device=device,
        )
        lease = SourceLease(
            lambda: s.tree_cache.inc_lock_ref(found.last_device_node),
            lambda: s.tree_cache.dec_lock_ref(found.last_device_node),
        )
        links = []
        record = state["requests"][p["request_id"]]
        record.update(role=role, phase="connecting", published=False)
        lease.begin()
        try:

            def connect(target_rank):
                link = Sender(
                    s.factor_endpoints[rank, target_rank],
                    _contract(p, rank, target_rank),
                    codec,
                )
                links.append(link)
                link.share(storage)

            parallel(targets, connect)
            count = 0
            for start in range(0, n, block_tokens):
                stop = min(start + block_tokens, n)
                pack_source_block(
                    pool,
                    found.device_indices,
                    start=start,
                    stop=stop,
                    workspace=storage,
                    block_tokens=min(n, block_tokens),
                )
                lease.export(start, stop)
                parallel(links, lambda link: link.offer(start, stop))
                lease.acknowledge(gpu_reads_complete=True)
                count += 1
            parallel(links, lambda link: link.finish())
            lease.close()
            record.update(
                phase="complete",
                outcome="complete",
                blocks=count,
                source_lease_released=True,
                consumers=targets,
                logical_delivery_bytes=n
                * 2
                * 40
                * LOCAL_KV_HEADS
                * 128
                * 2
                * len(targets),
            )
        except BaseException:
            state["quarantined"] = True
            raise
        finally:
            if not state["quarantined"]:
                for link in links:
                    link.close()

    def receive_blocks(p, indices, workspace, links):
        """Identical Dense receive/map loop; no allocation, radix or TP vote."""
        ids = p["history_ids"]
        n = len(ids)
        method = p["mapper_kind"]
        block_tokens = p.get("reference_block_tokens", _config.BLOCK_TOKENS)
        sources = target_sources(method)
        count = 0
        for source_rank in sources:
            head_start = 0 if method == "CacheBridge" else source_rank * LOCAL_KV_HEADS
            links.append(
                Receiver(
                    listeners[source_rank],
                    _contract(p, source_rank, rank),
                    _SliceCodec(workspace, head_start),
                )
            )
        for start in range(0, n, block_tokens):
            stop = min(start + block_tokens, n)
            slot = count % 2
            workspace.select(slot)
            for link in links:
                observed_start, observed_stop, _ = next(link)
                if (observed_start, observed_stop) != (start, stop):
                    raise RuntimeError("delivered block span mismatch")
            block = workspace.slots[slot][:, :, : stop - start]
            with torch.cuda.stream(workspace.compute_stream):
                map_write_received(
                    mappers[method],
                    block,
                    pool,
                    indices,
                    start=start,
                    source_head_offset=(
                        rank * LOCAL_KV_HEADS if method == "CacheBridge" else 0
                    ),
                    block_tokens=block_tokens,
                )
                workspace.done[slot].record(workspace.compute_stream)
                workspace.used[slot] = True
            count += 1
            if p.get("execution") == "serial":
                workspace.compute_stream.synchronize()
            for link in links:
                link.acknowledge()
        for link in links:
            try:
                next(link)
            except StopIteration:
                pass
            else:
                raise RuntimeError("unexpected extra source block")
        workspace.finish()
        return count

    def target_run(p):
        block_tokens = p.get("reference_block_tokens", _config.BLOCK_TOKENS)
        ids = p["history_ids"]
        n = len(ids)
        method = p["mapper_kind"]
        workspace_key = (
            method
            if block_tokens == _config.BLOCK_TOKENS
            else f"{method}-b{block_tokens}"
        )
        if workspace_key not in workspaces:
            workspaces[workspace_key] = _Workspace(device, method, block_tokens)
        workspace = workspaces[workspace_key]
        sources = target_sources(method)
        found = s.tree_cache.match_prefix(ids)
        if len(found.device_indices):
            raise RuntimeError("target prefix already exists or shares a radix root")
        indices = None
        links = []
        error = None
        safe = True
        count = 0
        record = state["requests"][p["request_id"]]
        record.update(role=role, phase="receiving", published=False)
        try:
            indices = s.token_to_kv_pool_allocator.alloc(n)
            if indices is None:
                raise RuntimeError("target token pool exhausted")
            record["pool_indices"] = indices
            workspace.begin()
            count = receive_blocks(p, indices, workspace, links)
        except BaseException as exc:
            error = repr(exc)
            try:
                workspace.finish()
                for link in links:
                    link.abort()
            except BaseException as cleanup:
                safe = False
                error += "; cleanup=" + repr(cleanup)
        ready, all_safe = vote(error is None, safe)
        if not all_safe:
            state["quarantined"] = True
            raise RuntimeError("cannot certify remote reads/local writes")
        if not ready:
            if indices is not None:
                s.token_to_kv_pool_allocator.free(indices)
            record.update(phase="rolled_back", outcome="rolled_back", error=error)
            raise RuntimeError(error)
        try:
            inserted = s.tree_cache.insert(ids, indices)
            if inserted != 0:
                raise RuntimeError(f"unexpected existing target prefix: {inserted}")
            matched = s.tree_cache.match_prefix(ids)
            if not torch.equal(matched.device_indices, indices):
                raise RuntimeError("published radix mapping differs from allocation")
        except BaseException as exc:
            error = repr(exc)
        published, _ = vote(error is None, True)
        if not published:
            state["quarantined"] = True
            raise RuntimeError("publication failed; ownership retained")
        record.pop("pool_indices", None)
        record.update(
            phase="published",
            outcome="published",
            published=True,
            tokens=n,
            blocks=count,
            sources=sources,
            logical_copy_bytes=sum(link.bytes_copied for link in links),
            input_slots=2,
            workspace_bytes=workspace.bytes,
            execution=p.get("execution", "double"),
        )

    def status(p):
        histories = p.get("history_ids_list", [])
        rows = []
        for ids in histories:
            match = s.tree_cache.match_prefix(ids)
            rows.append(
                {
                    "history_sha256": _history_identity(ids),
                    "tokens": len(ids),
                    "resident_tokens": len(match.device_indices),
                }
            )
        mapper_bytes = {
            method: sum(
                tensor.numel() * tensor.element_size()
                for layer in mapper.layers
                for tensor in (
                    layer.k_weight,
                    layer.k_bias,
                    layer.v_weight,
                    layer.v_bias,
                )
            )
            for method, mapper in mappers.items()
        }
        write(
            root / f"{role}-concurrency-status-rank{rank}.json",
            {
                "phase": p["phase"],
                "role": role,
                "rank": rank,
                "pid": os.getpid(),
                "idle": not s.waiting_queue
                and s.running_batch.is_empty()
                and s.chunked_req is None,
                "active_request": state["active_request"],
                "quarantined": state["quarantined"],
                "serial_handoffs_only": True,
                "completed_request_ids": sorted(state["requests"]),
                "free_tokens": s.token_to_kv_pool_allocator.available_size(),
                "protected_tokens": s.tree_cache.protected_size(),
                "tree_tokens": s.tree_cache.total_size(),
                "histories": rows,
                "mapper_parameter_bytes": mapper_bytes,
                "workspace_bytes": {
                    method: workspace.bytes for method, workspace in workspaces.items()
                },
            },
        )

    def rpc(self, p):
        action = p["action"]
        if not action.startswith("concurrency_"):
            return original(p)
        if action == "concurrency_status":
            status(p)
            return
        if action == "concurrency_clear_records":
            idle()
            if state["active_request"] is not None:
                raise RuntimeError("cannot clear an active request")
            state["requests"].clear()
            return
        if action != "concurrency_run":
            raise ValueError(action)
        idle()
        if state["quarantined"] or state["active_request"] is not None:
            raise RuntimeError("handoff state is active or quarantined")
        if p["mapper_kind"] not in ("CacheBridge", "Dense"):
            raise ValueError(p["mapper_kind"])
        if p.get("execution", "double") not in ("double", "serial"):
            raise ValueError(p.get("execution"))
        request_id = p["request_id"]
        if request_id in state["requests"]:
            raise RuntimeError("request identity already used")
        state["active_request"] = request_id
        state["requests"][request_id] = {
            "request_id": request_id,
            "history_sha256": _history_identity(p["history_ids"]),
            "method": p["mapper_kind"],
            "phase": "active",
        }
        try:
            return source_run(p) if role == "source" else target_run(p)
        finally:
            state["active_request"] = None

    # The R013 adapter owns these endpoints but did not expose their names.
    socket_root = Path(os.environ["CACHEBRIDGE_BOUNDED_SOCKET_ROOT"])
    s.factor_endpoints = {
        (source_rank, target_rank): socket_root / f"f{source_rank}-{target_rank}.sock"
        for source_rank in range(_config.TARGET_TP)
        for target_rank in range(_config.TARGET_TP)
    }
    s.concurrency_receive_blocks = receive_blocks
    s.concurrency_state = state
    s.concurrency_mappers = mappers
    s.concurrency_workspaces = workspaces
    s.audit_handoff = MethodType(rpc, s)
    write(
        root / f"{role}-concurrency-ready-rank{rank}.json",
        {
            "pid": os.getpid(),
            "rank": rank,
            "role": role,
            "serial_handoffs_only": True,
            "request_scoped_records": True,
            "max_running_requests": s.server_args.max_running_requests,
            "mapper_ids": {
                name: value["mapper_id"] for name, value in artifacts.items()
            },
            "mapper_parameter_bytes": {
                method: sum(
                    tensor.numel() * tensor.element_size()
                    for layer in mapper.layers
                    for tensor in (
                        layer.k_weight,
                        layer.k_bias,
                        layer.v_weight,
                        layer.v_bias,
                    )
                )
                for method, mapper in mappers.items()
            },
            "workspace_bytes": {
                method: workspace.bytes for method, workspace in workspaces.items()
            },
        },
    )
