"""Reuse existing clean/cache/streaming adapters with an explicit equal-TP layout."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from types import MethodType

from cachebridge.backends.sglang.runtime import config as _config
from cachebridge.backends.sglang.runtime.overlap_bridge import (
    model_runner,
    require_drained,
    attach_work,
    attach_status,
)


def attach(s):
    from cachebridge.backends.sglang.runtime.clean_adapter import attach as clean_attach
    from cachebridge.backends.sglang.runtime.bounded_handoff_channel import listener
    from cachebridge.backends.sglang.runtime.concurrency_adapter import (
        attach as concurrency_attach,
    )
    from cachebridge.backends.sglang.runtime.source_mapper_adapter import (
        attach as streaming_attach,
    )
    from cachebridge.backends.sglang.runtime.resident_session import write

    attach_work(s)
    clean_attach(s)
    role = os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROLE"]
    socket_root = Path(os.environ["CACHEBRIDGE_BOUNDED_SOCKET_ROOT"])
    endpoints = {
        (a, b): socket_root / f"f{a}-{b}.sock"
        for a in range(_config.TARGET_TP)
        for b in range(_config.TARGET_TP)
    }
    s.factor_threads = (
        ThreadPoolExecutor(max_workers=_config.TARGET_TP) if role == "source" else None
    )
    s.factor_listeners = (
        {a: listener(endpoints[a, s.tp_rank]) for a in range(_config.TARGET_TP)}
        if role == "target"
        else {}
    )
    original = s.audit_handoff
    root = Path(os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROOT"])
    pool = s.token_to_kv_pool_allocator.get_kvcache()

    def rpc(self, p):
        if p["action"] in ("control_background_digest", "control_background_progress"):
            import torch

            # Instrumented correctness only. RPC executes between scheduler
            # steps; the observed first token proves the historical prefix is
            # computed. The native request still owns its radix protection.
            ids = p["history_ids"]
            if (
                role not in ("source", "target")
                or len(ids) != 512
                or s.running_batch.is_empty()
            ):
                raise RuntimeError(
                    "background snapshot requires live endpoint Native512"
                )
            requests = [
                r for r in s.running_batch.reqs if list(r.origin_input_ids[:512]) == ids
            ]
            if len(requests) != 1:
                raise RuntimeError(
                    "background request does not own the requested prefix"
                )
            request = requests[0]
            if len(request.prefix_indices) < 512 or request.last_node.lock_ref <= 0:
                raise RuntimeError("background prefix is incomplete or unprotected")
            if p["action"] == "control_background_progress":
                write(
                    root / f"{role}-background-progress-rank{s.tp_rank}.json",
                    dict(
                        phase=p["phase"],
                        rank=s.tp_rank,
                        native_generated_tokens=len(request.output_ids),
                    ),
                )
                return
            # Never release/publish pages or device-synchronize the decoder.
            # Only already-written history is read on this observation stream.
            device = pool._get_key_buffer(pool.start_layer).device
            read_stream = torch.cuda.Stream(device=device)
            read_stream.wait_stream(torch.cuda.current_stream(device))
            digest = hashlib.sha256()
            with torch.cuda.stream(read_stream):
                indices = s.req_to_token_pool.req_to_token[
                    request.req_pool_idx, :512
                ].clone()
                for layer in range(pool.layer_num):
                    for getter in (pool._get_key_buffer, pool._get_value_buffer):
                        value = (
                            getter(layer)[indices].cpu().contiguous().view(torch.uint8)
                        )
                        digest.update(value.numpy().tobytes())
            read_stream.synchronize()
            write(
                root / f"{role}-background-digest-rank{s.tp_rank}.json",
                dict(
                    phase=p["phase"],
                    rank=s.tp_rank,
                    tokens=512,
                    history_sha256=hashlib.sha256(
                        json.dumps(ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                    sha256=digest.hexdigest(),
                    native_request_owns_prefix=True,
                    native_generated_tokens=len(request.output_ids),
                    observation_only=True,
                    modifies_allocator_or_tree=False,
                ),
            )
            return
        if p["action"] == "system_release_reference":
            import gc
            import torch

            if (
                role != "target"
                or s.waiting_queue
                or not s.running_batch.is_empty()
                or s.chunked_req is not None
            ):
                raise RuntimeError("reference release requires idle target")
            if s.source_mapper_state["transaction"] is not None:
                raise RuntimeError("cannot release references during installation")
            require_drained(s)
            s.concurrency_mappers.clear()
            s.concurrency_workspaces.clear()
            gc.collect()
            torch.cuda.empty_cache()
            return
        if p["action"] in ("r010_digest", "system_snapshot"):
            import torch

            if (
                s.waiting_queue
                or not s.running_batch.is_empty()
                or s.chunked_req is not None
            ):
                raise RuntimeError("digest is a separate idle verification step")
            require_drained(s)
            torch.cuda.synchronize()
            match = s.tree_cache.match_prefix(p["history_ids"])
            if len(match.device_indices) != len(p["history_ids"]):
                raise RuntimeError("digest prefix is incomplete")
            if p["action"] == "system_snapshot":
                values = torch.stack(
                    [
                        torch.stack(
                            [
                                getter(layer)[match.device_indices].cpu()
                                for getter in (
                                    pool._get_key_buffer,
                                    pool._get_value_buffer,
                                )
                            ]
                        )
                        for layer in range(pool.layer_num)
                    ]
                )
                torch.save(values, root / f"{p['phase']}-snapshot-rank{s.tp_rank}.pt")
                return
            digest = hashlib.sha256()
            for layer in range(pool.layer_num):
                for getter in (pool._get_key_buffer, pool._get_value_buffer):
                    digest.update(
                        getter(layer)[match.device_indices]
                        .cpu()
                        .contiguous()
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    )
            write(
                root / f"{role}-r010-digest-rank{s.tp_rank}.json",
                dict(
                    phase=p["phase"],
                    tokens=len(p["history_ids"]),
                    rank=s.tp_rank,
                    sha256=digest.hexdigest(),
                ),
            )
            return
        if p["action"] == "system_memory":
            import torch

            device = pool._get_key_buffer(pool.start_layer).device
            if p.get("reset_peak"):
                torch.cuda.reset_peak_memory_stats(device)
            write(
                root / f"{role}-system-memory-rank{s.tp_rank}.json",
                dict(
                    phase=p["phase"],
                    rank=s.tp_rank,
                    allocated=torch.cuda.memory_allocated(device),
                    reserved=torch.cuda.memory_reserved(device),
                    peak_allocated=torch.cuda.max_memory_allocated(device),
                    peak_reserved=torch.cuda.max_memory_reserved(device),
                    pool_capacity=self.server_args.max_total_tokens,
                    free_tokens=self.token_to_kv_pool_allocator.available_size(),
                    tree_tokens=self.tree_cache.total_size(),
                ),
            )
            return
        return original(p)

    s.audit_handoff = MethodType(rpc, s)
    concurrency_attach(s)
    streaming_attach(s)
    from cachebridge.backends.sglang.runtime.probe_control import attach as probe_attach

    probe_attach(s)

    from cachebridge.backends.sglang.runtime.native_graph_control import (
        attach as graph_attach,
    )

    graph_attach(s)

    from cachebridge.backends.sglang.runtime.peer_adapter import attach as peer_attach

    peer_attach(s)

    attach_status(s)
    from cachebridge.backends.sglang.runtime.ready_publication import (
        attach as ready_attach,
    )

    ready_attach(s)

    from cachebridge.backends.sglang.runtime.generated_bg_witness import (
        attach as generated_bg_attach,
    )

    generated_bg_attach(s)

    from cachebridge.backends.sglang.runtime.fullhead_adapter import (
        attach as fullhead_attach,
    )

    fullhead_attach(s)

    from cachebridge.backends.sglang.runtime.namespace_adapter import (
        attach as namespace_attach,
    )

    namespace_attach(s)
