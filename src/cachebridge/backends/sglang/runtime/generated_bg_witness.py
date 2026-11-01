"""Offline candidate: observe retained Native continuation; never install implicitly."""

import hashlib
import os
import threading
from types import MethodType


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class GeneratedPrefixWitness:
    """One before/after functional observation, scheduler thread only."""

    def __init__(self, scheduler, read_prefix, work_state):
        self.s = scheduler
        self.read_prefix = read_prefix
        self.work_state = work_state
        self.owner = threading.get_ident()
        self.saved = None

    def prepare(self, parameters):
        s = self.s
        require(threading.get_ident() == self.owner, "observation outside scheduler")
        require(
            s.enable_overlap and s.spec_algorithm.is_none(), "unsupported execution"
        )
        require(not s.waiting_queue and s.chunked_req is None, "pending prefill")
        require(not s.concurrency_state["quarantined"], "quarantined ownership")
        require(
            all(
                s.source_mapper_state[k] is None
                for k in ["transaction", "source_transaction"]
            ),
            "unfinished transaction",
        )
        require(
            type(parameters.get("observation_id")) is str
            and 0 < len(parameters["observation_id"]) <= 128,
            "invalid observation ID",
        )
        require(
            type(parameters.get("phase")) is str and bool(parameters["phase"]),
            "missing observation phase",
        )
        reqs = s.running_batch.reqs
        require(
            len(reqs) == 1 and s.running_batch.forward_mode.is_decode(),
            "one Native decoder required",
        )
        req = reqs[0]
        ids = tuple(parameters["background_input_ids"])
        require(
            len(ids) == 526 and all(type(x) is int and x >= 0 for x in ids),
            "Native512 plus fixed query14 required",
        )
        require(tuple(req.origin_input_ids) == ids, "wrong Native input")
        require(
            not req.finished() and not req.is_retracted and req.is_chunked <= 0,
            "expired or retracted Native",
        )
        require(
            type(req.req_pool_idx) is int and req.req_pool_idx >= 0,
            "missing Native index",
        )
        require(
            type(req.rid) is str
            and req.rid
            and len(req.prefix_indices) >= len(ids)
            and req.last_node.lock_ref > 0,
            "Native prefix unprotected",
        )
        outputs = tuple(req.output_ids)
        require(
            len(outputs) >= 2 and all(type(x) is int and x >= 0 for x in outputs),
            "no resolved continuation KV",
        )
        work = self.work_state(s)
        pending = work["submitted"] - work["resolved"]
        require(
            not work["failed"]
            and pending in (0, 1)
            and work["queued_results"] == pending,
            "unaccounted async work",
        )
        batches = [
            ("running", s.running_batch),
            ("current", getattr(s, "cur_batch", None)),
            ("last", getattr(s, "last_batch", None)),
        ]
        batches += [
            (f"queued{i}", row[0])
            for i, row in enumerate(getattr(s, "result_queue", ()))
        ]
        context = []
        for name, batch in batches:
            if batch is not None and not batch.is_empty():
                require(
                    batch.forward_mode.is_decode()
                    and len(batch.reqs) == 1
                    and batch.reqs[0] is req,
                    "another request or prefill in pending work",
                )
                context.append(
                    dict(boundary_batch=name, Native_rids=[r.rid for r in batch.reqs])
                )
        # Output processing resolves copy_done before appending sampled IDs.
        # The last sampled token is deliberately excluded, even if an
        # overlapping forward has already consumed it.
        return dict(
            req=req,
            rid=req.rid,
            req_pool_idx=req.req_pool_idx,
            origin_ids=ids,
            resolved_output_ids=outputs,
            completed_ids=ids + outputs[:-1],
            work=work,
            context=context,
        )

    def capture(self, prepared, observation_id):
        require(self.saved is None, "previous observation not cleared")
        require(
            type(observation_id) is str and 0 < len(observation_id) <= 128,
            "invalid observation ID",
        )
        indices, digest, count = self.read_prefix(
            self.s, prepared["req_pool_idx"], len(prepared["completed_ids"])
        )
        self.check_read(indices, count, len(prepared["completed_ids"]))
        self.saved = dict(
            prepared,
            observation_id=observation_id,
            indices=tuple(indices),
            digest=digest,
            verified=False,
        )
        return self.receipt(prepared, observation_id, indices, digest, count)

    @staticmethod
    def check_read(indices, byte_count, tokens):
        require(
            len(indices) == tokens and all(type(x) is int and x > 0 for x in indices),
            "incomplete physical indices",
        )
        require(len(set(indices)) == len(indices), "aliased Native token slots")
        require(byte_count == tokens * 65536, "wrong complete Target64 BF16 geometry")

    def verify_identity(self, prepared, observation_id):
        old = self.saved
        require(
            old is not None
            and not old["verified"]
            and old["observation_id"] == observation_id,
            "wrong observation identity",
        )
        require(
            prepared["req"] is old["req"]
            and prepared["rid"] == old["rid"]
            and prepared["req_pool_idx"] == old["req_pool_idx"],
            "Native ownership changed",
        )
        require(prepared["origin_ids"] == old["origin_ids"], "Native prompt changed")
        require(
            prepared["resolved_output_ids"][: len(old["resolved_output_ids"])]
            == old["resolved_output_ids"],
            "resolved Native token prefix changed",
        )
        require(
            prepared["work"]["submitted"] >= old["work"]["submitted"]
            and prepared["work"]["resolved"] >= old["work"]["resolved"],
            "async generation changed",
        )
        return old

    def verify(self, prepared, observation_id, foreground_indices):
        old = self.verify_identity(prepared, observation_id)
        n = len(old["completed_ids"])
        indices, digest, count = self.read_prefix(self.s, prepared["req_pool_idx"], n)
        self.check_read(indices, count, n)
        require(tuple(indices) == old["indices"], "Native physical prefix moved")
        require(digest == old["digest"], "Native completed prefix bytes changed")
        require(
            foreground_indices
            and all(type(v) is int and v > 0 for v in foreground_indices)
            and len(set(foreground_indices)) == len(foreground_indices),
            "foreground prefix incomplete or aliased",
        )
        require(
            not set(indices).intersection(foreground_indices),
            "foreground aliases retained Native prefix",
        )
        old["verified"] = True
        receipt = self.receipt(prepared, observation_id, indices, digest, count)
        receipt.update(
            status="verified_retained_completed_Native_prefix",
            observed_before_ids=list(old["completed_ids"]),
            Native_continuation_KV_tokens_verified=n - len(old["origin_ids"]),
            same_request_before_after=True,
            Native_indices_and_bytes_unchanged=True,
            foreground_indices_disjoint=True,
        )
        return receipt

    @staticmethod
    def receipt(prepared, observation_id, indices, digest, count):
        return dict(
            observation_id=observation_id,
            Native_rid=prepared["rid"],
            Native_req_pool_idx=prepared["req_pool_idx"],
            observed_input_ids=list(prepared["completed_ids"][: len(indices)]),
            current_completed_input_ids=list(prepared["completed_ids"]),
            resolved_output_ids=list(prepared["resolved_output_ids"]),
            observed_tokens=len(indices),
            indices=list(indices),
            KV_sha256=digest,
            logical_bytes=count,
            boundary_context=prepared["context"],
            work=prepared["work"],
            last_sampled_token_excluded=True,
            observation_only=True,
            formal_result=False,
            normal_cost=False,
            exact_ordered_decode_history_verified=False,
        )

    def clear(self):
        require(threading.get_ident() == self.owner, "clear outside scheduler")
        require(
            self.s.running_batch.is_empty()
            and not self.s.waiting_queue
            and self.s.chunked_req is None,
            "clear while Native live",
        )
        require(self.work_state(self.s)["drained"], "clear before async drain")
        self.saved = None


def digest_pool(pool, indices):
    """Both K/V, every Target layer and local head, in logical token order."""
    import torch

    require(pool.layer_num == 64, "unsupported target layers")
    digest = hashlib.sha256()
    count = 0
    for layer in range(pool.layer_num):
        for getter in (pool._get_key_buffer, pool._get_value_buffer):
            value = getter(layer)[indices].cpu().contiguous()
            require(
                value.dtype == torch.bfloat16
                and tuple(value.shape) == (len(indices), 2, 128),
                "unsupported KV layout",
            )
            raw = value.view(torch.uint8).numpy().tobytes()
            digest.update(raw)
            count += len(raw)
    return digest.hexdigest(), count


def cuda_read_prefix(s, req_pool_idx, tokens):
    import torch

    pool = s.token_to_kv_pool_allocator.get_kvcache()
    device = pool._get_key_buffer(pool.start_layer).device
    read_stream = torch.cuda.Stream(device=device)
    read_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(read_stream):
        indices = s.req_to_token_pool.req_to_token[req_pool_idx, :tokens].clone()
        physical = indices.cpu().tolist()
        require(
            len(set(physical)) == tokens and all(v > 0 for v in physical),
            "unsafe Native indices",
        )
        digest, count = digest_pool(pool, indices)
    read_stream.synchronize()
    return physical, digest, count


def attach(s):
    # No imports, mutations, CUDA work or wrapping unless explicitly enabled.
    if os.environ.get("CACHEBRIDGE_GENERATED_BG_WITNESS") != "1":
        return False
    if os.environ.get("CACHEBRIDGE_SERVING_HANDOFF_ROLE") == "source":
        return False
    require(
        os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROLE"] == "target",
        "Target-only witness",
    )
    from pathlib import Path
    import torch
    from cachebridge.backends.sglang.runtime.overlap_bridge import work_state
    from cachebridge.backends.sglang.runtime.resident_session import write

    observer = GeneratedPrefixWitness(s, cuda_read_prefix, work_state)
    original = s.audit_handoff
    root = Path(os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROOT"])

    def rpc(self, parameters):
        action = parameters["action"]
        if action == "control_generated_bg_clear":
            observer.clear()
            return
        if action not in [
            "control_generated_bg_capture",
            "control_generated_bg_verify",
        ]:
            return original(parameters)
        prepared = None
        foreground_indices = None
        error = None
        try:
            prepared = observer.prepare(parameters)
            if action == "control_generated_bg_capture":
                require(observer.saved is None, "previous observation not cleared")
            else:
                observer.verify_identity(prepared, parameters["observation_id"])
                fg = parameters["foreground_history_ids"]
                require(
                    len(fg) == 16384 and all(type(v) is int and v >= 0 for v in fg),
                    "wrong foreground IDs",
                )
                found = self.tree_cache.match_prefix(fg)
                require(
                    len(found.device_indices) == len(fg),
                    "foreground full H not published",
                )
                foreground_indices = found.device_indices.cpu().tolist()
        except Exception as exc:
            error = repr(exc)
        eligible = torch.tensor([int(error is None)], dtype=torch.int32)
        torch.distributed.all_reduce(
            eligible, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        require(
            bool(eligible[0]),
            "all-rank Native witness eligibility rejected: "
            + (error or "peer rejection"),
        )
        signature = (
            parameters["observation_id"],
            prepared["rid"],
            prepared["origin_ids"],
            prepared["resolved_output_ids"],
        )
        peers = [None] * 4
        torch.distributed.all_gather_object(peers, signature, group=self.tp_cpu_group)
        require(
            all(value == signature for value in peers),
            "ranks disagree on resolved Native token context",
        )
        if action == "control_generated_bg_capture":
            receipt = observer.capture(prepared, parameters["observation_id"])
        else:
            receipt = observer.verify(
                prepared, parameters["observation_id"], foreground_indices
            )
        write(
            root
            / f'target-generated-bg-{action.rsplit("_", 1)[1]}-rank{self.tp_rank}.json',
            dict(phase=parameters["phase"], rank=self.tp_rank, **receipt),
        )
        return

    s.audit_handoff = MethodType(rpc, s)
    return True
