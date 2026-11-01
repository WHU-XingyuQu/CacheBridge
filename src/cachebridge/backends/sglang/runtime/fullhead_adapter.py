"""Live Full-Head installation using the unchanged four-source Dense loop.

One unpublished transaction, scheduler-owned allocation/publication, no worker
collectives or radix mutation. Source uses the existing concurrency_run fanout.
"""

import json
import math
import os
from pathlib import Path
import threading
import time
from types import MethodType

from cachebridge.backends.sglang.runtime.config import BLOCK_TOKENS
from cachebridge.backends.sglang.runtime.reservation_accounting import (
    validate as validate_idle_reservation,
)
from cachebridge.backends.sglang.runtime.resident_session import write


def valid_identity(tx, state, p):
    return bool(
        tx is not None
        and tx["request_id"] == p["request_id"]
        and tx["parameters"]["phase"] == p["transaction_phase"]
        and state["active_request"] == p["request_id"]
        and not state["quarantined"]
    )


def attach(s):
    if os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROLE"] != "target":
        return
    import torch

    root = Path(os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROOT"])
    owner = threading.get_ident()
    pool = s.token_to_kv_pool_allocator.get_kvcache()
    device = pool._get_key_buffer(pool.start_layer).device
    state = s.concurrency_state
    full = {"transaction": None, "records": {}}
    s.fullhead_state = full
    original = s.audit_handoff
    original_memory = s.check_memory

    def vote(*flags):
        tensor = torch.tensor([int(x) for x in flags], dtype=torch.int32)
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MIN, group=s.tp_cpu_group
        )
        return [bool(x) for x in tensor]

    def check_memory(self):
        tx = full["transaction"]
        if tx is None:
            return original_memory()
        if (
            self.waiting_queue
            or not self.running_batch.is_empty()
            or self.chunked_req is not None
        ):
            full["accounting"] = dict(
                status="live_unpublished_reservation_tracked",
                unpublished_tokens=len(tx["indices"]),
                scheduler_owned=True,
            )
        else:
            try:
                full["accounting"] = validate_idle_reservation(self, len(tx["indices"]))
            except BaseException:
                state["quarantined"] = True
                raise

    def worker(tx):
        links = []
        workspace = tx["workspace"]
        error = None
        safe = True
        try:
            with torch.cuda.device(device):
                tx["blocks"] = s.concurrency_receive_blocks(
                    tx["parameters"], tx["indices"], workspace, links
                )
        except BaseException as exc:
            error = repr(exc)
            try:
                workspace.finish()
                for link in links:
                    link.abort()
            except BaseException as cleanup:
                safe = False
                error += "; cleanup=" + repr(cleanup)
        tx.update(
            worker_error=error,
            worker_safe=safe,
            logical_copy_bytes=sum(link.bytes_copied for link in links),
            certified_data_completion_ns=(
                time.perf_counter_ns() if error is None and safe else None
            ),
        )
        tx["done"].set()

    def start(p):
        ids = p["history_ids"]
        block_tokens = p.get("reference_block_tokens", BLOCK_TOKENS)
        local_valid = (
            p.get("mapper_kind") == "Dense"
            and p.get("execution", "double") in ("double", "serial")
            and block_tokens == BLOCK_TOKENS
            and bool(ids)
            and full["transaction"] is None
            and state["active_request"] is None
            and s.source_mapper_state["transaction"] is None
            and not state["quarantined"]
            and p["request_id"] not in state["requests"]
            and len(s.tree_cache.match_prefix(ids).device_indices) == 0
        )
        if not vote(local_valid)[0]:
            raise RuntimeError(
                "all-rank Full-Head admission rejected before allocation"
            )
        indices = s.token_to_kv_pool_allocator.alloc(len(ids))
        if not vote(indices is not None)[0]:
            if indices is not None:
                s.token_to_kv_pool_allocator.free(indices)
            raise RuntimeError("all-rank Full-Head pages unavailable")
        workspace = s.concurrency_workspaces["Dense"]
        # Order workspace streams after already enqueued scheduler work without
        # waiting on an unrelated live decoder or exposing unpublished pages.
        workspace.begin()
        tx = dict(
            request_id=p["request_id"],
            parameters=dict(p),
            indices=indices,
            workspace=workspace,
            done=threading.Event(),
            worker_error=None,
            worker_safe=True,
            blocks=0,
        )
        state["active_request"] = p["request_id"]
        state["requests"][p["request_id"]] = dict(
            request_id=p["request_id"],
            method="DenseFullHeadLive",
            published=False,
            phase="staging",
            pool_tokens_held=len(ids),
            input_slots=2,
            workspace_bytes=workspace.bytes,
        )
        full["transaction"] = tx
        thread = threading.Thread(
            target=worker,
            args=(tx,),
            name=f"fullhead-stage-{s.tp_rank}-{p['request_id']}",
            daemon=True,
        )
        tx["thread"] = thread
        try:
            thread.start()
        except BaseException:
            state["quarantined"] = True
            raise

    def publish(p):
        began = time.perf_counter_ns()
        tx = full["transaction"]
        valid = valid_identity(tx, state, p)
        done = bool(valid and tx["done"].is_set())
        exited = bool(valid and not tx["thread"].is_alive())
        all_valid, all_ready = vote(valid, done and exited)
        if not all_valid:
            raise RuntimeError("all-rank Full-Head publication identity invalid")
        voted = time.perf_counter_ns()
        if all_ready:
            ready, safe = vote(
                tx["worker_error"] is None
                and tx["blocks"]
                == math.ceil(len(tx["parameters"]["history_ids"]) / BLOCK_TOKENS)
                and tx["logical_copy_bytes"]
                == len(tx["parameters"]["history_ids"]) * 2 * 40 * 8 * 128 * 2,
                tx["worker_safe"],
            )
            if not safe:
                state["quarantined"] = True
                raise RuntimeError("Full-Head remote reads/local writes not certified")
            record = state["requests"][p["request_id"]]
            if not ready:
                s.token_to_kv_pool_allocator.free(tx["indices"])
                record.update(
                    phase="rolled_back",
                    outcome="rolled_back",
                    error=tx["worker_error"],
                    pool_tokens_held=0,
                )
                full["transaction"] = None
                state["active_request"] = None
                raise RuntimeError(tx["worker_error"])
            error = None
            try:
                ids = tx["parameters"]["history_ids"]
                if s.tree_cache.insert(ids, tx["indices"]) != 0:
                    raise RuntimeError("Full-Head unexpected existing radix prefix")
                if not torch.equal(
                    s.tree_cache.match_prefix(ids).device_indices, tx["indices"]
                ):
                    raise RuntimeError("Full-Head radix allocation mismatch")
            except BaseException as exc:
                error = repr(exc)
            if not vote(error is None)[0]:
                state["quarantined"] = True
                raise RuntimeError("Full-Head publication failed; retain ownership")
            n = len(tx["parameters"]["history_ids"])
            if tx["blocks"] != math.ceil(n / BLOCK_TOKENS):
                state["quarantined"] = True
                raise RuntimeError("Full-Head incomplete history")
            record.update(
                phase="published",
                outcome="published",
                published=True,
                tokens=n,
                blocks=tx["blocks"],
                sources=list(range(s.tp_size)),
                logical_copy_bytes=tx["logical_copy_bytes"],
                pool_tokens_held=0,
                worker_thread_exited=True,
                full_H_before_query=True,
                certified_data_completion_ns=tx["certified_data_completion_ns"],
                publish_completed_ns=time.perf_counter_ns(),
                execution=tx["parameters"].get("execution", "double"),
            )
            full["records"][p["request_id"]] = dict(record)
            full["transaction"] = None
            state["active_request"] = None
        write(
            root / f"target-fullhead-publication-rank{s.tp_rank}.json",
            dict(
                phase=p["phase"],
                transaction_phase=p["transaction_phase"],
                request_id=p["request_id"],
                rank=s.tp_rank,
                outcome="published" if all_ready else "pending",
                worker_done=done,
                worker_thread_exited=exited,
                all_rank_ready=all_ready,
                all_rank_identity_valid=all_valid,
                certified_data_completion_ns=(
                    tx.get("certified_data_completion_ns") if done else None
                ),
                callback_started_ns=began,
                readiness_vote_finished_ns=voted,
                callback_finished_ns=time.perf_counter_ns(),
                pending_keeps_unpublished_ownership=not all_ready,
            ),
        )

    def rpc(self, p):
        if threading.get_ident() != owner:
            raise RuntimeError("Full-Head control requires scheduler owner")
        action = p["action"]
        if action == "fullhead_stage_start":
            return start(p)
        if action == "fullhead_stage_try_publish":
            return publish(p)
        if full["transaction"] is not None and action not in ("concurrency_status",):
            raise RuntimeError(
                "retain Full-Head unpublished ownership until completion"
            )
        result = original(p)
        if action == "concurrency_status":
            path = root / f"target-concurrency-status-rank{s.tp_rank}.json"
            value = json.loads(path.read_text())
            value["fullhead_records"] = full["records"]
            value["fullhead_unpublished_request"] = (
                full["transaction"]["request_id"]
                if full["transaction"] is not None
                else None
            )
            value["fullhead_accounting"] = full.get("accounting")
            write(path, value)
        if action == "concurrency_clear_records":
            full["records"].clear()
        return result

    s.audit_handoff = MethodType(rpc, s)
    s.check_memory = MethodType(check_memory, s)
