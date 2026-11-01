"""One CPU-group readiness vote, then the unchanged outer commit stack.

No worker/Event wait, GPU synchronization, allocator mutation or ACK inference
occurs in this adapter. Pending retains the complete unpublished transaction.
"""

import os
from pathlib import Path
import threading
import time
from types import MethodType
from cachebridge.backends.sglang.runtime import config as _config


def attach(s):
    if os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROLE"] != "target":
        return
    import torch
    from cachebridge.backends.sglang.runtime.resident_session import write

    root = Path(os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROOT"])
    assert (
        s.tp_size == _config.TARGET_TP
        and torch.distributed.get_backend(s.tp_cpu_group) == "gloo"
    )
    owner = threading.get_ident()
    original = s.audit_handoff  # Includes PeerControl's copy receipt checks.

    def rpc(self, p):
        if p["action"] != "online_stage_try_publish":
            return original(p)
        if threading.get_ident() != owner:
            raise RuntimeError("ready/publication requires scheduler owner")
        began = time.perf_counter_ns()
        tx = self.source_mapper_state["transaction"]
        record = self.concurrency_state["requests"].get(p["request_id"])
        valid = (
            tx is not None
            and tx["request_id"] == p["request_id"]
            and tx["parameters"]["phase"] == p["transaction_phase"]
            and not self.concurrency_state["quarantined"]
            and self.concurrency_state["active_request"] == p["request_id"]
            and record is not None
            and record["phase"] == "staging"
            and not record["published"]
        )
        done = bool(valid and tx["done"].is_set())
        exited = bool(valid and not tx["thread"].is_alive())
        # Always the same collective, including invalid/pending local states.
        # A local early return would strand another rank in the commit votes.
        flags = torch.tensor([int(valid), int(done and exited)], dtype=torch.int32)
        torch.distributed.all_reduce(
            flags, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        all_valid, all_ready = bool(flags[0]), bool(flags[1])
        voted = time.perf_counter_ns()
        if not all_valid:
            raise RuntimeError(
                "ready/publication all-rank transaction identity invalid"
            )
        if all_ready:
            # Preserve original worker-safe/error votes, tree insert/match,
            # publication vote, quarantine/rollback and copy-control observer.
            original(dict(p, action="online_stage_commit"))
        write(
            root / f"target-ready-publication-rank{self.tp_rank}.json",
            dict(
                phase=p["phase"],
                transaction_phase=p["transaction_phase"],
                request_id=p["request_id"],
                rank=self.tp_rank,
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
                original_commit_stack_called=all_ready,
                pending_keeps_unpublished_ownership=not all_ready,
            ),
        )

    s.audit_handoff = MethodType(rpc, s)
