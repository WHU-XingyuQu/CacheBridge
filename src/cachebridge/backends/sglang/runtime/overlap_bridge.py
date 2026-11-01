"""Explicit worker interface and read-only drain observation; no GPU waits."""

import json
import os
from pathlib import Path
import threading
from types import MethodType


def model_runner(s):
    worker = s.tp_worker.worker if s.enable_overlap else s.tp_worker
    return worker.model_runner


class WorkLedger:
    def __init__(self, s):
        self.s = s
        self.owner = threading.get_ident()
        self.submitted = 0
        self.resolved = 0
        self.failed = False

    def scheduler(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("Async work accounting belongs to scheduler thread")

    def snapshot(self):
        self.scheduler()
        s = self.s
        queued = len(getattr(s, "result_queue", ())) if s.enable_overlap else 0
        last = getattr(s, "last_batch", None) is not None if s.enable_overlap else False
        current = getattr(s, "cur_batch", None)
        current_live = current is not None and not current.is_empty()
        balanced = self.submitted == self.resolved
        return dict(
            enabled=s.enable_overlap,
            submitted=self.submitted,
            resolved=self.resolved,
            failed=self.failed,
            queued_results=queued,
            last_batch_present=last,
            current_batch_live=current_live,
            drained=not self.failed
            and balanced
            and queued == 0
            and not last
            and not current_live,
        )


def work_state(s):
    return s.overlap_work_ledger.snapshot()


def require_drained(s):
    # The original idle guards remain at their callers. This is additive.
    value = work_state(s)
    if not value["drained"]:
        raise RuntimeError("Idle-only operation requires resolved asynchronous work")
    return value


def attach_work(s):
    ledger = WorkLedger(s)
    s.overlap_work_ledger = ledger
    worker = s.tp_worker
    if s.enable_overlap:
        enqueue = worker.forward_batch_generation
        resolve = worker.resolve_last_batch_result

        def submitted(self, *args, **kwargs):
            ledger.scheduler()
            ledger.submitted += 1
            try:
                return enqueue(*args, **kwargs)
            except BaseException:
                ledger.failed = True
                raise

        def resolved(self, *args, **kwargs):
            ledger.scheduler()
            if ledger.resolved >= ledger.submitted:
                ledger.failed = True
                raise RuntimeError("Resolved work has no submitted identity")
            try:
                result = resolve(*args, **kwargs)
            except BaseException:
                ledger.failed = True
                raise
            ledger.resolved += 1
            return result

        worker.forward_batch_generation = MethodType(submitted, worker)
        worker.resolve_last_batch_result = MethodType(resolved, worker)


def attach_status(s):
    from cachebridge.backends.sglang.runtime.resident_session import write

    root = Path(os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROOT"])
    role = os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROLE"]
    original = s.audit_handoff

    def rpc(self, p):
        result = original(p)
        if p["action"] == "concurrency_status":
            path = root / f"{role}-concurrency-status-rank{self.tp_rank}.json"
            value = json.loads(path.read_text())
            value["overlap_work_state"] = work_state(self)
            write(path, value)
        return result

    s.audit_handoff = MethodType(rpc, s)
