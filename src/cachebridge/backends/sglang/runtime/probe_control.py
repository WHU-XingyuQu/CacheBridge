"""Isolate read-only probe RPC completion; retain every GPU ownership fence."""

import json
import os
from pathlib import Path
import threading
import time
from types import MethodType
from cachebridge.backends.sglang.runtime import config as _config
import torch


def attach(s):
    from sglang.srt.managers.io_struct import RpcReqInput, RpcReqOutput
    from cachebridge.backends.sglang.runtime.resident_session import write

    root = Path(os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROOT"])
    role = os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROLE"]
    assert s.pp_size == 1 and s.tp_size == _config.TARGET_TP
    assert torch.distributed.get_world_size() == s.tp_size
    assert torch.distributed.get_backend(s.tp_cpu_group) == "gloo"
    assert torch.distributed.get_world_size(s.tp_cpu_group) == s.tp_size
    assert torch.distributed.get_process_group_ranks(s.tp_cpu_group) == list(
        range(s.tp_size)
    )
    scheduler_thread = threading.get_ident()
    original_handler = s.handle_rpc_request
    records = []

    def handle(self, req):
        p = req.parameters
        if req.method != "audit_handoff" or p.get("action") not in (
            "source_producer_probe",
            "online_stage_probe",
        ):
            return original_handler(req)
        assert threading.get_ident() == scheduler_thread
        backend = p.get("probe_completion_backend", "default")
        if backend not in ("default", "cpu"):
            raise RuntimeError("unknown probe completion backend")
        begin = time.perf_counter_ns()
        error = None
        try:
            # Same audited read-only callback and existing receipt files.
            getattr(self, req.method)(p)
        except Exception as exc:
            error = exc
        callback_finished = time.perf_counter_ns()
        if backend == "cpu":
            torch.distributed.barrier(group=self.tp_cpu_group)
        else:
            # Original SGLang default NCCL/WORLD barrier semantics.
            torch.distributed.barrier()
        finished = time.perf_counter_ns()
        records.append(
            dict(
                request_id=p["request_id"],
                phase=p["phase"],
                action=p["action"],
                rank=self.tp_rank,
                backend=backend,
                callback_started_ns=begin,
                callback_finished_ns=callback_finished,
                barrier_finished_ns=finished,
                callback_success=error is None,
            )
        )
        return RpcReqOutput(error is None, "" if error is None else str(error))

    s.handle_rpc_request = MethodType(handle, s)
    matches = [
        i
        for i, (ty, fn) in enumerate(s._request_dispatcher._mapping)
        if ty is RpcReqInput
    ]
    assert len(matches) == 1
    s._request_dispatcher._mapping[matches[0]] = (RpcReqInput, s.handle_rpc_request)
    original_rpc = s.audit_handoff

    def rpc(self, p):
        result = original_rpc(p)
        if p["action"] == "concurrency_status":
            path = root / f"{role}-concurrency-status-rank{self.tp_rank}.json"
            value = json.loads(path.read_text())
            # Outside timing; no per-probe extra file I/O or CUDA observation.
            value["probe_control_receipts"] = list(records)
            write(path, value)
        return result

    s.audit_handoff = MethodType(rpc, s)
