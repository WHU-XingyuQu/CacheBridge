"""Ordinary Target decode graph; same capture inventory, scheduler-only switches."""

import json
import os
from pathlib import Path
import threading
from types import MethodType
from cachebridge.backends.sglang.runtime import config as _config
from cachebridge.backends.sglang.runtime.overlap_bridge import (
    model_runner,
    require_drained,
    attach_work,
    attach_status,
)


class NativeDecodeControl:
    def __init__(self, scheduler):
        self.s = scheduler
        self.runner = model_runner(scheduler)
        self.captured = self.runner.cuda_graph_runner
        if self.captured is None or set(self.captured.graphs) != set(
            range(1, _config.MAX_RUNNING_REQUESTS + 1)
        ):
            raise RuntimeError(
                "requires ordinary Target graphs captured for every batch1..MAX_RUNNING_REQUESTS"
            )
        self.lock = threading.RLock()
        self.owner = threading.get_ident()
        self.mode = "graph"
        self.switches = 0
        self._reset()

    def _reset(self):
        self.counts = dict(
            decode_forwards=0,
            decode_graph_replays=0,
            decode_eager_forwards=0,
            decode_token_steps=0,
            extend_forwards=0,
        )
        self.batches = {
            k: {i: 0 for i in range(1, _config.MAX_RUNNING_REQUESTS + 1)}
            for k in ["decode", "graph", "eager"]
        }

    def set_mode(self, mode):
        with self.lock:
            s = self.s
            if threading.get_ident() != self.owner:
                raise RuntimeError("mode change requires scheduler thread")
            if mode not in ("eager", "graph"):
                raise RuntimeError("unknown Target decode mode")
            if (
                s.waiting_queue
                or not s.running_batch.is_empty()
                or s.chunked_req is not None
            ):
                raise RuntimeError("mode change requires Target idle")
            require_drained(s)
            if any(
                s.source_mapper_state[k] is not None
                for k in ["source_transaction", "transaction"]
            ):
                raise RuntimeError("mode change during mapper transaction")
            if (
                s.concurrency_state["quarantined"]
                or s.concurrency_state["active_request"] is not None
                or s.tree_cache.protected_size()
            ):
                raise RuntimeError("mode change with protected/quarantined ownership")
            self.runner.cuda_graph_runner = self.captured if mode == "graph" else None
            self.mode = mode
            self.switches += 1
            self._reset()

    def observe(self, batch, result):
        with self.lock:
            if batch.forward_mode.is_decode():
                bs = int(batch.batch_size)
                if bs not in range(1, _config.MAX_RUNNING_REQUESTS + 1):
                    raise RuntimeError(
                        "actual Target decode exceeds configured request capacity"
                    )
                replay = bool(result[1])
                key = "graph" if replay else "eager"
                if replay != (self.mode == "graph"):
                    raise RuntimeError(
                        "actual ordinary Target graph execution differs from configured mode"
                    )
                self.counts["decode_forwards"] += 1
                self.counts[
                    "decode_graph_replays" if replay else "decode_eager_forwards"
                ] += 1
                self.counts["decode_token_steps"] += bs
                self.batches["decode"][bs] += 1
                self.batches[key][bs] += 1
            elif batch.forward_mode.is_extend():
                if result[1]:
                    raise RuntimeError(
                        "ordinary query/prefill unexpectedly graph replayed"
                    )
                self.counts["extend_forwards"] += 1

    def receipt(self):
        with self.lock:
            return dict(
                mode=self.mode,
                **self.counts,
                switches=self.switches,
                batch_histograms={k: dict(v) for k, v in self.batches.items()},
                captured_graph_batch_sizes=sorted(self.captured.graphs),
                captured_graph_object_id=id(self.captured),
                captured_graph_retained_in_both_modes=True,
                cuda_graph_mem_usage_GiB=self.runner.cuda_graph_mem_usage,
                replay_pointer_enabled=self.runner.cuda_graph_runner is self.captured,
                ordinary_decode_only=True,
                mapper_graph=False,
            )


def attach(s):
    if os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROLE"] != "target":
        return
    from cachebridge.backends.sglang.runtime.resident_session import write

    root = Path(os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROOT"])
    control = NativeDecodeControl(s)
    original_forward = control.runner._forward_raw

    def forward(self, forward_batch, *args, **kwargs):
        result = original_forward(forward_batch, *args, **kwargs)
        control.observe(forward_batch, result)
        return result

    control.runner._forward_raw = MethodType(forward, control.runner)
    original_rpc = s.audit_handoff

    def rpc(self, p):
        if p["action"] == "native_target_graph_mode":
            control.set_mode(p["mode"])
            write(
                root / f"target-native-graph-mode-rank{self.tp_rank}.json",
                dict(phase=p["phase"], rank=self.tp_rank, **control.receipt()),
            )
            return
        result = original_rpc(p)
        if p["action"] == "concurrency_status":
            path = root / f"target-concurrency-status-rank{self.tp_rank}.json"
            value = json.loads(path.read_text())
            value["native_decode_control"] = control.receipt()
            write(path, value)
        return result

    s.audit_handoff = MethodType(rpc, s)
