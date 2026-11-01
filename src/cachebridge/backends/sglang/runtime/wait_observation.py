"""Diagnostic host certificates; no synchronization or scheduling decision."""

import threading
import time


class WaitObservation:
    def __init__(self, enabled, request_id, rank):
        self.enabled = bool(enabled)
        self.request_id = request_id
        self.rank = rank
        self.events = []
        self.lock = threading.Lock()

    def mark(self, event, *, slot=None, start=None, stop=None):
        if not self.enabled:
            return
        now = time.perf_counter_ns()
        row = dict(
            event=event,
            host_ns=now,
            request_id=self.request_id,
            rank=self.rank,
            slot=slot,
            start=start,
            stop=stop,
            GPU_device_timestamp=False,
        )
        with self.lock:
            self.events.append(row)

    def complete(self):
        # Caller already completed all transport futures. No waiting, flushing,
        # CUDA work, generation or request-state changes occur here.
        with self.lock:
            return sorted(self.events, key=lambda x: x["host_ns"])
