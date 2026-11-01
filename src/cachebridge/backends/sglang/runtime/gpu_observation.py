"""CUDA stream windows only in separate diagnostics, never normal timings."""

import torch


class GpuWindows:
    def __init__(self, enabled):
        self.enabled = bool(enabled)
        self.pending = []

    def start(self, label, stream):
        if not self.enabled:
            return None
        begin = torch.cuda.Event(enable_timing=True)
        begin.record(stream)
        return (label, begin)

    def stop(self, handle, stream):
        if handle is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record(stream)
        self.pending.append((handle[0], handle[1], end))

    def complete(self):
        if not self.enabled:
            return []
        # Callers already certify the associated stream/copy completion.
        if any(not end.query() for _, _, end in self.pending):
            raise RuntimeError("GPU stage window read before completion")
        return [
            dict(
                label=label,
                stream_elapsed_ms=begin.elapsed_time(end),
                includes_stream_wait_and_host_launch_gaps=True,
                kernel_service_time_claim=False,
            )
            for label, begin, end in self.pending
        ]
