"""Two-slot source lease state machine; callbacks own actual GPU fences.

Use one instance per handoff. This class does not launch GPU work or transport.
"""


class PipelinedSourceLease:
    """Hold one radix lock while at most one export per output slot is live."""

    def __init__(self, acquire, release, slots=2):
        self.acquire = acquire
        self.release = release
        self.pending = [False] * slots
        self.active = False
        self.next_start = 0

    def begin(self):
        if self.active:
            raise RuntimeError("pipelined source lease already active")
        self.acquire()
        self.active = True

    def export(self, slot, start, stop):
        if (
            not self.active
            or self.pending[slot]
            or start != self.next_start
            or stop <= start
        ):
            raise RuntimeError("pipelined block ownership/order violation")
        self.pending[slot] = True
        self.next_start = stop

    def acknowledge(self, slot, *, gpu_reads_complete):
        if not self.active or not self.pending[slot] or not gpu_reads_complete:
            raise RuntimeError("slot-local GPU completion required before reuse")
        self.pending[slot] = False

    def close(self):
        if not self.active or any(self.pending):
            raise RuntimeError("cannot release source lease with live output slots")
        self.release()
        self.active = False
