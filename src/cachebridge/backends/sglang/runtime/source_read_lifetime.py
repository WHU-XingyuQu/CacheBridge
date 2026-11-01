"""One diagnostic source radix pin, with scheduler-owned acquire/release."""

import threading


def validate_idle_source_pin(scheduler, expected_tokens):
    """Account only the producer's real radix pin, retaining exact leak checks."""
    if scheduler.is_hybrid or scheduler.enable_hierarchical_cache:
        raise RuntimeError("source diagnostic requires ordinary nonhierarchical KV")
    if (
        scheduler.waiting_queue
        or not scheduler.running_batch.is_empty()
        or scheduler.chunked_req is not None
    ):
        raise RuntimeError("source idle invariant invoked during live generation")
    available = scheduler.token_to_kv_pool_allocator.available_size()
    evictable = scheduler.tree_cache.evictable_size()
    protected = scheduler.tree_cache.protected_size()
    if (
        protected != expected_tokens
        or available + evictable + protected != scheduler.max_total_num_tokens
    ):
        raise RuntimeError(
            "source pin/pool accounting mismatch: "
            f"free={available}, evictable={evictable}, protected={protected}, "
            f"expected_pin={expected_tokens}, total={scheduler.max_total_num_tokens}"
        )
    if len(scheduler.req_to_token_pool.free_slots) != scheduler.req_to_token_pool.size:
        raise RuntimeError("source request-slot leak while only producer pin is live")
    return dict(
        status="explicit_source_radix_pin_accounted",
        available_tokens=available,
        evictable_tokens=evictable,
        protected_tokens=protected,
        producer_pin_tokens=expected_tokens,
    )


class SourceReadLifetime:
    def __init__(self, acquire, release):
        self.owner = threading.get_ident()
        self.acquire = acquire
        self.release = release
        self.active = False
        self.done = threading.Event()
        self.safe = False
        self.error = None

    def _scheduler(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("radix ownership changes belong to the scheduler")

    def begin(self):
        self._scheduler()
        if self.active or self.done.is_set():
            raise RuntimeError("source lifetime is single-use")
        self.acquire()
        self.active = True

    def worker_complete(self, *, reads_and_acks_complete, error=None):
        if threading.get_ident() == self.owner:
            raise RuntimeError("worker completion must come from the producer")
        if not self.active or self.done.is_set():
            raise RuntimeError("source producer completion identity violation")
        self.safe = bool(reads_and_acks_complete) and error is None
        self.error = error
        self.done.set()

    def finish(self):
        self._scheduler()
        if not self.active or not self.done.is_set() or not self.safe:
            raise RuntimeError("retain radix pin until certified producer completion")
        self.release()
        self.active = False
