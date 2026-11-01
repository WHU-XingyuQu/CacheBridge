"""Exact idle accounting for one unpublished target-page reservation."""


def validate(scheduler, reserved):
    if scheduler.is_hybrid or scheduler.enable_hierarchical_cache:
        raise RuntimeError("OB10 reservation accounting requires ordinary MHA KV")
    if (
        scheduler.waiting_queue
        or not scheduler.running_batch.is_empty()
        or scheduler.chunked_req is not None
    ):
        raise RuntimeError("idle invariant invoked with a live request")
    available = scheduler.token_to_kv_pool_allocator.available_size()
    evictable = scheduler.tree_cache.evictable_size()
    protected = scheduler.tree_cache.protected_size()
    if protected:
        raise RuntimeError("unexpected protected radix pages while idle")
    if available + evictable + reserved != scheduler.max_total_num_tokens:
        raise RuntimeError(
            "unaccounted pool pages: "
            f"free={available}, tree_evictable={evictable}, "
            f"unpublished={reserved}, total={scheduler.max_total_num_tokens}"
        )
    if len(scheduler.req_to_token_pool.free_slots) != scheduler.req_to_token_pool.size:
        raise RuntimeError("request-slot leak while an unpublished stage is idle")
    return {
        "status": "explicit_unpublished_reservation_accounted",
        "available_tokens": available,
        "evictable_tokens": evictable,
        "unpublished_tokens": reserved,
    }
