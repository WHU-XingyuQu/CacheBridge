"""Audited PP1/non-speculative logical reset without allocator cache eviction.

Mirrors Scheduler.flush_cache state reset at SGLang base 8cd34458, but keeps
PyTorch allocator reserves. Deliberately not a general scheduler API.
"""


def reset_prefix_preserving_allocator(s):
    if s.pp_size != 1 or not s.spec_algorithm.is_none():
        raise RuntimeError("Warm reset only supports audited PP1/non-speculative path")
    if s.waiting_queue or not s.running_batch.is_empty() or s.chunked_req is not None:
        raise RuntimeError("Warm reset requires idle scheduler")
    if hasattr(s, "audit_pending") or getattr(s, "audit_transaction_active", False):
        raise RuntimeError("Cannot reset unpublished transaction")
    s.cur_batch = None
    s.last_batch = None
    s.tree_cache.reset()
    if s.grammar_backend:
        s.grammar_backend.reset()
    s.req_to_token_pool.clear()
    s.token_to_kv_pool_allocator.clear()
    s.num_generated_tokens = 0
    s.forward_ct_decode = 0
    s.spec_num_total_accepted_tokens = 0
    s.spec_num_total_forward_ct = 0
    s.cum_spec_accept_length = 0
    s.cum_spec_accept_count = 0
    assert s.tree_cache.total_size() == 0
    return True
