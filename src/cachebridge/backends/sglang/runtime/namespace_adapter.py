"""Target radix namespaces: cache keys only, never model input token changes."""

from contextlib import contextmanager
from types import MethodType
import hashlib
import json
import os
from pathlib import Path
import threading


def install_tree(tree):
    assert not hasattr(tree, "request_namespace")
    local = threading.local()

    @contextmanager
    def namespace(value):
        prior = getattr(local, "namespace", None)
        local.namespace = value
        try:
            yield
        finally:
            local.namespace = prior

    def key_with_namespace(key, rid=None):
        ns = rid if rid is not None else getattr(local, "namespace", None)
        if isinstance(ns, str) and ns.startswith("cachebridge:"):
            ns = ns[len("cachebridge:") :].split("::", 1)[0]
        if not key or ns is None:
            return key
        result = list(key)
        assert isinstance(result[0], int), "Only raw token keys enter namespace wrapper"
        result[0] = ("cachebridge-request", str(ns), result[0])
        return result

    original_match, original_insert = tree.match_prefix, tree.insert

    def match(self, key, **kwargs):
        return original_match(key_with_namespace(key, kwargs.get("rid")), **kwargs)

    def insert(self, key, value=None):
        return original_insert(key_with_namespace(key), value)

    tree.match_prefix = MethodType(match, tree)
    tree.insert = MethodType(insert, tree)
    tree.request_namespace = namespace
    for name in ("cache_finished_req", "cache_unfinished_req"):
        original = getattr(tree, name)

        def wrapped(self, req, _original=original):
            with namespace(req.rid):
                return _original(req)

        setattr(tree, name, MethodType(wrapped, tree))


def attach(s):
    if os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROLE"] != "target":
        return
    from sglang.srt.managers.schedule_batch import Req
    from cachebridge.backends.sglang.runtime.overlap_bridge import require_drained
    import torch

    assert s.server_args.schedule_policy == "fcfs"
    assert not s.enable_kv_cache_events and not s.enable_metrics
    tree = s.tree_cache
    install_tree(tree)
    assert not getattr(Req, "_cachebridge_request_namespace", False)
    original_init = Req.init_next_round_input

    def init(req, tree_cache=None):
        if tree_cache is not None and hasattr(tree_cache, "request_namespace"):
            with tree_cache.request_namespace(req.rid):
                return original_init(req, tree_cache)
        return original_init(req, tree_cache)

    Req.init_next_round_input = init
    Req._cachebridge_request_namespace = True
    original_rpc = s.audit_handoff
    root = Path(os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROOT"])

    def rpc(self, p):
        namespace = p.get("cache_namespace", p.get("request_id"))
        with tree.request_namespace(namespace):
            if p["action"] == "poisson_cache_probe":
                rows = []
                for item in p["requests"]:
                    with tree.request_namespace(item["request_id"]):
                        match = tree.match_prefix(item["history_ids"])
                    rows.append(
                        dict(
                            request_id=item["request_id"],
                            resident_tokens=len(match.device_indices),
                        )
                    )
                (root / f"target-poisson-probe-rank{s.tp_rank}.json").write_text(
                    json.dumps(
                        dict(
                            phase=p["phase"],
                            rank=s.tp_rank,
                            requests=rows,
                            free_tokens=s.token_to_kv_pool_allocator.available_size(),
                        )
                    )
                    + "\n"
                )
                return
            if p["action"] in ("online_stage_start", "fullhead_stage_start"):
                # Identical bounded admission limits keep live/pending request KV
                # below half the pool. Reclaim only unprotected completed cache.
                required = len(p["history_ids"]) + 65536
                available = s.token_to_kv_pool_allocator.available_size()
                if available < required:
                    tree.evict(required - available)
                assert s.token_to_kv_pool_allocator.available_size() >= required
            return original_rpc(p)

    s.audit_handoff = MethodType(rpc, s)
