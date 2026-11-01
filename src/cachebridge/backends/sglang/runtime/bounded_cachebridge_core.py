"""CacheBridge block primitives for R004; not installed in resident workers.

Caller owns the source radix lease and an unpublished target transaction across
the entire operation. Transport/TP publication remain the adapter's obligation.
No CUDA event timing, profiling, event-file polling, or full-history KV copy.
"""

import torch


class SourceLease:
    """Single in-flight block; release is legal only after consumer completion."""

    def __init__(self, acquire, release):
        self.acquire = acquire
        self.release = release
        self.active = False
        self.pending = False
        self.next_start = 0

    def begin(self):
        if self.active:
            raise RuntimeError("source lease already active")
        self.acquire()
        self.active = True
        self.next_start = 0

    def export(self, start, stop):
        if not self.active or self.pending or start != self.next_start or stop <= start:
            raise RuntimeError("block ownership/order violation")
        self.pending = True
        self.next_start = stop

    def acknowledge(self, *, gpu_reads_complete):
        if not self.active or not self.pending or not gpu_reads_complete:
            raise RuntimeError("consumer GPU completion required before buffer reuse")
        self.pending = False

    def close(self):
        if not self.active or self.pending:
            raise RuntimeError("cannot release source lease with outstanding GPU reads")
        self.release()
        self.active = False


@torch.inference_mode()
def pack_source_block(pool, indices, *, start, stop, workspace, block_tokens):
    """Pack real (possibly non-contiguous) pool rows into contiguous tail storage.

    workspace is a one-dimensional, request-charged tensor. Reshape its prefix
    for a tail; slicing a [2,L,b,h,d] tensor along tokens would be non-contiguous
    and violate the existing CUDA IPC transport contract.
    """
    n = stop - start
    if not 0 <= start < stop <= len(indices) or n > block_tokens:
        raise ValueError("invalid block span")
    layers = pool.layer_num
    first = pool._get_key_buffer(pool.start_layer)
    heads, dim = first.shape[1:]
    required = 2 * layers * block_tokens * heads * dim
    if (
        workspace.ndim != 1
        or workspace.numel() != required
        or not workspace.is_contiguous()
    ):
        raise ValueError("workspace must be exactly one contiguous block")
    if (
        workspace.device != first.device
        or workspace.dtype != first.dtype
        or indices.device != first.device
    ):
        raise ValueError(
            "pool, indices and workspace must share device and dtype as applicable"
        )
    if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("one-dimensional integer pool indices required")
    block = workspace[: 2 * layers * n * heads * dim].view(2, layers, n, heads, dim)
    loc = indices[start:stop]
    for i, layer in enumerate(range(pool.start_layer, pool.start_layer + layers)):
        torch.index_select(pool._get_key_buffer(layer), 0, loc, out=block[0, i])
        torch.index_select(pool._get_value_buffer(layer), 0, loc, out=block[1, i])
    return block


@torch.inference_mode()
def map_write_block(
    mapper, block, target_pool, target_indices, *, start, source_head_offset
):
    """Keep existing mapper arithmetic and positions; materialize one output block."""
    n = block.shape[2]
    if start < 0 or start + n > len(target_indices) or mapper.method != "cachebridge":
        raise ValueError("invalid target range or non-CacheBridge method")
    device = target_pool._get_key_buffer(target_pool.start_layer).device
    mapped = mapper.map_page(
        block,
        position_start=start,
        source_head_offset=source_head_offset,
        output_device=device,
    )
    if (
        mapped.shape[:3] != (2, target_pool.layer_num, n)
        or mapped.dtype != target_pool.dtype
    ):
        raise ValueError("mapper output geometry/dtype mismatch")
    loc = target_indices[start : start + n]
    for i, layer in enumerate(
        range(target_pool.start_layer, target_pool.start_layer + target_pool.layer_num)
    ):
        target_pool.set_kv_buffer(
            None, loc, mapped[0, i], mapped[1, i], layer_id_override=layer
        )


def install_unpublished(
    *,
    token_count,
    allocate,
    free,
    blocks,
    write,
    synchronize,
    ready_vote,
    publish,
    cancel_after_blocks=None
):
    """Execute inside ONE target worker handler, not across scheduler idle turns.

    blocks yields (start, stop, consumer_owned_tensor); the adapter handles IPC
    ACKs and source cancellation. ready_vote must run on all target ranks even
    when local work fails. publish must be atomic locally and must not throw
    after taking ownership; asynchronous rank/process failure is outside scope.
    """
    indices = None
    error = None
    count = 0
    cursor = 0
    try:
        indices = allocate(token_count)
        if indices is None:
            raise RuntimeError("target pool allocation failed")
        for start, stop, value in blocks:
            if (
                start != cursor
                or not start < stop <= token_count
                or value.shape[2] != stop - start
            ):
                raise ValueError("missing, duplicate or overlapping block")
            write(value, indices, start)
            synchronize()
            cursor = stop
            count += 1
            if cancel_after_blocks == count:
                raise RuntimeError("controlled cancellation before publication")
        if cursor != token_count:
            raise ValueError("incomplete prefix")
    except Exception as exc:
        error = exc
    # No request pages are published while any rank is still writing.
    try:
        synchronize()
        all_ready = ready_vote(error is None)
    except Exception:
        # A broken distributed group cannot certify global GPU completion.
        # Keep pages owned for recovery; do not pretend this is a safe rollback.
        raise
    if error is not None or not all_ready:
        if indices is not None:
            free(indices)
        raise RuntimeError("unpublished target transaction rolled back") from error
    publish(indices)
    return dict(tokens=cursor, blocks=count, published=True)
