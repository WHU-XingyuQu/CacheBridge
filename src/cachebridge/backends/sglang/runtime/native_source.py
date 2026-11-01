"""Read-only native-pool export under the unchanged scheduler-owned Source lifetime."""

import time
import torch
from cachebridge.backends.sglang.runtime.native_channel import (
    NativeSender,
    process_identity,
)
from cachebridge.backends.sglang.runtime.native_cuda import NativeExportCodec


def native_contract(base, p, rank):
    if (
        p.get("conversion_placement") != "target"
        or p.get("source_representation") != "source_native_pages"
    ):
        raise ValueError("native receiver-only scope")
    if len(p["source_worker_roster"]) != 4 or len(p["target_worker_roster"]) != 4:
        raise ValueError("native TP4 generation roster")
    return dict(
        base,
        version=13,
        representation="source_native_pages",
        source_generation=p["source_worker_roster"][rank],
        target_generation=p["target_worker_roster"][rank],
    )


def run_native_source(
    *, p, managed, pool, device, rank, compute_stream, path, contract
):
    if (
        managed is None
        or not managed["lifetime"].active
        or managed["lifetime"].done.is_set()
    ):
        raise RuntimeError("native source lacks scheduler-owned H pin")
    if (pool.layer_num, contract["layers"], contract["heads"], contract["dim"]) != (
        40,
        40,
        2,
        128,
    ):
        raise ValueError("native Qwen14 TP4 geometry")
    if process_identity() != contract["source_generation"]:
        raise RuntimeError("native source generation changed")
    buffers = []
    first = pool._get_key_buffer(pool.start_layer)
    rows = first.shape[0]
    for kind in range(2):
        for layer in range(pool.layer_num):
            t = (pool._get_key_buffer if kind == 0 else pool._get_value_buffer)(
                pool.start_layer + layer
            )
            if (
                t.shape != (rows, 2, 128)
                or not t.is_cuda
                or t.dtype != torch.bfloat16
                or not t.is_contiguous()
                or t.device != first.device
            ):
                raise ValueError("native Source pool ABI")
            flat = t.view(-1)
            assert flat.data_ptr() == t.data_ptr()
            buffers.append(flat)
    # The original source_start waits this stream on already enqueued scheduler
    # work and clones the protected indices. Capture/synchronize only that stream.
    with torch.cuda.device(device), torch.cuda.stream(compute_stream):
        indices = managed["indices"].to("cpu").tolist()
    compute_stream.synchronize()
    if len(indices) != contract["tokens"]:
        raise ValueError("native H descriptor truncated")
    sender = None
    blocks = 0
    try:
        sender = NativeSender(path, contract, NativeExportCodec(device))
        with torch.cuda.device(device), torch.cuda.stream(compute_stream):
            sender.share(buffers, indices, rows)
        for a in range(0, len(indices), 2048):
            b = min(a + 2048, len(indices))
            sender.offer(blocks % 2, a, b)
            blocks += 1
        sender.finish()
        managed["record"].update(
            phase="complete",
            outcome="complete",
            blocks=blocks,
            consumers=[rank],
            certified_data_completion_ns=time.perf_counter_ns(),
            mapper_compute_role="target",
            conversion_placement="target",
            logical_delivery_bytes=len(indices) * 40960,
            source_lease_released=False,
            mapped_output_slots=2,
            source_representation="source_native_pages",
            native_buffer_exports=len(buffers),
            source_pack_operations=0,
            source_metadata_rows=len(indices),
            source_index_snapshot_bytes=managed["indices"].numel()
            * managed["indices"].element_size(),
            source_workspace_used_bytes=0,
            common_inventory_allocation_unchanged=True,
            gpu_observation=[],
            pipeline_wait_timeline=[],
        )
    finally:
        if sender is not None:
            sender.close()
