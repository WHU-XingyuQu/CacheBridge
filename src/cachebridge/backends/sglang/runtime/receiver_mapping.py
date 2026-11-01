"""Receiver placement: copy source KV ahead of local mapping, with safe reuse.

The I/O thread alone owns the socket; its ACK certifies remote reads. Local
slots have a separate lifetime through map/install completion. No collectives,
radix mutation or page allocation run on either background worker.
"""

import queue
import threading
import time
from cachebridge.backends.sglang.runtime.gpu_observation import GpuWindows
import torch
from cachebridge.backends.sglang.runtime.bounded_handoff_channel import CudaCodec


class LocalSlotOwnership:
    def __init__(self, slots=2):
        self.free = [threading.Event() for _ in range(slots)]
        for event in self.free:
            event.set()
        self.cancelled = threading.Event()

    def acquire(self, slot):
        if not self.free[slot].wait(timeout=60):
            raise TimeoutError("receiver local input slot remains in use")
        if self.cancelled.is_set():
            raise RuntimeError("receiver mapping cancelled")
        self.free[slot].clear()

    def release(self, slot, *, compute_reads_complete):
        if not compute_reads_complete or self.free[slot].is_set():
            raise RuntimeError("receiver local slot requires completed map/install")
        self.free[slot].set()

    def cancel_after_completion(self):
        self.cancelled.set()
        for event in self.free:
            event.set()


class InputCopyCodec(CudaCodec):
    def __init__(self, device, storages, heads, ownership):
        super().__init__(device)
        self.storages = storages
        self.heads = heads
        self.ownership = ownership
        self.copy_stream = torch.cuda.Stream(device=device)
        self.start = self.stop = None

    @property
    def workspace_bytes(self):
        return sum(x.numel() * x.element_size() for x in self.storages)

    def select(self, start, stop):
        self.start, self.stop = start, stop

    def copy(self, remote, local, elements):
        del local
        slot = (self.start // 2048) % 2
        self.ownership.acquire(slot)
        n = self.stop - self.start
        if elements != 2 * 40 * n * self.heads * 128:
            raise ValueError("receiver source KV span mismatch")
        value = self.storages[slot][:elements].view(2, 40, n, self.heads, 128)
        with torch.cuda.stream(self.copy_stream):
            value.copy_(remote[:elements].view_as(value), non_blocking=True)
        self.copy_stream.synchronize()
        return value

    def synchronize(self):
        self.copy_stream.synchronize()


def certify_receiver_counts(
    *, tokens, heads, remote_bytes, local_install_bytes, blocks, local_blocks
):
    expected_remote = tokens * 2 * 40 * heads * 128 * 2
    expected_local = tokens * 2 * 64 * heads * 128 * 2
    expected_blocks = (tokens + 2047) // 2048
    if (
        remote_bytes != expected_remote
        or local_install_bytes != expected_local
        or blocks != expected_blocks
        or local_blocks != expected_blocks
    ):
        raise RuntimeError("receiver raw delivery/local installation counts disagree")


def run_receiver_mapping(
    transaction,
    *,
    device,
    pool,
    mapper,
    storages,
    output_storage,
    compute_stream,
    listener,
    contract,
    receiver_class,
    pool_codec_class,
    rank,
    heads,
):
    ownership = LocalSlotOwnership()
    p = transaction["parameters"]
    native = p.get("source_representation") == "source_native_pages"
    if native or p.get("raw_input_copy") == "target_peer":
        from cachebridge.backends.sglang.runtime.native_cuda import TargetPeerInputCodec

        codec = TargetPeerInputCodec(device, storages, heads, ownership)
    else:
        codec = InputCopyCodec(device, storages, heads, ownership)
    if native:
        from cachebridge.backends.sglang.runtime.native_channel import NativeReceiver
        from cachebridge.backends.sglang.runtime.native_source import native_contract

        contract = native_contract(contract, p, rank)
        receiver_class = NativeReceiver
    installer = pool_codec_class(device, pool, transaction["indices"])
    observation = GpuWindows(
        transaction["parameters"].get("instrument_gpu_stages", False)
    )
    installer.observation = observation
    # Map and segmented local pool writes are ordered on the same stream.
    installer.install_stream = compute_stream
    ready = queue.Queue(maxsize=2)
    result = {"error": None, "safe": True, "bytes": 0}
    n = contract["tokens"]

    def receive_blocks():
        link = None
        try:
            with torch.cuda.device(device):
                link = receiver_class(listener, contract, codec)
                for start in range(0, n, 2048):
                    if ownership.cancelled.is_set():
                        raise RuntimeError("receiver mapping consumer cancelled")
                    slot = (start // 2048) % 2
                    a, b, value = link.next(slot)
                    # Remote export may be reused after copy; local slot may not.
                    link.acknowledge()
                    ready.put((a, b, slot, value), timeout=60)
                link.finish()
                result["bytes"] = link.bytes_copied
        except BaseException as exc:
            result["error"] = repr(exc)
            try:
                codec.synchronize()
                if link is not None:
                    link.abort()
            except BaseException as cleanup:
                result["safe"] = False
                result["error"] += "; cleanup=" + repr(cleanup)
            try:
                ready.put(("error", result["error"]), timeout=1)
            except queue.Full:
                pass

    reader = threading.Thread(
        target=receive_blocks, daemon=True, name=f"receiver-map-copy-{rank}"
    )
    reader.start()
    blocks = 0
    error = None
    try:
        with torch.cuda.device(device):
            for start in range(0, n, 2048):
                item = ready.get(timeout=120)
                if item[0] == "error":
                    raise RuntimeError(item[1])
                a, b, slot, value = item
                if a != start or b != min(start + 2048, n) or slot != blocks % 2:
                    raise RuntimeError("receiver mapping sequence mismatch")
                elements = 2 * 64 * (b - a) * heads * 128
                output = output_storage[:elements].view(2, 64, b - a, heads, 128)
                with torch.cuda.stream(compute_stream):
                    map_window = observation.start(f"map-{a}-{b}", compute_stream)
                    mapped = mapper.map_page(
                        value,
                        position_start=a,
                        source_head_offset=rank * heads,
                        output_device=device,
                        output=output,
                    )
                    observation.stop(map_window, compute_stream)
                    if mapped.shape != output.shape or mapped.dtype != pool.dtype:
                        raise RuntimeError("receiver mapper output contract changed")
                    installer.select(a, b)
                    installer.copy(mapped.reshape(-1), None, elements)
                compute_stream.synchronize()
                ownership.release(slot, compute_reads_complete=True)
                blocks += 1
    except BaseException as exc:
        error = repr(exc)
    finally:
        # Wake any backpressured copy only once all local reads/writes completed.
        try:
            compute_stream.synchronize()
            ownership.cancel_after_completion()
        except BaseException as cleanup:
            result["safe"] = False
            error = (error or "") + "; compute_cleanup=" + repr(cleanup)
        reader.join(timeout=120)
        if reader.is_alive():
            result["safe"] = False
            error = (error or "") + "; copy_thread_did_not_finish"
    error = error or result["error"]
    if error is None and result["safe"]:
        try:
            certify_receiver_counts(
                tokens=n,
                heads=heads,
                remote_bytes=result["bytes"],
                local_install_bytes=installer.local_install_bytes,
                blocks=blocks,
                local_blocks=installer.blocks_completed,
            )
        except BaseException as invalid:
            # Original scheduler commit votes consume worker_error before any
            # publication. No rank-local early raise strands its peers.
            error = repr(invalid)
    if hasattr(codec, "receipt"):
        transaction["native_read_control"] = codec.receipt()
    transaction["target_local_install_bytes"] = installer.local_install_bytes
    transaction["certified_data_completion_ns"] = (
        time.perf_counter_ns() if error is None and result["safe"] else None
    )
    transaction.update(
        worker_error=error,
        worker_safe=result["safe"],
        blocks=blocks,
        logical_copy_bytes=result["bytes"],
        target_transfer_workspace_bytes=codec.workspace_bytes,
        target_mapper_workspace_bytes=output_storage.numel()
        * output_storage.element_size(),
        target_placement_runs=len(installer.runs),
        conversion_placement="target",
        worker_done=True,
    )
    if error is None and result["safe"]:
        transaction["gpu_observation"] = observation.complete()
