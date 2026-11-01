"""Isolated same-mapper placement control; scheduler owns allocation/publication.

SourceMap delivers64-layer outputs; ReceiverMap delivers40-layer inputs and
performs pipelined reconstruction on the target. All resources are charged.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import socket
import threading
import time
from types import MethodType

import torch

from cachebridge.backends.sglang.runtime.bounded_cachebridge_core import (
    pack_source_block,
)
from cachebridge.backends.sglang.runtime.bounded_handoff_channel import (
    CudaCodec,
    peer_check,
    receive,
    recv_bytes,
    send,
    send_bytes,
)
from cachebridge.backends.sglang.runtime.concurrency_adapter import (
    _history_identity,
    _verify_files,
)
from cachebridge.backends.sglang.runtime import config as _config
from cachebridge.backends.sglang.runtime.reservation_accounting import (
    validate as validate_idle_reservation,
)
from cachebridge.backends.sglang.runtime.resident_session import write
from cachebridge.backends.sglang.runtime.source_read_lifetime import (
    SourceReadLifetime,
    validate_idle_source_pin,
)
from cachebridge.backends.sglang.runtime.gpu_observation import GpuWindows
from cachebridge.backends.sglang.runtime.overlap_bridge import (
    model_runner,
    require_drained,
    attach_work,
    attach_status,
)


TARGET_LAYERS = 64
LOCAL_KV_HEADS = _config.PAIR.kv_heads // _config.PAIR.tensor_parallel_size
HEAD_DIM = 128
SOURCE_BLOCK_TOKENS = 2048


def _mapped_contract(p, rank):
    if p.get("conversion_placement") == "mixed":
        from cachebridge.backends.sglang.runtime.mixed_channel import mixed_contract

        return mixed_contract(p, rank)
    ids = p["history_ids"]
    placement = p.get("conversion_placement", "source")
    if placement not in ("source", "target"):
        raise ValueError("unknown conversion placement")
    layers = TARGET_LAYERS if placement == "source" else 40
    return {
        "version": 12,
        "request": p["request_id"],
        "source_rank": rank,
        "target_rank": rank,
        "slots": 2,
        "method": "CacheBridgePlacement",
        "conversion_placement": placement,
        "mode": "C",
        "tokens": len(ids),
        "block_tokens": SOURCE_BLOCK_TOKENS,
        "layers": layers,
        "heads": LOCAL_KV_HEADS,
        "dim": HEAD_DIM,
        "elements": (
            2 * layers * min(len(ids), SOURCE_BLOCK_TOKENS) * LOCAL_KV_HEADS * HEAD_DIM
        ),
        "history_sha256": _history_identity(ids),
    }


class _PipelinedSourceLease:
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


class _SlotCodec(CudaCodec):
    """Make Sender.offer wait for its slot event, not the whole source GPU."""

    def __init__(self, device, export_stream=None):
        super().__init__(device)
        self.export_stream = export_stream
        self.event = torch.cuda.Event()
        self.recorded = False
        self.exporting = False

    def export(self, value):
        self.exporting = True
        try:
            if self.export_stream is not None:
                with torch.cuda.stream(self.export_stream):
                    return super().export(value)
            return super().export(value)
        finally:
            self.exporting = False

    def record(self, stream):
        self.event.record(stream)
        self.recorded = True

    def synchronize(self):
        if self.exporting and self.export_stream is not None:
            return self.export_stream.synchronize()
        if self.exporting or not self.recorded:
            return torch.cuda.synchronize(self.device)
        return self.event.synchronize()


from cachebridge.backends.sglang.runtime.wait_observation import WaitObservation


class _DoubleBufferedSender:
    """One ordered control stream exporting two independently reusable slots."""

    def __init__(self, path, contract, codecs, timeout=60, wait_observer=None):
        self._wait_log = wait_observer
        self.contract = contract
        self.codecs = codecs
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.cursor = 0
        self.completed = False
        try:
            self.sock.connect(str(path))
            peer_check(self.sock)
            send(self.sock, contract)
            if receive(self.sock) != {"op": "accept"}:
                raise ValueError("consumer rejected double-buffer identity")
        except BaseException:
            self.sock.close()
            raise

    def share(self, storages):
        if len(storages) != 2:
            raise ValueError("double-buffer channel requires two storages")
        for slot, (codec, storage) in enumerate(zip(self.codecs, storages)):
            send_bytes(
                self.sock,
                codec.export(_ipc_storage_view(storage, self.contract["elements"])),
            )
            if receive(self.sock) != {"op": "mapped", "slot": slot}:
                raise ValueError("consumer failed to import mapped-output slot")

    def offer(self, slot, start, stop):
        step = self.contract["block_tokens"]
        if (
            slot != (start // step) % 2
            or start != self.cursor
            or stop != min(start + step, self.contract["tokens"])
        ):
            raise ValueError("double-buffer source block order/span mismatch")
        if self._wait_log is not None:
            self._wait_log.mark("ready_wait_begin", slot=slot, start=start, stop=stop)
        self.codecs[slot].synchronize()
        if self._wait_log is not None:
            self._wait_log.mark(
                "source_ready_certified", slot=slot, start=start, stop=stop
            )
        send(
            self.sock,
            {"op": "ready", "slot": slot, "start": start, "stop": stop},
        )
        if self._wait_log is not None:
            self._wait_log.mark("ready_sent", slot=slot, start=start, stop=stop)
        reply = receive(self.sock)
        expected = {
            "op": "ack",
            "slot": slot,
            "start": start,
            "stop": stop,
            "gpu_reads_complete": True,
        }
        if reply.get("op") == "cancel" and reply.get("gpu_reads_complete") is True:
            raise RuntimeError("consumer cancelled after GPU completion")
        if reply != expected:
            raise RuntimeError("consumer did not certify double-buffer block")
        if self._wait_log is not None:
            self._wait_log.mark("read_ACK_certified", slot=slot, start=start, stop=stop)
        self.cursor = stop

    def finish(self):
        if self.cursor != self.contract["tokens"]:
            raise RuntimeError("cannot finish incomplete double-buffer transfer")
        send(self.sock, {"op": "end"})
        if receive(self.sock) != {"op": "closed", "gpu_reads_complete": True}:
            raise RuntimeError("consumer double-buffer imports remain live")
        self.completed = True

    def close(self):
        self.sock.close()


def _ipc_storage_view(storage, elements):
    """Share exactly the contract span while retaining the reusable slot."""

    flat = storage.reshape(-1)
    if elements <= 0 or elements > flat.numel():
        raise ValueError("IPC contract exceeds mapped-output slot")
    return flat[:elements]


class _DoubleBufferedReceiver:
    """Receive globally ordered blocks from two pre-shared remote slots."""

    def __init__(self, server, contract, codec):
        self.contract = contract
        self.codec = codec
        self.sock, _ = server.accept()
        self.sock.settimeout(server.gettimeout())
        self.remotes = []
        self.cursor = 0
        self.pending = None
        self.closed = False
        self.bytes_copied = 0
        try:
            peer_check(self.sock)
            if receive(self.sock) != contract:
                send(self.sock, {"op": "reject"})
                raise ValueError("source double-buffer contract mismatch")
            send(self.sock, {"op": "accept"})
            for slot in range(2):
                remote = codec.open(recv_bytes(self.sock), contract["elements"])
                self.remotes.append(remote)
                send(self.sock, {"op": "mapped", "slot": slot})
        except BaseException:
            self.sock.close()
            raise

    def next(self, expected_slot):
        if self.pending is not None or self.closed:
            raise RuntimeError("invalid double-buffer receive state")
        message = receive(self.sock)
        step = self.contract["block_tokens"]
        stop = min(self.cursor + step, self.contract["tokens"])
        expected = {
            "op": "ready",
            "slot": expected_slot,
            "start": self.cursor,
            "stop": stop,
        }
        if message != expected:
            raise ValueError("double-buffer target block order/span mismatch")
        start = self.cursor
        n = stop - start
        layers = self.contract["layers"]
        elements = 2 * layers * n * LOCAL_KV_HEADS * HEAD_DIM
        self.codec.select(start, stop)
        value = self.codec.copy(self.remotes[expected_slot], None, elements).view(
            2, layers, n, LOCAL_KV_HEADS, HEAD_DIM
        )
        self.bytes_copied += value.numel() * value.element_size()
        self.pending = (expected_slot, start, stop)
        return start, stop, value

    def acknowledge(self):
        if self.pending is None:
            raise RuntimeError("no double-buffer block to acknowledge")
        self.codec.synchronize()
        slot, start, stop = self.pending
        send(
            self.sock,
            {
                "op": "ack",
                "slot": slot,
                "start": start,
                "stop": stop,
                "gpu_reads_complete": True,
            },
        )
        self.cursor = stop
        self.pending = None

    def finish(self):
        if self.pending is not None or self.cursor != self.contract["tokens"]:
            raise RuntimeError("cannot close incomplete double-buffer receive")
        if receive(self.sock) != {"op": "end"}:
            raise ValueError("unexpected double-buffer terminal message")
        self.codec.synchronize()
        self.remotes = []
        send(self.sock, {"op": "closed", "gpu_reads_complete": True})
        self.closed = True
        self.sock.close()

    def abort(self):
        if self.closed:
            return
        self.codec.synchronize()
        self.remotes = []
        try:
            send(self.sock, {"op": "cancel", "gpu_reads_complete": True})
        finally:
            self.closed = True
            self.sock.close()


def _contiguous_runs(values):
    """Return (logical_start, logical_stop, physical_start) placement runs."""
    values = [int(value) for value in values]
    if not values:
        return []
    runs = []
    logical_start = 0
    physical_start = values[0]
    for logical_stop in range(1, len(values)):
        if values[logical_stop] != values[logical_stop - 1] + 1:
            runs.append((logical_start, logical_stop, physical_start))
            logical_start = logical_stop
            physical_start = values[logical_stop]
    runs.append((logical_start, len(values), physical_start))
    return runs


class _DirectMappedPoolCodec(CudaCodec):
    """P2P-copy mapped KV directly into physical runs of unpublished pages."""

    def __init__(self, device, pool, indices):
        super().__init__(device)
        self.pool = pool
        self.indices = indices
        self.runs = _contiguous_runs(indices.detach().cpu().tolist())
        self.start = None
        self.stop = None
        self.install_stream = torch.cuda.Stream(device=device)
        self.observation = GpuWindows(False)

    @property
    def workspace_bytes(self):
        return 0

    def select(self, start, stop):
        self.start, self.stop = start, stop

    def allocate(self, elements):
        return None

    def copy(self, remote, local, elements):
        if self.start is None or self.stop is None:
            raise RuntimeError("mapped target span was not selected")
        n = self.stop - self.start
        expected = 2 * TARGET_LAYERS * n * LOCAL_KV_HEADS * HEAD_DIM
        if elements != expected:
            raise ValueError("mapped target block geometry mismatch")
        mapped = remote[:elements].view(2, TARGET_LAYERS, n, LOCAL_KV_HEADS, HEAD_DIM)
        with torch.cuda.stream(self.install_stream):
            observation = self.observation.start(
                f"install-{self.start}-{self.stop}", self.install_stream
            )
            for offset, layer in enumerate(
                range(self.pool.start_layer, self.pool.start_layer + TARGET_LAYERS)
            ):
                key_pool = self.pool._get_key_buffer(layer)
                value_pool = self.pool._get_value_buffer(layer)
                for logical_start, logical_stop, physical_start in self.runs:
                    overlap_start = max(self.start, logical_start)
                    overlap_stop = min(self.stop, logical_stop)
                    if overlap_start >= overlap_stop:
                        continue
                    source_start = overlap_start - self.start
                    source_stop = overlap_stop - self.start
                    target_start = physical_start + overlap_start - logical_start
                    target_stop = target_start + overlap_stop - overlap_start
                    key_pool[target_start:target_stop].copy_(
                        mapped[0, offset, source_start:source_stop], non_blocking=True
                    )
                    value_pool[target_start:target_stop].copy_(
                        mapped[1, offset, source_start:source_stop], non_blocking=True
                    )
            self.observation.stop(observation, self.install_stream)
        # The ACK may be sent only after all P2P reads and pool writes complete.
        self.install_stream.synchronize()
        self.start = self.stop = None
        return remote[:elements]

    def synchronize(self):
        self.install_stream.synchronize()


def _mapper_bytes(mapper):
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in mapper.layers
        for tensor in (layer.k_weight, layer.k_bias, layer.v_weight, layer.v_bias)
    )


def attach(s):
    if not hasattr(s, "concurrency_state") or hasattr(s, "source_mapper_state"):
        raise RuntimeError("OB11 requires exactly one reviewed concurrency adapter")
    role = os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROLE"]
    rank = s.tp_rank
    root = Path(os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROOT"])
    pool = s.token_to_kv_pool_allocator.get_kvcache()
    device = pool._get_key_buffer(pool.start_layer).device
    base_state = s.concurrency_state
    original = s.audit_handoff
    original_check_memory = s.check_memory
    artifacts = _config.artifact_contract()
    live_stage_enabled = (
        role == "target" and os.environ.get("CACHEBRIDGE_LIVE_STAGE_ENABLED") == "1"
    )

    mapper = None
    input_storage = None
    output_storages = None
    compute_stream = None
    if role == "source":
        from cachebridge.online.mapping import PageMapper

        item = artifacts["cachebridge"]
        _verify_files(item["mapper_dir"], item["file_hashes"])
        mapper = PageMapper(
            item["mapper_dir"],
            method="cachebridge",
            source_rope_theta=1_000_000.0,
            target_rope_theta=1_000_000.0,
            device=device,
            target_tp_rank=rank,
            target_tp_size=_config.TARGET_TP,
        )
        input_storage = torch.empty(
            2 * 40 * SOURCE_BLOCK_TOKENS * LOCAL_KV_HEADS * HEAD_DIM,
            dtype=pool.dtype,
            device=device,
        )
        output_storages = [
            torch.empty(
                2 * TARGET_LAYERS * SOURCE_BLOCK_TOKENS * LOCAL_KV_HEADS * HEAD_DIM,
                dtype=pool.dtype,
                device=device,
            )
            for _ in range(2)
        ]
        compute_stream = torch.cuda.Stream(device=device)
        gc.collect()
        torch.cuda.empty_cache()

    source_export_storages = None
    receiver_input_storages = None
    receiver_output_storage = None
    if role == "source":
        source_export_storages = [torch.empty_like(input_storage) for _ in range(2)]
    else:
        mapper = s.concurrency_mappers["CacheBridge"]
        compute_stream = torch.cuda.Stream(device=device)
        receiver_input_storages = [
            torch.empty(
                2 * 40 * SOURCE_BLOCK_TOKENS * LOCAL_KV_HEADS * HEAD_DIM,
                dtype=pool.dtype,
                device=device,
            )
            for _ in range(2)
        ]
        receiver_output_storage = torch.empty(
            2 * TARGET_LAYERS * SOURCE_BLOCK_TOKENS * LOCAL_KV_HEADS * HEAD_DIM,
            dtype=pool.dtype,
            device=device,
        )
    torch.cuda.synchronize(device)

    state = {
        "transaction": None,
        "source_transaction": None,
        "records": {},
        "max_staged_transactions": 1,
        "mapper_compute_role": "source",
        "serial_target_publication": True,
        "query_handoff_overlap": True,
        "live_stage_enabled": live_stage_enabled,
    }
    source_producer = ThreadPoolExecutor(max_workers=1) if role == "source" else None

    def idle():
        if (
            s.waiting_queue
            or not s.running_batch.is_empty()
            or s.chunked_req is not None
        ):
            raise RuntimeError("stage ownership changes require an idle scheduler")
        require_drained(s)
        torch.cuda.synchronize(device)

    def target_ownership_boundary():
        if live_stage_enabled:
            # collective_rpc executes on the scheduler thread between decode
            # steps, so allocation/publication remain scheduler-serialized.
            # Do not device-synchronize the unrelated decoder stream.
            return
        idle()

    def vote(ok, safe):
        flags = torch.tensor([int(ok), int(safe)], dtype=torch.int32)
        torch.distributed.all_reduce(
            flags, op=torch.distributed.ReduceOp.MIN, group=s.tp_cpu_group
        )
        return bool(flags[0]), bool(flags[1])

    def check_memory(self):
        managed = state["source_transaction"]
        if role == "source" and managed is not None:
            if (
                s.waiting_queue
                or not s.running_batch.is_empty()
                or s.chunked_req is not None
            ):
                state["idle_accounting"] = dict(
                    status="live_source_pin_tracked",
                    producer_pin_tokens=len(managed["parameters"]["history_ids"]),
                )
                return
            try:
                state["idle_accounting"] = validate_idle_source_pin(
                    self, len(managed["parameters"]["history_ids"])
                )
            except BaseException:
                base_state["quarantined"] = True
                raise
            return
        transaction = state["transaction"]
        if role != "target" or transaction is None:
            return original_check_memory()
        if live_stage_enabled and (
            s.waiting_queue
            or not s.running_batch.is_empty()
            or s.chunked_req is not None
        ):
            state["idle_accounting"] = {
                "status": "live_unpublished_reservation_tracked",
                "unpublished_tokens": len(transaction["indices"]),
                "available_tokens": s.token_to_kv_pool_allocator.available_size(),
                "scheduler_serialized_ownership": True,
            }
            return
        try:
            reserved = len(transaction["indices"])
            state["idle_accounting"] = validate_idle_reservation(self, reserved)
        except BaseException as exc:
            base_state["quarantined"] = True
            state["idle_accounting"] = {
                "status": "quarantined_invariant_failure",
                "error": repr(exc),
                "no_automatic_release": True,
            }

    def source_run(p, managed=None):
        if p.get("conversion_placement") == "mixed":
            from cachebridge.backends.sglang.runtime.mixed_cuda import run_source

            if role != "source":
                raise RuntimeError("mixed Source callback reached wrong role")
            return run_source(
                p=p,
                managed=managed,
                pool=pool,
                mapper=mapper,
                device=device,
                rank=rank,
                input_storage=input_storage,
                raw_storages=source_export_storages,
                mapped_storages=output_storages,
                compute_stream=compute_stream,
                path=s.factor_endpoints[rank, rank],
                contract=_mapped_contract(p, rank),
                slot_codec=_SlotCodec,
            )
        if p.get("source_representation") == "source_native_pages":
            from cachebridge.backends.sglang.runtime.native_source import (
                run_native_source,
                native_contract,
            )

            return run_native_source(
                p=p,
                managed=managed,
                pool=pool,
                device=device,
                rank=rank,
                compute_stream=compute_stream,
                path=s.factor_endpoints[rank, rank],
                contract=native_contract(_mapped_contract(p, rank), p, rank),
            )
        if role != "source" or mapper is None:
            raise RuntimeError("source mapper action reached the wrong role")
        if managed is None:
            idle()
            if base_state["quarantined"] or base_state["active_request"] is not None:
                raise RuntimeError("source mapper state is active or quarantined")
        request_id = p["request_id"]
        if managed is None and request_id in base_state["requests"]:
            raise RuntimeError("source mapper request identity already used")
        pipeline = p.get("source_pipeline", "double")
        if pipeline not in ("double", "serial"):
            raise ValueError("source_pipeline must be double or serial")
        placement = p.get("conversion_placement", "source")
        if placement not in ("source", "target"):
            raise ValueError("unknown conversion placement")
        storages = output_storages if placement == "source" else source_export_storages
        ids = p["history_ids"]
        n = len(ids)
        found = s.tree_cache.match_prefix(ids) if managed is None else managed["found"]
        indices = found.device_indices if managed is None else managed["indices"]
        if len(indices) != n:
            raise RuntimeError("source history is not fully resident")
        if managed is None:
            base_state["active_request"] = request_id
        record = (
            managed["record"]
            if managed is not None
            else {
                "request_id": request_id,
                "history_sha256": _history_identity(ids),
                "method": "CacheBridgeSourceMapped",
                "role": role,
                "phase": "mapping",
                "published": False,
            }
        )
        base_state["requests"][request_id] = record
        lease = _PipelinedSourceLease(
            (
                (lambda: None)
                if managed is not None
                else lambda: s.tree_cache.inc_lock_ref(found.last_device_node)
            ),
            (
                (lambda: None)
                if managed is not None
                else lambda: s.tree_cache.dec_lock_ref(found.last_device_node)
            ),
        )
        link = None
        codecs = []
        futures = [None, None]
        observation = GpuWindows(p.get("instrument_gpu_stages", False))
        _wait_log = WaitObservation(
            p.get("instrument_pipeline_waits", False), request_id, rank
        )
        transport = ThreadPoolExecutor(max_workers=1)
        lease.begin()
        try:
            for slot in range(2):
                codecs.append(
                    _SlotCodec(
                        device,
                        export_stream=compute_stream if managed is not None else None,
                    )
                )
            link = _DoubleBufferedSender(
                s.factor_endpoints[rank, rank],
                _mapped_contract(p, rank),
                codecs,
                wait_observer=_wait_log,
            )
            link.share(storages)
            blocks = 0
            for start in range(0, n, SOURCE_BLOCK_TOKENS):
                stop = min(start + SOURCE_BLOCK_TOKENS, n)
                slot = blocks % 2
                if futures[slot] is not None:
                    _wait_log.mark(
                        "slot_reuse_wait_begin", slot=slot, start=start, stop=stop
                    )
                    futures[slot].result()
                    _wait_log.mark(
                        "slot_reuse_wait_end", slot=slot, start=start, stop=stop
                    )
                    lease.acknowledge(slot, gpu_reads_complete=True)
                _wait_log.mark("block_submit_begin", slot=slot, start=start, stop=stop)
                with torch.cuda.stream(compute_stream):
                    pack_window = observation.start(
                        f"pack-{start}-{stop}", compute_stream
                    )
                    block = pack_source_block(
                        pool,
                        indices,
                        start=start,
                        stop=stop,
                        workspace=(
                            input_storage if placement == "source" else storages[slot]
                        ),
                        block_tokens=SOURCE_BLOCK_TOKENS,
                    )
                    observation.stop(pack_window, compute_stream)
                    if placement == "source":
                        expected = (
                            2,
                            TARGET_LAYERS,
                            stop - start,
                            LOCAL_KV_HEADS,
                            HEAD_DIM,
                        )
                        mapped_output = output_storages[slot][
                            : 2
                            * TARGET_LAYERS
                            * (stop - start)
                            * LOCAL_KV_HEADS
                            * HEAD_DIM
                        ].view(expected)
                        map_window = observation.start(
                            f"map-{start}-{stop}", compute_stream
                        )
                        mapped = mapper.map_page(
                            block,
                            position_start=start,
                            source_head_offset=rank * LOCAL_KV_HEADS,
                            output_device=device,
                            output=mapped_output,
                        )
                        observation.stop(map_window, compute_stream)
                        if (
                            tuple(mapped.shape) != expected
                            or mapped.dtype != pool.dtype
                        ):
                            raise ValueError(
                                "source mapper output geometry/dtype mismatch"
                            )
                    codecs[slot].record(compute_stream)
                _wait_log.mark("block_submit_end", slot=slot, start=start, stop=stop)
                lease.export(slot, start, stop)
                futures[slot] = transport.submit(link.offer, slot, start, stop)
                _wait_log.mark("transport_queued", slot=slot, start=start, stop=stop)
                if pipeline == "serial":
                    # Same two slots, map/pack/channel and GPU-read ACK. Only
                    # prevent submitting block i+1 before block i is consumed.
                    _wait_log.mark(
                        "serial_wait_begin", slot=slot, start=start, stop=stop
                    )
                    futures[slot].result()
                    _wait_log.mark("serial_wait_end", slot=slot, start=start, stop=stop)
                    lease.acknowledge(slot, gpu_reads_complete=True)
                    futures[slot] = None
                blocks += 1
            for slot in range(2):
                if futures[slot] is not None:
                    _wait_log.mark("tail_wait_begin", slot=slot)
                    futures[slot].result()
                    _wait_log.mark("tail_wait_end", slot=slot)
                    lease.acknowledge(slot, gpu_reads_complete=True)
            link.finish()
            record["certified_data_completion_ns"] = time.perf_counter_ns()
            lease.close()
            record.update(
                phase="complete",
                outcome="complete",
                blocks=blocks,
                source_lease_released=managed is None,
                consumers=[rank],
                mapper_compute_role=placement,
                conversion_placement=placement,
                logical_delivery_bytes=(
                    n
                    * 2
                    * (TARGET_LAYERS if placement == "source" else 40)
                    * LOCAL_KV_HEADS
                    * HEAD_DIM
                    * 2
                ),
                input_workspace_bytes=input_storage.numel()
                * input_storage.element_size(),
                mapped_output_workspace_bytes=(
                    sum(
                        value.numel() * value.element_size()
                        for value in output_storages
                    )
                ),
                mapped_output_slots=2,
                source_pipeline=pipeline,
                gpu_observation=observation.complete(),
                pipeline_wait_timeline=_wait_log.complete(),
            )
        except BaseException:
            base_state["quarantined"] = True
            raise
        finally:
            if managed is None:
                base_state["active_request"] = None
            transport.shutdown(wait=True)
            if link is not None and not base_state["quarantined"]:
                link.close()

    def source_background(managed):
        error = None
        try:
            source_run(managed["parameters"], managed=managed)
        except BaseException as exc:
            error = repr(exc)
            base_state["quarantined"] = True
        finally:
            # A success includes each slot event and every remote GPU-copy ACK.
            # Errors retain the real radix pin until process teardown; do not
            # infer remote read completion from a local CUDA synchronization.
            managed["lifetime"].worker_complete(
                reads_and_acks_complete=error is None,
                error=error,
            )

    def source_start(p):
        if role != "source" or mapper is None:
            raise RuntimeError("only source launches its read-only producer")
        if (
            base_state["quarantined"]
            or base_state["active_request"] is not None
            or state["source_transaction"] is not None
        ):
            raise RuntimeError("source producer active or quarantined")
        if p.get("mapper_kind") != "CacheBridge" or p.get(
            "conversion_placement"
        ) not in ("source", "target", "mixed"):
            raise ValueError("invalid frozen-mapper placement")
        if p.get("conversion_placement") == "mixed":
            _mapped_contract(
                p, rank
            )  # Reject unsupported plan before acquiring Source pin.
        request_id = p["request_id"]
        if request_id in base_state["requests"]:
            raise RuntimeError("source producer request identity already used")
        found = s.tree_cache.match_prefix(p["history_ids"])
        if len(found.device_indices) != len(p["history_ids"]):
            raise RuntimeError("source producer requires fully resident history")
        lifetime = SourceReadLifetime(
            lambda: s.tree_cache.inc_lock_ref(found.last_device_node),
            lambda: s.tree_cache.dec_lock_ref(found.last_device_node),
        )
        lifetime.begin()
        record = dict(
            request_id=request_id,
            history_sha256=_history_identity(p["history_ids"]),
            method="CacheBridgeSourceProducer",
            role=role,
            phase="mapping",
            published=False,
            source_lease_released=False,
            scheduler_owned_pin=True,
            background_producer=True,
        )
        managed = dict(
            parameters=dict(p), found=found, record=record, lifetime=lifetime
        )
        base_state["active_request"] = request_id
        base_state["requests"][request_id] = record
        state["source_transaction"] = managed
        try:
            # Snapshot page IDs while pinning prevents eviction/reuse. Capture
            # only scheduler work already enqueued; no device-wide synchronize.
            managed["indices"] = found.device_indices.clone()
            compute_stream.wait_stream(torch.cuda.current_stream(device))
            managed["future"] = source_producer.submit(source_background, managed)
        except BaseException:
            base_state["quarantined"] = True
            raise

    def source_probe(p):
        managed = state["source_transaction"]
        if (
            role != "source"
            or managed is None
            or managed["parameters"]["request_id"] != p["request_id"]
        ):
            raise RuntimeError("source producer probe identity mismatch")
        lifetime = managed["lifetime"]
        write(
            root / f"source-producer-probe-rank{rank}.json",
            dict(
                phase=p["phase"],
                rank=rank,
                request_id=p["request_id"],
                worker_done=lifetime.done.is_set(),
                pin_active=lifetime.active,
                reads_and_acks_complete=(
                    lifetime.safe if lifetime.done.is_set() else False
                ),
                error=lifetime.error if lifetime.done.is_set() else None,
                certified_data_completion_ns=(
                    managed["record"].get("certified_data_completion_ns")
                    if lifetime.done.is_set()
                    else None
                ),
            ),
        )

    def source_finish(p):
        managed = state["source_transaction"]
        if (
            role != "source"
            or managed is None
            or managed["parameters"]["request_id"] != p["request_id"]
        ):
            raise RuntimeError("source producer finish identity mismatch")
        lifetime = managed["lifetime"]
        # This collective and radix release run on the scheduler between steps.
        # Never block the scheduler on the producer or synchronize the decoder.
        ready, safe = vote(
            lifetime.done.is_set(), lifetime.safe and not base_state["quarantined"]
        )
        if not ready or not safe:
            base_state["quarantined"] = True
            raise RuntimeError("source finish lacks certified all-rank completion")
        lifetime.finish()
        managed["record"].update(
            source_lease_released=True,
            phase="complete",
            scheduler_released_pin=True,
            source_pin_release_completed_ns=time.perf_counter_ns(),
        )
        state["records"][p["request_id"]] = dict(managed["record"])
        state["source_transaction"] = None
        base_state["active_request"] = None

    def worker_source(transaction):
        p = transaction["parameters"]
        n = len(p["history_ids"])
        codec = None
        link = None
        error = None
        safe = True
        blocks = 0
        try:
            installation = p.get("installation", "direct")
            if installation not in ("direct", "staged_copy"):
                raise ValueError("unknown Target installation mode")
            codec = _DirectMappedPoolCodec(device, pool, transaction["indices"])
            codec.observation = GpuWindows(p.get("instrument_gpu_stages", False))
            with torch.cuda.device(device):
                link = _DoubleBufferedReceiver(
                    s.factor_listeners[rank], _mapped_contract(p, rank), codec
                )
                for start in range(0, n, SOURCE_BLOCK_TOKENS):
                    stop = min(start + SOURCE_BLOCK_TOKENS, n)
                    observed_start, observed_stop, value = link.next(blocks % 2)
                    if (observed_start, observed_stop) != (start, stop):
                        raise RuntimeError("source-mapped block span mismatch")
                    if tuple(value.shape) != (
                        2,
                        TARGET_LAYERS,
                        stop - start,
                        LOCAL_KV_HEADS,
                        HEAD_DIM,
                    ):
                        raise RuntimeError("source-mapped target geometry changed")
                    blocks += 1
                    link.acknowledge()
                link.finish()
                if installation == "staged_copy":
                    codec.finish_install()
                codec.synchronize()
        except BaseException as exc:
            error = repr(exc)
            try:
                if codec is not None:
                    codec.synchronize()
                if link is not None:
                    link.abort()
            except BaseException as cleanup:
                safe = False
                error += "; cleanup=" + repr(cleanup)
        transaction["certified_data_completion_ns"] = (
            time.perf_counter_ns() if error is None and safe else None
        )
        transaction.update(
            worker_done=True,
            worker_error=error,
            worker_safe=safe,
            blocks=blocks,
            logical_copy_bytes=0 if link is None else link.bytes_copied,
            target_transfer_workspace_bytes=(
                0 if codec is None else codec.workspace_bytes
            ),
            installation=p.get("installation", "direct"),
            additional_local_copy_bytes=0 if codec is None else codec.workspace_bytes,
            target_placement_runs=0 if codec is None else len(codec.runs),
            gpu_observation=(
                []
                if codec is None or error is not None
                else codec.observation.complete()
            ),
        )
        write(
            root / f"{role}-source-mapper-last-worker-rank{rank}.json",
            {
                "request_id": transaction["request_id"],
                "worker_error": error,
                "worker_safe": safe,
                "blocks": blocks,
                "logical_copy_bytes": 0 if link is None else link.bytes_copied,
                "target_transfer_workspace_bytes": (
                    0 if codec is None else codec.workspace_bytes
                ),
                "target_placement_runs": 0 if codec is None else len(codec.runs),
            },
        )
        transaction["done"].set()

    def worker(transaction):
        if transaction["parameters"].get("conversion_placement") == "mixed":
            from cachebridge.backends.sglang.runtime.mixed_cuda import run_receiver

            try:
                run_receiver(
                    transaction,
                    device=device,
                    pool=pool,
                    mapper=mapper,
                    storages=receiver_input_storages,
                    output_storage=receiver_output_storage,
                    compute_stream=compute_stream,
                    listener=s.factor_listeners[rank],
                    contract=_mapped_contract(transaction["parameters"], rank),
                    pool_codec_class=_DirectMappedPoolCodec,
                    rank=rank,
                )
            except BaseException as exc:
                transaction.update(
                    worker_done=True, worker_error=repr(exc), worker_safe=False
                )
            transaction["done"].set()
            return
        if transaction["parameters"].get("conversion_placement", "source") == "source":
            return worker_source(transaction)
        from cachebridge.backends.sglang.runtime.receiver_mapping import (
            run_receiver_mapping,
        )

        try:
            run_receiver_mapping(
                transaction,
                device=device,
                pool=pool,
                mapper=mapper,
                storages=receiver_input_storages,
                output_storage=receiver_output_storage,
                compute_stream=compute_stream,
                listener=s.factor_listeners[rank],
                contract=_mapped_contract(transaction["parameters"], rank),
                receiver_class=_DoubleBufferedReceiver,
                pool_codec_class=_DirectMappedPoolCodec,
                rank=rank,
                heads=LOCAL_KV_HEADS,
            )
        except BaseException as exc:
            transaction.update(
                worker_done=True, worker_error=repr(exc), worker_safe=False
            )
        transaction["done"].set()

    def start(p):
        if role != "target":
            raise RuntimeError("only target owns unpublished source-mapped staging")
        if p.get("conversion_placement") == "mixed":
            _mapped_contract(p, rank)  # Reject unsupported plan before allocation.
        target_ownership_boundary()
        if base_state["quarantined"] or state["transaction"] is not None:
            raise RuntimeError("source-mapped transaction is active or quarantined")
        if p.get("mapper_kind") != "CacheBridge":
            raise ValueError("source mapper stages CacheBridge only")
        ids = p["history_ids"]
        if len(s.tree_cache.match_prefix(ids).device_indices):
            raise RuntimeError("target source-mapped prefix already exists")
        indices = s.token_to_kv_pool_allocator.alloc(len(ids))
        if indices is None:
            raise RuntimeError("target token pool exhausted before staging")
        request_id = p["request_id"]
        if request_id in base_state["requests"]:
            s.token_to_kv_pool_allocator.free(indices)
            raise RuntimeError("source-mapped request identity already used")
        record = {
            "request_id": request_id,
            "method": "CacheBridgeSourceWavefront",
            "phase": "staging",
            "published": False,
            "pool_tokens_held": len(ids),
        }
        base_state["requests"][request_id] = record
        transaction = {
            "request_id": request_id,
            "parameters": p,
            "indices": indices,
            "done": threading.Event(),
            "worker_done": False,
            "worker_error": None,
            "worker_safe": True,
        }
        thread = threading.Thread(
            target=worker,
            args=(transaction,),
            name=f"cachebridge-source-map-stage-{rank}-{request_id}",
            daemon=True,
        )
        transaction["thread"] = thread
        state["transaction"] = transaction
        base_state["active_request"] = request_id
        thread.start()

    def commit(p):
        if role != "target":
            raise RuntimeError("only target publishes source-mapped staging")
        transaction = state["transaction"]
        if transaction is None or transaction["request_id"] != p["request_id"]:
            raise RuntimeError("source-mapped transaction identity mismatch")
        if base_state["quarantined"]:
            raise RuntimeError(
                "quarantined source-mapped ownership needs reconciliation"
            )
        if not transaction["done"].is_set():
            raise RuntimeError(
                "probe all rank workers before scheduler commit; never wait for mapper on scheduler"
            )
        target_ownership_boundary()
        transaction["thread"].join(timeout=600)
        if transaction["thread"].is_alive():
            transaction["worker_error"] = "source-mapped worker did not finish"
            transaction["worker_safe"] = False
        ready, all_safe = vote(
            transaction["worker_error"] is None, transaction["worker_safe"]
        )
        record = base_state["requests"][transaction["request_id"]]
        if not all_safe:
            base_state["quarantined"] = True
            record.update(phase="quarantined", outcome="quarantined")
            raise RuntimeError("cannot certify source-mapped reads/pool writes")
        if not ready:
            s.token_to_kv_pool_allocator.free(transaction["indices"])
            record.update(
                phase="rolled_back",
                outcome="rolled_back",
                error=transaction["worker_error"],
            )
            state["transaction"] = None
            base_state["active_request"] = None
            raise RuntimeError(transaction["worker_error"])
        error = None
        try:
            inserted = s.tree_cache.insert(
                transaction["parameters"]["history_ids"], transaction["indices"]
            )
            if inserted != 0:
                raise RuntimeError(
                    f"unexpected existing source-mapped prefix: {inserted}"
                )
            matched = s.tree_cache.match_prefix(
                transaction["parameters"]["history_ids"]
            )
            if not torch.equal(matched.device_indices, transaction["indices"]):
                raise RuntimeError("published source-mapped allocation differs")
        except BaseException as exc:
            error = repr(exc)
        published, _ = vote(error is None, True)
        if not published:
            base_state["quarantined"] = True
            record.update(phase="quarantined", outcome="publication_failed")
            raise RuntimeError("source-mapped publication failed; ownership retained")
        record.update(
            phase="published",
            outcome="published",
            published=True,
            certified_data_completion_ns=transaction.get(
                "certified_data_completion_ns"
            ),
            publish_completed_ns=time.perf_counter_ns(),
            tokens=len(transaction["parameters"]["history_ids"]),
            blocks=transaction["blocks"],
            logical_copy_bytes=transaction["logical_copy_bytes"],
            mapper_compute_role=transaction["parameters"].get(
                "conversion_placement", "source"
            ),
            conversion_placement=transaction["parameters"].get(
                "conversion_placement", "source"
            ),
            target_mapper_workspace_bytes=transaction.get(
                "target_mapper_workspace_bytes", 0
            ),
            installation=transaction.get("installation", "direct"),
            additional_local_copy_bytes=transaction.get(
                "additional_local_copy_bytes", 0
            ),
            gpu_observation=transaction.get("gpu_observation", []),
            native_read_control=transaction.get("native_read_control"),
            target_transfer_workspace_bytes=transaction[
                "target_transfer_workspace_bytes"
            ],
            target_placement_runs=transaction["target_placement_runs"],
            pool_tokens_held=0,
        )
        state["records"][transaction["request_id"]] = dict(record)
        state["transaction"] = None
        base_state["active_request"] = None

    def augment_status(p):
        original(p)
        path = root / f"{role}-concurrency-status-rank{rank}.json"
        value = json.loads(path.read_text())
        transaction = state["transaction"]
        details = {
            "active_request": (
                transaction["request_id"] if transaction is not None else None
            ),
            "worker_done": (
                transaction["worker_done"] if transaction is not None else None
            ),
            "max_staged_transactions": 1,
            "mapper_compute_role": "source",
            "serial_target_publication": True,
            "query_handoff_overlap": True,
            "live_stage_enabled": live_stage_enabled,
            "completed_request_ids": sorted(state["records"]),
            "idle_accounting": state.get("idle_accounting", {"status": "ordinary"}),
            "source_mapper_parameter_bytes": (
                0 if mapper is None else _mapper_bytes(mapper)
            ),
            "source_input_workspace_bytes": (
                0
                if input_storage is None
                else input_storage.numel() * input_storage.element_size()
            ),
            "source_mapped_output_workspace_bytes": (
                0
                if output_storages is None
                else sum(
                    value.numel() * value.element_size() for value in output_storages
                )
            ),
            "source_mapped_output_slots": 0 if output_storages is None else 2,
            "target_mapper_workspace_bytes": 0,
            "target_transfer_workspace_bytes": 0,
            "target_placement": "segmented_direct_p2p",
        }
        # Keep the OB03 controller/runner status contract while naming the mechanism.
        details["placement_common_inventory"] = {
            "mapper_parameter_bytes": _mapper_bytes(mapper),
            "source_export_slots_bytes": (
                0
                if source_export_storages is None
                else sum(x.numel() * x.element_size() for x in source_export_storages)
            ),
            "receiver_input_slots_bytes": (
                0
                if receiver_input_storages is None
                else sum(x.numel() * x.element_size() for x in receiver_input_storages)
            ),
            "receiver_output_bytes": (
                0
                if receiver_output_storage is None
                else receiver_output_storage.numel()
                * receiver_output_storage.element_size()
            ),
            "both_placement_resources_preloaded": True,
            "minimal_placement_capacity_claim": False,
        }
        details["mapper_compute_role"] = "per_transaction"
        managed = state["source_transaction"]
        details["live_source_producer"] = dict(
            supported=True,
            scheduler_owned_pin=True,
            active_request=(
                None if managed is None else managed["parameters"]["request_id"]
            ),
            worker_done=None if managed is None else managed["lifetime"].done.is_set(),
        )
        value["placement_records"] = {
            request_id: dict(record)
            for request_id, record in base_state["requests"].items()
        }
        value["online_mapper_pipeline"] = details
        value["source_mapper_pipeline"] = details
        write(path, value)

    def rpc(self, p):
        action = p["action"]
        if action == "source_producer_start":
            return source_start(p)
        if action == "source_producer_probe":
            return source_probe(p)
        if action == "source_producer_finish":
            return source_finish(p)
        if action == "online_stage_probe":
            transaction = state["transaction"]
            if (
                role != "target"
                or transaction is None
                or transaction["request_id"] != p["request_id"]
            ):
                raise RuntimeError("placement readiness probe identity mismatch")
            write(
                root / f"target-placement-probe-rank{rank}.json",
                {
                    "phase": p["phase"],
                    "request_id": p["request_id"],
                    "rank": rank,
                    "worker_done": transaction["done"].is_set(),
                    "certified_data_completion_ns": (
                        transaction.get("certified_data_completion_ns")
                        if transaction["done"].is_set()
                        else None
                    ),
                },
            )
            return
        if action == "online_stage_source":
            return source_run(p)
        if action == "online_stage_start":
            return start(p)
        if action == "online_stage_commit":
            return commit(p)
        if action == "concurrency_status":
            return augment_status(p)
        if action == "concurrency_clear_records":
            if (
                state["transaction"] is not None
                or state["source_transaction"] is not None
            ):
                raise RuntimeError("cannot clear an active source-mapped transaction")
            result = original(p)
            state["records"].clear()
            return result
        return original(p)

    s.source_mapper_state = state
    s.source_mapper = mapper
    s.source_mapper_input_storage = input_storage
    s.source_mapper_output_storages = output_storages
    s.source_mapper_original_check_memory = original_check_memory
    s.check_memory = MethodType(check_memory, s)
    s.audit_handoff = MethodType(rpc, s)
    write(
        root / f"{role}-source-mapper-ready-rank{rank}.json",
        {
            "role": role,
            "rank": rank,
            "pid": os.getpid(),
            "mapper_compute_role": "source",
            "mapper_shape_preserved": True,
            "block_tokens": SOURCE_BLOCK_TOKENS,
            "mapped_delivery_layers": TARGET_LAYERS,
            "source_mapper_parameter_bytes": (
                0 if mapper is None else _mapper_bytes(mapper)
            ),
            "source_input_workspace_bytes": (
                0
                if input_storage is None
                else input_storage.numel() * input_storage.element_size()
            ),
            "source_mapped_output_workspace_bytes": (
                0
                if output_storages is None
                else sum(
                    value.numel() * value.element_size() for value in output_storages
                )
            ),
            "source_mapped_output_slots": 0 if output_storages is None else 2,
            "target_mapper_workspace_bytes": 0,
            "target_transfer_workspace_bytes": 0,
            "target_placement": "segmented_direct_p2p",
            "unpublished_pages_enter_idle_invariant": True,
            "live_stage_enabled": live_stage_enabled,
        },
    )
