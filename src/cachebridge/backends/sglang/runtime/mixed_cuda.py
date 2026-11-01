"""Existing CUDA packing/mapper/peer installers behind typed block drivers.

No new storage banks, mapper or kernel. Actual CUDA behavior remains unverified
until the separate eight-GPU functional protocol runs on an installed candidate.
"""

import time
import torch
from cachebridge.backends.sglang.runtime.bounded_cachebridge_core import (
    pack_source_block,
)
from cachebridge.backends.sglang.runtime.receiver_mapping import (
    InputCopyCodec,
    LocalSlotOwnership,
)
from cachebridge.backends.sglang.runtime.mixed_channel import (
    RAW,
    MAPPED,
    Sender,
    Receiver,
    blocks,
)
from cachebridge.backends.sglang.runtime.mixed_execution import (
    source_pipeline,
    receiver_pipeline,
)


class SourceOperations:
    def __init__(
        self,
        *,
        pool,
        indices,
        mapper,
        device,
        rank,
        input_storage,
        storages,
        stream,
        codecs
    ):
        self.pool = pool
        self.indices = indices
        self.mapper = mapper
        self.device = device
        self.rank = rank
        self.input = input_storage
        self.storages = storages
        self.stream = stream
        self.codecs = codecs

    def prepare(self, b):
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            workspace = self.input if b.mode == MAPPED else self.storages[RAW][b.slot]
            value = pack_source_block(
                self.pool,
                self.indices,
                start=b.start,
                stop=b.stop,
                workspace=workspace,
                block_tokens=2048,
            )
            if b.mode == MAPPED:
                elements = 2 * 64 * (b.stop - b.start) * 2 * 128
                out = self.storages[MAPPED][b.slot][:elements].view(
                    2, 64, b.stop - b.start, 2, 128
                )
                result = self.mapper.map_page(
                    value,
                    position_start=b.start,
                    source_head_offset=self.rank * 2,
                    output_device=self.device,
                    output=out,
                )
                if result.shape != out.shape or result.dtype != self.pool.dtype:
                    raise ValueError("mixed Source mapper contract changed")
            self.codecs[b.bank][b.slot].record(self.stream)


def run_source(
    *,
    p,
    managed,
    pool,
    mapper,
    device,
    rank,
    input_storage,
    raw_storages,
    mapped_storages,
    compute_stream,
    path,
    contract,
    slot_codec
):
    if managed is None:
        raise RuntimeError("mixed candidate requires scheduler-owned async Source pin")
    storages = {RAW: raw_storages, MAPPED: mapped_storages}
    codecs = {
        mode: [slot_codec(device, export_stream=compute_stream) for _ in range(2)]
        for mode in (RAW, MAPPED)
    }
    operations = SourceOperations(
        pool=pool,
        indices=managed["indices"],
        mapper=mapper,
        device=device,
        rank=rank,
        input_storage=input_storage,
        storages=storages,
        stream=compute_stream,
        codecs=codecs,
    )
    with torch.cuda.device(device):
        link = Sender(path, contract, codecs)
        try:
            link.share(storages)
        except BaseException:
            link.close()
            raise
        result = source_pipeline(link, operations)
    managed["record"].update(
        result,
        phase="mapped",
        outcome="complete",
        mapper_compute_role="mixed",
        conversion_placement="mixed",
        source_lease_released=False,
        consumers=[rank],
        certified_data_completion_ns=time.perf_counter_ns(),
        input_workspace_bytes=input_storage.numel() * input_storage.element_size(),
        mapped_output_workspace_bytes=sum(
            x.numel() * x.element_size() for x in mapped_storages
        ),
        mapped_output_slots=2,
        gpu_observation=[],
        pipeline_wait_timeline=[],
    )


class TypedRawCodec(InputCopyCodec):
    def select_block(self, b):
        if b.mode != RAW:
            raise ValueError("raw input codec received mapped block")
        self.block = b

    def copy(self, remote, local, elements):
        del local
        b = self.block
        self.ownership.acquire(b.slot)
        expected = 2 * 40 * (b.stop - b.start) * self.heads * 128
        if elements != expected:
            raise ValueError("mixed raw source geometry")
        value = self.storages[b.slot][:elements].view(
            2, 40, b.stop - b.start, self.heads, 128
        )
        with torch.cuda.stream(self.copy_stream):
            value.copy_(remote[:elements].view_as(value), non_blocking=True)
        self.copy_stream.synchronize()
        return value


class TypedMappedCodec:
    def __init__(self, installer):
        self.installer = installer

    def open(self, payload, elements):
        return self.installer.open(payload, elements)

    def select_block(self, b):
        if b.mode != MAPPED:
            raise ValueError("mapped codec received raw block")
        self.installer.select(b.start, b.stop)

    def copy(self, remote, local, elements):
        return self.installer.copy(remote, local, elements)

    def synchronize(self):
        self.installer.synchronize()


class ReceiverOperations:
    def __init__(
        self,
        *,
        device,
        mapper,
        rank,
        pool,
        storages,
        output_storage,
        stream,
        raw_installer,
        mapped_installer
    ):
        self.device = device
        self.mapper = mapper
        self.rank = rank
        self.pool = pool
        self.output_storage = output_storage
        self.stream = stream
        self.installer = raw_installer
        self.ownership = LocalSlotOwnership()
        self.codecs = {
            RAW: TypedRawCodec(device, storages, 2, self.ownership),
            MAPPED: TypedMappedCodec(mapped_installer),
        }

    def device_context(self):
        return torch.cuda.device(self.device)

    def cancelled(self):
        return self.ownership.cancelled.is_set()

    def synchronize_copies(self):
        for codec in self.codecs.values():
            codec.synchronize()

    def map_install(self, b, value):
        elements = 2 * 64 * (b.stop - b.start) * 2 * 128
        out = self.output_storage[:elements].view(2, 64, b.stop - b.start, 2, 128)
        with torch.cuda.stream(self.stream):
            mapped = self.mapper.map_page(
                value,
                position_start=b.start,
                source_head_offset=self.rank * 2,
                output_device=self.device,
                output=out,
            )
            if mapped.shape != out.shape or mapped.dtype != self.pool.dtype:
                raise ValueError("mixed Target mapper contract changed")
            self.installer.select(b.start, b.stop)
            self.installer.copy(mapped.reshape(-1), None, elements)
        self.stream.synchronize()
        self.ownership.release(b.slot, compute_reads_complete=True)

    def finish_compute_and_cancel(self):
        self.stream.synchronize()
        self.ownership.cancel_after_completion()


def run_receiver(
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
    pool_codec_class,
    rank
):
    with torch.cuda.device(device):
        raw = pool_codec_class(
            device, pool, transaction["indices"], _mixed_install_mode="raw_local"
        )
        raw.install_stream = compute_stream
        mapped = pool_codec_class(
            device, pool, transaction["indices"], _mixed_install_mode="mapped_peer"
        )
        operations = ReceiverOperations(
            device=device,
            mapper=mapper,
            rank=rank,
            pool=pool,
            storages=storages,
            output_storage=output_storage,
            stream=compute_stream,
            raw_installer=raw,
            mapped_installer=mapped,
        )
        result = receiver_pipeline(listener, contract, operations, Receiver)
    if result["error"] is None and result["safe"]:
        expected_raw = sum(
            2 * 64 * (b.stop - b.start) * 2 * 128 * 2
            for b in blocks(contract)
            if b.mode == RAW
        )
        expected_mapped = sum(
            2 * 64 * (b.stop - b.start) * 2 * 128 * 2
            for b in blocks(contract)
            if b.mode == MAPPED
        )
        if (
            raw.local_install_bytes != expected_raw
            or mapped.copied_bytes != expected_mapped
            or raw.blocks_completed + mapped.blocks_completed != result["blocks"]
        ):
            result["error"] = "mixed local install counters disagree"
    transaction.update(
        worker_error=result["error"],
        worker_safe=result["safe"],
        worker_done=True,
        blocks=result["blocks"],
        logical_copy_bytes=sum(result["bytes_by_mode"].values()),
        mixed_raw_delivery_bytes=result["bytes_by_mode"][RAW],
        mixed_mapped_delivery_bytes=result["bytes_by_mode"][MAPPED],
        target_local_install_bytes=raw.local_install_bytes + mapped.copied_bytes,
        target_transfer_workspace_bytes=operations.codecs[RAW].workspace_bytes,
        target_mapper_workspace_bytes=output_storage.numel()
        * output_storage.element_size(),
        target_placement_runs=len(raw.runs),
        conversion_placement="mixed",
        gpu_observation=[],
        certified_data_completion_ns=(
            time.perf_counter_ns()
            if result["error"] is None and result["safe"]
            else None
        ),
    )
    if result["error"] is None and result["safe"]:
        # Receipt validation participates in worker_error BEFORE the unchanged
        # all-rank commit, even though the outer wrapper rechecks after commit.
        from cachebridge.backends.sglang.runtime.peer_adapter import mixed_receipt

        try:
            mixed_receipt(transaction)
        except BaseException as invalid:
            transaction.update(
                worker_error=repr(invalid), certified_data_completion_ns=None
            )
