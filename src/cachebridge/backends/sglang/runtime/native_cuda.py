"""Existing CUDA IPC + explicit Target-device peer-copy adapter, no new kernels."""

import pickle
import torch
from cachebridge.backends.sglang.runtime.bounded_handoff_channel import CudaCodec
from cachebridge.backends.sglang.runtime.peer_runtime import PeerRuntime


class NativeExportCodec(CudaCodec):
    def export(self, value):
        # Caller has already fenced the Source snapshot stream once. Do not
        # synchronize the entire Source device for each of80 native descriptors.
        from torch.multiprocessing.reductions import reduce_tensor, rebuild_cuda_tensor

        if (
            not value.is_cuda
            or value.dtype != torch.bfloat16
            or value.ndim != 1
            or not value.is_contiguous()
        ):
            raise ValueError("flat native BF16 storage required")
        rebuild, args = reduce_tensor(value)
        if rebuild is not rebuild_cuda_tensor:
            raise ValueError("native CUDA IPC ABI")
        return pickle.dumps(args, protocol=5)


class TargetPeerInputCodec(CudaCodec):
    def __init__(self, device, storages, heads, ownership):
        super().__init__(device)
        self.storages = storages
        self.heads = heads
        self.ownership = ownership
        self.copy_stream = torch.cuda.Stream(device=device)
        self.runtime = PeerRuntime()
        self.start = self.stop = None
        self.peer_calls = 0
        self.logical_bytes = 0
        self.block_runs = []

    @property
    def workspace_bytes(self):
        return sum(x.numel() * x.element_size() for x in self.storages)

    def select(self, start, stop):
        self.start, self.stop = start, stop

    def synchronize(self):
        self.copy_stream.synchronize()

    def _destination(self, elements):
        slot = (self.start // 2048) % 2
        self.ownership.acquire(slot)
        return self.storages[slot][:elements].view(
            2, 40, self.stop - self.start, self.heads, 128
        )

    def _copy(self, remote, value, source_offset, destination_offset, size):
        if (
            not remote.is_cuda
            or remote.dtype != torch.bfloat16
            or not remote.is_contiguous()
            or remote.device.index == self.device.index
        ):
            raise ValueError("native cross-device BF16 input")
        if (
            source_offset < 0
            or destination_offset < 0
            or source_offset + size > remote.numel() * 2
            or destination_offset + size > value.numel() * 2
        ):
            raise ValueError("native copy byte bounds")
        self.runtime.copy(
            value.data_ptr() + destination_offset,
            self.device.index,
            remote.data_ptr() + source_offset,
            remote.device.index,
            size,
            self.copy_stream.cuda_stream,
        )
        self.peer_calls += 1

    def copy(self, remote, local, elements):
        del local
        expected = 2 * 40 * (self.stop - self.start) * self.heads * 128
        if elements != expected:
            raise ValueError("packed raw input geometry")
        value = self._destination(elements)
        with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
            self._copy(remote, value, 0, 0, elements * 2)
        self.synchronize()
        self.logical_bytes += elements * 2
        return value

    def copy_native(self, remotes, plan):
        if (plan.layers, plan.heads, plan.dimension, plan.element_bytes) != (
            40,
            self.heads,
            128,
            2,
        ) or (plan.start, plan.stop) != (self.start, self.stop):
            raise ValueError("native receiver plan geometry")
        if len(remotes) != 80:
            raise ValueError("native layer descriptor count")
        elements = plan.useful_bytes // 2
        value = self._destination(elements)
        with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
            for span in plan.spans():
                remote = remotes[span["kind"] * 40 + span["layer"]]
                if remote.numel() != plan.pool_rows * self.heads * 128:
                    raise ValueError("native imported pool extent")
                self._copy(
                    remote,
                    value,
                    span["source_byte_offset"],
                    span["destination_byte_offset"],
                    span["bytes"],
                )
        self.synchronize()
        self.logical_bytes += plan.useful_bytes
        self.block_runs.append(len(plan.runs))
        return value

    def receipt(self):
        return dict(
            input_copy_api="cudaMemcpyPeerAsync",
            input_copy_stream_device=self.device.index,
            peer_calls=self.peer_calls,
            logical_input_bytes=self.logical_bytes,
            native_runs_per_block=list(self.block_runs),
            read_ACK_requires="all_input_reads_GPU_complete",
            local_slot_release_requires="map_and_install_GPU_complete",
        )
