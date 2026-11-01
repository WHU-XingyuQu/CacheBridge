"""Isolated copy mode; existing Source-ready/Target-exclusive lifetime retained."""

import json
import os
from pathlib import Path
import threading
from types import MethodType
from cachebridge.backends.sglang.runtime.peer_geometry import operations
from cachebridge.backends.sglang.runtime.peer_runtime import PeerRuntime
from cachebridge.backends.sglang.runtime import config as _config

LOCAL_KV_HEADS = _config.PAIR.kv_heads // _config.PAIR.tensor_parallel_size
from cachebridge.backends.sglang.runtime.overlap_bridge import (
    model_runner,
    require_drained,
    work_state,
    attach_work,
    attach_status,
)


class PeerControl:
    def __init__(self, s):
        self.s = s
        self.owner = threading.get_ident()
        self.mode = "tensor"
        self.switches = 0
        self.runtime = PeerRuntime()

    def set_mode(self, mode):
        s = self.s
        if threading.get_ident() != self.owner or mode not in ["tensor", "peer"]:
            raise RuntimeError("copy mode requires scheduler thread and frozen mode")
        if (
            s.waiting_queue
            or not s.running_batch.is_empty()
            or s.chunked_req is not None
        ):
            raise RuntimeError("copy mode requires Target idle")
        if any(
            s.source_mapper_state[k] is not None
            for k in ["transaction", "source_transaction"]
        ):
            raise RuntimeError("copy mode with pending handoff")
        if (
            s.concurrency_state["quarantined"]
            or s.concurrency_state["active_request"] is not None
            or s.tree_cache.protected_size()
        ):
            raise RuntimeError("copy mode with protected ownership")
        require_drained(s)
        self.mode = mode
        self.switches += 1

    def _admit_receiver_idle(self):
        s = self.s
        if threading.get_ident() != self.owner or self.mode != "tensor":
            raise RuntimeError(
                "receiver placement requires scheduler-owned tensor mode"
            )
        if (
            s.waiting_queue
            or not s.running_batch.is_empty()
            or s.chunked_req is not None
        ):
            raise RuntimeError("this receiver control requires Target idle")
        if any(
            s.source_mapper_state[k] is not None
            for k in ["transaction", "source_transaction"]
        ):
            raise RuntimeError("receiver placement with pending transaction")
        if (
            s.concurrency_state["quarantined"]
            or s.concurrency_state["active_request"] is not None
            or s.tree_cache.protected_size()
        ):
            raise RuntimeError("receiver placement with unresolved ownership")
        require_drained(s)

    def _admit_receiver_decoder(self):
        s = self.s
        if threading.get_ident() != self.owner or self.mode != "tensor":
            raise RuntimeError(
                "receiver placement requires scheduler-owned tensor mode"
            )
        if not s.source_mapper_state.get("live_stage_enabled"):
            raise RuntimeError("receiver overlap requires live scheduler ownership")
        if s.waiting_queue or s.chunked_req is not None:
            raise RuntimeError("bounded receiver overlap excludes pending prefill")
        if any(
            s.source_mapper_state[k] is not None
            for k in ["transaction", "source_transaction"]
        ):
            raise RuntimeError("receiver placement with pending transaction")
        if (
            s.concurrency_state["quarantined"]
            or s.concurrency_state["active_request"] is not None
        ):
            raise RuntimeError("receiver placement with unresolved handoff ownership")
        reqs = s.running_batch.reqs
        if len(reqs) != 1 or not s.running_batch.forward_mode.is_decode():
            raise RuntimeError("bounded receiver overlap requires one Native decoder")
        req = reqs[0]
        if (
            req.finished()
            or not req.output_ids
            or req.req_pool_idx is None
            or req.req_pool_idx < 0
            or len(req.prefix_indices) < len(req.origin_input_ids)
            or req.last_node.lock_ref <= 0
        ):
            raise RuntimeError(
                "Native background must retain its complete protected prefix"
            )
        if s.tree_cache.protected_size() != len(req.prefix_indices):
            raise RuntimeError(
                "bounded receiver overlap excludes unaccounted protected pages"
            )
        work = work_state(s)
        pending = work["submitted"] - work["resolved"]
        if (
            work["failed"]
            or pending < 0
            or pending > 1
            or work["queued_results"] != pending
        ):
            raise RuntimeError("unsupported asynchronous work ownership")
        batches = [
            s.running_batch,
            getattr(s, "cur_batch", None),
            getattr(s, "last_batch", None),
        ]
        batches += [row[0] for row in getattr(s, "result_queue", ())]
        for batch in batches:
            if batch is not None and not batch.is_empty():
                if not batch.forward_mode.is_decode() or any(
                    r is not req for r in batch.reqs
                ):
                    raise RuntimeError("pending batch owns another request or prefill")

    def admit_receiver(self):
        # Offline candidate: default retains the exact original idle contract.
        if os.environ.get("CACHEBRIDGE_RECEIVER_DECODER_OVERLAP") != "1":
            return self._admit_receiver_idle()
        # Every rank votes BEFORE the original start allocates unpublished pages
        # or starts an IPC worker. A local reject must not strand its peers.
        error = None
        try:
            if self.s.running_batch.is_empty():
                self._admit_receiver_idle()
            else:
                self._admit_receiver_decoder()
        except BaseException as exc:
            error = repr(exc)
        import torch

        flag = torch.tensor([int(error is None)], dtype=torch.int32)
        torch.distributed.all_reduce(
            flag, op=torch.distributed.ReduceOp.MIN, group=self.s.tp_cpu_group
        )
        if not bool(flag[0]):
            raise RuntimeError(
                "all-rank receiver eligibility rejected: " + (error or "peer rejection")
            )

    def receipt(self):
        return dict(
            mode=self.mode, switches=self.switches, CUDA_runtime=self.runtime.version
        )


def codec_type(original, control, torch):
    class CertifiedCodec(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            tx = control.s.source_mapper_state["transaction"]
            rid = tx["request_id"]
            if tx["parameters"].get("conversion_placement", "source") not in (
                "source",
                "mixed",
            ):
                raise RuntimeError("control excludes receiver mapping")
            if control.s.concurrency_state["requests"][rid]["published"]:
                raise RuntimeError("peer destination must remain unpublished")
            self.tx = tx
            self.copy_mode = (
                "peer"
                if tx["parameters"].get("conversion_placement") == "mixed"
                else control.mode
            )
            self.peer_stream = None
            self.peer_device = None
            self.ready = None
            self.blocks_completed = 0
            self.copied_bytes = 0
            self.peer_calls = 0
            tx[
                (
                    "mixed_peer_observer"
                    if tx["parameters"].get("conversion_placement") == "mixed"
                    else "copy_observer"
                )
            ] = self

        def select(self, start, stop):
            if self.ready is not None:
                raise RuntimeError("unconsumed validated ready block")
            super().select(start, stop)
            self.ready = (start, stop)

        def copy(self, remote, local, elements):
            if self.ready != (self.start, self.stop) or self.ready is None:
                raise RuntimeError("requires validated ready message")
            n = self.stop - self.start
            expected = 2 * 64 * n * LOCAL_KV_HEADS * 128
            if elements != expected:
                raise ValueError("peer source geometry mismatch")
            if control.s.concurrency_state["requests"][self.tx["request_id"]][
                "published"
            ]:
                raise RuntimeError("peer-copy destination already published")
            if self.copy_mode == "tensor":
                result = super().copy(remote, local, elements)
            else:
                if (
                    remote.dtype != torch.bfloat16
                    or not remote.is_cuda
                    or not remote.is_contiguous()
                    or remote.numel() < elements
                ):
                    raise ValueError("peer source must be contiguous BF16 IPC")
                if remote.device.index == self.device.index:
                    raise ValueError("requires cross-device peer path")
                if self.peer_stream is None:
                    self.peer_device = remote.device
                    self.peer_stream = torch.cuda.Stream(device=self.peer_device)
                if self.peer_device != remote.device:
                    raise RuntimeError("IPC source device changed")
                with torch.cuda.device(self.peer_device), torch.cuda.stream(
                    self.peer_stream
                ):
                    window = self.observation.start(
                        f"install-{self.start}-{self.stop}", self.peer_stream
                    )
                    for kind, layer, src_offset, dst_offset, size in operations(
                        self.start, self.stop, self.runs, heads=LOCAL_KV_HEADS
                    ):
                        getter = (
                            self.pool._get_key_buffer
                            if kind == 0
                            else self.pool._get_value_buffer
                        )
                        target = getter(self.pool.start_layer + layer)
                        if (
                            target.dtype != torch.bfloat16
                            or target.device != self.device
                            or not target.is_contiguous()
                            or target.shape[1:] != (LOCAL_KV_HEADS, 128)
                        ):
                            raise ValueError("peer destination pool format changed")
                        if (
                            dst_offset + size > target.numel() * 2
                            or src_offset + size > elements * 2
                        ):
                            raise ValueError("peer pointer span exceeds allocation")
                        control.runtime.copy(
                            target.data_ptr() + dst_offset,
                            self.device.index,
                            remote.data_ptr() + src_offset,
                            self.peer_device.index,
                            size,
                            self.peer_stream.cuda_stream,
                        )
                        self.peer_calls += 1
                    self.observation.stop(window, self.peer_stream)
                    self.peer_stream.synchronize()
                self.start = self.stop = None
                result = remote[:elements]
            self.copied_bytes += elements * 2
            self.blocks_completed += 1
            self.ready = None
            return result

        def synchronize(self):
            if self.copy_mode == "tensor":
                return super().synchronize()
            if self.peer_stream is not None:
                with torch.cuda.device(self.peer_device):
                    self.peer_stream.synchronize()

        def receipt(self):
            return dict(
                mode=self.copy_mode,
                blocks_completed=self.blocks_completed,
                copied_bytes=self.copied_bytes,
                peer_API_calls=self.peer_calls,
                validated_ready_before_copy=True,
                unpublished_destination_before_copy=True,
                block_GPU_completion_before_ACK=True,
                CUDA_runtime=control.runtime.version,
                copy_stream_device=(
                    None if self.peer_device is None else self.peer_device.index
                ),
                no_transfer_workspace=True,
                per_tensor_stream_wait_inserted=self.copy_mode == "tensor",
            )

    class StagedCertifiedCodec(CertifiedCodec):
        """Use the identical peer-copy implementation with a scratch pool.

        Only the destination changes. The extra Target-local installation is
        completed after all remote blocks arrive, before publication.
        """

        def __init__(self, device, pool, indices):
            from types import SimpleNamespace

            self.real_pool = pool
            self.real_indices = indices
            self.staging = torch.empty(
                (2, 64, len(indices), LOCAL_KV_HEADS, 128),
                dtype=pool._get_key_buffer(pool.start_layer).dtype,
                device=device,
            )
            self.staging_bytes = self.staging.numel() * self.staging.element_size()
            scratch = SimpleNamespace(
                start_layer=pool.start_layer,
                _get_key_buffer=lambda layer: self.staging[0, layer - pool.start_layer],
                _get_value_buffer=lambda layer: self.staging[
                    1, layer - pool.start_layer
                ],
            )
            # This logical index array is CPU-only metadata, not model input.
            super().__init__(device, scratch, torch.arange(len(indices)))
            self.local_copy_bytes = 0

        @property
        def workspace_bytes(self):
            return self.staging_bytes

        def finish_install(self):
            # 'original' is the unchanged tensor-copy codec, before peer wrapping.
            installer = original(self.device, self.real_pool, self.real_indices)
            installer.select(0, len(self.real_indices))
            installer.copy(self.staging.reshape(-1), None, self.staging.numel())
            installer.synchronize()
            self.local_copy_bytes = self.workspace_bytes
            # tx -> copy_observer -> tx forms a Python reference cycle in the
            # inherited receipt mechanism. Drop GPU scratch explicitly after
            # the install fence, rather than depending on cyclic GC timing.
            self.staging = None

        def receipt(self):
            result = super().receipt()
            result.update(
                no_transfer_workspace=False,
                target_staging_bytes=self.workspace_bytes,
                additional_local_copy_bytes=self.local_copy_bytes,
                staging_released_after_install=self.staging is None,
                full_local_completion_before_publication=self.local_copy_bytes
                == self.workspace_bytes,
            )
            return result

    class ReceiverInstallCodec(original):
        """Target-local writes; raw remote copy has a separate codec/lifetime."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            tx = control.s.source_mapper_state["transaction"]
            rid = tx["request_id"]
            if (
                tx["parameters"].get("conversion_placement") not in ("target", "mixed")
                or control.mode != "tensor"
            ):
                raise RuntimeError(
                    "local receiver installation requires target/tensor scope"
                )
            if control.s.concurrency_state["requests"][rid]["published"]:
                raise RuntimeError(
                    "receiver local installation requires unpublished pages"
                )
            self.tx = tx
            self.blocks_completed = 0
            self.local_install_bytes = 0
            tx["copy_observer"] = self

        def copy(self, remote, local, elements):
            if control.s.concurrency_state["requests"][self.tx["request_id"]][
                "published"
            ]:
                raise RuntimeError("receiver local destination already published")
            result = super().copy(remote, local, elements)
            # The unchanged parent synchronizes installation before returning.
            self.local_install_bytes += elements * 2
            self.blocks_completed += 1
            return result

        def receipt(self):
            return dict(
                mode="tensor",
                conversion_placement="target",
                transported_representation="Source40",
                installed_representation="Target64",
                raw_remote_delivery_bytes=self.tx.get(
                    "mixed_raw_delivery_bytes", self.tx["logical_copy_bytes"]
                ),
                local_mapped_install_bytes=self.local_install_bytes,
                blocks_completed=self.blocks_completed,
                peer_API_calls=0,
                remote_read_ACK_certifies="input_copy_complete",
                local_slot_release_requires="map_and_install_GPU_complete",
                full_local_completion_before_publication=True,
                no_transfer_workspace=False,
            )

    def scoped_codec(*args, **kwargs):
        tx = control.s.source_mapper_state["transaction"]
        placement = tx["parameters"].get("conversion_placement", "source")
        if placement == "mixed":
            scope = kwargs.pop("_mixed_install_mode", None)
            if control.mode != "tensor":
                raise RuntimeError(
                    "mixed candidate requires original tensor admission plus explicit mapped peer copy"
                )
            if scope == "raw_local":
                return ReceiverInstallCodec(*args, **kwargs)
            if scope == "mapped_peer":
                return CertifiedCodec(*args, **kwargs)
            raise RuntimeError(
                "mixed installer requires an explicit representation role"
            )
        if placement == "target":
            return ReceiverInstallCodec(*args, **kwargs)
        if placement != "source":
            raise RuntimeError("unknown copy/conversion placement")
        if tx["parameters"].get("installation", "direct") == "staged_copy":
            return StagedCertifiedCodec(*args, **kwargs)
        return CertifiedCodec(*args, **kwargs)

    return scoped_codec


def mixed_receipt(tx):
    raw = tx["copy_observer"].receipt()
    mapped = tx["mixed_peer_observer"].receipt()
    if (
        raw["raw_remote_delivery_bytes"] != tx["mixed_raw_delivery_bytes"]
        or mapped["copied_bytes"] != tx["mixed_mapped_delivery_bytes"]
        or raw["raw_remote_delivery_bytes"] + mapped["copied_bytes"]
        != tx["logical_copy_bytes"]
        or raw["local_mapped_install_bytes"] + mapped["copied_bytes"]
        != tx["target_local_install_bytes"]
        or raw["blocks_completed"] + mapped["blocks_completed"] != tx["blocks"]
        or mapped["mode"] != "peer"
        or not mapped["block_GPU_completion_before_ACK"]
        or not raw["full_local_completion_before_publication"]
    ):
        raise RuntimeError(
            "mixed remote/local copy receipts disagree before publication"
        )
    return dict(
        conversion_placement="mixed",
        mode="raw-tensor/mapped-peer",
        raw=raw,
        mapped=mapped,
        logical_copy_bytes=tx["logical_copy_bytes"],
        target_local_install_bytes=tx["target_local_install_bytes"],
        blocks_completed=tx["blocks"],
        full_local_completion_before_publication=True,
    )


def attach(s):
    if os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROLE"] != "target":
        return
    import torch
    from cachebridge.backends.sglang.runtime import source_mapper_adapter as mapper
    from cachebridge.backends.sglang.runtime.resident_session import write

    root = Path(os.environ["CACHEBRIDGE_SERVING_HANDOFF_ROOT"])
    control = PeerControl(s)
    mapper._DirectMappedPoolCodec = codec_type(
        mapper._DirectMappedPoolCodec, control, torch
    )
    original = s.audit_handoff

    def rpc(self, p):
        if p["action"] == "peer_copy_mode":
            control.set_mode(p["mode"])
            write(
                root / f"target-copy-mode-rank{self.tp_rank}.json",
                dict(phase=p["phase"], rank=self.tp_rank, **control.receipt()),
            )
            return
        if p["action"] == "online_stage_start" and p.get("conversion_placement") in (
            "target",
            "mixed",
        ):
            control.admit_receiver()
        tx = (
            self.source_mapper_state["transaction"]
            if p["action"] == "online_stage_commit"
            else None
        )
        result = original(p)
        if tx is not None:
            receipt = tx["copy_observer"].receipt()
            if tx["parameters"].get("conversion_placement") == "mixed":
                receipt = mixed_receipt(tx)
            elif tx["parameters"].get("conversion_placement", "source") == "source":
                if (
                    receipt["copied_bytes"] != tx["logical_copy_bytes"]
                    or receipt["blocks_completed"] != tx["blocks"]
                ):
                    raise RuntimeError("copy receipt disagrees with data certificate")
            elif (
                receipt["raw_remote_delivery_bytes"] != tx["logical_copy_bytes"]
                or receipt["local_mapped_install_bytes"]
                != tx["target_local_install_bytes"]
                or receipt["blocks_completed"] != tx["blocks"]
            ):
                raise RuntimeError(
                    "receiver remote/local receipt disagrees with data certificate"
                )
            self.source_mapper_state["records"][p["request_id"]][
                "copy_control"
            ] = receipt
            self.concurrency_state["requests"][p["request_id"]][
                "copy_control"
            ] = receipt
        if p["action"] == "concurrency_status":
            path = root / f"target-concurrency-status-rank{self.tp_rank}.json"
            value = json.loads(path.read_text())
            value["copy_mode_control"] = control.receipt()
            write(path, value)
        return result

    s.audit_handoff = MethodType(rpc, s)
