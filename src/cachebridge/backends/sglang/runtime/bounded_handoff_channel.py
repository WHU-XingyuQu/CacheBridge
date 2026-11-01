"""R004 single-node control channel; CUDA KV never travels through the socket.

One source rank, one consumer rank, one exported storage, one in-flight block.
Blocking Unix sockets replace event-file polling. Only same-UID peers in a
private job directory may exchange the pinned PyTorch CUDA IPC descriptor.
No timing/profile hooks; the caller charges this whole protocol to handoff.
"""

import json
import os
import pickle
import socket
import stat
import struct
from pathlib import Path


class Cancelled(RuntimeError):
    """Consumer certified completion and dropped its imported storage."""


def _read(sock, size):
    result = bytearray()
    while len(result) < size:
        part = sock.recv(size - len(result))
        if not part:
            raise EOFError("handoff peer disconnected")
        result.extend(part)
    return bytes(result)


def send_bytes(sock, payload):
    if len(payload) > 1024 * 1024:
        raise ValueError("control payload too large")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_bytes(sock):
    size = struct.unpack("!I", _read(sock, 4))[0]
    if size > 1024 * 1024:
        raise ValueError("control payload too large")
    return _read(sock, size)


def send(sock, obj):
    send_bytes(sock, json.dumps(obj, separators=(",", ":")).encode())


def receive(sock):
    value = json.loads(recv_bytes(sock))
    if not isinstance(value, dict):
        raise ValueError("control message must be an object")
    return value


def peer_check(sock):
    _, uid, _ = struct.unpack(
        "3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    )
    if uid != os.getuid():
        raise PermissionError("foreign control peer")


def listener(path, timeout=60):
    path = Path(path)
    parent = path.parent.stat()
    if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o077:
        raise PermissionError("socket directory must be private and owned")
    if len(os.fsencode(path)) >= 108:
        raise ValueError("Unix socket path too long")
    # Never unlink an existing endpoint; a stale/live listener needs reconciliation.
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(path))
        sock.settimeout(timeout)
        sock.listen(1)
    except BaseException:
        sock.close()
        raise
    return sock


class CudaCodec:
    def __init__(self, device):
        import torch

        self.torch = torch
        self.device = device

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def export(self, value):
        from torch.multiprocessing.reductions import reduce_tensor, rebuild_cuda_tensor

        if not value.is_cuda or value.ndim != 1 or not value.is_contiguous():
            raise ValueError("export exactly one flat CUDA storage")
        self.synchronize()
        rebuild, args = reduce_tensor(value)
        if rebuild is not rebuild_cuda_tensor:
            raise ValueError("unexpected CUDA IPC reduction ABI")
        return pickle.dumps(args, protocol=5)

    def open(self, payload, elements):
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        # Same-job trusted descriptor, after directory/Unix peer UID checks.
        remote = rebuild_cuda_tensor(*pickle.loads(payload))
        if (
            not remote.is_cuda
            or remote.shape != (elements,)
            or remote.dtype != self.torch.bfloat16
        ):
            raise ValueError("IPC storage geometry/dtype mismatch")
        return remote

    def allocate(self, elements):
        return self.torch.empty(elements, dtype=self.torch.bfloat16, device=self.device)

    def copy(self, remote, local, elements):
        local[:elements].copy_(remote[:elements])
        self.synchronize()
        return local[:elements]


class Sender:
    def __init__(self, path, contract, codec, timeout=60):
        self.codec = codec
        self.contract = contract
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.cursor = 0
        self.completed = False
        self.safe_cancel = False
        try:
            self.sock.connect(str(path))
            peer_check(self.sock)
            send(self.sock, contract)
            if receive(self.sock) != {"op": "accept"}:
                raise ValueError("consumer rejected request identity")
        except BaseException:
            self.sock.close()
            raise

    def share(self, storage):
        send_bytes(self.sock, self.codec.export(storage))
        if receive(self.sock) != {"op": "mapped"}:
            raise ValueError("consumer failed to import storage")

    def offer(self, start, stop):
        step = (
            self.contract["block_tokens"]
            if self.contract["mode"] == "C"
            else self.contract["tokens"]
        )
        if start != self.cursor or stop != min(start + step, self.contract["tokens"]):
            raise ValueError("source block order/span mismatch")
        self.codec.synchronize()
        send(self.sock, {"op": "ready", "start": start, "stop": stop})
        reply = receive(self.sock)
        if reply.get("op") == "cancel" and reply.get("gpu_reads_complete") is True:
            self.safe_cancel = True
            raise Cancelled("consumer cancelled after GPU completion")
        if reply != {
            "op": "ack",
            "start": start,
            "stop": stop,
            "gpu_reads_complete": True,
        }:
            raise RuntimeError("consumer did not certify the expected block")
        self.cursor = stop

    def finish(self):
        if self.cursor != self.contract["tokens"]:
            raise RuntimeError("cannot finish an incomplete transfer")
        send(self.sock, {"op": "end"})
        if receive(self.sock) != {"op": "closed", "gpu_reads_complete": True}:
            raise RuntimeError("consumer import still live")
        self.completed = True

    def close(self):
        self.sock.close()


class Receiver:
    def __init__(self, server, contract, codec):
        self.codec = codec
        self.contract = contract
        self.sock, _ = server.accept()
        self.sock.settimeout(server.gettimeout())
        self.remote = self.local = None
        self.cursor = 0
        self.pending = None
        self.closed = False
        self.bytes_copied = 0
        try:
            peer_check(self.sock)
            if receive(self.sock) != contract:
                send(self.sock, {"op": "reject"})
                raise ValueError("source request identity/geometry mismatch")
            send(self.sock, {"op": "accept"})
            self.remote = codec.open(recv_bytes(self.sock), contract["elements"])
            self.local = codec.allocate(contract["elements"])
            send(self.sock, {"op": "mapped"})
        except BaseException:
            self.sock.close()
            raise

    def __iter__(self):
        return self

    def __next__(self):
        if self.pending is not None:
            raise RuntimeError("acknowledge writes before requesting another block")
        if self.closed:
            raise StopIteration
        message = receive(self.sock)
        if message == {"op": "end"}:
            if self.cursor != self.contract["tokens"]:
                raise ValueError("source ended before full coverage")
            self._drop_import()
            send(self.sock, {"op": "closed", "gpu_reads_complete": True})
            self.closed = True
            self.sock.close()
            raise StopIteration
        step = (
            self.contract["block_tokens"]
            if self.contract["mode"] == "C"
            else self.contract["tokens"]
        )
        stop = min(self.cursor + step, self.contract["tokens"])
        if message != {"op": "ready", "start": self.cursor, "stop": stop}:
            raise ValueError("source block order/span mismatch")
        start = self.cursor
        shape = (
            2,
            self.contract["layers"],
            stop - start,
            self.contract["heads"],
            self.contract["dim"],
        )
        elements = 2 * shape[1] * shape[2] * shape[3] * shape[4]
        value = self.codec.copy(self.remote, self.local, elements).view(shape)
        self.bytes_copied += value.numel() * value.element_size()
        self.pending = (start, stop)
        return start, stop, value

    def acknowledge(self):
        if self.pending is None:
            raise RuntimeError("no block to acknowledge")
        self.codec.synchronize()
        start, stop = self.pending
        send(
            self.sock,
            {"op": "ack", "start": start, "stop": stop, "gpu_reads_complete": True},
        )
        self.cursor = stop
        self.pending = None

    def _drop_import(self):
        self.codec.synchronize()
        self.remote = None

    def abort(self):
        """Return only after local GPU work/imports are safe; failed sync propagates."""
        if self.closed:
            return
        self._drop_import()
        try:
            send(self.sock, {"op": "cancel", "gpu_reads_complete": True})
        finally:
            self.closed = True
            self.sock.close()
