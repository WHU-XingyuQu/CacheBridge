"""Isolated native-pool descriptor channel; KV data uses injected IPC/DMA codec."""

from pathlib import Path
import os, socket, struct
from cachebridge.backends.sglang.runtime.bounded_handoff_channel import (
    send,
    receive,
    send_bytes,
    recv_bytes,
    peer_check,
)
from cachebridge.backends.sglang.runtime.native_plan import build_plan


def process_identity(pid=None):
    pid = os.getpid() if pid is None else pid
    f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, proc_start=f[19], pgid=int(f[2]))


def _peer_generation(sock, expected):
    peer_check(sock)
    pid, _, _ = struct.unpack(
        "3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    )
    if process_identity(pid) != expected:
        raise RuntimeError("native endpoint process generation mismatch")


def validate_contract(c):
    if (
        c.get("representation") != "source_native_pages"
        or c.get("version") != 13
        or c.get("conversion_placement") != "target"
    ):
        raise ValueError("native contract scope")
    if (
        type(c.get("slots")) is not int
        or c["slots"] != 2
        or c.get("mode") != "C"
        or any(
            not isinstance(c.get(k), str) or not c[k]
            for k in ["request", "history_sha256"]
        )
    ):
        raise ValueError("native contract identity")
    for k in ["tokens", "block_tokens", "layers", "heads", "dim"]:
        if type(c[k]) is not int or c[k] <= 0:
            raise ValueError("native integer geometry")
    for k in ["source_generation", "target_generation"]:
        if (
            set(c[k]) != {"pid", "proc_start", "pgid"}
            or type(c[k]["pid"]) is not int
            or type(c[k]["pgid"]) is not int
            or min(c[k]["pid"], c[k]["pgid"]) <= 0
            or not isinstance(c[k]["proc_start"], str)
            or not c[k]["proc_start"].isdigit()
        ):
            raise ValueError("native generation fields")


class NativeSender:
    def __init__(self, path, contract, codec, timeout=60):
        validate_contract(contract)
        if process_identity() != contract["source_generation"]:
            raise RuntimeError("native Source generation")
        self.contract = contract
        self.codec = codec
        self.cursor = 0
        self.completed = False
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        try:
            self.sock.connect(str(path))
            _peer_generation(self.sock, contract["target_generation"])
            send(self.sock, contract)
            if receive(self.sock) != {"op": "accept"}:
                raise ValueError("native request rejected")
        except BaseException:
            self.sock.close()
            raise

    def share(self, buffers, indices, pool_rows):
        c = self.contract
        if len(buffers) != 2 * c["layers"] or len(indices) != c["tokens"]:
            raise ValueError("native descriptor coverage")
        build_plan(
            indices,
            start=0,
            stop=len(indices),
            pool_rows=pool_rows,
            layers=c["layers"],
            heads=c["heads"],
            dimension=c["dim"],
        )
        send(
            self.sock,
            dict(
                op="native_pool",
                pool_rows=pool_rows,
                buffers=len(buffers),
                indices=list(indices),
            ),
        )
        for buffer in buffers:
            send_bytes(self.sock, self.codec.export(buffer))
        if receive(self.sock) != {"op": "native_mapped", "buffers": len(buffers)}:
            raise RuntimeError("native imports incomplete")

    def offer(self, slot, start, stop):
        c = self.contract
        if (
            slot != (start // c["block_tokens"]) % 2
            or start != self.cursor
            or stop != min(start + c["block_tokens"], c["tokens"])
        ):
            raise ValueError("native source order")
        send(self.sock, dict(op="native_ready", slot=slot, start=start, stop=stop))
        reply = receive(self.sock)
        if reply.get("op") == "cancel" and reply.get("gpu_reads_complete") is True:
            raise RuntimeError("native read cancelled after completion")
        if reply != dict(
            op="ack", slot=slot, start=start, stop=stop, gpu_reads_complete=True
        ):
            raise RuntimeError("native missing read ACK")
        self.cursor = stop

    def finish(self):
        if self.cursor != self.contract["tokens"]:
            raise RuntimeError("native incomplete source")
        send(self.sock, dict(op="end"))
        if receive(self.sock) != dict(op="closed", gpu_reads_complete=True):
            raise RuntimeError("native imports remain live")
        self.completed = True

    def close(self):
        self.sock.close()


class NativeReceiver:
    def __init__(self, server, contract, codec):
        validate_contract(contract)
        if process_identity() != contract["target_generation"]:
            raise RuntimeError("native Target generation")
        self.contract = contract
        self.codec = codec
        self.cursor = 0
        self.pending = None
        self.closed = False
        self.remotes = []
        self.bytes_copied = 0
        self.sock, _ = server.accept()
        self.sock.settimeout(server.gettimeout())
        try:
            _peer_generation(self.sock, contract["source_generation"])
            if receive(self.sock) != contract:
                send(self.sock, {"op": "reject"})
                raise ValueError("native request/history mismatch")
            send(self.sock, {"op": "accept"})
            meta = receive(self.sock)
            if (
                set(meta) != {"op", "pool_rows", "buffers", "indices"}
                or meta["op"] != "native_pool"
                or meta["buffers"] != 2 * contract["layers"]
                or len(meta["indices"]) != contract["tokens"]
            ):
                raise ValueError("native pool descriptor mismatch")
            self.pool_rows = meta["pool_rows"]
            self.indices = tuple(meta["indices"])
            self.plans = {}
            for a in range(0, contract["tokens"], contract["block_tokens"]):
                self.plans[a] = build_plan(
                    self.indices,
                    start=a,
                    stop=min(a + contract["block_tokens"], contract["tokens"]),
                    pool_rows=self.pool_rows,
                    layers=contract["layers"],
                    heads=contract["heads"],
                    dimension=contract["dim"],
                )
            elements = self.pool_rows * contract["heads"] * contract["dim"]
            for _ in range(2 * contract["layers"]):
                self.remotes.append(codec.open(recv_bytes(self.sock), elements))
            send(self.sock, dict(op="native_mapped", buffers=len(self.remotes)))
        except BaseException:
            self.codec.synchronize()
            self.remotes = []
            self.sock.close()
            raise

    def next(self, expected_slot):
        if self.pending is not None or self.closed:
            raise RuntimeError("native receive state")
        c = self.contract
        stop = min(self.cursor + c["block_tokens"], c["tokens"])
        message = receive(self.sock)
        if expected_slot != (self.cursor // c["block_tokens"]) % 2 or message != dict(
            op="native_ready", slot=expected_slot, start=self.cursor, stop=stop
        ):
            raise ValueError("native target order")
        p = self.plans[self.cursor]
        self.codec.select(self.cursor, stop)
        value = self.codec.copy_native(self.remotes, p)
        self.bytes_copied += p.useful_bytes
        self.pending = (expected_slot, self.cursor, stop)
        return self.cursor, stop, value

    def acknowledge(self):
        if self.pending is None:
            raise RuntimeError("native ACK without read")
        self.codec.synchronize()
        slot, a, b = self.pending
        send(
            self.sock,
            dict(op="ack", slot=slot, start=a, stop=b, gpu_reads_complete=True),
        )
        self.cursor = b
        self.pending = None

    def finish(self):
        if self.pending is not None or self.cursor != self.contract["tokens"]:
            raise RuntimeError("native incomplete receiver")
        if receive(self.sock) != dict(op="end"):
            raise ValueError("native terminal frame")
        self.codec.synchronize()
        self.remotes = []
        self.indices = ()
        self.plans = {}
        send(self.sock, dict(op="closed", gpu_reads_complete=True))
        self.closed = True
        self.sock.close()

    def abort(self):
        if self.closed:
            return
        self.codec.synchronize()
        self.remotes = []
        self.indices = ()
        self.plans = {}
        try:
            send(self.sock, dict(op="cancel", gpu_reads_complete=True))
        finally:
            self.closed = True
            self.sock.close()
