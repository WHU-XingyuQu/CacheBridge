"""Typed dual-representation channel, using the existing framed Unix transport.

CUDA storage remains in IPC descriptors, never in socket data. Completion
codecs are injected; CPU tests exercise actual sockets but replace CUDA IPC.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import socket
from cachebridge.backends.sglang.runtime.bounded_handoff_channel import (
    peer_check,
    send,
    receive,
    send_bytes,
    recv_bytes,
)

RAW = "raw40"
MAPPED = "mapped64"


@dataclass(frozen=True)
class Block:
    request: str
    rank: int
    mapper: str
    mode: str
    bank: str
    slot: int
    generation: int
    start: int
    stop: int

    @property
    def scope(self):
        return (
            "raw_input_copy_complete"
            if self.mode == RAW
            else "mapped_copy_install_complete"
        )

    def frame(self, op):
        return dict(op=op, block=asdict(self))

    def ack(self):
        return dict(self.frame("ack"), scope=self.scope, gpu_reads_complete=True)


def blocks(contract):
    result = []
    counts = {RAW: 0, MAPPED: 0}
    step = contract["block_tokens"]
    n = contract["tokens"]
    for index, start in enumerate(range(0, n, step)):
        mode = MAPPED if index == contract["source_block_index"] else RAW
        ordinal = counts[mode]
        counts[mode] += 1
        result.append(
            Block(
                contract["request"],
                contract["target_rank"],
                contract["mapper"],
                mode,
                mode,
                ordinal % 2,
                ordinal // 2 + 1,
                start,
                min(start + step, n),
            )
        )
    return tuple(result)


def contract_for(*, request, rank, mapper, history_sha, tokens, step, source_index):
    if tokens <= 0 or step <= 0 or not 0 <= source_index < (tokens + step - 1) // step:
        raise ValueError("mixed block plan geometry")
    return dict(
        version=13,
        request=request,
        source_rank=rank,
        target_rank=rank,
        mapper=mapper,
        history_sha256=history_sha,
        method="CacheBridgeMixedBlocks",
        conversion_placement="mixed",
        tokens=tokens,
        block_tokens=step,
        source_block_index=source_index,
        heads=2,
        dim=128,
        element_bytes=2,
        banks={
            RAW: dict(
                slots=2, layers=40, elements=2 * 40 * min(tokens, step) * 2 * 128
            ),
            MAPPED: dict(
                slots=2, layers=64, elements=2 * 64 * min(tokens, step) * 2 * 128
            ),
        },
    )


def mixed_contract(p, rank):
    # Only the single selected production question is admitted, no ratio search.
    if (
        p.get("mapper_kind") != "CacheBridge"
        or p.get("conversion_placement") != "mixed"
        or len(p["history_ids"]) != 16384
        or p.get("mixed_source_block") not in ("early", "tail")
        or p.get("source_representation", "packed") != "packed"
        or p.get("raw_input_copy", "tensor") != "tensor"
    ):
        raise ValueError(
            "mixed candidate requires frozen 16K packed/tensor early-or-tail plan"
        )
    from cachebridge.backends.sglang.runtime import config as _config

    if _config.TARGET_TP != 4:
        raise ValueError("mixed candidate requires SourceTP4/TargetTP4")
    a = _config.artifact_contract()["cachebridge"]
    mapper = hashlib.sha256(
        json.dumps(a["file_hashes"], sort_keys=True).encode()
    ).hexdigest()
    history = hashlib.sha256(
        json.dumps(p["history_ids"], separators=(",", ":")).encode()
    ).hexdigest()
    return contract_for(
        request=p["request_id"],
        rank=rank,
        mapper=mapper,
        history_sha=history,
        tokens=16384,
        step=2048,
        source_index=0 if p["mixed_source_block"] == "early" else 7,
    )


class Sender:
    def __init__(self, path, contract, codecs, timeout=60):
        self.contract = contract
        self.codecs = codecs
        self.plan = blocks(contract)
        self.cursor = 0
        self.completed = False
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        try:
            self.sock.connect(str(path))
            peer_check(self.sock)
            send(self.sock, contract)
            if receive(self.sock) != {"op": "accept"}:
                raise ValueError("mixed contract rejected")
        except BaseException:
            self.sock.close()
            raise

    def share(self, storages):
        for bank in (RAW, MAPPED):
            if len(storages[bank]) != 2 or len(self.codecs[bank]) != 2:
                raise ValueError("mixed bank slots")
            elements = self.contract["banks"][bank]["elements"]
            for slot in range(2):
                flat = storages[bank][slot].reshape(-1)
                if flat.numel() < elements:
                    raise ValueError("mixed storage extent")
                send_bytes(self.sock, self.codecs[bank][slot].export(flat[:elements]))
                if receive(self.sock) != {"op": "mapped", "bank": bank, "slot": slot}:
                    raise ValueError("mixed bank import receipt")

    def offer(self, b):
        if self.cursor >= len(self.plan) or b != self.plan[self.cursor]:
            raise ValueError("mixed ready cursor/identity")
        self.codecs[b.bank][b.slot].synchronize()
        send(self.sock, b.frame("ready"))
        reply = receive(self.sock)
        if reply != b.ack():
            raise RuntimeError("mixed stale/unsafe/wrong-scope ACK")
        self.cursor += 1

    def finish(self):
        if self.cursor != len(self.plan):
            raise RuntimeError("mixed incomplete Source transfer")
        send(self.sock, {"op": "end", "request": self.contract["request"]})
        if receive(self.sock) != {
            "op": "closed",
            "request": self.contract["request"],
            "gpu_reads_complete": True,
        }:
            raise RuntimeError("mixed imported reads remain uncertified")
        self.completed = True

    def close(self):
        self.sock.close()


class Receiver:
    def __init__(self, listener, contract, codecs):
        self.contract = contract
        self.codecs = codecs
        self.plan = blocks(contract)
        self.sock, _ = listener.accept()
        self.sock.settimeout(listener.gettimeout())
        self.cursor = 0
        self.pending = None
        self.remotes = {}
        self.closed = False
        self.bytes_by_mode = {RAW: 0, MAPPED: 0}
        try:
            peer_check(self.sock)
            if receive(self.sock) != contract:
                send(self.sock, {"op": "reject"})
                raise ValueError("mixed receiver contract identity")
            send(self.sock, {"op": "accept"})
            for bank in (RAW, MAPPED):
                elements = contract["banks"][bank]["elements"]
                for slot in range(2):
                    self.remotes[bank, slot] = codecs[bank].open(
                        recv_bytes(self.sock), elements
                    )
                    send(self.sock, {"op": "mapped", "bank": bank, "slot": slot})
        except BaseException:
            self.sock.close()
            raise

    def next(self, b):
        if (
            self.pending is not None
            or self.closed
            or self.cursor >= len(self.plan)
            or b != self.plan[self.cursor]
        ):
            raise RuntimeError("mixed receive state/cursor")
        if receive(self.sock) != b.frame("ready"):
            raise ValueError("mixed ready identity mismatch")
        codec = self.codecs[b.mode]
        codec.select_block(b)
        layers = self.contract["banks"][b.bank]["layers"]
        elements = (
            2
            * layers
            * (b.stop - b.start)
            * self.contract["heads"]
            * self.contract["dim"]
        )
        value = codec.copy(self.remotes[b.bank, b.slot], None, elements)
        self.bytes_by_mode[b.mode] += elements * self.contract["element_bytes"]
        self.pending = b
        return value if b.mode == RAW else None

    def acknowledge(self):
        b = self.pending
        if b is None:
            raise RuntimeError("mixed no pending ACK")
        self.codecs[b.mode].synchronize()
        send(self.sock, b.ack())
        self.pending = None
        self.cursor += 1

    def synchronize(self):
        for c in self.codecs.values():
            c.synchronize()

    def finish(self):
        if self.pending is not None or self.cursor != len(self.plan):
            raise RuntimeError("mixed incomplete receiver")
        if receive(self.sock) != {"op": "end", "request": self.contract["request"]}:
            raise ValueError("mixed terminal identity")
        self.synchronize()
        self.remotes.clear()
        send(
            self.sock,
            {
                "op": "closed",
                "request": self.contract["request"],
                "gpu_reads_complete": True,
            },
        )
        self.closed = True
        self.sock.close()

    def abort(self):
        if self.closed:
            return
        self.synchronize()
        self.remotes.clear()
        try:
            send(self.sock, {"op": "cancel", "gpu_reads_complete": True})
        finally:
            self.closed = True
            self.sock.close()
