"""Single-node trusted-job CUDA IPC transport; no CPU KV payload.

Private PyTorch reduction ABI is intentionally pinned to the existing environment.
Each descriptor has exactly one consumer. Exporter must retain the tensor until
all consumers have completed their local copies and acknowledged the request.
"""

import hashlib
import os
import pickle
from pathlib import Path

import torch
from torch.multiprocessing.reductions import reduce_tensor, rebuild_cuda_tensor


def publish(value, path, rank, consumers):
    assert value.is_cuda and value.is_contiguous()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    # The existing protocol waits for export RPC completion before installation.
    torch.cuda.synchronize(value.device)
    descriptors = {}
    for consumer in consumers:
        rebuild, args = reduce_tensor(value)
        assert rebuild is rebuild_cuda_tensor
        payload = pickle.dumps(args, protocol=5)
        name = f"rank{rank}-consumer{consumer}.ipc"
        target = path / name
        with target.open("xb") as f:
            os.chmod(target, 0o600)
            f.write(payload)
        descriptors[str(consumer)] = dict(
            name=name, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)
        )
    return descriptors


def receive(path, metadata, consumer, device):
    """Import once, copy into consumer-owned storage, wait before dropping import."""
    desc = metadata["ipc_descriptors"][str(consumer)]
    file = Path(path) / desc["name"]
    assert file.parent.resolve() == Path(path).resolve()
    stat = file.stat()
    assert stat.st_uid == os.getuid() and stat.st_mode & 0o077 == 0
    payload = file.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == desc["sha256"]
    args = pickle.loads(payload)  # Only this job's private, authenticated descriptor.
    remote = rebuild_cuda_tensor(*args)
    assert list(remote.shape) == metadata["shape"] and remote.dtype == torch.bfloat16
    assert remote.is_cuda
    value = remote.to(device=device, copy=True)
    torch.cuda.synchronize(device)
    del remote
    return value
