"""Exact installed CUDA12.6 peer API; no kernels or implicit tensor fences."""

import ctypes
from pathlib import Path

import os
import importlib.util


def cuda_runtime_library():
    explicit = os.environ.get("CACHEBRIDGE_CUDART")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    package = importlib.util.find_spec("nvidia.cuda_runtime")
    if package and package.submodule_search_locations:
        for directory in package.submodule_search_locations:
            path = Path(directory) / "lib/libcudart.so.12"
            if path.is_file():
                return path
    raise RuntimeError("Set CACHEBRIDGE_CUDART to the tested CUDA12.6 libcudart.so.12")


LIB = cuda_runtime_library()


class PeerRuntime:
    def __init__(self):
        self.lib = ctypes.CDLL(str(LIB))
        self.fn = self.lib.cudaMemcpyPeerAsync
        self.fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self.fn.restype = ctypes.c_int
        self.lib.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cudaRuntimeGetVersion.restype = ctypes.c_int
        version = ctypes.c_int()
        error = self.lib.cudaRuntimeGetVersion(ctypes.byref(version))
        if error or version.value != 12060:
            raise RuntimeError(f"Unexpected installed peer ABI {error}/{version.value}")
        self.version = version.value

    def copy(self, dst, dst_device, src, src_device, size, stream):
        if (
            min(dst, src, stream, size) <= 0
            or min(dst_device, src_device) < 0
            or dst_device == src_device
        ):
            raise ValueError("invalid asynchronous peer-copy pointers/devices/stream")
        result = self.fn(dst, dst_device, src, src_device, size, stream)
        if result:
            raise RuntimeError(f"cudaMemcpyPeerAsync error {result}")
