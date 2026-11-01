"""Verify a complete mapper instance before loading its tensors."""

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_descriptor(path, *, check_tensors=True):
    path = Path(path).expanduser().resolve()
    d = json.loads(path.read_text())
    root = Path(d["mapper_dir"])
    root = root.resolve() if root.is_absolute() else (path.parent / root).resolve()
    depth = d.get("target_layers", 64)
    expected = {f"layer_{i:03d}.safetensors" for i in range(depth)}
    if (
        set(d["file_hashes"]) != expected
        or {p.name for p in root.glob("layer_*.safetensors")} != expected
    ):
        raise ValueError("Missing, extra or noncontiguous mapper layers")
    tensor_bytes = 0
    if check_tensors:
        import torch
        from safetensors.torch import load_file
    for name, digest in d["file_hashes"].items():
        p = root / name
        if file_sha256(p) != digest:
            raise ValueError(f"Mapper hash mismatch: {name}")
        if not check_tensors:
            continue
        values = load_file(str(p))
        needed = {"selected", "headwise", "k_weight", "k_bias", "v_weight", "v_bias"}
        if set(values) != needed or not bool(values["headwise"]):
            raise ValueError("Invalid head-local mapper schema")
        selected = values["selected"]
        if (
            selected.ndim != 1
            or selected.dtype != torch.int64
            or len(set(selected.tolist())) != len(selected)
        ):
            raise ValueError("Invalid source-layer support")
        if (
            selected.numel() == 0
            or int(selected.min()) < 0
            or int(selected.max()) >= d.get("source_layers", 40)
        ):
            raise ValueError("Source-layer support out of range")
        heads, dim = d.get("kv_heads", 8), d.get("head_dim", 128)
        for kind in ("k", "v"):
            w, b = values[kind + "_weight"], values[kind + "_bias"]
            if w.shape != (heads, len(selected) * dim, dim) or b.shape != (heads, dim):
                raise ValueError("Mapper geometry mismatch")
            if (
                w.dtype != torch.float32
                or b.dtype != torch.float32
                or not torch.isfinite(w).all()
                or not torch.isfinite(b).all()
            ):
                raise ValueError("Mapper coefficients must be finite FP32")
            tensor_bytes += w.numel() * w.element_size() + b.numel() * b.element_size()
    if check_tensors and tensor_bytes != d["logical_parameter_bytes"]:
        raise ValueError("Logical parameter byte mismatch")
    return dict(
        status="passed",
        mapper_id=d["mapper_id"],
        mapper_dir=str(root),
        layers=depth,
        logical_parameter_bytes=(
            tensor_bytes if check_tensors else d["logical_parameter_bytes"]
        ),
        descriptor_sha256=file_sha256(path),
        quality=d.get("quality", "unmeasured"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("descriptor", type=Path)
    args = parser.parse_args(argv)
    import torch

    torch.set_num_threads(2)
    print(json.dumps(verify_descriptor(args.descriptor), indent=2))


if __name__ == "__main__":
    main()
