"""Load and validate explicit source support from an existing initializer."""

from pathlib import Path


def load_support(directory, *, source_layers, target_layers, support_size=None):
    from safetensors import safe_open

    directory = Path(directory)
    paths = sorted(directory.glob("layer_*.safetensors"))
    if {p.name for p in paths} != {
        f"layer_{i:03d}.safetensors" for i in range(target_layers)
    }:
        raise ValueError(f"Expected {target_layers} complete initializer layers")
    support = []
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor("selected")
        ids = tensor.tolist()
        if (
            tensor.ndim != 1
            or not ids
            or any(type(i) is not int or not 0 <= i < source_layers for i in ids)
            or len(set(ids)) != len(ids)
            or (support_size is not None and len(ids) != support_size)
        ):
            raise ValueError(f"Invalid source-layer support: {path.name}")
        support.append(tuple(ids))
    return tuple(support)
