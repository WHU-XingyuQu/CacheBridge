"""CPU-reviewed transfer geometry only; no installed CUDA or service endpoint."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Run:
    source_row: int
    destination_token: int
    rows: int


@dataclass(frozen=True)
class Plan:
    start: int
    stop: int
    pool_rows: int
    layers: int
    heads: int
    dimension: int
    element_bytes: int
    runs: tuple

    @property
    def useful_bytes(self):
        return (
            2
            * self.layers
            * (self.stop - self.start)
            * self.heads
            * self.dimension
            * self.element_bytes
        )

    @property
    def copy_calls(self):
        return 2 * self.layers * len(self.runs)

    def spans(self):
        """Per-native-layer source offsets; contiguous packed-block destination offsets."""
        n = self.stop - self.start
        row_bytes = self.heads * self.dimension * self.element_bytes
        for kind in range(2):
            for layer in range(self.layers):
                for run in self.runs:
                    yield dict(
                        kind=kind,
                        layer=layer,
                        source_byte_offset=run.source_row * row_bytes,
                        destination_byte_offset=(
                            (kind * self.layers + layer) * n + run.destination_token
                        )
                        * row_bytes,
                        bytes=run.rows * row_bytes,
                    )


def build_plan(
    indices,
    *,
    start,
    stop,
    pool_rows,
    layers=40,
    heads=2,
    dimension=128,
    element_bytes=2
):
    """Coalesce adjacent physical rows, preserving original logical token order."""
    if any(
        type(x) is not int
        for x in [start, stop, pool_rows, layers, heads, dimension, element_bytes]
    ):
        raise ValueError("integer geometry required")
    if (
        not 0 <= start < stop <= len(indices)
        or min(pool_rows, layers, heads, dimension, element_bytes) <= 0
        or element_bytes != 2
    ):
        raise ValueError("invalid geometry")
    selected = tuple(indices[start:stop])
    if any(type(row) is not int or not 0 <= row < pool_rows for row in selected):
        raise ValueError("native row outside certified pool geometry")
    runs = []
    i = 0
    while i < len(selected):
        j = i + 1
        while j < len(selected) and selected[j] == selected[j - 1] + 1:
            j += 1
        runs.append(Run(selected[i], i, j - i))
        i = j
    return Plan(
        start, stop, pool_rows, layers, heads, dimension, element_bytes, tuple(runs)
    )


def cpu_materialize(pool, plan, workspace):
    """CPU semantic oracle for future receiver-owned DMA; never executes CUDA."""
    import torch

    first = pool._get_key_buffer(pool.start_layer)
    if first.device.type != "cpu" or workspace.device.type != "cpu":
        raise RuntimeError("CPU prototype only")
    if (
        plan.element_bytes != first.element_size()
        or first.dtype != torch.bfloat16
        or workspace.dtype != first.dtype
        or not workspace.is_contiguous()
    ):
        raise ValueError("BF16 contiguous workspace required")
    if (
        first.shape != (plan.pool_rows, plan.heads, plan.dimension)
        or pool.layer_num != plan.layers
    ):
        raise ValueError("native pool shape mismatch")
    n = plan.stop - plan.start
    required = 2 * plan.layers * n * plan.heads * plan.dimension
    if workspace.ndim != 1 or workspace.numel() < required:
        raise ValueError("short receiver workspace")
    target = workspace[:required].view(2, plan.layers, n, plan.heads, plan.dimension)
    for kind in range(2):
        for layer in range(plan.layers):
            source = (pool._get_key_buffer if kind == 0 else pool._get_value_buffer)(
                pool.start_layer + layer
            )
            if (
                source.device.type != "cpu"
                or source.dtype != first.dtype
                or source.shape != first.shape
                or not source.is_contiguous()
            ):
                raise ValueError("native layer geometry mismatch")
            for run in plan.runs:
                target[
                    kind,
                    layer,
                    run.destination_token : run.destination_token + run.rows,
                ].copy_(source[run.source_row : run.source_row + run.rows])
    return target
