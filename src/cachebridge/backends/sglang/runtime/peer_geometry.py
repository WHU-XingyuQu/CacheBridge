"""Pure byte geometry for existing contiguous BF16 MHA pool runs."""


def operations(start, stop, runs, *, layers=64, heads=4, dim=128, itemsize=2):
    if (
        not 0 <= start < stop
        or stop - start > 2048
        or (layers, dim, itemsize) != (64, 128, 2)
        or heads not in (2, 4)
    ):
        raise ValueError("unsupported fixed peer-copy geometry")
    n = stop - start
    rowbytes = heads * dim * itemsize
    out = []
    covered = 0
    cursor = 0
    physical = []
    for a, b, p in runs:
        if (
            not all(isinstance(x, int) for x in (a, b, p))
            or not a == cursor < b
            or p < 0
        ):
            raise ValueError("invalid ordered logical partition")
        cursor = b
        physical.append((p, p + b - a))
        left = max(start, a)
        right = min(stop, b)
        if left < right:
            covered += right - left
    ordered = sorted(physical)
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("target pages alias")
    if covered != n:
        raise ValueError("target runs do not cover exactly one block")
    for layer in range(layers):
        for a, b, p in runs:
            left = max(start, a)
            right = min(stop, b)
            if left >= right:
                continue
            for kind in [0, 1]:
                out.append(
                    (
                        kind,
                        layer,
                        ((kind * layers + layer) * n + left - start) * rowbytes,
                        (p + left - a) * rowbytes,
                        (right - left) * rowbytes,
                    )
                )
    if sum(v[4] for v in out) != 2 * layers * n * rowbytes:
        raise ValueError("copy-byte invariant")
    return out
