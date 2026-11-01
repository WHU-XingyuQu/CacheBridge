import pytest

from cachebridge.online.pipeline import PipelinedSourceLease
from cachebridge.online.peer_geometry import operations


def test_two_live_slots_require_individual_completion():
    events = []
    lease = PipelinedSourceLease(
        lambda: events.append("acquire"), lambda: events.append("release")
    )
    lease.begin()
    lease.export(0, 0, 2048)
    lease.export(1, 2048, 4096)
    with pytest.raises(RuntimeError):
        lease.export(0, 4096, 4097)
    with pytest.raises(RuntimeError):
        lease.close()
    with pytest.raises(RuntimeError):
        lease.acknowledge(0, gpu_reads_complete=False)
    lease.acknowledge(0, gpu_reads_complete=True)
    lease.export(0, 4096, 4097)
    lease.acknowledge(1, gpu_reads_complete=True)
    with pytest.raises(RuntimeError):
        lease.close()
    lease.acknowledge(0, gpu_reads_complete=True)
    lease.close()
    assert events == ["acquire", "release"]


@pytest.mark.parametrize("tokens", [1, 512, 2048, 4096])
def test_peer_geometry_covers_fragmented_destination_without_overlap(tokens):
    split = max(1, tokens // 2)
    runs = [(0, split, 5)]
    if split < tokens:
        runs.append((split, tokens, split + 20))
    row_bytes = 2 * 128 * 2
    copies = operations(0, tokens, runs, heads=2)
    assert sum(size for _, _, _, _, size in copies) == 2 * 64 * tokens * row_bytes
    for kind in range(2):
        for layer in range(64):
            spans = [
                (src, dst, size)
                for k, l, src, dst, size in copies
                if (k, l) == (kind, layer)
            ]
            cursor = (kind * 64 + layer) * tokens * row_bytes
            for (logical_start, logical_stop, physical), (src, dst, size) in zip(
                runs, spans
            ):
                assert src == cursor
                assert dst == physical * row_bytes
                assert size == (logical_stop - logical_start) * row_bytes
                cursor += size
            assert cursor == (kind * 64 + layer + 1) * tokens * row_bytes


def test_peer_geometry_rejects_alias_and_unsupported_block():
    with pytest.raises(ValueError, match="alias"):
        operations(0, 8, [(0, 4, 5), (4, 8, 7)], heads=2)
    with pytest.raises(ValueError, match="unsupported"):
        operations(0, 4097, [(0, 4097, 0)], heads=2)
