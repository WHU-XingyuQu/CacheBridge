"""CPU lifecycle and layout checks; not evidence of deployed GPU/TP behavior."""

import unittest
from types import SimpleNamespace
import torch
from cachebridge.online.blocks import (
    SourceLease,
    pack_source_block,
    install_unpublished,
)


class BoundedCoreTests(unittest.TestCase):
    def test_source_cannot_reuse_or_release_before_gpu_ack(self):
        events = []
        lease = SourceLease(
            lambda: events.append("lock"), lambda: events.append("unlock")
        )
        lease.begin()
        lease.export(0, 512)
        with self.assertRaises(RuntimeError):
            lease.export(512, 513)
        with self.assertRaises(RuntimeError):
            lease.close()
        with self.assertRaises(RuntimeError):
            lease.acknowledge(gpu_reads_complete=False)
        lease.acknowledge(gpu_reads_complete=True)
        lease.export(512, 513)
        lease.acknowledge(gpu_reads_complete=True)
        lease.close()
        self.assertEqual(events, ["lock", "unlock"])

    def test_partial_block_contiguous_same_workspace_noncontiguous_pages(self):
        k = torch.arange(3 * 700 * 2 * 4).reshape(3, 700, 2, 4).float()
        v = -k
        pool = SimpleNamespace(
            layer_num=3,
            start_layer=0,
            _get_key_buffer=lambda l: k[l],
            _get_value_buffer=lambda l: v[l],
        )
        indices = (
            torch.randperm(650, generator=torch.Generator().manual_seed(7))[:513] + 1
        )
        workspace = torch.empty(2 * 3 * 512 * 2 * 4)
        for start, stop in [(0, 65), (0, 512), (512, 513)]:
            packed = pack_source_block(
                pool,
                indices,
                start=start,
                stop=stop,
                workspace=workspace,
                block_tokens=512,
            )
            self.assertTrue(packed.is_contiguous())
            self.assertEqual(packed.data_ptr(), workspace.data_ptr())
            expected = torch.stack(
                [
                    torch.stack([k[l, indices[start:stop]], v[l, indices[start:stop]]])
                    for l in range(3)
                ],
                dim=1,
            )
            self.assertTrue(torch.equal(packed, expected))
        self.assertFalse(torch.empty(2, 3, 512, 2, 4)[:, :, :1].is_contiguous())

    def transaction(self, *, foreign_failure=False, cancel=None, broken_span=False):
        events = []
        value = torch.zeros(2, 1, 2, 1, 1)
        blocks = [(0, 2, value), (3 if broken_span else 2, 4, value)]
        kwargs = dict(
            token_count=4,
            allocate=lambda n: torch.tensor([5, 2, 9, 3]),
            free=lambda indices: events.append(("free", indices.tolist())),
            blocks=blocks,
            write=lambda value, indices, start: events.append(("write", start)),
            synchronize=lambda: events.append(("sync",)),
            ready_vote=lambda local: events.append(("vote", local))
            or (local and not foreign_failure),
            publish=lambda indices: events.append(("publish", indices.tolist())),
            cancel_after_blocks=cancel,
        )
        return events, kwargs

    def test_publish_only_after_complete_writes_and_ready_vote(self):
        events, kwargs = self.transaction()
        result = install_unpublished(**kwargs)
        self.assertTrue(result["published"])
        self.assertEqual(events[-2][0], "vote")
        self.assertEqual(events[-1][0], "publish")
        self.assertNotIn("free", [e[0] for e in events])

    def test_cancellation_or_foreign_rank_failure_frees_unpublished_once(self):
        for options in [
            dict(cancel=1),
            dict(foreign_failure=True),
            dict(broken_span=True),
        ]:
            events, kwargs = self.transaction(**options)
            with self.assertRaisesRegex(RuntimeError, "rolled back"):
                install_unpublished(**kwargs)
            self.assertEqual([e[0] for e in events].count("free"), 1)
            self.assertNotIn("publish", [e[0] for e in events])

    def test_broken_vote_does_not_claim_safe_global_reclamation(self):
        events, kwargs = self.transaction()

        def fail(_):
            raise RuntimeError("distributed group unavailable")

        kwargs["ready_vote"] = fail
        with self.assertRaisesRegex(RuntimeError, "distributed group"):
            install_unpublished(**kwargs)
        self.assertNotIn("free", [e[0] for e in events])


if __name__ == "__main__":
    unittest.main()
