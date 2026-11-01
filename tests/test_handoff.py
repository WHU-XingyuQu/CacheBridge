import pytest

from cachebridge.online.handoff import HandoffRequest, run_handoff


class Endpoint:
    def __init__(self, *, bad=None):
        self.events = []
        self.polls = 0
        self.bad = bad

    def start(self, role, parameters):
        self.events.append(("start", role, parameters))

    def receive(self, role, deadline):
        assert [e[1] for e in self.events[:2]] == ["target", "source"]
        self.events.append(("receive", role))
        return {"ready": True}

    def try_publish(self, label, parameters):
        self.polls += 1
        rows = [
            dict(
                phase=label,
                transaction_phase=parameters["phase"],
                request_id=parameters["request_id"],
                rank=k,
                all_rank_identity_valid=True,
                outcome="pending" if self.polls == 1 else "published",
                worker_done=True,
                worker_thread_exited=True,
                all_rank_ready=True,
                original_commit_stack_called=True,
                certified_data_completion_ns=123,
            )
            for k in range(2)
        ]
        if self.bad:
            self.bad(rows)
        return rows


def request():
    return HandoffRequest(
        "request-a", "phase-a", (1, 2, 3), "episode-a", target_ranks=2
    )


def test_handoff_starts_both_peers_before_waiting_and_requires_all_rank_completion():
    endpoint = Endpoint()
    result = run_handoff(endpoint, request(), poll_interval=0)
    assert len(result["readiness"]) == 2
    assert result["parameters"]["history_ids"] == [1, 2, 3]
    assert result["committed_ns"] >= result["mapping_started_ns"]
    assert endpoint.events[0][2]["action"] == "online_stage_start"
    assert endpoint.events[1][2]["action"] == "online_stage_source"


@pytest.mark.parametrize(
    "bad",
    [
        lambda rows: rows.pop(),
        lambda rows: rows[0].update(request_id="another-request"),
        lambda rows: rows[0].update(transaction_phase="stale-phase"),
        lambda rows: rows[0].update(rank=1),
        lambda rows: rows[0].update(all_rank_identity_valid=False),
        lambda rows: rows[0].update(outcome="failed"),
        lambda rows: rows[0].update(worker_thread_exited=False),
        lambda rows: rows[0].update(certified_data_completion_ns=0),
    ],
)
def test_handoff_never_accepts_stale_partial_or_unfenced_publication(bad):
    with pytest.raises(RuntimeError):
        run_handoff(Endpoint(bad=bad), request(), poll_interval=0)


def test_handoff_timeout_retains_backend_ownership(monkeypatch):
    ticks = iter([0, 2])
    monkeypatch.setattr(
        "cachebridge.online.handoff.time.monotonic", lambda: next(ticks)
    )
    endpoint = Endpoint()
    with pytest.raises(TimeoutError):
        run_handoff(endpoint, request(), timeout=1)
    assert endpoint.polls == 0
    assert (
        len(endpoint.events) == 4
    )  # Both peers remain owned; no unsafe implicit release.
