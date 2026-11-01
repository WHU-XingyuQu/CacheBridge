"""Engine-independent coordination of an unpublished cache handoff."""

from dataclasses import dataclass
import time
from typing import Protocol


class HandoffEndpoint(Protocol):
    def start(self, role, parameters): ...
    def receive(self, role, deadline): ...
    def try_publish(self, label, parameters): ...


@dataclass(frozen=True)
class HandoffRequest:
    request_id: str
    phase: str
    history_ids: tuple[int, ...]
    namespace: str
    target_ranks: int = 4
    block_tokens: int = 2048

    def __post_init__(self):
        if not all(
            isinstance(s, str) and s
            for s in (self.request_id, self.phase, self.namespace)
        ):
            raise ValueError(
                "A handoff requires request, phase and namespace identities"
            )
        if not self.history_ids or any(
            type(i) is not int or i < 0 for i in self.history_ids
        ):
            raise ValueError("history_ids must contain non-negative token IDs")
        if any(
            type(n) is not int or n <= 0 for n in (self.target_ranks, self.block_tokens)
        ):
            raise ValueError("Rank and block counts must be positive integers")

    def parameters(self):
        return dict(
            action="online_stage_start",
            phase=self.phase,
            request_id=self.request_id,
            history_ids=list(self.history_ids),
            mapper_kind="CacheBridge",
            execution="double",
            block_tokens=self.block_tokens,
            cache_namespace=self.namespace,
            source_pipeline="double",
            instrument_gpu_stages=False,
        )


def run_handoff(
    endpoint: HandoffEndpoint,
    request: HandoffRequest,
    *,
    timeout=120,
    poll_interval=0.002,
):
    """Start both peers, then require complete all-rank publication evidence.

    Failure leaves outstanding resources owned by the backend. This function
    never assumes a timeout proves GPU operations have stopped or frees pages.
    """
    if timeout <= 0 or poll_interval < 0:
        raise ValueError("Invalid handoff timeout or polling interval")
    parameters = request.parameters()
    began = time.perf_counter_ns()
    deadline = time.monotonic() + timeout
    for role in ("target", "source"):
        endpoint.start(
            role,
            dict(
                parameters,
                action=(
                    "online_stage_start" if role == "target" else "online_stage_source"
                ),
            ),
        )
    replies = {role: endpoint.receive(role, deadline) for role in ("target", "source")}
    observations = []
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError("mapper publication")
        label = request.phase + "-ready-" + str(len(observations))
        rows = endpoint.try_publish(label, parameters)
        if len(rows) != request.target_ranks or any(
            row.get("phase") != label
            or row.get("transaction_phase") != request.phase
            or row.get("request_id") != request.request_id
            or row.get("rank") != rank
            or not row.get("all_rank_identity_valid")
            for rank, row in enumerate(rows)
        ):
            raise RuntimeError("Stale or incomplete all-rank publication identity")
        outcomes = {row.get("outcome") for row in rows}
        observations.append(rows)
        if outcomes == {"published"}:
            if not all(
                all(
                    row.get(field)
                    for field in (
                        "worker_done",
                        "worker_thread_exited",
                        "all_rank_ready",
                        "original_commit_stack_called",
                        "certified_data_completion_ns",
                    )
                )
                for row in rows
            ):
                raise RuntimeError(
                    "Publication lacks complete GPU-read and worker completion evidence"
                )
            break
        if outcomes != {"pending"}:
            raise RuntimeError(f"Inconsistent publication outcomes: {outcomes}")
        time.sleep(poll_interval)
    return dict(
        parameters=parameters,
        replies=replies,
        readiness=observations,
        mapping_started_ns=began,
        committed_ns=time.perf_counter_ns(),
    )
