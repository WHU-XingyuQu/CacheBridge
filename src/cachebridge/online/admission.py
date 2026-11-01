"""Concurrency-safe admission for the online CacheBridge mapper.

The mainline path maps each request after arrival while older requests decode.
Mapper compute and its two output slots live on the source ranks; the target
only reserves unpublished pages, installs mapped KV directly, and publishes
through its scheduler thread.  Cross-request publications remain FIFO and
serial even though mapping/transfer overlap target decode.

The older phase-isolated cohort state machine remains available as a safety
fallback for runtimes that cannot prove the live-path capability contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterable, Literal, TypeVar


_QueryResult = TypeVar("_QueryResult")
_RouteResult = TypeVar("_RouteResult")


@dataclass(frozen=True)
class MapperRequest:
    request_id: str
    history_tokens: int


@dataclass(frozen=True)
class AdmissionPlan:
    admitted: bool
    reason: str
    requests: tuple[MapperRequest, ...]
    required_target_tokens: int
    available_target_tokens: int
    mapper_transactions_in_flight: int
    mapper_workspace_scales_with_cohort: bool
    query_coalescing: bool


@dataclass(frozen=True)
class LiveMapperMechanism:
    """Capabilities required before mapper work may overlap target decode.

    This is a fail-closed contract, not a performance hint.  A runtime should
    instantiate it from observed topology and workspace configuration rather
    than assuming that a generic mapper is concurrency-safe.
    """

    mapper_compute_role: Literal["source", "target"] = "source"
    mapped_output_slots: int = 2
    direct_target_placement: bool = True
    target_mapper_workspace_bytes: int = 0
    target_transfer_workspace_bytes: int = 0
    scheduler_serialized_allocation: bool = True
    scheduler_serialized_publication: bool = True
    source_slot_gpu_ack: bool = True
    max_staged_transactions: int = 1

    def validate(self) -> None:
        if self.mapper_compute_role != "source":
            raise ValueError("live mapper compute must be source-isolated")
        if self.mapped_output_slots < 2:
            raise ValueError("live mapper requires at least two output slots")
        if not self.direct_target_placement:
            raise ValueError("live mapper requires direct target placement")
        if self.target_mapper_workspace_bytes != 0:
            raise ValueError("live mapper cannot allocate target mapper workspace")
        if self.target_transfer_workspace_bytes != 0:
            raise ValueError("live mapper cannot allocate target transfer workspace")
        if not self.scheduler_serialized_allocation:
            raise ValueError("target allocation must be scheduler-serialized")
        if not self.scheduler_serialized_publication:
            raise ValueError("target publication must be scheduler-serialized")
        if not self.source_slot_gpu_ack:
            raise ValueError("source slots require target GPU-complete ACKs")
        if self.max_staged_transactions != 1:
            raise ValueError("exactly one unpublished transaction is supported")


@dataclass(frozen=True)
class LiveMapperCostPolicy:
    """Evidence-calibrated break-even gate for the live mapper.

    The mechanism contract answers whether concurrent mapping is safe.  This
    policy separately answers whether its fixed transform/install cost is
    worthwhile for a request's history length.  It never waits for decoder
    idle and does not change the live path for eligible requests.
    """

    min_history_tokens: int
    objective: Literal["ttft"] = "ttft"

    def validate(self) -> None:
        if self.min_history_tokens <= 0:
            raise ValueError("min_history_tokens must be positive")
        if self.objective != "ttft":
            raise ValueError("only the measured ttft objective is supported")

    def admits(self, request: MapperRequest) -> bool:
        return request.history_tokens >= self.min_history_tokens


@dataclass(frozen=True)
class LiveAdmissionPlan:
    admitted: bool
    reason: str
    request: MapperRequest
    required_target_tokens: int
    available_target_tokens: int
    decoder_active: bool


@dataclass(frozen=True)
class PublicationProof:
    gpu_reads_complete: bool
    tp_vote_passed: bool


@dataclass(frozen=True)
class RouterDecision:
    mode: Literal[
        "cachebridge_live", "cachebridge_cohort", "native_reprefill", "backpressure"
    ]
    reason: str
    requests: tuple[MapperRequest, ...]
    canceled_request_ids: tuple[str, ...]
    decoder_active: bool
    service_class: Literal["latency", "throughput"]
    cohort_preformed: bool


class OnlineMapperRouter:
    """Route arrivals to the source-isolated live mapper when it is proven safe.

    Decoder activity, latency class, and trickle arrival are not reasons to
    abandon KV reuse.  They use ``cachebridge_live`` whenever the runtime has
    supplied a validated :class:`LiveMapperMechanism`.  The former idle-cohort
    policy is retained only for runtimes without that capability.
    """

    def __init__(
        self,
        *,
        max_cohort_requests: int = 4,
        live_mechanism: LiveMapperMechanism | None = None,
        live_cost_policy: LiveMapperCostPolicy | None = None,
    ) -> None:
        if max_cohort_requests <= 0:
            raise ValueError("max_cohort_requests must be positive")
        self.max_cohort_requests = max_cohort_requests
        if live_mechanism is not None:
            live_mechanism.validate()
        if live_cost_policy is not None:
            live_cost_policy.validate()
            if live_mechanism is None:
                raise ValueError("live cost policy requires a live mapper mechanism")
        self.live_mechanism = live_mechanism
        self.live_cost_policy = live_cost_policy

    def route(
        self,
        requests: Iterable[MapperRequest],
        *,
        decoder_active: bool,
        service_class: Literal["latency", "throughput"],
        cohort_preformed: bool,
        canceled_request_ids: Iterable[str] = (),
    ) -> RouterDecision:
        cohort = tuple(requests)
        if not cohort:
            raise ValueError("route requires at least one request")
        identities = [request.request_id for request in cohort]
        if any(not identity for identity in identities):
            raise ValueError("request identity must be non-empty")
        if len(set(identities)) != len(identities):
            raise ValueError("request identities must be unique")
        canceled = tuple(canceled_request_ids)
        if len(set(canceled)) != len(canceled):
            raise ValueError("canceled request identities must be unique")
        unknown = set(canceled) - set(identities)
        if unknown:
            raise ValueError(f"unknown canceled request identities: {sorted(unknown)}")
        canceled_set = set(canceled)
        active = tuple(row for row in cohort if row.request_id not in canceled_set)
        if not active:
            return RouterDecision(
                "backpressure",
                "cohort_canceled",
                (),
                canceled,
                decoder_active,
                service_class,
                cohort_preformed,
            )
        if len(active) > self.max_cohort_requests:
            return RouterDecision(
                "backpressure",
                "cohort_request_limit",
                active,
                canceled,
                decoder_active,
                service_class,
                cohort_preformed,
            )
        if (
            self.live_mechanism is not None
            and self.live_cost_policy is not None
            and any(not self.live_cost_policy.admits(request) for request in active)
        ):
            mode, reason = "native_reprefill", "below_live_mapper_break_even"
        elif self.live_mechanism is not None:
            mode, reason = "cachebridge_live", "source_isolated_live_mapper"
        elif decoder_active:
            mode, reason = "native_reprefill", "decoder_active"
        elif service_class == "latency":
            mode, reason = "native_reprefill", "latency_service_class"
        elif not cohort_preformed:
            mode, reason = "native_reprefill", "trickle_arrival_no_wait"
        else:
            mode, reason = "cachebridge_cohort", "idle_preformed_throughput_cohort"
        return RouterDecision(
            mode,
            reason,
            active,
            canceled,
            decoder_active,
            service_class,
            cohort_preformed,
        )

    def dispatch(
        self,
        decision: RouterDecision,
        *,
        cachebridge_live: (
            Callable[[tuple[MapperRequest, ...]], _RouteResult] | None
        ) = None,
        cachebridge_cohort: Callable[[tuple[MapperRequest, ...]], _RouteResult],
        native_reprefill: Callable[[tuple[MapperRequest, ...]], _RouteResult],
    ) -> _RouteResult:
        """Invoke exactly one already-selected serving path."""

        if decision.mode == "cachebridge_live":
            if cachebridge_live is None:
                raise RuntimeError("cachebridge_live callback is required")
            return cachebridge_live(decision.requests)
        if decision.mode == "cachebridge_cohort":
            return cachebridge_cohort(decision.requests)
        if decision.mode == "native_reprefill":
            return native_reprefill(decision.requests)
        raise RuntimeError(f"cannot dispatch backpressured route: {decision.reason}")


class LiveMapperAdmission:
    """Single-unpublished-transaction state machine for post-arrival mapping.

    The transaction may run while the decoder is active.  Only allocator and
    radix-tree mutations are serialized by the target scheduler.  Publication
    is impossible until target GPU reads/writes and all TP-rank votes complete.
    """

    def __init__(
        self,
        *,
        mechanism: LiveMapperMechanism,
        query_reserve_tokens: int = 32,
    ) -> None:
        if query_reserve_tokens < 0:
            raise ValueError("query reserve must be non-negative")
        mechanism.validate()
        self.mechanism = mechanism
        self.query_reserve_tokens = query_reserve_tokens
        self._active: MapperRequest | None = None
        self._phase: Literal[
            "idle", "mapping", "source_ready", "published", "quarantined"
        ] = "idle"
        self._decoder_active = False
        self._completed: list[str] = []
        self._error: str | None = None

    def plan(
        self,
        request: MapperRequest,
        *,
        available_target_tokens: int,
        decoder_active: bool,
    ) -> LiveAdmissionPlan:
        if not request.request_id:
            raise ValueError("request identity must be non-empty")
        if request.history_tokens <= 0:
            raise ValueError("history_tokens must be positive")
        required = request.history_tokens + self.query_reserve_tokens
        admitted = self._phase == "idle" and required <= available_target_tokens
        if self._phase != "idle":
            reason = "mapper_transaction_in_flight"
        elif not admitted:
            reason = "target_token_backpressure"
        else:
            reason = "admitted_during_decode" if decoder_active else "admitted"
        return LiveAdmissionPlan(
            admitted,
            reason,
            request,
            required,
            available_target_tokens,
            decoder_active,
        )

    def begin(self, plan: LiveAdmissionPlan) -> MapperRequest:
        if self._phase != "idle":
            raise RuntimeError(f"cannot begin while state={self._phase}")
        if not plan.admitted:
            raise RuntimeError(f"cannot execute rejected plan: {plan.reason}")
        if plan.request.request_id in self._completed:
            raise RuntimeError("request identity was already published")
        self._active = plan.request
        self._decoder_active = plan.decoder_active
        self._phase = "mapping"
        self._error = None
        return plan.request

    def mark_source_complete(self, request_id: str) -> None:
        self._require_owner(request_id, "mapping")
        self._phase = "source_ready"

    def publish(self, request_id: str, proof: PublicationProof) -> None:
        self._require_owner(request_id, "source_ready")
        if not proof.gpu_reads_complete or not proof.tp_vote_passed:
            self.quarantine("publication proof failed")
            raise RuntimeError("cannot publish without GPU completion and TP vote")
        self._phase = "published"

    def mark_submitted(self, request_id: str) -> None:
        self._require_owner(request_id, "published")
        self._completed.append(request_id)
        self._active = None
        self._decoder_active = False
        self._phase = "idle"

    def rollback(
        self,
        request_id: str,
        *,
        gpu_reads_complete: bool,
        target_pages_released: bool,
    ) -> None:
        if self._active is None or self._active.request_id != request_id:
            raise RuntimeError("rollback does not own the active transaction")
        if self._phase not in ("mapping", "source_ready"):
            raise RuntimeError(f"cannot roll back while state={self._phase}")
        if not gpu_reads_complete or not target_pages_released:
            self.quarantine("unsafe rollback")
            raise RuntimeError("rollback requires completed reads and released pages")
        self._active = None
        self._decoder_active = False
        self._phase = "idle"

    def quarantine(self, error: BaseException | str) -> None:
        self._phase = "quarantined"
        self._error = str(error)

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self._phase,
            "active_request": (
                None if self._active is None else self._active.request_id
            ),
            "decoder_active_during_mapping": self._decoder_active,
            "completed_request_ids": tuple(self._completed),
            "error": self._error,
            "mapper_compute_role": self.mechanism.mapper_compute_role,
            "mapped_output_slots": self.mechanism.mapped_output_slots,
            "max_staged_transactions": self.mechanism.max_staged_transactions,
            "target_mapper_workspace_bytes": (
                self.mechanism.target_mapper_workspace_bytes
            ),
            "target_transfer_workspace_bytes": (
                self.mechanism.target_transfer_workspace_bytes
            ),
            "direct_target_placement": self.mechanism.direct_target_placement,
            "scheduler_serialized_publication": (
                self.mechanism.scheduler_serialized_publication
            ),
        }

    def _require_owner(self, request_id: str, phase: str) -> None:
        if (
            self._phase != phase
            or self._active is None
            or self._active.request_id != request_id
        ):
            raise RuntimeError("operation does not own the active mapper transaction")


class OnlineMapperAdmission:
    """One-cohort state machine for the production throughput policy."""

    def __init__(
        self,
        *,
        max_cohort_requests: int = 4,
        query_reserve_tokens_per_request: int = 32,
    ) -> None:
        if max_cohort_requests <= 0:
            raise ValueError("max_cohort_requests must be positive")
        if query_reserve_tokens_per_request < 0:
            raise ValueError("query reserve must be non-negative")
        self.max_cohort_requests = max_cohort_requests
        self.query_reserve_tokens_per_request = query_reserve_tokens_per_request
        self._state: Literal[
            "idle", "admitted", "mapping", "ready", "dispatched", "quarantined"
        ] = "idle"
        self._requests: tuple[MapperRequest, ...] = ()
        self._pending: list[str] = []
        self._published: list[str] = []
        self._active_request: str | None = None
        self._error: str | None = None

    def plan(
        self,
        requests: Iterable[MapperRequest],
        *,
        available_target_tokens: int,
    ) -> AdmissionPlan:
        cohort = tuple(requests)
        if not cohort:
            raise ValueError("cohort must be non-empty")
        if len(cohort) > self.max_cohort_requests:
            return AdmissionPlan(
                False,
                "cohort_request_limit",
                cohort,
                0,
                available_target_tokens,
                0,
                False,
                True,
            )
        identities = [request.request_id for request in cohort]
        if any(not identity for identity in identities):
            raise ValueError("request identity must be non-empty")
        if len(set(identities)) != len(identities):
            raise ValueError("request identities must be unique")
        if any(request.history_tokens <= 0 for request in cohort):
            raise ValueError("history_tokens must be positive")
        required = sum(request.history_tokens for request in cohort) + len(cohort) * (
            self.query_reserve_tokens_per_request
        )
        admitted = required <= available_target_tokens
        return AdmissionPlan(
            admitted,
            "admitted" if admitted else "target_token_backpressure",
            cohort,
            required,
            available_target_tokens,
            1 if admitted else 0,
            False,
            True,
        )

    def admit(self, plan: AdmissionPlan) -> None:
        if self._state != "idle":
            raise RuntimeError(f"cannot admit while state={self._state}")
        if not plan.admitted:
            raise RuntimeError(f"cannot execute rejected plan: {plan.reason}")
        self._requests = plan.requests
        self._pending = [request.request_id for request in plan.requests]
        self._published = []
        self._active_request = None
        self._error = None
        self._state = "admitted"

    def begin_next_mapper(self) -> MapperRequest:
        if self._state not in ("admitted", "mapping"):
            raise RuntimeError(f"cannot start mapper while state={self._state}")
        if self._active_request is not None:
            raise RuntimeError("another mapper transaction is active")
        if not self._pending:
            raise RuntimeError("cohort has no pending mapper request")
        request_id = self._pending[0]
        self._active_request = request_id
        self._state = "mapping"
        return next(
            request for request in self._requests if request.request_id == request_id
        )

    def publish_active_mapper(self, request_id: str) -> None:
        if self._state != "mapping" or self._active_request != request_id:
            raise RuntimeError("publication does not own the active mapper transaction")
        if not self._pending or self._pending[0] != request_id:
            raise RuntimeError("mapper completion violated FIFO admission order")
        self._pending.pop(0)
        self._published.append(request_id)
        self._active_request = None
        self._state = "ready" if not self._pending else "mapping"

    def coalesced_query_batch(self) -> tuple[str, ...]:
        if self._state != "ready" or self._active_request is not None or self._pending:
            raise RuntimeError(
                "queries cannot dispatch before every prefix is published"
            )
        expected = tuple(request.request_id for request in self._requests)
        if tuple(self._published) != expected:
            raise RuntimeError("published request set differs from admitted cohort")
        self._state = "dispatched"
        return expected

    def finish_queries(self) -> None:
        if self._state != "dispatched":
            raise RuntimeError(f"cannot finish queries while state={self._state}")
        self._state = "idle"
        self._requests = ()
        self._pending = []
        self._published = []
        self._active_request = None
        self._error = None

    def execute_cohort(
        self,
        plan: AdmissionPlan,
        *,
        mapper_transaction: Callable[[MapperRequest], None],
        query_batch: Callable[[tuple[str, ...]], _QueryResult],
    ) -> _QueryResult:
        """Execute the phase-isolated policy through serving RPC callbacks.

        ``mapper_transaction`` must return only after mapping, installation,
        safety checks, and the TP publication vote have completed atomically
        for that request.  The callback is invoked synchronously so there can
        be only one cross-request mapper transaction in flight.  ``query_batch``
        is invoked once, after every admitted prefix has been published.
        Callback failures quarantine the cohort and are re-raised.
        """

        self.admit(plan)
        try:
            while self._pending:
                request = self.begin_next_mapper()
                mapper_transaction(request)
                self.publish_active_mapper(request.request_id)
            request_ids = self.coalesced_query_batch()
            result = query_batch(request_ids)
        except BaseException as error:
            self.quarantine(error)
            raise
        self.finish_queries()
        return result

    def quarantine(self, error: BaseException | str) -> None:
        self._state = "quarantined"
        self._error = str(error)

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self._state,
            "active_request": self._active_request,
            "pending_request_ids": tuple(self._pending),
            "published_request_ids": tuple(self._published),
            "error": self._error,
            "max_mapper_transactions_in_flight": 1,
            "mapper_workspace_scales_with_cohort": False,
            "query_coalescing": True,
        }
