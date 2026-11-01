import pytest

from cachebridge.online.admission import (
    LiveMapperAdmission,
    LiveMapperCostPolicy,
    LiveMapperMechanism,
    MapperRequest,
    OnlineMapperAdmission,
    OnlineMapperRouter,
    PublicationProof,
)


def cohort(*tokens: int):
    return [MapperRequest(f"r{index}", value) for index, value in enumerate(tokens)]


def test_capacity_rejects_without_partial_state():
    admission = OnlineMapperAdmission(query_reserve_tokens_per_request=4)
    plan = admission.plan(cohort(100, 200), available_target_tokens=307)
    assert not plan.admitted
    assert plan.reason == "target_token_backpressure"
    assert plan.required_target_tokens == 308
    assert admission.snapshot()["state"] == "idle"


def test_mapper_is_fifo_serial_and_queries_coalesce_after_publication():
    admission = OnlineMapperAdmission(query_reserve_tokens_per_request=0)
    plan = admission.plan(cohort(8, 8, 8), available_target_tokens=24)
    admission.admit(plan)

    assert admission.begin_next_mapper().request_id == "r0"
    with pytest.raises(RuntimeError, match="another mapper transaction"):
        admission.begin_next_mapper()
    admission.publish_active_mapper("r0")
    with pytest.raises(RuntimeError, match="every prefix"):
        admission.coalesced_query_batch()

    assert admission.begin_next_mapper().request_id == "r1"
    admission.publish_active_mapper("r1")
    assert admission.begin_next_mapper().request_id == "r2"
    admission.publish_active_mapper("r2")
    assert admission.coalesced_query_batch() == ("r0", "r1", "r2")
    admission.finish_queries()
    assert admission.snapshot()["state"] == "idle"


def test_identity_and_ownership_mismatch_fail_closed():
    admission = OnlineMapperAdmission(query_reserve_tokens_per_request=0)
    with pytest.raises(ValueError, match="unique"):
        admission.plan(
            [MapperRequest("same", 8), MapperRequest("same", 8)],
            available_target_tokens=16,
        )
    plan = admission.plan(cohort(8), available_target_tokens=8)
    admission.admit(plan)
    admission.begin_next_mapper()
    with pytest.raises(RuntimeError, match="does not own"):
        admission.publish_active_mapper("foreign")
    admission.quarantine("ownership mismatch")
    assert admission.snapshot()["state"] == "quarantined"
    assert admission.snapshot()["error"] == "ownership mismatch"


def test_execute_cohort_connects_serial_mapper_to_one_query_batch():
    admission = OnlineMapperAdmission(query_reserve_tokens_per_request=0)
    plan = admission.plan(cohort(8, 12, 16), available_target_tokens=36)
    events = []

    def mapper_transaction(request):
        events.append(("map_and_publish", request.request_id))

    def query_batch(request_ids):
        events.append(("query_batch", request_ids))
        return {request_id: "token" for request_id in request_ids}

    result = admission.execute_cohort(
        plan,
        mapper_transaction=mapper_transaction,
        query_batch=query_batch,
    )

    assert events == [
        ("map_and_publish", "r0"),
        ("map_and_publish", "r1"),
        ("map_and_publish", "r2"),
        ("query_batch", ("r0", "r1", "r2")),
    ]
    assert result == {"r0": "token", "r1": "token", "r2": "token"}
    assert admission.snapshot()["state"] == "idle"


def test_execute_cohort_quarantines_callback_failure():
    admission = OnlineMapperAdmission(query_reserve_tokens_per_request=0)
    plan = admission.plan(cohort(8, 8), available_target_tokens=16)

    def mapper_transaction(request):
        if request.request_id == "r1":
            raise OSError("TP publication vote failed")

    with pytest.raises(OSError, match="publication vote"):
        admission.execute_cohort(
            plan,
            mapper_transaction=mapper_transaction,
            query_batch=lambda request_ids: request_ids,
        )

    snapshot = admission.snapshot()
    assert snapshot["state"] == "quarantined"
    assert snapshot["active_request"] == "r1"
    assert snapshot["error"] == "TP publication vote failed"


def test_router_uses_cachebridge_only_for_idle_preformed_throughput_cohort():
    router = OnlineMapperRouter()
    decision = router.route(
        cohort(8, 8),
        decoder_active=False,
        service_class="throughput",
        cohort_preformed=True,
    )
    assert decision.mode == "cachebridge_cohort"
    assert decision.reason == "idle_preformed_throughput_cohort"


@pytest.mark.parametrize(
    ("decoder_active", "service_class", "cohort_preformed"),
    [
        (True, "throughput", True),
        (True, "latency", False),
        (False, "latency", False),
    ],
)
def test_source_isolated_live_mapper_handles_decoder_active_and_trickle_arrivals(
    decoder_active, service_class, cohort_preformed
):
    router = OnlineMapperRouter(live_mechanism=LiveMapperMechanism())
    decision = router.route(
        cohort(8),
        decoder_active=decoder_active,
        service_class=service_class,
        cohort_preformed=cohort_preformed,
    )
    assert decision.mode == "cachebridge_live"
    assert decision.reason == "source_isolated_live_mapper"

    calls = []
    result = router.dispatch(
        decision,
        cachebridge_live=lambda requests: calls.append(("live", requests)) or "ok",
        cachebridge_cohort=lambda requests: calls.append(("cohort", requests)),
        native_reprefill=lambda requests: calls.append(("native", requests)),
    )
    assert result == "ok"
    assert [name for name, _ in calls] == ["live"]


@pytest.mark.parametrize("decoder_active", [False, True])
def test_live_cost_policy_routes_below_break_even_to_native_without_idle_wait(
    decoder_active,
):
    router = OnlineMapperRouter(
        live_mechanism=LiveMapperMechanism(),
        live_cost_policy=LiveMapperCostPolicy(min_history_tokens=4096),
    )
    short = router.route(
        cohort(2048),
        decoder_active=decoder_active,
        service_class="latency",
        cohort_preformed=False,
    )
    assert short.mode == "native_reprefill"
    assert short.reason == "below_live_mapper_break_even"

    long = router.route(
        cohort(4096),
        decoder_active=decoder_active,
        service_class="latency",
        cohort_preformed=False,
    )
    assert long.mode == "cachebridge_live"
    assert long.reason == "source_isolated_live_mapper"


def test_live_cost_policy_routes_an_atomic_mixed_cohort_to_native():
    router = OnlineMapperRouter(
        live_mechanism=LiveMapperMechanism(),
        live_cost_policy=LiveMapperCostPolicy(min_history_tokens=4096),
    )
    decision = router.route(
        cohort(8192, 2048),
        decoder_active=True,
        service_class="throughput",
        cohort_preformed=True,
    )
    assert decision.mode == "native_reprefill"
    assert decision.reason == "below_live_mapper_break_even"


@pytest.mark.parametrize("threshold", [0, -1])
def test_live_cost_policy_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="min_history_tokens"):
        OnlineMapperRouter(
            live_mechanism=LiveMapperMechanism(),
            live_cost_policy=LiveMapperCostPolicy(min_history_tokens=threshold),
        )


def test_live_cost_policy_requires_live_mechanism():
    with pytest.raises(ValueError, match="requires a live mapper mechanism"):
        OnlineMapperRouter(
            live_cost_policy=LiveMapperCostPolicy(min_history_tokens=4096)
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mapper_compute_role", "target", "source-isolated"),
        ("mapped_output_slots", 1, "two output slots"),
        ("direct_target_placement", False, "direct target placement"),
        ("target_mapper_workspace_bytes", 1, "target mapper workspace"),
        ("target_transfer_workspace_bytes", 1, "target transfer workspace"),
        ("scheduler_serialized_allocation", False, "allocation"),
        ("scheduler_serialized_publication", False, "publication"),
        ("source_slot_gpu_ack", False, "GPU-complete ACKs"),
        ("max_staged_transactions", 2, "one unpublished transaction"),
    ],
)
def test_live_mechanism_fails_closed_when_an_isolation_property_is_missing(
    field, value, message
):
    values = vars(LiveMapperMechanism()) | {field: value}
    with pytest.raises(ValueError, match=message):
        OnlineMapperRouter(live_mechanism=LiveMapperMechanism(**values))


def test_live_admission_publishes_during_active_decode_with_proof():
    admission = LiveMapperAdmission(
        mechanism=LiveMapperMechanism(), query_reserve_tokens=4
    )
    request = MapperRequest("late-arrival", 8)
    plan = admission.plan(request, available_target_tokens=12, decoder_active=True)
    assert plan.admitted
    assert plan.reason == "admitted_during_decode"
    admission.begin(plan)
    admission.mark_source_complete(request.request_id)
    admission.publish(
        request.request_id,
        PublicationProof(gpu_reads_complete=True, tp_vote_passed=True),
    )
    admission.mark_submitted(request.request_id)
    snapshot = admission.snapshot()
    assert snapshot["state"] == "idle"
    assert snapshot["completed_request_ids"] == (request.request_id,)
    assert snapshot["target_mapper_workspace_bytes"] == 0
    assert snapshot["target_transfer_workspace_bytes"] == 0


def test_live_admission_serializes_unpublished_ownership_and_fails_closed():
    admission = LiveMapperAdmission(mechanism=LiveMapperMechanism())
    first = admission.plan(
        cohort(8)[0], available_target_tokens=40, decoder_active=True
    )
    admission.begin(first)
    second = admission.plan(
        MapperRequest("second", 8),
        available_target_tokens=40,
        decoder_active=True,
    )
    assert not second.admitted
    assert second.reason == "mapper_transaction_in_flight"
    admission.mark_source_complete("r0")
    with pytest.raises(RuntimeError, match="GPU completion and TP vote"):
        admission.publish(
            "r0", PublicationProof(gpu_reads_complete=True, tp_vote_passed=False)
        )
    assert admission.snapshot()["state"] == "quarantined"


def test_live_admission_safe_rollback_releases_transaction():
    admission = LiveMapperAdmission(mechanism=LiveMapperMechanism())
    plan = admission.plan(
        cohort(8)[0], available_target_tokens=40, decoder_active=False
    )
    admission.begin(plan)
    admission.rollback("r0", gpu_reads_complete=True, target_pages_released=True)
    assert admission.snapshot()["state"] == "idle"


def test_live_admission_cannot_roll_back_published_prefix():
    admission = LiveMapperAdmission(mechanism=LiveMapperMechanism())
    plan = admission.plan(cohort(8)[0], available_target_tokens=40, decoder_active=True)
    admission.begin(plan)
    admission.mark_source_complete("r0")
    admission.publish(
        "r0", PublicationProof(gpu_reads_complete=True, tp_vote_passed=True)
    )
    with pytest.raises(RuntimeError, match="state=published"):
        admission.rollback("r0", gpu_reads_complete=True, target_pages_released=True)


@pytest.mark.parametrize(
    ("decoder_active", "service_class", "cohort_preformed", "reason"),
    [
        (True, "throughput", True, "decoder_active"),
        (False, "latency", True, "latency_service_class"),
        (False, "throughput", False, "trickle_arrival_no_wait"),
    ],
)
def test_router_falls_back_to_native_without_waiting(
    decoder_active, service_class, cohort_preformed, reason
):
    decision = OnlineMapperRouter().route(
        cohort(8),
        decoder_active=decoder_active,
        service_class=service_class,
        cohort_preformed=cohort_preformed,
    )
    assert decision.mode == "native_reprefill"
    assert decision.reason == reason


def test_router_removes_canceled_request_before_dispatch():
    router = OnlineMapperRouter()
    decision = router.route(
        cohort(8, 8, 8),
        decoder_active=True,
        service_class="throughput",
        cohort_preformed=True,
        canceled_request_ids=["r1"],
    )
    calls = []
    result = router.dispatch(
        decision,
        cachebridge_cohort=lambda requests: calls.append(("cachebridge", requests)),
        native_reprefill=lambda requests: calls.append(("native", requests)) or "ok",
    )
    assert result == "ok"
    assert [request.request_id for request in decision.requests] == ["r0", "r2"]
    assert decision.canceled_request_ids == ("r1",)
    assert [
        (name, tuple(row.request_id for row in requests)) for name, requests in calls
    ] == [("native", ("r0", "r2"))]


def test_router_backpressures_oversized_cohort():
    router = OnlineMapperRouter(max_cohort_requests=2)
    decision = router.route(
        cohort(8, 8, 8),
        decoder_active=False,
        service_class="throughput",
        cohort_preformed=True,
    )
    assert decision.mode == "backpressure"
    with pytest.raises(RuntimeError, match="cohort_request_limit"):
        router.dispatch(
            decision,
            cachebridge_cohort=lambda requests: requests,
            native_reprefill=lambda requests: requests,
        )
