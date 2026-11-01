from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from cachebridge.trajectory import Generation, Message, TaskRef, Trajectory
from cachebridge.online.agent import CacheBridgePolicy, sample_trajectory
from cachebridge.offline.trajectory import (
    HandoffSample,
    handoff_samples,
    select_training_trajectories,
)

spec = importlib.util.spec_from_file_location(
    "trajectory_example", Path(__file__).parents[1] / "examples/sample_trajectory.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def sample(enabled=False, **overrides):
    args = dict(
        task=TaskRef("fixture", "orders", "train", "0"),
        run_id="fixture/mapped" if enabled else "fixture/native",
        environment=example.ExampleEnvironment(),
        router=example.ExampleRouter(),
        backend=example.ExampleBackend(),
        models=("small", "large"),
        cache_policy=CacheBridgePolicy(
            {("small", "large"): "mapper-v1"}, enabled=enabled
        ),
    )
    args.update(overrides)
    return sample_trajectory(**args)


def test_model_route_is_independent_of_cache_mode():
    native, mapped = sample(), sample(True)
    assert native.status == mapped.status == "terminal"
    assert native.trajectory_id != mapped.trajectory_id
    assert (
        sample(run_id="fixture/native/repetition2").trajectory_id
        != native.trajectory_id
    )
    assert [t.route for t in native.turns] == [t.route for t in mapped.turns]
    assert [t.generation.executed_mode for t in native.turns] == [
        "native_prefill",
        "native_prefill",
        "reuse",
    ]
    assert [t.generation.executed_mode for t in mapped.turns] == [
        "native_prefill",
        "cachebridge",
        "reuse",
    ]
    # A directed registry must never reuse the reverse mapper automatically.
    assert (
        CacheBridgePolicy({("small", "large"): "m"}).plan("large", "small").mode
        == "native_prefill"
    )


def test_serialization_and_no_overwrite(tmp_path):
    trajectory = sample(True)
    assert Trajectory.from_dict(trajectory.to_dict()) == trajectory
    trajectory.save(tmp_path / "record.json")
    with pytest.raises(FileExistsError):
        trajectory.save(tmp_path / "record.json")


def test_native_teacher_continuation_positions_and_truncation():
    full = handoff_samples(sample())[0]
    assert full.history_ids == (1, 2, 3)
    assert full.query_ids == (4, 5, 6, 7, 8, 9)
    assert full.assistant_prediction_positions == (0, 1, 3, 4)
    short = handoff_samples(sample(), max_query_tokens=3)[0]
    assert short.query_ids == (4, 5, 6)
    assert short.assistant_prediction_positions == (0, 1)
    assert short.truncated and not full.truncated
    boundary = handoff_samples(sample(), max_query_tokens=2)[0]
    assert boundary.query_ids == (4, 5)
    assert boundary.assistant_prediction_positions == (0, 1)


def test_mapped_teacher_and_changed_token_prefix_are_rejected():
    with pytest.raises(ValueError, match="native target"):
        handoff_samples(sample(True))
    native = sample()
    turn = native.turns[-1]
    changed = replace(
        turn,
        generation=replace(
            turn.generation, prompt_ids=(90,) + turn.generation.prompt_ids[1:]
        ),
    )
    with pytest.raises(ValueError, match="exact token prefix"):
        handoff_samples(replace(native, turns=native.turns[:-1] + (changed,)))


def test_train_selection_is_nested_order_invariant_and_retains_failures():
    base = sample()
    bank = [
        replace(
            base,
            task=replace(base.task, task_id=str(i)),
            trajectory_id=str(i),
            status="error" if i % 2 else "terminal",
            error="fixture" if i % 2 else None,
        )
        for i in range(8)
    ]
    first = select_training_trajectories(bank, count=3, seed="fixed")
    all_items = select_training_trajectories(reversed(bank), count=8, seed="fixed")
    assert first == all_items[:3]
    assert sum(t.status == "error" for t in all_items) == 4
    with pytest.raises(ValueError, match="duplicate"):
        select_training_trajectories(bank + [bank[0]], count=3, seed="fixed")
    with pytest.raises(ValueError, match="dev/test"):
        select_training_trajectories(
            [replace(base, task=replace(base.task, split="test"))],
            count=1,
            seed="fixed",
        )
    with pytest.raises(ValueError, match="only TRAIN"):
        handoff_samples(replace(base, task=replace(base.task, split="test")))


def test_failure_and_limit_records_survive_and_backend_is_closed():
    class Failing(example.ExampleBackend):
        closed = False

        def generate(self, request):
            if self.calls == 1:
                raise RuntimeError("fixture generation failed")
            return super().generate(request)

        def end_episode(self, identity):
            self.closed = True

    backend = Failing()
    trajectory = sample(backend=backend)
    assert (
        trajectory.status == "error" and len(trajectory.turns) == 1 and backend.closed
    )
    assert "fixture generation failed" in trajectory.error
    limited = sample(max_steps=1)
    assert limited.status == "step_limit" and len(limited.turns) == 1


def test_partial_begin_failure_still_calls_backend_cleanup():
    class Partial(example.ExampleBackend):
        closed = False

        def begin_episode(self, identity):
            raise RuntimeError("partial setup")

        def end_episode(self, identity):
            self.closed = True

    backend = Partial()
    assert sample(backend=backend).status == "error"
    assert backend.closed


def test_backend_cannot_substitute_another_model():
    class WrongModel(example.ExampleBackend):
        def generate(self, request):
            return replace(super().generate(request), model_id="unrequested-model")

    trajectory = sample(backend=WrongModel())
    assert trajectory.status == "error"
    assert "different model" in trajectory.error


def test_fallback_requires_explicit_reason():
    class FallingBack(example.ExampleBackend):
        reason = None

        def generate(self, request):
            value = super().generate(request)
            if request.cache_plan.mode == "cachebridge":
                return replace(
                    value,
                    executed_mode="native_prefill",
                    reused_tokens=0,
                    mapper_id=None,
                    fallback_reason=self.reason,
                )
            return value

    backend = FallingBack()
    assert sample(True, backend=backend).status == "error"
    backend.reason = "source_cache_evicted"
    trajectory = sample(True, backend=backend)
    assert trajectory.status == "terminal"
    assert trajectory.turns[1].generation.fallback_reason == "source_cache_evicted"


@pytest.mark.parametrize(
    "values",
    [
        dict(prompt_ids=(-1,)),
        dict(executed_mode="native_prefill", reused_tokens=1),
        dict(handoff_prefix_ids=(9,)),
        dict(executed_mode="cachebridge", mapper_id="m", reused_tokens=1),
        dict(ttft_ms=float("nan")),
    ],
)
def test_generation_rejects_misleading_cache_receipts(values):
    args = dict(
        response=Message("assistant", "x"),
        prompt_ids=(1, 2),
        completion_ids=(3,),
        tokenizer_id="fixture",
        model_id="small",
        executed_mode="native_prefill",
    )
    args.update(values)
    with pytest.raises(ValueError):
        Generation(**args)
