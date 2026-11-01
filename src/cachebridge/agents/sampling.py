"""Agent sampling with separate model-routing and KV-execution decisions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Protocol

from ..trajectory import (
    CachePlan,
    Generation,
    Message,
    Observation,
    RouteDecision,
    TaskRef,
    Trajectory,
    Turn,
)


@dataclass(frozen=True)
class RouteContext:
    step: int
    messages: tuple[Message, ...]
    previous_model: str | None
    available_models: tuple[str, ...]


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    model_id: str
    messages: tuple[Message, ...]
    cache_plan: CachePlan
    seed: int
    max_new_tokens: int


class AgentEnvironment(Protocol):
    """Expose observations/tools only; keep task answers and scores private."""

    def reset(self, task: TaskRef, *, seed: int) -> Observation: ...
    def step(self, response: Message) -> Observation: ...


class ModelRouter(Protocol):
    def choose(self, context: RouteContext) -> RouteDecision: ...


class AgentBackend(Protocol):
    """Own actual models, cache handles, tokenizer, transfer and completion fences.

    Begin/end scope all caches to one episode. The backend must release only its
    request resources, never model residents shared with other jobs. A fallback
    must be explicit in Generation; generation calls must not execute tools.
    """

    def begin_episode(self, trajectory_id: str) -> None: ...
    def generate(self, request: GenerationRequest) -> Generation: ...
    def end_episode(self, trajectory_id: str) -> None: ...


class CacheBridgePolicy:
    """Plan KV reuse after a router chooses a model; never change that choice."""

    def __init__(self, mappers: Mapping[tuple[str, str], str], *, enabled: bool = True):
        self.mappers = dict(mappers)
        if any(
            len(pair) != 2 or not all(pair) or not artifact
            for pair, artifact in self.mappers.items()
        ):
            raise ValueError(
                "mapper registry needs directed model pairs and artifact IDs"
            )
        self.enabled = enabled

    def plan(self, source: str | None, target: str) -> CachePlan:
        if source is None:
            return CachePlan("native_prefill")
        if source == target:
            return CachePlan("reuse", source_model=source)
        mapper = self.mappers.get((source, target)) if self.enabled else None
        return CachePlan("cachebridge" if mapper else "native_prefill", source, mapper)


def _validate_execution(request: GenerationRequest, generation: Generation) -> None:
    plan = request.cache_plan
    if generation.model_id != request.model_id:
        raise ValueError("backend ran a different model than the router selected")
    if len(generation.completion_ids) > request.max_new_tokens:
        raise ValueError("backend exceeded the requested generation budget")
    if generation.executed_mode != plan.mode:
        if (
            generation.executed_mode != "native_prefill"
            or not generation.fallback_reason
        ):
            raise ValueError(
                "cache execution mismatch without explicit native fallback"
            )
    elif generation.fallback_reason:
        raise ValueError("fallback reason supplied without a fallback")
    if generation.executed_mode == "cachebridge":
        if (
            generation.mapper_id != plan.mapper_id
            or not plan.source_model
            or plan.source_model == request.model_id
        ):
            raise ValueError("wrong directed mapper for the selected model switch")


def sample_trajectory(
    task: TaskRef,
    *,
    run_id: str,
    environment: AgentEnvironment,
    router: ModelRouter,
    backend: AgentBackend,
    models: tuple[str, ...],
    cache_policy: CacheBridgePolicy,
    seed: int = 0,
    max_steps: int = 30,
    max_new_tokens: int = 1024,
) -> Trajectory:
    """Run one complete episode, retaining errors and limits as explicit records.

    Routing sees only the current conversation and previous model. Scoring is a
    separate operation after this function, never a filter on training samples.
    Use a fresh environment/router per episode when implementations hold state.
    All engine calls and tool execution are synchronous in this reference loop.
    """
    if (
        type(max_steps) is not int
        or max_steps <= 0
        or type(max_new_tokens) is not int
        or max_new_tokens <= 0
    ):
        raise ValueError("step and generation budgets must be positive integers")
    models = tuple(models)
    if (
        not models
        or len(set(models)) != len(models)
        or any(not isinstance(m, str) or not m for m in models)
    ):
        raise ValueError("models must contain unique nonempty model IDs")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must identify the experiment arm and repetition")
    identity = hashlib.sha256(f"{run_id}:{task.key}:{seed}".encode()).hexdigest()[:24]
    turns = []
    observation = None
    previous = None
    begun = False
    status, reason, error = "step_limit", "max_agent_steps", None
    try:
        begun = True
        backend.begin_episode(identity)
        observation = environment.reset(task, seed=seed)
        for step in range(max_steps):
            if observation.done:
                status, reason = "terminal", observation.stop_reason
                break
            decision = router.choose(
                RouteContext(step, observation.messages, previous, models)
            )
            if decision.model_id not in models or not decision.reason:
                raise ValueError("router returned an unknown model or missing reason")
            plan = cache_policy.plan(previous, decision.model_id)
            request = GenerationRequest(
                f"{identity}:{step}",
                decision.model_id,
                observation.messages,
                plan,
                seed + step,
                max_new_tokens,
            )
            generated = backend.generate(request)
            _validate_execution(request, generated)
            if (
                generated.executed_mode == "cachebridge"
                and turns[-1].generation.tokenizer_id != generated.tokenizer_id
            ):
                raise ValueError(
                    "cross-tokenizer cache reuse requires a separate alignment adapter"
                )
            turns.append(Turn(step, observation.messages, decision, plan, generated))
            if generated.finish_reason == "length":
                status, reason = "generation_limit", "max_new_tokens"
                break
            observation = environment.step(generated.response)
            previous = decision.model_id
            if observation.done:
                status, reason = "terminal", observation.stop_reason
                break
    except Exception as exc:
        status, reason, error = (
            "error",
            "sampling_error",
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        if begun:
            try:
                backend.end_episode(identity)
            except Exception as exc:
                status, reason = "error", "backend_cleanup_error"
                error = (error + "; " if error else "") + f"{type(exc).__name__}: {exc}"
    return Trajectory(
        trajectory_id=identity,
        run_id=run_id,
        task=task,
        seed=seed,
        turns=tuple(turns),
        final_observation=observation,
        status=status,
        stop_reason=reason,
        error=error,
    )
