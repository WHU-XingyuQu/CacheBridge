"""Serializable agent trajectories shared by routing, fitting and evaluation.

No scores or reference answers are passed through the sampling interface.
Backend receipts describe observed execution; token checks alone do not prove
GPU cache correctness, which remains the serving adapter's responsibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

CacheMode = Literal["native_prefill", "reuse", "cachebridge"]


def token_digest(ids: tuple[int, ...]) -> str:
    return hashlib.sha256(
        json.dumps(list(ids), separators=(",", ":")).encode()
    ).hexdigest()


def validate_ids(ids: tuple[int, ...], name: str) -> None:
    if not isinstance(ids, tuple) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError(f"{name} must be a tuple of nonnegative integer token IDs")


@dataclass(frozen=True)
class TaskRef:
    benchmark: str
    domain: str
    split: Literal["train", "dev", "test"]
    task_id: str

    def __post_init__(self):
        if not all(
            isinstance(v, str) and v
            for v in (self.benchmark, self.domain, self.task_id)
        ):
            raise ValueError("task identity fields must be nonempty strings")
        if self.split not in ("train", "dev", "test"):
            raise ValueError("unknown task split")

    @property
    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def __post_init__(self):
        if self.role not in ("system", "user", "assistant", "tool") or not isinstance(
            self.content, str
        ):
            raise ValueError("invalid message")


@dataclass(frozen=True)
class Observation:
    messages: tuple[Message, ...]
    done: bool = False
    stop_reason: str | None = None

    def __post_init__(self):
        if not isinstance(self.messages, tuple) or not all(
            isinstance(m, Message) for m in self.messages
        ):
            raise ValueError("observation messages must be an immutable tuple")
        if self.done and not self.stop_reason:
            raise ValueError("terminal observation needs a stop reason")


@dataclass(frozen=True)
class RouteDecision:
    model_id: str
    reason: str

    def __post_init__(self):
        if not all(isinstance(v, str) and v for v in (self.model_id, self.reason)):
            raise ValueError("route needs a model ID and a reason")


@dataclass(frozen=True)
class CachePlan:
    mode: CacheMode
    source_model: str | None = None
    mapper_id: str | None = None


@dataclass(frozen=True)
class Generation:
    response: Message
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    tokenizer_id: str
    model_id: str
    executed_mode: CacheMode
    reused_tokens: int = 0
    # Actual source prefix at a switch, including any preparation by the backend.
    # Native collection should also expose this candidate boundary for fitting.
    handoff_prefix_ids: tuple[int, ...] = ()
    mapper_id: str | None = None
    fallback_reason: str | None = None
    finish_reason: str = "stop"
    ttft_ms: float | None = None
    completion_ms: float | None = None

    def __post_init__(self):
        for name in ("prompt_ids", "completion_ids", "handoff_prefix_ids"):
            validate_ids(getattr(self, name), name)
        if (
            not self.prompt_ids
            or not self.tokenizer_id
            or not self.model_id
            or self.response.role != "assistant"
        ):
            raise ValueError(
                "generation needs a prompt, tokenizer identity and assistant response"
            )
        if self.executed_mode not in ("native_prefill", "reuse", "cachebridge"):
            raise ValueError("unknown executed cache mode")
        if type(self.reused_tokens) is not int or not 0 <= self.reused_tokens <= len(
            self.prompt_ids
        ):
            raise ValueError("invalid reused-token count")
        if self.prompt_ids[: len(self.handoff_prefix_ids)] != self.handoff_prefix_ids:
            raise ValueError("handoff source prefix differs from the target prompt")
        if self.executed_mode == "native_prefill" and self.reused_tokens:
            raise ValueError("native prefill cannot report reused tokens")
        if self.executed_mode == "reuse" and not self.reused_tokens:
            raise ValueError("same-model reuse must report an actual cached prefix")
        if self.executed_mode == "cachebridge":
            if (
                not self.mapper_id
                or not self.handoff_prefix_ids
                or self.reused_tokens != len(self.handoff_prefix_ids)
            ):
                raise ValueError(
                    "CacheBridge needs an artifact and complete transferred-prefix count"
                )
        elif self.mapper_id is not None:
            raise ValueError("only actual CacheBridge execution may report a mapper")
        for value in (self.ttft_ms, self.completion_ms):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("timings must be finite and nonnegative")


@dataclass(frozen=True)
class Turn:
    step: int
    messages: tuple[Message, ...]
    route: RouteDecision
    cache_plan: CachePlan
    generation: Generation


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    run_id: str
    task: TaskRef
    seed: int
    turns: tuple[Turn, ...]
    final_observation: Observation | None
    status: Literal["terminal", "step_limit", "generation_limit", "error"]
    stop_reason: str
    error: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Create a new JSON record; never overwrite an earlier trajectory."""
        with Path(path).open("x") as stream:
            json.dump(
                self.to_dict(), stream, ensure_ascii=False, allow_nan=False, indent=2
            )
            stream.write("\n")

    @classmethod
    def from_dict(cls, data: dict) -> Trajectory:
        data = dict(data)
        if data.get("schema_version") != 1:
            raise ValueError("unsupported trajectory schema")
        task = TaskRef(**data.pop("task"))
        turns = []
        for raw in data.pop("turns"):
            raw = dict(raw)
            generation = dict(raw.pop("generation"))
            generation["response"] = Message(**generation["response"])
            for name in ("prompt_ids", "completion_ids", "handoff_prefix_ids"):
                generation[name] = tuple(generation[name])
            turns.append(
                Turn(
                    step=raw["step"],
                    messages=tuple(Message(**m) for m in raw["messages"]),
                    route=RouteDecision(**raw["route"]),
                    cache_plan=CachePlan(**raw["cache_plan"]),
                    generation=Generation(**generation),
                )
            )
        observation = data.pop("final_observation")
        if observation is not None:
            observation = dict(observation)
            observation["messages"] = tuple(
                Message(**m) for m in observation["messages"]
            )
            observation = Observation(**observation)
        return cls(task=task, turns=tuple(turns), final_observation=observation, **data)
