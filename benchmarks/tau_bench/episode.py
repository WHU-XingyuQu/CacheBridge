import hashlib, json, os, time
from pathlib import Path
from client import write
from live_adapter import TauEnvironment, IntakeRouter, SGLangBackend
from cachebridge.agents import CacheBridgePolicy, sample_trajectory
from cachebridge.trajectory import TaskRef


def episode(client, tokenizer, domain, task_id, arm, *, split="test", functional=False):
    mapper_id = os.environ.get("CB_AGENT_MAPPER_ID", "unused-native-control")
    backend = SGLangBackend(client, tokenizer, mapper_id, functional=functional)
    env = TauEnvironment(client, tokenizer, domain)
    trajectory = sample_trajectory(
        TaskRef("tau-bench", domain, split, str(task_id)),
        run_id="agent-trajectory-20261031-"
        + arm
        + ("-functional" if functional else ""),
        environment=env,
        router=IntakeRouter(),
        backend=backend,
        models=("Qwen3-14B", "Qwen3-32B"),
        cache_policy=CacheBridgePolicy(
            {("Qwen3-14B", "Qwen3-32B"): mapper_id}, enabled=arm != "Native"
        ),
        seed=20261031,
        max_steps=30,
        max_new_tokens=1024,
    )
    record = dict(
        trajectory=trajectory.to_dict(),
        events=env.events,
        backend_events=backend.events,
        user_events=env.user.events if hasattr(env, "user") else [],
        initial_user=getattr(env, "initial", None),
        request=backend.req,
        transaction=backend.transaction,
        source_before=backend.source_before,
        source_after=getattr(backend, "source_after", None),
        target_before=backend.target_before,
        target_after=getattr(backend, "target_after", None),
    )
    # Infrastructure errors do not become scored agent failures.
    allowed = (
        "ValueError: context_limit_no_truncation",
        "ValueError: Malformed function-call object",
        "ValueError: Invalid function name",
        "ValueError: Unclosed tool call",
        "JSONDecodeError:",
    )
    if trajectory.error and not trajectory.error.startswith(allowed):
        record["status"] = "infrastructure_failure_preserved"
        write(
            client.output / f"{split}-{domain}-{task_id}-failed-{time.time_ns()}.json",
            record,
        )
        raise RuntimeError(trajectory.error)
    if split == "test":
        quality = env.b.official_score(
            env.env, env.done and trajectory.status == "terminal"
        )
        success = int(quality["reward"] == 1)
    else:
        quality = None
        success = 0
    source_turns = [t for t in trajectory.turns if t.route.model_id == "Qwen3-14B"]
    signature = dict(
        initial=env.initial,
        turns=[
            dict(
                prompt_ids=t.generation.prompt_ids,
                response=t.generation.response.content,
            )
            for t in source_turns
        ],
        observations=env.events[: len(source_turns)],
    )
    intake = hashlib.sha256(
        json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    row = dict(
        domain=domain,
        task_id=task_id,
        arm=arm,
        success=success,
        status=trajectory.status,
        source_intake_sha256=intake,
        handoff_count=int(backend.req is not None),
        mapped_handoffs=int(backend.transaction is not None),
        native_fallbacks=0,
        source_H_sha256=(
            hashlib.sha256(json.dumps(backend.req["history_ids"]).encode()).hexdigest()
            if backend.req
            else None
        ),
        source_KV_digest=backend.source_before,
        quality=quality,
        stop_reason=trajectory.stop_reason,
        assistant_calls=len(trajectory.turns),
        user_calls=len(env.user.events),
        generation_limit_hits=int(trajectory.status == "generation_limit"),
        malformed_tool_call_generations=int(
            bool(trajectory.error) and trajectory.error.startswith(allowed[1:])
        ),
        executed_tool_calls=sum(
            x["action"]["name"] != env.b.RESPOND_ACTION_NAME for x in env.events
        ),
        multiple_tool_call_outputs=sum(x["tool_calls_emitted"] > 1 for x in env.events),
        functional=functional,
        formal_result=False,
    )
    record["row"] = row
    path = client.output / f"{split}-{domain}-{task_id}.json"
    assert not path.exists()
    write(path, record)
    return row
