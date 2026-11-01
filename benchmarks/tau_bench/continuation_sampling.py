"""Scoped longer TRAIN continuation with exact original-prefix certification."""

import shutil, time
from pathlib import Path
import json, torch
import tau_adapter as b
from sampling_helpers import stream, LocalUser, live_request
from client import write
from cachebridge.artifacts import file_sha256


def collect(client, tok, index, protocol, reference):
    phase = f"tau-add32-{index}"
    client.episode_namespace = phase
    client.reset("source", phase + "-empty-source")
    client.reset("target", phase + "-empty-target")
    env = b.new_env("train", index)
    user = LocalUser(client, tok, phase)
    env.user = user
    initial = env.reset(task_index=index).observation
    # Remove evaluation labels before any agent/tool continuation. The ordinary
    # environment loader constructs Task objects, but calibration never consumes
    # their gold actions/outputs and cannot score or choose trajectories by them.
    env.task = env.task.model_copy(update={"actions": [], "outputs": []})
    env.tasks = []
    text = b.initial_text(env, tok, initial)
    events = []
    req = None
    source_status = None
    transition_step = None
    stop = None
    done = False
    target_calls = 0
    teacher_ids = None
    files = []
    checkpoint_checked = False

    def snapshot(role, ids):
        label = phase + "-" + role
        client.rpc(
            role,
            "system_snapshot",
            label,
            history_ids=ids,
            cache_namespace=phase + "-agent",
        )
        for rank in range(4):
            source = client.root / role / f"{label}-snapshot-rank{rank}.pt"
            destination = client.output / source.name
            assert not destination.exists()
            shutil.move(str(source), str(destination))
            files.append(
                dict(
                    role=role,
                    rank=rank,
                    path=str(destination),
                    bytes=destination.stat().st_size,
                )
            )

    for step in range(6 + protocol["max_target_calls"]):
        ids = tok.encode(text + b.ASSISTANT, add_special_tokens=False)
        role = "source" if req is None else "target"
        out, _ = stream(
            client,
            role,
            ids,
            phase + f"-agent{step}",
            time.perf_counter_ns(),
            b.SAMPLING,
        )
        if req is not None:
            target_calls += 1
            assert ids[: req["H"]] == req["history_ids"]
            if target_calls == 1:
                assert out["result"]["meta_info"]["cached_tokens"] == 0
        event = dict(
            step=step,
            role=role,
            prompt_ids=ids,
            output=out,
            action=None,
            observation=None,
        )
        raw = out["result"]["text"]
        # Preserve native generated text even on a bounded/invalid completion.
        teacher_ids = tok.encode(text + b.ASSISTANT + raw, add_special_tokens=False)
        if out["result"]["meta_info"]["finish_reason"]["type"] == "length":
            event["limit"] = "agent_output_token_limit"
            events.append(event)
            stop = event["limit"]
            break
        try:
            action, count = b.parse_action(raw)
        except (ValueError, TypeError) as exc:
            event.update(limit="malformed_tool_call", parse_error=str(exc))
            events.append(event)
            stop = event["limit"]
            break
        # DeferredRetailEnv.calculate_reward is the existing no-scoring override.
        # Neither official_score nor task.actions/task.outputs is accessed here.
        response = env.step(action)
        event.update(
            action=action.model_dump(),
            observation=response.observation,
            tool_calls_emitted=count,
        )
        events.append(event)
        text = b.append_response(text, raw, action, response.observation)
        teacher_ids = tok.encode(text + b.ASSISTANT, add_special_tokens=False)
        done = response.done
        if req is not None and target_calls == reference["native_target_calls"]:
            assert not checkpoint_checked
            assert initial == reference["initial_user"]
            assert req["history_ids"] == reference["history_ids"]
            assert (
                teacher_ids[: reference["H"] + reference["Q"]]
                == reference["history_ids"] + reference["query_ids"]
            )
            assert len(teacher_ids) - reference["H"] == reference["full_trajectory_Q"]
            checkpoint_root = client.output / ("checkpoint_" + str(index))
            checkpoint_root.mkdir()
            old_raw_path = Path(reference["record_path"])
            old_raw = json.loads(old_raw_path.read_text())
            assert len(events) == len(old_raw["events"])
            for observed, expected in zip(events, old_raw["events"], strict=True):
                for field in ("role", "prompt_ids", "action", "observation"):
                    assert observed[field] == expected[field], (index, field)
                assert (
                    observed["output"]["result"]["text"]
                    == expected["output"]["result"]["text"]
                )
            checkpoint_ids = reference["history_ids"] + reference["query_ids"]
            stream(
                client,
                "target",
                checkpoint_ids,
                phase + "-short-prefix-query",
                time.perf_counter_ns(),
                dict(b.SAMPLING, max_new_tokens=1),
            )
            label = phase + "-short-target"
            client.rpc(
                "target",
                "system_snapshot",
                label,
                history_ids=checkpoint_ids,
                cache_namespace=phase + "-agent",
            )
            checkpoint_files = list(files)
            for rank in range(4):
                source = (
                    client.root
                    / "target"
                    / (label + "-snapshot-rank" + str(rank) + ".pt")
                )
                destination = checkpoint_root / source.name
                shutil.move(str(source), str(destination))
                checkpoint_files.append(
                    dict(role="target", rank=rank, path=str(destination))
                )
            comparisons = []
            for observed in checkpoint_files:
                expected = next(
                    f
                    for f in reference["files"]
                    if (f["role"], f["rank"]) == (observed["role"], observed["rank"])
                )
                a = torch.load(
                    observed["path"], map_location="cpu", weights_only=True, mmap=True
                )
                z = torch.load(
                    expected["path"], map_location="cpu", weights_only=True, mmap=True
                )
                equal = a.shape == z.shape and torch.equal(a, z)
                comparisons.append(
                    dict(
                        role=observed["role"],
                        rank=observed["rank"],
                        shape=list(a.shape),
                        bitwise_equal=equal,
                    )
                )
                del a, z
            review = dict(
                status=(
                    "passed"
                    if all(c["bitwise_equal"] for c in comparisons)
                    else "failed_preserved"
                ),
                task=index,
                original_H_Q_tokens_exact=True,
                all_prior_agent_text_actions_observations_exact=True,
                comparisons=comparisons,
                old_raw_sha256=file_sha256(old_raw_path),
            )
            write(checkpoint_root / "review.json", review)
            assert len(comparisons) == 8 and review["status"] == "passed", (
                index,
                review,
            )
            checkpoint_checked = True
        if done:
            stop = "native_trajectory_ended" if req else "no_handoff_source_ended"
            break
        if req is None:
            reason = b.boundary(action, response.observation, step + 1)
            if reason:
                text += b.TRANSITION
                initial_ids = tok.encode(text + b.ASSISTANT, add_special_tokens=False)
                req, source_status = live_request(client, initial_ids, phase + "-H")
                req["transition_reason"] = reason
                transition_step = step + 1
                snapshot("source", req["history_ids"])
        elif len(teacher_ids) - req["H"] >= protocol["max_query_tokens"]:
            stop = "calibration_query_budget"
            break
        elif target_calls >= protocol["max_target_calls"]:
            stop = "calibration_target_call_budget"
            break
    if stop is None:
        stop = "calibration_target_call_budget"
    common = dict(
        task_id=index,
        split="train",
        events=events,
        user_events=user.events,
        user_messages=user.messages,
        initial_user=initial,
        task_answers_read=False,
        stop_reason=stop,
        native_target_calls=target_calls,
        terminal=done,
        transition_step=transition_step,
        formal_result=False,
    )
    if req is None or target_calls == 0:
        return dict(common, status="no_usable_handoff", files=files)
    assert checkpoint_checked, "Continuation may not bypass the original prefix gate"
    assert teacher_ids[: req["H"]] == req["history_ids"]
    full_q = len(teacher_ids) - req["H"]
    q = min(full_q, protocol["max_query_tokens"])
    ids = teacher_ids[: req["H"] + q]
    native, _ = stream(
        client,
        "target",
        ids,
        phase + "-long-native-snapshot-query",
        time.perf_counter_ns(),
        dict(b.SAMPLING, max_new_tokens=1),
    )
    snapshot("target", ids)
    return dict(
        common,
        status="calibration_input_complete",
        H=req["H"],
        Q=q,
        full_trajectory_Q=full_q,
        calibration_query_truncated=full_q > q,
        history_ids=req["history_ids"],
        query_ids=ids[req["H"] :],
        input_sha256=b.hash_value(ids),
        files=files,
        source_status=source_status,
        request=req,
        native_query_observation=native,
        teacher="Native model continuation and ordinary tool/user observations; no gold actions or answers",
        original_short_prefix_checkpoint_verified=checkpoint_checked,
    )
