"""Frozen original sampling helpers; no original control-plane side effects."""

import copy, json, time, shutil

from pathlib import Path

import tau_adapter as b


def stream(client, role, ids, phase, epoch, sampling, transfer=None):
    client.phase = phase
    client.system_capture_progress = False
    client.system_offload_rpc = True
    client.call(role, dict(op="trace_open", trace_epoch_ns=epoch))
    tx = transfer(epoch) if transfer else None
    namespace = client.episode_namespace + ("-user" if "-user" in phase else "-agent")
    client.call(
        role,
        dict(
            op="trace_submit",
            request=dict(
                request_id=phase,
                cache_namespace=namespace,
                ids=ids,
                sampling=sampling,
                arrival_offset_ms=0,
                return_token_ids=True,
            ),
        ),
    )
    outputs = client.call(role, dict(op="trace_wait"), timeout=900)["result"]
    client.system_offload_rpc = False
    assert len(outputs) == 1 and outputs[0]["request_id"] == phase
    out = outputs[0]
    meta = out["result"]["meta_info"]
    token_ids = [int(x[1]) for x in meta["output_token_logprobs"]]
    assert len(token_ids) == meta["completion_tokens"]
    assert (
        client.tokenizer.decode(token_ids, skip_special_tokens=True)
        == out["result"]["text"]
    ), "engine-token/text mismatch"
    assert (
        meta["prompt_tokens"] == len(ids)
        and 0 <= meta["completion_tokens"] <= sampling["max_new_tokens"]
    )
    assert (
        out["arrival_ns"]
        == epoch
        <= out["submitted_ns"]
        <= out["host_first_token_ns"]
        <= out["host_finished_ns"]
    )
    return out, tx


class LocalUser:
    def __init__(self, client, tok, phase):
        self.client, self.tok, self.phase = client, tok, phase
        self.messages, self.events = [], []

    def generate(self):
        started = time.perf_counter_ns()
        # Distinct first token isolates the simulator radix namespace from the
        # mapped assistant prefix; user text/official system prompt is unchanged.
        # Restate the existing upstream termination rule on every simulator turn.
        # This message sees no tools, database state, scorer or gold actions.
        reminder = (
            "Follow the user-simulation rules above. If your requested help "
            "is complete, output exactly ###STOP### now. Do not exchange "
            "repeated thanks or farewells after resolution. Otherwise send "
            "the next customer message; provide only facts from your "
            "instruction or the conversation."
        )
        visible_messages = self.messages + [dict(role="system", content=reminder)]
        text = "\n" + self.tok.apply_chat_template(
            visible_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = self.tok.encode(text, add_special_tokens=False)
        assert ids[0] != self.tok.encode(b.ASSISTANT, add_special_tokens=False)[0]
        if len(ids) + 512 > 40960:
            raise RuntimeError("User simulator context limit")
        out, _ = stream(
            self.client,
            "target",
            ids,
            self.phase + f"-user{len(self.events)}",
            started,
            b.USER_SAMPLING,
        )
        text = out["result"]["text"]
        self.messages.append(dict(role="assistant", content=text))
        self.events.append(
            dict(
                started_ns=started,
                finished_ns=time.perf_counter_ns(),
                prompt_ids=ids,
                output=out,
            )
        )
        if out["result"]["meta_info"]["finish_reason"]["type"] == "length":
            raise RuntimeError(
                "User simulator output limit; preserve as simulator-invalid"
            )
        return text

    def reset(self, instruction=None):
        prompt = b.LLMUserSimulationEnv.build_system_prompt(self, instruction)
        self.messages = [
            dict(role="system", content=prompt),
            dict(role="user", content="Hi! How can I help you today?"),
        ]
        return self.generate()

    def step(self, content):
        self.messages.append(dict(role="user", content=content))
        return self.generate()

    def get_total_cost(self):
        return 0.0


def live_request(client, ids, phase):
    status = client.status("source", phase + "-source-H", [ids])
    lengths = {x["histories"][0]["resident_tokens"] for x in status}
    assert len(lengths) == 1
    h = lengths.pop()
    assert 0 < h < len(ids)
    return (
        dict(
            request_id=phase + "-tx",
            history_ids=ids[:h],
            query_ids=ids[h:],
            H=h,
            Q=len(ids) - h,
            max_new_tokens=1024,
            arrival_offset_ms=0,
        ),
        status,
    )


def collect(client, tok, index, protocol):
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
    assert teacher_ids[: req["H"]] == req["history_ids"]
    full_q = len(teacher_ids) - req["H"]
    q = min(full_q, protocol["max_query_tokens"])
    ids = teacher_ids[: req["H"] + q]
    native, _ = stream(
        client,
        "target",
        ids,
        phase + "-native-snapshot-query",
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
    )
