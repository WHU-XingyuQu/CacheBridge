"""Original tau environment and real SGLang backend for the public sampler."""

import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
from cachebridge.trajectory import Generation, Message, Observation, RouteDecision
from sampling_helpers import LocalUser, live_request


def domain_adapter(domain):
    spec = importlib.util.spec_from_file_location(
        "tau_domain_" + domain, HERE / "tau_adapter.py"
    )
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    if domain == "airline":
        from tau_bench.envs.airline import MockAirlineDomainEnv

        class DeferredAirlineEnv(MockAirlineDomainEnv):
            calculate_reward = b.DeferredRetailEnv.calculate_reward

        b.new_env = lambda split, index: DeferredAirlineEnv(
            user_strategy="human", task_split=split, task_index=index
        )

        def boundary(action, observation, calls):
            if action.name == "get_reservation_details" and not observation.startswith(
                ("Error", "Unknown")
            ):
                return "first_successful_detail_query"
            return "six_call_intake_cap" if calls >= 6 else None

        b.boundary = boundary
        b.FORMAT = b.FORMAT.replace("retail policy", "airline policy")
        b.TRANSITION = b.TRANSITION.replace("retail policy", "airline policy")
    else:
        assert domain == "retail"
    return b


class TauEnvironment:
    def __init__(self, client, tokenizer, domain):
        self.client, self.tok, self.b = client, tokenizer, domain_adapter(domain)
        self.events = []
        self.transition_reason = None
        self.done = False

    def observation(self):
        messages = (
            Message(
                "system", self.text + self.b.ASSISTANT, name="tau_serialized_prompt"
            ),
        )
        if self.transition_reason:
            messages += (
                Message("system", self.transition_reason, name="intake_complete"),
            )
        return Observation(
            messages,
            done=self.done,
            stop_reason="environment_terminal" if self.done else None,
        )

    def reset(self, task, *, seed):
        self.env = self.b.new_env(task.split, int(task.task_id))
        self.user = LocalUser(self.client, self.tok, self.client.episode_namespace)
        self.env.user = self.user
        initial = self.env.reset(task_index=int(task.task_id)).observation
        if task.split == "train":
            self.env.task = self.env.task.model_copy(
                update={"actions": [], "outputs": []}
            )
            self.env.tasks = []
        self.initial = initial
        self.text = self.b.initial_text(self.env, self.tok, initial)
        return self.observation()

    def step(self, response):
        action, count = self.b.parse_action(response.content)
        result = self.env.step(action)
        self.events.append(
            dict(
                response=response.content,
                action=action.model_dump(),
                observation=result.observation,
                tool_calls_emitted=count,
            )
        )
        self.text = self.b.append_response(
            self.text, response.content, action, result.observation
        )
        self.done = result.done
        if not self.done and self.transition_reason is None:
            self.transition_reason = self.b.boundary(
                action, result.observation, len(self.events)
            )
            if self.transition_reason:
                self.text += self.b.TRANSITION
        return self.observation()


class IntakeRouter:
    def choose(self, context):
        marker = next(
            (m.content for m in context.messages if m.name == "intake_complete"), None
        )
        return RouteDecision(
            "Qwen3-32B" if marker else "Qwen3-14B", marker or "source_intake"
        )


def digest(client, role, req, phase):
    client.rpc(
        role,
        "r010_digest",
        phase,
        history_ids=req["history_ids"],
        cache_namespace=client.episode_namespace + "-agent",
    )
    rows = [
        json.loads(
            (client.root / role / f"{role}-r010-digest-rank{k}.json").read_text()
        )
        for k in range(4)
    ]
    assert all(
        x["phase"] == phase and x["tokens"] == req["H"] and x["rank"] == k
        for k, x in enumerate(rows)
    )
    return [x["sha256"] for x in rows]


from cachebridge.backends.sglang.handoff import transfer


class SGLangBackend:
    def __init__(self, client, tokenizer, mapper_id, *, functional=False):
        self.client, self.tok, self.mapper_id = client, tokenizer, mapper_id
        self.events = []
        self.req = None
        self.transaction = None
        self.source_before = None
        self.target_before = None
        self.functional = functional

    def begin_episode(self, trajectory_id):
        c = self.client
        c.episode_namespace = trajectory_id
        c.reset("source", trajectory_id + "-empty-source")
        c.reset("target", trajectory_id + "-empty-target")
        c.rpc(
            "target", "peer_copy_mode", trajectory_id + "-tensor-target", mode="tensor"
        )

    def generate(self, request):
        c = self.client
        role = "source" if request.model_id == "Qwen3-14B" else "target"
        prompt = next(
            m.content for m in request.messages if m.name == "tau_serialized_prompt"
        )
        ids = self.tok.encode(prompt, add_special_tokens=False)
        if len(ids) + request.max_new_tokens > 40960:
            raise ValueError("context_limit_no_truncation")
        switching = (
            request.cache_plan.source_model
            and request.cache_plan.source_model != request.model_id
        )
        if switching:
            assert role == "target" and self.req is None
            self.req, self.source_status = live_request(
                c, ids, request.request_id + "-H"
            )
            assert self.events and self.events[-1]["role"] == "source"
            assert self.req["H"] >= len(self.events[-1]["prompt_ids"])
            self.source_before = digest(
                c, "source", self.req, request.request_id + "-source-before"
            )
        epoch = time.perf_counter_ns()
        c.system_offload_rpc = True
        c.call(role, dict(op="trace_open", trace_epoch_ns=epoch))
        if request.cache_plan.mode == "cachebridge":
            if self.functional:
                params = dict(
                    action="concurrency_run",
                    phase=request.request_id + "-reference",
                    request_id=request.request_id + "-reference",
                    history_ids=self.req["history_ids"],
                    mapper_kind="CacheBridge",
                    execution="serial",
                    reference_block_tokens=2048,
                    cache_namespace=c.episode_namespace + "-agent",
                )
                for endpoint in ("target", "source"):
                    c.pipes[endpoint].send(dict(op="rpc", parameters=params))
                deadline = time.monotonic() + 600
                for endpoint in ("target", "source"):
                    c.receive(endpoint, deadline)
                reference = digest(
                    c, "target", self.req, request.request_id + "-reference-digest"
                )
                c.reset("target", request.request_id + "-reference-release")
            self.transaction = transfer(c, self.req, request.request_id + "-mapping")
            self.target_before = digest(
                c, "target", self.req, request.request_id + "-target-before"
            )
            if self.functional:
                assert (
                    self.target_before == reference
                ), "complete-history serial/pipeline KV mismatch"
                self.transaction["serial_reference_KV_bitwise_equal"] = True
        from tau_adapter import SAMPLING

        c.call(
            role,
            dict(
                op="trace_submit",
                request=dict(
                    request_id=request.request_id,
                    cache_namespace=c.episode_namespace + "-agent",
                    ids=ids,
                    sampling=dict(SAMPLING, max_new_tokens=request.max_new_tokens),
                    arrival_offset_ms=0,
                    return_token_ids=True,
                ),
            ),
        )
        outputs = c.call(role, dict(op="trace_wait"), timeout=900)["result"]
        c.system_offload_rpc = False
        assert len(outputs) == 1 and outputs[0]["request_id"] == request.request_id
        out = outputs[0]
        meta = out["result"]["meta_info"]
        raw = out["result"]["text"]
        completion = tuple(int(x[1]) for x in meta["output_token_logprobs"])
        assert len(completion) == meta["completion_tokens"]
        assert self.tok.decode(completion, skip_special_tokens=True) == raw
        cached = meta["cached_tokens"]
        mode = request.cache_plan.mode
        if mode == "native_prefill":
            assert cached == 0
        if mode == "reuse":
            assert cached > 0
        if mode == "cachebridge":
            assert cached == self.req["H"]
        self.events.append(dict(role=role, prompt_ids=ids, output=out))
        return Generation(
            Message("assistant", raw),
            tuple(ids),
            completion,
            "Qwen3-shared-tokenizer",
            request.model_id,
            mode,
            reused_tokens=cached,
            handoff_prefix_ids=tuple(self.req["history_ids"]) if switching else (),
            mapper_id=self.mapper_id if mode == "cachebridge" else None,
            finish_reason=meta["finish_reason"]["type"],
            ttft_ms=out["ttft_ms"],
            completion_ms=out["completion_ms"],
        )

    def end_episode(self, trajectory_id):
        if self.req:
            self.source_after = digest(
                self.client, "source", self.req, trajectory_id + "-source-after"
            )
            assert self.source_after == self.source_before
            if self.transaction:
                self.target_after = digest(
                    self.client, "target", self.req, trajectory_id + "-target-after"
                )
                assert self.target_after == self.target_before
        self.client.reset("source", trajectory_id + "-release-source")
        self.client.reset("target", trajectory_id + "-release-target")
