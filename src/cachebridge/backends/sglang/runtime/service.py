"""Resident service that keeps streaming decode live while accepting mapper RPCs."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

from cachebridge.backends.sglang.runtime import config as _config


from cachebridge.backends.sglang.runtime import engine_base as _base


def engine_args(role):
    from cachebridge.backends.sglang.runtime import config as _config

    _config.validate_model_pair()
    args = _base.engine_args(role)
    args.update(
        tp_size=_config.TARGET_TP,
        base_gpu_id=0 if role == "source" else _config.TARGET_TP,
        port=(_config.SOURCE_PORT if role == "source" else _config.TARGET_PORT),
        max_total_tokens=_config.POOL_TOKENS,
        max_running_requests=_config.MAX_RUNNING_REQUESTS,
        schedule_policy="fcfs",
    )
    # All cold evaluation arms use the same initialization/sampling RNG seed.
    args["random_seed"] = 20261031
    args.update(
        disable_overlap_schedule=False,
        disable_cuda_graph=False,
        cuda_graph_bs=[1, 2],
        cuda_graph_max_bs=2,
        disable_cuda_graph_padding=True,
        enable_mixed_chunk=False,
    )
    if role == "target":
        args.update(
            disable_cuda_graph=False,
            cuda_graph_bs=list(range(1, _config.MAX_RUNNING_REQUESTS + 1)),
            cuda_graph_max_bs=_config.MAX_RUNNING_REQUESTS,
            disable_cuda_graph_padding=True,
            enable_mixed_chunk=False,
        )
    return args


async def _trace_one(engine, request, trace_epoch_ns, progress):
    arrival_ns = trace_epoch_ns + int(request["arrival_offset_ms"] * 1e6)
    delay = (arrival_ns - time.perf_counter_ns()) / 1e9
    if delay > 0:
        await asyncio.sleep(delay)
    submitted_ns = time.perf_counter_ns()
    progress["submitted_ns"] = submitted_ns
    from sglang.srt.managers.io_struct import GenerateReqInput

    # Explicit rid selects a radix namespace without changing any input IDs.
    obj = GenerateReqInput(
        input_ids=request["ids"],
        sampling_params=request["sampling"],
        rid=(
            ("cachebridge:" + request["cache_namespace"] + "::" + request["request_id"])
            if "cache_namespace" in request
            else request["request_id"]
        ),
        stream=True,
        return_logprob=request.get("return_token_ids", False),
        logprob_start_len=-1,
    )
    stream = engine.tokenizer_manager.generate_request(obj, None)
    first_token_ns = None
    final = None
    chunks = 0
    chunk_progress = [] if request.get("capture_chunk_progress", False) else None
    async for value in stream:
        observed_ns = time.perf_counter_ns()
        if first_token_ns is None:
            first_token_ns = observed_ns
            progress["first_token_ns"] = first_token_ns
        final = value
        if chunk_progress is not None:
            chunk_progress.append(
                dict(
                    host_observed_ns=observed_ns,
                    completion_tokens=value.get("meta_info", {}).get(
                        "completion_tokens"
                    ),
                )
            )
        chunks += 1
    finished_ns = time.perf_counter_ns()
    progress["finished_ns"] = finished_ns
    if first_token_ns is None or final is None:
        raise RuntimeError(f"empty trace stream for {request['request_id']}")
    return {
        "request_id": request["request_id"],
        "arrival_ns": arrival_ns,
        "submitted_ns": submitted_ns,
        "host_first_token_ns": first_token_ns,
        "host_finished_ns": finished_ns,
        "frontend_queue_ms": (submitted_ns - arrival_ns) / 1e6,
        "ttft_ms": (first_token_ns - arrival_ns) / 1e6,
        "completion_ms": (finished_ns - arrival_ns) / 1e6,
        "stream_chunks": chunks,
        "chunk_progress": chunk_progress,
        "result": final,
    }


# Total outcomes retained by an observer trace; Engine active requests stay 16.
MAX_TRACE_REQUESTS = 512


async def _serve(role, engine, pipe):
    needs_recovery = False
    trace = None
    while True:
        command = await asyncio.to_thread(pipe.recv)
        received_ns = time.perf_counter_ns()
        if command["op"] == "stop":
            if trace is not None and any(
                not task.done() for task in trace["tasks"].values()
            ):
                raise RuntimeError("cannot stop with live trace requests")
            pipe.send({"ok": True, "result": "stopping"})
            return
        try:
            op = command["op"]
            if op == "rpc":
                if trace is not None and trace["offload_control_rpc"]:
                    await asyncio.to_thread(
                        engine.collective_rpc, "audit_handoff", **command["parameters"]
                    )
                else:
                    engine.collective_rpc("audit_handoff", **command["parameters"])
                if command["parameters"]["action"] == "flush":
                    needs_recovery = False
                result = True
            elif op == "generate":
                if needs_recovery:
                    raise RuntimeError("reconcile failed handoff before generation")
                result = await engine.async_generate(
                    input_ids=command["ids"], sampling_params=command["sampling"]
                )
            elif op == "batch_stream":
                if needs_recovery or trace is not None or role != "target":
                    raise RuntimeError(
                        "batch stream requires recovered Target and no open trace"
                    )
                from cachebridge.backends.sglang.runtime.batch_stream import (
                    run as stream_batch,
                )

                result = await stream_batch(
                    engine,
                    command["requests"],
                    command["arrival_ns"],
                    time.perf_counter_ns,
                )
            elif op == "trace_open":
                if trace is not None:
                    raise RuntimeError("trace endpoint is single-owner")
                trace = {
                    "epoch_ns": int(command["trace_epoch_ns"]),
                    "capture_chunk_progress": bool(
                        command.get("capture_chunk_progress", False)
                    ),
                    "offload_control_rpc": bool(
                        command.get("offload_control_rpc", False)
                    ),
                    "tasks": {},
                    "progress": {},
                }
                result = {"trace_epoch_ns": trace["epoch_ns"]}
            elif op == "trace_submit":
                if trace is None:
                    raise RuntimeError("no open endpoint trace")
                request = dict(command["request"])
                request["capture_chunk_progress"] = trace["capture_chunk_progress"]
                request_id = request["request_id"]
                if (
                    request_id in trace["tasks"]
                    or len(trace["tasks"]) >= MAX_TRACE_REQUESTS
                ):
                    raise RuntimeError("duplicate or over-capacity trace request")
                progress = {
                    "submitted_ns": None,
                    "first_token_ns": None,
                    "finished_ns": None,
                }
                trace["progress"][request_id] = progress
                trace["tasks"][request_id] = asyncio.create_task(
                    _trace_one(engine, request, trace["epoch_ns"], progress)
                )
                result = {"request_id": request_id, "submitted_to_frontend": True}
            elif op == "trace_status":
                if trace is None:
                    raise RuntimeError("no open endpoint trace")
                result = {
                    "request_ids": list(trace["tasks"]),
                    "done": {
                        request_id: task.done()
                        for request_id, task in trace["tasks"].items()
                    },
                    "progress": trace["progress"],
                }
            elif op == "trace_wait":
                if trace is None:
                    raise RuntimeError("no open endpoint trace")
                request_ids = list(trace["tasks"])
                values = await asyncio.gather(
                    *(trace["tasks"][request_id] for request_id in request_ids)
                )
                result = values
                trace = None
            else:
                raise ValueError(op)
            pipe.send(
                {
                    "ok": True,
                    "result": result,
                    "service_received_ns": received_ns,
                    "service_finished_ns": time.perf_counter_ns(),
                }
            )
        except BaseException:
            if command["op"] == "rpc":
                needs_recovery = True
            pipe.send({"ok": False, "error": traceback.format_exc()})


def service(role, root, pipe, socket_root):
    os.environ["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "1"
    # Both tokenizer and worker scheduler see this before Engine construction.
    os.environ["SGLANG_ENABLE_COLOCATED_BATCH_GEN"] = (
        "true" if role == "target" else "false"
    )
    os.environ.update(
        CACHEBRIDGE_SERVING_HANDOFF_AUDIT="1",
        CACHEBRIDGE_SERVING_HANDOFF_ROOT=str(root),
        CACHEBRIDGE_SERVING_HANDOFF_ROLE=role,
        CACHEBRIDGE_BOUNDED_ENABLED="1",
        CACHEBRIDGE_BOUNDED_SOCKET_ROOT=str(socket_root),
        CACHEBRIDGE_LOCAL_HEAD_PREP="1",
        CACHEBRIDGE_LIVE_STAGE_ENABLED="1",
        R013_TARGET_TP=str(_config.TARGET_TP),
        R010_RUNTIME_CONFIG_JSON=json.dumps(_config.runtime_config()),
    )
    from sglang import Engine

    engine = None
    loop = asyncio.new_event_loop()
    try:
        args = engine_args(role)
        Path(root).mkdir(parents=True, exist_ok=False)
        (Path(root) / f"{role}-engine.json").write_text(
            json.dumps(args, indent=2) + "\n"
        )
        (Path(root) / f"{role}-admission-config.json").write_text(
            json.dumps(
                dict(
                    role=role,
                    enabled=role == "target",
                    flag=os.environ["SGLANG_ENABLE_COLOCATED_BATCH_GEN"],
                    set_before_Engine_construction=True,
                    prior_engine_functionality=True,
                ),
                indent=2,
            )
            + "\n"
        )
        engine = Engine(**args)
        asyncio.set_event_loop(loop)
        pipe.send({"ready": True, "pid": os.getpid(), "role": role})
        loop.run_until_complete(_serve(role, engine, pipe))
    except BaseException:
        try:
            pipe.send({"ok": False, "error": traceback.format_exc()})
        except BaseException:
            pass
        raise
    finally:
        if engine is not None:
            engine.shutdown()
        loop.close()
