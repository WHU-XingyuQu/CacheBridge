"""Native TRAIN sampling or complete TEST episodes through sample_trajectory."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import traceback

from client import Client, write
from cachebridge.backends.sglang.runtime import config as _config
from episode import episode
from sampling_helpers import collect
from cachebridge.artifacts import file_sha256
from cachebridge.offline.cli import prediction_positions
from cachebridge.offline.trajectory import HandoffSample
from cachebridge.trajectory import TaskRef


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["sample", "evaluate"])
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="CacheBridge")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--functional-only", action="store_true")
    parser.add_argument("--max-query-tokens", type=int)
    parser.add_argument("--max-target-calls", type=int)
    args = parser.parse_args()
    if args.mode != "sample" and any(
        value is not None for value in (args.max_query_tokens, args.max_target_calls)
    ):
        parser.error("Continuation limits apply only to sample mode")
    limits = dict(
        max_query_tokens=(
            2048 if args.max_query_tokens is None else args.max_query_tokens
        ),
        max_target_calls=6 if args.max_target_calls is None else args.max_target_calls,
    )
    if (
        not 1 <= limits["max_query_tokens"] <= 4096
        or not 1 <= limits["max_target_calls"] <= 24
    ):
        parser.error(
            "This recipe supports query limits1..4096 and target-call limits1..24"
        )
    tasks = json.loads(args.tasks.read_text())
    if args.mode == "sample":
        if (
            not isinstance(tasks, list)
            or not tasks
            or any(type(task) is not int or task < 0 for task in tasks)
            or len(set(tasks)) != len(tasks)
        ):
            raise ValueError("Sampling requires a unique list of retail TRAIN task IDs")
    else:
        if not isinstance(tasks, dict) or not set(tasks) <= {"retail", "airline"}:
            raise ValueError("Evaluation requires a domain-to-task-IDs object")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    descriptor = _config.artifact_contract()["cachebridge"]
    os.environ["CB_AGENT_MAPPER_ID"] = descriptor["mapper_id"]
    client = Client(
        output / "runtime", Path(tempfile.mkdtemp(prefix="cachebridge-")), output
    )
    state = dict(status="starting", mode=args.mode, completed=[], started=time.time())
    try:
        write(output / "occupancy_before.json", client.assert_isolation())
        write(output / "ready.json", client.start())
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(_config.SOURCE_MODEL, local_files_only=True)
        client.tokenizer = tok
        if args.mode == "sample":
            rows = []
            for task in tasks:
                record = collect(client, tok, task, limits)
                for f in record["files"]:
                    f["sha256"] = file_sha256(f["path"])
                write(output / f"case_{task}.json", record)
                rows.append(record)
                state.update(status="sampling", completed=[r["task_id"] for r in rows])
                write(output / "status.json", state)
            fields = (
                "events",
                "user_events",
                "user_messages",
                "source_status",
                "native_query_observation",
            )
            for name, chosen in [
                ("attempts.jsonl", rows),
                (
                    "inputs.jsonl",
                    [r for r in rows if r["status"] == "calibration_input_complete"],
                ),
            ]:
                (output / name).write_text(
                    "".join(
                        json.dumps({k: v for k, v in r.items() if k not in fields})
                        + "\n"
                        for r in chosen
                    )
                )
            samples = []
            for r in rows:
                if r["status"] != "calibration_input_complete":
                    continue
                positions = prediction_positions(tok, r)[
                    "assistant_prediction_positions"
                ]
                sample = HandoffSample(
                    TaskRef("tau-bench", "retail", "train", str(r["task_id"])),
                    "native-train-" + str(r["task_id"]),
                    "Qwen3-14B",
                    "Qwen3-32B",
                    "Qwen3-shared-tokenizer",
                    tuple(r["history_ids"]),
                    tuple(r["query_ids"]),
                    tuple(positions),
                    "native_target",
                    truncated=bool(r["calibration_query_truncated"]),
                )
                samples.append(sample.to_dict())
            (output / "handoff_samples.jsonl").write_text(
                "".join(json.dumps(s) + "\n" for s in samples)
            )
            state.update(
                attempted=len(rows),
                usable=sum(r["status"] == "calibration_input_complete" for r in rows),
                outcome_filter=False,
                continuation_limits=limits,
            )
        else:
            arm = "Native" if args.native else args.arm
            if not args.native:
                gate = episode(
                    client, tok, "retail", 157, arm, split="train", functional=True
                )
                assert gate["mapped_handoffs"] == 1
            if not args.functional_only:
                for domain, ids in tasks.items():
                    for task in ids:
                        row = episode(client, tok, domain, task, arm)
                        state["completed"].append(row)
                        state["status"] = "evaluating"
                        write(output / "status.json", state)
            state["scope"] = (
                "Complete episodes, fixed greedy seed20261031 and router; no quality or timing inference from a functional-only run"
            )
        state["status"] = "complete_pending_review"
    except BaseException:
        state.update(status="failed_preserved", error=traceback.format_exc())
        raise
    finally:
        client.stop()
        state["finished"] = time.time()
        write(output / "status.json", state)


if __name__ == "__main__":
    main()
