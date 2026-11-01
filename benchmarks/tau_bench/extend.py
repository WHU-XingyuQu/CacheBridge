"""Extend budget-stopped native TRAIN trajectories with exact prefix checks."""

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

from client import Client, write
from cachebridge.backends.sglang.runtime import config as _config
from continuation_sampling import collect
from cachebridge.artifacts import file_sha256
from cachebridge.offline.cli import prediction_positions


def load_references(path):
    import torch

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if (
            row.get("split") != "train"
            or row.get("task_answers_read", True)
            or row.get("status") != "calibration_input_complete"
            or row["task_id"] in seen
        ):
            raise ValueError(
                "Expected unique usable native TRAIN inputs, without task-answer fitting"
            )
        seen.add(row["task_id"])
        if row["H"] != len(row["history_ids"]) or row["Q"] != len(row["query_ids"]):
            raise ValueError("H/Q token geometry mismatch")
        record = Path(row.get("record_path", f"case_{row['task_id']}.json"))
        if not record.is_absolute():
            record = (path.parent / record).resolve()
        if not record.is_file():
            raise ValueError(f"Missing native trajectory record: {record}")
        row["record_path"] = str(record)
        pairs = {(f["role"], f["rank"]) for f in row["files"]}
        if len(row["files"]) != 8 or pairs != {
            (role, rank) for role in ("source", "target") for rank in range(4)
        }:
            raise ValueError("Expected exactly four native KV shards per model")
        for shard in row["files"]:
            source = Path(shard["path"])
            if not source.is_absolute():
                source = (path.parent / source).resolve()
            if not shard.get("sha256") or file_sha256(source) != shard["sha256"]:
                raise ValueError(f"Missing or changed reference SHA256: {source}")
            tensor = torch.load(
                source, weights_only=True, map_location="cpu", mmap=True
            )
            expected = (
                40 if shard["role"] == "source" else 64,
                2,
                row["H"] if shard["role"] == "source" else row["H"] + row["Q"],
                2,
                128,
            )
            if tensor.shape != expected or tensor.dtype != torch.bfloat16:
                raise ValueError(f"Invalid native shard geometry/dtype: {source}")
            del tensor
            shard["path"] = str(source)
    if not rows:
        raise ValueError("Empty reference bank")
    return rows


def check_prefixes(record, reference):
    import torch

    checks = []
    for shard in record["files"]:
        old = next(
            f
            for f in reference["files"]
            if (f["role"], f["rank"]) == (shard["role"], shard["rank"])
        )
        current = torch.load(
            shard["path"], weights_only=True, map_location="cpu", mmap=True
        )
        previous = torch.load(
            old["path"], weights_only=True, map_location="cpu", mmap=True
        )
        equal = current.dtype == previous.dtype and torch.equal(
            current[:, :, : previous.shape[2]], previous
        )
        checks.append(
            dict(role=shard["role"], rank=shard["rank"], old_prefix_bitwise_equal=equal)
        )
        del current, previous
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-query-tokens", type=int, default=4096)
    parser.add_argument("--max-target-calls", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.max_query_tokens <= 4096 or not 1 <= args.max_target_calls <= 24:
        parser.error(
            "This recipe supports query limits1..4096 and target-call limits1..24"
        )
    inputs = args.inputs.resolve()
    rows = load_references(inputs)
    eligible = [
        row
        for row in rows
        if not row["terminal"]
        and row["stop_reason"]
        in ("calibration_query_budget", "calibration_target_call_budget")
    ]
    if any(
        args.max_query_tokens < row["Q"]
        or args.max_target_calls < row["native_target_calls"]
        for row in eligible
    ):
        parser.error(
            "Continuation limits must retain every selected reference prefix and call"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import torch

    torch.set_num_threads(4)
    limits = dict(
        max_query_tokens=args.max_query_tokens, max_target_calls=args.max_target_calls
    )
    state = dict(
        status="starting",
        pid=os.getpid(),
        started=time.time(),
        completed=[],
        reference_inputs_sha256=file_sha256(inputs),
        continuation_limits=limits,
        reference_cases=len(rows),
        extended_cases=len(eligible),
        reward_filter=False,
    )
    ledger = next(
        (
            inputs.parent / name
            for name in ("attempts.jsonl", "reference_attempts.jsonl")
            if (inputs.parent / name).exists()
        ),
        inputs.parent / "attempts.jsonl",
    )
    if ledger.exists():
        shutil.copy2(ledger, output / "reference_attempts.jsonl")
        state["reference_attempt_ledger_sha256"] = file_sha256(ledger)
    else:
        state["reference_attempt_ledger_sha256"] = None
    if not eligible:
        (output / "inputs.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        state.update(
            status="complete_no_extension_needed",
            unchanged_roster=True,
            task_quality_claim=False,
            finished=time.time(),
        )
        write(output / "status.json", state)
        return
    client = Client(
        output / "runtime", Path(tempfile.mkdtemp(prefix="cb-extend-")), output
    )
    extended = {}
    try:
        write(output / "occupancy_before.json", client.assert_isolation())
        write(output / "ready.json", client.start())
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            _config.SOURCE_MODEL, local_files_only=True
        )
        client.tokenizer = tokenizer
        for reference in eligible:
            task = reference["task_id"]
            state.update(status="sampling", current_task=task)
            write(output / "status.json", state)
            record = collect(client, tokenizer, task, limits, reference)
            assert record["status"] == "calibration_input_complete"
            assert record["history_ids"] == reference["history_ids"]
            assert record["query_ids"][: reference["Q"]] == reference["query_ids"]
            checks = check_prefixes(record, reference)
            write(
                output / f"post_extension_{task}.json",
                dict(
                    status=(
                        "passed"
                        if all(c["old_prefix_bitwise_equal"] for c in checks)
                        else "failed_preserved"
                    ),
                    checks=checks,
                ),
            )
            assert all(c["old_prefix_bitwise_equal"] for c in checks)
            old_mask = set(
                prediction_positions(tokenizer, reference)[
                    "assistant_prediction_positions"
                ]
            )
            new_mask = set(
                prediction_positions(tokenizer, record)[
                    "assistant_prediction_positions"
                ]
            )
            prefix_mask = {i for i in new_mask if i < reference["Q"]}
            assert old_mask <= prefix_mask and prefix_mask - old_mask <= {
                reference["Q"] - 1
            }
            for shard in record["files"]:
                shard["sha256"] = file_sha256(shard["path"])
            record["record_path"] = str(output / f"case_{task}.json")
            write(Path(record["record_path"]), record)
            extended[task] = record
            state["completed"].append(task)
        omitted = {
            "events",
            "user_events",
            "user_messages",
            "source_status",
            "native_query_observation",
        }
        merged = [
            {
                k: v
                for k, v in extended.get(row["task_id"], row).items()
                if k not in omitted
            }
            for row in rows
        ]
        (output / "inputs.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in merged)
        )
        state.update(
            status="complete_pending_fitting",
            unchanged_roster=True,
            task_quality_claim=False,
            new_inputs_sha256=file_sha256(output / "inputs.jsonl"),
        )
    except BaseException:
        state.update(status="failed_preserved", error=traceback.format_exc())
        raise
    finally:
        client.stop()
        state["finished"] = time.time()
        write(output / "status.json", state)


if __name__ == "__main__":
    main()
