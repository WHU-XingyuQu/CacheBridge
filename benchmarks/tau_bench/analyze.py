"""Replay official task rewards and compare three complete, paired task arms on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

from cachebridge.trajectory import Trajectory
from live_adapter import domain_adapter
from paired_analysis import analyze


class RecordedUser:
    def __init__(self, events):
        self.values = [event["output"]["result"]["text"] for event in events]
        self.cursor = 0

    def take(self):
        value = self.values[self.cursor]
        self.cursor += 1
        return value

    def reset(self, instruction=None):
        return self.take()

    def step(self, content):
        return self.take()

    def get_total_cost(self):
        return 0.0


def review(path, domain, task, adapter):
    raw = path.read_bytes()
    record = json.loads(raw)
    trajectory = Trajectory.from_dict(record["trajectory"])
    row = record["row"]
    if (
        trajectory.task.split != "test"
        or trajectory.task.domain != domain
        or int(trajectory.task.task_id) != task
    ):
        raise ValueError("Wrong task/split in episode")
    if row["functional"]:
        raise ValueError("Functional probes are not formal task outcomes")
    env = adapter.new_env("test", task)
    env.user = RecordedUser(record["user_events"])
    if env.reset(task_index=task).observation != record["initial_user"]:
        raise ValueError("Initial user observation differs")
    done = False
    for event in record["events"]:
        action, count = adapter.parse_action(event["response"])
        if (
            action.model_dump() != event["action"]
            or count != event["tool_calls_emitted"]
        ):
            raise ValueError("Recorded tool action differs from generated text")
        response = env.step(action)
        done = response.done
        if response.observation != event["observation"]:
            raise ValueError("Official tool observation differs")
    if env.user.cursor != len(env.user.values):
        raise ValueError("Unused user responses in episode")
    official = adapter.official_score(env, done and trajectory.status == "terminal")
    if official != row["quality"] or int(official["reward"] == 1) != row["success"]:
        raise ValueError("Official reward replay differs")
    if len(record["backend_events"]) != len(trajectory.turns):
        raise ValueError("Generation receipt count differs")
    for turn, event in zip(trajectory.turns, record["backend_events"], strict=True):
        meta = event["output"]["result"]["meta_info"]
        generation = turn.generation
        if (
            tuple(event["prompt_ids"]) != generation.prompt_ids
            or tuple(int(x[1]) for x in meta["output_token_logprobs"])
            != generation.completion_ids
        ):
            raise ValueError("Recorded token IDs differ from actual backend outputs")
        if (
            meta["cached_tokens"] != generation.reused_tokens
            or event["output"]["result"]["text"] != generation.response.content
        ):
            raise ValueError("Backend cache/text receipt differs")
    if record["request"] and record["source_before"] != record["source_after"]:
        raise ValueError("Source KV changed during handoff")
    if record["transaction"]:
        if record["target_before"] != record["target_after"]:
            raise ValueError("Installed target KV changed before query")
        receipts = record["transaction"]["readiness"][-1]
        if len(receipts) != 4 or not all(
            x["outcome"] == "published"
            and x["worker_done"]
            and x["worker_thread_exited"]
            and x["all_rank_ready"]
            and x["all_rank_identity_valid"]
            for x in receipts
        ):
            raise ValueError("Incomplete target publication")
        handoff = next(
            turn
            for turn in trajectory.turns
            if turn.generation.executed_mode == "cachebridge"
        )
        if handoff.generation.reused_tokens != record["request"]["H"]:
            raise ValueError("Mapped prefix was not fully reused")
    return row, trajectory.seed, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("native", "baseline", "candidate", "tasks", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous analysis; choose a fresh output path")
    roster = json.loads(args.tasks.read_text())
    if not roster or not set(roster) <= {"retail", "airline"}:
        raise ValueError("Supply a domain-to-TEST-task-ID roster")
    if any(not ids or len(ids) != len(set(ids)) for ids in roster.values()):
        raise ValueError("Task rosters must be nonempty and unique")
    rows, arms, hashes, seeds = [], [], {}, {}
    for root in (args.native, args.baseline, args.candidate):
        names = set()
        for domain, tasks in roster.items():
            adapter = domain_adapter(domain)
            for task in tasks:
                path = root / f"test-{domain}-{task}.json"
                row, seed, digest = review(path, domain, task, adapter)
                rows.append(row)
                names.add(row["arm"])
                hashes[str(path.resolve())] = digest
                seeds.setdefault((domain, task), set()).add(seed)
        if len(names) != 1:
            raise ValueError("Each input directory must contain exactly one arm")
        arms.append(names.pop())
    if len(set(arms)) != 3 or any(len(values) != 1 for values in seeds.values()):
        raise ValueError("Require three distinct arms with paired episode seeds")
    index = {}
    for row in rows:
        index.setdefault((row["domain"], row["task_id"]), []).append(row)
    for group in index.values():
        if len({r["source_H_sha256"] for r in group}) != 1 or any(
            r["source_KV_digest"] != group[0]["source_KV_digest"] for r in group
        ):
            raise ValueError("Source handoff tokens/KV differ across arms")
    result = analyze(rows, dict(evaluation=dict(task_ids=roster, arms=arms)))
    result.update(
        status="passed_official_replay_and_paired_analysis",
        episodes=len(rows),
        raw_sha256=hashes,
        official_tool_and_reward_replay=True,
        comparison="candidate minus baseline; Native supplies matched retention denominator",
        scope="One-seed exposed development comparison. Caller must freeze model/runtime/router/budgets; no automatic adoption or latency claim.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                status=result["status"],
                episodes=len(rows),
                pooled=result["results"]["pooled"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
