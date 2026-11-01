"""Replay official rewards and compare one complete mapper arm with Native on CPU."""

import argparse
import json
from pathlib import Path

from analyze import review
from live_adapter import domain_adapter
from paired_compare import analyze, validate_roster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("native", "candidate", "tasks", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261031)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous analysis; choose a fresh output path")
    roster = json.loads(args.tasks.read_text())
    validate_roster(roster)
    if not set(roster) <= {"retail", "airline"}:
        raise ValueError("The supplied official adapters support Retail and Airline")
    if args.bootstrap < 1:
        raise ValueError("Bootstrap count must be positive")
    rows, arms, hashes, seeds = [], [], {}, set()
    for root in (args.native, args.candidate):
        names = set()
        for domain, tasks in roster.items():
            adapter = domain_adapter(domain)
            for task in tasks:
                path = root / f"test-{domain}-{task}.json"
                row, seed, digest = review(path, domain, task, adapter)
                rows.append(row)
                seeds.add(seed)
                names.add(row["arm"])
                hashes[str(path.resolve())] = digest
        if len(names) != 1:
            raise ValueError("Each directory must contain exactly one arm")
        arms.append(names.pop())
    if len(seeds) != 1:
        raise ValueError("This recipe requires one matched generation seed")
    result = analyze(
        rows,
        dict(evaluation=dict(task_ids=roster, arms=arms)),
        bootstrap=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    result.update(
        status="passed_official_replay_and_paired_analysis",
        episodes=len(rows),
        raw_sha256=hashes,
        official_tool_and_reward_replay=True,
        generation_seed=seeds.pop(),
        comparison=f"{arms[1]} minus {arms[0]}; same-router Native is the retention denominator",
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
