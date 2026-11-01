"""Paired complete-task analysis; requires all predeclared evaluation rows.

Expected JSONL fields: domain, task_id, arm, success (0/1), status,
source_intake_sha256, handoff_count, mapped_handoffs, native_fallbacks.
This consumes official-scored episodes; it is not itself a task scorer.
"""

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import random


def interval(values):
    values = sorted(values)

    def quantile(p):
        x = (len(values) - 1) * p
        i = int(x)
        fraction = x - i
        return (
            values[i] * (1 - fraction) + values[min(i + 1, len(values) - 1)] * fraction
        )

    return [quantile(0.025), quantile(0.975)]


def analyze(rows, protocol, *, bootstrap=10000):
    roster = protocol["evaluation"]["task_ids"]
    arms = protocol["evaluation"]["arms"]
    native, baseline, expanded = arms
    expected = {
        (domain, int(task), arm)
        for domain, tasks in roster.items()
        for task in tasks
        for arm in arms
    }
    index = {}
    for row in rows:
        key = (row["domain"], int(row["task_id"]), row["arm"])
        if key in index or key not in expected:
            raise ValueError(f"duplicate/unplanned result: {key}")
        if row["success"] not in (0, 1) or row["status"] not in (
            "terminal",
            "step_limit",
            "generation_limit",
            "error",
        ):
            raise ValueError("invalid task outcome")
        if row["status"] != "terminal" and row["success"]:
            raise ValueError("unfinished task cannot score a success")
        for field in ("handoff_count", "mapped_handoffs", "native_fallbacks"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f"invalid {field}")
        if row["handoff_count"] > 1:
            raise ValueError(
                "pilot protocol permits one causal intake-to-resolution handoff"
            )
        if row["native_fallbacks"]:
            raise ValueError(
                "fallback in a forced-mapping quality arm; preserve and review before analysis"
            )
        if row["arm"] == native:
            if row["mapped_handoffs"]:
                raise ValueError("Native control contains mapping")
        elif row["mapped_handoffs"] != row["handoff_count"]:
            raise ValueError("incomplete mapped handoff receipt")
        digest = row["source_intake_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("missing source-intake fingerprint")
        index[key] = row
    if set(index) != expected:
        raise ValueError(
            f"incomplete results: {len(index)}/{len(expected)}; no partial quality claim"
        )
    groups = defaultdict(list)
    for domain, tasks in roster.items():
        for task in tasks:
            matched = [index[domain, int(task), arm] for arm in arms]
            if (
                len({r["source_intake_sha256"] for r in matched}) != 1
                or len({r["handoff_count"] for r in matched}) != 1
            ):
                raise ValueError("source intake or handoff differs across paired arms")
            groups[domain].append(tuple(r["success"] for r in matched))
    rng = random.Random(20261031)
    outputs = {}
    for label in [*groups, "pooled"]:
        selected = list(groups.values()) if label == "pooled" else [groups[label]]
        cases = [r for g in selected for r in g]
        counts = [sum(r[i] for r in cases) for i in range(3)]
        draws, ratios = [], [[], []]
        zero_native = 0
        for _ in range(bootstrap):
            draw = [rng.choice(g) for g in selected for _ in range(len(g))]
            totals = [sum(r[i] for r in draw) for i in range(3)]
            draws.append((totals[2] - totals[1]) / len(draw))
            if totals[0]:
                for i in (0, 1):
                    ratios[i].append(totals[i + 1] / totals[0])
            else:
                zero_native += 1
        outputs[label] = dict(
            N=len(cases),
            successes=dict(zip(arms, counts)),
            expanded_minus_baseline_success_pp=100
            * (counts[2] - counts[1])
            / len(cases),
            paired_95CI_pp=[100 * x for x in interval(draws)],
            retention={
                arm: (counts[i] / counts[0] if counts[0] else None)
                for i, arm in enumerate(arms[1:], 1)
            },
            retention_95CI={
                arm: (interval(ratios[i]) if not zero_native else None)
                for i, arm in enumerate(arms[1:])
            },
            zero_native_bootstrap_draws=zero_native,
            expanded_only_success=sum(r[2] == 1 and r[1] == 0 for r in cases),
            baseline_only_success=sum(r[1] == 1 and r[2] == 0 for r in cases),
        )
        relevant = [r for r in rows if label == "pooled" or r["domain"] == label]
        outputs[label]["execution"] = {}
        for arm in arms:
            values = [r for r in relevant if r["arm"] == arm]
            fields = (
                "handoff_count",
                "mapped_handoffs",
                "native_fallbacks",
                "generation_limit_hits",
                "malformed_tool_call_generations",
                "multiple_tool_call_outputs",
                "executed_tool_calls",
                "assistant_calls",
                "user_calls",
            )
            outputs[label]["execution"][arm] = {
                field: sum(r.get(field, 0) for r in values) for field in fields
            }
            outputs[label]["execution"][arm]["terminal_episodes"] = sum(
                r["status"] == "terminal" for r in values
            )
    return dict(
        status="complete_paired_analysis_requires_raw_episode_review",
        bootstrap=bootstrap,
        results=outputs,
        scope="One-seed exposed TEST pilot, fixed router and matched runtime; no automatic mapper adoption",
    )
