"""Statistics for two complete, source-paired fixed-router task panels."""

from collections import Counter
import math
import random

from paired_analysis import interval

SOURCE_FIELDS = (
    "source_intake_sha256",
    "source_H_sha256",
    "source_KV_digest",
    "handoff_count",
)


def validate_roster(roster):
    if not isinstance(roster, dict) or not roster:
        raise ValueError("Supply a nonempty domain-to-task-ID roster")
    for domain, tasks in roster.items():
        if (
            not isinstance(domain, str)
            or not domain
            or not isinstance(tasks, list)
            or not tasks
        ):
            raise ValueError("Each domain must have a nonempty task list")
        if any(type(task) is not int or task < 0 for task in tasks):
            raise ValueError("Task IDs must be nonnegative integers")
        if len(tasks) != len(set(tasks)):
            raise ValueError("Duplicate task IDs in roster")


def analyze(rows, protocol, *, bootstrap=10000, bootstrap_seed=20261031):
    """Consume officially reviewed rows; this function does not score episodes."""
    roster = protocol["evaluation"]["task_ids"]
    arms = protocol["evaluation"]["arms"]
    validate_roster(roster)
    if (
        len(arms) != 2
        or any(not isinstance(arm, str) or not arm for arm in arms)
        or len(set(arms)) != 2
    ):
        raise ValueError("Require distinct Native and candidate arms")
    if type(bootstrap) is not int or bootstrap < 1:
        raise ValueError("Bootstrap count must be positive")
    native_arm, candidate_arm = arms
    expected = {
        (domain, task, arm)
        for domain, tasks in roster.items()
        for task in tasks
        for arm in arms
    }
    index = {}
    for row in rows:
        key = (row["domain"], row["task_id"], row["arm"])
        if key in index or key not in expected:
            raise ValueError("Duplicate or unplanned task result")
        if row["success"] not in (0, 1) or row["status"] not in (
            "terminal",
            "step_limit",
            "generation_limit",
            "error",
        ):
            raise ValueError("Invalid task outcome")
        if row["success"] and row["status"] != "terminal":
            raise ValueError("Unfinished tasks cannot count as successes")
        for field in ("handoff_count", "mapped_handoffs", "native_fallbacks"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError("Invalid execution count")
        if row["handoff_count"] > 1 or row["native_fallbacks"]:
            raise ValueError(
                "This forced-mapping recipe requires at most one handoff and zero fallback"
            )
        if row["mapped_handoffs"] != (
            0 if row["arm"] == native_arm else row["handoff_count"]
        ):
            raise ValueError("Incomplete or wrongly labelled mapped handoff")
        digest = row["source_intake_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Missing Source intake fingerprint")
        index[key] = row
    if set(index) != expected:
        raise ValueError(f"Incomplete panel: {len(index)}/{len(expected)} records")
    groups = {}
    for domain, tasks in roster.items():
        groups[domain] = []
        for task in tasks:
            pair = [index[domain, task, arm] for arm in arms]
            if any(pair[0][field] != pair[1][field] for field in SOURCE_FIELDS):
                raise ValueError(
                    "Source history, KV, intake or handoff differs across arms"
                )
            groups[domain].append(tuple(row["success"] for row in pair))
    rng = random.Random(bootstrap_seed)
    result = {}
    for label in [*groups, "pooled"]:
        selected = list(groups.values()) if label == "pooled" else [groups[label]]
        pairs = [pair for group in selected for pair in group]
        n = len(pairs)
        native, mapped = (sum(pair[i] for pair in pairs) for i in (0, 1))
        gains = sum(pair == (0, 1) for pair in pairs)
        losses = sum(pair == (1, 0) for pair in pairs)
        differences, ratios, zeros = [], [], 0
        for _ in range(bootstrap):
            draw = [rng.choice(group) for group in selected for _ in range(len(group))]
            a, b = (sum(pair[i] for pair in draw) for i in (0, 1))
            differences.append(100 * (b - a) / n)
            if a:
                ratios.append(b / a)
            else:
                zeros += 1
        discordant = gains + losses
        exact_p = (
            min(
                1.0,
                2
                * sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1))
                / 2**discordant,
            )
            if discordant
            else 1.0
        )
        relevant = [row for row in rows if label == "pooled" or row["domain"] == label]
        execution = {}
        for arm in arms:
            values = [row for row in relevant if row["arm"] == arm]
            execution[arm] = {
                field: sum(row.get(field, 0) for row in values)
                for field in (
                    "handoff_count",
                    "mapped_handoffs",
                    "native_fallbacks",
                    "generation_limit_hits",
                    "malformed_tool_call_generations",
                    "executed_tool_calls",
                    "assistant_calls",
                    "user_calls",
                )
            }
            execution[arm]["statuses"] = dict(Counter(row["status"] for row in values))
            execution[arm]["successes_with_handoff"] = sum(
                row["success"] for row in values if row["handoff_count"]
            )
        result[label] = dict(
            N=n,
            successes=dict(zip(arms, [native, mapped])),
            mapped_minus_native_pp=100 * (mapped - native) / n,
            paired_95CI_pp=interval(differences),
            retention=mapped / native if native else None,
            retention_95CI=interval(ratios) if not zeros else None,
            zero_native_draws=zeros,
            mapped_only_success=gains,
            native_only_success=losses,
            exact_discordance_two_sided_p=exact_p,
            execution=execution,
        )
    return dict(
        status="complete_paired_analysis_requires_raw_episode_review",
        results=result,
        bootstrap=bootstrap,
        bootstrap_seed=bootstrap_seed,
        pooled="Paired task bootstrap stratified by domain; fixed task-count weights",
        scope="Complete fixed-router two-arm comparison. Freeze tasks, models, runtime and budgets before evaluation. Intervals do not establish equivalence or noninferiority. Exact discordance p-values are supplementary and unadjusted.",
    )
