import copy
import importlib.util
from pathlib import Path
import sys

import pytest

BENCHMARK = Path(__file__).parents[1] / "benchmarks/tau_bench"
sys.path.insert(0, str(BENCHMARK))
spec = importlib.util.spec_from_file_location(
    "public_paired_compare", BENCHMARK / "paired_compare.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def panel():
    rows = []
    for arm, outcomes in [("Native", [1, 0, 0]), ("Mapped", [0, 1, 1])]:
        for task, outcome in enumerate(outcomes):
            rows.append(
                dict(
                    domain="fixture",
                    task_id=task,
                    arm=arm,
                    success=outcome,
                    status="terminal",
                    handoff_count=1,
                    mapped_handoffs=int(arm == "Mapped"),
                    native_fallbacks=0,
                    source_intake_sha256=str(task) * 64,
                    source_H_sha256="a" * 64,
                    source_KV_digest=["b" * 64],
                )
            )
    return rows, dict(
        evaluation=dict(task_ids={"fixture": [0, 1, 2]}, arms=["Native", "Mapped"])
    )


def test_complete_pair_counts_discordant_outcomes_and_mapping_receipts():
    rows, protocol = panel()
    result = module.analyze(rows, protocol, bootstrap=100)["results"]["pooled"]
    assert result["successes"] == {"Native": 1, "Mapped": 2}
    assert result["mapped_only_success"] == 2 and result["native_only_success"] == 1
    assert result["mapped_minus_native_pp"] == pytest.approx(100 / 3)
    assert result["execution"]["Mapped"]["mapped_handoffs"] == 3


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unplanned"])
def test_partial_or_duplicate_panels_are_rejected(mutation):
    rows, protocol = panel()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows[-1]["task_id"] = 100
    with pytest.raises(ValueError, match="Incomplete|Duplicate|unplanned"):
        module.analyze(rows, protocol, bootstrap=10)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_H_sha256", "c" * 64),
        ("source_KV_digest", ["c" * 64]),
        ("source_intake_sha256", "c" * 64),
    ],
)
def test_source_mismatch_is_rejected(field, value):
    rows, protocol = panel()
    rows[-1][field] = value
    with pytest.raises(ValueError, match="Source"):
        module.analyze(rows, protocol, bootstrap=10)


@pytest.mark.parametrize(
    "field,value",
    [("native_fallbacks", 1), ("mapped_handoffs", 0), ("status", "step_limit")],
)
def test_fallback_missing_mapping_or_unfinished_success_is_rejected(field, value):
    rows, protocol = panel()
    rows[-1][field] = value
    with pytest.raises(ValueError):
        module.analyze(rows, protocol, bootstrap=10)


def test_zero_native_denominator_is_explicit():
    rows, protocol = panel()
    for row in rows:
        if row["arm"] == "Native":
            row["success"] = 0
    result = module.analyze(rows, protocol, bootstrap=20)["results"]["pooled"]
    assert result["retention"] is None and result["retention_95CI"] is None
    assert result["zero_native_draws"] == 20
