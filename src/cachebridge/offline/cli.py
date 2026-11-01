"""Prepare and execute the complete Qwen3 14B-to-32B TRAIN-only recipe."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import io
from .recipes import get_recipe
from .support import load_support
from ..architectures.pairs import get_pair


from .traces.positions import prediction_positions


def prepare(args):
    from transformers import AutoTokenizer
    import torch

    recipe = get_recipe(args.recipe)
    pair = get_pair(recipe.pair)
    io.validate_target_model(args.target_model, recipe=recipe.name)

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    records = [json.loads(s) for s in args.inputs.read_text().splitlines() if s.strip()]
    if len(records) < 16:
        raise ValueError(
            "The complete recipe requires at least16 usable native TRAIN handoffs"
        )
    seen = set()
    files = {}
    for r in records:
        if r["split"] != "train" or r.get("task_answers_read", True):
            raise ValueError(
                "Only TRAIN records with task_answers_read=false are eligible"
            )
        if r["status"] != "calibration_input_complete" or r["task_id"] in seen:
            raise ValueError("Incomplete or duplicate handoff input")
        seen.add(r["task_id"])
        assert r["H"] == len(r["history_ids"]) and r["Q"] == len(r["query_ids"])
        pairs = [(f["role"], f["rank"]) for f in r["files"]]
        if len(pairs) != 8 or set(pairs) != {
            (role, rank) for role in ("source", "target") for rank in range(4)
        }:
            raise ValueError(
                "Expected exactly four distinct source and target KV shards"
            )
        for f in r["files"]:
            path = Path(f["path"]).expanduser()
            if not path.is_absolute():
                path = (args.inputs.resolve().parent / path).resolve()
            digest = io.sha(path)
            if f.get("sha256") and f["sha256"] != digest:
                raise ValueError(f"Trace hash mismatch: {path}")
            tensor = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            expected = (
                40 if f["role"] == "source" else 64,
                2,
                r["H"] if f["role"] == "source" else r["H"] + r["Q"],
                2,
                128,
            )
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != expected
                or tensor.dtype != torch.bfloat16
            ):
                raise ValueError(f"Invalid native KV geometry/dtype: {path}")
            del tensor
            f.update(path=str(path), sha256=digest)
            files[str(path)] = digest
    if args.input_order:
        input_order = json.loads(args.input_order.read_text())
        if len(input_order) != len(records) or set(input_order) != seen:
            raise ValueError(
                "Input order must cover every usable TRAIN case exactly once"
            )
        indexed = {r["task_id"]: r for r in records}
        records = [indexed[task] for task in input_order]
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    positions = [prediction_positions(tokenizer, r) for r in records]
    (root / "inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    io.write(
        root / "input_review.json",
        dict(status="passed", usable=len(records), files=files, TRAIN_only=True),
    )
    io.write(
        root / "positions.json", dict(status="passed", N=len(records), cases=positions)
    )
    initializer = args.initializer.resolve()
    weights = list(initializer.glob("layer_*.safetensors"))
    load_support(
        initializer,
        source_layers=pair.source_layers,
        target_layers=pair.target_layers,
        support_size=8,
    )
    order = (
        json.loads(args.k_order.read_text())
        if args.k_order
        else [r["task_id"] for r in records]
    )
    if not args.freeze_keys and (set(order) != seen or len(order) != len(seen)):
        raise ValueError(
            "K allocation order must cover the same TRAIN tasks exactly once"
        )
    if not args.freeze_keys:
        for r in records:
            n = 16384 // len(records) + (
                order.index(r["task_id"]) < 16384 % len(records)
            )
            if r["H"] < n or 151645 not in r["history_ids"]:
                raise ValueError(
                    "Insufficient history for fixed K budget, or missing Qwen ChatML system boundary"
                )
    key_source = root / "key_run"
    if args.freeze_keys:
        key_source.mkdir()
        io.write(
            key_source / "complete.json",
            dict(
                status="passed",
                frozen_external_initializer=str(initializer),
                key_refit=False,
                coefficient_hashes={p.name: io.sha(p) for p in weights},
            ),
        )
    protocol = dict(
        recipe=recipe.name,
        model_pair=recipe.pair,
        target_model=str(args.target_model.resolve()),
        source_layers=40,
        expected_cases=len(records),
        inputs="inputs.jsonl",
        input_review="input_review.json",
        position_manifest="positions.json",
        base_initializer=str(initializer),
        current_mapper=str(initializer),
        key_source="key_run",
        initializer=(
            str(initializer) if args.freeze_keys else "key_run/mappers/initializer"
        ),
        K_observations=16384,
        K_ridge=1.0,
        K_allocation_order=order,
        V_mean_diag_fraction=0.01,
        prediction_positions=sum(
            len(r["assistant_prediction_positions"]) for r in positions
        ),
        no_gradient=True,
        keys_frozen=args.freeze_keys,
        files={
            str(p): io.sha(p)
            for p in [
                *weights,
                root / "inputs.jsonl",
                root / "positions.json",
                root / "input_review.json",
            ]
        },
    )
    io.write(root / "fit_protocol.json", protocol)
    print(root / "fit_protocol.json")


def run(args):
    root = args.directory.resolve()
    p = io.load_protocol(root / "fit_protocol.json")
    environment = dict(os.environ, CB_AGENT_FIT_DIR=str(root))
    for path, digest in p["files"].items():
        if io.sha(path) != digest:
            raise ValueError(f"Changed frozen input: {path}")
    for module, receipt in [
        ("keys", "key_run/complete.json"),
        ("calibration.sequential", "fit_run/complete.json"),
    ]:
        if (root / receipt).exists():
            if json.loads((root / receipt).read_text())["status"] != "passed":
                raise RuntimeError("Previous step did not pass")
            if module == "calibration.sequential":
                directory = root / "fit_run/mappers/TauBalancedReadoutV"
                manifest = json.loads((directory / "manifest.json").read_text())
                if manifest["fit_protocol_sha256"] != io.sha(
                    root / "fit_protocol.json"
                ):
                    raise ValueError("Completed mapper belongs to another protocol")
                for name, digest in manifest["files"].items():
                    if io.sha(directory / name) != digest:
                        raise ValueError("Completed mapper changed")
            if module == "keys" and not p.get("keys_frozen", False):
                receipt_path = root / "key_run/coefficient_hashes.json"
                if not receipt_path.exists():
                    raise ValueError(
                        "Missing key-stage integrity receipt; inspect previous attempt"
                    )
                for name, digest in json.loads(receipt_path.read_text()).items():
                    if io.sha(root / "key_run/mappers/initializer" / name) != digest:
                        raise ValueError("Completed K initializer changed")
            continue
        subprocess.run(
            [sys.executable, "-m", "cachebridge.offline." + module],
            env=environment,
            check=True,
        )
        if module == "keys":
            io.write(
                root / "key_run/coefficient_hashes.json",
                {
                    f.name: io.sha(f)
                    for f in sorted(
                        (root / "key_run/mappers/initializer").glob(
                            "layer_*.safetensors"
                        )
                    )
                },
            )
    from .export import export_mapper

    export_mapper(root, args.mapper_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("inputs", "target-model", "initializer", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--recipe", default="qwen3_14b_32b_trajectory")
    p.add_argument("--freeze-keys", action="store_true")
    p.add_argument("--k-order", type=Path)
    p.add_argument("--input-order", type=Path)
    p = sub.add_parser("run")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--mapper-id", required=True)
    args = parser.parse_args(argv)
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
