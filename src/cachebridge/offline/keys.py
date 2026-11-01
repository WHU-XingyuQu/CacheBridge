"""Matched fixed-budget K initialization using the frozen original solver."""

import gc
import importlib.util
import json
import os
from pathlib import Path
import time

from . import weighted_keys as old
from . import io as b

torch = old.torch


@torch.inference_mode()
def main():
    os.environ.update(
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    arm = Path(os.environ["CB_AGENT_FIT_DIR"])
    p = b.load_protocol(arm / "fit_protocol.json")
    for path, digest in p["files"].items():
        assert old.b.sha(path) == digest, path
    out = arm / "key_run"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    inputs = [json.loads(x) for x in Path(p["inputs"]).read_text().splitlines()]
    review = json.loads(Path(p["input_review"]).read_text())
    source_bank, target_bank, weights_bank, sampling = [], [], [], []
    for rank, record in enumerate(inputs):
        assert record["split"] == "train" and not record["task_answers_read"]
        for f in record["files"]:
            assert old.b.sha(f["path"]) == review["files"][f["path"]], f["path"]
        source, target = old.read_cache(record, "source"), old.read_cache(
            record, "target"
        )
        allocation_rank = p["K_allocation_order"].index(record["task_id"])
        h = record["H"]
        n = 16384 // len(inputs) + (allocation_rank < 16384 % len(inputs))
        assert source.shape == (40, 2, h, 8, 128) and target.shape == (
            64,
            2,
            h + record["Q"],
            8,
            128,
        )
        ids = old.positions(h, n)
        boundary = record["history_ids"].index(151645) + 1
        anchor, static, dialogue = (
            ids < 4,
            (ids >= 4) & (ids < boundary),
            ids >= boundary,
        )
        assert int(anchor.sum()) == 4 and bool(static.any()) and bool(dialogue.any())
        weights = torch.ones(n, dtype=torch.float64)
        weights[static] = (n - 4) / 2 / int(static.sum())
        weights[dialogue] = (n - 4) / 2 / int(dialogue.sum())
        assert abs(float(weights.sum()) - n) < 1e-9
        cos, sin = old.qwen_rope(ids.to("cuda:0"), head_dim=128, theta=1e6)
        source_bank.append(
            old.remove_rope(source[:, 0, ids].to("cuda:0").float(), cos, sin).cpu()
        )
        target_bank.append(
            old.remove_rope(target[:, 0, ids].to("cuda:0").float(), cos, sin).cpu()
        )
        weights_bank.append(weights)
        sampling.append(
            dict(
                task=record["task_id"], positions=ids.tolist(), weights=weights.tolist()
            )
        )
        del source, target, cos, sin
        old.b.write(
            out / "status.json",
            dict(
                stage="collect_keys",
                completed=len(sampling),
                elapsed_seconds=time.monotonic() - started,
            ),
        )
    source, target, weights = (
        torch.cat(source_bank, 1),
        torch.cat(target_bank, 1),
        torch.cat(weights_bank),
    )
    del source_bank, target_bank, weights_bank
    assert source.shape[1] == target.shape[1] == len(weights) == 16384
    old.b.write(out / "sampling.json", sampling)
    destination = out / "mappers/initializer"
    destination.mkdir(parents=True)
    stats = out / "statistics"
    stats.mkdir()
    solves, replay = [], []
    for layer in range(64):
        base = old.load_file(
            str(Path(p["base_initializer"]) / f"layer_{layer:03d}.safetensors")
        )
        values, statistics, residual = old.fit(source, target[layer], base, weights)
        assert all(
            torch.equal(v, base[k])
            for k, v in values.items()
            if k not in ("k_weight", "k_bias")
        )
        old.save_file(values, str(destination / f"layer_{layer:03d}.safetensors"))
        torch.save(statistics, stats / f"layer_{layer:03d}.pt")
        solves.append(dict(layer=layer, normal_residual=residual))
        old.b.write(
            out / "status.json",
            dict(
                stage="fit_keys",
                completed=layer + 1,
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        del base, values, statistics
        gc.collect()
    old.b.write(
        out / "complete.json",
        dict(
            status="passed",
            cases=len(inputs),
            observations=16384,
            solves=solves,
            baseline_historical_replay=replay,
            elapsed_seconds=time.monotonic() - started,
            gradient_training=False,
        ),
    )


if __name__ == "__main__":
    main()
