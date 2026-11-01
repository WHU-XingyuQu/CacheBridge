"""Same conditional V solve, selecting assistant prediction positions on TRAIN."""

import gc
import json
import os
from pathlib import Path
import sys
import time
from .. import io as b
from ..traces.io import read_cache
from ..traces.hf_cache import dynamic_cache_from_data
from ..recipes import get_recipe
from ...architectures import get_adapter
from .algebra import probabilities, gram, solve
from ...mapper import PageMapper

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def append(path, value):
    with path.open("a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")


def data(value):
    # value [2,64,H,8,128], retain its shared CPU backing storage.
    return [
        (value[0, l].permute(1, 0, 2)[None], value[1, l].permute(1, 0, 2)[None])
        for l in range(64)
    ]


def mapped(source, mapper):
    parts = []
    for start in range(0, source.shape[2], 512):
        page = source[:, :, start : start + 512].permute(1, 0, 2, 3, 4).contiguous()
        parts.append(mapper.map_page(page, position_start=start))
    return data(torch.cat(parts, dim=2))


def restore(values, model):
    return dynamic_cache_from_data(
        [
            (k.to(next(l.parameters()).device), v.to(next(l.parameters()).device))
            for (k, v), l in zip(values, model.model.layers, strict=True)
        ],
        config=model.config,
    )


@torch.inference_mode()
def query(model, ids, values):
    cache = restore(values, model)
    h = cache.get_seq_length()
    dev = model.model.embed_tokens.weight.device
    pos = torch.arange(h, h + len(ids), device=dev)
    out = model.model(
        input_ids=torch.tensor([ids], device=dev),
        past_key_values=cache,
        use_cache=True,
        position_ids=pos[None],
        cache_position=pos,
        attention_mask=torch.ones(1, h + len(ids), dtype=torch.long, device=dev),
    )
    return (
        model.lm_head(out.last_hidden_state[:, -1:].to(model.lm_head.weight.device))[
            0, -1
        ]
        .float()
        .cpu()
    )


def features(item, layer, mapper, device):
    return torch.cat(
        [
            item["source_v"][i].permute(1, 0, 2)
            for i in mapper.layers[layer].selected.tolist()
        ],
        dim=-1,
    ).to(device=device, dtype=torch.float32)


def folded(x, weights, dtype):
    w = weights["v_weight"].to(x.device).float()
    bias = weights["v_bias"].to(x.device).float()
    parts = []
    for start in range(0, x.shape[1], 512):
        page = x[:, start : start + 512].permute(1, 0, 2).contiguous()
        parts.append(
            (torch.einsum("thf,hfd->thd", page, w) + bias)
            .to(dtype)
            .permute(1, 0, 2)
            .cpu()
        )
    return torch.cat(parts, dim=1)[None]


@torch.inference_mode()
def main():
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    arm = Path(os.environ["CB_AGENT_FIT_DIR"])
    p = b.load_protocol(arm / "fit_protocol.json")
    b.validate_target_model(p["target_model"])
    out = arm / "fit_run"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    stage = "preflight"
    done = 0

    def status(**kw):
        b.write(
            out / "status.json",
            dict(
                stage=stage,
                completed=done,
                elapsed_seconds=time.monotonic() - started,
                pid=os.getpid(),
                **kw,
            ),
        )

    for path, digest in p["files"].items():
        assert b.sha(path) == digest, path
    review = json.loads(Path(p["input_review"]).read_text())
    assert review["status"] == "passed"
    # Snapshot content hashes were computed by the CPU review and frozen there;
    # recheck each only as loaded, avoiding a second full read before model load.
    status()
    model = AutoModelForCausalLM.from_pretrained(
        p["target_model"],
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={i: "24GiB" for i in range(4)},
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    model.requires_grad_(False)
    assert all(
        x.device.type == "cuda" and x.device.index in range(4)
        for x in model.parameters()
    )
    tok = AutoTokenizer.from_pretrained(p["target_model"], local_files_only=True)
    mapper = PageMapper(
        p["initializer"],
        method="cachebridge",
        source_rope_theta=1e6,
        target_rope_theta=1e6,
        device="cuda:0",
    )
    current = PageMapper(
        p["current_mapper"],
        method="cachebridge",
        source_rope_theta=1e6,
        target_rope_theta=1e6,
        device="cuda:0",
    )
    for a, z in zip(mapper.layers, current.layers, strict=True):
        assert (
            torch.equal(a.selected, z.selected)
            and a.k_weight.shape == z.k_weight.shape
            and a.k_bias.shape == z.k_bias.shape
        )
    selection = json.loads(Path(p["position_manifest"]).read_text())
    by_task = {r["task"]: r for r in selection["cases"]}
    assert selection["status"] == "passed" and selection["N"] == p["expected_cases"]
    recipe = get_recipe(p.get("recipe", "qwen3_14b_32b_trajectory"))
    adapter = get_adapter(model.config)
    recipe.validate_target(model.config)
    prepare, step, logits = adapter.prepare, adapter.step, adapter.logits
    impl = recipe.attention_implementation()
    original = impl.ALL_ATTENTION_FUNCTIONS["sdpa"]
    state = {}
    bank = []

    def capture(module, q, k, v, mask, **kwargs):
        output, weights = original(module, q, k, v, mask, **kwargs)
        if not state.get("enabled"):
            return output, weights
        item = state["item"]
        h = item["H"]
        layer = state["layer"]
        positions = list(range(q.shape[2]))
        att = probabilities(q, k, h, positions)
        actual = att @ v[0].float()
        recon = (
            actual.reshape(8, -1, len(positions), 128)
            .permute(2, 0, 1, 3)
            .reshape(len(positions), -1, 128)
        )
        ref = output[0].float()
        error = float(
            (recon - ref).square().sum() / ref.square().sum().clamp_min(1e-12)
        )
        assert error <= 0.001
        nk, nv = item["native"][layer]
        nativep = probabilities(
            q, torch.cat((nk.to(k.device), k[:, :, h:]), dim=2), h, positions
        )
        shadow = nativep @ torch.cat((nv.to(v.device), v[:, :, h:]), dim=2)[0].float()
        y = shadow - actual
        if state.get("parent_control"):
            full_a = att[:, :, :h] @ state["x"]
            fg, fc = gram(full_a.double(), y.double())
            state["full_G"].add_(fg.cpu())
            state["full_C"].add_(fc.cpu())
            del full_a, fg, fc
        selected = item["positions"]
        indices = torch.tensor(
            [
                group * q.shape[2] + position
                for group in range(8)
                for position in selected
            ],
            device=q.device,
        )
        att = att.index_select(1, indices)
        y = y.index_select(1, indices)
        shadow = shadow.index_select(1, indices)
        a = att[:, :, :h] @ state["x"]
        g, c = gram(a.double(), y.double())
        state["G"].add_(g.double().cpu())
        state["C"].add_(c.double().cpu())
        if state.get("numerical_replay"):
            g32, c32 = gram(a, y)
            state["G32"].add_(g32.double().cpu())
            state["C32"].add_(c32.double().cpu())
        name = (
            "parent_control_readouts.jsonl"
            if state.get("parent_control")
            else (
                "numerical_replay_readouts.jsonl"
                if state.get("numerical_replay")
                else "readouts.jsonl"
            )
        )
        append(
            out / name,
            dict(
                task=item["task"],
                split="train",
                layer=layer,
                H=h,
                Q=q.shape[2],
                query_positions=len(selected),
                target_kind="current_query_Native_history_shadow_assistant_predictions",
                reconstruction_relative_SSE=error,
                actual_native_SSE=float(y.square().sum()),
                native_shadow_energy=float(shadow.square().sum()),
            ),
        )
        return output, weights

    try:
        stage = "collect_bank"
        status()
        inputs = [json.loads(x) for x in Path(p["inputs"]).read_text().splitlines()]
        for item in inputs:
            if item["status"] != "calibration_input_complete":
                continue
            assert item["split"] == "train" and not item["task_answers_read"]
            for f in item["files"]:
                assert b.sha(f["path"]) == review["files"][f["path"]], f["path"]
            source = read_cache(item, "source")
            native_full = read_cache(item, "target")
            h = item["H"]
            assert source.shape == (p["source_layers"], 2, h, 8, 128)
            assert native_full.shape == (64, 2, h + item["Q"], 8, 128)
            local = mapped(source, mapper)
            native = data(native_full[:, :, :h].permute(1, 0, 2, 3, 4))
            hidden, meta = prepare(model, item["query_ids"], h)
            selected = by_task[item["task_id"]]
            assert selected["H"] == h and selected["Q"] == item["Q"]
            positions = selected["assistant_prediction_positions"]
            assert positions == sorted(set(positions)) and all(
                0 <= i < item["Q"] for i in positions
            )
            bank.append(
                dict(
                    task=item["task_id"],
                    H=h,
                    query_ids=item["query_ids"],
                    history_ids=item["history_ids"],
                    source_v=source[:, 1].clone(),
                    local=local,
                    native=native,
                    hidden=hidden,
                    meta=meta,
                    positions=positions,
                )
            )
            done += 1
            status(task=item["task_id"])
            del source, native_full, local, native, hidden, meta
        assert len(bank) == review["usable"] and len(bank) >= 16
        stage = "executor_gate"
        status()
        first = bank[0]
        hidden = first["hidden"]
        for layer in range(64):
            hidden, _ = step(model, layer, hidden, first["local"][layer], first["meta"])
        expected = logits(model, hidden)
        actual = query(model, first["query_ids"], first["local"])
        assert torch.equal(expected, actual), (
            "Layer executor parity",
            float((expected - actual).abs().max()),
        )
        b.write(
            out / "executor_gate.json",
            dict(status="passed", logits_bitwise_equal=True, train_task=first["task"]),
        )
        impl.ALL_ATTENTION_FUNCTIONS["sdpa"] = capture
        assert (
            json.loads((Path(p["key_source"]) / "complete.json").read_text())["status"]
            == "passed"
        )
        destination = out / "mappers/TauBalancedReadoutV"
        destination.mkdir(parents=True)
        (out / "statistics").mkdir()
        stage = "sequential_readout_solve"
        done = 0
        status()
        solves = []
        for layer in range(64):
            assert time.monotonic() - started < 7200, "bounded fit watchdog"
            dev = next(model.model.layers[layer].parameters()).device
            state.update(
                enabled=True,
                layer=layer,
                G=torch.zeros(8, 1025, 1025, dtype=torch.float64),
                C=torch.zeros(8, 1025, 128, dtype=torch.float64),
            )
            for item in bank:
                x = features(item, layer, mapper, dev)
                state.update(
                    item=item,
                    x=torch.cat((x, torch.ones(8, x.shape[1], 1, device=dev)), dim=-1),
                )
                scratch, kv = step(
                    model, layer, item["hidden"], item["local"][layer], item["meta"]
                )
                del scratch, kv, x
            state["enabled"] = False
            state.pop("x")
            g = state.pop("G")
            c = state.pop("C")
            torch.save(dict(G=g, C=c), out / "statistics" / f"layer_{layer:03d}.pt")
            delta, lam, residual = solve(g.to("cuda:0"), c.to("cuda:0"), fraction=0.01)
            assert torch.isfinite(delta).all() and float(residual.max()) <= 1e-7
            base = load_file(
                str(Path(p["initializer"]) / f"layer_{layer:03d}.safetensors")
            )
            values = {k: v.clone() for k, v in base.items()}
            delta = delta.cpu()
            values["v_weight"] = (base["v_weight"].double() + delta[:, :-1]).to(
                base["v_weight"].dtype
            )
            values["v_bias"] = (base["v_bias"].double() + delta[:, -1]).to(
                base["v_bias"].dtype
            )
            assert all(
                v.shape == base[k].shape
                and v.dtype == base[k].dtype
                and torch.isfinite(v).all()
                for k, v in values.items()
            )
            assert all(
                torch.equal(v, base[k])
                for k, v in values.items()
                if k not in ("v_weight", "v_bias")
            )
            save_file(values, str(destination / f"layer_{layer:03d}.safetensors"))
            solves.append(
                dict(
                    layer=layer,
                    lambda_values=lam.cpu().tolist(),
                    normal_residual_max=float(residual.max()),
                    delta_rms=float(delta.square().mean().sqrt()),
                )
            )
            for item in bank:
                x = features(item, layer, mapper, dev)
                newv = folded(x, values, item["local"][layer][1].dtype)
                item["hidden"], kv = step(
                    model,
                    layer,
                    item["hidden"],
                    (item["local"][layer][0], newv),
                    item["meta"],
                )
                del x, newv, kv
            done = layer + 1
            status()
            print(
                json.dumps(
                    dict(
                        stage=stage,
                        layer=layer,
                        residual=solves[-1]["normal_residual_max"],
                    )
                ),
                flush=True,
            )
            del g, c, delta, lam, residual, base, values
            gc.collect()
        b.write(
            out / "solve.json",
            dict(
                status="passed",
                layers=solves,
                train_cases=[x["task"] for x in bank],
                gradient_training=False,
                K_frozen_to_train_initializer=True,
                head_support_unchanged=True,
            ),
        )
        candidate = PageMapper(
            destination,
            method="cachebridge",
            source_rope_theta=1e6,
            target_rope_theta=1e6,
            device="cuda:0",
        )
        stage = "folded_candidate_gate"
        status()
        gates = []
        # Reopen only the two fixed length extremes to test actual PageMapper
        # coefficients against the states propagated during the layer solves.
        for item in (min(bank, key=lambda x: x["H"]), max(bank, key=lambda x: x["H"])):
            record = next(x for x in inputs if x["task_id"] == item["task"])
            source = read_cache(record, "source")
            converted = mapped(source, candidate)
            expected = logits(model, item["hidden"])
            actual = query(model, item["query_ids"], converted)
            equal = torch.equal(expected, actual)
            gap = float((expected - actual).abs().max())
            gates.append(
                dict(
                    train_task=item["task"],
                    H=item["H"],
                    logits_bitwise_equal=equal,
                    max_logit_gap=gap,
                    top1_equal=int(expected.argmax()) == int(actual.argmax()),
                )
            )
            assert equal, ("Folded candidate executor", item["task"], gap)
            del source, converted
        b.write(out / "gates.json", dict(status="passed", folded_executor=gates))
        files = {
            path.name: b.sha(path)
            for path in sorted(destination.glob("layer_*.safetensors"))
        }
        assert len(files) == 64
        b.write(
            destination / "manifest.json",
            dict(
                status="complete_calibration_candidate",
                method="head_local_balanced_K_sequential_assistant_prediction_readout_V",
                mapper_id="TauBalancedReadoutV",
                files=files,
                fit_protocol_sha256=b.sha(arm / "fit_protocol.json"),
                train_cases=[x["task"] for x in bank],
                test_dev_labels_used=False,
                gradient_training=False,
                K_frozen_to_train_initializer=True,
                support_shapes_dtype_unchanged=True,
                quality_and_online_cost="pending",
            ),
        )
        stage = "complete"
        status()
        b.write(
            out / "complete.json",
            dict(
                status="passed",
                mapper_dir=str(destination),
                manifest_sha256=b.sha(destination / "manifest.json"),
                quality_confirmed=False,
                online_cost_confirmed=False,
                elapsed_seconds=time.monotonic() - started,
            ),
        )
    finally:
        impl.ALL_ATTENTION_FUNCTIONS["sdpa"] = original


if __name__ == "__main__":
    main()
