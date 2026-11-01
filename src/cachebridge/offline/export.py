"""Export the complete trajectory recipe mapper descriptor."""

from . import io


def export_mapper(root, mapper_id):
    directory = root / "fit_run/mappers/TauBalancedReadoutV"
    io.write(
        root / "candidate.json",
        dict(
            schema=1,
            mapper_id=mapper_id,
            mapper_dir=str(directory.relative_to(root)),
            file_hashes={
                f.name: io.sha(f) for f in sorted(directory.glob("layer_*.safetensors"))
            },
            source_model="Qwen/Qwen3-14B",
            target_model="Qwen/Qwen3-32B",
            source_rope_theta=1e6,
            target_rope_theta=1e6,
            source_layers=40,
            target_layers=64,
            kv_heads=8,
            head_dim=128,
            logical_parameter_bytes=537395200,
            training_protocol="fit_protocol.json",
            training_protocol_sha256=io.sha(root / "fit_protocol.json"),
            quality="unmeasured",
            online_cost="unmeasured",
            no_default_adoption=True,
        ),
    )

    return root / "candidate.json"
