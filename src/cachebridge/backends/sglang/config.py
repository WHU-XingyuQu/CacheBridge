"""Explicit, immutable runtime configuration; no environment reads at import."""

from dataclasses import dataclass
import json
from pathlib import Path
from ...artifacts import file_sha256
from ...architectures.pairs import get_pair


@dataclass(frozen=True)
class RuntimeConfig:
    config_path: Path
    source_model: Path
    target_model: Path
    mapper_descriptor: Path
    model_pair: str = "qwen3_14b_32b"
    tensor_parallel_size: int = 4
    pool_tokens: int = 530000
    source_port: int = 34630
    target_port: int = 34631
    dense_descriptor: Path | None = None
    engine_path: Path | None = None

    def __post_init__(self):
        pair = get_pair(self.model_pair)
        if self.tensor_parallel_size != pair.tensor_parallel_size:
            raise ValueError("This serving recipe requires Source TP4 + Target TP4")
        if type(self.pool_tokens) is not int or self.pool_tokens <= 0:
            raise ValueError("pool_tokens must be a positive integer")
        if any(
            type(p) is not int or not 1 <= p <= 65535
            for p in (self.source_port, self.target_port)
        ):
            raise ValueError("Ports must be integers in 1..65535")
        if self.source_port == self.target_port:
            raise ValueError("Source and target ports must differ")

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        settings = json.loads(path.read_text())

        def resolve(value):
            p = Path(value).expanduser()
            return p.resolve() if p.is_absolute() else (path.parent / p).resolve()

        fields = {
            key: settings[key]
            for key in (
                "model_pair",
                "tensor_parallel_size",
                "pool_tokens",
                "source_port",
                "target_port",
            )
            if key in settings
        }
        for key in ("source_model", "target_model", "mapper_descriptor"):
            fields[key] = resolve(settings[key])
        for key in ("dense_descriptor", "engine_path"):
            if settings.get(key):
                fields[key] = resolve(settings[key])
        return cls(config_path=path, **fields)

    def validate_model_pair(self):
        pair = get_pair(self.model_pair)
        for role in ("source", "target"):
            path = getattr(self, role + "_model")
            pair.validate(json.loads((path / "config.json").read_text()), role)
        for name in ("tokenizer.json", "tokenizer_config.json"):
            if file_sha256(self.source_model / name) != file_sha256(
                self.target_model / name
            ):
                raise ValueError(
                    "This integration requires the exact shared Qwen3 tokenizer"
                )
        return dict(
            status="passed",
            source_config_sha256=file_sha256(self.source_model / "config.json"),
            target_config_sha256=file_sha256(self.target_model / "config.json"),
            tokenizer_sha256=file_sha256(self.source_model / "tokenizer.json"),
        )

    def artifact_contract(self, descriptor=None):
        path = (
            Path(descriptor).expanduser().resolve()
            if descriptor
            else self.mapper_descriptor
        )
        d = json.loads(path.read_text())
        directory = Path(d["mapper_dir"])
        if not directory.is_absolute():
            directory = (path.parent / directory).resolve()
        depth = get_pair(self.model_pair).target_layers
        if set(d["file_hashes"]) != {
            f"layer_{i:03d}.safetensors" for i in range(depth)
        }:
            raise ValueError(f"Complete {depth}-layer mapper required")
        result = dict(
            cachebridge=dict(
                mapper_id=d["mapper_id"],
                descriptor=str(path),
                descriptor_sha256=file_sha256(path),
                mapper_dir=str(directory),
                file_hashes=d["file_hashes"],
                schema="local",
                source_support_unchanged=True,
                logical_parameter_bytes=d["logical_parameter_bytes"],
            )
        )
        if self.dense_descriptor:
            dense = json.loads(self.dense_descriptor.read_text())
            directory = Path(dense["mapper_dir"])
            dense["mapper_dir"] = str(
                directory
                if directory.is_absolute()
                else (self.dense_descriptor.parent / directory).resolve()
            )
            result["dense"] = dense
        return result

    def runtime_parameters(self, descriptor=None):
        return dict(
            source_tp=self.tensor_parallel_size,
            target_tp=self.tensor_parallel_size,
            registry_path="",
            registry_sha256="",
            mapper_id=self.artifact_contract(descriptor)["cachebridge"]["mapper_id"],
            chunked_prefill_size=8192,
        )
