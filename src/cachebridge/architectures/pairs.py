"""Complete model-pair contracts, separate from individual family adapters."""

from dataclasses import dataclass
from . import describe
from .base import config_value, text_config


@dataclass(frozen=True)
class ModelPair:
    name: str
    family: str
    source_layers: int
    target_layers: int
    source_query_heads: int
    target_query_heads: int
    kv_heads: int
    head_dim: int
    hidden_size: int
    rope_theta: float
    vocab_size: int
    tensor_parallel_size: int

    def validate(self, config, role):
        if role not in ("source", "target"):
            raise ValueError("role must be source or target")
        values = text_config(config)
        spec = describe(values)
        expected = dict(
            model_type=self.family,
            num_hidden_layers=getattr(self, role + "_layers"),
            num_attention_heads=getattr(self, role + "_query_heads"),
            num_key_value_heads=self.kv_heads,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
            rope_theta=self.rope_theta,
            rope_scaling=None,
            vocab_size=self.vocab_size,
        )
        for key, value in expected.items():
            if config_value(values, key, None) != value:
                raise ValueError(
                    f"Unsupported {role} model geometry for {self.name}: {key}"
                )
        return spec


QWEN3_14B_32B = ModelPair(
    "qwen3_14b_32b", "qwen3", 40, 64, 40, 64, 8, 128, 5120, 1e6, 151936, 4
)


def get_pair(name="qwen3_14b_32b"):
    if name != QWEN3_14B_32B.name:
        raise ValueError(f"No complete model-pair recipe registered for {name!r}")
    return QWEN3_14B_32B
