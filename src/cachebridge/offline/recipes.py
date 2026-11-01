"""Explicit scientific recipes composed from reusable fitting stages."""

from dataclasses import dataclass
from importlib import import_module
from ..architectures.pairs import get_pair


@dataclass(frozen=True)
class FittingRecipe:
    name: str
    pair: str
    source_support: str = "supplied_initializer"
    key_observations: int = 16384
    key_ridge: float = 1.0
    value_mean_diag_fraction: float = 0.01

    @property
    def stages(self):
        return (
            "prepare_traces",
            "load_support",
            "initialize_keys",
            "calibrate_values",
            "export",
        )

    def validate_target(self, config):
        return get_pair(self.pair).validate(config, "target")

    def validate_protocol(self, protocol):
        pair = get_pair(self.pair)
        fixed = dict(
            source_layers=pair.source_layers,
            K_observations=self.key_observations,
            K_ridge=self.key_ridge,
            V_mean_diag_fraction=self.value_mean_diag_fraction,
            no_gradient=True,
        )
        for key, expected in fixed.items():
            if protocol.get(key) != expected:
                raise ValueError(f"Recipe {self.name} requires {key}={expected}")

    def attention_implementation(self):
        # This recipe captures the Qwen3 SDPA shadow readout.
        return import_module("transformers.models.qwen3.modeling_qwen3")


TRAJECTORY = FittingRecipe("qwen3_14b_32b_trajectory", "qwen3_14b_32b")


def get_recipe(name=TRAJECTORY.name):
    if name != TRAJECTORY.name:
        raise ValueError(f"Unknown complete fitting recipe: {name!r}")
    return TRAJECTORY
