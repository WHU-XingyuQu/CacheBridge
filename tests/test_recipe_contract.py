import json

import pytest

from cachebridge.offline.io import load_protocol


def test_fixed_recipe_rejects_silently_ignored_parameter_changes(tmp_path):
    path = tmp_path / "fit_protocol.json"
    recipe = dict(
        source_layers=40,
        K_observations=16384,
        K_ridge=1.0,
        V_mean_diag_fraction=0.01,
        no_gradient=True,
        inputs="inputs.jsonl",
        files={},
    )
    path.write_text(json.dumps(recipe))
    assert load_protocol(path)["inputs"] == str(tmp_path / "inputs.jsonl")
    for key, changed in [
        ("source_layers", 32),
        ("K_observations", 2048),
        ("K_ridge", 0.1),
        ("V_mean_diag_fraction", 0.1),
        ("no_gradient", False),
    ]:
        path.write_text(json.dumps(dict(recipe, **{key: changed})))
        with pytest.raises(ValueError, match=key):
            load_protocol(path)
