import pytest
import torch
from safetensors.torch import save_file

from cachebridge.offline.calibration import gram, solve
from cachebridge.offline.fitting import fit_layer
from cachebridge.offline.ridge import sufficient_stats, solve_stats
from cachebridge.online.mapper import PageMapper
from cachebridge.rope import apply_rope, qwen_rope


def test_two_pass_ridge_matches_centered_direct_system():
    rng = torch.Generator().manual_seed(17)
    x = torch.randn(97, 9, generator=rng, dtype=torch.float64) + 7
    y = torch.randn(97, 4, generator=rng, dtype=torch.float64) - 3
    weights = torch.rand(97, generator=rng, dtype=torch.float64) + 0.1
    ridge = 0.7
    mx = (weights[:, None] * x).sum(0) / weights.sum()
    my = (weights[:, None] * y).sum(0) / weights.sum()
    xc, yc = x - mx, y - my
    expected = torch.linalg.solve(
        xc.T @ (weights[:, None] * xc) + ridge * torch.eye(9),
        xc.T @ (weights[:, None] * yc),
    )
    with torch.inference_mode():
        stats = sufficient_stats(
            x, y, weights, device="cpu", dtype=torch.float64, chunk_rows=13
        )
        actual, bias, diagnostic = solve_stats(stats, ridge)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(bias, my - mx @ expected, atol=1e-9, rtol=1e-9)
    assert diagnostic["linear_system_relative_residual"] < 1e-12


def test_calibration_solve_satisfies_regularized_normal_equations():
    rng = torch.Generator().manual_seed(8)
    x = torch.randn(2, 31, 5, generator=rng, dtype=torch.float64)
    y = torch.randn(2, 31, 3, generator=rng, dtype=torch.float64)
    g, c = gram(x, y)
    delta, lam, residual = solve(g, c)
    expected_lam = 0.01 * x.square().sum((1, 2)) / (31 * 5)
    torch.testing.assert_close(lam, expected_lam)
    expected = torch.linalg.solve(g + lam[:, None, None] * torch.eye(5), c)
    torch.testing.assert_close(delta, expected)
    assert residual.max() < 1e-12


def test_fit_export_and_tp_mapping_on_held_out_observations(tmp_path):
    rng = torch.Generator().manual_seed(14)
    source = torch.randn(2, 3, 128, 4, 4, generator=rng)
    target = 0.7 * source[:, 2] - 0.2 * source[:, 0] + 0.13
    weights, diagnostics = fit_layer(
        source[:, :, :96],
        target[:, :96],
        selected=[2, 0],
        ridge=1e-4,
        chunk_rows=17,
    )
    save_file(weights, str(tmp_path / "layer_000.safetensors"))
    assert len(diagnostics) == 8
    assert not any(value.requires_grad for value in weights.values())
    positions = torch.arange(103, 135)
    page = source[:, :, 96:].clone()
    cos, sin = qwen_rope(positions, head_dim=4, theta=10000)
    page[0] = apply_rope(page[0], cos, sin)
    expected = target[:, None, 96:].clone()
    cos, sin = qwen_rope(positions, head_dim=4, theta=1000000)
    expected[0] = apply_rope(expected[0], cos, sin)
    before = page.clone()
    for rank in range(2):
        mapper = PageMapper(
            tmp_path,
            method="cachebridge",
            source_rope_theta=10000,
            target_rope_theta=1000000,
            target_tp_size=2,
            target_tp_rank=rank,
        )
        output = torch.empty(2, 1, 32, 2, 4)
        result = mapper.map_page(page, position_start=103, output=output)
        reference = expected[:, :, :, rank * 2 : (rank + 1) * 2]
        torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)
        assert result.data_ptr() == output.data_ptr()
        compact = mapper.map_page(
            page[:, :, :, rank * 2 : (rank + 1) * 2],
            position_start=103,
            source_head_offset=rank * 2,
        )
        torch.testing.assert_close(compact, result, atol=0, rtol=0)
    torch.testing.assert_close(page, before, atol=0, rtol=0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ridge": -1},
        {"ridge": float("nan")},
        {"selected": []},
        {"selected": [0, 0]},
        {"selected": [-1]},
        {"selected": [3]},
        {"chunk_rows": 0},
        {"dtype": torch.bfloat16},
    ],
)
def test_fit_rejects_invalid_contract(overrides):
    args = dict(selected=[0, 2], ridge=1)
    args.update(overrides)
    with pytest.raises(ValueError):
        fit_layer(torch.zeros(2, 3, 10, 2, 4), torch.zeros(2, 10, 2, 4), **args)
