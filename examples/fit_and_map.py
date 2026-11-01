"""CPU initialization-to-mapping example; synthetic data, no model downloads.

Run after installing the package: python examples/fit_and_map.py
This checks an affine fixture, not task quality or complete V calibration.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from cachebridge.offline.fitting import fit_layer
from cachebridge.online.mapper import PageMapper
from cachebridge.rope import apply_rope, qwen_rope


def main():
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(42)
    # Independent observations: first 192 for fitting, last 64 for validation.
    source = torch.randn(2, 3, 256, 2, 4, generator=rng)
    target = 0.6 * source[:, 0] + 0.3 * source[:, 2] + 0.1
    weights, diagnostics = fit_layer(
        source[:, :, :192], target[:, :192], selected=[0, 2], ridge=1e-4
    )
    with TemporaryDirectory(prefix="cachebridge-example-") as directory:
        save_file(weights, str(Path(directory) / "layer_000.safetensors"))
        mapper = PageMapper(
            directory,
            method="cachebridge",
            source_rope_theta=10000,
            target_rope_theta=1000000,
        )
        page = source[:, :, 192:].clone()
        positions = torch.arange(192, 256)
        source_cos, source_sin = qwen_rope(positions, head_dim=4, theta=10000)
        page[0] = apply_rope(page[0], source_cos, source_sin)
        output = torch.empty(2, 1, 64, 2, 4)
        result = mapper.map_page(page, position_start=192, output=output)
        expected = target[:, None, 192:].clone()
        target_cos, target_sin = qwen_rope(positions, head_dim=4, theta=1000000)
        expected[0] = apply_rope(expected[0], target_cos, target_sin)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
        assert result.data_ptr() == output.data_ptr()
        print(
            f"CPU fit -> save -> load -> map passed; max error = {(result - expected).abs().max():.3g}"
        )
        print(f"Solved {len(diagnostics)} head/component systems. No weights retained.")


if __name__ == "__main__":
    main()
