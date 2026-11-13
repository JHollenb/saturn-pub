"""Small independent mechanics check of sealed whole-image BF16 forecasts."""
import runpy
from pathlib import Path

import numpy as np
import torch


def test_complete_joint_forecasts_preserve_native_sequential_edges():
    module = runpy.run_path(str(Path(__file__).parents[1] / "qualification/flux_anime_origin_joint.py"))
    generator = torch.Generator().manual_seed(83)
    base = {name: (torch.randn(1, 7, 16, generator=generator) * scale).to(torch.bfloat16)
            for name, scale in (("residual", 20), ("attention", 4), ("mlp", 3))}
    base.update({name: torch.randn(1, 1, 16, generator=generator).to(torch.bfloat16) for name in ("gate_attention", "gate_mlp")})
    donor = {name: value.clone() for name, value in base.items()}
    for name in ("residual", "attention", "mlp"):
        donor[name][:, [1, 3, 5]] += torch.randn(1, 3, 16, generator=generator).to(torch.bfloat16)
    queries, dims = [1, 4], [0, 5, 11]
    predictions = module["forecasts"](base, donor, queries, dims)
    for mode, prediction in predictions.items():
        r, a, m = ((donor if letter in mode else base)[name] for letter, name in (("R", "residual"), ("A", "attention"), ("M", "mlp")))
        native = (r + base["gate_attention"] * a) + base["gate_mlp"] * m
        full = (prediction["image_full_bf16_bits"].astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(full, native.float().numpy())
        np.testing.assert_array_equal(prediction["image_bf16"], native[:, queries][:, :, dims].float().numpy())
