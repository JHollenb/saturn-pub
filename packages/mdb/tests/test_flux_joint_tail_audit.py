"""Qualify five-block own-state reference and ordered discrepancy instruments."""
from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parents[1]
A = runpy.run_path(str(HERE / "qualification/analyze_flux_joint_tail.py"))


def test_common_telescope_keeps_signed_cross_terms_and_shared_denominator():
    native = np.ones((1, 2, 3), np.float32)
    states = [native * value for value in (1, 2, -.5, 1.5, 1.25)]
    result = A["common_telescope"](states)
    assert result["max_abs_identity_residual"] == 0
    assert result["total"]["signed_mean"] == .25
    assert result["terms"]["joint2_addition"]["signed_mean"] == -2.5
    assert result["terms"]["joint4_addition"]["signed_mean"] == -.25
    assert result["terms"]["joint2_addition"]["native_normalized_relative_rms"] == 2.5
    assert any(item["dot"] < 0 for item in result["pairwise_cross_terms"].values())
    assert A["finite_difference"]([native, native * 2, native * 1.25])["local"]["signed_mean"] == -.75


def test_common_telescope_zero_denominator_stays_undefined():
    zero = np.zeros((2, 2), np.float32)
    result = A["common_telescope"]([zero] * 5)
    assert all(row["native_normalized_relative_rms"] is None for row in result["terms"].values())
    assert all(row["cosine"] is None for row in result["pairwise_cross_terms"].values())


def test_independent_five_joint_chain_feeds_its_own_two_streams():
    torch = pytest.importorskip("torch")
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

    torch.manual_seed(4103)
    model = Flux2Transformer2DModel(in_channels=4, out_channels=4, num_layers=5,
        num_single_layers=1, attention_head_dim=4, num_attention_heads=2,
        joint_attention_dim=6, timestep_guidance_channels=8, mlp_ratio=2,
        axes_dims_rope=(2, 2), guidance_embeds=False).eval().to(torch.bfloat16)
    fixture = runpy.run_path(str(HERE / "tests/test_flux_reentry_reference.py"))
    predictor = runpy.run_path(str(HERE / "qualification/flux_reentry_reference.py"))
    independent = A["NEW"]
    z, c, image_ids, text_ids = fixture["operands"]()
    params, wide = fixture["parameters"], fixture["wide"]
    weights = {key: value for key, value in params(model).items()
        if any(key.startswith(name + ".") for name in predictor["PROJECT_LINEAR_NAMES"])}
    project = predictor["predict_project"](wide(z), wide(c), image_ids.numpy(), text_ids.numpy(),
        np.float32(166.4853), weights, axes_dims_rope=[2, 2], timestep_channels=8)
    p, i = project, project
    profile = {"heads": 2, "layernorm_eps": 1e-6, "qk_eps": 1e-6}
    for index in range(5):
        weights = params(model.transformer_blocks[index])
        p = predictor["predict_joint"](p["image"], p["text"], project["context"], weights, **profile)
        i = independent["independent_joint"](i["image"], i["text"], project["context"], weights, **profile)
        for key in ("image", "text"):
            np.testing.assert_array_equal(p[key], i[key])
    wrong = predictor["predict_joint"](project["image"], project["text"], project["context"], weights, **profile)
    assert not np.array_equal(wrong["image"], p["image"])
