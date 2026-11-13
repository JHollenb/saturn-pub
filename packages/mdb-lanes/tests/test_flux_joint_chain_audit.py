"""Qualify matched-state decomposition and a second independently fed joint."""
from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parents[1]
A = runpy.run_path(str(HERE / "qualification/analyze_flux_joint_chain.py"))


def test_decomposition_preserves_opposition_and_zero_norm():
    native = np.zeros((1, 2, 3), np.float32)
    bridge = np.ones_like(native)
    prediction = np.full_like(native, .25)
    report = A["finite_difference"]([native, bridge, prediction])
    assert report["inherited"]["signed_mean"] == 1
    assert report["joint1_local"]["signed_mean"] == -.75
    assert report["total"]["signed_mean"] == .25
    assert report["cross_dot"] < 0
    assert report["max_abs_identity_residual"] == 0
    assert report["squared_norm_identity_residual"] == 0
    zero = A["finite_difference"]([native, native, native])
    assert zero["cross_cosine"] is None


def test_joint1_independent_reference_consumes_its_own_joint0_output():
    torch = pytest.importorskip("torch")
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

    torch.manual_seed(1921)
    model = Flux2Transformer2DModel(in_channels=4, out_channels=4, num_layers=3,
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
    for index in (0, 1):
        weights = params(model.transformer_blocks[index])
        p = predictor["predict_joint"](p["image"], p["text"], project["context"], weights, **profile)
        i = independent["independent_joint"](i["image"], i["text"], project["context"], weights, **profile)
        for key in ("image", "text"):
            np.testing.assert_array_equal(p[key], i[key])
    wrong = predictor["predict_joint"](project["image"], project["text"], project["context"], weights, **profile)
    assert not np.array_equal(wrong["image"], p["image"])
