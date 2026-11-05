"""Attention arithmetic and structural controls; no trained model execution."""
from pathlib import Path
import runpy

import numpy as np
import pytest

L = runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification/flux_attention_state_law.py"))


def specimen():
    rng = np.random.default_rng(101)
    return [rng.normal(size=(1, 2, 5, 4)) for _ in range(3)]


def test_changed_text_key_also_changes_unchanged_image_contribution():
    q, k, v = specimen()
    altered = k.copy()
    altered[:, :, :2] += 1
    c = L["routing_components"](q, k, v, altered, v, 2)
    assert np.max(np.abs(c["closure"])) < 1e-14
    assert np.count_nonzero(c["route_image"]) > 0
    assert np.count_nonzero(c["value"]) == np.count_nonzero(c["interaction"]) == 0
    assert not np.array_equal(c["text_mass_base"], c["text_mass_changed"])


def test_value_only_change_keeps_routes_and_has_no_interaction():
    q, k, v = specimen()
    altered = v.copy()
    altered[:, :, :2] *= -1
    c = L["routing_components"](q, k, v, k, altered, 2)
    assert np.count_nonzero(c["route_image"]) == np.count_nonzero(c["route_text"]) == 0
    assert np.count_nonzero(c["interaction"]) == 0
    assert np.allclose(c["value"], c["delta"], rtol=0, atol=1e-14)


def test_key_value_factorial_retains_the_interaction():
    q, k, v = specimen()
    c = L["routing_components"](q, k, v, -k, -v, 2)
    assert np.count_nonzero(c["interaction"]) > 0
    assert np.max(np.abs(c["closure"])) < 1e-14


def test_selected_queries_have_the_same_dense_reference_result():
    q, k, v = specimen()
    full, _ = L["dense"](q, k, v)
    subset, probability = L["dense"](q[:, :, [1, 4]], k, v)
    assert np.allclose(full[:, :, [1, 4]], subset, rtol=0, atol=1e-15)
    assert np.allclose(probability.sum(-1), 1, rtol=0, atol=1e-15)


def test_identity_rope_preserves_bf16_values():
    x = L["support"]()["quantize"](np.arange(48).reshape(1, 3, 2, 8)/8)
    assert np.array_equal(L["rotate"](x, np.ones((3, 8), np.float32), np.zeros((3, 8), np.float32)), x)


def test_raw_state_forecast_closes_tiny_native_residual_identity():
    torch = pytest.importorskip("torch")
    native = pytest.importorskip("diffusers.models.transformers.transformer_flux2")
    torch.manual_seed(71)
    block = native.Flux2TransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4, mlp_ratio=3).eval().to(torch.bfloat16)
    # With constant rows and zero shifts, both normalized streams are zero.
    # Bias-free projections then produce zero writes, preserving the residual.
    image = torch.ones(1, 4, 8, dtype=torch.bfloat16)
    text = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    mod = torch.zeros(1, 48, dtype=torch.bfloat16)
    mod[:, 16:24] = 1
    mod[:, 40:] = 1
    cos, sin = torch.ones(6, 4), torch.zeros(6, 4)
    weights = {n: p.detach().float().numpy() for n, p in block.named_parameters()}
    state = {n: v.float().numpy() for n, v in dict(image=image, text=text, img_mod=mod, txt_mod=mod, cos=cos, sin=sin).items()}
    forecast = L["forecast"](state, weights, [0, 3], [0, 1], heads=2)
    with torch.no_grad():
        _, actual = block(hidden_states=image, encoder_hidden_states=text, temb_mod_img=mod, temb_mod_txt=mod, image_rotary_emb=(cos, sin))
    assert np.array_equal(forecast["image_exit"], actual[:, [0, 3]].float().numpy())
    assert np.count_nonzero(forecast["attention"]) == np.count_nonzero(forecast["mlp"]) == 0
