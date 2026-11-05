"""Whole-suffix signed algebra and independently self-fed terminal recurrence."""
from __future__ import annotations

import runpy
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

HERE = Path(__file__).parents[1]
A = runpy.run_path(str(HERE / "qualification/analyze_flux_final_latent.py"))


def test_whole_suffix_signed_terms_use_shared_native_denominator():
    native = np.ones((2, 2), np.float32)
    result = A["suffix_difference"](native, native * 2, native * 1.25)
    assert result["additional_suffix"]["signed_mean"] == -.75
    assert result["native_normalized_relative_rms"] == {"inherited": 1., "additional_suffix": .75, "total": .25}
    assert result["cross_cosine"] == -1. and result["max_abs_identity_residual"] == 0
    zero = np.zeros((2, 2), np.float32)
    assert A["suffix_difference"](zero, zero, zero)["cross_cosine"] is None


def test_negative_cross_cosine_preserves_larger_total_as_observation():
    native = np.ones(2, np.float32)
    bridge = native + np.array([1., 0.], np.float32)
    predicted = bridge + np.array([-.125, 1.], np.float32)
    result = A["suffix_difference"](native, bridge, predicted)
    assert result["cross_cosine"] < 0
    assert result["total"]["l2"] > result["inherited"]["l2"]
    assert result["max_abs_identity_residual"] == 0


def test_context_identity_accepts_read_only_mapping_and_detects_changes():
    x = np.ones((1, 3), np.float32)
    before = {"temb": x, "rope": [x, x * 2]}
    assert A["context_descriptor"](MappingProxyType(before)) == A["context_descriptor"](before)
    assert A["context_descriptor"]({**before, "temb": x * 2}) != A["context_descriptor"](before)
    with pytest.raises(ValueError, match="actual FP32"):
        A["context_descriptor"]({**before, "temb": x.astype(np.float64)})


def test_twenty_distinct_single_banks_then_own_readout_and_final_commit():
    fixture = runpy.run_path(str(HERE / "tests/test_flux_transition_reference.py"))
    primary = fixture["REFERENCE"]
    boundary = runpy.run_path(str(HERE / "qualification/flux_step_boundary_reference.py"))
    residual, modulation, profile = fixture["inputs"]()
    base = fixture["arrays"](fixture["controlled_block"]())
    p, independent = residual.copy(), residual.copy()
    for index in range(20):
        weights = {**base, "to_out_bias": base["to_out_bias"] + np.float32(index * .125)}
        p = primary["predict_transition"](p, modulation, **weights, **{**profile, "text_rows": None, "retain_fields": False})["output"]
        independent = A["CHAIN"]["independent_transition"](independent, modulation, weights,
            heads=profile["heads"], layernorm_eps=profile["layernorm_eps"], qk_eps=profile["qk_eps"],
            rope_cos=profile["rope_cos"], rope_sin=profile["rope_sin"])
        np.testing.assert_array_equal(p, independent)
    temb = np.array([[.5, -.5]], np.float32)
    weights = {"modulation_weight": np.zeros((4, 2), np.float32), "modulation_bias": np.zeros(4, np.float32),
        "projection_weight": np.eye(2, dtype=np.float32), "projection_bias": np.zeros(2, np.float32)}
    noise = boundary["predict_readout"](p[:, 1:], temb, **weights, eps=1e-6)["noise"]
    other_noise = A["MATH"]["independent_readout"](independent[:, 1:], temb, **weights, eps=1e-6)["noise"]
    np.testing.assert_array_equal(noise, other_noise)
    z3 = np.array([[[.5, -.5]]], np.float32)
    sigma, next_sigma = np.float32(.3), np.float32(0.)
    final = boundary["predict_flow_commit"](z3, noise, sigma, next_sigma)["latent"]
    other = A["MATH"]["independent_flow_commit"](z3, other_noise, sigma, next_sigma)["latent"]
    np.testing.assert_array_equal(final, other)
    wrong = primary["predict_transition"](residual, modulation, **weights_for_last(base), **profile)["output"]
    assert not np.array_equal(wrong, p)


def weights_for_last(base):
    return {**base, "to_out_bias": base["to_out_bias"] + np.float32(19 * .125)}
