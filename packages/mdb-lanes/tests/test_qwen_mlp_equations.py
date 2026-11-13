"""Tiny scalar-loop references; no model, trained weights or Torch."""
from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np
import pytest

equations = runpy.run_path(str(Path(__file__).parents[1] / "qualification/qwen_mlp_equations.py"))


def fixture():
    return np.array([.7, -1.3]), {
        "postnorm": np.array([1.2, .8]), "gate_proj": np.array([[.2, -.7], [.5, .1], [-.9, .3]]),
        "up_proj": np.array([[.1, .4], [-.8, .5], [.3, -.2]]),
        "down_proj": np.array([[.2, -.4, .9], [-.3, .1, -.7]]),
        "finalnorm": np.array([.9, 1.1]), "selected_lm_head": np.array([[.2, -.1], [-.8, .7], [.4, .9]])}


def scalar_reference(cut, weights, eps, dose):
    def norm(values, scale):
        denominator = math.sqrt(math.fsum(float(x) ** 2 for x in values) / len(values) + eps)
        return [float(value) / denominator * float(weight) for value, weight in zip(values, scale, strict=True)]

    def linear(values, rows):
        return [math.fsum(float(a) * float(b) for a, b in zip(values, row, strict=True)) for row in rows]

    normalized = norm(cut, weights["postnorm"])
    gate, up = linear(normalized, weights["gate_proj"]), linear(normalized, weights["up_proj"])
    gated = [value / (1 + math.exp(-value)) * multiplier for value, multiplier in zip(gate, up, strict=True)]
    delta = linear(gated, weights["down_proj"])
    carrier = [float(value) + dose * change for value, change in zip(cut, delta, strict=True)]
    final = norm(carrier, weights["finalnorm"])
    return normalized, delta, carrier, final, linear(final, weights["selected_lm_head"])


@pytest.mark.parametrize("expanded", [False, True])
def test_full_chain_matches_independent_scalar_loops_and_retains_row_ids(expanded):
    cut, weights = fixture()
    supplied = cut.reshape(1, 1, -1) if expanded else cut
    forecast = equations["forecast"](supplied, weights, .03, candidate_ids=[17, 3, 99])
    assert forecast["candidate_ids"] == [17, 3, 99]
    for key, dose in (("1", 1.), ("0", 0.), ("0.5", .5)):
        normalized, delta, carrier, final, logits = scalar_reference(cut, weights, .03, dose)
        for actual, expected in ((forecast["mlp"]["postnorm"], normalized), (forecast["mlp"]["delta"], delta),
                                 (forecast["doses"][key]["residual"], carrier), (forecast["doses"][key]["final_norm"], final),
                                 (forecast["doses"][key]["logits"], logits)):
            np.testing.assert_allclose(actual.reshape(-1), expected, rtol=1e-14, atol=1e-14)
            assert actual.dtype == np.float64 and not actual.flags.writeable
        assert forecast["doses"][key]["logits"].shape == ((1, 1, 3) if expanded else (3,))
    assert np.array_equal(forecast["doses"]["0"]["residual"], supplied)


def test_silu_is_stable_at_large_magnitude_and_zero():
    result = equations["silu"](np.array([-1000., 0., 1000.]))
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result, [0., 0., 1000.])


def test_prepared_weights_and_published_cut_do_not_alias_callers():
    cut, weights = fixture()
    prepared = equations["prepare_weights"](weights)
    result = equations["forecast"](cut, weights, .01, candidate_ids=[2, 4, 6])
    weights["postnorm"][:] = 100.
    cut[:] = 100.
    np.testing.assert_array_equal(prepared["postnorm"], [1.2, .8])
    np.testing.assert_array_equal(result["cut"], [.7, -1.3])
    assert all(not value.flags.writeable for value in prepared.values())


def test_weight_geometry_and_bias_keys_cannot_silently_change_equations():
    cut, weights = fixture()
    weights["down_proj"] = np.ones((3, 2))
    with pytest.raises(ValueError, match="geometry"):
        equations["forecast"](cut, weights, .01, candidate_ids=[2, 4, 6])
    _, weights = fixture()
    weights["gate_bias"] = np.zeros(3)
    with pytest.raises(ValueError, match="expected exactly"):
        equations["prepare_weights"](weights)


@pytest.mark.parametrize("ids", [[1, 1, 2], [1, 2], [1, -2, 3], [True, 2, 3]])
def test_selected_vocabulary_row_identity_is_explicit_and_unambiguous(ids):
    cut, weights = fixture()
    with pytest.raises(ValueError, match="candidate_ids"):
        equations["forecast"](cut, weights, .01, candidate_ids=ids)


@pytest.mark.parametrize("eps,doses", [(0., (1.,)), (-.1, (1.,)), (float("nan"), (1.,)),
                                       (.01, (1., float("inf"))), (.01, (0., -0.)), (.01, ())])
def test_invalid_norm_or_dose_declarations_refuse_before_forecast(eps, doses):
    cut, weights = fixture()
    with pytest.raises(ValueError):
        equations["forecast"](cut, weights, eps, doses=doses, candidate_ids=[2, 4, 6])


def test_nonfinite_input_and_unintended_token_batch_are_rejected():
    cut, weights = fixture()
    with pytest.raises(ValueError, match="last-token"):
        equations["forecast"](np.zeros((1, 2, 2)), weights, .01, candidate_ids=[2, 4, 6])
    cut[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        equations["forecast"](cut, weights, .01, candidate_ids=[2, 4, 6])
