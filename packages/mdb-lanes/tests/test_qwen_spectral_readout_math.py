"""Small independent scalar/dense references for source-only NumPy equations."""
from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np
import pytest

M = runpy.run_path(str(Path(__file__).parents[1] / "qualification/qwen_spectral_readout_math.py"))


def factors():
    rng = np.random.default_rng(12)
    weights = dict(postnorm=np.array([1., .8, 1.1, .9]), gate=rng.normal(size=(7, 4)), up=rng.normal(size=(7, 4)),
        down=rng.normal(size=(4, 7)), finalnorm=np.array([.9, 1.1, .7, 1.2]))
    return M["prepare"](weights), np.array([.2, -.5, .7, .3])


def dense_operator(weights, consumer):
    gamma = weights["postnorm"]; gate = weights["gate"]*gamma; up = weights["up"]*gamma
    downstream = weights["down"].T @ consumer
    return sum(.25 * coefficient * (np.outer(g, u)+np.outer(u, g)) for coefficient, g, u in zip(downstream, gate, up, strict=True))


def test_matrix_free_action_and_full_lanczos_agree_with_independent_dense_matrix():
    weights, consumer = factors(); dense = dense_operator(weights, consumer)
    action = M["even_action"](weights, consumer)
    direction = np.array([.2, -.3, .5, .8])
    np.testing.assert_allclose(action(direction), dense @ direction, atol=1e-14)
    search = M["lanczos"](action, 4, steps=4, seed=7)
    expected = np.linalg.eigvalsh(dense)
    assert search["complete_space_covered"] and search["searched_dimension"] == 4
    for name, index in (("plus", -1), ("minus", 0)):
        result = search["extremes"][name]
        assert result["rayleigh_value"] == pytest.approx(expected[index], abs=1e-12)
        assert result["eigenaction_residual"] < 1e-12
        assert result["direction"][result["sign_pivot"]] >= 0 and not result["direction"].flags.writeable
    partial = M["lanczos"](action, 4, steps=2, seed=7)
    assert not partial["complete_space_covered"] and "Strongest found" in partial["scope"]


def test_original_order_antipodes_match_even_bilinear_identity_and_input_cast():
    weights, consumer = factors(); action = M["even_action"](weights, consumer)
    direction = M["lanczos"](action, 4, steps=4)["extremes"]["plus"]["direction"]
    result = M["antipodal"](direction, 2.3, weights, .03, consumer)
    assert result["positive_input"].dtype == result["write"].dtype == np.float32
    np.testing.assert_array_equal(result["negative_input"], -result["positive_input"])
    np.testing.assert_allclose(result["even"], result["even_bilinear"], atol=2e-15)
    np.testing.assert_allclose(result["positive"]["delta"], result["even"]+result["odd"], atol=2e-15)
    x = result["positive_input"].astype(np.float64)
    expected = float(x @ dense_operator(weights, consumer) @ x) / (float(np.mean(x*x))+.03)
    assert result["consumer_even"] == pytest.approx(expected, abs=2e-14)
    assert result["parity_error"]["max_abs"] < 2e-15
    assert result["realized_raw_norm"] == float(np.linalg.norm(x))


def test_full_grid_planes_and_rms_logits_match_scalar_all_vocabulary_equations():
    rng = np.random.default_rng(3)
    head = rng.normal(size=(6, 4)).astype(np.float32); head.flags.writeable = False
    base, gamma = np.array([.8, -.3, .5, .4]), np.array([.9, 1.1, .7, 1.2])
    writes = dict(plus=np.array([.2, -.1, .7, .1], dtype=np.float32), minus=np.array([-.6, .5, -.2, .3], dtype=np.float32))
    forecast = M["grid_forecast"](base, writes, gamma, head, .01, chunk_rows=2)
    assert len(forecast["rows"]) == 25 and forecast["coefficients"].shape == (6, 3)
    for row in forecast["rows"].values():
        x = base + row["alpha"]*writes["plus"].astype(np.float64) + row["beta"]*writes["minus"].astype(np.float64)
        denominator = math.sqrt(math.fsum(float(value)**2 for value in x)/4+.01)
        expected = [math.fsum(float(w)*float(g)*float(value) for w, g, value in zip(weights, gamma, x, strict=True))/denominator for weights in head]
        np.testing.assert_allclose(row["logits"], expected, rtol=1e-14, atol=1e-14)
        assert row["greedy_token"] == int(np.argmax(expected))
        winner = row["coefficient_greedy_token"]; a, b, c = forecast["coefficients"].T
        assert np.all((b-b[winner])*row["alpha"]+(c-c[winner])*row["beta"] <= a[winner]-a+1e-14)
    assert forecast["rows"]["0,0"]["response"]["max_abs"] == 0
    assert "all-vocabulary halfspace" in forecast["regions"]["representation"]


def test_coincident_rows_and_zero_carrier_keep_every_co_maximizer_and_low_id():
    head = np.array([[1., 0.], [1., 0.], [-1., 0.]], dtype=np.float32); head.flags.writeable = False
    writes = dict(plus=np.array([-1., 0.], dtype=np.float32), minus=np.array([0., -1.], dtype=np.float32))
    result = M["grid_forecast"](np.array([1., 1.]), writes, np.ones(2), head, .01)
    assert result["rows"]["0,0"]["co_maximizers"] == [0, 1]
    assert result["rows"]["1,1"]["co_maximizers"] == [0, 1, 2] and result["rows"]["1,1"]["greedy_token"] == 0
    assert result["rows"]["1,1"]["normalizer"] == .1


def test_source_arrays_are_immutable_copies_and_unqualified_geometry_is_rejected():
    weights, consumer = factors()
    assert all(not value.flags.writeable for value in weights.values())
    with pytest.raises(ValueError, match="exact bias-free"):
        M["prepare"]({**weights, "gate_bias": np.zeros(7)})
    with pytest.raises(ValueError, match="unit source"):
        M["antipodal"](np.ones(4), 2., weights, .01, consumer)
    with pytest.raises(ValueError, match="finite original"):
        M["even_action"](weights, consumer)(np.array([0., 0., 0., np.nan]))
    with pytest.raises(ValueError, match="read-only FP32"):
        M["grid_forecast"](np.ones(4), dict(plus=np.ones(4, dtype=np.float32), minus=np.ones(4, dtype=np.float32)), np.ones(4), np.ones((6, 4), dtype=np.float32), .01)


def test_effective_dimension_and_own_geometry_selection_do_not_invent_two_dimensions():
    parallel = dict(plus=np.array([1., 0.], dtype=np.float32), minus=np.array([-2., 0.], dtype=np.float32))
    geometry = M["payload_geometry"](parallel)
    assert geometry["numeric_rank"] == 1 and geometry["effective_dimension"] == pytest.approx(1.)
    head = np.array([[1., 0.], [-1., 0.], [0., 1.]], dtype=np.float32); head.flags.writeable = False
    forecast = M["grid_forecast"](np.array([.3, .2]), parallel, np.ones(2), head, .01)
    selection = M["select_futures"](forecast)
    assert selection["rows"][0] == "0,0" and len(set(selection["rows"])) == 4
    assert len(selection["rows"]) <= 4 and selection["predicted_regions"] >= 2
    assert any(row["reason"] == "distinct-own-predicted-winner-region" for row in selection["accounts"])
    # Outcomes are not inputs; irrelevant external result keys cannot alter selection.
    assert M["select_futures"]({**forecast, "actual_targets": "deliberately unavailable"}) == selection
