"""Tiny scalar references for an observed-state Klein MLP, no model imports."""
from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np
import pytest

equations = runpy.run_path(str(Path(__file__).parents[1] / "qualification/flux_mlp_equations.py"))


def fixture(bias=False):
    x = np.array([[[.5, -1.], [2., .25], [-.7, .9]]], dtype=np.float32)
    r = np.array([[[3., 4.], [5., 6.], [7., 8.]]], dtype=np.float32)
    gate = np.array([[[.25, -.5]]], dtype=np.float32)
    weights = {"linear_in": np.array([[.2, .4], [-.8, .1], [.6, -.2], [.3, -.7], [.1, .8], [-.4, .2]], dtype=np.float32),
               "linear_out": np.array([[.2, -.3, .7], [-.5, .4, .1]], dtype=np.float32)}
    if bias:
        weights.update(bias_in=np.array([.1, -.2, .3, -.4, .5, -.6], dtype=np.float32),
                       bias_out=np.array([.2, -.1], dtype=np.float32))
    for value in weights.values():
        value.flags.writeable = False
    return x, r, gate, weights


def scalar_linear(values, rows, bias):
    return [math.fsum(float(a) * float(b) for a, b in zip(values, row, strict=True))
            + (0. if bias is None else float(bias[index])) for index, row in enumerate(rows)]


@pytest.mark.parametrize("bias", [False, True])
def test_complete_formula_matches_scalar_loops_with_correct_half_order(bias):
    x, r, gate, weights = fixture(bias)
    output = equations["forecast"](x, r, gate, weights, chunk_tokens=2)
    expected = []
    for token in x[0]:
        projected = scalar_linear(token, weights["linear_in"], weights.get("bias_in"))
        a, b = projected[:3], projected[3:]
        gated = [first / (1 + math.exp(-first)) * second for first, second in zip(a, b, strict=True)]
        expected.append(scalar_linear(gated, weights["linear_out"], weights.get("bias_out")))
    np.testing.assert_allclose(output["y"], [expected], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(output["write"], np.asarray(expected)[None] * gate.astype(np.float64), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(output["residual"], r.astype(np.float64) + output["write"], rtol=1e-14, atol=1e-14)
    assert set(output) == {"y", "write", "residual", "summaries"}
    assert all(not output[key].flags.writeable for key in ("y", "write", "residual"))


def test_zero_gate_preserves_residual_and_tokenwise_gate_broadcast_is_explicit():
    x, r, gate, weights = fixture()
    zero = equations["forecast"](x, r, np.zeros_like(gate), weights)
    np.testing.assert_array_equal(zero["residual"], r)
    token_gate = np.broadcast_to(gate, x.shape).copy()
    token_gate[:, 1] = 0.
    output = equations["forecast"](x, r, token_gate, weights)
    np.testing.assert_array_equal(output["residual"][:, 1], r[:, 1])


def test_chunk_boundaries_and_input_mutability_do_not_change_the_formula():
    x, r, gate, weights = fixture()
    copies = [value.copy() for value in (x, r, gate, *weights.values())]
    first, second = (equations["forecast"](x, r, gate, weights, chunk_tokens=size) for size in (1, 3))
    for key in ("y", "write", "residual"):
        np.testing.assert_allclose(first[key], second[key], rtol=1e-14, atol=1e-14)
    for value, before in zip((x, r, gate, *weights.values()), copies, strict=True):
        np.testing.assert_array_equal(value, before)


def test_silu_remains_finite_at_extreme_native_values():
    result = equations["_silu"](np.array([-1000., 0., 1000.]))
    np.testing.assert_array_equal(result, [0., 0., 1000.])


@pytest.mark.parametrize("case", ["half", "readonly", "gate", "chunk", "nonfinite", "bias"])
def test_invalid_geometry_weights_or_observed_values_refuse_the_forecast(case):
    x, r, gate, weights = fixture()
    kwargs = {}
    if case == "half":
        weights["linear_in"] = weights["linear_in"][:-1]
    elif case == "readonly":
        weights["linear_out"] = weights["linear_out"].copy()
    elif case == "gate":
        gate = np.zeros((1, 2, 2), dtype=np.float32)
    elif case == "chunk":
        kwargs["chunk_tokens"] = 33
    elif case == "nonfinite":
        x[0, 0, 0] = np.nan
    elif case == "bias":
        weights["bias_in"] = np.zeros(5, dtype=np.float32)
        weights["bias_in"].flags.writeable = False
    with pytest.raises(ValueError):
        equations["forecast"](x, r, gate, weights, **kwargs)
