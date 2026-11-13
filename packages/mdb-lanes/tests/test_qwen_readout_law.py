"""Full readout law checked against scalar equations and brute-force lines."""
from __future__ import annotations

import math
import runpy
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

law = runpy.run_path(str(Path(__file__).parents[1] / "qualification/qwen_readout_law.py"))


def fixture():
    head = np.array([[.2, -.4, .9], [-.7, .5, .3], [.6, .1, -.2], [.1, .1, .1]], dtype=np.float32)
    head.flags.writeable = False
    return np.array([[[1.2, -.7, .3]]]), np.array([[[-.2, .9, .1]]]), np.array([.9, 1.1, .8]), head


def test_affine_law_matches_direct_scalar_rms_and_head_rows_at_all_doses():
    cut, delta, gamma, head = fixture()
    result = law["full_readout"](cut, delta, gamma, head, .03, doses=(0., .25, .5, 1.), include_logits=True, chunk_rows=2)
    for key, dose in (("0", 0.), ("0.25", .25), ("0.5", .5), ("1", 1.)):
        carrier = [float(r) + dose * float(m) for r, m in zip(cut.flat, delta.flat, strict=True)]
        denominator = math.sqrt(math.fsum(x * x for x in carrier) / len(carrier) + .03)
        normalized = [x * float(g) / denominator for x, g in zip(carrier, gamma, strict=True)]
        expected = [math.fsum(float(w) * value for w, value in zip(row, normalized, strict=True)) for row in head]
        actual = result["doses"][key]
        np.testing.assert_allclose(actual["logits"].reshape(-1), expected, rtol=1e-14, atol=1e-14)
        assert actual["normalizer"] == pytest.approx(denominator)
        order = sorted(range(len(expected)), key=lambda token: (-expected[token], token))
        assert actual["token_id"] == order[0] and actual["runner_up_token_id"] == order[1]
        assert actual["top2gap"] == pytest.approx(expected[order[0]] - expected[order[1]])
        assert not actual["logits"].flags.writeable
    assert not result["a"].flags.writeable and not result["b"].flags.writeable
    assert "head" not in result


def test_parallel_lines_and_zero_width_low_id_tie_are_preserved():
    envelope = law["upper_envelope"]([.5, 1., 0., -2., 1.], [1., 0., 2., 0., 0.])
    assert [row["token_id"] for row in envelope["segments"]] == [1, 2]
    crossing = next(row for row in envelope["breakpoints"] if row["dose_exact"] == "1/2")
    assert crossing["token_id"] == 0
    assert crossing["tied_token_ids"] == [0, 1, 2, 4]


def test_envelope_matches_brute_force_exact_lines_at_dense_points_and_breakpoints():
    rng = np.random.default_rng(23)
    a, b = rng.integers(-10, 11, 70).astype(float), rng.integers(-10, 11, 70).astype(float)
    envelope = law["upper_envelope"](a, b)
    for segment in envelope["segments"]:
        midpoint = (segment["left"] + segment["right"]) / 2
        assert segment["token_id"] == int(np.argmax(a + midpoint * b))
    for point in envelope["breakpoints"]:
        dose = Fraction(point["dose_exact"])
        heights = [Fraction(float(x)) + dose * Fraction(float(y)) for x, y in zip(a, b, strict=True)]
        expected = [token for token, height in enumerate(heights) if height == max(heights)]
        assert point["tied_token_ids"] == expected and point["token_id"] == min(expected)
    for dose in np.linspace(0, 1, 101):
        point = next((row for row in envelope["breakpoints"] if row["dose"] == dose), None)
        if point is not None:
            continue
        segment = next(row for row in envelope["segments"] if row["left"] < dose < row["right"])
        assert segment["token_id"] == int(np.argmax(a + dose * b))


def test_epsilon_keeps_zero_carrier_finite_and_lowest_token_wins_all_ties():
    cut, delta, gamma, head = fixture()
    delta = -cut
    result = law["full_readout"](cut, delta, gamma, head, .01, doses=(1.,), include_logits=True)
    np.testing.assert_allclose(result["doses"]["1"]["logits"], 0., atol=1e-14)
    assert result["doses"]["1"]["normalizer"] == .1
    assert result["doses"]["1"]["token_id"] == 0


def test_chunk_geometry_does_not_change_scalar_coefficients():
    cut, delta, gamma, head = fixture()
    first = law["full_readout"](cut, delta, gamma, head, .01, chunk_rows=1)
    second = law["full_readout"](cut, delta, gamma, head, .01, chunk_rows=4)
    np.testing.assert_allclose(first["a"], second["a"], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(first["b"], second["b"], rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("case", ["writable", "head64", "bias", "geometry", "eps", "nonfinite", "chunks"])
def test_invalid_native_matrix_or_equation_contract_is_rejected(case):
    cut, delta, gamma, head = fixture()
    kwargs, eps = {}, .01
    if case == "writable":
        head = head.copy()
    elif case == "head64":
        head = head.astype(np.float64)
        head.flags.writeable = False
    elif case == "bias":
        kwargs["head_bias"] = np.zeros(4)
    elif case == "geometry":
        delta = np.zeros((1, 2, 3))
    elif case == "eps":
        eps = 0.
    elif case == "nonfinite":
        cut[0, 0, 0] = np.nan
    elif case == "chunks":
        kwargs["chunk_rows"] = 4097
    with pytest.raises(ValueError):
        law["full_readout"](cut, delta, gamma, head, eps, **kwargs)
