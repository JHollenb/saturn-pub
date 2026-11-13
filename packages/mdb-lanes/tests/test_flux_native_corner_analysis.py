"""Signed native-response accounting, including cancellation and undefined ratios."""
import runpy
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

A = runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification/analyze_flux_native_corner.py"))


def test_immutable_forecast_container_normalization_preserves_descriptor_values():
    frozen = MappingProxyType({"descriptor": MappingProxyType({"shape": (1, 1024, 128), "sha256": "original"}),
        "witness": (np.float32(1.25), np.int64(7))})
    assert A["json_value"](frozen) == {"descriptor": {"shape": [1, 1024, 128], "sha256": "original"}, "witness": [1.25, 7]}


def test_response_error_uses_the_native_response_denominator():
    # DG=(2,0), DF=(1,0); endpoint signal scale must not hide this100% response error.
    m = np.array([1000., 1000.])
    result = A["response_account"](m + [2, 0], m, m + [1, 0], m, .5)
    assert result["error_DG_minus_DF"]["relative_rms"] == 1
    assert result["cosine_DG_DF"] == 1
    assert result["native_finite_response_ratio"] == pytest.approx(np.sqrt(.5) / .5)


def test_zero_native_response_keeps_absolute_error_and_undefined_ratios():
    zero = np.zeros(3)
    result = A["response_account"](np.ones(3), zero, zero, zero, 0)
    assert result["error_DG_minus_DF"]["rms"] == 1
    assert result["error_DG_minus_DF"]["relative_rms"] is None
    assert result["cosine_DG_DF"] is None
    assert result["native_finite_response_ratio"] is None


def test_opposing_responses_remain_a_missed_alignment_prediction():
    zero = np.zeros(2)
    result = A["response_account"](np.array([1., -1.]), zero, np.array([-1., 1.]), zero, 1)
    assert result["cosine_DG_DF"] == -1
    assert result["dot_DG_DF"] == -2


def test_three_term_identity_keeps_every_signed_cross_term():
    # A=2, B=-3, C=1 cancel completely; summing individual energies gives14, not0.
    result = A["continuation_account"](np.array([0.]), np.array([-2.]), np.array([1.]), np.array([0.]))
    assert len(result["pairwise_cross_terms"]) == 3
    assert sorted(v["twice_dot"] for v in result["pairwise_cross_terms"].values()) == [-12, -6, 4]
    assert result["P_minus_N_old"]["rms"] == 0
    assert result["squared_norm_identity"]["residual"] == 0
    assert result["vector_identity_max_abs"] == 0
    assert result["P_minus_M_current"]["rms"] == 1


def test_rgb_uint8_is_widened_before_subtraction():
    result = A["continuation_account"](np.array([0], dtype=np.uint8), np.array([255], dtype=np.uint8),
        np.array([255], dtype=np.uint8), np.array([255], dtype=np.uint8))
    assert result["terms"]["implementation_P_minus_R"]["signed_mean"] == -255
    assert result["P_minus_N_old"]["rms"] == 255


@pytest.mark.parametrize("values", [np.array([np.nan]), np.array([])])
def test_nonfinite_or_empty_instrument_operands_are_refused(values):
    with pytest.raises(ValueError, match="finite"):
        A["stats"](values)
