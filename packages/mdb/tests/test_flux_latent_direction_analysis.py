"""Vector geometry tests, including magnitude agreement with wrong direction."""
import ast
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

A = runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification/analyze_flux_latent_direction_cross.py"))


def test_similar_norm_wrong_direction_is_not_success():
    a = A["account"](np.array([-1., 0]), np.zeros(2), np.array([1., 0]), np.zeros(2), .1)
    assert a["native_response"]["rms"] == a["mathematical_response"]["rms"]
    assert a["prediction_error"]["relative_rms"] == 2
    assert a["cosine"] == -1


def test_zero_native_response_has_undefined_relative_error():
    a = A["account"](np.ones(2), np.zeros(2), np.zeros(2), np.zeros(2), .1)
    assert a["prediction_error"]["relative_rms"] is None
    assert a["cosine"] is None
    assert a["prediction_error"]["rms"] == 1


def test_changed_minus_sham_cancels_common_implementation_error():
    a = A["account"](np.array([4., 5]), np.array([3., 4]), np.array([2., 3]), np.array([1., 2]), .1)
    assert a["prediction_error"]["rms"] == 0
    assert a["cosine"] == 1
    assert a["native_finite_response_ratio"] == 10


def test_factorial_additive_and_interacting_vectors_remain_distinct():
    h, l = A["CELLS"]
    native = {(h, h): np.array([3., 2.]), (h, l): np.array([2., 2.]),
        (l, h): np.array([2., 1.]), (l, l): np.array([1., 1.])}
    a = A["factorial"](native, native)
    assert a["native_interaction"]["rms"] == 0
    assert a["interaction_prediction_error"]["relative_rms"] is None
    edited = {k: v.copy() for k, v in native.items()}
    edited[h, h] += np.array([2., 0.])
    a = A["factorial"](native, edited)
    assert a["native_interaction"]["squared_l2"] == 4
    assert a["interaction_prediction_error"]["relative_rms"] == 1
    assert a["direction_contrasts"][h]["prediction_error"]["relative_rms"] == pytest.approx(2 / 3)


def test_raw_uint8_consumer_values_are_widened_before_subtraction():
    a = A["account"](np.array([255], np.uint8), np.array([0], np.uint8),
        np.array([0], np.uint8), np.array([255], np.uint8), 1)
    assert a["native_response"]["signed_mean"] == -255
    assert a["prediction_error"]["rms"] == 510
    assert a["cosine"] == -1


def test_geometry_rejects_mismatched_or_nonfinite_inputs():
    with pytest.raises(ValueError):
        A["cosine"](np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        A["cosine"](np.array([np.nan]), np.ones(1))


def test_recovery_custody_change_preserves_frozen_numerical_reduction_region():
    path = Path(__file__).resolve().parents[1] / "qualification/analyze_flux_latent_direction_cross.py"
    fn = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "analyze")
    start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "n")
    end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "mkdir")
    def canonical(value):
        if isinstance(value, ast.AST):
            # Python3.12 added empty FunctionDef.type_params; ignore that
            # schema-only field so the source freeze is portable to3.10.
            return {"node": type(value).__name__, **{key: canonical(item) for key, item in ast.iter_fields(value)
                if key != "type_params"}}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value
    region = json.dumps(canonical(ast.Module(body=fn.body[start:end], type_ignores=[])), sort_keys=True, separators=(",", ":"))
    # Original pre-outcome ac09 analyzer region, covering every saved reduction.
    assert hashlib.sha256(region.encode()).hexdigest() == "bdfee4d0ef2837e913c47e72524fc77baa2fcb3ea3b9167e2bde0bf11795a1f7"
