"""Framework-free algebra and graph-candidate mechanics for physical traces."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mdb.forward_algebra import (
    AffineScan,
    ForwardAlgebraError,
    acquire_symbol_candidates,
    apply_affine_scan,
    bilinear_factorial_identity,
    bilinear_interaction_remainder,
    bind_empirical_observation,
    compose_affine_scan,
    factorial_contrast,
    factorial_contrasts,
    finite_difference,
    log_clock_residual,
    mobius_corner_coefficients,
    multilinear_predict,
    structural_identities,
)
from mdb.forward_readable import ForwardReadableProgram
from mdb.forward_symbols import FORWARD_SYMBOLS_SCHEMA, seal_trace


def _value(index: int, *, origin: str = "operation-output", producer: int | None = None):
    return {
        "shape": [1, 1], "dtype": "torch.float32", "sha256": f"{index + 1:064x}",
        "origin": origin, "producer": producer, "preview": [],
    }


def _event(sequence: int, operator: str, args: list[object], output: str):
    refs: list[str] = []

    def visit(value):
        if isinstance(value, dict) and set(value) == {"$value"}:
            if value["$value"] not in refs:
                refs.append(value["$value"])
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for argument in args:
        visit(argument)
    return {
        "sequence": sequence, "operator": operator, "module_path": "fixture.native",
        "inputs": refs, "outputs": [output],
        "arguments": {"args": {"$tuple": args}, "kwargs": {}},
        "result": {"$value": output},
        "clock": {"forward_call": 0, "operation": sequence},
    }


def _trace(*, add_alpha: int = 1):
    producers = {
        5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 10: 5, 11: 6, 12: 7,
        15: 8, 16: 9, 17: 10, 18: 11, 19: 12, 20: 13,
    }
    values = {}
    for index in range(21):
        if index in producers:
            origin, producer = "operation-output", producers[index]
        else:
            origin, producer = ("parameter" if index == 14 else "input"), None
        values[f"v{index:03}"] = _value(index, origin=origin, producer=producer)

    add_args = [{"$value": "v008"}, {"$value": "v002"}]
    if add_alpha != 1:
        add_args.append(add_alpha)
    events = [
        _event(0, "aten.mm.default", [{"$value": "v000"}, {"$value": "v001"}], "v005"),
        _event(1, "aten.softmax.int", [{"$value": "v005"}, -1], "v006"),
        _event(2, "aten.mm.default", [{"$value": "v006"}, {"$value": "v002"}], "v007"),
        _event(3, "aten.mul.Tensor", [{"$value": "v000"}, {"$value": "v001"}], "v008"),
        _event(4, "aten.add.Tensor", add_args, "v009"),
        _event(5, "aten.mul.Tensor", [{"$value": "v003"}, {"$value": "v004"}], "v010"),
        _event(6, "aten.log.default", [{"$value": "v010"}], "v011"),
        _event(7, "aten.sum.dim_IntList", [{"$value": "v008"}, [0], False], "v012"),
        _event(8, "aten.softplus.default", [{"$value": "v013"}], "v015"),
        _event(9, "aten.unsqueeze.default", [{"$value": "v015"}, 0], "v016"),
        _event(10, "aten.exp.default", [{"$value": "v014"}], "v017"),
        _event(11, "aten.neg.default", [{"$value": "v017"}], "v018"),
        _event(12, "aten.mul.Tensor", [{"$value": "v016"}, {"$value": "v018"}], "v019"),
        _event(13, "aten.exp.default", [{"$value": "v019"}], "v020"),
    ]
    return seal_trace({
        "schema": FORWARD_SYMBOLS_SCHEMA,
        "model_identity": {"model_id": "sealed-algebra-fixture"},
        "execution": {"backend": "metadata-only"},
        "input_identity": {"case": "graph-signature-fixture"},
        "values": values,
        "events": events,
        "outputs": {
            "read": {"$value": "v007"},
            "sum": {"$value": "v009"},
            "log_product": {"$value": "v011"},
            "contraction": {"$value": "v012"},
            "decay_expression": {"$value": "v020"},
        },
        "capture_status": "complete",
        "semantic_labels_supplied": False,
        "metadata": {"purpose": "tiny algebra signature fixture"},
    })


def _xy_corners(function):
    return {
        (0, 0): function(0, 0),
        (1, 0): function(1, 0),
        (0, 1): function(0, 1),
        (1, 1): function(1, 1),
    }


def test_finite_difference_and_factorial_contrasts_are_sampled_effects():
    assert finite_difference((1, [2, 4]), (4, [3, 9])) == (3, [1, 5])

    cells = _xy_corners(lambda x, y: 1 + 2 * x + 3 * y + 4 * x * y)
    contrast = factorial_contrast(cells, factors=("x", "y"))
    all_contrasts = factorial_contrasts(cells, factors=("x", "y"))
    assert contrast["value"] == 4
    assert contrast["provenance"] == "empirical-observation"
    assert "does not establish continuous multilinearity" in contrast["interpretation_limit"]
    assert [row["value"] for row in all_contrasts["contrasts"]] == [4, 5, 4]
    assert all_contrasts["corner_mobius"]["terms"] == [
        {"factors": [], "coefficient": 1},
        {"factors": ["x"], "coefficient": 2},
        {"factors": ["y"], "coefficient": 3},
        {"factors": ["x", "y"], "coefficient": 4},
    ]
    json.dumps(all_contrasts)


def test_same_corner_coefficients_do_not_establish_the_midpoint_behavior():
    bilinear = _xy_corners(lambda x, y: x * y)
    corner_agreeing_nonlinear = _xy_corners(lambda x, y: x * y + 4 * x * (1 - x))
    first = mobius_corner_coefficients(bilinear, factors=("x", "y"))
    second = mobius_corner_coefficients(corner_agreeing_nonlinear, factors=("x", "y"))
    first_coefficients = {
        tuple(term["factors"]): term["coefficient"] for term in first["terms"]
    }
    second_coefficients = {
        tuple(term["factors"]): term["coefficient"] for term in second["terms"]
    }
    assert first_coefficients == second_coefficients
    interpolated = multilinear_predict(
        first_coefficients, factors=("x", "y"), coordinates={"x": 0.5, "y": 0.5}
    )
    assert interpolated == 0.25
    assert multilinear_predict(
        first, factors=("x", "y"), coordinates={"x": 0.5, "y": 0.5}
    ) == interpolated
    assert bilinear[0, 0] == corner_agreeing_nonlinear[0, 0]
    assert bilinear[1, 0] == corner_agreeing_nonlinear[1, 0]
    assert bilinear[0, 1] == corner_agreeing_nonlinear[0, 1]
    assert bilinear[1, 1] == corner_agreeing_nonlinear[1, 1]
    assert 0.5 * 0.5 != 0.5 * 0.5 + 4 * 0.5 * (1 - 0.5)
    assert "do not establish that the underlying function is multilinear" in first[
        "interpretation_limit"
    ]


def test_three_factor_corners_produce_all_mobius_terms_and_interpolation():
    factors = ("h", "c", "g")

    def function(h, c, g):
        return 3 + h + 2 * c + 4 * g + 5 * h * c + 6 * h * g + 7 * c * g + 8 * h * c * g

    cells = {
        (h, c, g): function(h, c, g)
        for h in (0, 1) for c in (0, 1) for g in (0, 1)
    }
    contrasts = factorial_contrasts(cells, factors=factors)
    mobius = contrasts["corner_mobius"]
    assert len(contrasts["contrasts"]) == 7
    assert [(term["factors"], term["coefficient"]) for term in mobius["terms"]] == [
        ([], 3), (["h"], 1), (["c"], 2), (["g"], 4),
        (["h", "c"], 5), (["h", "g"], 6), (["c", "g"], 7),
        (["h", "c", "g"], 8),
    ]
    assert multilinear_predict(
        mobius, factors=factors, coordinates={"h": 0.5, "c": 0.5, "g": 0.5}
    ) == 12


def test_factorial_grid_must_be_complete_and_shapes_must_match():
    with pytest.raises(ForwardAlgebraError, match="complete factorial grid"):
        factorial_contrast({(0, 0): 1}, factors=("x", "y"))
    with pytest.raises(ForwardAlgebraError, match="matching lengths"):
        factorial_contrast(
            {(0,): [1], (1,): [2, 3]}, factors=("x",)
        )


def test_affine_scan_composition_matches_sequential_application_and_is_associative():
    first = AffineScan(decay=5, write=7)
    second = AffineScan(decay=2, write=3)
    third = AffineScan(decay=4, write=-1)
    composed = compose_affine_scan(second, first)
    assert composed == AffineScan(decay=10, write=17)
    assert apply_affine_scan(composed, 11) == apply_affine_scan(
        second, apply_affine_scan(first, 11)
    ) == 127
    assert compose_affine_scan(third, composed) == compose_affine_scan(
        compose_affine_scan(third, second), first
    )

    vector = compose_affine_scan(
        AffineScan(decay=[2, 3], write=[4, 5]),
        AffineScan(decay=[6, 7], write=[8, 9]),
    )
    assert vector == AffineScan(decay=[12, 21], write=[20, 32])


def test_log_character_and_bilinear_interaction_retain_remainders():
    assert log_clock_residual(2, 3, 6) == pytest.approx(0)
    with pytest.raises(ForwardAlgebraError, match="positive"):
        log_clock_residual(0, 3)

    result = bilinear_factorial_identity(
        [1, 2], [2, 0], [-1, 3], [2, 1], [[1, 2], [3, 4]]
    )
    assert result["mixed_difference"] == result["bilinear_cross_term"]
    assert result["remainder"] == 0
    assert result["provenance"] == "structural-identity"

    observed = {(0, 0): [0], (1, 0): [1], (0, 1): [2], (1, 1): [11]}
    remainder = bilinear_interaction_remainder(observed, [3])
    assert remainder["observed_interaction"] == [8]
    assert remainder["nonlinear_remainder"] == [5]
    assert remainder["provenance"] == "empirical-observation"
    assert "omitted paths" in remainder["remainder_scope"]


def test_graph_candidates_are_exact_role_neutral_and_bindings_stay_separate():
    program = ForwardReadableProgram(_trace())
    result = acquire_symbol_candidates(program)
    assert result["trace_fingerprint"] == program.fingerprint
    assert [item["provenance"] for item in result["structural_identities"]] == [
        "structural-identity", "structural-identity", "structural-identity",
    ]
    assert result["empirical_bindings"] == []

    candidates = {item["kind"]: item for item in result["graph_candidates"]}
    assert set(candidates) == {
        "product-plus-addend", "softmax-weighted-bilinear-read", "log-of-product",
        "sum-of-product-contraction", "exp-softplus-negative-exp",
    }
    assert candidates["product-plus-addend"]["affine_views"] == [
        {"coefficient_value": "v000", "variable_value": "v001", "offset_value": "v002"},
        {"coefficient_value": "v001", "variable_value": "v000", "offset_value": "v002"},
    ]
    reader = candidates["softmax-weighted-bilinear-read"]
    assert reader["score"]["input_values"] == ["v000", "v001"]
    assert reader["weights"]["output_value"] == "v006"
    assert reader["read"]["payload_value"] == "v002"
    assert reader["events"] == [0, 1, 2]
    assert candidates["log-of-product"]["factor_values"] == ["v003", "v004"]
    assert candidates["sum-of-product-contraction"]["factor_values"] == ["v000", "v001"]
    assert candidates["exp-softplus-negative-exp"]["events"] == [8, 9, 10, 11, 12, 13]
    assert "positive output" in str(
        candidates["exp-softplus-negative-exp"]["conditional_identity"]["conditions"]
    )
    assert "underflow to 0" in candidates["exp-softplus-negative-exp"][
        "conditional_identity"
    ]["native_numeric_caveat"]

    binding = bind_empirical_observation(
        "affine-scan-compose-v1",
        trace_fingerprint=program.fingerprint,
        scope={"kind": "value", "id": "v009"},
        observed=[5.5, -2],
        expected=[5, -3],
        evidence_refs=["fixture-arm-1"],
    )
    assert binding["residual"] == [0.5, 1]
    assert binding["provenance"] == "empirical-observation"
    assert result["empirical_bindings"] == []  # measurements are returned separately by the binder


def test_native_add_alpha_prevents_unscaled_affine_view():
    program = ForwardReadableProgram(_trace(add_alpha=2))
    product = next(
        item for item in acquire_symbol_candidates(program)["graph_candidates"]
        if item["kind"] == "product-plus-addend"
    )
    assert product["add_alpha"] == 2
    assert product["affine_views"] == []
    assert product["affine_views_omitted_reason"] == "native add alpha changes the operand coefficient"


def test_algebra_module_imports_without_torch_in_a_clean_process():
    package_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, ("src", env.get("PYTHONPATH", ""))))
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import mdb.forward_algebra; assert 'torch' not in sys.modules"],
        cwd=package_root, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_forward_symbols_algebra_cli_emits_machine_readable_candidates(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(_trace()), encoding="utf-8")
    package_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, ("src", env.get("PYTHONPATH", ""))))
    result = subprocess.run(
        [sys.executable, "-m", "mdb.forward_symbols", str(trace_path), "--algebra", "--json"],
        cwd=package_root, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["trace_fingerprint"] == _trace()["fingerprint"]
    assert report["empirical_bindings"] == []
    assert {row["kind"] for row in report["graph_candidates"]} >= {
        "softmax-weighted-bilinear-read", "sum-of-product-contraction",
        "exp-softplus-negative-exp",
    }


def test_identity_catalog_has_only_explicit_formulae():
    identities = structural_identities()
    assert [identity["id"] for identity in identities] == [
        "affine-scan-compose-v1", "positive-log-character-v1", "bilinear-reader-interaction-v1",
    ]
    assert all(identity["provenance"] == "structural-identity" for identity in identities)
    assert "carry" not in str(identities).lower()
