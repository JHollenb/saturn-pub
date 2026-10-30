import json
import subprocess
import sys

import numpy as np
import pytest

from mdb.equation_comparison import EquationCatalog, applicability, compare_equations, main


def test_independent_prediction_and_native_effects_retain_collateral():
    baseline = {"regs": [[-2., -2.]], "length": [[0., 1.]]}
    observed = {"regs": [[3., 1.]], "length": [[0., 1.]]}
    predicted = {"regs": [[3.1, .9]], "length": [[.001, .999]]}
    report = compare_equations(predicted, observed, baseline=baseline,
                               predicted_baseline={"regs": [[-3., -1.]]},
                               decision_rules={"regs": "positive", "length": "argmax"},
                               targets={"regs": [[0, 0]]}, include_residuals=True)
    regs = report["fields"]["regs"]
    assert report["decision_mismatches"] == 0
    np.testing.assert_allclose(regs["residual"]["signed_values"], [[-.1, .1]])
    assert regs["observed_effect"]["target_decision_changes"] == 1
    assert regs["observed_effect"]["collateral_decision_changes"] == 1
    assert regs["predicted_effect"]["signed_delta"]["signed_values"] == [[6.1, 1.9]]
    assert report["terminal_status"] == "unassessed"


def test_no_targets_or_rule_does_not_infer_consumer():
    report = compare_equations({"x": [1.]}, {"x": [2.]})
    assert report["target_scope"] == "unassigned"
    assert report["decision_fields"] == 0
    assert "observed_effect" not in report["fields"]["x"]
    assert "collateral" not in report["fields"]["x"]


def test_components_distinct_from_forecast_residual():
    report = compare_equations({"x": [3.]}, {"x": [3.2]},
                               components={"x": {"skip": [1.], "post": [2.]}},
                               observed_components={"x": {"skip": [1.], "post": [2.]}})
    row = report["fields"]["x"]
    assert row["prediction_component_closure"]["residual"]["max_abs"] == 0
    assert row["observation_component_closure"]["residual"]["max_abs"] == pytest.approx(.2)
    assert row["residual"]["max_abs"] == pytest.approx(.2)


@pytest.mark.parametrize("options", [
    {"observed": {"x": [[1.]]}}, {"observed": {"x": [float("nan")]}},
    {"observed": {"other": [1.]}}, {"decision_rules": {"x": "guess"}},
    {"targets": {"x": [[-1]]}}, {"targets": {"x": [[True]]}},
    {"baseline": {"x": [[1.]]}}, {"max_elements": 0},
    {"components": {"x": {"bad": [[1.]]}}},
])
def test_invalid_geometry_and_policies_fail_integrity(options):
    kwargs = {"predicted": {"x": [1.]}, "observed": {"x": [1.]}, **options}
    with pytest.raises(ValueError):
        compare_equations(**kwargs)


def test_empty_and_scalar_arrays():
    report = compare_equations({"x": np.empty((0, 2)), "y": 2.},
                               {"x": np.empty((0, 2)), "y": -2.}, decision_rules={"y": "positive"})
    assert report["fields"]["x"]["residual"]["elements"] == 0
    assert report["decision_mismatches"] == 1


def test_budget_and_unknown_fields():
    with pytest.raises(ValueError, match="element budget"):
        compare_equations({"x": [1., 2.]}, {"x": [1., 2.]}, max_elements=1)
    with pytest.raises(ValueError, match="unknown"):
        compare_equations({"x": [1.]}, {"x": [1.]}, baseline={"wrong": [1.]})


def binding():
    return {"model_identity": "a" * 64, "execution": {"backend": "cpu"},
            "input_identity": "b" * 64, "frontier": "raw-input", "horizon": 3,
            "source_identity": {"equation.py": "c" * 64}}


def test_immutable_catalog_reload_conditions_and_observed_scope(tmp_path):
    catalog = EquationCatalog(tmp_path)
    report = compare_equations({"x": [1.]}, {"x": [1.]})
    conditions = [{"field": "arch", "op": "eq", "value": "r3"},
                  {"field": "state.rdi", "op": "range", "minimum": 0, "maximum": 100}]
    row = catalog.record(law_id="law_001", route_id="route_002", alias="skip reader",
                         binding=binding(), report=report, conditions=conditions,
                         readers=["regs:7:0"], tested_context={"rdi": 42})
    assert catalog.record(law_id="law_001", route_id="route_002", alias="skip reader",
                          binding=binding(), report=report, conditions=conditions,
                          readers=["regs:7:0"], tested_context={"rdi": 42}) == row
    reopened = EquationCatalog(tmp_path)
    assert reopened.show(row["fingerprint"]) == row
    assert len(reopened.find(law_id="law_001", reader="regs:7:0", text="skip")) == 1
    assert reopened.find(law_id="law_002") == []
    assert len(reopened.find(applicable_to={"arch": "r3", "state": {"rdi": 50}})) == 1
    assert reopened.find(tested_context={"rdi": 50}) == []
    assert reopened.find(applicable_to={"arch": "r3"}) == []
    assert applicability(conditions, {"arch": "r3"})["status"] == "unknown"
    assert applicability([], {})["status"] == "unspecified"
    assert applicability(conditions, {"arch": "other"})["status"] == "does-not-match"


def test_membership_conditions_and_invalid_conditions():
    assert applicability([{"field": "arm", "op": "in", "values": ["native", "zero"]}], {"arm": "zero"})["status"] == "matches"
    with pytest.raises(ValueError):
        applicability([{"field": "x", "op": "less", "value": 1}], {})
    with pytest.raises(ValueError):
        applicability([{"field": "x", "op": "range", "minimum": 2, "maximum": 1}], {})


def test_catalog_rejects_tampering_and_bad_bindings(tmp_path):
    catalog = EquationCatalog(tmp_path)
    report = compare_equations({"x": [1.]}, {"x": [1.]})
    with pytest.raises(ValueError):
        catalog.record(law_id="law", binding={}, report=report)
    row = catalog.record(law_id="law", binding=binding(), report=report)
    path = tmp_path / (row["fingerprint"] + ".json")
    body = json.loads(path.read_text());body["alias"] = "tampered"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="fingerprint"):
        catalog.show(row["fingerprint"])
    with pytest.raises(ValueError, match="changed"):
        catalog.record(law_id="law", binding=binding(), report=report)


def test_framework_free_catalog_import():
    code = "import sys; from mdb.equation_comparison import EquationCatalog; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_compare_cli(tmp_path, capsys):
    p, o = tmp_path / "p.json", tmp_path / "o.json"
    p.write_text('{"x":[1]}');o.write_text('{"x":[2]}')
    assert main(["compare", str(p), str(o)]) == 0
    assert json.loads(capsys.readouterr().out)["max_abs"] == 1
