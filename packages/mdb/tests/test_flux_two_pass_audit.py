"""Independent two-pass recurrence and prefix-derived state provenance."""
from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

HERE = Path(__file__).parents[1]
A = runpy.run_path(str(HERE / "qualification/analyze_flux_two_pass.py"))


@pytest.fixture(scope="module")
def specimen(tmp_path_factory):
    fixture = runpy.run_path(str(HERE / "tests/test_flux_two_pass_campaign.py"))
    folder = tmp_path_factory.mktemp("independent-two-pass")
    with pytest.MonkeyPatch.context() as patch:
        fixture["test_two_complete_passes_predicted_image_first"](folder, patch, None)
    data = folder / "result"
    e = A["BASE"]["SavedEvidence"](data)
    tree = A["H"]["NumericalTree"](data / "transaction")
    encoded = json.loads((data / "transaction/forecasts/two_pass.json").read_text())
    forecast = tree.decode(encoded["predictions"])
    binding = json.loads((data / "binding.json").read_text())
    from mdb.tensor_archive import load_tensor_archive
    inputs = {key: load_tensor_archive(data, value) for key, value in binding["inputs"].items()}
    return data, e, binding, forecast, inputs


def comparisons(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "sealed_minus_independent":
                yield child
            else:
                yield from comparisons(child)
    elif isinstance(value, list):
        for child in value:
            yield from comparisons(child)


def test_both_independent_passes_consume_their_own_latent_and_context(specimen):
    data, e, binding, forecast, inputs = specimen
    segment = json.loads((data / "native-segment.json").read_text())
    independent, sealed = inputs["latent"], inputs["latent"]
    checked = 0
    for index, prediction in enumerate(forecast["passes"]):
        schedule = {key: e.array(binding["pass_schedules"][str(index + 2)][key])
            for key in ("timestep", "sigma", "sigma_next")}
        report, independent = A["audit_pass"](e, binding, prediction, schedule, independent, segment, index,
            inputs["conditioning"], inputs["img_ids"], inputs["txt_ids"], sealed)
        fields = list(comparisons(report)) + list(report["project"]["context"].values()) + report["project"]["rope"]
        assert len(fields) == 40
        assert all(value["bitwise_exact"] for value in fields)
        checked += sum(value["sample_count"] for value in fields)
        sealed = prediction["commit"]["latent"]
    assert checked == 3456
    np.testing.assert_array_equal(independent.view(np.uint32), sealed.view(np.uint32))


def test_independent_numeric_disagreement_is_retained_not_a_mechanics_abort(specimen):
    data, e, binding, forecast, inputs = specimen
    segment = json.loads((data / "native-segment.json").read_text())
    schedule = {key: e.array(binding["pass_schedules"]["2"][key]) for key in ("timestep", "sigma", "sigma_next")}
    altered = A["MATH"]["exact_bf16"](inputs["latent"] * np.float32(.5))
    report, _ = A["audit_pass"](e, binding, forecast["passes"][0], schedule, altered, segment, 0,
        inputs["conditioning"], inputs["img_ids"], inputs["txt_ids"], inputs["latent"])
    assert report["commit"]["sealed_minus_independent"]["bitwise_mismatch_count"] > 0


def test_derived_terminal_payload_needs_no_future_native_operand(specimen):
    data, e, binding, forecast, _ = specimen
    record = json.loads((data / "derived-terminal.json").read_text())
    row = json.loads((data / "rows/full.predicted.json").read_text())
    report = A["check_derived_terminal"](e, record, binding["input_cut"], row["acts"][0],
        forecast["passes"][1]["commit"]["latent"], record["source_hashes"]["derivation_helper"])
    assert report["all13_fields_prefix_derived"] is True


@pytest.mark.parametrize("field", ["source_preconditions", "scaffold_point"])
def test_resealed_wrong_source_or_projection_address_is_rejected(specimen, field):
    data, e, binding, forecast, _ = specimen
    record = copy.deepcopy(json.loads((data / "derived-terminal.json").read_text()))
    if field == "source_preconditions":
        record[field]["conditioning"] = "0" * 64
    else:
        record[field]["index"] = 0
    record["fingerprint"] = A["BASE"]["canonical_hash"]({key: value for key, value in record.items() if key != "fingerprint"})
    row = json.loads((data / "rows/full.predicted.json").read_text())
    with pytest.raises(ValueError):
        A["check_derived_terminal"](e, record, binding["input_cut"], row["acts"][0],
            forecast["passes"][1]["commit"]["latent"], record["source_hashes"]["derivation_helper"])


def test_tree_comparison_accepts_readonly_maps_and_preserves_signed_zero():
    value = np.array([0., 1.], np.float32)
    assert A["tree_exact"](MappingProxyType({"value": value}), {"value": value.copy()})
    changed = value.copy()
    changed[0] = -0.
    assert not A["tree_exact"]({"value": value}, {"value": changed})
