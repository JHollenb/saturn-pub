"""Synthetic descriptor/transaction mechanics; no actual campaign inputs."""
import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "qualification"


@pytest.fixture
def control():
    return runpy.run_path(str(HERE / "verify_flux_latent_direction_controls.py"))


def program():
    rows = []
    for position in range(112):
        local = position % 28
        fields = (["image", "text"] + ["context." + str(i) for i in range(6)] if local == 0 else
            ["image", "text"] if local <= 25 else ["noise"] if local == 26 else ["latent"])
        rows.append({"logical_step": position // 28, "local_index": local,
            "outputs": {name: {"descriptor": {"shape": [1, 2], "dtype": "float32", "bf16_le_sha256": str(position) + name},
                "archive": None} for name in fields}})
    return {"transitions": rows, "terminal_latent": rows[-1]["outputs"]["latent"]}


def test_complete_84_control_retains_all_180_checks(control):
    a = program()["transitions"][28:]
    result = control["compare"](a, copy.deepcopy(a), name="synthetic", expected_rows=84)
    assert result["fields_compared"] == 180
    assert result["field_misses"] == result["clock_misses"] == 0
    assert len(result["rows"]) == 84


def test_misses_missing_fields_and_clock_are_preserved_not_gated(control):
    a = program()["transitions"][28:]
    b = copy.deepcopy(a)
    b[3]["outputs"]["image"]["descriptor"]["shape"] = [2, 1]
    del b[4]["outputs"]["text"]
    b[5]["logical_step"] = 99
    result = control["compare"](a, b, name="synthetic-mismatch", expected_rows=84)
    assert result["fields_compared"] == 180 and result["field_misses"] == 2 and result["clock_misses"] == 1
    assert result["rows"][4]["fields"][1]["right"] is None


def test_coverage_is_a_mechanics_contract(control):
    with pytest.raises(ValueError, match="coverage"):
        control["compare"]([], [], name="missing", expected_rows=84)


def test_metadata_decode_does_not_materialize_arrays(control):
    encoded = {"kind": "mapping", "items": {"descriptor": {"kind": "mapping", "items": {
        "shape": {"kind": "list", "items": [{"kind": "scalar", "value": 2}]}}},
        "array": {"kind": "array", "path": "MUST-NOT-BE-OPENED.npy", "shape": [2], "dtype": "float32"}}}
    result = control["plain_metadata"](encoded)
    assert result["descriptor"] == {"shape": [2]}
    assert result["array"]["unmaterialized_forecast_array"]["path"] == "MUST-NOT-BE-OPENED.npy"


def test_real_public_transaction_verifier_and_descriptor_only_control(control, tmp_path):
    from mdb.forecast_transaction import ForecastTransaction

    data = tmp_path / "result"
    txroot = data / "transactions/c0-o1"
    p = program()
    with ForecastTransaction(txroot, model_identity={"model": "tiny"}, input_identity={"input": "tiny"},
            source_identity={"source": "synthetic"}, frontier={"kind": "raw-input", "clock": 0}) as tx:
        tx.predict("program", lambda _: {"predictions": {**p, "sample_array": np.array([1., 2.], np.float32)}})
        seal = tx.seal()
    parent = {"data": data, "summary": {"seals": {"c0-o1": seal}}, "request": {"config": {"transaction_max_bytes": 1 << 20}}}
    decoded, binding = control["qualified_program"](parent, "c0-o1")
    assert binding["public_transaction_verification"]["verified"] is True
    assert decoded["transitions"] == p["transitions"]
    assert decoded["sample_array"]["unmaterialized_forecast_array"]["kind"] == "array"
    assert json.loads((txroot / "forecasts/program.json").read_text())["name"] == "program"
