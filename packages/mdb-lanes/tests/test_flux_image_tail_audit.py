"""Full tiny native capture -> saved-only catalog, plus hostile custody controls."""
from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parents[1]
A = runpy.run_path(str(HERE / "qualification/audit_flux_image_tail_microscope.py"))
S = runpy.run_path(str(HERE / "qualification/stage_flux_image_tail_microscope.py"))


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    # Reuse the source-qualified original tiny block, complete LocalStore entries
    # and original checkpoint recipes; this is not a fake numerical executor.
    fixture = runpy.run_path(str(Path(__file__).with_name("test_flux_image_tail_microscope.py")))
    root = tmp_path_factory.mktemp("image-tail-audit")
    config, _, _ = fixture["fixture"](root)
    config.update(request_sha256="fixture-request", source_job="fixture-parent", scope="Tiny observed-operand qualification.")
    core = fixture["C"]
    summary = core["run"](config)
    ready = core["finalize"](config["output"], job_id=summary["job_id"], request_sha256="fixture-request")
    return Path(config["output"]), ready, root


def test_full_capture_to_fifteen_searchable_saved_only_equations(observed):
    raw, ready, root = observed
    result, count, manifest = A["audit"](raw, root / "compact", expected_manifest=ready["files_manifest_sha256"], expected_request="fixture-request")
    assert count == 15 and result["new_model_calls"] == result["new_reference_forward_calls"] == 0
    assert result["counts"]["native_blocks"] == 2 and result["counts"]["mathematical_affines"] == 22
    assert manifest["files"] and result["boundary_substitution_squares"]["changed-minus-sham"]["closure_rms"] == 0
    queries = json.loads((root / "compact/catalog/provenance.json").read_text())["queries"]
    assert len(queries) == 15 and all(row["exact"] == 1 and row["changed_clock"] == 0 for row in queries)
    assert result["historical_replay"]["edit"]["image"]["exact_value_identity"]
    assert result["entry_changes"]["image"]["changed_values"] > 0
    assert all(result["entry_changes"][key]["changed_values"] == 0 for key in ("text", "img_mod", "txt_mod"))
    assert all(result["positioned_QKV_changes"][key]["text_rows"]["changed_values"] == 0 for key in ("q", "k", "v"))
    records = [json.loads(path.read_text()) for path in (root / "compact/catalog/equations").glob("*.json")]
    toggled = [row["tested_context"] for row in records if row["tested_context"]["lane"] == "native.opposite-reduction"]
    assert len(toggled) == 3
    assert all(row["numerical"]["captured_native"]["source_reduction_policy"] is True
               and row["numerical"]["candidate"]["bf16_reduced_precision_reduction"] is False for row in toggled)
    assert all(row["tested_context"]["operand_frontier"] for row in records)


def test_bad_array_byte_is_rejected_before_comparing_or_cataloging(observed, tmp_path):
    raw, ready, _ = observed
    record = next(iter(json.loads((raw / "files.json").read_text())["files"].values()))
    path = next(path for path in (raw / "arrays").iterdir() if path.is_file())
    original = path.read_bytes()
    try:
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with pytest.raises(ValueError, match="evidence changed"):
            A["audit"](raw, tmp_path / "refused", expected_manifest=ready["files_manifest_sha256"], expected_request="fixture-request")
        assert not (tmp_path / "refused").exists()
    finally:
        path.write_bytes(original)
    assert record["bytes"] > 0


def test_equal_magnitude_reversed_effect_remains_a_large_error():
    measure = runpy.run_path(str(HERE / "qualification/analyze_flux_reader_arithmetic.py"))["measure"]
    result = measure(np.array([1., -1.]), np.array([-1., 1.]))
    assert result["actual_rms"] == result["predicted_rms"] == 1
    assert result["relative_to_actual_rms"] == 2 and result["cosine"] == pytest.approx(-1)
    assert measure(np.zeros(2), np.ones(2))["relative_to_actual_rms"] is None


def test_reinventoried_wrong_role_cannot_pass_source_join(observed, tmp_path):
    raw, _, _ = observed
    hostile = tmp_path / "hostile"
    shutil.copytree(raw, hostile)
    cases_path = hostile / "cases.json"
    cases = json.loads(cases_path.read_text())
    cases["edit"]["source_cut"] = cases["sham"]["source_cut"]
    cases_path.write_text(json.dumps(cases))
    inventory_path = hostile / "files.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["files"]["cases.json"] = {"sha256": A["sha"](cases_path), "bytes": cases_path.stat().st_size}
    inventory_path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="source-role/model join"):
        A["audit"](hostile, tmp_path / "refused-role", expected_manifest=A["sha"](inventory_path), expected_request="fixture-request")
    assert not (tmp_path / "refused-role").exists()


def test_signed_square_retains_interaction_and_cross_terms():
    result = A["square_account"](np.array([0., 0.]), np.array([3., 0.]), np.array([-2., 1.]), np.array([2., 3.]))
    assert result["closure_rms"] == 0
    assert result["total_squared_l2"] == pytest.approx(result["energy_sum"])
    assert result["signed_cross_terms"]["head__tail"] < 0
    assert result["term_squared_l2"]["interaction"] > 0


def test_source_selection_refuses_corrupt_ledger_before_staging(tmp_path):
    path = tmp_path / "native/c1-o0.sham.jsonl"
    path.parent.mkdir()
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="authenticated source changed"):
        S["select_inputs"](tmp_path, {"native/c1-o0.sham.jsonl": {"sha256": "0" * 64, "bytes": path.stat().st_size}})
