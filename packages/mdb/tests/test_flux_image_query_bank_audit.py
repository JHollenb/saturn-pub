"""Saved-only fixed-recipient audit, hostile consumed-operand joins and signed math."""
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parents[1]
A = runpy.run_path(str(HERE / "qualification/audit_flux_image_query_bank.py"))
C = runpy.run_path(str(HERE / "tests/test_flux_image_query_bank.py"))


def captured(root):
    class Factory:
        def mktemp(self, name):
            path = root / name
            path.mkdir(parents=True, exist_ok=False)
            return path
    source = C["saved_source"].__wrapped__(Factory())
    config = C["configuration"](source, root / "cross")
    result = C["C"]["run"](config)
    data = Path(config["output"])
    ready = C["C"]["finalize_directory"](data, job_id=result["job_id"], request_sha256=result["request_sha256"])
    return data, result, ready


def test_tiny_actual_intervention_to_signed_catalog(tmp_path):
    data, summary, ready = captured(tmp_path)
    result, count, _ = A["audit"](data, tmp_path / "audit", expected_manifest=ready["files_manifest_sha256"], expected_request=summary["request_sha256"])
    assert count == 10 and result["transaction_verification"]["verified"]
    assert all(row["exact"] and row["bitwise_fp32_widening_exact"] for row in result["errors"].values())
    assert all(row["exact"] for row in result["query_only_text_exits"].values())
    assert result["responses"]["image"]["square"]["closure_rms"] == 0
    assert result["new_model_calls"] == result["new_reference_forward_calls"] == 0
    index = json.loads((tmp_path / "audit/catalog/provenance.json").read_text())
    assert all(row["exact"] == 1 and row["wrong_clock"] == 0 for row in index["queries"])


def test_coherent_inventory_cannot_hide_wrong_consumed_query(tmp_path):
    data, summary, _ = captured(tmp_path)
    path = data / "targets/q0-b1.json"
    target = json.loads(path.read_text())
    wrong = json.loads((data / "targets/q1-b1.json").read_text())
    target["consumed_qkv"]["q"] = wrong["consumed_qkv"]["q"]
    path.write_text(json.dumps(target))
    inventory = json.loads((data / "files.json").read_text())
    inventory["files"]["targets/q0-b1.json"].update(sha256=A["sha"](path), bytes=path.stat().st_size)
    (data / "files.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="sidecar differs from closed observation"):
        A["audit"](data, tmp_path / "audit", expected_manifest=A["sha"](data / "files.json"), expected_request=summary["request_sha256"])
    assert not (tmp_path / "audit").exists()


@pytest.mark.parametrize("kind", ["census", "seal"])
def test_coherent_inventory_cannot_hide_wrong_execution_ledger(tmp_path, kind):
    data, summary, _ = captured(tmp_path)
    relative = "summary.json" if kind == "census" else "timeline.jsonl"
    path = data / relative
    if kind == "census":
        body = json.loads(path.read_text())
        body["target_counts"]["native_blocks"] = 3
        path.write_text(json.dumps(body))
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["seal_sha256"] = "wrong-forecast-seal"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    inventory = json.loads((data / "files.json").read_text())
    inventory["files"][relative].update(sha256=A["sha"](path), bytes=path.stat().st_size)
    (data / "files.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="census|seal"):
        A["audit"](data, tmp_path / "audit", expected_manifest=A["sha"](data / "files.json"), expected_request=summary["request_sha256"])
    assert not (tmp_path / "audit").exists()


def test_corrupt_array_refused_before_output(tmp_path):
    data, summary, ready = captured(tmp_path)
    path = next((data / "arrays").glob("*.npy"))
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="native evidence changed"):
        A["audit"](data, tmp_path / "audit", expected_manifest=ready["files_manifest_sha256"], expected_request=summary["request_sha256"])
    assert not (tmp_path / "audit").exists()


def test_signed_square_cancellation_and_undefined_normalization():
    measure = runpy.run_path(str(HERE / "qualification/analyze_flux_reader_arithmetic.py"))["measure"]
    actual = {"q0-b0": np.array([0., 0.]), "q1-b0": np.array([3., 1.]),
              "q0-b1": np.array([-2., -1.]), "q1-b1": np.array([1., 2.])}
    predicted = {key: -value for key, value in actual.items()}
    report = A["response_account"](actual, predicted, measure)
    assert report["contrasts"]["total"]["comparison"]["relative_to_actual_rms"] == 2
    assert report["square"]["closure_rms"] == 0
    assert report["square"]["signed_cross_terms"]["query_at_bank0__bank_at_query0"] < 0
    assert report["square"]["energy_sum"] == report["square"]["total_squared_l2"]
    actual["q1-b1"] = actual["q0-b0"].copy()
    zero = A["response_account"](actual, predicted, measure)
    assert zero["native_total_rms"] == 0
    assert zero["contrasts"]["interaction"]["native_relative_to_total_rms"] is None
    assert zero["contrasts"]["interaction"]["signed_projection_onto_total"] is None
