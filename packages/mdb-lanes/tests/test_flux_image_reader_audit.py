"""Saved-only audit checks using the actual tiny prefix/seal/target instrument."""
import json
import runpy
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1]
A = runpy.run_path(str(HERE / "qualification/audit_flux_image_reader_forecast.py"))
C = runpy.run_path(str(HERE / "tests/test_flux_image_reader_forecast.py"))


def captured(root):
    config, _, _ = C["fixture"](root)
    config["request_sha256"] = "tiny-audit-request"
    summary = C["C"]["run"](config)
    data = Path(config["output"])
    m, _ = C["C"]["helpers"](config)
    ready = m["finalize"](data, job_id=summary["job_id"], request_sha256=summary["request_sha256"])
    return data, config, summary, ready


def test_actual_tiny_seal_to_full_array_audit_and_search(tmp_path):
    data, _, summary, ready = captured(tmp_path / "source")
    result, count, _ = A["audit"](data, tmp_path / "audit", expected_manifest=ready["files_manifest_sha256"], expected_request=summary["request_sha256"])
    assert result["transaction_verification"]["verified"]
    assert result["errors"]["midpoint"]["exact"] and result["response"]["exact"]
    assert result["wrong_fixed_head"]["effect"]["max_abs"] > 0
    assert result["novel_input"] and count == 5
    assert result["new_model_calls"] == result["new_reference_forward_calls"] == 0
    provenance = json.loads((tmp_path / "audit/catalog/provenance.json").read_text())
    assert all(row["exact"] == 1 and row["wrong_clock"] == 0 for row in provenance["queries"])


def test_array_byte_corruption_refused_before_audit(tmp_path):
    data, _, summary, ready = captured(tmp_path / "source")
    inventory = json.loads((data / "files.json").read_text())
    name = next(name for name in inventory["files"] if name.endswith(".npy"))
    target = data / name
    target.write_bytes(target.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="native evidence changed"):
        A["audit"](data, tmp_path / "audit", expected_manifest=ready["files_manifest_sha256"], expected_request=summary["request_sha256"])
    assert not (tmp_path / "audit").exists()


def test_coherent_inventory_with_wrong_source_role_refused(tmp_path):
    data, _, summary, ready = captured(tmp_path / "source")
    source = json.loads((data / "source.json").read_text())
    source["source_cuts"]["edit"]["cut"] = "incorrect-source-role"
    (data / "source.json").write_text(json.dumps(source))
    inventory = json.loads((data / "files.json").read_text())
    inventory["files"]["source.json"].update(sha256=A["sha"](data / "source.json"), bytes=(data / "source.json").stat().st_size)
    (data / "files.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="source role join differs"):
        A["audit"](data, tmp_path / "audit", expected_manifest=A["sha"](data / "files.json"), expected_request=summary["request_sha256"])
    assert not (tmp_path / "audit").exists()


def test_coherent_inventory_with_prefix_qkv_substitution_refused(tmp_path):
    data, _, summary, _ = captured(tmp_path / "source")
    path = data / "prefixes/midpoint.json"
    prefix = json.loads(path.read_text())
    prefix["values"]["q"] = prefix["values"]["k"]
    path.write_text(json.dumps(prefix))
    inventory = json.loads((data / "files.json").read_text())
    inventory["files"]["prefixes/midpoint.json"].update(sha256=A["sha"](path), bytes=path.stat().st_size)
    (data / "files.json").write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="sealed frontier QKV differs"):
        A["audit"](data, tmp_path / "audit", expected_manifest=A["sha"](data / "files.json"), expected_request=summary["request_sha256"])
    assert not (tmp_path / "audit").exists()
