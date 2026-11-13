"""Source and actual-result joins for native continuations of prior forecasts."""
import base64
import gzip
import hashlib
import json
import runpy
from pathlib import Path

import pytest

L = runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification/submit_flux_native_corner.py"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def request(value):
    value = dict(value)
    value["request_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return value


def matched_fixture(tmp_path, monkeypatch, *, parent_id="job-parent", native=0):
    # Generic authenticated payload-envelope validation is separately qualified.
    # These fixtures exercise the new parent/report/Q-seal/actual-log joins.
    monkeypatch.setitem(L["C"], "validate_job", lambda *args: "unused-result-path")
    parent, matched = tmp_path / "parent", tmp_path / "matched"
    write(parent / "collection.json", {"job_id": "job-parent"})
    data = matched / "extracted/matched-pass-output"
    write(data / "Q-seal.json", {"retrospective": True, "outputs": ["tiny-saved-descriptor"]})
    report = {"Q_seal_sha256": L["sha"](data / "Q-seal.json"), "native_forwards": native}
    write(data / "report.json", report)
    result = {"report": report, "near_data_execution": {"job_id": "job-matched", "parent_job_id": parent_id,
        "mathematical_transitions": 56, "native_framework_forwards": native}}
    write(matched / "result.json", result)
    raw = (matched / "result.json").read_bytes()
    zipped = gzip.compress(raw, mtime=0)
    capsule = {"schema": "mdb-actual-audit-capsule-v1", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "gzip_bytes": len(zipped), "gzip_sha256": hashlib.sha256(zipped).hexdigest()}
    log = (L["L"]["MARKER"] + json.dumps({**capsule, "base64_gzip": base64.b64encode(zipped).decode()}) + "\n").encode()
    (matched / "worker.log").write_bytes(log)
    write(matched / "request.json", request({"schema": "tiny-fixture"}))
    write(matched / "job.json", {"job_id": "job-matched", "result": {"diagnostics": {"command": ["sealed"]},
        "artifacts": {"job_log": {"bytes": len(log), "sha256": hashlib.sha256(log).hexdigest()}}}})
    write(matched / "admission-command.json", {"command": ["sealed"]})
    write(matched / "collection-local-inventory.json", L["C"]["inventory"](data))
    names = ("request.json", "job.json", "result.json", "worker.log", "admission-command.json", "collection-local-inventory.json")
    write(matched / "collection.json", {"job_id": "job-matched",
        "request_sha256": json.loads((matched / "request.json").read_text())["request_sha256"],
        "capsule": capsule, "bindings": {name: L["sha"](matched / name) for name in names}})
    return parent, matched


def test_source_bound_completed_q_can_anchor_native_corner(tmp_path, monkeypatch):
    parent, matched = matched_fixture(tmp_path, monkeypatch)
    assert L["validate_matched"](parent, matched) == ("job-matched", L["sha"](matched / "extracted/matched-pass-output/report.json"))


@pytest.mark.parametrize("options", [{"parent_id": "job-another"}, {"native": 1}])
def test_wrong_parent_or_different_q_work_cannot_anchor_native_corner(tmp_path, monkeypatch, options):
    parent, matched = matched_fixture(tmp_path, monkeypatch, **options)
    with pytest.raises(ValueError, match="completed Q"):
        L["validate_matched"](parent, matched)


def test_changed_q_bytes_after_collection_are_rejected(tmp_path, monkeypatch):
    parent, matched = matched_fixture(tmp_path, monkeypatch)
    (matched / "extracted/matched-pass-output/new-array.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="completed Q"):
        L["validate_matched"](parent, matched)


def test_changed_actual_log_after_collection_is_rejected(tmp_path, monkeypatch):
    parent, matched = matched_fixture(tmp_path, monkeypatch)
    with (matched / "worker.log").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="collection/job"):
        L["validate_matched"](parent, matched)


def test_request_digest_covers_the_declared_native_work():
    sealed = request({"native_adapter_transitions": 336})
    L["validate_request"](sealed)
    sealed["native_adapter_transitions"] = 56
    with pytest.raises(ValueError, match="self digest"):
        L["validate_request"](sealed)


def test_native_bootstrap_preserves_cuda_visibility_and_public_dependencies():
    environment = L["B"]["execution_intent"]()
    cmd = L["command"]({"execution_environment": environment})
    assert cmd[-2:] == ["mdb_lanes.worker_bootstrap", "flux_native_corner.py"]
    assert "CUDA_VISIBLE_DEVICES=" not in cmd
    assert [cmd[i + 1] for i, value in enumerate(cmd) if value == "--with" and not cmd[i + 1].startswith("./")] == environment["requirements"]


def test_physical_transitions_do_not_exceed_the_scheduler_logical_work_cap():
    contract = L["work_contract"]()
    assert contract["logical_branches"] == 4
    assert contract["native_adapter_transitions"] == 336
    assert contract["max_local_work_items"] <= 256


def test_missing_paired_response_contract_is_rejected_before_staging(tmp_path):
    with pytest.raises(ValueError, match="paired response contract"):
        L["stage"](tmp_path / "run", parent=tmp_path, audit=tmp_path, matched=tmp_path, expectations=tmp_path)


def test_payload_import_check_refuses_omitted_transitive_collector(tmp_path):
    source = Path(L["HERE"])
    (tmp_path / "collect_flux_prompt_factorial.py").write_bytes((source / "collect_flux_prompt_factorial.py").read_bytes())
    with pytest.raises(FileNotFoundError, match="collect_flux_initial_noise"):
        L["validate_payload_helpers"](tmp_path)
