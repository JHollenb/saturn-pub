"""Zero-forward remote audit staging, nested copies and compact authority joins."""
from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

import mdb.job_support
import mdb.payload

HERE = Path(__file__).parents[1] / "qualification"


def wrapper():
    return runpy.run_path(str(HERE / "remote_flux_family_audit.py"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def test_stage_is_one_model_free_cpu_job_with_offline_wheels_and_declared_paths(tmp_path, monkeypatch):
    api, staged, launches = wrapper(), {}, []
    producer = tmp_path / "producer"
    source = producer / "payload/nested/source.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n")
    request = {"source_hashes": {"nested/source.py": api["sha"](source)}, "config": {"limits": {"max_files": 12}}}
    request["request_sha256"] = api["request_seal"](request)
    write(producer / "request.json", request)
    write(producer / "submission.json", {"job_id": "job-producer"})
    monkeypatch.setattr(mdb.payload, "stage_worker", lambda *_args, **kwargs: staged.update(kwargs))
    monkeypatch.setattr(mdb.payload, "worker_command", lambda name: ["uv", "run", "--offline", "python", name])

    def launch(command, **kwargs):
        launches.append(kwargs)
        assert "--offline" in command and kwargs["gpu"] is False
        assert kwargs["needs"] == {"cpu": True} and kwargs["pin"] == "beast"
        assert kwargs["retry_on_kill"] is False and kwargs["preflight"] is True and kwargs["timeout_s"] == 600
        contract = kwargs["request"]["scheduling_contract"]
        assert contract["model_free"] is True and contract["model_loads"] == 0 and contract["task_family"] == "analysis"
        assert contract["logical_checkpoint_replays"] == 0 and kwargs["request"]["new_model_calls"] == 0
        extras = dict((name, path) for path, name in staged["additional_files"])
        assert "input-run/payload/nested/source.py" in extras
        preflight = json.loads(extras["preflight.v1.json"].read_text())
        assert {"remote_flux_family_audit.py", "input-run", "analyze_flux_family.py"} <= set(preflight["hashed_roots"])
        assert all(api["sha"](path) == kwargs["request"]["source_hashes"][name] for name, path in extras.items())
        return "job-audit"

    monkeypatch.setattr(mdb.job_support, "guarded_mrun_launch", launch)
    assert api["stage"](tmp_path / "audit", producer)["job_id"] == "job-audit"
    assert len(launches) == 1


def test_nested_image_copy_preserves_bytes_and_refuses_escape_or_overwrite(tmp_path):
    api = wrapper()
    source, target = tmp_path / "source", tmp_path / "output"
    image = source / "nested/images/row.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"saved-native-image")
    api["copy_relative"](source, target, "nested/images/row.png")
    assert (target / "nested/images/row.png").read_bytes() == image.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        api["copy_relative"](source, target, "nested/images/row.png")
    with pytest.raises(ValueError, match="unsafe"):
        api["copy_relative"](source, target, "../row.png")


def compact_fixture(tmp_path, monkeypatch):
    api = wrapper()
    data = tmp_path / "extracted/flux-family-audit-result"
    data.mkdir(parents=True)
    producer = {"config": {"rows": [{"id": "one"}]}, "source_hashes": {}}
    producer["request_sha256"] = api["request_seal"](producer)
    write(data / "producer-request.json", producer)
    (data / "producer-analysis-source.py").write_text("# independent saved-array audit\n")
    archive_hash = "a" * 64
    analysis = {"mechanics_status": "valid", "new_model_calls": 0, "counts": {"images": 1},
                "auditor_source_sha256": api["sha"](data / "producer-analysis-source.py"),
                "custody": {"job_id": "job-producer", "request_sha256": producer["request_sha256"], "archive_sha256": archive_hash}}
    write(data / "producer-analysis.json", analysis)
    summary = {"job_id": "job-producer", "request_sha256": producer["request_sha256"]}
    write(data / "producer-job.json", {"job_id": "job-producer", "state": "succeeded"})
    write(data / "producer-archive-custody.json", {"job_id": "job-producer", "complete": True, "archive_sha256": archive_hash})
    write(data / "producer-log-custody.json", {"job_id": "job-producer", "complete": True,
                                              "job_json_sha256": api["sha"](data / "producer-job.json")})
    originals = {"plan.json": producer["config"], "summary.json": summary, "sources.json": {}, "resolver.json": {}, "support_selection.json": {}}
    for name, value in originals.items():
        write(data / ("producer-" + name), value)
    (data / "producer-timeline.jsonl").write_text("{}\n")
    (data / "nested").mkdir()
    (data / "nested/one.png").write_bytes(b"native-image")
    row = {"id": "one", "image": {"path": "nested/one.png", "sha256": api["sha"](data / "nested/one.png")}}
    write(data / "rows/one.json", row)
    manifest = {name: api["sha"](data / ("producer-" + name)) for name in (*originals, "timeline.jsonl")}
    manifest.update({"rows/one.json": api["sha"](data / "rows/one.json"), "nested/one.png": row["image"]["sha256"]})
    write(data / "producer-files.json", manifest)
    write(data / "producer-report.json", {"job_id": "job-producer", "request_sha256": producer["request_sha256"],
          "archive_sha256": archive_hash, "files_manifest_sha256": api["sha"](data / "producer-files.json"), "report": summary})
    request = {"request_sha256": "outer-request", "new_model_calls": 0,
               "config": {"producer_job_id": "job-producer", "producer_request_sha256": producer["request_sha256"]},
               "source_hashes": {"input-run/request.json": api["sha"](data / "producer-request.json"),
                                 "analyze_flux_family.py": analysis["auditor_source_sha256"]}}
    write(tmp_path / "request.json", request)
    report = {"job_id": "job-audit", "request_sha256": "outer-request", "new_model_calls": 0,
              "producer_job_id": "job-producer", "producer_request_sha256": producer["request_sha256"],
              "producer_archive_sha256": archive_hash, "producer_analysis_sha256": api["sha"](data / "producer-analysis.json"),
              "mechanics_status": "valid", "images": 1}
    write(data / "audit-receipt.json", report)
    write(tmp_path / "report.json", {"archive_root": "flux-family-audit-result", "report": report})
    monkeypatch.setattr(api["verify_compact"].__globals__["runpy"], "run_path", lambda _path: {
        "verify": lambda root, limits: {"job_id": "job-audit", "mechanics_status": "verified"}})
    return api, data


def test_compact_joins_remote_audit_to_producer_archive_request_and_original_nested_png(tmp_path, monkeypatch):
    api, _ = compact_fixture(tmp_path, monkeypatch)
    result = api["verify_compact"](tmp_path)
    assert result["compact_binding_status"] == "verified" and result["producer_job_id"] == "job-producer"
    assert result["images"] == 1 and result["new_model_calls"] == 0
    assert "raw producer archive remains remote" in result["scope"]


@pytest.mark.parametrize("tamper", ["analysis", "archive", "image", "source"])
def test_compact_rejects_consistent_outer_transport_with_broken_inner_authority(tmp_path, monkeypatch, tamper):
    api, data = compact_fixture(tmp_path, monkeypatch)
    target = {"analysis": "producer-analysis.json", "archive": "producer-report.json", "image": "nested/one.png",
              "source": "producer-analysis-source.py"}[tamper]
    if tamper == "archive":
        report = json.loads((data / target).read_text())
        report["archive_sha256"] = "b" * 64
        write(data / target, report)
    else:
        with (data / target).open("ab") as stream:
            stream.write(b" \n")
    with pytest.raises(ValueError, match="differs"):
        api["verify_compact"](tmp_path)


def test_repeat_verification_rechecks_custody_and_preserves_existing_receipt_bytes(tmp_path, monkeypatch):
    api, _ = compact_fixture(tmp_path, monkeypatch)
    original = api["verify_compact"]
    calls = []

    def counted(root):
        calls.append(root)
        return original(root)

    monkeypatch.setitem(api["verify_and_record"].__globals__, "verify_compact", counted)
    first = api["verify_and_record"](tmp_path)
    receipt = tmp_path / "compact-verification.json"
    assert json.loads(receipt.read_text()) == first
    # Semantic JSON equality must not rewrite even noncanonical original bytes.
    receipt.write_text(json.dumps(first, indent=1) + "\n\n")
    before, modified = receipt.read_bytes(), receipt.stat().st_mtime_ns
    assert api["verify_and_record"](tmp_path) == first
    assert len(calls) == 2 and receipt.read_bytes() == before and receipt.stat().st_mtime_ns == modified


def test_existing_mismatched_receipt_is_rejected_after_fresh_verification(tmp_path, monkeypatch):
    api, _ = compact_fixture(tmp_path, monkeypatch)
    original = api["verify_compact"]
    calls = []

    def counted(root):
        calls.append(root)
        return original(root)

    monkeypatch.setitem(api["verify_and_record"].__globals__, "verify_compact", counted)
    receipt = tmp_path / "compact-verification.json"
    receipt.write_text('{"compact_binding_status":"old-or-wrong"}\n')
    before = receipt.read_bytes()
    with pytest.raises(ValueError, match="existing compact verification receipt differs"):
        api["verify_and_record"](tmp_path)
    assert len(calls) == 1 and receipt.read_bytes() == before
