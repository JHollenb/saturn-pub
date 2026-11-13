"""Finite launcher and authenticated collection mechanics; no scheduler calls."""
from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parents[1] / "qualification"
L = runpy.run_path(str(HERE / "submit_flux_joint_tail.py"))
C = runpy.run_path(str(HERE / "collect_flux_joint_tail.py"))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_storage_includes_remaining_step_and_new_project_joint_banks():
    config = {"transport_limits": {"max_expanded_bytes": 12 << 30, "max_member_bytes": 1 << 30},
        "transaction_max_bytes": 768 << 20}
    result = L["storage_plan"](config)
    assert result["prior_parameter_bytes_lower_bound"] == 3435138560
    assert 335 << 20 < result["project_parameter_bytes_lower_bound"] < 336 << 20
    assert result["joint_parameter_banks"] == 5
    assert 468 << 20 < result["one_joint_parameter_bank_bytes_lower_bound"] < 469 << 20
    assert result["joint_parameter_bytes_lower_bound"] == 5 * result["one_joint_parameter_bank_bytes_lower_bound"]
    assert result["members"]["fused_weight"]["bytes"] == 169869440
    assert result["complete_output_bytes_lower_bound"] == 377489920
    for key, bound in (("max_member_bytes", 128 << 20), ("max_expanded_bytes", 4 << 30)):
        config["transport_limits"][key] = bound
        with pytest.raises(ValueError, match="custody bounds"):
            L["storage_plan"](config)
        config["transport_limits"] = {"max_expanded_bytes": 12 << 30, "max_member_bytes": 1 << 30}


def test_sealed_command_offline_native_linux_paths():
    from mdb_lanes.execution_environment import intent

    environment = intent()
    environment.update(uv_cache_dir="/mnt/hdd8tb/mdb/uv-cache", torch_home="/mnt/hdd8tb/mdb/torch",
        extra_model_roots="/mnt/big/llm-models", python="3.14.6")
    with patch.dict("os.environ", {"MDB_WORKER_UV_CACHE_DIR": "/Users/wrong", "MDB_WORKER_PYTHON": "9.9"}):
        command = L["sealed_worker_command"]({"execution_environment": environment})
    assert command[-1] == "flux_joint_tail_campaign.py" and "--offline" in command
    assert command[command.index("--python") + 1] == "3.14.6"
    assert "UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache" in command
    assert not any("/Users/" in value for value in command)


def test_scheduler_fallback_is_explicit_and_sealed_endpoint_cannot_drift(monkeypatch):
    monkeypatch.delenv("MRUN_URL", raising=False)
    assert L["scheduler_endpoint"]() == "http://192.168.68.68:9025"
    assert L["scheduler_endpoint"]({"scheduler_url": "http://192.168.68.68:9025"}) == "http://192.168.68.68:9025"
    with pytest.raises(ValueError, match="endpoint differs"):
        L["scheduler_endpoint"]({"scheduler_url": "http://other:9025"})


def test_prior_source_tampering_rejected_and_dirty_live_source_irrelevant(tmp_path, monkeypatch):
    prior, live = tmp_path / "prior", tmp_path / "live"
    (prior / "payload").mkdir(parents=True)
    live.mkdir()
    expected = {}
    for name in L["PRIOR_HELPERS"]:
        raw = ("qualified " + name).encode()
        (prior / "payload" / name).write_bytes(raw)
        (live / name).write_text("unrelated dirty source")
        expected[name] = digest(raw)
    request = {"source_hashes": expected}
    request["request_sha256"] = digest(json.dumps(request, sort_keys=True, allow_nan=False).encode())
    save(prior / "request.json", request)
    monkeypatch.setitem(L["prior_sources"].__globals__, "PRIOR_WORKER_SHA256", expected["flux_joint_chain_campaign.py"])
    paths, _ = L["prior_sources"](prior)
    assert all(path.parent == prior / "payload" for path in paths.values())
    (prior / "payload/flux_single_contexts.py").write_text("tampered")
    with pytest.raises(ValueError, match="prior scientific helper differs"):
        L["prior_sources"](prior)


def test_independent_analysis_preserves_old_math_and_explicit_corrected_reader(tmp_path):
    prior, here = tmp_path / "prior", tmp_path / "repo/packages/mdb/qualification"
    here.mkdir(parents=True)
    hashes = {}
    for name in ("flux_reentry_audit_math.py", "analyze_flux_reentry.py", "analyze_flux_step_boundary.py"):
        raw = ("prior math " + name).encode()
        path = prior / "independent-math-freeze" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        hashes[name] = digest(raw)
        (here / name).write_text("dirty live math")
    save(prior / "request.json", {"independent_math_sources": hashes})
    for name in L["NEW_ANALYZERS"]:
        (here / name).write_text("new numerical source " + name)
    def git_show(command, **kwargs):
        assert command == ["git", "show", "94a8e75:packages/mdb/qualification/analyze_flux_reentry.py"]
        assert kwargs["cwd"] == here.parents[2]
        return SimpleNamespace(stdout=b"tracked corrected typed-node reader")
    with patch.object(subprocess, "run", side_effect=git_show):
        values, authorities = L["independent_sources"](here, prior)
    assert digest(values["flux_reentry_audit_math.py"]) == hashes["flux_reentry_audit_math.py"]
    assert digest(values["analyze_flux_step_boundary.py"]) == hashes["analyze_flux_step_boundary.py"]
    assert values["analyze_flux_reentry.py"] == b"tracked corrected typed-node reader"
    assert authorities["analyze_flux_reentry.py"]["original_preoutcome_sha256"] == hashes["analyze_flux_reentry.py"]
    assert authorities["analyze_flux_reentry.py"]["authority"] == "corrected reader, tracked Git 94a8e75"
    assert set(values) == set(hashes) | set(L["NEW_ANALYZERS"])


def test_installed_public_closure_modules_have_distinct_qualified_source_pins():
    assert L["closure_module_sha256"]() == "45dd999fe54103ca7676248652798dd295fbc727d0fdf38cd912fa48b8011c23"
    assert L["joint_closure_module_sha256"]() == "bb352ae4859335abdd751d33ccc2774fa4d37c53f5167c5012183c693b0d709c"
    assert L["tail_closure_module_sha256"]() == "2788c984045098532009e105219e7c9cfcc38732a54d892bdc296149a979b586"


def test_refuse_launcher_drift_before_any_admission(tmp_path):
    request = {"launcher_source_sha256": "0" * 64, "source_hashes": {}, "independent_math_sources": {}}
    request["request_sha256"] = digest(json.dumps(request, sort_keys=True, allow_nan=False).encode())
    save(tmp_path / "request.json", request)
    with patch("mdb_lanes.job_support.guarded_mrun_launch") as launch:
        with pytest.raises(ValueError, match="launcher changed"):
            L["submit_staged"](tmp_path)
        launch.assert_not_called()


def test_collect_manifest_binds_new_result_job_and_exact_files(tmp_path):
    request = {"request_sha256": "1" * 64, "source_hashes": {"new.py": "2" * 64}, "config": {"seed": 97}}
    save(tmp_path / "source.json", {"job_id": "job-fixture", "request_sha256": "1" * 64,
        "source_hashes": request["source_hashes"]})
    save(tmp_path / "summary.json", {"job_id": "job-fixture", "request_sha256": "1" * 64})
    save(tmp_path / "plan.json", request["config"])
    save(tmp_path / "transaction/seal.json", {"fixture": True})
    first = C["inventory"](tmp_path)
    save(tmp_path / "remote-manifest.json", {"schema": "mdb-step-boundary-remote-manifest-v1",
        "job_id": "job-fixture", "request_sha256": "1" * 64, "files": first["files"]})
    inventory = C["inventory"](tmp_path)
    assert C["RESULT"] == "flux-joint-tail-result"
    assert C["verify_manifest"](tmp_path, inventory, "job-fixture", request)
    (tmp_path / "extra").write_text("undeclared")
    with pytest.raises(ValueError, match="exact result"):
        C["verify_manifest"](tmp_path, C["inventory"](tmp_path), "job-fixture", request)


def test_collection_rejects_symlink_and_wrong_job_state(tmp_path):
    (tmp_path / "target").write_bytes(b"bytes")
    (tmp_path / "alias").symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="nonregular"):
        C["inventory"](tmp_path)
    with pytest.raises(ValueError, match="succeeded"):
        C["validate_job"]({"job_id": "job-fixture", "state": "failed"}, "job-fixture", {})



def test_tail_collection_limits_and_result_are_distinct_from_old_lane():
    assert C["MAX_TOTAL"] == 12 << 30
    assert C["MAX_MEMBER"] == 1 << 30
    script = C["remote_script"]("/mnt/hdd8tb/mrun-work/job-fixture/flux-joint-tail-result")
    assert "MAX_TOTAL=12884901888" in script
    assert "MAX_MEMBER=1073741824" in script
    assert "flux-joint-tail-result" in script


def test_inventory_rejects_actual_member_and_total_bound_overflow(tmp_path):
    (tmp_path / "data").write_bytes(b"12345678")
    with pytest.raises(ValueError, match="collection bounds"):
        C["inventory"](tmp_path, max_member=7)
    with pytest.raises(ValueError, match="collection bounds"):
        C["inventory"](tmp_path, max_total=7)
