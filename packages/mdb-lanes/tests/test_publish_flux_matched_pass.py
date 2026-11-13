"""Saved tiny-report publisher checks; no native forwards or learned math.

Set MDB_MATCHED_PASS_TINY_RESULT to a saved tiny replay output and
MDB_FACTORIAL_TINY_DATA to its original saved tiny producer result. These are
explicit fixture inputs, never discovered from sibling workspace checkouts.
"""
from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")
pytest.importorskip("PIL")

HERE = Path(__file__).parents[1]
PUBLISHER_SHA256 = "543ccecec787c2ebcb199f7ae3663733abe6344a5967fec64558f3f7395e2e6f"


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    tiny, original = (os.environ.get(name) for name in ("MDB_MATCHED_PASS_TINY_RESULT", "MDB_FACTORIAL_TINY_DATA"))
    if not tiny or not original:
        pytest.skip("requires explicit existing saved tiny fixtures; no models are constructed")
    path = HERE / "qualification/publish_flux_matched_pass.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PUBLISHER_SHA256
    m = runpy.run_path(str(path))
    h = m["helpers"](HERE / "qualification/publish_flux_prompt_factorial.py")
    data = tmp_path_factory.mktemp("saved-Q") / "data"
    shutil.copytree(tiny, data)
    return {"m": m, "h": h, "data": data, "report": m["read"](data / "report.json"),
        "seal": m["read"](data / "Q-seal.json"), "parent": h["inventory"](Path(original))}


@pytest.fixture
def capsule(saved, tmp_path):
    raw = b'{"scope":"synthetic saved report only","value":17}\n'
    packed = gzip.compress(raw, mtime=0)
    metadata = {"schema": "mdb-actual-audit-capsule-v1", "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "gzip_bytes": len(packed),
        "gzip_sha256": hashlib.sha256(packed).hexdigest()}
    line = "MDB_FACTORIAL_AUDIT_CAPSULE=" + json.dumps({**metadata, "base64_gzip": base64.b64encode(packed).decode()})
    log = tmp_path / "worker.log"
    log.write_text("Synthetic fixture; not an actual scheduler job.\n" + line + "\n")
    (tmp_path / "result.json").write_bytes(raw)
    job = {"result": {"artifacts": {"job_log": {"sha256": saved["m"]["sha"](log), "bytes": log.stat().st_size}}}}
    return {"directory": tmp_path, "raw": raw, "line": line, "log": log, "job": job, "collection": {"capsule": metadata}}


def test_all28_signed_accounts_qbytes_and_parent_archive_references(saved):
    result = saved["m"]["validate_accounts"](saved["h"], saved["data"], saved["report"], saved["seal"], saved["parent"])
    assert result["full_field_accounts"] == 28
    assert result["complete_Q_transitions"] == 56
    assert result["P_N_archive_refs_bound_to_parent_inventory"] == 56


def test_parent_archive_tampering_refused(saved):
    bad = copy.deepcopy(saved["report"])
    bad["comparisons"]["c1-o0"]["rows"][0]["fields"]["image"]["P"]["archive"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parent inventory"):
        saved["m"]["validate_accounts"](saved["h"], saved["data"], bad, saved["seal"], saved["parent"])


def test_signed_orientation_cannot_flip(saved):
    bad = copy.deepcopy(saved["report"])
    bad["comparisons"]["c1-o0"]["rows"][0]["fields"]["image"]["account"]["signs"]["B"] = "Q-P"
    with pytest.raises(ValueError, match="orientation"):
        saved["m"]["validate_accounts"](saved["h"], saved["data"], bad, saved["seal"], saved["parent"])


def test_q_shape_and_byte_identity(saved):
    record = saved["report"]["comparisons"]["c1-o0"]["rows"][0]["fields"]["image"]["Q"]["archive"]
    path = saved["data"] / record["path"]
    original = path.read_bytes()
    path.write_bytes(original + b"tampered")
    try:
        with pytest.raises(ValueError, match="archive bytes"):
            saved["m"]["validate_accounts"](saved["h"], saved["data"], saved["report"], saved["seal"], saved["parent"])
    finally:
        path.write_bytes(original)


def test_exact_capsule_bytes_then_wrong_saved_result_refused(saved, capsule):
    saved["m"]["verify_capsule"](saved["h"], capsule["directory"], capsule["collection"], capsule["job"], "result.json")
    (capsule["directory"] / "result.json").write_bytes(capsule["raw"] + b"wrong")
    with pytest.raises(ValueError, match="scheduler-log capsule"):
        saved["m"]["verify_capsule"](saved["h"], capsule["directory"], capsule["collection"], capsule["job"], "result.json")


def test_log_mismatch_and_duplicate_capsule_refused(saved, capsule):
    log = capsule["log"]
    log.write_text(log.read_text() + capsule["line"] + "\n")
    with pytest.raises(ValueError, match="scheduler log"):
        saved["m"]["verify_capsule"](saved["h"], capsule["directory"], capsule["collection"], capsule["job"], "result.json")
    duplicate_job = copy.deepcopy(capsule["job"])
    duplicate_job["result"]["artifacts"]["job_log"] = {"sha256": saved["m"]["sha"](log), "bytes": log.stat().st_size}
    with pytest.raises(ValueError, match="one actual log capsule"):
        saved["m"]["verify_capsule"](saved["h"], capsule["directory"], capsule["collection"], duplicate_job, "result.json")
