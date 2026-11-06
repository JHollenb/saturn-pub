"""Factorial command, collection bounds and recipe custody; no model or scheduler."""
import base64
import contextlib
import hashlib
import io
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "qualification"
L = runpy.run_path(str(HERE / "submit_flux_prompt_factorial.py"))
C = runpy.run_path(str(HERE / "collect_flux_prompt_factorial.py"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def recipe(index):
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {"descr": "<u2", "fortran_order": False, "shape": (1,)})
    prefix, payload = stream.getvalue(), index.to_bytes(2, "little")
    return {"path": f"arrays/{index:064x}.npy", "bytes": len(prefix) + 2,
        "sha256": hashlib.sha256(prefix + payload).hexdigest(), "npy_prefix_b64": base64.b64encode(prefix).decode(),
        "source": {"file": "transformer/weights.safetensors", "file_sha256": "a" * 64,
            "tensor": f"bank.{index}", "dtype": "BF16", "shape": [1], "data_offsets": [2 * index, 2 * index + 2],
            "payload_sha256": hashlib.sha256(payload).hexdigest()}}


def fixture(root):
    helper = b"# source-bound synthetic standalone retriever\n"
    request = {"request_sha256": "1" * 64, "source_hashes": {"flux_prompt_factorial.py": "2" * 64},
        "config": {"expected_parameter_artifacts_module_sha256": hashlib.sha256(helper).hexdigest()}}
    identity = {"job_id": "job-fixture", "request_sha256": request["request_sha256"]}
    save(root / "source.json", {**identity, "source_hashes": request["source_hashes"]})
    save(root / "summary.json", identity)
    save(root / "plan.json", request["config"])
    for cell in C["CELLS"]:
        save(root / "transactions" / cell / "seal.json", {"cell": cell})
    save(root / "cohort-seal.json", {"all_four_seals": True})
    save(root / "parameter-artifacts.json", {"schema": "mdb-parameter-artifacts-v1",
        "artifacts": [recipe(index) for index in range(169)],
        "provenance": {**identity, "producer_source_sha256": "2" * 64}})
    (root / "retrieve_weights.py").write_bytes(helper)
    return request


def manifest(root, request):
    actual = C["inventory"](root)
    save(root / "remote-manifest.json", {"schema": "mdb-step-boundary-remote-manifest-v1",
        "job_id": "job-fixture", "request_sha256": request["request_sha256"],
        "files": {key: value for key, value in actual["files"].items() if key != "remote-manifest.json"}})
    return C["inventory"](root)


def test_command_retains_qualified_environment_and_changes_only_worker():
    environment = L["B"]["execution_intent"]()
    request = {"execution_environment": environment}
    prior = L["B"]["sealed_worker_command"](request)
    actual = L["command"](request)
    assert actual[:-1] == prior[:-1]
    assert actual[-1] == "flux_prompt_factorial.py"
    assert "TMPDIR=/mnt/hdd8tb/mdb/tmp" in actual and "--offline" in actual
    assert not any("/Users/" in value for value in actual)


def test_command_rejects_unexpected_bootstrap_abi(monkeypatch):
    monkeypatch.setitem(L["B"], "sealed_worker_command", lambda request: ["python", "other.py"])
    with pytest.raises(ValueError, match="bootstrap ABI"):
        L["command"]({})


def test_remote_inventory_and_local_inventory_have_identical_bounded_semantics(tmp_path):
    (tmp_path / "one").write_bytes(b"a")
    script = C["remote_script"](str(tmp_path))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(script, "remote-inventory-fixture", "exec"), {})
    assert json.loads(output.getvalue()) == C["inventory"](tmp_path)
    assert f"MAX_TOTAL={8 << 30}" in script and f"MAX_MEMBER={1 << 30}" in script


def test_inventory_refuses_symlinks_instead_of_following_them(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="non-directory"):
        C["inventory"](tmp_path)


def test_exact_four_seals_cohort_and_169_recipe_records_are_collected(tmp_path):
    request = fixture(tmp_path)
    assert C["verify_manifest"](tmp_path, manifest(tmp_path, request), "job-fixture", request)
    assert C["RESULT"] == "flux-prompt-factorial-result"


@pytest.mark.parametrize("cell", C["CELLS"])
def test_missing_one_seal_is_not_hidden_by_other_three(tmp_path, cell):
    request = fixture(tmp_path)
    (tmp_path / "transactions" / cell / "seal.json").unlink()
    with pytest.raises(ValueError, match="metadata absent"):
        C["verify_manifest"](tmp_path, manifest(tmp_path, request), "job-fixture", request)


def test_missing_cohort_seal_is_refused(tmp_path):
    request = fixture(tmp_path)
    (tmp_path / "cohort-seal.json").unlink()
    with pytest.raises(ValueError, match="metadata absent"):
        C["verify_manifest"](tmp_path, manifest(tmp_path, request), "job-fixture", request)


@pytest.mark.parametrize("key", ("job_id", "request_sha256", "producer_source_sha256"))
def test_recipe_provenance_cannot_belong_to_old_worker_or_history(tmp_path, key):
    request = fixture(tmp_path)
    path = tmp_path / "parameter-artifacts.json"
    value = json.loads(path.read_text())
    value["provenance"][key] = "wrong-source"
    save(path, value)
    with pytest.raises(ValueError, match="producer identity"):
        C["verify_manifest"](tmp_path, manifest(tmp_path, request), "job-fixture", request)


def test_persistent_weight_copy_refused_even_when_in_manifest(tmp_path):
    request = fixture(tmp_path)
    path = tmp_path / recipe(0)["path"]
    path.parent.mkdir()
    path.write_bytes(b"duplicate pretrained constant")
    with pytest.raises(ValueError, match="omitted parameter archives"):
        C["verify_manifest"](tmp_path, manifest(tmp_path, request), "job-fixture", request)


def test_modified_standalone_retriever_refused(tmp_path):
    request = fixture(tmp_path)
    (tmp_path / "retrieve_weights.py").write_text("changed")
    with pytest.raises(ValueError, match="exact checkpoint recipes"):
        C["verify_manifest"](tmp_path, manifest(tmp_path, request), "job-fixture", request)
