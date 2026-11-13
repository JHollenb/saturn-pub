"""CPU audit transport and immutable-neighbor mechanics, no learned model."""
import base64
import gzip
import hashlib
import json
import runpy
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "qualification"
W = runpy.run_path(str(HERE / "audit_flux_prompt_factorial.py"))
L = runpy.run_path(str(HERE / "submit_flux_prompt_factorial_audit.py"))


def inventory(path):
    value = path.read_bytes()
    return {"files": {"nested/a": {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest(), "filetype": "regular"}}}


def test_hardlink_preserves_original_inode_and_exact_bytes(tmp_path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    path = source / "nested/a"
    path.write_bytes(b"immutable native evidence")
    before = path.stat()
    W["hardlink_inventory"](source, tmp_path / "target", inventory(path))
    copy = tmp_path / "target/nested/a"
    assert copy.read_bytes() == path.read_bytes()
    assert (copy.stat().st_dev, copy.stat().st_ino) == (before.st_dev, before.st_ino)
    assert path.stat().st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize("name", ["../a", "/a", "nested/../../a", "nested//a"])
def test_hardlink_rejects_unsafe_members(tmp_path, name):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="unsafe"):
        W["hardlink_inventory"](source, tmp_path / "target", {"files": {name: {}}})


def test_hardlink_rejects_symlink_parent(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside/a").write_bytes(b"x")
    (source / "nested").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="ancestor"):
        W["hardlink_inventory"](source, tmp_path / "target", inventory(tmp_path / "outside/a"))


def test_hardlink_rejects_changed_source(tmp_path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    path = source / "nested/a"
    path.write_bytes(b"x")
    expected = inventory(path)
    path.write_bytes(b"y")
    with pytest.raises(ValueError, match="member differs"):
        W["hardlink_inventory"](source, tmp_path / "target", expected)


def capsule(raw):
    compressed = gzip.compress(raw, mtime=0)
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "gzip_bytes": len(compressed),
        "gzip_sha256": hashlib.sha256(compressed).hexdigest(), "base64_gzip": base64.b64encode(compressed).decode()}


def test_actual_capsule_decodes_exact_report_bytes():
    raw = b'{"independent":true,"native_framework_forwards":0}\n'
    marker = L["MARKER"] + json.dumps(capsule(raw))
    actual, meta = L["decode_capsule"]("progress\n" + marker + "\n")
    assert actual == raw
    assert meta["sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("mutation", ["digest", "size", "duplicate"])
def test_corrupt_or_duplicate_capsule_refused(mutation):
    value = capsule(b'{"a":1}\n')
    if mutation == "digest":
        value["sha256"] = "0" * 64
    elif mutation == "size":
        value["bytes"] = 33 << 20
    log = L["MARKER"] + json.dumps(value)
    if mutation == "duplicate":
        log += "\n" + log
    with pytest.raises(ValueError):
        L["decode_capsule"](log)


def test_cpu_command_keeps_sealed_environment_and_hides_cuda():
    request = {"execution_environment": L["B"]["execution_intent"](), "config": {"cpu_threads": 4}}
    command = L["command"](request)
    assert command[-1] == "audit_flux_prompt_factorial.py"
    assert "CUDA_VISIBLE_DEVICES=" in command
    assert "UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache" in command
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "VECLIB_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert variable + "=4" in command


def test_audit_contract_is_accepted_by_installed_scheduler_contract():
    contract = L["audit_contract"]()
    assert contract["task_family"] == "analysis"
    assert contract["model_free"] is True and contract["model_loads"] == 0
    assert contract["native_framework_forwards"] == 0
    assert contract["mathematical_transitions"] == 448


def test_memoized_resource_profile_bounds_constants_and_finite_wall_time():
    profile = L["audit_resources"](True)
    assert profile["ram_mb"] == 28672 and profile["cpu_threads"] == 4
    assert profile["vram_mb"] == 0 and profile["disk_gb"] == 10
    assert profile["max_wall_seconds"] == 6000 < profile["timeout_s"] == 6300
    assert profile["est_wall_s"] < profile["max_wall_seconds"]
    assert L["audit_resources"](False)["ram_mb"] == 12288


def test_lossless_bf16_cache_has_independently_reviewed_resource_profile():
    profile = L["audit_resources"](True, parameter_codec="bf16-bits")
    assert profile["ram_mb"] == 14336 and profile["cpu_threads"] == 4
    assert profile["vram_mb"] == 0 and profile["disk_gb"] == 10
    assert profile["max_wall_seconds"] == 6000 < profile["timeout_s"] == 6300
    assert profile["est_wall_s"] < profile["max_wall_seconds"]
    with pytest.raises(ValueError, match="codec"):
        L["audit_resources"](True, parameter_codec="rounded")
