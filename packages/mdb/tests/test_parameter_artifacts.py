"""Exact-byte restore/transport mechanics, with synthetic tensors and no frameworks."""
import base64
import copy
import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys

import pytest

from mdb import parameter_artifacts as pa


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def npy_prefix(dtype, shape):
    text = repr({"descr": dtype, "fortran_order": False, "shape": tuple(shape)}).encode()
    header = text + b" " * ((64 - (10 + len(text) + 1) % 64) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header


@pytest.fixture
def recipe(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    specs = [("word", "BF16", "<u2", [3], b"\x00\x00\x00\x80\x80\x3f"),
             ("float", "F32", "<f4", [2], struct.pack("<ff", -0.0, 1.25)),
             ("double", "F64", "<f8", [], struct.pack("<d", -3.5)),
             ("integer", "I64", "<i8", [1, 2], struct.pack("<qq", -7, 2**50)),
             ("empty", "F32", "<f4", [0, 2], b"")]
    header, payload, entries, originals = {}, bytearray(), [], {}
    for name, dtype, descr, shape, raw in specs:
        start = len(payload)
        payload.extend(raw)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
        prefix = npy_prefix(descr, shape)
        path = "nested/" + name + ".npy"
        originals[path] = prefix + raw
        entries.append({"path": path, "bytes": len(prefix) + len(raw), "sha256": sha(prefix + raw),
                        "npy_prefix_b64": base64.b64encode(prefix).decode(),
                        "source": {"file": "model.safetensors", "file_sha256": "0" * 64,
                                   "tensor": name, **header[name], "payload_sha256": sha(raw)}})
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    source = struct.pack("<Q", len(encoded)) + encoded + bytes(payload)
    (checkpoint / "model.safetensors").write_bytes(source)
    for entry in entries:
        entry["source"]["file_sha256"] = sha(source)
    manifest = {"schema": pa.SCHEMA, "artifacts": entries,
                "provenance": {"original_inventory_unchanged": True},
                "defaults": {"artifact_root": "extracted/result", "checkpoint_root": str(checkpoint),
                             "ssh_host": "deliberately-unreachable-default", "python": sys.executable}}
    return manifest, checkpoint, originals


def test_all_supported_native_bytes_roundtrip_and_correct_existing_retained(recipe, tmp_path):
    manifest, checkpoint, originals = recipe
    root = tmp_path / "restored"
    assert not pa.verify_present(manifest, root)["complete"]
    result = pa.restore(manifest, root, checkpoint)
    assert result["checkpoint_files_verified"] == 1
    assert result["conversions"] == result["model_calls"] == 0
    assert {row["status"] for row in result["artifacts"]} == {"restored"}
    before = {}
    for path, raw in originals.items():
        target = root / path
        assert target.read_bytes() == raw
        before[path] = target.stat().st_ino
    assert pa.verify_present(manifest, root)["complete"]
    assert {row["status"] for row in pa.restore(manifest, root, checkpoint)["artifacts"]} == {"present"}
    assert all((root / path).stat().st_ino == inode for path, inode in before.items())


def test_validate_returns_private_copy(recipe):
    manifest, _, _ = recipe
    value = pa.validate_manifest(manifest)
    value["artifacts"][0]["source"]["shape"][0] = 99
    assert manifest["artifacts"][0]["source"]["shape"] == [3]


@pytest.mark.parametrize("path", ["../escape.npy", "/absolute.npy", "a/../b.npy", "a//b.npy", "./a.npy", "a\\b.npy", "."])
def test_unsafe_archive_paths_refused(recipe, path):
    manifest, _, _ = recipe
    manifest["artifacts"][0]["path"] = path
    with pytest.raises(pa.ParameterArtifactError):
        pa.validate_manifest(manifest)


@pytest.mark.parametrize("field,value", [("sha256", "A" * 64), ("bytes", True), ("bytes", 1),
                                        ("npy_prefix_b64", "not base64!")])
def test_malformed_archive_metadata_refused(recipe, field, value):
    manifest, _, _ = recipe
    manifest["artifacts"][0][field] = value
    with pytest.raises(pa.ParameterArtifactError):
        pa.validate_manifest(manifest)


@pytest.mark.parametrize("field,value", [("file", "../model.safetensors"), ("shape", [True]),
                                        ("shape", [4]), ("dtype", "F16"), ("data_offsets", [0, 7])])
def test_source_geometry_and_conversion_refused(recipe, field, value):
    manifest, _, _ = recipe
    manifest["artifacts"][0]["source"][field] = value
    with pytest.raises(pa.ParameterArtifactError):
        pa.validate_manifest(manifest)


def test_duplicate_or_directory_collision_refused(recipe):
    manifest, _, _ = recipe
    manifest["artifacts"].append(copy.deepcopy(manifest["artifacts"][0]))
    with pytest.raises(pa.ParameterArtifactError, match="duplicate"):
        pa.validate_manifest(manifest)
    manifest["artifacts"][-1]["path"] = "nested/word.npy/child.npy"
    with pytest.raises(pa.ParameterArtifactError, match="collides"):
        pa.validate_manifest(manifest)


def test_npy_fortran_and_dtype_conversion_refused(recipe):
    manifest, _, _ = recipe
    entry = manifest["artifacts"][0]
    entry["npy_prefix_b64"] = base64.b64encode(npy_prefix("<f4", [3])).decode()
    with pytest.raises(pa.ParameterArtifactError, match="dtype"):
        pa.validate_manifest(manifest)
    prefix = npy_prefix("<u2", [3]).replace(b"False", b" True")
    entry["npy_prefix_b64"] = base64.b64encode(prefix).decode()
    with pytest.raises(pa.ParameterArtifactError, match="order"):
        pa.validate_manifest(manifest)


@pytest.mark.parametrize("kind", ["filehash", "payloadhash", "artifacthash", "tensorname", "headermetadata", "shortfile"])
def test_bad_source_never_installs_unverified_artifact(recipe, tmp_path, kind):
    manifest, checkpoint, _ = recipe
    manifest["artifacts"] = manifest["artifacts"][:1]
    entry = manifest["artifacts"][0]
    if kind == "filehash":
        entry["source"]["file_sha256"] = "0" * 64
    elif kind == "payloadhash":
        entry["source"]["payload_sha256"] = "0" * 64
    elif kind == "artifacthash":
        entry["sha256"] = "0" * 64
    elif kind == "tensorname":
        entry["source"]["tensor"] = "absent"
    else:
        source = checkpoint / "model.safetensors"
        raw = source.read_bytes()
        if kind == "headermetadata":
            raw = raw.replace(b'"shape":[3]', b'"shape":[4]', 1)
        else:
            raw = raw[:9]
        source.write_bytes(raw)
        entry["source"]["file_sha256"] = sha(raw)
    target = tmp_path / "bad"
    with pytest.raises(pa.ParameterArtifactError):
        pa.restore(manifest, target, checkpoint)
    assert not (target / entry["path"]).exists()
    assert not list(target.rglob(".mdb-parameter-*"))


@pytest.mark.parametrize("kind", ["wrong", "file-symlink", "parent-symlink", "root-symlink"])
def test_existing_wrong_files_and_symlink_destinations_refused(recipe, tmp_path, kind):
    manifest, checkpoint, _ = recipe
    root = tmp_path / "output"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    if kind == "root-symlink":
        root.symlink_to(elsewhere, target_is_directory=True)
    else:
        root.mkdir()
        if kind == "parent-symlink":
            (root / "nested").symlink_to(elsewhere, target_is_directory=True)
        else:
            (root / "nested").mkdir()
            target = root / manifest["artifacts"][0]["path"]
            if kind == "wrong":
                target.write_bytes(b"wrong")
            else:
                (elsewhere / "keep").write_bytes(b"unchanged")
                target.symlink_to(elsewhere / "keep")
    with pytest.raises((pa.ParameterArtifactError, OSError)):
        pa.restore(manifest, root, checkpoint)
    assert not list(elsewhere.glob("*.npy"))


def framed(header, payload=b""):
    raw = json.dumps(header).encode()
    return struct.pack(">I", len(raw)) + raw + payload


@pytest.mark.parametrize("kind", ["unknown", "duplicate", "size", "truncated", "tail", "missing", "payloadhash"])
def test_hostile_transport_allowlist_and_size_checks(recipe, tmp_path, kind):
    manifest, _, originals = recipe
    manifest["artifacts"] = manifest["artifacts"][:1]
    entry = manifest["artifacts"][0]
    header = {k: entry[k] for k in ("path", "bytes", "sha256")}
    payload = originals[entry["path"]]
    if kind == "unknown":
        header["path"] = "../escape"
    if kind == "size":
        header["bytes"] += 1
    if kind == "payloadhash":
        entry["source"]["payload_sha256"] = "0" * 64
    body = framed(header, payload)
    if kind == "duplicate":
        body += framed(header, payload)
    if kind == "truncated":
        body = body[:-1]
    else:
        body += framed({"done": True, "artifacts": 1})
    if kind == "tail":
        body += b"unlisted"
    if kind == "missing":
        body = framed({"done": True, "artifacts": 0})
    with pa._root_fd(tmp_path / "stream", create=True) as root:
        with pytest.raises(pa.ParameterArtifactError):
            pa._receive_stream(manifest, root, io.BytesIO(pa.MAGIC + body))


def test_standalone_cli_defaults_verify_and_explicit_local_override(recipe, tmp_path):
    manifest, checkpoint, originals = recipe
    run = tmp_path / "run"
    run.mkdir()
    script = run / "retrieve_weights.py"
    shutil.copyfile(pa.__file__, script)
    (run / "parameter-artifacts.json").write_text(json.dumps(manifest))
    command = [sys.executable, "-I", str(script)]
    missing = subprocess.run(command + ["--verify-only"], capture_output=True, text=True, cwd=tmp_path)
    assert missing.returncode == 1 and not json.loads(missing.stdout)["complete"]
    # Explicit checkpoint must override the deliberately unreachable default SSH host.
    restored = subprocess.run(command + ["--checkpoint", str(checkpoint)], capture_output=True, text=True, cwd=tmp_path)
    assert restored.returncode == 0, restored.stderr
    assert json.loads(restored.stdout)["mode"] == "local"
    assert all((run / "extracted/result" / path).read_bytes() == raw for path, raw in originals.items())
    assert subprocess.run(command + ["--verify-only"], capture_output=True).returncode == 0


def test_same_source_one_ssh_process_and_bare_script_defaults(recipe, tmp_path, monkeypatch):
    manifest, checkpoint, originals = recipe
    run = tmp_path / "remote-run"
    run.mkdir()
    shutil.copyfile(pa.__file__, run / "retrieve_weights.py")
    manifest["defaults"]["ssh_host"] = "fixture-host"
    (run / "parameter-artifacts.json").write_text(json.dumps(manifest))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "ssh-calls"
    ssh = bin_dir / "ssh"
    ssh.write_text("#!/bin/sh\nprintf x >> \"$MDB_TEST_SSH_CALLS\"\nwhile [ \"$1\" != -- ]; do shift; done\n[ \"$2\" = fixture-host ] || exit 92\nexec /bin/sh -c \"$3\"\n")
    ssh.chmod(0o700)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("MDB_TEST_SSH_CALLS", str(calls))
    result = subprocess.run([sys.executable, "-I", str(run / "retrieve_weights.py")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["mode"] == "ssh"
    assert calls.read_text() == "x"
    assert all((run / "extracted/result" / path).read_bytes() == raw for path, raw in originals.items())


def test_checkpoint_file_hash_once_per_invocation(recipe, tmp_path, monkeypatch):
    manifest, checkpoint, _ = recipe
    original = pa._file_hash
    seen = []

    def counting(stream):
        seen.append(stream.name)
        return original(stream)

    monkeypatch.setattr(pa, "_file_hash", counting)
    pa.restore(manifest, tmp_path / "fresh", checkpoint)
    assert len(seen) == 1


def test_symlinked_root_ancestor_and_checkpoint_source_refused(recipe, tmp_path):
    manifest, checkpoint, _ = recipe
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(pa.ParameterArtifactError, match="ancestor"):
        pa.restore(manifest, alias / "descendant", checkpoint)
    target = checkpoint / "model.safetensors"
    target.rename(checkpoint / "real.safetensors")
    target.symlink_to("real.safetensors")
    with pytest.raises((pa.ParameterArtifactError, OSError)):
        pa.restore(manifest, tmp_path / "new", checkpoint)


def test_cli_manifest_bound_and_duplicate_json_keys(recipe, tmp_path):
    manifest, _, _ = recipe
    script = tmp_path / "retrieve_weights.py"
    shutil.copyfile(pa.__file__, script)
    path = tmp_path / "parameter-artifacts.json"
    with path.open("wb") as stream:
        stream.seek(pa.MAX_MANIFEST_BYTES)
        stream.write(b"x")
    result = subprocess.run([sys.executable, "-I", str(script), "--verify-only"], capture_output=True, text=True)
    assert result.returncode == 2 and "exceeds bound" in result.stderr
    with pytest.raises(pa.ParameterArtifactError, match="duplicate"):
        pa._json(b'{"a":1,"a":2}')


def test_ssh_timeout_is_finite_and_does_not_install(recipe, tmp_path, monkeypatch):
    manifest, checkpoint, _ = recipe
    bin_dir = tmp_path / "timeout-bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text("#!/bin/sh\nexec sleep 30\n")
    ssh.chmod(0o700)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    with pytest.raises(pa.ParameterArtifactError, match="timed out"):
        pa.restore(manifest, tmp_path / "timeout-output", checkpoint, ssh="fixture-host", timeout=1)
    assert not list((tmp_path / "timeout-output").rglob("*.npy"))
