"""Restore exact NPY parameter artifacts from pinned safetensors, using only stdlib.

This file can be copied verbatim to a run as ``retrieve_weights.py``. No model,
framework, repository import, persistent cache, or tensor conversion is used.
"""
from __future__ import annotations

import argparse
import ast
import base64
import binascii
import contextlib
import hashlib
import json
import os
import shlex
import stat
import struct
import subprocess
import sys
import threading
import uuid
from pathlib import Path, PurePosixPath

SCHEMA = "mdb-parameter-artifacts-v1"
MAX_MANIFEST_BYTES = 32 << 20
MAX_HEADER_BYTES = 16 << 20
MAX_NPY_PREFIX_BYTES = 64 << 10
MAX_ARTIFACT_BYTES = 64 << 30
MAX_TOTAL_BYTES = 1 << 40
MAX_FILES = 100_000
CHUNK = 8 << 20
MAGIC = b"MDBPARAM1\n"
DTYPES = {"BF16": (2, {"<u2", "<V2", "|V2"}), "F32": (4, {"<f4"}),
          "F64": (8, {"<f8"}), "I64": (8, {"<i8"})}


class ParameterArtifactError(ValueError):
    """A recipe, source, destination, or transport integrity check failed."""


def _require(condition, message):
    if not condition:
        raise ParameterArtifactError(message)


def _integer(value, *, minimum=0, maximum=MAX_TOTAL_BYTES):
    _require(type(value) is int and minimum <= value <= maximum, "invalid bounded integer")
    return value


def _hash(value):
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), "invalid SHA256")
    return value


def _relative(value, *, dot=False):
    _require(isinstance(value, str) and 0 < len(value) <= 4096
             and "\\" not in value and "\0" not in value, "invalid relative path")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and str(path) == value
             and ".." not in path.parts and (dot or value != "."), "unsafe relative path")
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _json(raw):
    _require(len(raw) <= MAX_MANIFEST_BYTES, "JSON exceeds bound")
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ParameterArtifactError("nonfinite JSON")))
    except (ValueError, UnicodeError) as exc:
        raise ParameterArtifactError("invalid JSON: " + str(exc)) from exc


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _prefix(entry):
    try:
        prefix = base64.b64decode(entry["npy_prefix_b64"], validate=True)
    except (ValueError, TypeError, binascii.Error) as exc:
        raise ParameterArtifactError("invalid NPY prefix base64") from exc
    _require(10 <= len(prefix) <= MAX_NPY_PREFIX_BYTES and prefix[:6] == b"\x93NUMPY", "invalid NPY prefix")
    version = tuple(prefix[6:8])
    _require(version in ((1, 0), (2, 0), (3, 0)), "unsupported NPY version")
    width = 2 if version == (1, 0) else 4
    _require(len(prefix) >= 8 + width, "truncated NPY prefix")
    length = int.from_bytes(prefix[8:8 + width], "little")
    _require(length + 8 + width == len(prefix), "NPY prefix must contain exactly its header")
    try:
        header = ast.literal_eval(prefix[8 + width:].decode("utf-8" if version == (3, 0) else "latin1"))
    except (ValueError, SyntaxError, UnicodeError, RecursionError) as exc:
        raise ParameterArtifactError("invalid NPY header") from exc
    source = entry["source"]
    _require(isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"}
             and header["fortran_order"] is False and type(header["shape"]) is tuple
             and all(type(n) is int for n in header["shape"])
             and list(header["shape"]) == source["shape"]
             and header["descr"] in DTYPES[source["dtype"]][1], "NPY/native dtype, shape or order differs")
    return prefix


def validate_manifest(manifest):
    """Validate and return a private JSON copy; unknown provenance is retained.

    Each artifact has path/bytes/sha256/npy_prefix_b64 and source fields
    file/file_sha256/tensor/dtype/shape/data_offsets/payload_sha256. Only exact
    little-endian BF16-word, F32, F64 and I64 payloads are admitted.
    """
    try:
        raw = _canonical(manifest)
    except (TypeError, ValueError) as exc:
        raise ParameterArtifactError("manifest is not finite JSON") from exc
    value = _json(raw)
    _require(isinstance(value, dict) and value.get("schema") == SCHEMA, "unsupported manifest schema")
    artifacts = value.get("artifacts")
    _require(isinstance(artifacts, list) and 0 < len(artifacts) <= MAX_FILES, "invalid artifact list")
    paths, sources, total = set(), {}, 0
    for entry in artifacts:
        _require(isinstance(entry, dict) and isinstance(entry.get("source"), dict), "invalid artifact entry")
        try:
            path = _relative(entry["path"])
            _require(path not in paths, "duplicate artifact path")
            paths.add(path)
            size = _integer(entry["bytes"], minimum=1, maximum=MAX_ARTIFACT_BYTES)
            _hash(entry["sha256"])
            source = entry["source"]
            filename = _relative(source["file"])
            file_hash = _hash(source["file_sha256"])
            _require(filename not in sources or sources[filename] == file_hash, "conflicting checkpoint file hashes")
            sources[filename] = file_hash
            _require(isinstance(source["tensor"], str) and 0 < len(source["tensor"]) <= 4096, "invalid tensor name")
            _require(source["dtype"] in DTYPES, "unsupported native dtype or conversion")
            shape = source["shape"]
            _require(isinstance(shape, list) and len(shape) <= 32, "invalid tensor shape")
            elements = 1
            for dimension in shape:
                elements *= _integer(dimension, maximum=MAX_ARTIFACT_BYTES)
                _require(elements <= MAX_ARTIFACT_BYTES, "shape exceeds bound")
            offsets = source["data_offsets"]
            _require(isinstance(offsets, list) and len(offsets) == 2, "invalid tensor offsets")
            start, end = (_integer(n) for n in offsets)
            _require(end >= start and end - start == elements * DTYPES[source["dtype"]][0], "tensor byte geometry differs")
            _hash(source["payload_sha256"])
            _require(len(_prefix(entry)) + end - start == size, "NPY artifact byte count differs")
            total += size
            _require(total <= MAX_TOTAL_BYTES, "artifact total exceeds bound")
        except (KeyError, TypeError) as exc:
            raise ParameterArtifactError("missing or malformed artifact core field") from exc
    for path in paths:
        _require(not any(str(parent) in paths for parent in PurePosixPath(path).parents if str(parent) != "."),
                 "artifact file collides with parent directory")
    defaults = value.get("defaults", {})
    _require(isinstance(defaults, dict), "invalid defaults")
    if "artifact_root" in defaults:
        _relative(defaults["artifact_root"], dot=True)
    if "checkpoint_root" in defaults:
        _require(isinstance(defaults["checkpoint_root"], str) and Path(defaults["checkpoint_root"]).is_absolute(),
                 "default checkpoint root must be absolute")
    for key in ("ssh_host", "python"):
        if key in defaults:
            _require(isinstance(defaults[key], str) and defaults[key] and not defaults[key].startswith("-")
                     and "\0" not in defaults[key] and "\n" not in defaults[key], "invalid transport default")
    return value


@contextlib.contextmanager
def _root_fd(root, *, create=False):
    path = Path(os.path.abspath(root))
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            for part in path.parts[1:]:
                if create:
                    try:
                        os.mkdir(part, dir_fd=fd)
                    except FileExistsError:
                        pass
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = next_fd
        except OSError as exc:
            raise ParameterArtifactError("root and every ancestor must be nonsymlink directories") from exc
        yield fd
    finally:
        os.close(fd)


@contextlib.contextmanager
def _parent_fd(root_fd, path, *, create=False):
    parts = PurePosixPath(path).parts
    fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            if create:
                try:
                    os.mkdir(part, dir_fd=fd)
                except FileExistsError:
                    pass
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd, parts[-1]
    finally:
        os.close(fd)


def _file_hash(stream):
    digest = hashlib.sha256()
    while chunk := stream.read(CHUNK):
        digest.update(chunk)
    return digest.hexdigest()


def _existing(parent, name, entry):
    try:
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    _require(stat.S_ISREG(info.st_mode) and info.st_size == entry["bytes"], "wrong existing destination type or size")
    with os.fdopen(os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent), "rb") as stream:
        _require(_file_hash(stream) == entry["sha256"], "wrong existing destination SHA256")
    return True


def verify_present(manifest, artifact_root):
    """Report present/missing artifacts, refusing any wrong existing file/symlink."""
    manifest = validate_manifest(manifest)
    rows = []
    if not Path(artifact_root).exists() and not Path(artifact_root).is_symlink():
        rows = [{"path": e["path"], "status": "missing"} for e in manifest["artifacts"]]
    else:
        with _root_fd(artifact_root) as root:
            for entry in manifest["artifacts"]:
                try:
                    with _parent_fd(root, entry["path"]) as (parent, name):
                        present = _existing(parent, name, entry)
                except FileNotFoundError:
                    present = False
                rows.append({"path": entry["path"], "status": "present" if present else "missing"})
    return {"schema": "mdb-parameter-artifact-presence-v1", "complete": all(r["status"] == "present" for r in rows),
            "artifacts": rows}


def _read_exact(stream, size):
    parts, remaining = [], size
    while remaining:
        chunk = stream.read(min(remaining, CHUNK))
        _require(bool(chunk), "truncated source or stream")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _stable(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _source_entries(manifest, checkpoint_root):
    """Keep one source file open at a time; hash each pinned file once."""
    groups = {}
    for entry in manifest["artifacts"]:
        groups.setdefault(entry["source"]["file"], []).append(entry)
    with _root_fd(checkpoint_root) as root:
        for filename, entries in sorted(groups.items()):
            with _parent_fd(root, filename) as (parent, name):
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                _require(stat.S_ISREG(before.st_mode), "checkpoint source is not regular")
                _require(_file_hash(stream) == entries[0]["source"]["file_sha256"], "checkpoint file SHA256 differs")
                stream.seek(0)
                header_size = int.from_bytes(_read_exact(stream, 8), "little")
                _require(2 <= header_size <= MAX_HEADER_BYTES and 8 + header_size <= before.st_size,
                         "safetensors header exceeds file/bound")
                header = _json(_read_exact(stream, header_size))
                _require(isinstance(header, dict), "invalid safetensors header")
                for entry in entries:
                    source = entry["source"]
                    actual = header.get(source["tensor"])
                    _require(isinstance(actual, dict) and all(_canonical(actual.get(k)) == _canonical(source[k])
                             for k in ("dtype", "shape", "data_offsets")), "safetensors tensor metadata differs")
                    start, end = source["data_offsets"]
                    _require(8 + header_size + end <= before.st_size, "safetensors tensor exceeds source")
                    stream.seek(8 + header_size + start)
                    yield entry, stream, end - start
                _require(_stable(os.fstat(stream.fileno())) == _stable(before), "checkpoint changed during retrieval")


def _install(root, entry, chunks):
    with _parent_fd(root, entry["path"], create=True) as (parent, name):
        exists = _existing(parent, name, entry)
        temporary = ".mdb-parameter-" + uuid.uuid4().hex
        fd = None if exists else os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        digest, size = hashlib.sha256(), 0
        try:
            with contextlib.ExitStack() as stack:
                output = stack.enter_context(os.fdopen(fd, "wb")) if fd is not None else None
                for chunk in chunks:
                    size += len(chunk)
                    _require(size <= entry["bytes"], "artifact stream exceeds declared size")
                    digest.update(chunk)
                    if output:
                        output.write(chunk)
                _require(size == entry["bytes"] and digest.hexdigest() == entry["sha256"], "reconstructed artifact hash/size differs")
                if output:
                    output.flush()
                    os.fsync(output.fileno())
            if not exists:
                # link is atomic and cannot replace a concurrent or wrong destination.
                os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
                os.unlink(temporary, dir_fd=parent)
                os.fsync(parent)
        finally:
            if fd is not None:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
    return {"path": entry["path"], "status": "present" if exists else "restored", "sha256": entry["sha256"], "bytes": size}


def _artifact_chunks(entry, source, size):
    yield _prefix(entry)
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = source.read(min(CHUNK, remaining))
        _require(bool(chunk), "truncated tensor payload")
        remaining -= len(chunk)
        digest.update(chunk)
        yield chunk
    _require(digest.hexdigest() == entry["source"]["payload_sha256"], "tensor payload SHA256 differs")


def _send_stream(manifest, checkpoint_root, output):
    output.write(MAGIC)
    count = 0
    for entry, source, size in _source_entries(manifest, checkpoint_root):
        header = _canonical({k: entry[k] for k in ("path", "bytes", "sha256")})
        output.write(struct.pack(">I", len(header)))
        output.write(header)
        digest = hashlib.sha256()
        for chunk in _artifact_chunks(entry, source, size):
            output.write(chunk)
            digest.update(chunk)
        _require(digest.hexdigest() == entry["sha256"], "remote reconstructed NPY SHA256 differs")
        count += 1
    header = _canonical({"done": True, "artifacts": count})
    output.write(struct.pack(">I", len(header)))
    output.write(header)
    output.flush()


def _receive_stream(manifest, root, stream):
    _require(_read_exact(stream, len(MAGIC)) == MAGIC, "invalid parameter stream magic")
    allowed = {entry["path"]: entry for entry in manifest["artifacts"]}
    rows = []
    while True:
        length = struct.unpack(">I", _read_exact(stream, 4))[0]
        _require(0 < length <= 16 << 10, "stream header exceeds bound")
        header = _json(_read_exact(stream, length))
        _require(isinstance(header, dict), "invalid stream header")
        if header.get("done") is True:
            _require(not allowed and header == {"done": True, "artifacts": len(rows)}, "incomplete stream inventory")
            _require(not stream.read(1), "extra bytes after stream inventory")
            return rows
        entry = allowed.pop(header.get("path"), None)
        _require(entry is not None and header == {k: entry[k] for k in ("path", "bytes", "sha256")},
                 "unexpected/duplicate/mismatched stream artifact")

        def chunks():
            prefix = _prefix(entry)
            _require(_read_exact(stream, len(prefix)) == prefix, "stream NPY prefix differs")
            yield prefix
            remaining = entry["bytes"] - len(prefix)
            payload_hash = hashlib.sha256()
            while remaining:
                chunk = _read_exact(stream, min(remaining, CHUNK))
                remaining -= len(chunk)
                payload_hash.update(chunk)
                yield chunk
            _require(payload_hash.hexdigest() == entry["source"]["payload_sha256"], "stream tensor payload SHA256 differs")

        rows.append(_install(root, entry, chunks()))


def restore(manifest, artifact_root, checkpoint_root, *, ssh=None, python="python3", timeout=3600):
    """Restore exact artifacts locally or through one SSH Python process.

    Existing correct files are retained; wrong files and all destination symlinks
    are refused. Completed earlier files remain evidence if a later file fails.
    Source checkpoint hashes are checked once per source file per invocation.
    """
    manifest = validate_manifest(manifest)
    _integer(timeout, minimum=1, maximum=86400)
    verify_present(manifest, artifact_root)  # Refuse wrong existing files before source I/O.
    with _root_fd(artifact_root, create=True) as root:
        if ssh is None:
            rows = [_install(root, entry, _artifact_chunks(entry, stream, size))
                    for entry, stream, size in _source_entries(manifest, checkpoint_root)]
        else:
            _require(isinstance(ssh, str) and ssh and not ssh.startswith("-") and "\0" not in ssh, "invalid SSH host")
            _require(isinstance(python, str) and python and not python.startswith("-") and "\0" not in python, "invalid remote Python")
            code = Path(__file__).read_text()
            command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
                       "-o", "ServerAliveCountMax=3", "--", ssh,
                       shlex.join([python, "-c", code, "--_stream", "--checkpoint", str(checkpoint_root)])]
            with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
                timed_out = threading.Event()

                def expire():
                    timed_out.set()
                    process.kill()

                timer = threading.Timer(timeout, expire)
                timer.daemon = True
                timer.start()
                try:
                    process.stdin.write(_canonical(manifest))
                    process.stdin.close()
                    rows = _receive_stream(manifest, root, process.stdout)
                    _require(process.wait() == 0, "remote retrieval failed")
                except BaseException as exc:
                    process.kill()
                    process.wait()
                    if timed_out.is_set():
                        raise ParameterArtifactError("SSH retrieval timed out") from exc
                    raise
                finally:
                    timer.cancel()
    return {"schema": "mdb-parameter-artifact-restoration-v1", "manifest_sha256": hashlib.sha256(_canonical(manifest)).hexdigest(),
            "mode": "local" if ssh is None else "ssh", "artifacts": rows,
            "checkpoint_files_verified": len({e["source"]["file"] for e in manifest["artifacts"]}),
            "conversions": 0, "model_calls": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--ssh", help="One SSH host; explicit --checkpoint defaults to local retrieval")
    parser.add_argument("--python", help="Remote Python executable")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600, help="SSH total timeout in seconds (1..86400)")
    parser.add_argument("--_stream", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args._stream:
            _require(args.checkpoint is not None, "stream checkpoint root required")
            manifest = validate_manifest(_json(sys.stdin.buffer.read(MAX_MANIFEST_BYTES + 1)))
            _send_stream(manifest, args.checkpoint, sys.stdout.buffer)
            return 0
        script_root = Path(globals().get("__file__", "retrieve_weights.py")).resolve().parent
        manifest_path = args.manifest or script_root / "parameter-artifacts.json"
        with manifest_path.open("rb") as stream:
            manifest = validate_manifest(_json(stream.read(MAX_MANIFEST_BYTES + 1)))
        defaults = manifest.get("defaults", {})
        artifact_root = args.root or script_root / defaults.get("artifact_root", ".")
        if args.verify_only:
            result = verify_present(manifest, artifact_root)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["complete"] else 1
        checkpoint = args.checkpoint or defaults.get("checkpoint_root")
        _require(checkpoint is not None, "--checkpoint or defaults.checkpoint_root required")
        ssh = args.ssh if args.ssh is not None else (None if args.checkpoint is not None else defaults.get("ssh_host"))
        result = restore(manifest, artifact_root, checkpoint, ssh=ssh,
                         python=args.python or defaults.get("python", "python3"), timeout=args.timeout)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ParameterArtifactError, OSError, KeyError, TypeError) as exc:
        print("parameter artifact retrieval refused: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
