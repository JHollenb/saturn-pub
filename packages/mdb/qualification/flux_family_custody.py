"""Bounded multipart log custody for one qualification campaign, without pickle.

The fleet's ordinary log endpoint is the fallback transport. This module does
not launch or retry model jobs and does not change runtime artifact protocols.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import stat
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

PART = "MDB_FLUX_FAMILY_PART_V1 "
REPORT = "MDB_FLUX_FAMILY_REPORT_V1 "
SCHEMA = "mdb-flux-family-transport-v1"
MIB = 1024 * 1024
DEFAULTS = {"max_archive_bytes": 1536 * MIB, "max_expanded_bytes": 3072 * MIB,
            "max_files": 20000, "max_member_bytes": 128 * MIB,
            "chunk_bytes": 8 * MIB, "max_log_bytes": 2304 * MIB}
TERMINAL = {"succeeded", "failed", "timeout", "cancelled", "lost", "killed_ram", "killed_vram"}
IO_CHUNK = MIB


class CustodyError(ValueError):
    pass


def _limits(limits):
    result = dict(DEFAULTS)
    if limits is not None:
        if set(limits) - set(result):
            raise CustodyError("unknown custody limit")
        result.update(limits)
    if any(type(value) is not int or value < 1 for value in result.values()):
        raise CustodyError("custody limits must be positive finite integers")
    result["max_line_bytes"] = ((result["chunk_bytes"] + 2) // 3) * 4 + MIB
    return result


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _save(path, value):
    with path.open("x") as stream:
        stream.write(_json(value) + "\n")


def _sha(path):
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(IO_CHUNK), b""):
            digest.update(chunk)
            count += len(chunk)
    return digest.hexdigest(), count


def _relative(name):
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
        raise CustodyError("unsafe archive/source path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts) or path.as_posix() != name:
        raise CustodyError("unsafe archive/source path")
    return path


def _regular(path):
    if path.is_symlink() or not path.is_file():
        raise CustodyError("custody requires regular files without symlinks")


def _inventory(root, limits):
    if root.is_symlink() or not root.is_dir():
        raise CustodyError("output root must be a real directory")
    inventory, expanded = {}, 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise CustodyError("output contains a symlink directory")
        for name in sorted(files):
            path = Path(directory) / name
            _regular(path)
            relative = path.relative_to(root).as_posix()
            _relative(relative)
            size = path.stat().st_size
            if size > limits["max_member_bytes"]:
                raise CustodyError("archive member exceeds byte bound")
            expanded += size
            if expanded > limits["max_expanded_bytes"] or len(inventory) >= limits["max_files"]:
                raise CustodyError("archive expanded bytes or file count exceeds bound")
            digest, checked_size = _sha(path)
            if checked_size != size:
                raise CustodyError("output file changed during inventory")
            inventory[relative] = {"sha256": digest, "bytes": size}
    return dict(sorted(inventory.items()))


def _check_manifest(root, inventory, limits):
    if "files.json" not in inventory:
        raise CustodyError("worker files.json manifest is missing")
    if inventory["files.json"]["bytes"] > limits["max_line_bytes"]:
        raise CustodyError("worker file manifest exceeds bounded metadata size")
    manifest = json.loads((root / "files.json").read_text())
    expected = {name: item["sha256"] for name, item in inventory.items() if name != "files.json"}
    if manifest != expected:
        raise CustodyError("worker files.json manifest differs from complete file inventory")


class _BoundedWriter:
    def __init__(self, stream, ceiling):
        self.stream, self.ceiling, self.count = stream, ceiling, 0

    def write(self, data):
        if self.count + len(data) > self.ceiling:
            raise CustodyError("compressed archive exceeds byte bound")
        written = self.stream.write(data)
        self.count += written
        return written

    def flush(self):
        self.stream.flush()


def emit(output: Path, report: dict, limits: dict | None = None):
    """Spool one bounded USTAR/gzip archive, then print ordered transport parts.

    Worker ``files.json`` must map every other relative output file to SHA256.
    The final envelope contains the original report under ``report``. Parts are
    only transport divisions of the exact archive byte stream.
    """
    bounds, output = _limits(limits), Path(output)
    if not isinstance(report, dict) or not report.get("job_id") or not report.get("request_sha256"):
        raise CustodyError("report requires job_id and request_sha256")
    _relative(output.name)
    inventory = _inventory(output, bounds)
    _check_manifest(output, inventory, bounds)
    descriptor, archive_name = tempfile.mkstemp(prefix="." + output.name + "-", suffix=".tar.gz", dir=output.parent)
    archive_path = Path(archive_name)
    succeeded = False
    try:
        with os.fdopen(descriptor, "wb") as destination:
            bounded = _BoundedWriter(destination, bounds["max_archive_bytes"])
            with gzip.GzipFile(filename="", fileobj=bounded, mode="wb", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                    for name, item in inventory.items():
                        path = output / name
                        _regular(path)
                        handle = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                        with os.fdopen(handle, "rb") as stream:
                            before = os.fstat(stream.fileno())
                            if not stat.S_ISREG(before.st_mode) or before.st_size != item["bytes"]:
                                raise CustodyError("output file changed before archive")
                            header = tarfile.TarInfo(output.name + "/" + name)
                            header.size, header.mode, header.mtime = item["bytes"], 0o644, 0
                            archive.addfile(header, stream)
                            after = os.fstat(stream.fileno())
                            if any(getattr(after, key) != getattr(before, key) for key in (
                                    "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
                                raise CustodyError("output file changed during archive")
            destination.flush()
        digest, archive_bytes = _sha(archive_path)
        # Verify the actual snapshot against the manifest before sending bytes.
        actual = _scan_archive(archive_path, bounds, output.name)
        if actual != inventory:
            raise CustodyError("archived snapshot differs from worker file inventory")
        parts, logged = [], 0
        with archive_path.open("rb") as stream:
            for index, chunk in enumerate(iter(lambda: stream.read(bounds["chunk_bytes"]), b"")):
                part = {"index": index, "sha256": hashlib.sha256(chunk).hexdigest(),
                        "base64": base64.b64encode(chunk).decode("ascii")}
                line = PART + _json(part)
                logged += len(line.encode()) + 1
                if len(line.encode()) > bounds["max_line_bytes"] or logged > bounds["max_log_bytes"]:
                    raise CustodyError("multipart transport log exceeds bound")
                print(line, flush=True)
                parts.append({"index": index, "sha256": part["sha256"], "bytes": len(chunk)})
        envelope = {"schema": SCHEMA, "job_id": report["job_id"],
                    "request_sha256": report["request_sha256"], "archive_sha256": digest,
                    "archive_bytes": archive_bytes, "archive_root": output.name,
                    "archive_member_count": len(inventory),
                    "archive_expanded_bytes": sum(item["bytes"] for item in inventory.values()),
                    "files_manifest_sha256": inventory["files.json"]["sha256"],
                    "part_count": len(parts), "parts": parts, "report": report}
        line = REPORT + _json(envelope)
        if len(line.encode()) > bounds["max_line_bytes"] or logged + len(line.encode()) + 1 > bounds["max_log_bytes"]:
            raise CustodyError("final report transport exceeds log bound")
        print(line, flush=True)
        succeeded = True
        return envelope
    finally:
        # A failed emission leaves its partial spool available for diagnosis.
        if succeeded:
            archive_path.unlink()


def _scan_archive(path, limits, archive_root, destination=None):
    """Read USTAR directly in bounded blocks; reject extensions before payloads.

    Avoid automatic PAX/GNU extension parsing, which can allocate attacker-sized
    metadata before ordinary member bounds are checked.
    """
    _regular(Path(path))
    _relative(archive_root)
    if "/" in archive_root:
        raise CustodyError("archive root must be one directory name")
    inventory, expanded, tar_bytes, zeros = {}, 0, 0, 0
    max_tar_bytes = limits["max_expanded_bytes"] + limits["max_files"] * 1024 + 10240
    with gzip.open(path, "rb") as stream:
        while True:
            block = stream.read(512)
            tar_bytes += len(block)
            if tar_bytes > max_tar_bytes:
                raise CustodyError("expanded tar stream exceeds bound")
            if not block:
                if zeros < 2:
                    raise CustodyError("truncated tar end markers")
                break
            if len(block) != 512:
                raise CustodyError("truncated archive header")
            if block == b"\0" * 512:
                zeros += 1
                continue
            if zeros:
                raise CustodyError("nonzero data after archive end marker")
            try:
                member = tarfile.TarInfo.frombuf(block, "utf-8", "strict")
            except (tarfile.HeaderError, UnicodeError) as error:
                raise CustodyError("invalid archive header") from error
            if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                raise CustodyError("archive permits only regular files, without links or extensions")
            relative = _relative(member.name)
            if len(relative.parts) < 2 or relative.parts[0] != archive_root:
                raise CustodyError("archive member escapes declared root")
            name = PurePosixPath(*relative.parts[1:]).as_posix()
            if name in inventory:
                raise CustodyError("duplicate archive member")
            if member.size < 0 or member.size > limits["max_member_bytes"]:
                raise CustodyError("archive member exceeds byte bound")
            expanded += member.size
            if expanded > limits["max_expanded_bytes"] or len(inventory) >= limits["max_files"]:
                raise CustodyError("archive expanded bytes or file count exceeds bound")
            output = None
            if destination is not None:
                target = destination / relative.as_posix()
                target.parent.mkdir(parents=True, exist_ok=True)
                if any(parent.is_symlink() for parent in (target, *target.parents)):
                    raise CustodyError("extraction path contains a symlink")
                output = target.open("xb")
            digest, remaining = hashlib.sha256(), member.size
            try:
                while remaining:
                    chunk = stream.read(min(IO_CHUNK, remaining))
                    if not chunk:
                        raise CustodyError("truncated archive member")
                    digest.update(chunk)
                    if output is not None:
                        output.write(chunk)
                    remaining -= len(chunk)
                    tar_bytes += len(chunk)
            finally:
                if output is not None:
                    output.close()
            padding = (-member.size) % 512
            if stream.read(padding) != b"\0" * padding:
                raise CustodyError("invalid archive member padding")
            tar_bytes += padding
            inventory[name] = {"sha256": digest.hexdigest(), "bytes": member.size}
    return dict(sorted(inventory.items()))


class _Parts:
    def __init__(self, sink, limits):
        self.sink, self.limits = sink, limits
        self.buffer, self.parts, self.report = b"", [], None
        self.digest, self.count = hashlib.sha256(), 0

    def feed(self, data):
        self.buffer += data
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            self.line(line)
        if len(self.buffer) > self.limits["max_line_bytes"]:
            raise CustodyError("log line exceeds bound")

    def line(self, line):
        if len(line) > self.limits["max_line_bytes"]:
            raise CustodyError("log line exceeds bound")
        if line.startswith(PART.encode()):
            if self.report is not None:
                raise CustodyError("part appeared after final report")
            try:
                part = json.loads(line[len(PART):])
                raw = base64.b64decode(part["base64"], validate=True)
            except (ValueError, KeyError, TypeError) as error:
                raise CustodyError("invalid part encoding") from error
            if (type(part.get("index")) is not int or part["index"] != len(self.parts)
                    or not raw or len(raw) > self.limits["chunk_bytes"]
                    or hashlib.sha256(raw).hexdigest() != part.get("sha256")):
                raise CustodyError("part sequence, size or digest differs")
            if self.parts and self.parts[-1]["bytes"] != self.limits["chunk_bytes"]:
                raise CustodyError("short part is not the final part")
            if self.count + len(raw) > self.limits["max_archive_bytes"]:
                raise CustodyError("reconstructed archive exceeds byte bound")
            self.sink.write(raw)
            self.digest.update(raw)
            self.count += len(raw)
            self.parts.append({"index": part["index"], "sha256": part["sha256"], "bytes": len(raw)})
        elif line.startswith(REPORT.encode()):
            if self.report is not None:
                raise CustodyError("duplicate final report")
            self.report = json.loads(line[len(REPORT):])

    def finish(self):
        if self.buffer:
            self.line(self.buffer)
            self.buffer = b""
        if self.report is None:
            return None
        report = self.report
        if (report.get("schema") != SCHEMA or report.get("part_count") != len(self.parts)
                or report.get("parts") != self.parts or report.get("archive_bytes") != self.count
                or report.get("archive_sha256") != self.digest.hexdigest()):
            raise CustodyError("final multipart archive binding differs")
        return report


def _open_logs(api, path):
    """Stream with the public Api's canonical authentication/header behavior."""
    base = api.base_url or api._resolve()
    headers = {}
    if api.token:
        headers["X-Mrun-Token"] = api.token
    if api.agent_token:
        headers["X-MRun-Agent-Token"] = api.agent_token
    request = urllib.request.Request(base + path, headers=headers, method="GET")
    return urllib.request.urlopen(request, timeout=60.0)


def _request(run_dir):
    request = json.loads((run_dir / "request.json").read_text())
    unsealed = dict(request)
    claimed = unsealed.pop("request_sha256", None)
    if claimed != hashlib.sha256(_json(unsealed).encode()).hexdigest():
        raise CustodyError("request self seal differs")
    for name, digest in request.get("source_hashes", {}).items():
        relative = _relative(name)
        candidates = [run_dir / "payload" / relative.as_posix(), run_dir / relative.as_posix()]
        source = next((path for path in candidates if path.exists()), candidates[0])
        _regular(source)
        if any(parent.is_symlink() for parent in source.parents if parent != run_dir.parent):
            raise CustodyError("staged source path contains a symlink")
        if _sha(source)[0] != digest:
            raise CustodyError("request source digest differs: " + name)
    return request


def _binding(envelope, job, request, job_id):
    nested = envelope.get("report")
    if (envelope.get("job_id") != job_id or envelope.get("request_sha256") != request["request_sha256"]
            or not isinstance(nested, dict) or nested.get("job_id") != job_id
            or nested.get("request_sha256") != request["request_sha256"]
            or job.get("job_id", job.get("id", job_id)) != job_id):
        raise CustodyError("job/report/request identity differs")
    declared = (job.get("config") or {}).get("request_sha256")
    if declared is not None and declared != request["request_sha256"]:
        raise CustodyError("scheduler job request binding differs")


def collect(run_dir: Path, job_id: str, limits: dict | None = None):
    """Collect one direct terminal job, preserving bounded exact raw log bytes.

    The canonical endpoint returns the whole remaining tail. Its HTTP response
    is streamed in 1 MiB reads instead of calling Api.request's unbounded read.
    X-Next-Offset and subsequent offset requests retain canonical byte semantics.
    Existing collection files are never replaced, including partial failures.
    """
    from mdb.job_support import scheduler_api

    bounds, run_dir = _limits(limits), Path(run_dir)
    api = scheduler_api()
    status, raw_job, _ = api.request("GET", f"/api/jobs/{job_id}", timeout_s=60.0)
    if status >= 400:
        raise CustodyError("job read failed: HTTP " + str(status))
    job = json.loads(raw_job)
    retry = (job.get("meta") or {}).get("auto_retry_job_id")
    if retry:
        raise CustodyError("direct collection refuses an automatic retry lineage")
    if job.get("state") not in TERMINAL:
        return {"job_id": job_id, "state": job.get("state"), "collection_status": "not-terminal", "job": job}
    request = _request(run_dir)
    for name in ("job.json", "report.json", "worker.log.gz", "archive.tar.gz", "log-custody.json",
                 "archive-custody.json", "extracted"):
        if (run_dir / name).exists() or (run_dir / name).is_symlink():
            raise CustodyError("collection output exists; refusing overwrite: " + name)
    _save(run_dir / "job.json", job)
    raw_digest, raw_count, log_complete, transport, failure = hashlib.sha256(), 0, False, None, None
    parser = None
    try:
        with (run_dir / "archive.tar.gz").open("xb") as archive:
            parser = _Parts(archive, bounds)
            with (run_dir / "worker.log.gz").open("xb") as stored:
                with gzip.GzipFile(filename="", fileobj=stored, mode="wb", mtime=0) as compressed:
                    offset = 0
                    while True:
                        initial, received = offset, 0
                        with _open_logs(api, f"/api/jobs/{job_id}/logs?offset={offset}&wait_s=1") as response:
                            if getattr(response, "status", 200) >= 400:
                                raise CustodyError("log read failed: HTTP " + str(response.status))
                            while True:
                                chunk = response.read(IO_CHUNK)
                                if not chunk:
                                    break
                                if raw_count + len(chunk) > bounds["max_log_bytes"]:
                                    raise CustodyError("raw log exceeds byte bound")
                                compressed.write(chunk)
                                raw_digest.update(chunk)
                                raw_count += len(chunk)
                                received += len(chunk)
                                parser.feed(chunk)
                            header = response.headers.get("X-Next-Offset", str(initial + received))
                            if int(header) != initial + received:
                                raise CustodyError("scheduler byte offset differs from received log bytes")
                        offset += received
                        if not received:
                            log_complete = True
                            break
                    transport = parser.finish()
        if transport is not None:
            _binding(transport, job, request, job_id)
            inventory = _scan_archive(run_dir / "archive.tar.gz", bounds, transport["archive_root"])
            if (len(inventory) != transport["archive_member_count"]
                    or sum(v["bytes"] for v in inventory.values()) != transport["archive_expanded_bytes"]
                    or inventory.get("files.json", {}).get("sha256") != transport["files_manifest_sha256"]):
                raise CustodyError("archive inventory differs from final report")
            destination = run_dir / "extracted"
            destination.mkdir(exist_ok=False)
            _scan_archive(run_dir / "archive.tar.gz", bounds, transport["archive_root"], destination)
            _check_manifest(destination / transport["archive_root"], inventory, bounds)
            _save(run_dir / "report.json", transport)
    except Exception as error:
        failure = error
    finally:
        compressed_digest, compressed_bytes = _sha(run_dir / "worker.log.gz") if (run_dir / "worker.log.gz").exists() else (None, 0)
        _save(run_dir / "log-custody.json", {"schema": "mdb-flux-family-log-custody-v1", "storage": "gzip",
            "job_id": job_id, "complete": log_complete,
            "job_json_sha256": _sha(run_dir / "job.json")[0],
            "uncompressed_sha256": raw_digest.hexdigest(), "uncompressed_bytes": raw_count,
            "compressed_sha256": compressed_digest, "compressed_bytes": compressed_bytes,
            "scope": "exact scheduler response bytes; no Unicode decoding/re-encoding",
            "error": None if failure is None else f"{type(failure).__name__}: {failure}"})
        archive_digest, archive_bytes = _sha(run_dir / "archive.tar.gz") if (run_dir / "archive.tar.gz").exists() else (None, 0)
        _save(run_dir / "archive-custody.json", {"schema": "mdb-flux-family-archive-custody-v1", "job_id": job_id,
            "complete": transport is not None and failure is None, "archive_sha256": archive_digest,
            "archive_bytes": archive_bytes, "parts": [] if parser is None else parser.parts,
            "error": None if failure is None else f"{type(failure).__name__}: {failure}"})
    if failure is not None:
        raise CustodyError("collection failed; partial custody retained: " + str(failure)) from failure
    if transport is None:
        return {"job_id": job_id, "state": job["state"], "collection_status": "incomplete", "report": False}
    verified = verify(run_dir, limits)
    return {"job_id": job_id, "state": job["state"], "collection_status": "verified",
            "report": str(run_dir / "report.json"), "verification": verified}


class _CompareArchive:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        if self.stream.read(len(data)) != data:
            raise CustodyError("retained archive differs from exact transport bytes")


def verify(run_dir: Path, limits: dict | None = None):
    """Offline verification of transport, archive, extraction and source custody."""
    bounds, run_dir = _limits(limits), Path(run_dir)
    request = _request(run_dir)
    job = json.loads((run_dir / "job.json").read_text())
    envelope = json.loads((run_dir / "report.json").read_text())
    log = json.loads((run_dir / "log-custody.json").read_text())
    archive = json.loads((run_dir / "archive-custody.json").read_text())
    if (log.get("schema") != "mdb-flux-family-log-custody-v1"
            or archive.get("schema") != "mdb-flux-family-archive-custody-v1"
            or log.get("complete") is not True or archive.get("complete") is not True):
        raise CustodyError("partial collection cannot verify as complete")
    if _sha(run_dir / "job.json")[0] != log.get("job_json_sha256"):
        raise CustodyError("retained scheduler job record digest differs")
    job_id = envelope["job_id"]
    _binding(envelope, job, request, job_id)
    if job.get("state") not in TERMINAL or log.get("job_id") != job_id or archive.get("job_id") != job_id:
        raise CustodyError("terminal job/custody identity differs")
    for path in (run_dir / "worker.log.gz", run_dir / "archive.tar.gz"):
        _regular(path)
    if _sha(run_dir / "worker.log.gz") != (log["compressed_sha256"], log["compressed_bytes"]):
        raise CustodyError("compressed raw log digest differs")
    archive_hash = _sha(run_dir / "archive.tar.gz")
    if (archive_hash != (archive["archive_sha256"], archive["archive_bytes"])
            or archive_hash != (envelope["archive_sha256"], envelope["archive_bytes"])
            or archive_hash[1] > bounds["max_archive_bytes"]):
        raise CustodyError("retained archive digest/size differs")
    raw_hash, raw_size = hashlib.sha256(), 0
    with (run_dir / "archive.tar.gz").open("rb") as incoming:
        parser = _Parts(_CompareArchive(incoming), bounds)
        with gzip.open(run_dir / "worker.log.gz", "rb") as stream:
            for chunk in iter(lambda: stream.read(IO_CHUNK), b""):
                raw_size += len(chunk)
                if raw_size > bounds["max_log_bytes"]:
                    raise CustodyError("raw log exceeds byte bound")
                raw_hash.update(chunk)
                parser.feed(chunk)
        parsed = parser.finish()
        if incoming.read(1):
            raise CustodyError("retained archive has untransported trailing bytes")
    if (parsed != envelope or parser.parts != archive["parts"]
            or (raw_hash.hexdigest(), raw_size) != (log["uncompressed_sha256"], log["uncompressed_bytes"])):
        raise CustodyError("raw log/report/part custody differs")
    inventory = _scan_archive(run_dir / "archive.tar.gz", bounds, envelope["archive_root"])
    extracted = run_dir / "extracted"
    roots = list(extracted.iterdir())
    if len(roots) != 1 or roots[0].name != envelope["archive_root"]:
        raise CustodyError("extracted root inventory differs")
    actual = _inventory(roots[0], bounds)
    if (actual != inventory or len(actual) != envelope["archive_member_count"]
            or sum(v["bytes"] for v in actual.values()) != envelope["archive_expanded_bytes"]
            or actual.get("files.json", {}).get("sha256") != envelope["files_manifest_sha256"]):
        raise CustodyError("extracted file inventory differs from archive/report")
    _check_manifest(roots[0], actual, bounds)
    return {"schema": "mdb-flux-family-custody-verification-v1", "job_id": job_id,
            "request_sha256": request["request_sha256"], "archive_sha256": archive_hash[0],
            "archive_bytes": archive_hash[1], "part_count": len(parser.parts), "files": len(actual),
            "mechanics_status": "verified", "model_reexecution": False}


__all__ = ["emit", "collect", "verify", "CustodyError", "DEFAULTS", "PART", "REPORT"]
