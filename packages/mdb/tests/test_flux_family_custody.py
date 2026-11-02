"""Bounded multipart custody without scheduler or model execution."""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import runpy
import tarfile
from pathlib import Path

import pytest

custody = runpy.run_path(str(Path(__file__).parents[1] / "qualification/flux_family_custody.py"))
LIMITS = {"max_archive_bytes": 32768, "max_expanded_bytes": 32768, "max_files": 16,
          "max_member_bytes": 8192, "chunk_bytes": 97, "max_log_bytes": 65536}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def request_dir(path):
    path.mkdir()
    (path / "payload").mkdir()
    (path / "payload/source.py").write_text("value = 1\n")
    request = {"schema": "test-request", "source_hashes": {"source.py": sha(b"value = 1\n")}}
    request["request_sha256"] = sha(json.dumps(request, sort_keys=True, allow_nan=False).encode())
    write_json(path / "request.json", request)
    return request


def output_dir(path, request):
    path.mkdir()
    (path / "nested").mkdir()
    (path / "nested/result.bin").write_bytes(bytes(range(256)) * 3)
    (path / "summary.json").write_text('{"observed":true}\n')
    manifest = {file.relative_to(path).as_posix(): sha(file.read_bytes())
                for file in path.rglob("*") if file.is_file()}
    write_json(path / "files.json", manifest)
    return {"job_id": "job-one", "request_sha256": request["request_sha256"], "mechanics_status": "valid"}


class Response(io.BytesIO):
    def __init__(self, body, offset):
        super().__init__(body)
        self.headers, self.status = {"X-Next-Offset": str(offset + len(body))}, 200


def scheduler(monkeypatch, raw, *, state="succeeded", response_bytes=83, bad_offset=False):
    import mdb.job_support
    job = {"job_id": "job-one", "state": state}
    class Api:
        def request(self, method, path, **kwargs):
            assert method == "GET" and path == "/api/jobs/job-one"
            return 200, json.dumps(job).encode(), {}
    monkeypatch.setattr(mdb.job_support, "scheduler_api", Api)
    offsets = []
    def open_logs(api, path):
        offset = int(path.split("offset=")[1].split("&")[0])
        offsets.append(offset)
        response = Response(raw[offset:offset + response_bytes], offset)
        if bad_offset:
            response.headers["X-Next-Offset"] = str(offset + 9999)
        return response
    monkeypatch.setitem(custody["collect"].__globals__, "_open_logs", open_logs)
    return offsets


def emitted(tmp_path, capsys):
    run = tmp_path / "run"
    request = request_dir(run)
    output = tmp_path / "worker-result"
    report = output_dir(output, request)
    envelope = custody["emit"](output, report, LIMITS)
    return run, output, envelope, capsys.readouterr().out.encode()


def test_roundtrip_streams_parts_exact_bytes_and_verifies_offline(tmp_path, capsys, monkeypatch):
    run, output, envelope, raw = emitted(tmp_path, capsys)
    assert envelope["part_count"] > 1 and envelope["report"]["mechanics_status"] == "valid"
    raw = b"ordinary bytes \xff\xfe\n" + raw + b"after-report diagnostic\n"
    offsets = scheduler(monkeypatch, raw)
    result = custody["collect"](run, "job-one", LIMITS)
    assert result["collection_status"] == "verified"
    assert offsets[0] == 0 and offsets[-1] == len(raw)
    assert gzip.decompress((run / "worker.log.gz").read_bytes()) == raw
    assert (run / "extracted/worker-result/nested/result.bin").read_bytes() == (output / "nested/result.bin").read_bytes()
    assert custody["verify"](run, LIMITS)["files"] == 3
    with pytest.raises(custody["CustodyError"], match="refusing overwrite"):
        custody["collect"](run, "job-one", LIMITS)


@pytest.mark.parametrize("mutation", ["part-digest", "part-index", "report-job", "report-size", "duplicate-report"])
def test_transport_tamper_rejected_with_partial_custody(tmp_path, capsys, monkeypatch, mutation):
    run, _, _, raw = emitted(tmp_path, capsys)
    lines = raw.decode().splitlines()
    if mutation.startswith("part"):
        row = json.loads(lines[0][len(custody["PART"]):])
        row["sha256" if mutation == "part-digest" else "index"] = "0" * 64 if mutation == "part-digest" else 3
        lines[0] = custody["PART"] + json.dumps(row)
    elif mutation == "duplicate-report":
        lines.append(lines[-1])
    else:
        row = json.loads(lines[-1][len(custody["REPORT"]):])
        row["job_id" if mutation == "report-job" else "archive_bytes"] = "wrong-job" if mutation == "report-job" else 1
        lines[-1] = custody["REPORT"] + json.dumps(row)
    scheduler(monkeypatch, ("\n".join(lines) + "\n").encode())
    with pytest.raises(custody["CustodyError"], match="partial custody retained"):
        custody["collect"](run, "job-one", LIMITS)
    assert (run / "archive.tar.gz").exists() and (run / "worker.log.gz").exists()
    assert json.loads((run / "archive-custody.json").read_text())["complete"] is False


@pytest.mark.parametrize("target", ["extracted/worker-result/nested/result.bin", "archive.tar.gz",
                                    "worker.log.gz", "payload/source.py", "report.json", "job.json"])
def test_offline_verification_rejects_retained_artifact_tamper(tmp_path, capsys, monkeypatch, target):
    run, _, _, raw = emitted(tmp_path, capsys)
    scheduler(monkeypatch, raw)
    custody["collect"](run, "job-one", LIMITS)
    with (run / target).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises((custody["CustodyError"], ValueError)):
        custody["verify"](run, LIMITS)


def test_nonterminal_job_does_not_collect_and_missing_final_report_is_retained(tmp_path, capsys, monkeypatch):
    run, _, _, raw = emitted(tmp_path, capsys)
    offsets = scheduler(monkeypatch, raw, state="running")
    assert custody["collect"](run, "job-one", LIMITS)["collection_status"] == "not-terminal"
    assert not offsets and not (run / "job.json").exists()
    raw = b"\n".join(raw.splitlines()[:-1]) + b"\n"
    scheduler(monkeypatch, raw)
    assert custody["collect"](run, "job-one", LIMITS)["collection_status"] == "incomplete"
    assert (run / "worker.log.gz").exists() and not (run / "extracted").exists()


def archive_at(path, *, name="root/file", payload=b"hello", kind=tarfile.REGTYPE):
    with tarfile.open(path, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo(name)
        member.type, member.size = kind, len(payload) if kind == tarfile.REGTYPE else 0
        member.linkname = "../escape" if kind in {tarfile.SYMTYPE, tarfile.LNKTYPE} else ""
        archive.addfile(member, io.BytesIO(payload) if kind == tarfile.REGTYPE else None)


@pytest.mark.parametrize("name,kind", [("../escape", tarfile.REGTYPE), ("/escape", tarfile.REGTYPE),
    ("root/../escape", tarfile.REGTYPE), ("different/file", tarfile.REGTYPE),
    ("root/link", tarfile.SYMTYPE), ("root/link", tarfile.LNKTYPE), ("root/dir", tarfile.DIRTYPE)])
def test_archive_rejects_paths_and_nonregular_members_before_extraction(tmp_path, name, kind):
    path = tmp_path / "bad.tar.gz"
    archive_at(path, name=name, kind=kind)
    with pytest.raises(custody["CustodyError"]):
        custody["_scan_archive"](path, custody["_limits"](LIMITS), "root")
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("limit,value", [("max_member_bytes", 4), ("max_expanded_bytes", 4)])
def test_archive_member_and_expanded_bounds(tmp_path, limit, value):
    path = tmp_path / "large.tar.gz"
    archive_at(path)
    with pytest.raises(custody["CustodyError"], match="bound"):
        custody["_scan_archive"](path, custody["_limits"]({**LIMITS, limit: value}), "root")


def test_duplicate_members_and_file_count_are_rejected(tmp_path):
    path = tmp_path / "duplicate.tar.gz"
    with tarfile.open(path, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
        for _ in range(2):
            member = tarfile.TarInfo("root/file")
            archive.addfile(member, io.BytesIO())
    with pytest.raises(custody["CustodyError"], match="duplicate"):
        custody["_scan_archive"](path, custody["_limits"](LIMITS), "root")
    with tarfile.open(path, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
        for name in ("one", "two"):
            archive.addfile(tarfile.TarInfo("root/" + name), io.BytesIO())
    with pytest.raises(custody["CustodyError"], match="count"):
        custody["_scan_archive"](path, custody["_limits"]({**LIMITS, "max_files": 1}), "root")


def test_emitter_rejects_symlink_and_mismatched_manifest(tmp_path):
    output = tmp_path / "output"
    request = request_dir(tmp_path / "run")
    report = output_dir(output, request)
    (output / "link").symlink_to(output / "summary.json")
    with pytest.raises(custody["CustodyError"], match="symlink"):
        custody["emit"](output, report, LIMITS)
    (output / "link").unlink()
    (output / "summary.json").write_text("changed")
    with pytest.raises(custody["CustodyError"], match="manifest"):
        custody["emit"](output, report, LIMITS)


def test_log_size_and_offsets_are_bounded(tmp_path, capsys, monkeypatch):
    run, _, _, raw = emitted(tmp_path, capsys)
    scheduler(monkeypatch, raw)
    with pytest.raises(custody["CustodyError"], match="raw log exceeds"):
        custody["collect"](run, "job-one", {**LIMITS, "max_log_bytes": 10})
    other = tmp_path / "other"
    request_dir(other)
    scheduler(monkeypatch, raw, bad_offset=True)
    with pytest.raises(custody["CustodyError"], match="byte offset"):
        custody["collect"](other, "job-one", LIMITS)


def test_streaming_seam_keeps_canonical_auth_headers_out_of_evidence(monkeypatch):
    captured = {}
    class Api:
        base_url, token, agent_token = "http://scheduler", "secret-user", "secret-agent"
    def opened(request, timeout):
        captured.update({"headers": dict(request.header_items()), "url": request.full_url, "timeout": timeout})
        return Response(b"", 0)
    monkeypatch.setattr(custody["_open_logs"].__globals__["urllib"].request, "urlopen", opened)
    custody["_open_logs"](Api(), "/api/jobs/job-one/logs?offset=0&wait_s=1")
    assert {key.lower(): value for key, value in captured["headers"].items()} == {
        "x-mrun-token": "secret-user", "x-mrun-agent-token": "secret-agent"}
    assert captured["url"].endswith("offset=0&wait_s=1")


def test_oversized_line_and_archive_are_rejected_without_unbounded_reads(tmp_path):
    parser = custody["_Parts"](io.BytesIO(), custody["_limits"](LIMITS))
    with pytest.raises(custody["CustodyError"], match="line exceeds"):
        parser.feed(b"x" * (custody["_limits"](LIMITS)["max_line_bytes"] + 1))
    parser = custody["_Parts"](io.BytesIO(), custody["_limits"]({**LIMITS, "max_archive_bytes": 1}))
    line = custody["PART"] + json.dumps({"index": 0, "sha256": sha(b"xx"), "base64": base64.b64encode(b"xx").decode()})
    with pytest.raises(custody["CustodyError"], match="archive exceeds"):
        parser.feed(line.encode() + b"\n")


@pytest.mark.parametrize("limit,value", [("max_archive_bytes", 1), ("max_member_bytes", 10),
    ("max_expanded_bytes", 10), ("max_files", 1), ("max_log_bytes", 1)])
def test_producer_enforces_declared_bounds(tmp_path, limit, value):
    request = request_dir(tmp_path / "run")
    output = tmp_path / "output"
    report = output_dir(output, request)
    with pytest.raises(custody["CustodyError"], match="bound"):
        custody["emit"](output, report, {**LIMITS, limit: value})


def test_extraction_refuses_existing_file_and_symlink_directory(tmp_path):
    path = tmp_path / "archive.tar.gz"
    archive_at(path)
    destination = tmp_path / "extracted"
    (destination / "root").mkdir(parents=True)
    existing = destination / "root/file"
    existing.write_bytes(b"preserve me")
    with pytest.raises(FileExistsError):
        custody["_scan_archive"](path, custody["_limits"](LIMITS), "root", destination)
    assert existing.read_bytes() == b"preserve me"
    existing.unlink()
    (destination / "root").rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "root").symlink_to(outside, target_is_directory=True)
    with pytest.raises(custody["CustodyError"], match="symlink"):
        custody["_scan_archive"](path, custody["_limits"](LIMITS), "root", destination)
    assert not (outside / "file").exists()
