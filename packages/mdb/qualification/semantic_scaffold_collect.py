"""Collect only a named finite semantic job and verify its bounded archive."""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(value):
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def collect(root):
    from mdb.job_support import guarded_job_receipt, terminal_job
    request = json.loads((root / "request.json").read_text())
    job_id = json.loads((root / "submission.json").read_text())["job_id"]
    identifier, job, logs = terminal_job(job_id)
    if job.get("state") not in {"succeeded", "failed", "timeout", "killed_ram", "killed_vram", "cancelled", "lost"}:
        print(json.dumps({"job_id": identifier, "state": job.get("state")}))
        return
    (root / "worker.log.gz").write_bytes(gzip.compress(logs.encode(), mtime=0))
    save(root / "job.json", job)
    save(root / "custody.json", guarded_job_receipt(identifier))
    marker = "SCAFFOLD_REPORT_V1 "
    encoded = next((s[len(marker):] for s in logs.splitlines() if s.startswith(marker)), None)
    if encoded is None:
        print(json.dumps({"job_id": identifier, "state": job.get("state"), "report": False,
                          "tail": "\n".join(logs.splitlines()[-15:])}))
        return
    report = json.loads(gzip.decompress(base64.b64decode(encoded, validate=True)))
    count, chunks = None, {}
    for line in logs.splitlines():
        if line.startswith("SCAFFOLD_ARCHIVE_V1 "):
            head, body = line[len("SCAFFOLD_ARCHIVE_V1 "):].split(" ", 1)
            index, total = map(int, head.split("/"))
            if index in chunks or (count is not None and total != count):
                raise ValueError("inconsistent archive chunks")
            chunks[index], count = base64.b64decode(body, validate=True), total
    if count is None or sorted(chunks) != list(range(count)):
        raise ValueError("incomplete artifact transport")
    raw = b"".join(chunks[i] for i in range(count))
    if (len(raw) > request["workload_geometry"]["output_archive_max_bytes"]
        or len(raw) != report["archive"]["bytes"] or sha(raw) != report["archive"]["sha256"]):
        raise ValueError("archive identity/size mismatch")
    if request["request_sha256"] != report["request_sha256"]:
        raise ValueError("report request identity mismatch")
    (root / "result.tar.gz").write_bytes(raw)
    results = root / "results"
    results.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        if sum(p.size for p in archive.getmembers()) > request["workload_geometry"]["output_expanded_max_bytes"]:
            raise ValueError("expanded artifact transport exceeds declared limit")
        for member in archive.getmembers():
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe archive member")
            target = results / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(archive.extractfile(member).read())
    manifest = json.loads((results / "output-manifest.json").read_text())
    for entry in manifest["files"]:
        path = results / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha(path.read_bytes()) != entry["sha256"]:
            raise ValueError("retained inventory differs: " + entry["path"])
    save(root / "report.json", report)
    print(json.dumps({"job_id": identifier, "state": job.get("state"), "archive_bytes": len(raw),
                      "files_verified": len(manifest["files"]), "mechanics_status": report["mechanics_status"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    collect(parser.parse_args().root.resolve())
