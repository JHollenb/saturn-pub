"""Safely unpack an owned job archive and independently audit exported closures."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import tarfile

from mdb.bench_verify import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--archive", default="export.tar.gz")
    parser.add_argument("--allow-missing-report", action="store_true")
    args = parser.parse_args()
    job = args.job_dir.resolve()
    report_path = job / "report.json"
    if not report_path.exists() and not args.allow_missing_report:
        raise RuntimeError("terminal report required; diagnostic audits must explicitly allow its absence")
    archive = job / args.archive
    digest = sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    destination = job / "posttermination"
    destination.mkdir(exist_ok=False)
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unsafe archive member: {member.name}")
        handle.extractall(destination, members=members, filter="data")
    metadata = {"sha256": digest.hexdigest(), "bytes": archive.stat().st_size,
                "members": len(members)}
    (job / "archive.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    for selected in sorted(destination.rglob("selected.json")):
        result = verify(selected.parent)
        rows.append({"root": str(selected.parent.relative_to(destination)), "verification": result})
        print(json.dumps({"root": rows[-1]["root"], "counts": result["counts"]}), flush=True)
    if not rows:
        raise RuntimeError("archive has no exported sessions")
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if "closed_verification" in report:
        if len(rows) != 1 or rows[0]["verification"] != report["closed_verification"]:
            raise RuntimeError("posttermination closure differs from closed worker report")
    incomplete = [str(p.parent.relative_to(destination)) for p in destination.rglob("payloads")
        if (p.parent / "artifacts").is_dir() and not (p.parent / "selected.json").exists()]
    (job / "posttermination-verification.json").write_text(json.dumps(
        {"schema": "mdb-job-archive-audit-v1", "archive": metadata, "rows": rows,
         "terminal_report_present": bool(report), "incomplete_stores": incomplete}, indent=2) + "\n")


if __name__ == "__main__":
    main()
