"""Transport only: authenticate a completed corrected audit and fetch metadata.

No raw activation/weight member is fetched. Request/job identities are CLI
inputs, so admitting audit-02 never requires changing this frozen collector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
from pathlib import Path, PurePosixPath

BASE = Path("/Users/jakeholl/domains")
QUAL = BASE / "saturn-pub/packages/mdb/qualification"
COLLECTOR_SHA256 = "007098eb72842fe1f5d585f64eae272812303d1c18eb67a17ba43508c646b18d"
READER_SHA256 = "d995c3c5ca3c194a3ce87bfca58e9352d251246d798e21c6e04a7938bd9b5838"
ALLOWED = {".json", ".jsonl", ".py", ".md", ".txt"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def canonical_request(path):
    value = read(path)
    body = {key: row for key, row in value.items() if key != "request_sha256"}
    require(hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest() == value["request_sha256"], "Request identity differs")
    return value


def validate_requests(audit_run, native_run, expected_request):
    audit, native = canonical_request(audit_run / "request.json"), canonical_request(native_run / "request.json")
    require(audit["request_sha256"] == expected_request, "Corrected audit request differs")
    require(audit["source_hashes"]["audit_qwen_row_sign.py"] == READER_SHA256, "Corrected reader pin differs")
    require(audit["config"]["native_request_sha256"] == native["request_sha256"], "Native study request differs")
    native_job = read(native_run / "submission.json")["job_id"]
    require(audit["config"]["source_job"] == native_job, "Native study job differs")
    return audit, native_job


def compact_scope(record, *, max_expanded_bytes, max_archive_bytes):
    files = record["files"]
    require(files and record["verification"]["archive_bytes"] <= max_archive_bytes, "Compact archive budget exceeded")
    for name, item in files.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts and relative.suffix in ALLOWED,
                "Nonmetadata/unsafe compact member refused before retrieval: " + name)
        require(isinstance(item["bytes"], int) and item["bytes"] >= 0, "Invalid compact member size")
    require(sum(row["bytes"] for row in files.values()) <= max_expanded_bytes, "Compact expanded budget exceeded")


def scheduler_url(explicit=None):
    # Generic saved-audit requests intentionally do not contain scheduler_url.
    return explicit or os.environ.get("MRUN_URL") or "http://192.168.68.68:9025"


def collect(audit_run, native_run, output, *, expected_request, encoder_source, pointer=None,
            catalog="/mnt/hdd8tb/mdb/custody/qwen-row-sign-20261106", scheduler=None,
            max_expanded_bytes=64 << 20, max_archive_bytes=32 << 20):
    audit_run, native_run, output, encoder_source = map(lambda path: Path(path).resolve(), (audit_run, native_run, output, encoder_source))
    audit, native_job = validate_requests(audit_run, native_run, expected_request)
    source = QUAL / "collect_flux_remote_directory.py"
    require(sha(source) == COLLECTOR_SHA256, "Qualified collector changed")
    pointer = Path(pointer).resolve() if pointer else output.with_name(output.name + "-custody-pointer.json")
    require(not output.exists() and not output.is_symlink(), "Fresh collection destination required")
    helper = runpy.run_path(str(source))
    # This validates a separately request-sealed collection encoder before any service call.
    encoder, encoder_path, encoder_receipt = helper["custody_encoder"](audit_run, encoder_source)
    os.environ["MRUN_URL"] = scheduler_url(scheduler)
    if not pointer.exists():
        helper["collect"](audit_run, pointer, encoder_source=encoder_source, catalog=catalog)
    record = helper["checked_pointer"](pointer)
    require(Path(read(pointer)["local_source"]).resolve() == audit_run and record["verification"]["request_sha256"] == expected_request,
            "Pointer belongs to another audit")
    compact_scope(record, max_expanded_bytes=max_expanded_bytes, max_archive_bytes=max_archive_bytes)
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "archive.tar.gz"
    helper["retrieve"](pointer, "archive.tar.gz", archive)
    bounds = encoder["_limits"](None)
    inventory = encoder["_scan_archive"](archive, bounds, "binding-minimal-result")
    require(inventory == record["files"], "Compact archive inventory differs")
    extracted = output / "extracted"
    extracted.mkdir()
    encoder["_scan_archive"](archive, bounds, "binding-minimal-result", extracted)
    pack = extracted / "binding-minimal-result"
    encoder["_check_manifest"](pack, inventory, bounds)
    summary = read(pack / "summary.json")
    require(sha(pack / "files.json") == record["files_manifest_sha256"] and summary["schema"] == "mdb-qwen-row-sign-compact-v1"
        and summary["request_sha256"] == expected_request and summary["job_id"] == record["verification"]["job_id"]
        and summary["source_job"] == native_job and read(pack / "request.json") == audit, "Compact source/job/request joins differ")
    receipt = dict(schema="mdb-qwen-row-sign-compact-retrieval-v1", job_id=summary["job_id"], request_sha256=expected_request,
        native_job_id=native_job, reader_sha256=READER_SHA256, collector_sha256=sha(__file__),
        files_manifest_sha256=record["files_manifest_sha256"], pointer_sha256=sha(pointer),
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size, file_count=len(inventory),
        expanded_bytes=sum(row["bytes"] for row in inventory.values()), raw_tensor_arrays_collected=False,
        collection_encoder=encoder_receipt, encoder_source_sha256=sha(encoder_path))
    with (output / "collection-receipt.json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit-run", "native-run", "out", "encoder-source"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-audit-request", required=True)
    parser.add_argument("--pointer", type=Path)
    parser.add_argument("--scheduler-url", help="Defaults to current MRUN_URL or the known Beast scheduler")
    parser.add_argument("--catalog", default="/mnt/hdd8tb/mdb/custody/qwen-row-sign-20261106")
    args = parser.parse_args()
    print(json.dumps(collect(args.audit_run, args.native_run, args.out, expected_request=args.expected_audit_request,
        encoder_source=args.encoder_source, pointer=args.pointer, catalog=args.catalog, scheduler=args.scheduler_url), sort_keys=True))


if __name__ == "__main__":
    main()
