"""Reuse qualified custody transport; retrieve only the saved audit metadata."""
import hashlib
import json
import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior = ROOT.parent / "mdb-qwen-row-sign-20261106/collect_compact.py"
    generic = BASE / "saturn-pub/packages/mdb/qualification/collect_flux_remote_directory.py"
    if sha(prior) != "b621d8e12dc06476abf356643f22693a48a5a52bd43c7684d66e788126475015" or sha(generic) != "007098eb72842fe1f5d585f64eae272812303d1c18eb67a17ba43508c646b18d":
        raise ValueError("Qualified transport/scope sources changed")
    scope, core = runpy.run_path(str(prior)), runpy.run_path(str(generic))
    request = scope["canonical_request"](ROOT / "audit-01/request.json")
    native = scope["canonical_request"](ROOT / "run-01/request.json")
    if native["request_sha256"] != "d0bb13015d8bcb735f0de5cc2a4587d8aa96eb13b7af51d6c6af391015cdf827":
        raise ValueError("Original native request changed")
    if request["source_hashes"]["audit_qwen_error_consumer.py"] != "c0adfae449b426c52c91a5cb2e05fd27453cdd704354dbd7dd465a194d19723b":
        raise ValueError("Qualified saved reader changed")
    if request["config"]["native_request_sha256"] != native["request_sha256"] or request["config"]["source_job"] != "job-f74bf3f78621":
        raise ValueError("Exact original source job/request differs")
    os.environ["MRUN_URL"] = "http://192.168.68.68:9025"
    encoder_source = ROOT.parent / "mdb-flux-functional-decoder-20261106/recovery-collection-01"
    pointer = ROOT / "audit-custody-pointer-v1.json"
    if not pointer.exists():
        core["collect"](ROOT / "audit-01", pointer, encoder_source=encoder_source,
                        catalog="/mnt/hdd8tb/mdb/custody/qwen-error-consumer-20261106")
    record = core["checked_pointer"](pointer)
    if record["verification"]["request_sha256"] != request["request_sha256"]:
        raise ValueError("Collected saved-audit request differs")
    scope["compact_scope"](record, max_expanded_bytes=64 << 20, max_archive_bytes=32 << 20)
    encoder = runpy.run_path(str(encoder_source / "payload/flux_family_custody.py"))
    output = ROOT / "compact-collection-01"
    output.mkdir(exist_ok=False)
    archive = output / "archive.tar.gz"
    core["retrieve"](pointer, "archive.tar.gz", archive)
    bounds = encoder["_limits"](None)
    members = encoder["_scan_archive"](archive, bounds, "binding-minimal-result")
    if members != record["files"]:
        raise ValueError("Exact compact member inventory differs")
    extracted = output / "extracted"
    extracted.mkdir()
    encoder["_scan_archive"](archive, bounds, "binding-minimal-result", extracted)
    pack = extracted / "binding-minimal-result"
    encoder["_check_manifest"](pack, members, bounds)
    summary = json.loads((pack / "summary.json").read_text())
    if summary["schema"] != "mdb-qwen-error-consumer-compact-v1" or summary["request_sha256"] != request["request_sha256"] or summary["source_job"] != "job-f74bf3f78621":
        raise ValueError("Compact publication/source/schema differs")
    receipt = dict(schema="mdb-error-consumer-compact-retrieval-v1", job_id=summary["job_id"], request_sha256=request["request_sha256"],
        native_job_id=summary["source_job"], files_manifest_sha256=sha(pack / "files.json"),
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size, file_count=len(members),
        expanded_bytes=sum(value["bytes"] for value in members.values()), pointer_sha256=sha(pointer),
        collector_sha256=sha(Path(__file__)), generic_transport_sha256=sha(generic), metadata_scope_sha256=sha(prior),
        raw_tensor_arrays_collected=False, model_calls=0)
    (output / "collection-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
