"""Collect the authenticated saved-only Qwen shortcut compact; never fetch raw arrays."""
import hashlib
import json
import os
import runpy
from pathlib import Path

BASE = Path("/Users/jakeholl/domains")
ROOT = Path(__file__).resolve().parent
QUAL = BASE / "saturn-pub/packages/mdb/qualification"
ENCODER = BASE / "artifacts/mdb-flux-functional-decoder-20261106/recovery-collection-01"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = QUAL / "collect_flux_remote_directory.py"
    if sha(source) != "007098eb72842fe1f5d585f64eae272812303d1c18eb67a17ba43508c646b18d":
        raise ValueError("Qualified collector changed")
    request = json.loads((ROOT / "audit-01/request.json").read_text())
    unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("Audit request changed")
    if (request["config"]["native_request_sha256"] != json.loads((ROOT / "run-02/request.json").read_text())["request_sha256"]
            or request["source_hashes"]["audit_qwen_sign_shortcut.py"] != "d33613bbb1d71ee3b2007fb007e5266acaf141994091264bed50d27342237e2c"):
        raise ValueError("Original Qwen shortcut cohort differs")
    os.environ["MRUN_URL"] = "http://192.168.68.68:9025"
    collect = runpy.run_path(str(source))
    pointer = ROOT / "audit-custody-pointer-v1.json"
    if not pointer.exists():
        collect["collect"](ROOT / "audit-01", pointer, encoder_source=ENCODER,
                           catalog="/mnt/hdd8tb/mdb/custody/qwen-sign-shortcut-20261106")
    record = collect["checked_pointer"](pointer)
    encoder = runpy.run_path(str(ENCODER / "payload/flux_family_custody.py"))
    output = ROOT / "compact-collection-01"
    output.mkdir(exist_ok=False)
    archive = output / "archive.tar.gz"
    collect["retrieve"](pointer, "archive.tar.gz", archive)
    bounds = encoder["_limits"](None)
    inventory = encoder["_scan_archive"](archive, bounds, "binding-minimal-result")
    if inventory != record["files"] or any(n.endswith((".npy", ".pt", ".pth", ".safetensors")) for n in inventory):
        raise ValueError("Compact inventory or payload scope differs")
    extracted = output / "extracted"
    extracted.mkdir()
    encoder["_scan_archive"](archive, bounds, "binding-minimal-result", extracted)
    pack = extracted / "binding-minimal-result"
    encoder["_check_manifest"](pack, inventory, bounds)
    if sha(pack / "files.json") != record["files_manifest_sha256"]:
        raise ValueError("Compact wrapper differs")
    receipt = dict(schema="mdb-qwen-shortcut-compact-retrieval-v1", job_id=record["verification"]["job_id"],
                   request_sha256=record["verification"]["request_sha256"],
                   files_manifest_sha256=record["files_manifest_sha256"], pointer_sha256=sha(pointer),
                   archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                   expanded_bytes=sum(r["bytes"] for r in inventory.values()), file_count=len(inventory),
                   raw_tensor_arrays_collected=False, collection_encoder=record["recovery_encoder"])
    with (output / "collection-receipt.json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: receipt[k] for k in ("file_count", "expanded_bytes", "archive_bytes", "files_manifest_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
