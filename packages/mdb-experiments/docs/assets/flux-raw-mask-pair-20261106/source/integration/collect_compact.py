"""Collect one authenticated saved-only report; leave full states near compute."""
import hashlib
import json
import os
import runpy
from pathlib import Path

BASE = Path("/Users/jakeholl/domains")
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "audit-03"
POINTER = ROOT / "audit-custody-pointer-v1.json"
COLLECTION = ROOT / "compact-collection-01"
QUALIFICATION = BASE / "saturn-pub/packages/mdb/qualification"
ENCODER_SOURCE = BASE / "artifacts/mdb-flux-functional-decoder-20261106/recovery-collection-01"
AUDITOR_SHA256 = "70408046bb83f1c0719addb57191c7d4bdf4c5a1121195c299c4cb493cf11d88"
NATIVE_REQUEST = "783e8dc0c0a731c458eed739c5d8b68c932bf622ef5f426d7f484596a8c011dd"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    collector_source = QUALIFICATION / "collect_flux_remote_directory.py"
    if sha(collector_source) != "007098eb72842fe1f5d585f64eae272812303d1c18eb67a17ba43508c646b18d":
        raise ValueError("Qualified collector changed")
    request = read(SOURCE / "request.json")
    actual = hashlib.sha256(json.dumps(
        {key: value for key, value in request.items() if key != "request_sha256"},
        sort_keys=True, allow_nan=False,
    ).encode()).hexdigest()
    if (actual != request["request_sha256"]
            or request["config"]["native_request_sha256"] != NATIVE_REQUEST
            or request["source_hashes"]["audit_flux_raw_mask_pair.py"] != AUDITOR_SHA256):
        raise ValueError("Frozen original/audit specimen differs")
    os.environ["MRUN_URL"] = "http://192.168.68.68:9025"
    # This encoder is an explicit transport dependency from a prior sealed
    # request. It does not replace either the model or saved-audit payload.
    collector = runpy.run_path(str(collector_source))
    if not POINTER.exists():
        collector["collect"](
            SOURCE, POINTER, encoder_source=ENCODER_SOURCE,
            catalog="/mnt/hdd8tb/mdb/custody/raw-mask-pair-20261106",
        )
    record = collector["checked_pointer"](POINTER)
    encoder = runpy.run_path(str(ENCODER_SOURCE / "payload/flux_family_custody.py"))
    COLLECTION.mkdir(exist_ok=False)
    archive = COLLECTION / "archive.tar.gz"
    collector["retrieve"](POINTER, "archive.tar.gz", archive)
    bounds = encoder["_limits"](None)
    inventory = encoder["_scan_archive"](archive, bounds, "binding-minimal-result")
    if inventory != record["files"]:
        raise ValueError("Compact archive inventory differs")
    extracted = COLLECTION / "extracted"
    extracted.mkdir()
    encoder["_scan_archive"](archive, bounds, "binding-minimal-result", extracted)
    pack = extracted / "binding-minimal-result"
    encoder["_check_manifest"](pack, inventory, bounds)
    if sha(pack / "files.json") != record["files_manifest_sha256"]:
        raise ValueError("Compact file manifest differs")
    if any(name.endswith((".npy", ".pt", ".safetensors", ".pth")) for name in inventory):
        raise ValueError("Raw array or weight payload in compact publication")
    receipt = {
        "schema": "mdb-raw-mask-pair-compact-retrieval-v1",
        "job_id": record["verification"]["job_id"],
        "request_sha256": record["verification"]["request_sha256"],
        "files_manifest_sha256": record["files_manifest_sha256"],
        "pointer_sha256": sha(POINTER), "archive_sha256": sha(archive),
        "expanded_bytes": sum(row["bytes"] for row in inventory.values()),
        "archive_bytes": archive.stat().st_size, "file_count": len(inventory),
        "raw_tensor_arrays_collected": False,
        "collection_encoder": record["recovery_encoder"],
    }
    with (COLLECTION / "collection-receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
