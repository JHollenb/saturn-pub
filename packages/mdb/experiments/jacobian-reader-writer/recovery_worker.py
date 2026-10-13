#!/usr/bin/env python3
"""Model-free, same-host transport for a timed-out J-space assay's sealed files."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import tarfile
from pathlib import Path

META_MARKER = "MDB_JSPACE_RECOVERY_META_BASE64 "
CHUNK_MARKER = "MDB_JSPACE_RECOVERY_BUNDLE_CHUNK "
COUNT_MARKER = "MDB_JSPACE_RECOVERY_BUNDLE_CHUNKS "
SHA_MARKER = "MDB_JSPACE_RECOVERY_BUNDLE_SHA256 "
RAW_CHUNK_BYTES = 65_535


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_files(root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"recovery source is not a real directory: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise RuntimeError(f"unsafe recovery source member: {path}")
        if stat.S_ISREG(mode):
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_raw = args.config.read_bytes()
    config = json.loads(config_raw)
    payload_manifest = json.loads(Path("payload-manifest.json").read_text())
    for name in ("recovery_worker.py", "recovery.json", "expected-protocol.json"):
        if sha256_file(Path(name)) != payload_manifest.get(name):
            raise RuntimeError(f"staged recovery payload identity mismatch: {name}")
    origin = Path(config["origin_workdir"])
    if origin != Path(f"/mnt/hdd8tb/mrun-work/{config['origin_job_id']}"):
        raise RuntimeError("origin workdir is not the sealed job-local path")
    if not origin.is_dir() or origin.is_symlink():
        raise RuntimeError("origin workdir is unavailable on this host")

    expected = config["expected_source_sha256"]
    for name in ("real_bf16_worker.py", "protocol.json", config["wheel_name"]):
        path = origin / name
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected[name]:
            raise RuntimeError(f"origin source identity mismatch: {name}")

    output = origin / "output"
    lens = output / "lens"
    exported = output / "bench-export"
    fallback = output / "bench"
    if not lens.is_dir():
        raise RuntimeError("origin output/lens is missing")
    if exported.is_dir():
        bench = exported
        recovery_status = "export-directory-recovered"
    elif fallback.is_dir():
        bench = fallback
        recovery_status = "live-bench-fallback-recovered"
    else:
        raise RuntimeError("origin has neither output/bench-export nor output/bench")

    selected = [lens, bench]
    inventory: list[dict[str, object]] = []
    selected_files: list[Path] = []
    for root in selected:
        for path in regular_files(root):
            selected_files.append(path)
            inventory.append(
                {
                    "path": path.relative_to(origin).as_posix(),
                    "bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "sha256": sha256_file(path),
                }
            )
    protocol = origin / "protocol.json"
    manifest = {
        "schema": "mdb-jspace-recovery-manifest-v1",
        "origin_job_id": config["origin_job_id"],
        "origin_workdir": str(origin),
        "origin_job_record_sha256": config["origin_job_record_sha256"],
        "recovery_status": recovery_status,
        "bench_source": bench.relative_to(origin).as_posix(),
        "output_presence": {
            name: {
                "exists": path.exists(),
                "is_directory": path.is_dir(),
                "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
            }
            for name, path in {
                "lens": lens,
                "bench-export": exported,
                "bench": fallback,
                "report.json": output / "report.json",
                "protocol.json": output / "protocol.json",
            }.items()
        },
        "expected_source_sha256": expected,
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "inventory": inventory,
    }
    manifest_path = Path("recovery-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        "MDB_JSPACE_RECOVERY_PROGRESS "
        + json.dumps(
            {
                "phase": "archive-start",
                "selected_file_count": len(inventory),
                "selected_bytes": sum(int(row["bytes"]) for row in inventory),
                "output_presence": manifest["output_presence"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    bundle = Path("recovery-bundle.tar.gz")
    with tarfile.open(bundle, mode="w:gz") as archive:
        for path in selected_files:
            archive.add(path, arcname=path.relative_to(origin).as_posix(), recursive=False)
        archive.add(protocol, arcname="source/protocol.json", recursive=False)
        archive.add(manifest_path, arcname=manifest_path.name, recursive=False)
    bundle_sha = sha256_file(bundle)
    metadata = {
        "schema": "mdb-jspace-recovery-transport-v1",
        "origin_job_id": config["origin_job_id"],
        "recovery_status": recovery_status,
        "bundle_sha256": bundle_sha,
        "bundle_bytes": bundle.stat().st_size,
        "inventory_count": len(inventory),
        "manifest_sha256": sha256_file(manifest_path),
    }
    print(META_MARKER + base64.b64encode(json.dumps(metadata, sort_keys=True).encode()).decode())
    count = 0
    with bundle.open("rb") as handle:
        while raw := handle.read(RAW_CHUNK_BYTES):
            print(CHUNK_MARKER + f"{count} " + base64.b64encode(raw).decode())
            count += 1
    print(COUNT_MARKER + str(count))
    print(SHA_MARKER + bundle_sha)


if __name__ == "__main__":
    os.environ["PYTHONNOUSERSITE"] = "1"
    main()
