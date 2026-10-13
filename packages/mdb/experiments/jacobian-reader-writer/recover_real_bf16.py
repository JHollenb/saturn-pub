#!/usr/bin/env python3
"""Launch and collect one model-free same-host recovery of the BF16 assay."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from mdb.job_support import (  # noqa: E402
    guarded_job_receipt,
    guarded_mrun_launch,
    one_lease_scheduling_contract,
    terminal_job,
)

ORIGIN_JOB = "job-4ba3cc18d86a"
ORIGIN_WORKDIR = f"/mnt/hdd8tb/mrun-work/{ORIGIN_JOB}"
ORIGIN = HERE / "results" / ORIGIN_JOB
RESULT_ROOT = HERE / "results" / f"recovery-{ORIGIN_JOB}"
WORKER = HERE / "recovery_worker.py"
WHEEL_NAME = "mdb_runtime-0.1.0-py3-none-any.whl"
EXPECTED_SOURCE_SHA256 = {
    "real_bf16_worker.py": "82f1a5917327864d7c63cfdebf5883af9e485017faf6a00d0bb5c686c33a3d21",
    "protocol.json": "2f9a406ef66d3096daca9f60b66bef485fa04b9ca910579eaa1f899d1a41b20c",
    WHEEL_NAME: "1120c5299ffe096a76db84ceaa71cd44348db8f03a46e6bf17e72e433b96ea2e",
}
META_MARKER = "MDB_JSPACE_RECOVERY_META_BASE64 "
CHUNK_MARKER = "MDB_JSPACE_RECOVERY_BUNDLE_CHUNK "
COUNT_MARKER = "MDB_JSPACE_RECOVERY_BUNDLE_CHUNKS "
SHA_MARKER = "MDB_JSPACE_RECOVERY_BUNDLE_SHA256 "
TERMINAL = {"succeeded", "failed", "killed_ram", "killed_vram", "timeout", "cancelled", "lost"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"immutable recovery evidence conflict: {path}")
        return
    path.write_bytes(data)


def origin_custody() -> tuple[dict[str, Any], dict[str, Any]]:
    required = (
        "job-record.json",
        "job-receipt.json",
        "submitted-request.json",
        "payload-manifest.json",
    )
    if any(not (ORIGIN / name).is_file() for name in required):
        raise RuntimeError("collected origin custody is incomplete")
    record = json.loads((ORIGIN / "job-record.json").read_text())
    if record.get("state") != "timeout" or record.get("assigned_host") != "beast":
        raise RuntimeError("origin is not the retained beast timeout")
    source_hashes = (record.get("config") or {}).get("source_sha256")
    if not isinstance(source_hashes, dict) or any(
        source_hashes.get(name) != digest
        for name, digest in EXPECTED_SOURCE_SHA256.items()
    ):
        raise RuntimeError("origin scheduler record has unexpected source identities")
    payload_manifest = json.loads((ORIGIN / "payload-manifest.json").read_text())
    if any(
        payload_manifest.get(name) != digest
        for name, digest in EXPECTED_SOURCE_SHA256.items()
    ):
        raise RuntimeError("origin payload custody has unexpected source identities")
    for name, digest in EXPECTED_SOURCE_SHA256.items():
        if sha256_file(ORIGIN / "submitted-payload" / name) != digest:
            raise RuntimeError(f"origin submitted payload differs at {name}")
    custody = {name: sha256_file(ORIGIN / name) for name in required}
    custody["job.log"] = sha256_file(ORIGIN / "job.log")
    return record, custody


def specification() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    _, custody = origin_custody()
    config = {
        "schema": "mdb-jspace-recovery-config-v1",
        "origin_job_id": ORIGIN_JOB,
        "origin_workdir": ORIGIN_WORKDIR,
        "origin_job_record_sha256": custody["job-record.json"],
        "expected_source_sha256": EXPECTED_SOURCE_SHA256,
        "wheel_name": WHEEL_NAME,
    }
    request: dict[str, Any] = {
        "schema": "mdb-jspace-recovery-request-v1",
        "experiment": "mdb-jspace-real-bf16-recovery",
        "task_family": "analysis",
        "device": "cpu",
        "model": None,
        "model_free": True,
        "needs": {},
        "origin_job_id": ORIGIN_JOB,
        "origin_host": "beast",
        "origin_workdir": ORIGIN_WORKDIR,
        "origin_custody_sha256": custody,
        "expected_source_sha256": EXPECTED_SOURCE_SHA256,
        "source_sha256": {"recovery_worker.py": sha256_file(WORKER)},
        "scheduler": {"strict_preflight": True, "retry_on_kill": False, "hard_pin": "beast"},
        "scheduling_contract": one_lease_scheduling_contract(
            model_free=True,
            model_loads=0,
            logical_operations=1,
            logical_checkpoint_replays=0,
            max_local_work_items=1,
            checkpoint_policy=(
                "one stdlib-only CPU worker; read retained origin output once, archive and stream; "
                "no model load, mutation, replay, or retry"
            ),
            extra={"task_family": "analysis", "origin_job_id": ORIGIN_JOB},
        ),
    }
    request["request_fingerprint"] = canonical_sha256(request)
    command = ["/usr/bin/python3", "-I", "-u", "recovery_worker.py", "--config", "recovery.json"]
    return request, config, command


def stage_payload(destination: Path, request: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(WORKER, destination / WORKER.name)
    (destination / "recovery.json").write_bytes(json_bytes(config))
    shutil.copyfile(HERE / "protocol.json", destination / "expected-protocol.json")
    hashes = {path.name: sha256_file(path) for path in sorted(destination.iterdir())}
    if hashes["expected-protocol.json"] != EXPECTED_SOURCE_SHA256["protocol.json"]:
        raise RuntimeError("client protocol no longer matches the origin")
    (destination / "payload-manifest.json").write_bytes(json_bytes(hashes))
    hashes["payload-manifest.json"] = sha256_file(destination / "payload-manifest.json")
    return hashes


def submit() -> str:
    request, config, command = specification()
    custody = RESULT_ROOT / "source-custody" / request["request_fingerprint"][:20]
    with tempfile.TemporaryDirectory(prefix="mdb-jspace-recovery-") as temporary:
        payload = Path(temporary) / "payload"
        hashes = stage_payload(payload, request, config)
        for path in payload.iterdir():
            write_once(custody / "payload" / path.name, path.read_bytes())
        write_once(custody / "submitted-request.json", json_bytes(request))
        write_once(custody / "payload-manifest.json", json_bytes(hashes))
        job = guarded_mrun_launch(
            command,
            request=request,
            experiment=request["experiment"],
            task="analysis",
            model=None,
            payload=payload,
            pin="beast",
            gpu=False,
            needs={},
            mrun_config=request,
            timeout_s=300,
            detach=True,
            retry_on_kill=False,
            preflight=True,
            note=(
                f"Model-free recovery of retained {ORIGIN_JOB} lens and Bench export; "
                "one beast-local stdlib CPU lease, no model, mutation, or retry."
            ),
        )
    write_once(custody / f"{job}.json", json_bytes({"job_id": job}))
    print(json.dumps({"job": job, "custody": str(custody)}, sort_keys=True))
    return job


def decode(logs: str) -> tuple[dict[str, Any], bytes, str]:
    metadata: list[dict[str, Any]] = []
    chunks: dict[int, bytes] = {}
    counts: list[int] = []
    digests: list[str] = []
    for line in logs.splitlines():
        if line.startswith(META_MARKER):
            metadata.append(json.loads(base64.b64decode(line[len(META_MARKER) :], validate=True)))
        elif line.startswith(CHUNK_MARKER):
            index, encoded = line[len(CHUNK_MARKER) :].split(maxsplit=1)
            number = int(index)
            if number in chunks:
                raise RuntimeError("duplicate recovery chunk")
            chunks[number] = base64.b64decode(encoded, validate=True)
        elif line.startswith(COUNT_MARKER):
            counts.append(int(line[len(COUNT_MARKER) :]))
        elif line.startswith(SHA_MARKER):
            digests.append(line[len(SHA_MARKER) :].strip())
    if len(metadata) != 1 or len(counts) != 1 or len(digests) != 1:
        raise RuntimeError("recovery transport completion markers are incomplete")
    if sorted(chunks) != list(range(counts[0])):
        raise RuntimeError("recovery transport chunks are incomplete")
    bundle = b"".join(chunks[index] for index in range(counts[0]))
    if sha256_bytes(bundle) != digests[0] or metadata[0].get("bundle_sha256") != digests[0]:
        raise RuntimeError("recovery bundle SHA-256 mismatch")
    return metadata[0], bundle, digests[0]


def safe_extract(bundle: bytes, destination: Path) -> dict[str, str]:
    if len(bundle) > 512 << 20:
        raise RuntimeError("compressed recovery bundle exceeds 512 MiB")
    allowed = ("output/lens/", "output/bench-export/", "output/bench/", "source/protocol.json")
    hashes: dict[str, str] = {}
    expanded = 0
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            name = Path(member.name)
            relative = name.as_posix()
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or relative in hashes
                or not (relative == "recovery-manifest.json" or relative.startswith(allowed))
            ):
                raise RuntimeError("unsafe recovery bundle member")
            expanded += member.size
            if expanded > 2 << 30:
                raise RuntimeError("expanded recovery bundle exceeds 2 GiB")
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError("recovery bundle member cannot be read")
            data = handle.read()
            if len(data) != member.size:
                raise RuntimeError("truncated recovery bundle member")
            hashes[relative] = sha256_bytes(data)
            write_once(destination / name, data)
    if "recovery-manifest.json" not in hashes or "source/protocol.json" not in hashes:
        raise RuntimeError("recovery bundle lacks its manifest or protocol")
    if not any(name.startswith("output/lens/") for name in hashes):
        raise RuntimeError("recovery bundle lacks the lens")
    if not any(
        name.startswith("output/bench-export/") or name.startswith("output/bench/")
        for name in hashes
    ):
        raise RuntimeError("recovery bundle lacks a Bench store")
    return hashes


def verify_recovered_artifacts(
    metadata: dict[str, Any],
    bundle: bytes,
    artifacts: Path,
    hashes: dict[str, str],
    request: dict[str, Any],
    custody: Path,
) -> dict[str, Any]:
    manifest_path = artifacts / "recovery-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if metadata.get("bundle_bytes") != len(bundle):
        raise RuntimeError("recovery metadata has the wrong bundle size")
    if metadata.get("manifest_sha256") != hashes["recovery-manifest.json"]:
        raise RuntimeError("recovery metadata has the wrong manifest SHA-256")
    if (
        manifest.get("origin_job_id") != ORIGIN_JOB
        or manifest.get("origin_job_record_sha256")
        != request["origin_custody_sha256"]["job-record.json"]
        or manifest.get("expected_source_sha256") != EXPECTED_SOURCE_SHA256
        or manifest.get("recovery_status") != metadata.get("recovery_status")
    ):
        raise RuntimeError("recovery manifest origin or status identity mismatch")
    if manifest.get("config_sha256") != sha256_file(custody / "payload/recovery.json"):
        raise RuntimeError("recovery manifest has the wrong staged config identity")
    if hashes["source/protocol.json"] != EXPECTED_SOURCE_SHA256["protocol.json"]:
        raise RuntimeError("recovered origin protocol has the wrong SHA-256")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list) or metadata.get("inventory_count") != len(inventory):
        raise RuntimeError("recovery inventory count mismatch")
    declared: set[str] = set()
    for row in inventory:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError("recovery inventory row is malformed")
        relative = row["path"]
        if relative in declared or relative not in hashes:
            raise RuntimeError("recovery inventory is duplicate or incomplete")
        path = artifacts / relative
        if row.get("bytes") != path.stat().st_size or row.get("sha256") != hashes[relative]:
            raise RuntimeError("recovery inventory size or SHA-256 mismatch")
        declared.add(relative)
    extracted_payload = {
        name
        for name in hashes
        if name.startswith("output/lens/")
        or name.startswith("output/bench-export/")
        or name.startswith("output/bench/")
    }
    if declared != extracted_payload:
        raise RuntimeError("recovery inventory is not exact for extracted output files")
    presence = manifest.get("output_presence")
    if not isinstance(presence, dict) or set(presence) != {
        "lens",
        "bench-export",
        "bench",
        "report.json",
        "protocol.json",
    }:
        raise RuntimeError("recovery manifest lacks the finalization presence inventory")
    return manifest


def collect(job: str) -> int:
    bindings = list((RESULT_ROOT / "source-custody").glob(f"*/{job}.json"))
    if len(bindings) != 1:
        raise RuntimeError("recovery collection requires one source-custody binding")
    custody = bindings[0].parent
    collected, record, logs = terminal_job(job, wait=False)
    if collected != job:
        raise RuntimeError("recovery collection returned a different job")
    state = str(record.get("state"))
    if state not in TERMINAL:
        print(json.dumps({"job": job, "state": state, "collection": "deferred"}))
        return 0
    output = RESULT_ROOT / job
    write_once(output / "job-record.json", json_bytes(record))
    write_once(output / "job.log", logs.encode())
    guarded = guarded_job_receipt(job)
    request_raw = (custody / "submitted-request.json").read_bytes()
    request = json.loads(request_raw)
    submitted = guarded.get("submitted_request")
    config = submitted.get("config") if isinstance(submitted, dict) else None
    if (
        guarded.get("job_id") != job
        or guarded.get("state") != state
        or not isinstance(config, dict)
        or config.get("saturn_request_sha256") != canonical_sha256(request)
    ):
        raise RuntimeError("guarded recovery receipt is not bound to this request")
    if request["origin_custody_sha256"]["job-record.json"] != sha256_file(
        ORIGIN / "job-record.json"
    ):
        raise RuntimeError("native origin record changed after recovery submission")
    write_once(output / "job-receipt.json", json_bytes(guarded))
    write_once(output / "submitted-request.json", request_raw)
    if state != "succeeded":
        write_once(
            output / "collection-receipt.json",
            json_bytes(
                {
                    "schema": "mdb-jspace-recovery-collection-v1",
                    "recovery_job_id": job,
                    "origin_job_id": ORIGIN_JOB,
                    "state": state,
                    "recovery_status": "transport-failed",
                    "terminal_claim": "none; model-free artifact recovery only",
                }
            ),
        )
        print(json.dumps({"job": job, "state": state, "output": str(output)}, sort_keys=True))
        return 2
    metadata, bundle, digest = decode(logs)
    if metadata.get("origin_job_id") != ORIGIN_JOB:
        raise RuntimeError("recovery metadata has the wrong origin")
    write_once(output / "transport-metadata.json", json_bytes(metadata))
    write_once(output / "recovery-bundle.tar.gz", bundle)
    hashes = safe_extract(bundle, output / "artifacts")
    manifest = verify_recovered_artifacts(
        metadata, bundle, output / "artifacts", hashes, request, custody
    )
    write_once(output / "artifact-manifest.json", json_bytes(hashes))
    receipt = {
        "schema": "mdb-jspace-recovery-collection-v1",
        "recovery_job_id": job,
        "origin_job_id": ORIGIN_JOB,
        "state": state,
        "native_origin_job_record_sha256": sha256_file(ORIGIN / "job-record.json"),
        "bundle_sha256": digest,
        "artifact_count": len(hashes),
        "recovery_status": metadata["recovery_status"],
        "bench_source": manifest["bench_source"],
        "output_presence": manifest["output_presence"],
        "terminal_claim": "none; model-free artifact recovery only",
    }
    write_once(output / "collection-receipt.json", json_bytes(receipt))
    print(json.dumps({"job": job, "state": state, "output": str(output)}, sort_keys=True))
    return 0 if state == "succeeded" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "submit", "collect"))
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.action == "collect":
        if not args.job:
            parser.error("collect requires --job")
        return collect(args.job)
    if args.job:
        parser.error("--job is valid only for collect")
    request, config, command = specification()
    if args.action == "preflight":
        print(json.dumps({"request": request, "config": config, "command": command}, indent=2, sort_keys=True))
        return 0
    submit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
