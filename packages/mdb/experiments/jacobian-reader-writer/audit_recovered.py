#!/usr/bin/env python3
"""Independent model-free audit of a recovered real-BF16 J-space export."""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

CANONICAL_WHEEL_SHA256 = "1120c5299ffe096a76db84ceaa71cd44348db8f03a46e6bf17e72e433b96ea2e"
JACOBIAN_SCHEMAS = {
    "mdb-bench-jacobian-pin-v1",
    "mdb-bench-jacobian-read-v1",
    "mdb-bench-jacobian-candidate-v1",
}
SIDECAR_SCHEMAS = JACOBIAN_SCHEMAS | {
    "saturn-hidden-memory-probe-v1",
    "saturn-bench-hidden-memory-write-v1",
}


class _UpstreamBlocker(importlib.abc.MetaPathFinder):
    blocked = {"saturn", "saturn_pub", "common", "common_harness", "mrun"}

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        if fullname.split(".", 1)[0] in self.blocked:
            raise ImportError(f"recovered audit blocked upstream import: {fullname}")
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def progress(phase: str, **details: Any) -> None:
    print(
        "MDB_JSPACE_RECOVERED_AUDIT_PROGRESS "
        + json.dumps({"phase": phase, **details}, sort_keys=True),
        flush=True,
    )


def load_installed_runtime(target: Path):
    target = target.expanduser().resolve()
    if not target.is_dir() or target.is_symlink():
        raise RuntimeError("installed target must be a real directory")
    already = sorted(
        name for name in sys.modules if name.split(".", 1)[0] in _UpstreamBlocker.blocked
    )
    if already:
        raise RuntimeError(f"upstream modules were loaded before audit isolation: {already}")
    sys.path.insert(0, str(target))
    sys.meta_path.insert(0, _UpstreamBlocker())
    import mdb
    from mdb.bench_verify import verify
    from mdb.circuit_runtime import DirectoryArtifactStore

    mdb_file = Path(mdb.__file__).resolve()
    if not mdb_file.is_relative_to(target):
        raise RuntimeError(f"MDB did not import from installed target: {mdb_file}")
    forbidden = sorted(
        name for name in sys.modules if name.split(".", 1)[0] in _UpstreamBlocker.blocked
    )
    if forbidden:
        raise RuntimeError(f"upstream modules imported during audit: {forbidden}")
    try:
        distribution = importlib.metadata.distribution("mdb-runtime")
        version = distribution.version
    except importlib.metadata.PackageNotFoundError:
        version = None
    return verify, DirectoryArtifactStore, {
        "installed_target": str(target),
        "mdb_file": str(mdb_file),
        "import_root_verified": True,
        "blocked_import_roots": sorted(_UpstreamBlocker.blocked),
        "distribution_version": version,
    }


def recovery_layout(root: Path) -> dict[str, Path]:
    root = root.expanduser().resolve()
    artifacts = root / "artifacts"
    required = {
        "collection": root / "collection-receipt.json",
        "artifact_manifest": root / "artifact-manifest.json",
        "recovery_manifest": artifacts / "recovery-manifest.json",
        "protocol": artifacts / "source/protocol.json",
        "lens": artifacts / "output/lens",
    }
    for name, path in required.items():
        if not path.exists() or path.is_symlink():
            raise RuntimeError(f"recovery root lacks safe {name}: {path}")
    manifest = strict_json(required["recovery_manifest"])
    source = manifest.get("bench_source")
    if source not in {"output/bench-export", "output/bench"}:
        raise RuntimeError("recovery manifest names an unsupported Bench source")
    bench = artifacts / source
    if not bench.is_dir() or bench.is_symlink():
        raise RuntimeError("recovered Bench source is not a real directory")
    return {"root": root, "artifacts": artifacts, "bench": bench, **required}


def verify_recovery_inventory(layout: dict[str, Path]) -> dict[str, Any]:
    hashes = strict_json(layout["artifact_manifest"])
    manifest = strict_json(layout["recovery_manifest"])
    if not isinstance(hashes, dict) or not isinstance(manifest, dict):
        raise RuntimeError("recovery manifests must be JSON objects")
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError("artifact manifest contains an invalid row")
        path = layout["artifacts"] / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise RuntimeError(f"recovered artifact identity mismatch: {relative}")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list):
        raise RuntimeError("recovery manifest inventory is missing")
    seen: set[str] = set()
    for row in inventory:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError("recovery inventory contains a malformed row")
        relative = row["path"]
        if relative in seen or relative not in hashes:
            raise RuntimeError("recovery inventory is duplicate or absent from artifact manifest")
        path = layout["artifacts"] / relative
        if row.get("bytes") != path.stat().st_size or row.get("sha256") != hashes[relative]:
            raise RuntimeError(f"recovery inventory differs at {relative}")
        seen.add(relative)
    payload = {
        name
        for name in hashes
        if name.startswith("output/lens/")
        or name.startswith("output/bench-export/")
        or name.startswith("output/bench/")
    }
    if seen != payload:
        raise RuntimeError("recovery inventory does not exactly cover the recovered payload")
    if hashes.get("recovery-manifest.json") != sha256_file(layout["recovery_manifest"]):
        raise RuntimeError("artifact manifest does not bind the recovery manifest")
    if hashes.get("source/protocol.json") != sha256_file(layout["protocol"]):
        raise RuntimeError("artifact manifest does not bind the recovered protocol")
    return {"hashes": hashes, "manifest": manifest}


def artifact_rows(artifacts, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict]]:
    rows: list[dict[str, Any]] = []
    candidates: list[dict] = []
    for path in sorted(artifacts.root.glob("*/*.json")):
        reference = f"cas://sha256/{path.stem}"
        value = artifacts.get(reference)
        schema = value.get("schema") if isinstance(value, dict) else None
        if schema not in SIDECAR_SCHEMAS:
            continue
        row: dict[str, Any] = {"ref": reference, "schema": schema}
        if schema == "mdb-bench-jacobian-candidate-v1":
            consumer = value.get("write", {}).get("consumer", {})
            row.update(
                attempt_id=value.get("attempt_id"),
                consumer=consumer.get("name"),
                dose=value.get("write", {}).get("recipe", {}).get("dose"),
            )
            candidates.append({"ref": reference, "value": value})
        elif schema == "saturn-hidden-memory-probe-v1":
            row.update(
                name=value.get("name"),
                reinstall_closure_exact=value.get("reinstall_effect", {}).get("closure_exact"),
                selected_parent_restored_exact=value.get("selected_parent_restored_exact"),
            )
        rows.append(row)
    return rows, candidates


def optional_candidate_norms(candidates: list[dict], artifacts, bench_root: Path) -> dict[str, Any]:
    try:
        import torch

        from mdb.vm_payload_cache import PayloadRef, PayloadStore
    except ImportError as exc:
        return {"available": False, "reason": f"optional torch/payload runtime unavailable: {exc}"}
    payloads = PayloadStore(bench_root / "vm/radix/cas", read_only=True)

    def tensor(descriptor):
        reference = PayloadRef.from_dict(descriptor)
        raw = payloads.audit(reference).payload
        dtype = getattr(torch, reference.dtype.removeprefix("torch."))
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(reference.shape).clone()

    result: list[dict[str, Any]] = []
    for item in candidates:
        value = item["value"]
        observed = artifacts.get(value["observation_ref"])
        source = tensor(observed["source_ref"])
        candidate = tensor(value["candidate_ref"])
        delta = tensor(value["delta_ref"])
        actual = candidate.float() - source.float()
        result.append(
            {
                "candidate_ref": item["ref"],
                "attempt_id": value.get("attempt_id"),
                "consumer": value.get("write", {}).get("consumer", {}).get("name"),
                "dose": value.get("write", {}).get("recipe", {}).get("dose"),
                "candidate_minus_source_l2": float(actual.norm()),
                "stored_delta_l2": float(delta.float().norm()),
                "stored_delta_exact": bool(torch.equal(actual, delta.float())),
                "declared_actual_delta_norm": value.get("write", {})
                .get("recipe", {})
                .get("algebra", {})
                .get("actual_delta_norm"),
            }
        )
    return {"available": True, "rows": result}


def reopen_commit_status(candidates: list[dict], manifest: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item
        for item in candidates
        if item["value"].get("write", {}).get("consumer", {}).get("name")
        == "export-reopen-native-recall"
    ]
    rows = []
    for item in matches:
        previews = [
            event
            for event in manifest.get("events", [])
            if event.get("kind") == "jacobian-preview"
            and event.get("jacobian_candidate_ref") == item["ref"]
        ]
        terminals = [
            event
            for event in manifest.get("events", [])
            if event.get("kind") in {"residual-act", "act-rejected"}
            and event.get("jacobian_candidate_ref") == item["ref"]
        ]
        rows.append(
            {
                "candidate_ref": item["ref"],
                "attempt_id": item["value"].get("attempt_id"),
                "preview_count": len(previews),
                "terminal_events": [event.get("kind") for event in terminals],
                "committed": len(terminals) == 1 and terminals[0].get("kind") == "residual-act",
            }
        )
    close_cuts = [
        reference
        for reference, row in manifest.get("cuts", {}).items()
        if row.get("label") == "close"
    ]
    return {
        "consumer": "export-reopen-native-recall",
        "candidate_count": len(matches),
        "candidates": rows,
        "close_cut_count": len(close_cuts),
        "close_cuts": close_cuts,
        "interpretation": "existence and native commit/close mechanics only; replay equality is not inferred",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed-target", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError("audit output already exists")
    output.mkdir(parents=True)

    progress("installed-runtime-load")
    verify, artifact_store_type, runtime = load_installed_runtime(args.installed_target)
    progress("recovery-inventory-verify")
    layout = recovery_layout(args.recovery_root)
    recovery = verify_recovery_inventory(layout)
    collection = strict_json(layout["collection"])
    protocol = strict_json(layout["protocol"])
    expected = recovery["manifest"].get("expected_source_sha256", {})
    protocol_sha = sha256_file(layout["protocol"])
    if protocol_sha != expected.get("protocol.json"):
        raise RuntimeError("recovered protocol differs from origin source custody")
    if (
        protocol.get("wheel_sha256") != expected.get("mdb_runtime-0.1.0-py3-none-any.whl")
        or protocol.get("wheel_sha256") != CANONICAL_WHEEL_SHA256
    ):
        raise RuntimeError("recovered protocol/wheel custody join differs")

    audit_copy = output / "audit-copy"
    progress("bench-copy-start", source=str(layout["bench"]), destination=str(audit_copy))
    shutil.copytree(layout["bench"], audit_copy, symlinks=False)
    progress("bench-copy-complete")
    progress("canonical-verify-start")
    audit = verify(audit_copy)
    progress("canonical-verify-complete", counts=audit.get("counts"))
    artifacts = artifact_store_type(audit_copy / "artifacts")
    selected = strict_json(audit_copy / "selected.json")
    manifest = artifacts.get(selected["manifest"])
    sidecars, candidates = artifact_rows(artifacts, manifest)

    lens_manifest = strict_json(layout["lens"] / "manifest.json")
    lens_metadata = lens_manifest.get("metadata", {})
    lens_fingerprint = lens_metadata.get("fingerprint")
    if lens_manifest.get("tensor_file_sha256") != sha256_file(layout["lens"] / "tensors.safetensors"):
        raise RuntimeError("standalone lens tensor file identity differs")
    pins = [row for row in sidecars if row["schema"] == "mdb-bench-jacobian-pin-v1"]
    pin_values = [artifacts.get(row["ref"]) for row in pins]
    if not pin_values or any(
        pin.get("lens_manifest", {}).get("fingerprint") != lens_fingerprint for pin in pin_values
    ):
        raise RuntimeError("standalone transport does not join the pinned Bench lens")

    progress("sidecar-and-norm-audit-start", candidate_count=len(candidates))
    norms = optional_candidate_norms(candidates, artifacts, audit_copy)
    reopen = reopen_commit_status(candidates, manifest)
    receipt = {
        "schema": "mdb-jspace-recovered-independent-audit-v1",
        "origin_job_id": recovery["manifest"].get("origin_job_id"),
        "origin_job_record_sha256": recovery["manifest"].get("origin_job_record_sha256"),
        "recovery_job_id": collection.get("recovery_job_id"),
        "recovery_status": collection.get("recovery_status"),
        "recovery_collection_sha256": sha256_file(layout["collection"]),
        "artifact_manifest_sha256": sha256_file(layout["artifact_manifest"]),
        "recovery_manifest_sha256": sha256_file(layout["recovery_manifest"]),
        "source_identity": {
            "protocol_sha256": protocol_sha,
            "worker_sha256": expected.get("real_bf16_worker.py"),
            "wheel_sha256": expected.get("mdb_runtime-0.1.0-py3-none-any.whl"),
            "model_safetensors_sha256": protocol.get("model_safetensors_sha256"),
        },
        "runtime": runtime,
        "transport": {
            "lens_fingerprint": lens_fingerprint,
            "transport_sha256": lens_metadata.get("transport", {}).get("sha256"),
            "tensor_file_sha256": lens_manifest.get("tensor_file_sha256"),
            "pinned_lens_count": len(pins),
        },
        "closed_export_audit": audit,
        "sidecars": sidecars,
        "sidecar_counts": {
            schema: sum(row["schema"] == schema for row in sidecars)
            for schema in sorted({row["schema"] for row in sidecars})
        },
        "candidate_norms": norms,
        "export_reopen_consumer": reopen,
        "mechanics_status": "recovered-closed-export-byte-audit-valid",
        "instrument_status": "recovered-sidecars-verified",
        "trend_status": "not-assessed-by-model-free-audit",
        "terminal_status": "original-worker-report-criteria-not-assessed",
        "semantic_writer_qualified": False,
        "scope": (
            "Independent installed-wheel, model-free byte/reference/software audit of the recovered "
            "closed Bench export; no reconstruction of the missing worker report or semantic result."
        ),
    }
    write_json(output / "receipt.json", receipt)
    progress("receipt-written", receipt=str(output / "receipt.json"))
    print(json.dumps({"receipt": str(output / "receipt.json"), "audit": audit}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
