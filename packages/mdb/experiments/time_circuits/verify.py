"""Offline reference/byte audit for a collected native-research archive."""
import argparse
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

from mdb.bench_publication import recover_locked
from mdb.circuit_runtime import DirectoryArtifactStore
from mdb.model_os_contracts import content_digest
from mdb.vm_payload_cache import PayloadRef, PayloadStore


def verify_source(root, *, request_sha256):
    """Audit sealed worker/package bytes without importing the archived wheels."""
    root = Path(root)
    request = json.loads((root / "saturn_debug_request.json").read_text())
    body = {key: value for key, value in request.items() if key != "request_sha256"}
    actual = sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if actual != request_sha256 or actual != request["request_sha256"]:
        raise ValueError("archived source request identity changed")
    seal = json.loads((root / "payload-seal.json").read_text())
    config_sha = sha256((root / "saturn_run_config.json").read_bytes()).hexdigest()
    if config_sha != seal["config_bytes_sha256"] or config_sha != request["config_bytes_sha256"]:
        raise ValueError("archived configuration changed")
    sources = {}
    for name, expected in seal["wheels"].items():
        if Path(name).name != name:
            raise ValueError("invalid archived wheel path")
        path = root / name
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("archived wheel changed: " + name)
        with ZipFile(path) as wheel:
            for package, members in seal["packages"].items():
                for member, digest in members.items():
                    target = package + "/" + member
                    if target not in wheel.namelist():
                        continue
                    if sha256(wheel.read(target)).hexdigest() != digest:
                        raise ValueError("archived package member changed: " + target)
                    sources[target] = digest
    for name, expected in seal["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("archived source path escapes packet")
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("archived worker file changed: " + name)
        sources[name] = actual
    worker = seal["worker"]
    if sources.get(worker) != seal["worker_sha256"]:
        raise ValueError("archived worker identity changed")
    expected_sources = request["source_hashes"]
    if sources != expected_sources:
        raise ValueError("archived source closure does not match request")
    return {"schema": "mdb-native-research-source-audit-v1", "mechanics_status": "valid",
            "request_sha256": request_sha256, "source_files": len(sources),
            "wheels": seal["wheels"], "config_bytes_sha256": config_sha,
            "authority": "sealed local source bytes; external instrument/model identity is in the environment receipt"}


def verify_session(root):
    root = Path(root)
    writer, recovery = recover_locked(root)
    try:
        artifacts = DirectoryArtifactStore(root / "artifacts")
        pointer = json.loads((root / "selected.json").read_text())
        manifest = artifacts.get(pointer["manifest"])
        if manifest["schema"] != "mdb-native-research-session-v1":
            raise ValueError("unsupported native research archive")
        payloads = PayloadStore(root / "pages", read_only=True)
        references = {}
        def walk(value):
            if isinstance(value, dict) and ("$tensor" in value or "$chunks" in value):
                for descriptor in value.get("$chunks", [value.get("$tensor")]):
                    ref = PayloadRef.from_dict(descriptor)
                    references[ref.fingerprint] = ref
            elif isinstance(value, dict):
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        handles = {row["cut"]["handle"] for row in manifest["cuts"].values()}
        abis = set()
        for row in manifest["cuts"].values():
            cut = row["cut"]
            if content_digest({k: v for k, v in cut.items() if k != "content_sha256"}) != cut["content_sha256"]:
                raise ValueError("research cut identity changed")
            if cut["parent_handle"] is not None and cut["parent_handle"] not in handles:
                raise ValueError("research cut parent missing")
            if content_digest({"state_schema": cut["state_schema_fingerprint"], "state": row["state"]}) != cut["state_fingerprint"]:
                raise ValueError("research tensor input closure changed")
            abis.add(cut["model_abi_fingerprint"])
            walk(row["state"])
        if len(abis) != 1:
            raise ValueError("research cuts have incompatible native ABIs")
        for reference in manifest["receipts"]:
            receipt = artifacts.get(reference)
            if receipt["parent_handle"] not in handles or receipt["child_handle"] not in handles:
                raise ValueError("native receipt has missing cut references")
            if receipt["model_abi_fingerprint"] not in abis:
                raise ValueError("native receipt ABI changed")
        for reference in references.values():
            payloads.audit(reference)
        return {"schema": "mdb-native-research-audit-v1", "mechanics_status": "valid",
            "session_id": manifest["session_id"], "identity": manifest["identity"],
            "manifest": pointer["manifest"], "cuts": len(handles), "receipts": len(manifest["receipts"]),
            "payload_audit": payloads.audit_summary(), "selected_pointer": recovery,
            "authority": "stored input/reference integrity; native outputs require the execution report"}
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="extracted worker result root")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source", type=Path, help="extracted sealed worker/source packet")
    args = parser.parse_args()
    sessions = [verify_session(path.parent) for path in sorted(args.root.glob("*/selected.json"))]
    if not sessions:
        raise ValueError("no native research sessions found")
    graphs = DirectoryArtifactStore(args.root / "graphs") if (args.root / "graphs").exists() else None
    report = json.loads((args.root / "report.json").read_text())
    for reference in report["mdb"].get("graphs", []):
        graphs.get(reference["reference"])
    audit = {"schema": "mdb-native-research-job-audit-v1", "mechanics_status": "valid", "sessions": sessions,
             "graph_references_verified": len(report["mdb"].get("graphs", [])),
             "request_sha256": report["mdb"]["request_sha256"]}
    if args.source:
        audit["source_audit"] = verify_source(args.source, request_sha256=report["mdb"]["request_sha256"])
    if args.out:
        args.out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"mechanics_status": "valid", "sessions": len(sessions),
                      "cuts": sum(s["cuts"] for s in sessions),
                      "verified_bytes": sum(s["payload_audit"]["unique_verified_bytes"] for s in sessions)}))


if __name__ == "__main__":
    main()
