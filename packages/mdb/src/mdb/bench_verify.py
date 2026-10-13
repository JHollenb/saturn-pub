"""Dependency-light byte custody and reference closure checks for an exported bench."""
import json
from pathlib import Path

from mdb.ar.program_record import read_program_record
from mdb.causal_radix_store import CausalRadixStore
from mdb.circuit_runtime import DirectoryArtifactStore
from mdb.future_forest import PersistentFutureForest
from mdb.hyperion import HYPERION_REPORT_SCHEMA, verify_hyperion_report
from mdb.vm_cache import VMCacheStore
from mdb.vm_payload_cache import PayloadRef


def verify(root):
    root = Path(root)
    from mdb.bench_publication import recover_locked, OWNERS
    from mdb.errors import BenchError
    try: writer, _ = recover_locked(root)
    except BenchError:
        owner = OWNERS.get(str(root.resolve()))
        if owner is None or owner.owner_lock is None: raise
        # The resident owner already has the materialized closure. Serialize
        # verification with that owner; recovery never races an active writer.
        with owner.owner_lock:
            try: return _verify_tree(root)
            except Exception as exc:
                _persist_failed_audit(root, exc)
                raise
    try: return _verify_tree(root)
    except Exception as exc:
        _persist_failed_audit(root, exc)
        raise
    finally: writer.close()


def _persist_failed_audit(root, exc):
    """One durable invalid aggregate, separate from immutable payload objects."""
    from hashlib import sha256
    from .vm_payload_cache import PayloadStore
    summary = getattr(exc, "audit_summary", None)
    if summary is None: return
    summary = {**summary, "manifest_ref": json.loads((root/"selected.json").read_text())["manifest"]}
    raw = json.dumps(summary, sort_keys=True).encode()
    target = root/"byte-audits"/(sha256(raw).hexdigest()+".json")
    target.parent.mkdir(exist_ok=True)
    PayloadStore._atomic_publish(target, raw+b"\n")


def _verify_tree(root):
    result = _verify(root)
    manifest = DirectoryArtifactStore(root/"artifacts").get(result["manifest_ref"])
    contexts = []
    for child in sorted((root/"discovery-contexts").glob("*/selected.json")):
        if child.is_symlink() or child.parent.is_symlink(): raise ValueError("linked discovery context")
        value = _verify(child.parent)
        child_manifest = DirectoryArtifactStore(child.parent/"artifacts").get(value["manifest_ref"])
        if child_manifest["identity"] != manifest["identity"]: raise ValueError("discovery context model identity changed")
        contexts.append({"root_relative": child.parent.relative_to(root).as_posix(), **value})
    result["discovery_contexts"] = contexts
    return result


def _verify(root):
    artifacts = DirectoryArtifactStore(root/"artifacts")
    vm_artifacts = DirectoryArtifactStore(root/"vm/artifacts")
    manifest_ref = json.loads((root/"selected.json").read_text())["manifest"]
    manifest = artifacts.get(manifest_ref)
    if manifest.get("schema") in {"saturn-open-bench-v2", "saturn-open-bench-v3"}:
        return verify_family(root, artifacts, manifest_ref, manifest)
    radix = CausalRadixStore(root/"vm/radix", read_only=True)
    counts = {"json_artifacts": 0, "payload_references": 0, "cuts": 0, "symbols": 0, "vm_bundles": 0}
    seen_payloads = set()

    def walk(value):
        if isinstance(value, dict):
            if value.get("schema") == PayloadRef.schema:
                ref = PayloadRef.from_dict(value)
                if ref.fingerprint not in seen_payloads:
                    radix.payloads.audit(ref)
                    seen_payloads.add(ref.fingerprint)
                    counts["payload_references"] += 1
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for store in (artifacts, vm_artifacts):
        for path in sorted(store.root.glob("*/*.json")):
            value = store.get(f"cas://sha256/{path.stem}")
            counts["json_artifacts"] += 1
            walk(value)
            if store is artifacts:
                from .bench_jacobian_verify import verify_artifact as verify_jacobian
                if verify_jacobian(value, artifacts, manifest):
                    counts["jacobian_artifacts"] = counts.get("jacobian_artifacts", 0)+1
                from .bench_circuit_verify import verify_artifact
                if verify_artifact(value, artifacts, manifest, vm_artifacts):
                    counts["circuit_artifacts"] = counts.get("circuit_artifacts", 0)+1
                from .bench_memory_probe import verify_artifact as verify_memory_probe
                if verify_memory_probe(value, artifacts, manifest, vm_artifacts):
                    counts["memory_probes"] = counts.get("memory_probes", 0)+1
                from .bench_hidden_memory import WRITE_SCHEMA, verify_write_act
                if isinstance(value, dict) and value.get("schema") == WRITE_SCHEMA and value.get("cut_id"):
                    verify_write_act(value, artifacts, vm_artifacts)
                    counts["memory_writes"] = counts.get("memory_writes", 0)+1
            if isinstance(value, dict) and value.get("schema") == HYPERION_REPORT_SCHEMA:
                verify_hyperion_report(value)
                for row in value["observations"]:
                    artifacts.get(row["evidence_ref"])
                counts["symbols"] += 1
    for ref in manifest["cuts"]:
        snapshot = artifacts.get(ref)
        if snapshot["identity"] != manifest["identity"] or snapshot["lane"] != manifest["lane"]:
            raise ValueError("cut model/lane identity differs from manifest")
        checkpoint = vm_artifacts.get(snapshot["vm_checkpoint"])
        radix.read(checkpoint["mount_sha256"], checkpoint["page_id"],
                   model_abi_sha256=checkpoint["model_abi_sha256"], audit=True)
        if snapshot["parent_cut"]:
            artifacts.get(snapshot["parent_cut"])
        counts["cuts"] += 1
    for branch in manifest["branches"].values():
        artifacts.get(branch["cut"])
    vm = VMCacheStore(root/"vm/vm-cache")
    for path in sorted(vm.bundles.glob("*.json")):
        bundle = vm.get_bundle(path)
        for ref in bundle.plane_references.values():
            vm.get(ref)
        native = bundle.metadata["native_execution"]
        vm_artifacts.get(native["receipt_ref"])
        vm_artifacts.get(native["checkpoint_ref"])
        counts["vm_bundles"] += 1
    forest = PersistentFutureForest(root/"forest", forest_id=manifest["session_id"])
    forest.verify_closure()
    from .source_frame_stack import SourceFrameStack
    sources = SourceFrameStack(root/"source-frames")
    with sources.register.forest.verification_snapshot():
        sources.register.forest.verify_closure()
        for ref in manifest["cuts"]:
            frame = artifacts.get(ref).get("source_frames")
            if frame:
                sources.inspect(frame["active"])
                sources.cuts.verify_or_raise(frame["statecut_ref"])
    read_program_record(root/"programs.json")
    return {"schema": "saturn-open-bench-verification-v1", "mechanics_status": "valid",
            "session_id": manifest["session_id"], "manifest_ref": manifest_ref, "counts": counts,
            "payload_audit": {"manifest_ref": manifest_ref, **radix.payloads.audit_summary()},
            "boundary": "byte-integrity and reference closure; native execution is proved separately"}


def verify_family(root, artifacts, manifest_ref, manifest):
    from mdb.vm_payload_cache import PayloadStore
    payloads = PayloadStore(root/"payloads", read_only=True)
    seen = set()
    counts = {"json_artifacts": 0, "payload_references": 0, "cuts": 0, "symbols": 0, "vm_bundles": 0}
    def walk(value):
        if isinstance(value, dict):
            if value.get("schema") == PayloadRef.schema:
                ref = PayloadRef.from_dict(value)
                if ref.fingerprint not in seen:
                    payloads.audit(ref)
                    seen.add(ref.fingerprint); counts["payload_references"] += 1
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    for path in artifacts.root.glob("*/*.json"):
        value = artifacts.get(f"cas://sha256/{path.stem}"); counts["json_artifacts"] += 1; walk(value)
        from .bench_circuit_verify import verify_artifact
        if verify_artifact(value, artifacts, manifest):
            counts["circuit_artifacts"] = counts.get("circuit_artifacts", 0)+1
        if isinstance(value, dict) and value.get("schema") == HYPERION_REPORT_SCHEMA:
            verify_hyperion_report(value)
            for row in value["observations"]: artifacts.get(row["evidence_ref"])
            counts["symbols"] += 1
    from .source_frame_stack import SourceFrameStack
    sources = SourceFrameStack(root/"source-frames")
    with sources.register.forest.verification_snapshot():
        sources.register.forest.verify_closure()
        for ref, record in manifest["cuts"].items():
            cut = artifacts.get(ref)
            if any(cut[key] != manifest[key] for key in ("identity", "family", "lane", "numerical_contract")):
                raise ValueError("family cut identity differs from manifest")
            if cut["mutable_closure"] != manifest["capabilities"]["mutable_closure"]:
                raise ValueError("cut mutable closure differs from capabilities")
            if cut.get("parent_cut"): artifacts.get(cut["parent_cut"])
            if cut.get("receipt_ref"): artifacts.get(cut["receipt_ref"])
            if record.get("source_frame"): sources.inspect(record["source_frame"])
            counts["cuts"] += 1
    for branch in manifest["branches"].values(): artifacts.get(branch["cut"])
    vm = VMCacheStore(root/"vm-cache")
    for path in vm.bundles.glob("*.json"):
        bundle = vm.get_bundle(path)
        for ref in bundle.plane_references.values(): vm.get(ref)
        artifacts.get(bundle.metadata["cut"])
        if bundle.metadata.get("receipt_ref"): artifacts.get(bundle.metadata["receipt_ref"])
        counts["vm_bundles"] += 1
    PersistentFutureForest(root/"forest", forest_id=manifest["session_id"]).verify_closure()
    read_program_record(root/"programs.json")
    return {"schema": "saturn-open-bench-verification-v2", "mechanics_status": "valid",
            "family": manifest["family"], "session_id": manifest["session_id"], "manifest_ref": manifest_ref,
            "counts": counts, "payload_audit": {"manifest_ref": manifest_ref, **payloads.audit_summary()},
            "boundary": "byte/reference closure; model-backed native continuation is proved separately"}
