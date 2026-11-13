"""Thott-backed descriptive pointers and demand hydration of native dependencies.

Metadata never substitutes for native K/V or pending execution. A pointer keeps
external CAS backing; native restore still uses the unchanged Bench/VM consumer.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

from .circuit_runtime import DirectoryArtifactStore
from .future_forest import PersistentFutureForest
from .model_os_contracts import content_digest
from .vm_payload_cache import PayloadRef, PayloadStore

POINTER_SCHEMA = "mdb-cot-thott-pointer-v1"
CATALOG_SCHEMA = "mdb-cot-thott-catalog-v1"


def emit(bench, destination=None):
    """Seal derived Thott metadata in one separate journal pack per update."""
    from .bench_publication import BenchPublication, writer_path
    root = Path(destination or bench.root / "cot-pointer").resolve()
    root.mkdir(parents=True, exist_ok=True)
    if bench._edit:
        raise ValueError("pointer publication requires no pending Act")
    publishers = getattr(bench, "_cot_pointer_publishers", None)
    if publishers is None:
        publishers = bench._cot_pointer_publishers = {}
    with writer_path(root).open("a+") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publisher = publishers.get(str(root))
        if publisher is None:
            publisher = publishers[str(root)] = BenchPublication(root)
            publisher.owner_lock = bench._lock
        with publisher.transaction():
            result, catalog = _emit(bench, root)
            ref = DirectoryArtifactStore(root / "artifacts").put(catalog)
            publisher.select(ref)
    return {**result, "catalog_ref": ref}


def _emit(bench, root):
    from .bench_cot import view
    from .bench_cot_index import index
    from .bench_publication import stage_write
    selected = json.loads((bench.root / "selected.json").read_text())["manifest"]
    manifest = bench.artifacts.get(selected)
    # Called after command publication; never publish a speculative register.
    if manifest["branches"][manifest["active"]]["cut"] != bench.branch.cut:
        raise ValueError("pointer requires the selected durable native cut")
    catalog = {"schema": CATALOG_SCHEMA, "manifest_ref": selected,
               "model_identity": bench.identity, "session_id": bench.session_id,
               "selected_cut": bench.branch.cut, "register": view(bench),
               "native_window": [{"cut_ref": ref, "parent_cut": row.get("parent_cut"),
                                  "label": row.get("label"), "token_count": len(row["token_ids"])}
                                 for ref, row in index(bench).lineage(bench, 32)],
               "semantic_cot_qualified": False}
    forest = PersistentFutureForest(root / "thott", forest_id="mdb-cot-" + bench.session_id)
    blob = forest.put_json_object(catalog, media_type="application/vnd.mdb.cot-catalog+json")
    old = None
    if (root / "pointer.json").exists():
        old = json.loads((root / "pointer.json").read_text())
    node = (forest.fork(old["node_id"], transition={"kind": "selected-native-cut"},
                        diff_refs={"cot-view": blob}) if old else
            forest.create_root(state_refs={"cot-view": blob},
                               metadata={"session_id": bench.session_id, "authority": "descriptive"}))
    pointer = {"schema": POINTER_SCHEMA, "forest_id": forest.forest_id,
               "node_id": node, "native_root": os.path.relpath(bench.root, root),
               "catalog_sha256": content_digest(catalog)}
    raw = json.dumps(pointer, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    if not stage_write(root / "pointer.json", raw, immutable=False):
        descriptor, temporary = tempfile.mkstemp(prefix=".pointer-", dir=root)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, root / "pointer.json")
        finally:
            Path(temporary).unlink(missing_ok=True)
    return ({"path": str(root / "pointer.json"), "pointer_bytes": len(raw),
             "catalog_bytes": len(forest.get_blob(blob)), "selected_cut": bench.branch.cut}, catalog)


class BenchPointer:
    def __init__(self, path, *, native_root=None):
        from .bench_publication import OWNERS, recover_locked
        self.path = Path(path).resolve()
        owner = OWNERS.get(str(self.path.parent))
        # A live owner serializes metadata reads. Otherwise recover the derived
        # journal before opening its materialized cache; native payloads stay cold.
        if owner is None and (self.path.parent / "selected.json").exists():
            writer, _ = recover_locked(self.path.parent)
            try:
                self._open(native_root)
            finally:
                writer.close()
        else:
            with owner.owner_lock if owner is not None else nullcontext():
                self._open(native_root)

    def _open(self, native_root):
        pointer = json.loads(self.path.read_text())
        if pointer.get("schema") != POINTER_SCHEMA:
            raise ValueError("unsupported MDB pointer")
        forest = PersistentFutureForest.reopen(self.path.parent / "thott")
        if forest.forest_id != pointer["forest_id"]:
            raise ValueError("pointer forest identity changed")
        forest.verify_node(pointer["node_id"])
        state = forest.effective_state(pointer["node_id"])
        self._catalog = json.loads(forest.get_blob(state["cot-view"]))
        if (self.catalog.get("schema") != CATALOG_SCHEMA
                or content_digest(self.catalog) != pointer["catalog_sha256"]):
            raise ValueError("pointer catalog identity changed")
        self.root = Path(native_root or self.path.parent / pointer["native_root"]).resolve()
        self.artifacts = DirectoryArtifactStore(self.root / "artifacts")
        manifest = self.artifacts.get(self.catalog["manifest_ref"])
        if (manifest["identity"] != self.catalog["model_identity"]
                or manifest["session_id"] != self.catalog["session_id"]
                or manifest["branches"][manifest["active"]]["cut"] != self.catalog["selected_cut"]):
            raise ValueError("pointer native selection binding changed")
        self._manifest = manifest
        self.payloads = PayloadStore(self.root / "vm/radix/cas", read_only=True)

    @property
    def catalog(self):
        return deepcopy(self._catalog)

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    @classmethod
    def open(cls, path, *, native_root=None):
        return cls(path, native_root=native_root)

    def browse(self):
        """Bounded language/graph metadata; no native tensor payload reads."""
        return {"catalog": self.catalog, "hydration_audit": self.payloads.audit_summary()}

    def residual(self, register_ref=None):
        from .bench_cot import verify_artifact
        entries = self.catalog["register"]["entries"]
        if not entries:
            raise ValueError("pointer has no decoded residual")
        ref = register_ref or entries[-1]["register_ref"]
        if ref not in {row["register_ref"] for row in entries}:
            raise ValueError("residual selection is outside this pointer window")
        row = self.artifacts.get(ref)
        verify_artifact(row, self.artifacts, self.manifest)
        read = self.artifacts.get(row["observation_ref"])
        source = PayloadRef.from_dict(read["source_ref"])
        if source.sha256 != row["activation_sha256"]:
            raise ValueError("residual payload differs from bound observation")
        return self.payloads.audit(source)

    def native_closure(self):
        """Plan just the selected native K/V path, delta pages and pending token."""
        from .causal_radix_store import CausalRadixStore
        cut = self.artifacts.get(self.catalog["selected_cut"])
        if cut["identity"] != self.catalog["model_identity"]:
            raise ValueError("pointer selected model identity changed")
        vm = DirectoryArtifactStore(self.root / "vm/artifacts").get(cut["vm_checkpoint"])
        abi = content_digest(self.catalog["model_identity"])
        if vm["model_abi_sha256"] != abi:
            raise ValueError("pointer VM model ABI changed")
        radix = CausalRadixStore(self.root / "vm/radix", read_only=True)
        mount = radix._mount(vm["mount_sha256"])
        if mount["plan"]["model_abi_sha256"] != abi:
            raise ValueError("pointer radix ABI changed")
        path = next(row for row in mount["plan"]["paths"] if row["page_id"] == vm["page_id"])
        physical = {row["node_sha256"]: row["payload_ref"] for row in mount["physical"]}
        refs = [(physical[node], "ancestor-kv") for node in path["node_sha256s"]]
        refs += [(row["payload_ref"], "delta-kv") for row in vm.get("delta_pages", [])]
        if "delta_ref" in vm:
            refs.append((vm["delta_ref"], "delta-kv"))
        if cut.get("logits"):
            refs.append((cut["logits"], "last-logits"))
        execution = cut.get("execution")
        if execution:
            refs.append((execution["hidden"], "pending-residual"))
            refs.extend((ref, "pending-kv") for pair in execution["staged_kv"] for ref in pair)
            if execution.get("logits"):
                refs.append((execution["logits"], "pending-logits"))
        unique = {}
        for ref, use in refs:
            parsed = PayloadRef.from_dict(ref)
            row = unique.setdefault(parsed.sha256, {"payload_ref": parsed.to_dict(), "uses": []})
            if row["payload_ref"] != parsed.to_dict():
                raise ValueError("native closure descriptor identity conflict")
            row["uses"].append(use)
        return {"selected_cut": self.catalog["selected_cut"], "payloads": list(unique.values()),
                "payload_bytes": sum(row["payload_ref"]["bytes"] for row in unique.values()),
                "pretrained_weights": "external identity-bound resident model",
                "hydration_audit": self.payloads.audit_summary()}

    def hydrate_native(self):
        closure = self.native_closure()
        payloads = {row["payload_ref"]["sha256"]: self.payloads.audit(row["payload_ref"]).payload
                    for row in closure["payloads"]}
        return {"closure": closure, "payloads": payloads,
                "hydration_audit": self.payloads.audit_summary()}

    def resume(self, model, tokenizer=None, **options):
        """Native restore from this immutable selection, with other branches cold."""
        from .bench_publication import recover_locked
        from .bench_qwen import QwenBenchSession
        writer, _ = recover_locked(self.root)
        try:
            # Recover proves custody; the pointer chooses a historical admitted
            # manifest and cut rather than silently adopting a newer selection.
            bench = QwenBenchSession.resume(model, tokenizer, root=self.root,
                _writer=writer, _manifest=self.manifest, lazy_branches=True, **options)
            bench.restore(self.catalog["selected_cut"])
            return bench
        except BaseException:
            writer.close()
            raise
