"""Descriptive, sealed annotations for immutable Bench StateCuts.

Annotations are a sidecar evidence index.  They never participate in StateCut,
manifest, CAS, numerical, replay, or compiler identities.  A label describes a
cut; it does not establish a semantic circuit or authorize restoration.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .bench_publication import stage_write
from .circuit_runtime import DirectoryArtifactStore
from .model_os_contracts import canonical_json_bytes, content_digest

ANNOTATION_SCHEMA = "mdb-state-annotation-v1"
INDEX_SCHEMA = "mdb-state-annotation-index-v1"
VERIFICATION_SCHEMA = "mdb-state-annotation-verification-v1"
PROVENANCE_SCHEMA = "mdb-state-annotation-provenance-v1"
SIDECAR = "state-annotations"
MAX_TEXT = 512
MAX_REFS = 64
MAX_DETAILS_BYTES = 16 << 10
MAX_DETAILS_DEPTH = 5
GEOMETRY_FIELDS = {"components", "layers", "rows", "row_ranges", "addresses"}


class StateAnnotationError(ValueError):
    """The annotation sidecar is invalid or the requested lookup is ambiguous."""


def _text(value: Any, field: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT:
        raise StateAnnotationError(f"{field} must be a nonempty string of at most {MAX_TEXT} characters")
    if any(ord(char) < 32 for char in value):
        raise StateAnnotationError(f"{field} contains control characters")
    return value.strip()


def _refs(values: Sequence[str] | None, field: str) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) > MAX_REFS:
        raise StateAnnotationError(f"{field} must be a sequence of at most {MAX_REFS} strings")
    result = [_text(value, field) for value in values]
    if len(result) != len(set(result)):
        raise StateAnnotationError(f"{field} contains duplicates")
    return result


def _details(value: Any, *, depth: int = 0) -> Any:
    if depth > MAX_DETAILS_DEPTH:
        raise StateAnnotationError("details nesting is too deep")
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not (-float("inf") < value < float("inf")):
            raise StateAnnotationError("details contains a non-finite number")
        if isinstance(value, str) and len(value) > MAX_TEXT:
            raise StateAnnotationError(f"details strings must be at most {MAX_TEXT} characters")
        return value
    if isinstance(value, list):
        if len(value) > 256:
            raise StateAnnotationError("details list is too large")
        return [_details(item, depth=depth + 1) for item in value]
    if isinstance(value, Mapping):
        if len(value) > 128:
            raise StateAnnotationError("details object is too large")
        result = {}
        for key, item in value.items():
            key = _text(key, "details key")
            result[key] = _details(item, depth=depth + 1)
        return result
    raise StateAnnotationError(f"details has unsupported type {type(value).__name__}")


def _indices(values: Any, field: str) -> list[int]:
    if not isinstance(values, list) or len(values) > 256:
        raise StateAnnotationError(f"geometry {field} must be a list of at most 256 indices")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise StateAnnotationError(f"geometry {field} contains an invalid index")
    if len(values) != len(set(values)):
        raise StateAnnotationError(f"geometry {field} contains duplicate indices")
    return list(values)


def _range(value: Any, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {"start", "stop"}:
        raise StateAnnotationError(f"geometry {field} requires start and stop")
    start, stop = value["start"], value["stop"]
    if (isinstance(start, bool) or isinstance(stop, bool)
            or not isinstance(start, int) or not isinstance(stop, int)
            or start < 0 or stop <= start):
        raise StateAnnotationError(f"geometry {field} requires 0 <= start < stop")
    return {"start": start, "stop": stop}


def _geometry(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not set(value).issubset(GEOMETRY_FIELDS):
        raise StateAnnotationError(f"geometry fields must be drawn from {sorted(GEOMETRY_FIELDS)}")
    result: dict[str, Any] = {}
    if "components" in value:
        result["components"] = sorted(_refs(value["components"], "geometry components"))
    if "layers" in value:
        result["layers"] = _indices(value["layers"], "layers")
    if "rows" in value:
        result["rows"] = (_range(value["rows"], "rows") if isinstance(value["rows"], Mapping)
                          else _indices(value["rows"], "rows"))
    if "row_ranges" in value:
        ranges = value["row_ranges"]
        if not isinstance(ranges, list) or len(ranges) > 64:
            raise StateAnnotationError("geometry row_ranges must be a list of at most 64 ranges")
        result["row_ranges"] = [_range(item, "row_ranges") for item in ranges]
    if "addresses" in value:
        result["addresses"] = sorted(_refs(value["addresses"], "geometry addresses"))
    if len(canonical_json_bytes(result)) > MAX_DETAILS_BYTES:
        raise StateAnnotationError(f"geometry exceeds {MAX_DETAILS_BYTES} canonical bytes")
    return result


def _safe_path(root: Path, path: Path) -> Path:
    root = root.resolve()
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(root):
        raise StateAnnotationError("annotation path is outside the session root")
    resolved = path.resolve(strict=False)
    if resolved != path or not resolved.is_relative_to(root):
        raise StateAnnotationError("annotation path is outside the session root")
    for candidate in (path, *path.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise StateAnnotationError("annotation path traverses a symlink")
    return path


def _replace(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".state-annotation-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


class StateAnnotationStore:
    """A session-bound annotation index over immutable cut artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        session_binding: Mapping[str, Any],
        cut_loader: Callable[[str], Mapping[str, Any]],
        cut_registry: Mapping[str, Mapping[str, Any]],
        journal_required: bool = False,
        selected_manifest_ref: str | None = None,
        write_enabled: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.sidecar = _safe_path(self.root, self.root / SIDECAR)
        self.index_path = _safe_path(self.root, self.sidecar / "index.json")
        self.objects = _safe_path(self.root, self.sidecar / "objects")
        self.session_binding = dict(session_binding)
        self.cut_loader = cut_loader
        self.cut_registry = cut_registry
        self.journal_required = bool(journal_required)
        self.selected_manifest_ref = selected_manifest_ref
        self.write_enabled = bool(write_enabled)
        if self.sidecar.is_symlink() or self.objects.is_symlink():
            raise StateAnnotationError("annotation namespace cannot be a symlink")

    @classmethod
    def from_session(cls, session: Any) -> "StateAnnotationStore":
        selected = json.loads((session.root / "selected.json").read_bytes())
        binding = {
            "session_id": session.session_id,
            "family": getattr(session, "family", "qwen2"),
            "lane": session.lane,
            "model_identity_sha256": content_digest(session.identity),
            "publication_mode": session.publication.mode,
        }
        return cls(
            session.root,
            session_binding=binding,
            cut_loader=session.artifacts.get,
            cut_registry=session.cuts,
            journal_required=session.publication.mode == "journal",
            selected_manifest_ref=selected["manifest"],
        )

    @classmethod
    def open(cls, root: str | Path) -> "StateAnnotationStore":
        root = Path(root).expanduser().resolve()
        selected = json.loads((root / "selected.json").read_bytes())
        artifacts = DirectoryArtifactStore(root / "artifacts")
        manifest = artifacts.get(selected["manifest"])
        identity = manifest["identity"]
        binding = {
            "session_id": manifest["session_id"],
            "family": manifest.get("family", "qwen2"),
            "lane": manifest["lane"],
            "model_identity_sha256": content_digest(identity),
            "publication_mode": manifest.get("publication_mode", "legacy"),
        }
        return cls(root, session_binding=binding, cut_loader=artifacts.get,
                   cut_registry=manifest.get("cuts", {}),
                   journal_required=manifest.get("publication_mode", "legacy") == "journal",
                   selected_manifest_ref=selected["manifest"], write_enabled=False)

    def _empty_index(self) -> dict[str, Any]:
        body = {"schema": INDEX_SCHEMA, "session_binding": self.session_binding, "entries": []}
        return {**body, "content_sha256": content_digest(body)}

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return self._empty_index()
        if self.index_path.is_symlink():
            raise StateAnnotationError("annotation index cannot be a symlink")
        try:
            raw = self.index_path.read_bytes()
            index = json.loads(raw)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise StateAnnotationError("annotation index is unreadable") from exc
        if not isinstance(index, dict) or set(index) != {
                "schema", "session_binding", "entries", "content_sha256"}:
            raise StateAnnotationError("annotation index shape is invalid")
        body = {key: value for key, value in index.items() if key != "content_sha256"}
        if index.get("schema") != INDEX_SCHEMA or index.get("content_sha256") != content_digest(body):
            raise StateAnnotationError("annotation index seal changed")
        if index.get("session_binding") != self.session_binding:
            raise StateAnnotationError("annotation index session binding changed")
        if (not isinstance(index.get("entries"), list)
                or any(not isinstance(entry, Mapping) for entry in index["entries"])):
            raise StateAnnotationError("annotation index entries are invalid")
        return index

    def _write(self, path: Path, raw: bytes, *, immutable: bool) -> None:
        _safe_path(self.root, path)
        if path.exists() and immutable:
            if path.is_symlink() or path.read_bytes() != raw:
                raise StateAnnotationError("immutable annotation object collision")
            return
        staged = stage_write(path, raw, immutable=immutable)
        if self.journal_required and not staged:
            raise StateAnnotationError("journal annotations require an active publication transaction")
        if not staged:
            _replace(path, raw)

    def _cut(self, cut_id: str) -> dict[str, Any]:
        cut_id = _text(cut_id, "cut_id")
        if cut_id not in self.cut_registry:
            raise StateAnnotationError(f"cut is not registered in this session: {cut_id}")
        try:
            cut = dict(self.cut_loader(cut_id))
        except Exception as exc:
            raise StateAnnotationError(f"cut cannot be loaded: {cut_id}") from exc
        if cut.get("identity") is None or content_digest(cut["identity"]) != self.session_binding["model_identity_sha256"]:
            raise StateAnnotationError("cut model identity differs from the session binding")
        return cut

    def _clock(self, cut_id: str, cut: Mapping[str, Any]) -> dict[str, Any]:
        cursor = cut.get("cursor") or self.cut_registry[cut_id].get("cursor")
        if not isinstance(cursor, Mapping):
            raise StateAnnotationError("cut has no native cursor")
        cursor = _details(dict(cursor))
        phase = _text(cursor.get("phase", "unknown"), "native clock phase")
        next_layer = cursor.get("next_layer")
        boundary = "layer" if next_layer is not None else phase
        index = next_layer
        if index is None:
            index = cursor.get("token", cursor.get("step", cursor.get("index", 0)))
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise StateAnnotationError("native clock index is invalid")
        return {"boundary": boundary, "index": index, "cursor": cursor}

    def _input_binding(self, cut: Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(cut.get("token_ids"), list):
            tokens = cut["token_ids"]
            if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
                raise StateAnnotationError("cut token history is invalid")
            return {
                "kind": "committed-token-history",
                "token_count": len(tokens),
                "token_ids_sha256": content_digest(tokens),
                "prompt_length": cut.get("prompt_length"),
                "pending_token": cut.get("pending"),
            }
        if "state" in cut:
            return {"kind": "family-native-state", "state_sha256": content_digest(cut["state"])}
        return {"kind": "cut-bound", "cut_payload_sha256": content_digest(cut)}

    def annotate(
        self,
        label: str,
        *,
        cut_id: str,
        namespace: str = "default",
        role: str,
        stage: str,
        scientific_source_cut_ids: Sequence[str] = (),
        recipient_parent_cut_id: str | None = None,
        tags: Sequence[str] = (),
        native_act_refs: Sequence[str] = (),
        consumer_refs: Sequence[str] = (),
        geometry: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
        note: str | None = None,
        producer: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.write_enabled:
            raise StateAnnotationError("offline annotation stores are read-only; use the resident BenchSession API")
        label = _text(label, "label")
        namespace = _text(namespace, "namespace")
        role = _text(role, "role")
        stage = _text(stage, "stage")
        note = _text(note, "note", required=False)
        tags = sorted(_refs(tags, "tags"))
        source_ids = sorted(_refs(scientific_source_cut_ids, "scientific_source_cut_ids"))
        act_refs = sorted(_refs(native_act_refs, "native_act_refs"))
        consumer_refs = sorted(_refs(consumer_refs, "consumer_refs"))
        geometry = _geometry(geometry)
        cut = self._cut(cut_id)
        for source_id in source_ids:
            self._cut(source_id)
        if recipient_parent_cut_id is not None:
            self._cut(recipient_parent_cut_id)
        structured = _details(dict(details or {}))
        if len(canonical_json_bytes(structured)) > MAX_DETAILS_BYTES:
            raise StateAnnotationError(f"details exceeds {MAX_DETAILS_BYTES} canonical bytes")
        producer = _details(dict(producer or {
            "kind": "bench-session", "id": self.session_binding["session_id"], "tool": "mdb"
        }))
        payload = {
            "schema": ANNOTATION_SCHEMA,
            "session_binding": self.session_binding,
            "cut_id": cut_id,
            "immutable_parent_cut_id": cut.get("parent_cut"),
            "label": label,
            "namespace": namespace,
            "role": role,
            "stage": stage,
            "native_writer_clock": self._clock(cut_id, cut),
            "model_input_binding": self._input_binding(cut),
            "scientific_source_cut_ids": source_ids,
            "recipient_parent_cut_id": recipient_parent_cut_id,
            "native_act_refs": act_refs,
            "consumer_refs": consumer_refs,
            "tags": tags,
            "geometry": geometry,
            "details": structured,
            "note": note,
            "provenance": {"schema": PROVENANCE_SCHEMA, "producer": producer,
                           "selected_manifest_ref": self.selected_manifest_ref,
                           "authority": "descriptive-only", "semantic_claim": False},
        }
        body_sha256 = content_digest(payload)
        annotation_id = f"state-annotation://sha256/{body_sha256}"
        qualified_label = f"{namespace}::{label}@sha256:{body_sha256}"
        annotation = {**payload, "annotation_id": annotation_id,
                      "qualified_label": qualified_label, "body_sha256": body_sha256}
        raw = canonical_json_bytes(annotation)
        object_sha256 = sha256(raw).hexdigest()
        object_path = self.objects / f"{object_sha256}.json"
        index = self._load_index()
        existing = next((row for row in index["entries"] if row.get("annotation_id") == annotation_id), None)
        entry = {
            "annotation_id": annotation_id,
            "qualified_label": qualified_label,
            "label": label,
            "namespace": namespace,
            "cut_id": cut_id,
            "role": role,
            "stage": stage,
            "tags": tags,
            "body_sha256": body_sha256,
            "object_sha256": object_sha256,
            "object_path": object_path.relative_to(self.root).as_posix(),
        }
        if existing is not None:
            if existing != entry:
                raise StateAnnotationError("annotation identity collision")
            return annotation
        if any(row.get("qualified_label") == qualified_label for row in index["entries"]):
            raise StateAnnotationError("qualified annotation label collision")
        self._write(object_path, raw, immutable=True)
        entries = sorted([*index["entries"], entry], key=lambda row: row["qualified_label"])
        body = {"schema": INDEX_SCHEMA, "session_binding": self.session_binding, "entries": entries}
        sealed = {**body, "content_sha256": content_digest(body)}
        self._write(self.index_path, canonical_json_bytes(sealed), immutable=False)
        return annotation

    def _object(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(entry, Mapping):
            raise StateAnnotationError("annotation index entry is invalid")
        object_path = _text(entry.get("object_path"), "annotation object path")
        path = _safe_path(self.root, self.root / object_path)
        if path.is_symlink():
            raise StateAnnotationError("annotation object cannot be a symlink")
        try:
            raw = path.read_bytes()
            annotation = json.loads(raw)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise StateAnnotationError("annotation object is unreadable") from exc
        if sha256(raw).hexdigest() != entry.get("object_sha256") or canonical_json_bytes(annotation) != raw:
            raise StateAnnotationError("annotation object bytes changed")
        payload = {key: value for key, value in annotation.items()
                   if key not in {"annotation_id", "qualified_label", "body_sha256"}}
        body_sha256 = content_digest(payload)
        expected_id = f"state-annotation://sha256/{body_sha256}"
        expected_label = f"{payload['namespace']}::{payload['label']}@sha256:{body_sha256}"
        if (annotation.get("schema") != ANNOTATION_SCHEMA
                or annotation.get("body_sha256") != body_sha256
                or annotation.get("annotation_id") != expected_id
                or annotation.get("qualified_label") != expected_label
                or entry.get("annotation_id") != expected_id
                or entry.get("qualified_label") != expected_label
                or entry.get("body_sha256") != body_sha256):
            raise StateAnnotationError("annotation object seal changed")
        expected_entry = {
            "annotation_id": annotation["annotation_id"],
            "qualified_label": annotation["qualified_label"],
            "label": annotation["label"],
            "namespace": annotation["namespace"],
            "cut_id": annotation["cut_id"],
            "role": annotation["role"],
            "stage": annotation["stage"],
            "tags": annotation["tags"],
            "body_sha256": annotation["body_sha256"],
            "object_sha256": sha256(raw).hexdigest(),
            "object_path": path.relative_to(self.root).as_posix(),
        }
        if dict(entry) != expected_entry:
            raise StateAnnotationError("annotation index summary differs from its sealed object")
        if annotation.get("session_binding") != self.session_binding:
            raise StateAnnotationError("annotation object session binding changed")
        return annotation

    def list(
        self,
        *,
        label: str | None = None,
        namespace: str | None = None,
        role: str | None = None,
        stage: str | None = None,
        cut_id: str | None = None,
        tag: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = list(self._load_index()["entries"])
        predicates = {"label": label, "namespace": namespace, "role": role,
                      "stage": stage, "cut_id": cut_id}
        for field, value in predicates.items():
            if value is not None:
                rows = [row for row in rows if row.get(field) == value]
        if tag is not None:
            rows = [row for row in rows if tag in row.get("tags", [])]
        if query is not None:
            needle = query.casefold()
            rows = [row for row in rows if needle in " ".join(str(row.get(field, ""))
                    for field in ("qualified_label", "label", "namespace", "role", "stage", "cut_id")).casefold()]
        for row in rows:
            self._object(row)
        return rows

    def show(self, reference: str) -> dict[str, Any]:
        reference = _text(reference, "annotation reference")
        rows = self._load_index()["entries"]
        exact = [row for row in rows if reference in {row.get("annotation_id"), row.get("qualified_label")}]
        if len(exact) == 1:
            return self._object(exact[0])
        plain = [row for row in rows if row.get("label") == reference]
        if len(plain) == 1:
            return self._object(plain[0])
        if len(plain) > 1:
            choices = [row["qualified_label"] for row in plain]
            raise StateAnnotationError(f"ambiguous label {reference!r}; use one of {choices}")
        raise StateAnnotationError(f"unknown annotation reference: {reference}")

    def lineage(self, cut_id: str) -> dict[str, Any]:
        cut_id = _text(cut_id, "cut_id")
        chain = []
        seen = set()
        current = cut_id
        while current is not None:
            if current in seen:
                raise StateAnnotationError("cut lineage contains a cycle")
            seen.add(current)
            cut = self._cut(current)
            chain.append({
                "cut_id": current,
                "immutable_parent_cut_id": cut.get("parent_cut"),
                "native_writer_clock": self._clock(current, cut),
                "annotations": self.list(cut_id=current),
            })
            current = cut.get("parent_cut")
        chain.reverse()
        return {"schema": "mdb-state-annotation-lineage-v1", "session_binding": self.session_binding,
                "target_cut_id": cut_id, "cuts": chain}

    def verify(self) -> dict[str, Any]:
        index = self._load_index()
        annotation_ids, qualified_labels = set(), set()
        for entry in index["entries"]:
            annotation = self._object(entry)
            if annotation["annotation_id"] in annotation_ids or annotation["qualified_label"] in qualified_labels:
                raise StateAnnotationError("annotation index contains duplicate identities")
            annotation_ids.add(annotation["annotation_id"])
            qualified_labels.add(annotation["qualified_label"])
            cut = self._cut(annotation["cut_id"])
            if annotation["immutable_parent_cut_id"] != cut.get("parent_cut"):
                raise StateAnnotationError("annotation immutable parent changed")
            if annotation["native_writer_clock"] != self._clock(annotation["cut_id"], cut):
                raise StateAnnotationError("annotation native writer clock changed")
            if annotation["model_input_binding"] != self._input_binding(cut):
                raise StateAnnotationError("annotation model-input binding changed")
            for source in annotation["scientific_source_cut_ids"]:
                self._cut(source)
            if annotation["recipient_parent_cut_id"] is not None:
                self._cut(annotation["recipient_parent_cut_id"])
        indexed = {entry["object_sha256"] for entry in index["entries"]}
        objects = set()
        if self.objects.exists():
            for path in self.objects.iterdir():
                if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                    raise StateAnnotationError("annotation object namespace contains an invalid member")
                if len(path.stem) != 64 or any(char not in "0123456789abcdef" for char in path.stem):
                    raise StateAnnotationError("annotation object filename is invalid")
                objects.add(path.stem)
                if path.stem not in indexed:
                    try:
                        raw = path.read_bytes()
                        annotation = json.loads(raw)
                    except (OSError, ValueError, UnicodeDecodeError) as exc:
                        raise StateAnnotationError("orphan annotation object is unreadable") from exc
                    if canonical_json_bytes(annotation) != raw or sha256(raw).hexdigest() != path.stem:
                        raise StateAnnotationError("orphan annotation object bytes changed")
                    payload = {key: value for key, value in annotation.items()
                               if key not in {"annotation_id", "qualified_label", "body_sha256"}}
                    body_sha256 = content_digest(payload)
                    if (annotation.get("schema") != ANNOTATION_SCHEMA
                            or annotation.get("session_binding") != self.session_binding
                            or annotation.get("body_sha256") != body_sha256
                            or annotation.get("annotation_id") != f"state-annotation://sha256/{body_sha256}"
                            or annotation.get("qualified_label") !=
                            f"{annotation.get('namespace')}::{annotation.get('label')}@sha256:{body_sha256}"):
                        raise StateAnnotationError("orphan annotation object seal changed")
        return {
            "schema": VERIFICATION_SCHEMA,
            "mechanics_status": "valid",
            "session_binding": self.session_binding,
            "annotation_count": len(index["entries"]),
            "orphan_object_count": len(objects - indexed),
            "index_content_sha256": index["content_sha256"],
            "semantic_claims_established": False,
            "external_references_verified": False,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mdb states", description="Inspect descriptive StateCut annotations")
    parser.add_argument("--root", required=True, help="Bench session root")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    for name in ("label", "namespace", "role", "stage", "cut-id", "tag", "query"):
        listing.add_argument(f"--{name}")
    show = commands.add_parser("show")
    show.add_argument("reference")
    lineage = commands.add_parser("lineage")
    lineage.add_argument("cut_id")
    commands.add_parser("verify")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    store = StateAnnotationStore.open(args.root)
    if args.command == "list":
        result = store.list(label=args.label, namespace=args.namespace, role=args.role,
                            stage=args.stage, cut_id=args.cut_id, tag=args.tag, query=args.query)
    elif args.command == "show":
        result = store.show(args.reference)
    elif args.command == "lineage":
        result = store.lineage(args.cut_id)
    else:
        result = store.verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
