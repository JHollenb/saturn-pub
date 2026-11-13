"""Read-only reader of a recorded mdb project (``project.sqlite`` + saturn_pub ``store/``).

Standard library, ``saturn_pub`` and ``mdb.core`` only: no model, no framework, no write. This is
what ``mdb serve`` reads a project through. It opens SQLite strictly read-only and verifies every
row it returns against its seal and its session binding, the same checks the project writer's own
readers make, so a served value is never an unverified row.

Rows are verified one at a time. Every listing method takes ``refuse``: a callable
``refuse(kind, identifier, error)``. Without it the first bad row raises (strict, like the writer's
readers); with it a bad row is reported and skipped and the healthy rows are still returned. Cuts
and receipts are always strict: they are the session's execution history, and a session missing
part of it is refused as a whole by callers.

What is verified, per row:

* sessions: the row exists (``ValueError("unknown project session: ...")`` otherwise).
* cuts: the index manifest equals the store's sealed manifest (``LocalStore.inspect``), the
  manifest is bound to the session's model identity and execution, and the stored execution point
  and surface describe to the sealed ones.
* receipts: the body re-seals to its id and equals the stored receipt bytes.
* symbol bindings: ``SymbolBinding`` round-trip, fingerprint equals the row id, bound to the
  session's model identity and execution digest.
* slices: seal equals the id, the session matches, the embedded graph snapshot verifies.
* legacy circuit records: ``mdb.core.records.verify_legacy`` plus kind, session, model identity
  and execution.
* state annotations: the sealed index (schema, digest, session binding) and each object (bytes
  digest, canonical bytes, body digest, id, qualified label, index summary, session binding;
  no path outside the session directory, no symlink).
* symbol lifecycles: lifecycle seal (bounded canonical JSON), binding round-trip and fingerprint,
  child occurrence/use seals and binding, session binding, the recorded receipt/consumer join of
  every use and definition re-derived from verified cuts and receipts, and slice links resolved.
  Not re-derived here (the lifecycle writer checks them when it records): clock witnesses,
  invalidation clocks, location subsets and slot-metadata support of the binding.
* watch events: the event seal.

``session_graph`` returns the sealed session history graph (execution + knowledge) with the same
node, edge, event and snapshot identities the project writer's graph has for the same rows; a row
that cannot be placed (unreadable, or referring to a row that is not in the graph) is left out and
reported through ``refuse``.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

from saturn_pub.values import canonical, describe, digest

from . import graph_seal as gs
from .pointers import _sealed, _sha, check_bounded

Refuse = Callable[[str, str, BaseException], None]

LIFECYCLE_SCHEMA = "mdb-symbol-lifecycle-v1"
CIRCUIT_SCHEMA = "mdb-circuit-investigation-v1"
ANNOTATION_SCHEMA = "mdb-state-annotation-v1"
ANNOTATION_INDEX_SCHEMA = "mdb-state-annotation-index-v1"
READ_SCHEMA = "mdb-symbol-read-v2"
UNKNOWN_CAUSAL_LABEL = "unknown (observed only; no intervention)"
JOIN_FIELDS = frozenset({"definitions", "uses", "cuts", "receipts", "native_payload_reads",
                         "external_references", "authority", "causal_effect"})
COORDINATE_CLOCKS = frozenset({"execution-point", "token-position", "token", "depth",
                               "decoder-depth", "layer"})


class ProjectReadError(ValueError):
    """A stored row does not verify, or the project cannot be opened read-only."""


def _strict(kind, identifier, error):
    raise error


# --------------------------------------------------------------------------- bounded JSON

_LC_DEPTH, _LC_ITEMS, _LC_STRING, _LC_TOTAL = 12, 512, 16 * 1024, 256 * 1024


def _lc_plain(value, depth=0):
    """The lifecycle record's bounded JSON normalisation (text keys, finite floats, limits)."""
    if depth > _LC_DEPTH:
        raise ProjectReadError("lifecycle exceeds metadata nesting limit")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        if len(value.encode("utf-8")) > _LC_STRING:
            raise ProjectReadError("lifecycle string is too large")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProjectReadError("lifecycle contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if len(value) > _LC_ITEMS:
            raise ProjectReadError("lifecycle has too many metadata keys")
        out = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ProjectReadError("lifecycle contains a non-text key")
            out[key] = _lc_plain(item, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        if len(value) > _LC_ITEMS:
            raise ProjectReadError("lifecycle has too many metadata items")
        return [_lc_plain(item, depth + 1) for item in value]
    raise ProjectReadError(f"lifecycle contains unsupported {type(value).__name__}")


def lifecycle_digest(value) -> str:
    """Seal digest of symbol lifecycle records (and their occurrences and uses)."""
    encoded = json.dumps(_lc_plain(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > _LC_TOTAL:
        raise ProjectReadError("lifecycle record exceeds the bounded metadata budget")
    return sha256(encoded).hexdigest()


def _sealed_child(value, label):
    if not isinstance(value, Mapping):
        raise ProjectReadError(f"{label} is not an object")
    body = {k: v for k, v in value.items() if k != "fingerprint"}
    if value.get("fingerprint") != lifecycle_digest(body):
        raise ProjectReadError(f"{label} fingerprint mismatch")
    return value


# Key order of the lifecycle writer's ``to_dict`` (stored rows are canonical, i.e. key-sorted).
# Readers that truncate by position (``bound_json``) depend on it; opaque mappings inside a
# record (metadata, readings, observations) keep their stored order, as the writer's do.
_LC_KEYS = ("schema", "symbol", "binding", "binding_fingerprint", "definitions", "uses",
            "statuses", "invalidations", "claim_boundary", "authority", "metadata", "fingerprint")
_OCC_KEYS = ("schema", "occurrence_id", "binding_fingerprint", "binding", "session_id",
             "source_cut", "result_cut", "branch", "execution", "receipt", "locations",
             "location_set", "coalition", "location_sets", "clock", "spectral_address",
             "vm_bundle", "cot_refs", "definition_refs", "evidence_refs", "status",
             "validity_status", "validity_support", "authority", "metadata", "fingerprint")
_USE_KEYS = ("schema", "use_id", "binding_fingerprint", "session_id", "consumer", "source_cut",
             "parent_cut", "result_cut", "branch", "execution", "receipt", "locations", "clock",
             "signed_observations", "evidence_refs", "status", "validity_status",
             "validity_support", "authority", "metadata", "fingerprint")
_CLOCK_KEYS = ("schema", "clock_id", "kind", "domain", "reading", "provenance",
               "validity_status", "validity_support", "lifetime", "status", "metadata",
               "fingerprint")
_INVALIDATION_KEYS = ("scope", "reason", "status", "evidence_refs", "clock", "metadata")
_LOCATION_KEYS = ("slot", "selector", "role", "execution_point", "state_schema")
_AUTHORITY_KEYS = ("write", "causal_effect")
_STATUS_KEYS = ("mechanics", "instrument", "trend", "terminal")


def _ordered(value, keys, nested=None):
    if not isinstance(value, dict):
        return value
    out = {k: value[k] for k in keys if k in value}
    out.update({k: v for k, v in value.items() if k not in out})
    for key, order in (nested or {}).items():
        if key in out:
            out[key] = order(out[key])
    return out


def _ordered_binding(value):
    from saturn_pub.symbols import SymbolBinding

    return value if not isinstance(value, dict) else SymbolBinding.from_dict(value).to_dict()


def _ordered_clock(value):
    return _ordered(value, _CLOCK_KEYS)


def _locations(values):
    return [_ordered(v, _LOCATION_KEYS) for v in values] if isinstance(values, list) else values


def _event_order(keys):
    def order(value):
        return _ordered(value, keys, {"binding": _ordered_binding, "locations": _locations,
                                      "clock": _ordered_clock,
                                      "authority": lambda a: _ordered(a, _AUTHORITY_KEYS)})
    return order


def _lifecycle_order(body):
    """``body`` with the writer's key order (the content is unchanged)."""
    def each(order):
        return lambda rows: [order(r) for r in rows] if isinstance(rows, list) else rows
    return _ordered(body, _LC_KEYS, {
        "binding": _ordered_binding, "definitions": each(_event_order(_OCC_KEYS)),
        "uses": each(_event_order(_USE_KEYS)),
        "statuses": lambda s: _ordered(s, _STATUS_KEYS),
        "invalidations": each(lambda i: _ordered(i, _INVALIDATION_KEYS,
                                                 {"clock": _ordered_clock})),
        "authority": lambda a: _ordered(a, _AUTHORITY_KEYS)})


def _annotation_text(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 512 \
            or any(ord(c) < 32 for c in value):
        raise ProjectReadError(f"{label} must be short non-empty text")
    return value.strip()


def _safe_path(root: Path, path: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(root):
        raise ProjectReadError("annotation path is outside the session root")
    if path.resolve(strict=False) != path:
        raise ProjectReadError("annotation path is outside the session root")
    for candidate in (path, *path.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise ProjectReadError("annotation path traverses a symlink")
    return path


def _receipt_id(value):
    if value is not None and not isinstance(value, str):
        raise ProjectReadError("native reference must be a string")
    return value.removeprefix("cas://sha256/") if value else None


# --------------------------------------------------------------------------- reader

class ProjectReader:
    """A recorded project opened read-only: every connection is ``mode=ro`` + ``query_only``.

    * no ``-wal`` file next to the database: ``file:...?mode=ro&immutable=1`` (SQLite takes no
      locks and creates no side file; the database is assumed not to change while read);
    * a ``-wal`` file exists: ``mode=ro``, which needs a writable directory for SQLite's ``-shm``
      wal-index; when the directory is not writable this raises ``ValueError`` at open.
    """

    read_only = True

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        if (self.root / "project.sqlite-wal").exists() and not os.access(self.root, os.W_OK):
            raise ValueError(
                f"{self.root} is a WAL-mode project (it has a project.sqlite-wal file) in a "
                "directory that is not writable; SQLite needs to create its -shm wal-index there "
                "to read it. Make the directory writable, or serve a rollback-journal copy of the "
                "project (sqlite3 project.sqlite 'PRAGMA journal_mode=DELETE').")
        if not (self.root / "project.sqlite").is_file() or not (self.root / "store").is_dir():
            raise ValueError("read-only project requires an existing index and store")
        from saturn_pub.store import LocalStore

        self.store = LocalStore(self.root / "store")
        with self._db() as db:
            if db.execute("PRAGMA user_version").fetchone()[0] != 3:
                raise ValueError("read-only project requires schema 3; migrate a writable copy first")

    @contextmanager
    def _db(self):
        path = self.root / "project.sqlite"
        mode = "mode=ro" if path.with_name(path.name + "-wal").exists() else "mode=ro&immutable=1"
        db = sqlite3.connect(path.as_uri() + "?" + mode, uri=True, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA query_only=ON")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _json(value):
        return canonical(value).decode("utf-8")

    def _tables(self, db):
        return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    # ----------------------------------------------------------------------- sessions and cuts

    def sessions(self):
        """Session ids (one corrupt session row never hides the others)."""
        with self._db() as db:
            return [row["id"] for row in db.execute("SELECT id FROM sessions ORDER BY id")]

    def session(self, sid):
        with self._db() as db:
            row = db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if row is None:
            raise ValueError(f"unknown project session: {sid}")
        return dict(row)

    def _binding(self, sid):
        row = self.session(sid)
        return json.loads(row["identity"]), json.loads(row["execution"])

    def cut_ids(self, sid):
        self.session(sid)
        with self._db() as db:
            return [r[0] for r in db.execute(
                "SELECT id FROM cuts WHERE session_id=? ORDER BY rowid", (sid,))]

    def cut_descriptor(self, sid, cut_id):
        """Verified native cut metadata (no tensor page is read)."""
        with self._db() as db:
            row = db.execute("SELECT manifest FROM cuts WHERE session_id=? AND id=?",
                             (sid, cut_id)).fetchone()
        if row is None:
            raise ValueError("cut is not retained in this project session")
        manifest = self.store.inspect(cut_id)
        if self._json(manifest) != row[0]:
            raise ValueError("project cut index differs from native cut")
        identity, execution = self._binding(sid)
        if manifest["model_identity"] != identity or manifest["execution"] != execution:
            raise ValueError("native cut differs from the project session binding")
        metadata = json.loads((self.store.root / "cuts" / f"{cut_id}.json").read_bytes())
        point = metadata.get("execution_point")
        if describe(point) != manifest.get("execution_point"):
            raise ValueError("native execution point differs from its sealed manifest")
        surface = metadata.get("surface")
        if (None if surface is None else describe(surface)) != manifest.get("surface"):
            raise ValueError("native surface differs from its sealed manifest")
        return {"fingerprint": cut_id, "manifest": manifest,
                "model_identity": manifest["model_identity"], "parent": manifest["parent"],
                "boundary": manifest["boundary"], "execution_point": point or {},
                "surface": surface or {}}

    def receipts(self, sid):
        """Verified native receipts as dicts (``Receipt.to_dict()``), insertion order."""
        from saturn_pub.core import Receipt

        self.session(sid)
        with self._db() as db:
            rows = list(db.execute(
                "SELECT id, body FROM receipts WHERE session_id=? ORDER BY rowid", (sid,)))
        out = []
        for row in rows:
            data = json.loads(row["body"])
            fingerprint = data.pop("fingerprint")
            receipt = Receipt(canonical(data))
            if receipt.fingerprint != fingerprint or fingerprint != row["id"]:
                raise ValueError("project receipt seal changed")
            raw = (self.store.root / "receipts" / f"{fingerprint}.json").read_bytes()
            if raw != canonical(receipt.to_dict()):
                raise ValueError("native receipt bytes differ from the project index")
            out.append(receipt.to_dict())
        return out

    def tensor_info(self, ref):
        """Shape/dtype of a stored tensor (verified descriptor; no tensor is built)."""
        return self.store.tensor_info(ref)

    # ----------------------------------------------------------------------- symbols and slices

    def _symbol(self, sid, identifier, body, identity, execution):
        from saturn_pub.symbols import SymbolBinding

        value = SymbolBinding.from_dict(json.loads(body)).to_dict()
        if value["fingerprint"] != identifier:
            raise ValueError("project symbol index differs from its seal")
        if value["model_identity"] != identity or value["execution"] != digest(execution):
            raise ValueError("project symbol differs from its session binding")
        return value

    def symbols(self, sid, *, refuse: Refuse = _strict):
        """Verified symbol bindings (``SymbolBinding.to_dict()``), insertion order."""
        identity, execution = self._binding(sid)
        with self._db() as db:
            rows = db.execute("SELECT id, body FROM symbols WHERE session_id=? ORDER BY rowid",
                              (sid,)).fetchall()
        out = []
        for row in rows:
            try:
                out.append(self._symbol(sid, row["id"], row["body"], identity, execution))
            except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as error:
                refuse("symbol", str(row["id"]), error)
        return out

    def slice_ids(self, sid):
        self.session(sid)
        with self._db() as db:
            return [r[0] for r in db.execute(
                "SELECT id FROM slices WHERE session_id=? ORDER BY rowid", (sid,))]

    def slice_record(self, sid, identifier):
        self.session(sid)
        with self._db() as db:
            row = db.execute("SELECT body FROM slices WHERE session_id=? AND id=?",
                             (sid, identifier)).fetchone()
        if row is None:
            raise ValueError("slice is not retained in this project session")
        body = json.loads(row[0])
        if not isinstance(body, dict) or digest(body) != identifier \
                or body.get("session_id") != sid:
            raise ValueError("project slice seal changed")
        gs.verify_snapshot(body.get("graph"))
        return {"slice_id": identifier, **body}

    def slices(self, sid, *, refuse: Refuse = _strict):
        out = []
        for identifier in self.slice_ids(sid):
            try:
                out.append(self.slice_record(sid, identifier))
            except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as error:
                refuse("slice", str(identifier), error)
        return out

    # ----------------------------------------------------------------------- circuit records

    def circuit_records(self, sid, *, refuse: Refuse = _strict):
        """Verified legacy ``circuit_records`` rows ``{record_id, **body}``, insertion order."""
        from .records import verify_legacy

        identity, execution = self._binding(sid)
        with self._db() as db:
            if "circuit_records" not in self._tables(db):
                return []
            rows = db.execute("SELECT id, kind, body FROM circuit_records WHERE session_id=? "
                              "ORDER BY rowid", (sid,)).fetchall()
        out = []
        for row in rows:
            try:
                body = json.loads(row["body"])
                value = verify_legacy(body, row["id"])
                if (body.get("schema") != CIRCUIT_SCHEMA or body.get("session_id") != sid
                        or body.get("kind") != row["kind"]
                        or body.get("model_identity") != identity
                        or body.get("execution") != execution):
                    raise ProjectReadError("circuit record seal or session binding changed")
                out.append(value)
            except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as error:
                refuse("circuit-record", str(row["id"]), error)
        return out

    # ----------------------------------------------------------------------- annotations

    def _annotation_binding(self, sid):
        identity, execution = self._binding(sid)
        return {"session_id": sid, "family": "saturn-pub", "lane": "public-session",
                "model_identity_sha256": digest({"model_id": identity, "execution": execution}),
                "publication_mode": "project"}

    def _annotation_index(self, sid):
        """``(session root, session binding, index entries)`` of the state-annotation sidecar."""
        root = (self.root / "sessions" / digest(sid)).resolve()
        binding = self._annotation_binding(sid)
        sidecar = _safe_path(root, root / "state-annotations")
        index_path = _safe_path(root, sidecar / "index.json")
        objects = _safe_path(root, sidecar / "objects")
        if sidecar.is_symlink() or objects.is_symlink():
            raise ProjectReadError("annotation namespace cannot be a symlink")
        if not index_path.exists():
            return root, binding, []
        if index_path.is_symlink():
            raise ProjectReadError("annotation index cannot be a symlink")
        try:
            index = json.loads(index_path.read_bytes())
        except (OSError, ValueError, UnicodeDecodeError) as error:
            raise ProjectReadError("annotation index is unreadable") from error
        if not isinstance(index, dict) or set(index) != {
                "schema", "session_binding", "entries", "content_sha256"}:
            raise ProjectReadError("annotation index shape is invalid")
        body = {k: v for k, v in index.items() if k != "content_sha256"}
        if index["schema"] != ANNOTATION_INDEX_SCHEMA or index["content_sha256"] != gs.digest(body):
            raise ProjectReadError("annotation index seal changed")
        if index["session_binding"] != binding:
            raise ProjectReadError("annotation index session binding changed")
        if not isinstance(index["entries"], list) or any(
                not isinstance(e, Mapping) for e in index["entries"]):
            raise ProjectReadError("annotation index entries are invalid")
        return root, binding, index["entries"]

    @staticmethod
    def _annotation(root, binding, entry):
        path = _safe_path(root, root / _annotation_text(entry.get("object_path"),
                                                        "annotation object path"))
        if path.is_symlink():
            raise ProjectReadError("annotation object cannot be a symlink")
        try:
            raw = path.read_bytes()
            annotation = json.loads(raw)
        except (OSError, ValueError, UnicodeDecodeError) as error:
            raise ProjectReadError("annotation object is unreadable") from error
        if not isinstance(annotation, dict):
            raise ProjectReadError("annotation object is not an object")
        if sha256(raw).hexdigest() != entry.get("object_sha256") or gs.canonical(annotation) != raw:
            raise ProjectReadError("annotation object bytes changed")
        payload = {k: v for k, v in annotation.items()
                   if k not in {"annotation_id", "qualified_label", "body_sha256"}}
        body_sha256 = gs.digest(payload)
        expected_id = f"state-annotation://sha256/{body_sha256}"
        expected_label = f"{payload['namespace']}::{payload['label']}@sha256:{body_sha256}"
        if (annotation.get("schema") != ANNOTATION_SCHEMA
                or annotation.get("body_sha256") != body_sha256
                or annotation.get("annotation_id") != expected_id
                or annotation.get("qualified_label") != expected_label
                or entry.get("annotation_id") != expected_id
                or entry.get("qualified_label") != expected_label
                or entry.get("body_sha256") != body_sha256):
            raise ProjectReadError("annotation object seal changed")
        expected_entry = {
            "annotation_id": annotation["annotation_id"],
            "qualified_label": annotation["qualified_label"], "label": annotation["label"],
            "namespace": annotation["namespace"], "cut_id": annotation["cut_id"],
            "role": annotation["role"], "stage": annotation["stage"], "tags": annotation["tags"],
            "body_sha256": annotation["body_sha256"], "object_sha256": sha256(raw).hexdigest(),
            "object_path": path.relative_to(root).as_posix()}
        if dict(entry) != expected_entry:
            raise ProjectReadError("annotation index summary differs from its sealed object")
        if annotation.get("session_binding") != binding:
            raise ProjectReadError("annotation object session binding changed")
        return annotation

    def annotations(self, sid, *, tag=None, refuse: Refuse = _strict):
        """``(index entry, verified annotation)`` pairs in index order (optionally one tag)."""
        try:
            root, binding, entries = self._annotation_index(sid)
        except (ValueError, OSError) as error:
            refuse("annotation", "index", error)
            return []
        out = []
        for entry in entries:
            ident = str(entry.get("annotation_id") or entry.get("qualified_label") or "?")
            try:
                if tag is not None and tag not in entry.get("tags", []):
                    continue
                out.append((dict(entry), self._annotation(root, binding, entry)))
            except (ValueError, KeyError, TypeError, AttributeError, OSError,
                    RecursionError) as error:
                refuse("annotation", ident, error)
        return out

    def findings(self, sid, *, refuse: Refuse = _strict):
        """Annotations tagged ``finding`` (verified objects), index order."""
        return [annotation for _, annotation in self.annotations(
            sid, tag="finding", refuse=lambda kind, ident, error: refuse("finding", ident, error))]

    # ----------------------------------------------------------------------- watch events

    def watch_events(self, sid, *, refuse: Refuse = _strict):
        info = json.loads(self.session(sid)["debug_info"])
        events = info.get("watch_events", []) if isinstance(info, dict) else []
        if not isinstance(events, list):
            refuse("watch-event", "watch_events", ProjectReadError("watch_events is not a list"))
            return []
        out = []
        for position, event in enumerate(events):
            try:
                if not isinstance(event, dict):
                    raise ProjectReadError("watch event is not an object")
                body = {k: v for k, v in event.items()
                        if k not in {"fingerprint", "boundary", "authority"}}
                if digest(body) != event["fingerprint"]:
                    raise ProjectReadError("project watch event seal changed")
                out.append(event)
            except (ValueError, KeyError, TypeError, RecursionError) as error:
                ident = event.get("fingerprint") if isinstance(event, dict) else None
                refuse("watch-event", str(ident or position), error)
        return out

    # ----------------------------------------------------------------------- symbol lifecycles

    def symbol_lifecycles(self, sid, *, refuse: Refuse = _strict):
        """Verified lifecycle rows ``{lifecycle_id, lifecycle, joins}``, insertion order."""
        self.session(sid)
        with self._db() as db:
            if "symbol_lifecycles" not in self._tables(db):
                return []
            rows = db.execute("SELECT id, body, joins FROM symbol_lifecycles WHERE session_id=? "
                              "ORDER BY rowid", (sid,)).fetchall()
        context = {}
        out = []
        for row in rows:
            try:
                out.append(self._lifecycle(sid, row["id"], row["body"], row["joins"], context))
            except (ValueError, KeyError, TypeError, AttributeError, OSError,
                    RecursionError) as error:
                refuse("lifecycle", str(row["id"]), error)
        return out

    def _lifecycle(self, sid, identifier, body_text, joins_text, context):
        from saturn_pub.symbols import SymbolBinding

        body = json.loads(body_text)
        if not isinstance(body, dict) or body.get("schema") != LIFECYCLE_SCHEMA:
            raise ProjectReadError("unsupported symbol lifecycle schema")
        if body.get("fingerprint") != lifecycle_digest(
                {k: v for k, v in body.items() if k != "fingerprint"}) \
                or body["fingerprint"] != identifier:
            raise ProjectReadError("symbol lifecycle index differs from its seal")
        binding = body.get("binding")
        if not isinstance(binding, dict) or SymbolBinding.from_dict(binding).to_dict() != binding:
            raise ProjectReadError("lifecycle binding does not round-trip")
        binding_fp = binding["fingerprint"]
        if body.get("binding_fingerprint") != binding_fp or binding["symbol"] != body.get("symbol"):
            raise ProjectReadError("lifecycle binding fingerprint mismatch")
        identity, execution = self._binding(sid)
        if binding["model_identity"] != identity or binding["execution"] != digest(execution):
            raise ProjectReadError("lifecycle belongs to another model or execution")
        definitions = body.get("definitions") or []
        uses = body.get("uses") or []
        if not isinstance(definitions, list) or not isinstance(uses, list):
            raise ProjectReadError("lifecycle events are not lists")
        ids = []
        for row in definitions:
            _sealed_child(row, "symbol occurrence")
            ids.append(row.get("occurrence_id"))
        for row in uses:
            _sealed_child(row, "symbol use")
            ids.append(row.get("use_id"))
        if len(ids) != len(set(map(str, ids))):
            raise ProjectReadError("lifecycle occurrence/use IDs must be unique")
        for row in (*definitions, *uses):
            if row.get("binding_fingerprint") != binding_fp:
                raise ProjectReadError("lifecycle record belongs to another binding")
            if row.get("session_id") != sid:
                raise ProjectReadError("lifecycle event belongs to another project session")
            if row.get("execution") is not None and row["execution"] != binding["execution"]:
                raise ProjectReadError("symbol event execution changed")
        recorded = json.loads(joins_text)
        if not isinstance(recorded, dict) or set(recorded) != JOIN_FIELDS:
            raise ProjectReadError("lifecycle join contains unsupported fields")
        joins = self._joins(sid, body, binding, definitions, uses, context)
        self._check_joins(sid, recorded, joins, binding_fp)
        return {"lifecycle_id": identifier, "lifecycle": _lifecycle_order(body),
                "joins": recorded}

    def _joins(self, sid, body, binding, definitions, uses, context):
        """Re-derive the receipt/consumer join of every recorded event from verified rows."""
        if "receipts" not in context:
            context["receipts"] = {r["fingerprint"]: r for r in self.receipts(sid)}
            context["cuts"] = {}
        native_receipts, cut_cache = context["receipts"], context["cuts"]
        cuts, receipts = set(), {}

        def cut(identifier):
            if not isinstance(identifier, str):
                raise ProjectReadError("event cut reference must be a string")
            if identifier not in cut_cache:
                cut_cache[identifier] = self.cut_descriptor(sid, identifier)
            cuts.add(identifier)
            return cut_cache[identifier]

        def receipt(row, is_use):
            ident = _receipt_id(row.get("receipt"))
            if ident is None:
                return None
            if ident not in native_receipts:
                raise ProjectReadError("lifecycle receipt is not retained in this session")
            native = native_receipts[ident]
            if (native.get("model_identity") != binding["model_identity"]
                    or digest(native.get("execution")) != binding["execution"]):
                raise ProjectReadError("native receipt belongs to another model or execution")
            if native.get("status") == "refused":
                raise ProjectReadError("refused receipt cannot describe a completed symbol event")
            referenced = {native.get(k) for k in ("parent", "result", "candidate")} - {None}
            if row.get("result_cut") and row["result_cut"] != native.get("result"):
                raise ProjectReadError("symbol result cut differs from its native receipt")
            if row.get("parent_cut") and row["parent_cut"] != native.get("parent"):
                raise ProjectReadError("symbol parent cut differs from its native receipt")
            if row.get("source_cut") and row["source_cut"] not in referenced and not is_use:
                raise ProjectReadError("definition cut is not referenced by its native receipt")
            receipts[ident] = native
            return ident

        for row in (*definitions, *uses):
            for key in ("source_cut", "result_cut", "parent_cut"):
                if row.get(key):
                    cut(row[key])
        for invalidation in body.get("invalidations") or []:
            clock = invalidation.get("clock") if isinstance(invalidation, Mapping) else None
            if isinstance(clock, Mapping):
                source = _receipt_id(clock.get("metadata", {}).get("source_cut"))
                if source is not None:
                    cut(source)
        out_definitions, out_uses = [], []
        for row in definitions:
            cut(row.get("source_cut"))
            if not row.get("locations"):
                raise ProjectReadError("definition has no locations in the qualified binding")
            item ={"occurrence": row.get("occurrence_id"), "source_cut": row.get("source_cut"),
                    "receipt": receipt(row, False), "support": "metadata-support-verified",
                    "slice_basis": "full-root-binding-declared-dataflow"}
            spectral = row.get("spectral_address")
            if spectral:
                if "canonical_key" in spectral:
                    if spectral["canonical_key"]["symbol_id"] != body["symbol"]:
                        raise ProjectReadError("spectral symbol differs from the binding")
                    item["spectral_support"] = ("typed-identity-verified; "
                                                "supplied-native-association")
                else:
                    item["spectral_support"] = "opaque-unjoined-reference"
            bundle = row.get("vm_bundle")
            if bundle:
                if "schema" in bundle:
                    expected = (("model_identity", "id", binding["model_identity"]),
                                ("abi_identity", "execution", binding["execution"]),
                                ("input_identity", "source_cut", row.get("source_cut")),
                                ("input_identity", "context", binding.get("context")))
                    for part, name, value in expected:
                        if name in bundle[part] and bundle[part][name] != value:
                            raise ProjectReadError("VM bundle differs from the native source")
                    exact = (bundle["model_identity"] == {"id": binding["model_identity"]}
                             and bundle["abi_identity"].get("execution") == binding["execution"]
                             and bundle["input_identity"].get("source_cut") == row.get("source_cut")
                             and bundle["input_identity"].get("context") == binding.get("context"))
                    item["vm_support"] = ("native-source-metadata-matched" if exact else
                                          "bundle-seal-verified; source-association-unverified")
                else:
                    item["vm_support"] = "opaque-unjoined-reference"
            out_definitions.append(item)
        for row in uses:
            if row.get("source_cut"):
                cut(row["source_cut"])
            receipt_id = receipt(row, True)
            closed = bool(receipt_id and row.get("parent_cut") and row.get("result_cut"))
            if receipt_id and not closed and row.get("status") in {"observed", "validated"}:
                raise ProjectReadError("observed symbol use requires receipt parent/result closure")
            declared = set()
            for key in ("parent_cut", "result_cut"):
                if row.get(key):
                    consumer = cut(row[key])["surface"].get("consumer")
                    if consumer:
                        declared.add(consumer)
            if closed:
                for step in receipts[receipt_id].get("transitions", []):
                    consumer = step.get("native_operation", {}).get("consumer")
                    if consumer:
                        declared.add(consumer)
            if declared and row.get("consumer") not in declared:
                raise ProjectReadError("symbol consumer differs from the native contract")
            out_uses.append({
                "use": row.get("use_id"), "source_cut": row.get("source_cut"),
                "result_cut": row.get("result_cut"), "receipt": receipt_id,
                "consumer": row.get("consumer"),
                "consumer_support": ("native-contract-matched" if declared and closed
                                     else "unverified-label"),
                "receipt_closure": "native-parent-result-matched" if closed else "incomplete",
                "support": "recorded-consumer-association"})
        return {"definitions": out_definitions, "uses": out_uses, "cuts": sorted(cuts),
                "receipts": sorted(receipts), "native_payload_reads": 0,
                "external_references": ("retained descriptors; external payload availability "
                                        "unverified"),
                "authority": "read-only-evidence-index", "causal_effect": "not-established"}

    def _check_joins(self, sid, recorded, joins, binding_fp):
        if not isinstance(recorded["definitions"], list) or not isinstance(recorded["uses"], list):
            raise ProjectReadError("lifecycle join changed")
        for definition in recorded["definitions"]:
            if not isinstance(definition, dict):
                raise ProjectReadError("lifecycle definition join changed")
            if definition.get("slice_id"):
                sliced = self.slice_record(sid, definition["slice_id"])
                if (sliced["cut_id"] != definition["source_cut"]
                        or sliced["symbol_binding"]["fingerprint"] != binding_fp
                        or sliced["complete"] != definition.get("slice_complete")
                        or sliced["scope"] != definition.get("slice_scope")):
                    raise ProjectReadError("lifecycle slice differs from its native source")
        stripped = [{k: v for k, v in row.items()
                     if k not in {"slice_id", "slice_complete", "slice_scope"}}
                    for row in recorded["definitions"]]
        if len(stripped) != len(joins["definitions"]):
            raise ProjectReadError("lifecycle definition join changed")
        expected = [current if "slice_basis" in saved else
                    {k: v for k, v in current.items() if k != "slice_basis"}
                    for saved, current in zip(stripped, joins["definitions"])]
        if stripped != expected:
            raise ProjectReadError("lifecycle definition join changed")
        if len(recorded["uses"]) != len(joins["uses"]):
            raise ProjectReadError("lifecycle use join changed")
        expected = [current if isinstance(saved, dict) and "receipt_closure" in saved else
                    {k: v for k, v in current.items() if k != "receipt_closure"}
                    for saved, current in zip(recorded["uses"], joins["uses"])]
        if recorded["uses"] != expected:
            raise ProjectReadError("lifecycle use join changed")
        for field in JOIN_FIELDS - {"definitions", "uses"}:
            if joins[field] != recorded[field]:
                raise ProjectReadError("lifecycle native reference join changed")

    # ----------------------------------------------------------------------- session graph

    def session_graph(self, sid, *, refuse: Refuse = _strict):
        """The sealed session history graph (snapshot dict), built from verified rows only.

        Order and identities follow the project writer's session graph: cut states, their
        ``continues`` edges, receipt actions, state annotations, symbol bindings, slices, legacy
        circuit records, watch events. Cuts and receipts are strict (a failure raises); any other
        row that is unreadable or refers to a row not in the graph is left out and reported
        through ``refuse`` (records referring to a left-out record are left out in turn).
        """
        row = self.session(sid)
        identity = json.loads(row["identity"])
        session_binding = gs.binding(session=sid)
        g = gs.GraphBuilder(metadata={"source": "mdb-project",
                                      "execution_authority": "native-runtime"})
        nodes, cuts = {}, {}
        for identifier in self.cut_ids(sid):
            cut = self.cut_descriptor(sid, identifier)
            cuts[identifier] = cut
            nodes[identifier] = g.add_node(g.state_node(
                f"cas://sha256/{identifier}", identifier,
                binding_=gs.binding(model=cut["model_identity"], session=sid,
                                    clock=dict(cut["execution_point"])),
                metadata={"native_format": "saturn-pub", "boundary": cut["boundary"]}))
        for identifier, cut in cuts.items():
            if cut["parent"] in nodes:
                g.add_edge(nodes[cut["parent"]], nodes[identifier], "continues",
                           evidence_=gs.evidence("observed"))
        receipt_binding = gs.binding(model=identity, session=sid)
        for data in self.receipts(sid):
            parent, result = data.get("parent"), data.get("result")
            output = (nodes.get(result) if data.get("operation") not in {"restore", "abort"}
                      and result != parent else None)
            inputs = [nodes[parent]] if parent in nodes else []
            if data.get("operation") in {"commit", "abort"}:
                candidate = nodes.get(data.get("candidate"))
                if candidate and candidate not in inputs:
                    inputs.append(candidate)
            ref = f"cas://sha256/{data['fingerprint']}"
            action = g.add_action(ref, inputs=inputs, output=output, receipt_refs=(ref,),
                                  binding_=receipt_binding, metadata={"native_receipt": data},
                                  evidence_class="observed")
            if data.get("operation") == "restore" and result in nodes:
                g.add_edge(action, nodes[result], "describes", graph="knowledge",
                           evidence_=gs.evidence("observed"), metadata={"role": "restore-target"})

        def attempt(kind, ident, build):
            try:
                return build()
            except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as error:
                refuse(kind, ident, error)
                return None

        # ---- gather and verify every knowledge row first; place them in writer order after
        records = {r["record_id"]: r for r in self.circuit_records(sid, refuse=refuse)}
        record_nodes = {rid: attempt("circuit-record", rid, lambda r=r: g.source_node(
            "circuit://sha256/" + r["record_id"], binding_=session_binding,
            labels=("circuit-" + r["kind"],),
            metadata={"record_id": r["record_id"], "kind": r["kind"],
                      "focus_id": r.get("focus_id"),
                      "terminal_status": r.get("terminal_status", "not-established")}))
            for rid, r in records.items()}
        record_nodes = {rid: n for rid, n in record_nodes.items() if n is not None}

        def record_refs(r):
            refs = []
            for key in ("focus_id", "plan_id"):
                if r.get(key):
                    refs.append(r[key])
            arms = r.get("arms", [])
            if not isinstance(arms, list):
                raise ProjectReadError("circuit record arms is not a list")
            refs += arms
            for key in ("supporting_cut", "input_cut", "result_cut"):
                hash(r.get(key))
            return refs

        changed = True
        while changed:
            changed = False
            for rid in list(record_nodes):
                try:
                    refs = record_refs(records[rid])
                    missing = [x for x in refs if not isinstance(x, str) or x not in record_nodes]
                    if missing:
                        raise ProjectReadError("circuit graph has an unresolved record reference")
                except (ValueError, TypeError) as error:
                    refuse("circuit-record", rid, error)
                    del record_nodes[rid]
                    changed = True

        annotations = []
        for entry, annotation in self.annotations(sid, refuse=refuse):
            ident = entry.get("annotation_id", "?")

            def build(entry=entry, annotation=annotation):
                details = annotation.get("details", {})
                slice_id = details.get("slice_id")
                investigation = details.get("investigation_id")
                if entry["cut_id"] not in nodes:
                    raise ProjectReadError("annotation names a cut that is not in the session")
                if investigation and (not isinstance(investigation, str)
                                      or investigation not in record_nodes):
                    raise ProjectReadError("finding has an unresolved circuit record")
                node = g.source_node(entry["annotation_id"], labels=(entry["label"],),
                                     binding_=session_binding, metadata=annotation)
                return node, entry["cut_id"], slice_id, investigation
            built = attempt("annotation", str(ident), build)
            if built is not None:
                annotations.append(built)

        symbols = {}
        for value in self.symbols(sid, refuse=refuse):
            node = attempt("symbol", value["fingerprint"], lambda v=value: g.source_node(
                f"symbol://sha256/{v['fingerprint']}", binding_=session_binding, metadata=v,
                labels=(v["symbol"],)))
            if node is not None:
                symbols[value["fingerprint"]] = node

        slices = []
        for sliced in self.slices(sid, refuse=refuse):
            identifier = sliced["slice_id"]

            def build(sliced=sliced, identifier=identifier):
                if sliced["cut_id"] not in nodes:
                    raise ProjectReadError("slice names a cut that is not in the session")
                symbol = None
                if sliced["symbol_binding"]:
                    symbol = sliced["symbol_binding"]["fingerprint"]
                    if symbol not in symbols:
                        raise ProjectReadError("slice names a symbol binding not in the graph")
                node = g.source_node(f"slice://sha256/{identifier}", binding_=session_binding,
                                     labels=(sliced["component"],),
                                     metadata={"slice_id": identifier,
                                               "graph_snapshot": sliced["graph"]["snapshot_id"],
                                               "support": sliced["support"],
                                               "complete": sliced["complete"]})
                return identifier, node, sliced["cut_id"], symbol
            built = attempt("slice", identifier, build)
            if built is not None:
                slices.append(built)

        watches = []
        for event in self.watch_events(sid, refuse=refuse):
            ident = str(event.get("fingerprint"))

            def build(event=event):
                target = event["cut"]
                hash(target)
                node = g.output_node(f"watch://sha256/{event['fingerprint']}",
                                     binding_=receipt_binding, metadata=event,
                                     evidence_=gs.evidence("observed"),
                                     labels=("diagnostic-watch",))
                return node, target
            built = attempt("watch-event", ident, build)
            if built is not None:
                watches.append(built)

        # ---- place, in the writer's order
        annotation_ids = []
        for node, cut_id, slice_id, investigation in annotations:
            source = g.add_node(node)
            g.add_edge(source, nodes[cut_id], "annotates", graph="knowledge")
            annotation_ids.append((source, slice_id, investigation))
        for fingerprint, node in symbols.items():
            symbols[fingerprint] = g.add_node(node)
        for identifier, node, cut_id, symbol in slices:
            source = g.add_node(node)
            g.add_edge(source, nodes[cut_id], "describes", graph="knowledge")
            if symbol is not None:
                g.add_edge(symbols[symbol], source, "describes", graph="knowledge")
            for annotation, slice_id, _ in annotation_ids:
                if slice_id and slice_id == identifier:
                    g.add_edge(annotation, source, "related_to", graph="knowledge")
        placed = {rid: g.add_node(node) for rid, node in record_nodes.items()}
        for rid, node in placed.items():
            r = records[rid]
            for key in ("focus_id", "plan_id"):
                if r.get(key):
                    g.add_edge(node, placed[r[key]], "related_to", graph="knowledge")
            for key in ("supporting_cut", "input_cut", "result_cut"):
                if r.get(key) in nodes:
                    g.add_edge(node, nodes[r[key]], "describes", graph="knowledge")
            for arm in r.get("arms", []):
                g.add_edge(node, placed[arm], "related_to", graph="knowledge")
        for annotation, _, investigation in annotation_ids:
            if investigation:
                g.add_edge(annotation, placed[investigation], "related_to", graph="knowledge")
        for node, target in watches:
            output = g.add_node(node)
            if target in nodes:
                g.add_edge(output, nodes[target], "describes", graph="knowledge",
                           evidence_=gs.evidence("observed"))
        return g.seal()


# --------------------------------------------------------------------------- symbol reads

def lifecycle_reads(body) -> list[dict]:
    """Attention reads recorded in a verified symbol lifecycle body, as sealed
    ``mdb-symbol-read-v2`` dicts (one per query-head use; causal status ``unknown``).

    The read source is the project binding: rows from ``uses[].metadata.source_row`` (``[0]``
    when none), assignment ``supplied``. Empty when the lifecycle carries no attention trace or no
    uses, or when a use does not hold exactly one ``source_mass`` observation.
    """
    try:
        return _lifecycle_reads(body)
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError):
        return []


def _lifecycle_reads(body):
    metadata = body.get("metadata") or {}
    trace = metadata.get("attention_trace")
    uses = body.get("uses")
    if not trace or not uses:
        return []
    layer = trace.get("layer")
    if layer is None:
        return []
    binding_fp = body["binding_fingerprint"]
    _sha(binding_fp, "binding fingerprint")
    rows = sorted({int(u["metadata"].get("source_row", 0)) for u in uses
                   if isinstance(u.get("metadata"), dict) and "source_row" in u["metadata"]})
    rows = check_bounded([int(r) for r in (rows or [0])], "source rows")
    if not body.get("symbol") or any(r < 0 for r in rows):
        raise ProjectReadError("a read source needs a name and non-negative token rows")
    source = {"name": body["symbol"], "rows": rows, "assignment": "supplied",
              "provenance": check_bounded({"by": "project binding " + str(binding_fp)[:12]},
                                          "source provenance"),
              "binding_fingerprint": None}
    out = []
    for use in uses:
        meta = dict(use["metadata"])
        masses = [o["value"] for o in use["signed_observations"]
                  if isinstance(o, Mapping) and o.get("observable") == "source_mass"]
        if len(masses) != 1:
            raise ProjectReadError("use lacks exactly one observed mass")
        block, head = int(layer), int(meta["query_head"])
        if block < 0 or head < 0:
            raise ProjectReadError("reader block/head must be non-negative integers")
        mass = masses[0]
        if isinstance(mass, bool) or not isinstance(mass, (int, float)) or not math.isfinite(mass):
            raise ProjectReadError("observed mass must be a finite number")
        reader = {"block": block, "head": head, "step": None, "steps": None,
                  "kv_group": meta.get("kv_head"), "block_label": f"layer.{layer}",
                  "receivers": "query row", "receiver_row": meta.get("receiver_row")}
        read = {"schema": READ_SCHEMA, "source": source, "reader": reader, "mass": float(mass),
                "mass_kind": "attention probability on source row", "surface": "project-session",
                "backend": "qwen-gqa", "binding_fingerprint": binding_fp,
                "use_id": use["use_id"], "use_fingerprint": use["fingerprint"],
                "lifecycle_fingerprint": body["fingerprint"], "sets": [],
                "causal": {"status": "unknown", "label": UNKNOWN_CAUSAL_LABEL, "results": []},
                "evidence": check_bounded(list(use.get("evidence_refs") or ()), "read evidence"),
                "metadata": check_bounded({"trace_fingerprint": meta.get("trace_fingerprint")},
                                          "read metadata"),
                "model": None}
        out.append(_sealed(read, "symbol read"))
    return out
