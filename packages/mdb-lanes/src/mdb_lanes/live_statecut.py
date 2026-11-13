"""A local live-watch to durable StateCut bridge.

The worker-side :class:`saturn.debugger_worker.DebugWorkerController` already
owns watch predicates, safe-point ordering, pause state, and adapter checkpoint
authority.  This module wraps that controller; it does not replace it.  On a
watch fire, the bridge snapshots the caller-supplied live state into the local
physical payload CAS from :mod:`saturn.vm_payload_cache` and publishes an
immutable content-addressed cut manifest.

Boundary: this is local runtime mechanics.  It does not expose an mrun HTTP
endpoint, schedule work, or make a native-consumer semantic claim.  The
controller/adapter decides *when* a cut is valid; this module makes the bytes
reopenable and tamper-evident.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .debugger_protocol import DebugRequest, WatchSpec
from .vm_payload_cache import (
    PAYLOAD_AUTHORITY_CACHE_ONLY,
    PayloadCacheError,
    PayloadCacheReceipt,
    PayloadRef,
    PayloadStore,
)

LIVE_STATECUT_SCHEMA = "saturn-live-statecut-v1"
LIVE_STATECUT_REF_PREFIX = "cas://live-statecut/sha256/"
LIVE_STATECUT_EVENT_SCHEMA = "saturn-live-statecut-event-v1"
LIVE_STATECUT_AUTHORITY = "controller-safe-point-plus-local-payload-cas"


class LiveStateCutError(ValueError):
    """Raised when a live cut cannot be captured, reopened, or verified."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LiveStateCutError("state-cut metadata must be strict JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise LiveStateCutError(f"{label} must be a SHA-256 digest")
    text = value.lower().removeprefix("sha256:")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise LiveStateCutError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiveStateCutError(f"{label} must be non-empty text")
    return value.strip()


def _json_value(value: Any, label: str) -> Any:
    try:
        return json.loads(_canonical(value).decode("utf-8"))
    except (LiveStateCutError, json.JSONDecodeError) as exc:
        raise LiveStateCutError(f"{label} must be strict JSON") from exc


def _mapping(value: Any, label: str, *, allow_empty: bool = True) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LiveStateCutError(f"{label} must be a JSON object")
    normalized = _json_value(dict(value), label)
    if not allow_empty and not normalized:
        raise LiveStateCutError(f"{label} must not be empty")
    return normalized


def _lookup_dotted(value: Any, path: str) -> Any:
    cursor = value
    for segment in str(path).split("."):
        if not isinstance(cursor, Mapping) or segment not in cursor:
            return None
        cursor = cursor[segment]
    return cursor


def _safe_cut_id(value: Any) -> str:
    text = _text(value, "cut_id")
    # The controller's cut IDs are opaque; this ID is metadata, not a path.
    return text


def _boundary(
    value: str | Mapping[str, Any], *, index: int, total: int, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = _mapping(value, "boundary")
        if not {"kind", "index", "total"}.issubset(result):
            raise LiveStateCutError("boundary must include kind, index, and total")
        return result
    resolved_total = int(total)
    resolved_index = int(index)
    if resolved_total <= 0 or not 0 <= resolved_index <= resolved_total:
        raise LiveStateCutError("boundary requires total > 0 and 0 <= index <= total")
    # Debugger boundaries are family-specific strings; retain that exact name
    # in metadata while using the family-neutral StateCut step boundary kind.
    return {
        "kind": "step_cut",
        "index": resolved_index,
        "total": resolved_total,
        "runtime_boundary": _text(value, "runtime boundary"),
        "runtime_metadata": dict(metadata),
    }


def _state_value_kind(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "bytes"
    # JSON is the deliberately conservative fallback.  Array-like values are
    # only treated as physical arrays when put_array can prove shape/dtype.
    if isinstance(value, (Mapping, list, tuple, str, bool, int, float)) or value is None:
        return "json"
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return "array"
    return "json"


@dataclass(frozen=True, slots=True)
class DurableStateCut:
    """Metadata for one immutable local cut; payload bytes live in the CAS."""

    cut_id: str
    family: str
    state_schema_id: str
    model_identity: Mapping[str, Any]
    boundary: Mapping[str, Any]
    slots: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any]
    parent_cut: str | None
    fingerprint: str
    ref: str
    authority_boundary: Mapping[str, Any]

    schema: ClassVar[str] = LIVE_STATECUT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cut_id": self.cut_id,
            "family": self.family,
            "state_schema_id": self.state_schema_id,
            "model_identity": dict(self.model_identity),
            "boundary": dict(self.boundary),
            "slots": {name: dict(record) for name, record in self.slots.items()},
            "metadata": dict(self.metadata),
            "parent_cut": self.parent_cut,
            "fingerprint": self.fingerprint,
            "ref": self.ref,
            "authority_boundary": dict(self.authority_boundary),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DurableStateCut:
        if not isinstance(value, Mapping):
            raise LiveStateCutError("durable StateCut manifest must be an object")
        body = {key: value[key] for key in value if key not in {"fingerprint", "ref"}}
        declared = _digest(value.get("fingerprint"), "StateCut fingerprint")
        computed = _sha256(_canonical(body))
        if declared != computed:
            raise LiveStateCutError("StateCut manifest fingerprint mismatch")
        if value.get("schema", cls.schema) != cls.schema:
            raise LiveStateCutError("StateCut manifest schema changed")
        ref = value.get("ref")
        expected_ref = LIVE_STATECUT_REF_PREFIX + declared
        if ref not in (None, expected_ref):
            raise LiveStateCutError("StateCut manifest ref does not match its fingerprint")
        slots = value.get("slots")
        if not isinstance(slots, Mapping) or not slots:
            raise LiveStateCutError("StateCut manifest must contain slots")
        return cls(
            cut_id=_safe_cut_id(value.get("cut_id")),
            family=_text(value.get("family"), "family"),
            state_schema_id=_text(value.get("state_schema_id"), "state_schema_id"),
            model_identity=_mapping(
                value.get("model_identity"), "model_identity", allow_empty=False
            ),
            boundary=_mapping(value.get("boundary"), "boundary", allow_empty=False),
            slots={name: _mapping(record, f"slots.{name}") for name, record in slots.items()},
            metadata=_mapping(value.get("metadata") or {}, "metadata"),
            parent_cut=value.get("parent_cut"),
            fingerprint=declared,
            ref=expected_ref,
            authority_boundary=_mapping(
                value.get("authority_boundary") or {}, "authority_boundary", allow_empty=False
            ),
        )


@dataclass(frozen=True, slots=True)
class ReopenedStateCut:
    """A fully hydrated local cut and per-slot physical receipts."""

    cut: DurableStateCut
    slots: Mapping[str, Any]
    payload_receipts: Mapping[str, PayloadCacheReceipt]
    payload_bytes: int

    @property
    def state(self) -> Mapping[str, Any]:
        return self.slots

    @property
    def fingerprint(self) -> str:
        return self.cut.fingerprint

    @property
    def ref(self) -> str:
        return self.cut.ref


@dataclass(frozen=True, slots=True)
class LiveWatchPredicate:
    """Typed registration input lowered to the controller's native WatchSpec."""

    path: str
    predicate: str | None = None
    mode: str = "change"
    watch_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_controller_spec(self) -> WatchSpec:
        return WatchSpec(
            path=_text(self.path, "watch path"),
            predicate=self.predicate,
            mode=self.mode,
            watch_id=self.watch_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_controller_spec().to_dict()


@dataclass(frozen=True, slots=True)
class LiveStateCutEvent:
    """One typed watch fire joined to a durable local cut."""

    watch_id: str
    path: str
    predicate: str | None
    mode: str
    boundary: str
    index: int
    controller_cut_id: str
    statecut_ref: str
    statecut_fingerprint: str
    observed: Any
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    schema: ClassVar[str] = LIVE_STATECUT_EVENT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "event_id": self.event_id,
            "watch_id": self.watch_id,
            "path": self.path,
            "predicate": self.predicate,
            "mode": self.mode,
            "boundary": self.boundary,
            "index": self.index,
            "controller_cut_id": self.controller_cut_id,
            "statecut_ref": self.statecut_ref,
            "statecut_fingerprint": self.statecut_fingerprint,
            "observed": self.observed,
        }


class LiveStateCutStore:
    """Local state-cut manifest store backed by immutable physical payloads."""

    def __init__(self, root: str | Path, *, payload_store: PayloadStore | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.cuts = self.root / "cuts"
        self.cuts.mkdir(parents=True, exist_ok=True)
        self.payload_store = payload_store or PayloadStore(self.root / "payloads")
        self._write_store_manifest()

    def _write_store_manifest(self) -> None:
        body = {
            "schema": "saturn-live-statecut-store-v1",
            "storage": "content-addressed-cut-manifest-plus-physical-payload-cas",
            "controller_authority": "DebugWorkerController.safe_point",
            "payload_authority": PAYLOAD_AUTHORITY_CACHE_ONLY,
        }
        payload = {**body, "content_sha256": _sha256(_canonical(body))}
        target = self.root / "manifest.json"
        if target.exists():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LiveStateCutError(f"cannot read StateCut store manifest: {target}") from exc
            if existing != payload:
                raise LiveStateCutError("StateCut store manifest changed")
            return
        self._atomic_publish(target, _canonical(payload) + b"\n")

    @staticmethod
    def _atomic_publish(path: Path, payload: bytes) -> None:
        from .bench_publication import stage_write
        if stage_write(path, payload): return
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise LiveStateCutError(
                        f"immutable StateCut path contains different bytes: {path}"
                    ) from None
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @staticmethod
    def _ref_digest(value: DurableStateCut | str | Path) -> str:
        if isinstance(value, DurableStateCut):
            return value.fingerprint
        text = str(value)
        if text.startswith(LIVE_STATECUT_REF_PREFIX):
            return _digest(text[len(LIVE_STATECUT_REF_PREFIX) :], "StateCut reference")
        return _digest(text, "StateCut fingerprint")

    def manifest_path(self, value: DurableStateCut | str | Path) -> Path:
        return self.cuts / f"{self._ref_digest(value)}.json"

    def _read_manifest(self, value: DurableStateCut | str | Path) -> DurableStateCut:
        path = self.manifest_path(value)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise LiveStateCutError(f"StateCut is not present: {path.stem}") from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LiveStateCutError(f"cannot read StateCut manifest: {path}") from exc
        result = DurableStateCut.from_dict(payload)
        if result.fingerprint != path.stem:
            raise LiveStateCutError("StateCut manifest filename does not match its fingerprint")
        return result

    def _publish(self, cut: DurableStateCut) -> DurableStateCut:
        payload = _canonical(cut.to_dict()) + b"\n"
        self._atomic_publish(self.manifest_path(cut), payload)
        return cut

    def capture(
        self,
        *,
        cut_id: str,
        family: str,
        state_schema_id: str,
        model_identity: Mapping[str, Any],
        boundary: str | Mapping[str, Any],
        index: int,
        total: int,
        state: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
        parent_cut: str | None = None,
        native_consumer: str | None = None,
    ) -> DurableStateCut:
        """Capture actual live values, then publish the manifest last."""

        if not isinstance(state, Mapping) or not state:
            raise LiveStateCutError("StateCut capture requires a non-empty state mapping")
        resolved_metadata = _mapping(metadata or {}, "metadata")
        resolved_boundary = _boundary(
            boundary, index=index, total=total, metadata=resolved_metadata
        )
        slot_records: dict[str, dict[str, Any]] = {}
        for raw_name, value in state.items():
            name = _text(raw_name, "state slot name")
            if name in slot_records:
                raise LiveStateCutError(f"duplicate state slot {name!r}")
            kind = _state_value_kind(value)
            try:
                if kind == "bytes":
                    payload_ref = self.payload_store.put_bytes(
                        value, role=f"state:{name}", metadata={"slot": name, "kind": kind}
                    )
                elif kind == "array":
                    payload_ref = self.payload_store.put_array(
                        value, role=f"state:{name}", metadata={"slot": name, "kind": kind}
                    )
                else:
                    raw = _canonical(value)
                    payload_ref = self.payload_store.put_bytes(
                        raw,
                        dtype="json",
                        role=f"state:{name}",
                        metadata={"slot": name, "kind": kind},
                    )
            except (PayloadCacheError, TypeError, ValueError) as exc:
                raise LiveStateCutError(f"cannot capture state slot {name!r}: {exc}") from exc
            slot_records[name] = {
                "kind": kind,
                "payload": payload_ref.to_dict(),
                "captured_bytes": payload_ref.bytes,
            }
        body = {
            "schema": LIVE_STATECUT_SCHEMA,
            "cut_id": _safe_cut_id(cut_id),
            "family": _text(family, "family"),
            "state_schema_id": _text(state_schema_id, "state_schema_id"),
            "model_identity": _mapping(model_identity, "model_identity", allow_empty=False),
            "boundary": resolved_boundary,
            "slots": slot_records,
            "metadata": resolved_metadata,
            "parent_cut": parent_cut,
            "authority_boundary": {
                "controller": "DebugWorkerController.safe_point",
                "adapter_checkpoint": True,
                "payload_store": PAYLOAD_AUTHORITY_CACHE_ONLY,
                "native_consumer": native_consumer,
                "native_consumer_authority": "external" if native_consumer else "unbound",
                "semantic_equivalence": "not_assessed",
            },
        }
        fingerprint = _sha256(_canonical(body))
        cut = DurableStateCut(
            cut_id=body["cut_id"],
            family=body["family"],
            state_schema_id=body["state_schema_id"],
            model_identity=body["model_identity"],
            boundary=body["boundary"],
            slots=body["slots"],
            metadata=body["metadata"],
            parent_cut=body["parent_cut"],
            fingerprint=fingerprint,
            ref=LIVE_STATECUT_REF_PREFIX + fingerprint,
            authority_boundary=body["authority_boundary"],
        )
        return self._publish(cut)

    put = capture
    put_statecut = capture

    def query(self, value: DurableStateCut | str | Path) -> dict[str, Any]:
        """Read and verify metadata only; no physical payload bytes are read."""

        cut = self._read_manifest(value)
        return {
            **cut.to_dict(),
            "payload_hydrated": False,
            "payload_bytes_verified": False,
            "payload_bytes_declared": sum(
                int(record.get("captured_bytes", 0)) for record in cut.slots.values()
            ),
            "query_authority": "manifest-only",
        }

    @staticmethod
    def _expected_identity(cut: DurableStateCut, expected: Mapping[str, Any] | None) -> None:
        if expected is None:
            return
        identity = _mapping(expected, "expected identity", allow_empty=False)
        for key in ("family", "state_schema_id"):
            if key in identity and identity[key] != getattr(cut, key):
                raise LiveStateCutError(f"StateCut {key} does not match the live identity")
        if "model_identity" in identity and _sha256(
            _canonical(identity["model_identity"])
        ) != _sha256(_canonical(cut.model_identity)):
            raise LiveStateCutError("StateCut model identity does not match the live identity")

    def reopen(
        self,
        value: DurableStateCut | str | Path,
        *,
        expected_identity: Mapping[str, Any] | None = None,
        native_consumer: str | None = None,
    ) -> ReopenedStateCut:
        """Hydrate every slot and re-check all physical payload descriptors."""

        cut = self._read_manifest(value)
        self._expected_identity(cut, expected_identity)
        slots: dict[str, Any] = {}
        receipts: dict[str, PayloadCacheReceipt] = {}
        total_bytes = 0
        for name, record in cut.slots.items():
            payload_value = record.get("payload")
            try:
                descriptor = PayloadRef.from_dict(payload_value)
                hydration = self.payload_store.hydrate(
                    descriptor,
                    expected_shape=descriptor.shape,
                    expected_dtype=descriptor.dtype,
                    expected_bytes=descriptor.bytes,
                    native_consumer=native_consumer,
                )
            except (PayloadCacheError, TypeError, ValueError) as exc:
                raise LiveStateCutError(f"cannot reopen StateCut slot {name!r}: {exc}") from exc
            receipts[name] = hydration.receipt
            total_bytes += len(hydration.payload)
            kind = record.get("kind")
            if kind == "json":
                try:
                    slots[name] = json.loads(hydration.payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise LiveStateCutError(f"StateCut JSON slot {name!r} is invalid") from exc
            elif kind == "bytes":
                slots[name] = hydration.payload
            elif kind == "array":
                try:
                    slots[name] = hydration.as_numpy()
                except PayloadCacheError:
                    # A non-NumPy dtype is still reopenable as verified bytes;
                    # the caller must opt into a family-native decoder.
                    slots[name] = hydration.payload
            else:
                raise LiveStateCutError(f"StateCut slot {name!r} has unknown kind {kind!r}")
        return ReopenedStateCut(
            cut=cut,
            slots=slots,
            payload_receipts=receipts,
            payload_bytes=total_bytes,
        )

    load = reopen
    get = reopen

    def verify(self, value: DurableStateCut | str | Path) -> dict[str, Any]:
        """Verify manifest and physical payloads without leaking a false green."""

        try:
            reopened = self.reopen(value)
        except (LiveStateCutError, PayloadCacheError, OSError) as exc:
            return {
                "schema": "saturn-live-statecut-verification-v1",
                "ok": False,
                "error": str(exc),
                "payload_hydrated": False,
            }
        return {
            "schema": "saturn-live-statecut-verification-v1",
            "ok": True,
            "fingerprint": reopened.fingerprint,
            "ref": reopened.ref,
            "slots": len(reopened.slots),
            "payload_hydrated": True,
            "payload_bytes": reopened.payload_bytes,
            "receipts": {
                name: receipt.to_dict()
                for name, receipt in reopened.payload_receipts.items()
            },
        }

    def verify_or_raise(self, value: DurableStateCut | str | Path) -> ReopenedStateCut:
        return self.reopen(value)


class LiveStateCutBridge:
    """Adapt existing controller safe points to :class:`LiveStateCutStore`."""

    def __init__(
        self,
        controller: Any,
        store: LiveStateCutStore,
        *,
        state_provider: Callable[[], Mapping[str, Any]],
        family: str,
        state_schema_id: str,
        model_identity: Mapping[str, Any],
        total: int = 1,
        on_cut: Callable[[LiveStateCutEvent], None] | None = None,
        native_consumer: str | None = None,
    ) -> None:
        if not callable(getattr(controller, "safe_point", None)):
            raise LiveStateCutError("controller must expose safe_point")
        if not callable(state_provider):
            raise LiveStateCutError("state_provider must be callable")
        self.controller = controller
        self.store = store
        self.state_provider = state_provider
        self.family = _text(family, "family")
        self.state_schema_id = _text(state_schema_id, "state_schema_id")
        self.model_identity = _mapping(model_identity, "model_identity", allow_empty=False)
        self.total = int(total)
        if self.total <= 0:
            raise LiveStateCutError("total must be positive")
        self.on_cut = on_cut
        capability = getattr(controller, "capability", None)
        self.native_consumer = native_consumer or getattr(capability, "consumer", None)
        self._captured_controller_cuts: dict[str, DurableStateCut] = {}
        self._processed_watch_events: set[tuple[str, str, str, int]] = set()
        self._events: list[LiveStateCutEvent] = []

    @property
    def events(self) -> tuple[LiveStateCutEvent, ...]:
        return tuple(self._events)

    def add_watch(
        self,
        path: str,
        *,
        predicate: str | None = None,
        mode: str = "change",
        watch_id: str | None = None,
    ) -> dict[str, Any]:
        spec = LiveWatchPredicate(
            path=path,
            predicate=predicate,
            mode=mode,
            watch_id=watch_id or uuid.uuid4().hex[:12],
        ).to_controller_spec()
        response = self.controller.handle(
            DebugRequest(operation="watch.add", payload=spec.to_dict())
        )
        require_ok = getattr(response, "require_ok", None)
        if callable(require_ok):
            return dict(require_ok())
        if not getattr(response, "ok", False):
            raise LiveStateCutError(str(getattr(response, "error", "watch registration failed")))
        return dict(getattr(response, "result", {}))

    register_watch = add_watch

    def safe_point(
        self, boundary: str, index: int, metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call controller.safe_point, then persist only when it reports a fire."""

        event_metadata = dict(metadata or {})
        status = dict(self.controller.safe_point(boundary, int(index), event_metadata))
        reason = status.get("pause_reason") or {}
        watches = list(reason.get("watches") or []) if isinstance(reason, Mapping) else []
        durable: list[dict[str, Any]] = []
        state = None
        for fired in watches:
            if not isinstance(fired, Mapping):
                continue
            controller_cut = (reason.get("cut") or {}) if isinstance(reason, Mapping) else {}
            controller_cut_id = str(controller_cut.get("cut_id") or "")
            watch_id = str(fired.get("watch_id") or "")
            event_key = (controller_cut_id, watch_id, str(boundary), int(index))
            if event_key in self._processed_watch_events:
                continue
            self._processed_watch_events.add(event_key)
            cut = (
                self._captured_controller_cuts.get(controller_cut_id)
                if controller_cut_id
                else None
            )
            if cut is None:
                if state is None:
                    state = self.state_provider()
                if not isinstance(state, Mapping):
                    raise LiveStateCutError("state_provider must return a mapping")
                cut_id = controller_cut_id or f"live-{uuid.uuid4().hex}"
                cut = self.store.capture(
                    cut_id=cut_id,
                    family=self.family,
                    state_schema_id=self.state_schema_id,
                    model_identity=self.model_identity,
                    boundary=boundary,
                    index=int(index),
                    total=self.total,
                    state=state,
                    metadata={
                        **event_metadata,
                        "watch": dict(fired),
                        "controller_cut": dict(controller_cut),
                    },
                    parent_cut=controller_cut.get("parent_cut"),
                    native_consumer=self.native_consumer,
                )
                if controller_cut_id:
                    self._captured_controller_cuts[controller_cut_id] = cut
            if state is None:
                state = self.state_provider()
            path = str(fired.get("path") or "")
            event = LiveStateCutEvent(
                watch_id=watch_id,
                path=path,
                predicate=fired.get("predicate"),
                mode=str(fired.get("mode") or "change"),
                boundary=str(boundary),
                index=int(index),
                controller_cut_id=controller_cut_id or cut.cut_id,
                statecut_ref=cut.ref,
                statecut_fingerprint=cut.fingerprint,
                observed=_lookup_dotted(state, path),
            )
            self._events.append(event)
            durable.append(event.to_dict())
            if self.on_cut is not None:
                self.on_cut(event)
        return {
            **status,
            "durable_cuts": durable,
            "live_statecut_events": [event.to_dict() for event in self._events],
        }

    def resume(self) -> dict[str, Any]:
        response = self.controller.handle(DebugRequest(operation="continue"))
        require_ok = getattr(response, "require_ok", None)
        if callable(require_ok):
            return dict(require_ok())
        if not getattr(response, "ok", False):
            raise LiveStateCutError(str(getattr(response, "error", "resume failed")))
        return dict(getattr(response, "result", {}))

    continue_run = resume

    def reopen(
        self, event_or_ref: LiveStateCutEvent | DurableStateCut | str | Path
    ) -> ReopenedStateCut:
        value: Any = event_or_ref
        if isinstance(event_or_ref, LiveStateCutEvent):
            value = event_or_ref.statecut_ref
        return self.store.reopen(value, expected_identity={
            "family": self.family,
            "state_schema_id": self.state_schema_id,
            "model_identity": self.model_identity,
        }, native_consumer=self.native_consumer)


# Descriptive aliases make the local seam easy to discover from either side
# of the VM/StateCut vocabulary.
DurableStateCutStore = LiveStateCutStore
LocalStateCutStore = LiveStateCutStore
LiveWatch = LiveWatchPredicate


__all__ = [
    "DurableStateCut",
    "DurableStateCutStore",
    "LIVE_STATECUT_AUTHORITY",
    "LIVE_STATECUT_EVENT_SCHEMA",
    "LIVE_STATECUT_REF_PREFIX",
    "LIVE_STATECUT_SCHEMA",
    "LiveStateCutBridge",
    "LiveStateCutError",
    "LiveStateCutEvent",
    "LiveStateCutStore",
    "LiveWatch",
    "LiveWatchPredicate",
    "LocalStateCutStore",
    "ReopenedStateCut",
]
