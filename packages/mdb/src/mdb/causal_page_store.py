"""Physical storage and exact logical reads for factored causal pages.

``causal_page_dag`` decides which leading rows may be shared.  This module
turns that plan into an immutable physical layout backed by ``PayloadStore``:
one ancestor blob, one delta blob per child, and a content-addressed mount
manifest binding every child to the exact ancestor it requires.

The read path intentionally returns two ordered byte segments.  A segmented
native consumer can read them directly; the serial-equivalent reference may
call :meth:`CausalPageRead.materialize` to rebuild the original byte stream.
The store proves byte identity and custody only.  It does not claim that a
page is semantically equivalent under a model consumer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from mdb.causal_page_dag import (
    CausalPageDAGPlan,
    CausalPageManifest,
    CausalPageRow,
    verify_causal_page_dag,
)
from mdb.vm_payload_cache import PayloadHydration, PayloadRef, PayloadStore

CAUSAL_PAGE_MOUNT_SCHEMA = "saturn-causal-page-mount-v1"
CAUSAL_PAGE_READ_SCHEMA = "saturn-causal-page-read-v1"
CAUSAL_PAGE_MOUNT_REF_PREFIX = "cas://saturn-causal-page-mount/sha256/"
CAUSAL_PAGE_SEGMENT_ROLE = "causal-page-segment"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CausalPageStoreError(ValueError):
    """A causal-page payload, mount, or read contract is invalid."""


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
        raise CausalPageStoreError("value is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical(value))


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise CausalPageStoreError(f"{label} must be a SHA-256 hex digest")
    normalized = value.casefold()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise CausalPageStoreError(f"{label} must be a SHA-256 hex digest")
    return normalized


def _payload(value: bytes | bytearray | memoryview, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise CausalPageStoreError(f"{label} must be bytes-like")
    return bytes(value)


def _page_digest(rows: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row)
    return digest.hexdigest()


def manifest_from_row_payloads(
    *,
    page_id: str,
    model_abi_sha256: str,
    rows: Sequence[bytes | bytearray | memoryview],
) -> CausalPageManifest:
    """Build the exact manifest expected by the physical page store.

    A logical row is the caller's deterministic serialization of one causal
    token row across the stored layers/tensors.  The page payload digest is
    SHA-256 over those row byte strings in causal order.
    """

    payloads = tuple(_payload(row, f"{page_id} row") for row in rows)
    if not payloads:
        raise CausalPageStoreError("a causal page must contain at least one row")
    return CausalPageManifest(
        page_id=page_id,
        model_abi_sha256=model_abi_sha256,
        payload_sha256=_page_digest(payloads),
        rows=tuple(
            CausalPageRow(payload_sha256=_sha256_bytes(row), payload_bytes=len(row))
            for row in payloads
        ),
    )


@dataclass(frozen=True, slots=True)
class CausalPageSegment:
    """One physical blob and its exact ordered logical row boundaries."""

    payload_ref: PayloadRef
    rows: tuple[CausalPageRow, ...]

    def __post_init__(self) -> None:
        ref = (
            self.payload_ref
            if isinstance(self.payload_ref, PayloadRef)
            else PayloadRef.from_dict(self.payload_ref)
        )
        rows = tuple(
            row if isinstance(row, CausalPageRow) else CausalPageRow(**row)
            for row in self.rows
        )
        if not rows:
            raise CausalPageStoreError("causal page segment must contain at least one row")
        if ref.role != CAUSAL_PAGE_SEGMENT_ROLE:
            raise CausalPageStoreError("causal page segment payload role changed")
        if ref.bytes != sum(row.payload_bytes for row in rows):
            raise CausalPageStoreError("segment byte count disagrees with its row manifest")
        object.__setattr__(self, "payload_ref", ref)
        object.__setattr__(self, "rows", rows)

    @property
    def payload_bytes(self) -> int:
        return self.payload_ref.bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload_ref": self.payload_ref.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "payload_bytes": self.payload_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CausalPageSegment:
        if not isinstance(value, Mapping):
            raise CausalPageStoreError("causal page segment must be an object")
        segment = cls(
            payload_ref=PayloadRef.from_dict(value.get("payload_ref")),
            rows=tuple(CausalPageRow(**row) for row in value.get("rows", ())),
        )
        if value.get("payload_bytes", segment.payload_bytes) != segment.payload_bytes:
            raise CausalPageStoreError("declared segment byte count changed")
        return segment


@dataclass(frozen=True, slots=True)
class CausalPageBinding:
    """One child delta bound to its exact logical source and ancestor."""

    page_id: str
    source_payload_sha256: str
    required_ancestor_sha256: str
    delta: CausalPageSegment

    def __post_init__(self) -> None:
        if not isinstance(self.page_id, str) or not self.page_id.strip():
            raise CausalPageStoreError("page_id must be non-empty")
        object.__setattr__(
            self,
            "source_payload_sha256",
            _sha256(self.source_payload_sha256, "source payload_sha256"),
        )
        object.__setattr__(
            self,
            "required_ancestor_sha256",
            _sha256(self.required_ancestor_sha256, "required ancestor_sha256"),
        )
        if not isinstance(self.delta, CausalPageSegment):
            object.__setattr__(self, "delta", CausalPageSegment.from_dict(self.delta))

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "source_payload_sha256": self.source_payload_sha256,
            "required_ancestor_sha256": self.required_ancestor_sha256,
            "delta": self.delta.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CausalPageBinding:
        if not isinstance(value, Mapping):
            raise CausalPageStoreError("causal page binding must be an object")
        return cls(
            page_id=value.get("page_id"),
            source_payload_sha256=value.get("source_payload_sha256"),
            required_ancestor_sha256=value.get("required_ancestor_sha256"),
            delta=CausalPageSegment.from_dict(value.get("delta")),
        )


@dataclass(frozen=True, slots=True)
class CausalPageMount:
    """Content-addressed physical realization of one causal page DAG plan."""

    plan_fingerprint: str
    model_abi_sha256: str
    ancestor_sha256: str
    ancestor: CausalPageSegment
    children: tuple[CausalPageBinding, ...]
    exact_backing_bytes: int
    factored_backing_bytes: int
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = CAUSAL_PAGE_MOUNT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_fingerprint",
            _sha256(self.plan_fingerprint, "plan fingerprint"),
        )
        object.__setattr__(
            self,
            "model_abi_sha256",
            _sha256(self.model_abi_sha256, "model ABI sha256"),
        )
        object.__setattr__(
            self,
            "ancestor_sha256",
            _sha256(self.ancestor_sha256, "ancestor sha256"),
        )
        if not isinstance(self.ancestor, CausalPageSegment):
            object.__setattr__(
                self, "ancestor", CausalPageSegment.from_dict(self.ancestor)
            )
        children = tuple(
            child
            if isinstance(child, CausalPageBinding)
            else CausalPageBinding.from_dict(child)
            for child in self.children
        )
        children = tuple(sorted(children, key=lambda child: child.page_id))
        if len(children) < 2:
            raise CausalPageStoreError("a causal page mount requires at least two children")
        page_ids = [child.page_id for child in children]
        if len(page_ids) != len(set(page_ids)):
            raise CausalPageStoreError("causal page mount child IDs must be unique")
        if any(
            child.required_ancestor_sha256 != self.ancestor_sha256 for child in children
        ):
            raise CausalPageStoreError("a child requires a different ancestor")
        object.__setattr__(self, "children", children)
        expected_exact = sum(
            self.ancestor.payload_bytes + child.delta.payload_bytes for child in children
        )
        expected_factored = self.ancestor.payload_bytes + sum(
            child.delta.payload_bytes for child in children
        )
        if self.exact_backing_bytes != expected_exact:
            raise CausalPageStoreError("exact backing byte accounting changed")
        if self.factored_backing_bytes != expected_factored:
            raise CausalPageStoreError("factored backing byte accounting changed")
        object.__setattr__(self, "fingerprint", _sha256_json(self._body()))

    @property
    def ref(self) -> str:
        return CAUSAL_PAGE_MOUNT_REF_PREFIX + self.fingerprint

    @property
    def saved_bytes(self) -> int:
        return self.exact_backing_bytes - self.factored_backing_bytes

    @property
    def saved_fraction(self) -> float:
        return self.saved_bytes / self.exact_backing_bytes

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_fingerprint": self.plan_fingerprint,
            "model_abi_sha256": self.model_abi_sha256,
            "ancestor_sha256": self.ancestor_sha256,
            "ancestor": self.ancestor.to_dict(),
            "children": [child.to_dict() for child in self.children],
            "exact_backing_bytes": self.exact_backing_bytes,
            "factored_backing_bytes": self.factored_backing_bytes,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._body(),
            "saved_bytes": self.saved_bytes,
            "saved_fraction": self.saved_fraction,
            "ref": self.ref,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CausalPageMount:
        if not isinstance(value, Mapping):
            raise CausalPageStoreError("causal page mount must be an object")
        if value.get("schema", cls.schema) != cls.schema:
            raise CausalPageStoreError("causal page mount schema changed")
        mount = cls(
            plan_fingerprint=value.get("plan_fingerprint"),
            model_abi_sha256=value.get("model_abi_sha256"),
            ancestor_sha256=value.get("ancestor_sha256"),
            ancestor=CausalPageSegment.from_dict(value.get("ancestor")),
            children=tuple(
                CausalPageBinding.from_dict(child)
                for child in value.get("children", ())
            ),
            exact_backing_bytes=value.get("exact_backing_bytes"),
            factored_backing_bytes=value.get("factored_backing_bytes"),
        )
        if value.get("fingerprint", mount.fingerprint) != mount.fingerprint:
            raise CausalPageStoreError("causal page mount fingerprint changed")
        if value.get("ref", mount.ref) != mount.ref:
            raise CausalPageStoreError("causal page mount reference changed")
        if value.get("saved_bytes", mount.saved_bytes) != mount.saved_bytes:
            raise CausalPageStoreError("causal page mount saved-byte accounting changed")
        return mount

    def child(self, page_id: str) -> CausalPageBinding:
        for child in self.children:
            if child.page_id == page_id:
                return child
        raise CausalPageStoreError(f"causal page mount has no child {page_id!r}")


@dataclass(frozen=True, slots=True)
class CausalPageReadReceipt:
    """Receipt for an exact two-segment logical page read."""

    mount_fingerprint: str
    page_id: str
    source_payload_sha256: str
    ancestor_ref: str
    delta_ref: str
    row_count: int
    logical_bytes: int
    ancestor_source: str
    delta_source: str
    disk_bytes_read: int
    resident_bytes_read: int
    native_consumer: str | None
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = CAUSAL_PAGE_READ_SCHEMA

    def __post_init__(self) -> None:
        _sha256(self.mount_fingerprint, "mount fingerprint")
        _sha256(self.source_payload_sha256, "source payload sha256")
        if not self.page_id:
            raise CausalPageStoreError("read receipt page_id must be non-empty")
        if self.ancestor_source not in {"disk", "resident"}:
            raise CausalPageStoreError("read receipt ancestor source changed")
        if self.delta_source not in {"disk", "resident"}:
            raise CausalPageStoreError("read receipt delta source changed")
        for name in ("row_count", "logical_bytes", "disk_bytes_read", "resident_bytes_read"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CausalPageStoreError(f"read receipt {name} must be non-negative")
        object.__setattr__(self, "fingerprint", _sha256_json(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mount_fingerprint": self.mount_fingerprint,
            "page_id": self.page_id,
            "source_payload_sha256": self.source_payload_sha256,
            "ancestor_ref": self.ancestor_ref,
            "delta_ref": self.delta_ref,
            "row_count": self.row_count,
            "logical_bytes": self.logical_bytes,
            "segment_order": ["ancestor", "delta"],
            "ancestor_source": self.ancestor_source,
            "delta_source": self.delta_source,
            "disk_bytes_read": self.disk_bytes_read,
            "resident_bytes_read": self.resident_bytes_read,
            "full_page_materialized": False,
            "source_payload_verified": True,
            "native_consumer": self.native_consumer,
            "authority": "byte-identity-and-custody-only",
            "semantic_equivalence": "not_assessed",
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}


@dataclass(frozen=True, slots=True)
class CausalPageRead:
    """Verified logical page segments in original causal row order."""

    segments: tuple[bytes, bytes]
    receipt: CausalPageReadReceipt

    def materialize(self) -> bytes:
        """Build the serial-reference contiguous byte stream explicitly."""

        return b"".join(self.segments)


def _verify_segment_payload(
    segment: CausalPageSegment,
    hydration: PayloadHydration,
    *,
    label: str,
) -> None:
    if hydration.ref.to_dict() != segment.payload_ref.to_dict():
        raise CausalPageStoreError(f"{label} payload descriptor changed")
    offset = 0
    for index, row in enumerate(segment.rows):
        stop = offset + row.payload_bytes
        payload = hydration.payload[offset:stop]
        if len(payload) != row.payload_bytes:
            raise CausalPageStoreError(f"{label} row {index} byte count changed")
        if _sha256_bytes(payload) != row.payload_sha256:
            raise CausalPageStoreError(f"{label} row {index} payload changed")
        offset = stop
    if offset != len(hydration.payload):
        raise CausalPageStoreError(f"{label} row boundaries do not close")


class CausalPageStore:
    """Immutable causal page DAG mounts over Saturn's physical payload CAS."""

    def __init__(self, root: str | Path, *, resident_capacity_bytes: int = 0) -> None:
        self.root = Path(root).expanduser().resolve()
        self.mounts = self.root / "mounts"
        self.mounts.mkdir(parents=True, exist_ok=True)
        self.payload_store = PayloadStore(
            self.root / "payloads", resident_capacity_bytes=resident_capacity_bytes
        )

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
                    raise CausalPageStoreError(
                        f"immutable mount path contains different bytes: {path}"
                    ) from None
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def mount_path(self, value: CausalPageMount | str) -> Path:
        if isinstance(value, CausalPageMount):
            digest = value.fingerprint
        else:
            text = str(value)
            if text.startswith(CAUSAL_PAGE_MOUNT_REF_PREFIX):
                text = text[len(CAUSAL_PAGE_MOUNT_REF_PREFIX) :]
            digest = _sha256(text, "causal page mount reference")
        return self.mounts / f"{digest}.json"

    def publish(
        self,
        plan: CausalPageDAGPlan,
        sources: Sequence[CausalPageManifest],
        row_payloads: Mapping[str, Sequence[bytes | bytearray | memoryview]],
    ) -> CausalPageMount:
        """Validate, factor, and persist a causal page family."""

        verify_causal_page_dag(plan, sources)
        source_by_id = {source.page_id: source for source in sources}
        if set(row_payloads) != set(source_by_id):
            raise CausalPageStoreError("row payload page identities differ from the plan")
        normalized: dict[str, tuple[bytes, ...]] = {}
        for page_id, source in source_by_id.items():
            rows = tuple(
                _payload(row, f"{page_id} row {index}")
                for index, row in enumerate(row_payloads[page_id])
            )
            if len(rows) != len(source.rows):
                raise CausalPageStoreError(f"row count changed for {page_id}")
            for index, (payload, row) in enumerate(zip(rows, source.rows, strict=True)):
                if len(payload) != row.payload_bytes:
                    raise CausalPageStoreError(
                        f"row byte count changed for {page_id} at {index}"
                    )
                if _sha256_bytes(payload) != row.payload_sha256:
                    raise CausalPageStoreError(
                        f"row payload changed for {page_id} at {index}"
                    )
            if _page_digest(rows) != source.payload_sha256:
                raise CausalPageStoreError(f"source payload changed for {page_id}")
            normalized[page_id] = rows

        shared_count = len(plan.ancestor_rows)
        canonical_id = plan.children[0].page_id
        ancestor_rows = normalized[canonical_id][:shared_count]
        for page_id, rows in normalized.items():
            if rows[:shared_count] != ancestor_rows:
                raise CausalPageStoreError(
                    f"shared ancestor bytes differ despite row identity for {page_id}"
                )
        ancestor_ref = self.payload_store.put_bytes(
            b"".join(ancestor_rows), role=CAUSAL_PAGE_SEGMENT_ROLE
        )
        ancestor = CausalPageSegment(ancestor_ref, plan.ancestor_rows)
        child_by_id = {child.page_id: child for child in plan.children}
        bindings = []
        for page_id in sorted(source_by_id):
            child = child_by_id[page_id]
            delta_payload = b"".join(normalized[page_id][shared_count:])
            delta_ref = self.payload_store.put_bytes(
                delta_payload, role=CAUSAL_PAGE_SEGMENT_ROLE
            )
            bindings.append(
                CausalPageBinding(
                    page_id=page_id,
                    source_payload_sha256=child.source_payload_sha256,
                    required_ancestor_sha256=child.required_ancestor_sha256,
                    delta=CausalPageSegment(delta_ref, child.delta_rows),
                )
            )
        mount = CausalPageMount(
            plan_fingerprint=plan.fingerprint,
            model_abi_sha256=plan.model_abi_sha256,
            ancestor_sha256=plan.ancestor_sha256,
            ancestor=ancestor,
            children=tuple(bindings),
            exact_backing_bytes=plan.exact_backing_bytes,
            factored_backing_bytes=plan.factored_backing_bytes,
        )
        self._atomic_publish(
            self.mount_path(mount), _canonical(mount.to_dict()) + b"\n"
        )
        return mount

    def load_mount(self, value: CausalPageMount | str) -> CausalPageMount:
        if isinstance(value, CausalPageMount):
            # Reparse to apply every structural/fingerprint check even for an
            # in-memory mount supplied by a caller.
            return CausalPageMount.from_dict(value.to_dict())
        path = self.mount_path(value)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CausalPageStoreError(f"causal page mount is not present: {path}") from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CausalPageStoreError(f"cannot read causal page mount: {path}") from exc
        mount = CausalPageMount.from_dict(payload)
        if path.stem != mount.fingerprint:
            raise CausalPageStoreError("causal page mount filename changed")
        return mount

    def prefetch_ancestor(
        self,
        value: CausalPageMount | str,
        *,
        pin: bool = True,
        native_consumer: str | None = None,
    ) -> PayloadHydration:
        """Hydrate the shared ancestor once and optionally pin it resident."""

        mount = self.load_mount(value)
        hydration = self.payload_store.hydrate(
            mount.ancestor.payload_ref,
            expected_bytes=mount.ancestor.payload_bytes,
            native_consumer=native_consumer,
        )
        _verify_segment_payload(mount.ancestor, hydration, label="ancestor")
        if pin and not self.payload_store.pin(mount.ancestor.payload_ref):
            raise CausalPageStoreError(
                "ancestor did not fit in the configured resident tier and cannot be pinned"
            )
        return hydration

    def read(
        self,
        value: CausalPageMount | str,
        page_id: str,
        *,
        native_consumer: str | None = None,
        pin_ancestor: bool = False,
    ) -> CausalPageRead:
        """Read one exact logical page as ``(ancestor, delta)`` segments."""

        mount = self.load_mount(value)
        child = mount.child(page_id)
        if child.required_ancestor_sha256 != mount.ancestor_sha256:
            raise CausalPageStoreError("child ancestor dependency changed")
        ancestor = self.payload_store.hydrate(
            mount.ancestor.payload_ref,
            expected_bytes=mount.ancestor.payload_bytes,
            native_consumer=native_consumer,
        )
        _verify_segment_payload(mount.ancestor, ancestor, label="ancestor")
        if pin_ancestor and not self.payload_store.pin(mount.ancestor.payload_ref):
            raise CausalPageStoreError(
                "ancestor did not fit in the configured resident tier and cannot be pinned"
            )
        delta = self.payload_store.hydrate(
            child.delta.payload_ref,
            expected_bytes=child.delta.payload_bytes,
            native_consumer=native_consumer,
        )
        _verify_segment_payload(child.delta, delta, label=f"{page_id} delta")
        if _page_digest((ancestor.payload, delta.payload)) != child.source_payload_sha256:
            raise CausalPageStoreError(f"logical source payload changed for {page_id}")
        disk_bytes = ancestor.receipt.bytes_read + delta.receipt.bytes_read
        resident_bytes = (
            ancestor.receipt.resident_bytes_read + delta.receipt.resident_bytes_read
        )
        receipt = CausalPageReadReceipt(
            mount_fingerprint=mount.fingerprint,
            page_id=page_id,
            source_payload_sha256=child.source_payload_sha256,
            ancestor_ref=mount.ancestor.payload_ref.ref,
            delta_ref=child.delta.payload_ref.ref,
            row_count=len(mount.ancestor.rows) + len(child.delta.rows),
            logical_bytes=mount.ancestor.payload_bytes + child.delta.payload_bytes,
            ancestor_source=ancestor.source,
            delta_source=delta.source,
            disk_bytes_read=disk_bytes,
            resident_bytes_read=resident_bytes,
            native_consumer=native_consumer,
        )
        return CausalPageRead(
            segments=(ancestor.payload, delta.payload),
            receipt=receipt,
        )


__all__ = [
    "CAUSAL_PAGE_MOUNT_REF_PREFIX",
    "CAUSAL_PAGE_MOUNT_SCHEMA",
    "CAUSAL_PAGE_READ_SCHEMA",
    "CAUSAL_PAGE_SEGMENT_ROLE",
    "CausalPageBinding",
    "CausalPageMount",
    "CausalPageRead",
    "CausalPageReadReceipt",
    "CausalPageSegment",
    "CausalPageStore",
    "CausalPageStoreError",
    "manifest_from_row_payloads",
]
