"""Metadata-only cache planes for the bounded Saturn VM surface.

The native runtimes already own tensor caches (for example the prompt K/V
cache and Thott's persistent KV store).  This module is the small control
plane *above* those stores.  It records which source, address, operator, or
renderer object was selected, where its bytes live, and which model/input/ABI
and parent state make that selection meaningful.  It never hydrates or copies
those bytes.

One manifest represents one plane.  Keeping one plane per manifest makes it
possible to inspect or compare source, address, operator, and renderer
records independently while still joining them by common identity and parent
lineage.  Similarity fields are descriptive measurements only; a diff always
reports ``semantic_equivalence=not_assessed``.

This is intentionally stdlib-only so it can be used by evidence tooling and
debugger clients without importing torch, diffusers, or mrun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

VM_CACHE_MANIFEST_SCHEMA = "saturn-vm-cache-manifest-v1"
VM_CACHE_POINTER_SCHEMA = "saturn-vm-cache-pointer-v1"
VM_CACHE_ENTRY_SCHEMA = "saturn-vm-cache-entry-v1"
VM_CACHE_DIFF_SCHEMA = "saturn-vm-cache-diff-v1"
VM_CACHE_LIST_SCHEMA = "saturn-vm-cache-list-v1"
VM_CACHE_STORE_SCHEMA = "saturn-vm-cache-store-v1"
VM_CACHE_BUNDLE_SCHEMA = "saturn-vm-cache-execution-bundle-v1"
VM_CACHE_BUNDLE_DIFF_SCHEMA = "saturn-vm-cache-execution-bundle-diff-v1"
VM_CACHE_BUNDLE_LIST_SCHEMA = "saturn-vm-cache-execution-bundle-list-v1"
VM_CACHE_BUNDLE_REF_PREFIX = "cas://vm-cache-bundle/sha256/"

# Public aliases keep the short names convenient for callers while the
# explicit MANIFEST name is useful in receipts and schema registries.
VM_CACHE_SCHEMA = VM_CACHE_MANIFEST_SCHEMA
VM_CACHE_PLANES = ("source", "address", "operator", "renderer")
VM_CACHE_REF_PREFIX = "cas://sha256/"
VM_CACHE_MAX_METADATA_BYTES = 256 * 1024
VM_CACHE_MAX_SEQUENCE_ITEMS = 512
VM_CACHE_MAX_MAPPING_ITEMS = 512
VM_CACHE_MAX_STRING_BYTES = 16 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:(?://)?[^\s]+$")
_PAYLOAD_KEYS = frozenset(
    {
        "payload",
        "payloads",
        "tensor",
        "tensors",
        "weights",
        "embedding",
        "embeddings",
        "latents",
        "logits",
        "raw_bytes",
        "tensor_bytes",
    }
)
_SEMANTIC_EQUIVALENCE_KEYS = frozenset(
    {"semantic_equivalence", "semantically_equivalent", "equivalent"}
)


class VMCacheError(ValueError):
    """Raised when a VM cache record is malformed or unsafe."""


class VMCachePlane(str, Enum):
    """The four intentionally separable control/data reference planes."""

    SOURCE = "source"
    ADDRESS = "address"
    OPERATOR = "operator"
    RENDERER = "renderer"

    @classmethod
    def coerce(cls, value: VMCachePlane | str) -> str:
        if isinstance(value, cls):
            return value.value
        text = _text(value, "cache plane")
        if text not in VM_CACHE_PLANES:
            raise VMCacheError(
                f"unsupported cache plane {text!r}; expected one of {VM_CACHE_PLANES}"
            )
        return text


def _text(value: Any, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise VMCacheError(f"{label} must be non-empty text")
    result = value.strip()
    if not result:
        raise VMCacheError(f"{label} must be non-empty text")
    if len(result.encode("utf-8")) > VM_CACHE_MAX_STRING_BYTES:
        raise VMCacheError(f"{label} is too large for metadata")
    return result


def _sha(value: Any, label: str) -> str:
    result = _text(value, label).lower().removeprefix("sha256:")
    if _SHA256_RE.fullmatch(result) is None:
        raise VMCacheError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _jsonable(value: Any, *, label: str = "metadata", depth: int = 0) -> Any:
    """Normalize bounded JSON metadata and reject tensor-shaped objects.

    A shape list and a byte count are useful descriptors, but an actual tensor
    must never be silently converted through ``tolist`` or ``repr``.  The
    reserved-key check catches the common ``{"payload": ...}`` escape hatch;
    object-attribute checks catch torch, numpy, and similar tensor-likes
    without importing any of them.
    """

    if depth > 12:
        raise VMCacheError(f"{label} exceeds metadata nesting limit")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value.encode("utf-8")) > VM_CACHE_MAX_STRING_BYTES:
            raise VMCacheError(f"{label} string is too large for metadata")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VMCacheError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise VMCacheError(f"{label} cannot contain raw byte payloads")
    # Do this before handling generic sequences: numpy arrays and torch
    # tensors are sequence-like in some versions, but they are never metadata.
    if any(hasattr(value, attribute) for attribute in ("detach", "numpy", "ndim", "shape")):
        raise VMCacheError(
            f"{label} contains a tensor-like payload; store an artifact_ref or source_handle"
        )
    if isinstance(value, Mapping):
        if len(value) > VM_CACHE_MAX_MAPPING_ITEMS:
            raise VMCacheError(f"{label} has too many metadata keys")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise VMCacheError(f"{label} keys must be strings")
            key_text = key.strip()
            if not key_text:
                raise VMCacheError(f"{label} contains an empty key")
            if key_text.lower() in _PAYLOAD_KEYS:
                raise VMCacheError(
                    f"{label}.{key_text} is a payload field; use an artifact_ref or source_handle"
                )
            result[key_text] = _jsonable(item, label=f"{label}.{key_text}", depth=depth + 1)
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, (list, tuple)):
        if len(value) > VM_CACHE_MAX_SEQUENCE_ITEMS:
            raise VMCacheError(f"{label} has too many metadata items")
        return [
            _jsonable(item, label=f"{label}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        if len(value) > VM_CACHE_MAX_SEQUENCE_ITEMS:
            raise VMCacheError(f"{label} has too many metadata items")
        normalized = [
            _jsonable(item, label=f"{label}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    raise VMCacheError(f"{label} is not bounded JSON metadata: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _jsonable(value, label="canonical value"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VMCacheError("value is not canonical JSON metadata") from exc


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = {"id": value}
    if not isinstance(value, Mapping) or not value:
        raise VMCacheError(f"{label} must be a non-empty object or identifier")
    normalized = _jsonable(value, label=label)
    if not isinstance(normalized, dict):  # pragma: no cover - guarded above
        raise VMCacheError(f"{label} must be an object")
    return normalized


def _optional_sha(value: Any, label: str) -> str | None:
    if value in (None, ""):
        return None
    return _sha(value, label)


def _reference_text(value: Any, label: str, *, source_handle: bool) -> str:
    result = _text(value, label)
    if "\\" in result or result.startswith("/") or result.startswith("."):
        raise VMCacheError(f"{label} must be an exact handle/reference, not a filesystem path")
    if source_handle:
        return result
    if _URI_RE.fullmatch(result) is None:
        raise VMCacheError(
            f"{label} must be a URI-like artifact reference (for example cas:// or ar://)"
        )
    if result.lower().startswith("file:"):
        raise VMCacheError(f"{label} cannot be a local file path")
    return result


def _metadata(value: Any, label: str) -> dict[str, Any]:
    if value is None or (isinstance(value, Mapping) and not value):
        normalized = {}
    else:
        normalized = _mapping(value, label)
    try:
        encoded = canonical_bytes(normalized)
    except VMCacheError:
        raise
    if len(encoded) > VM_CACHE_MAX_METADATA_BYTES:
        raise VMCacheError(f"{label} exceeds the bounded metadata budget")
    return normalized


def _safe_similarity(value: Any, label: str) -> dict[str, Any]:
    result = _metadata(value, label)
    for key in _SEMANTIC_EQUIVALENCE_KEYS:
        if key in result and result[key] not in (None, "not_assessed"):
            raise VMCacheError(
                f"{label}.{key} cannot claim semantic equivalence; use 'not_assessed'"
            )
    # Scores are observations, not admission gates.  We only validate their
    # representation here; a caller can use any explicitly named metric.
    if "score" in result and (
        isinstance(result["score"], bool)
        or not isinstance(result["score"], (int, float))
        or not math.isfinite(float(result["score"]))
    ):
        raise VMCacheError(f"{label}.score must be a finite number")
    result.setdefault("semantic_equivalence", "not_assessed")
    return result


@dataclass(frozen=True, slots=True)
class VMCachePointer:
    """One exact external identity; no payload is stored in the pointer."""

    kind: str
    ref: str
    role: str = "opaque"
    sha256: str | None = None
    bytes: int | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = VM_CACHE_POINTER_SCHEMA

    def __post_init__(self) -> None:
        kind = _text(self.kind, "pointer kind")
        if kind not in {"source_handle", "artifact_ref"}:
            raise VMCacheError("pointer kind must be source_handle or artifact_ref")
        reference = _reference_text(self.ref, "pointer ref", source_handle=kind == "source_handle")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "ref", reference)
        object.__setattr__(self, "role", _text(self.role, "pointer role"))
        object.__setattr__(self, "sha256", _optional_sha(self.sha256, "pointer sha256"))
        if self.bytes is not None:
            if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
                raise VMCacheError("pointer bytes must be a non-negative integer")
        if self.shape is not None:
            try:
                shape = tuple(self.shape)
            except TypeError as exc:
                raise VMCacheError("pointer shape must be a sequence") from exc
            if not shape or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape
            ):
                raise VMCacheError("pointer shape must contain non-negative integers")
            if len(shape) > VM_CACHE_MAX_SEQUENCE_ITEMS:
                raise VMCacheError("pointer shape has too many dimensions")
            object.__setattr__(self, "shape", shape)
        if self.dtype is not None:
            object.__setattr__(self, "dtype", _text(self.dtype, "pointer dtype"))
        object.__setattr__(self, "metadata", _metadata(self.metadata, "pointer metadata"))
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    @classmethod
    def from_source_handle(cls, source_handle: str, **kwargs: Any) -> VMCachePointer:
        return cls(kind="source_handle", ref=source_handle, **kwargs)

    @classmethod
    def from_artifact_ref(cls, artifact_ref: str, **kwargs: Any) -> VMCachePointer:
        return cls(kind="artifact_ref", ref=artifact_ref, **kwargs)

    @property
    def source_handle(self) -> str | None:
        return self.ref if self.kind == "source_handle" else None

    @property
    def artifact_ref(self) -> str | None:
        return self.ref if self.kind == "artifact_ref" else None

    def _body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema": self.schema,
            "kind": self.kind,
            "ref": self.ref,
            "role": self.role,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "shape": None if self.shape is None else list(self.shape),
            "dtype": self.dtype,
            "metadata": dict(self.metadata),
        }
        body["source_handle" if self.kind == "source_handle" else "artifact_ref"] = self.ref
        return body

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VMCachePointer:
        if not isinstance(value, Mapping):
            raise VMCacheError("cache pointer must be an object")
        kind = value.get("kind")
        ref = value.get("ref")
        if kind is None:
            if value.get("source_handle") is not None:
                kind, ref = "source_handle", value.get("source_handle")
            elif value.get("artifact_ref") is not None:
                kind, ref = "artifact_ref", value.get("artifact_ref")
        result = cls(
            kind=kind,
            ref=ref,
            role=value.get("role", "opaque"),
            sha256=value.get("sha256"),
            bytes=value.get("bytes"),
            shape=None if value.get("shape") is None else tuple(value["shape"]),
            dtype=value.get("dtype"),
            metadata=value.get("metadata", {}),
        )
        if value.get("schema", cls.schema) != cls.schema:
            raise VMCacheError("cache pointer schema changed")
        if value.get("fingerprint") not in (None, result.fingerprint):
            raise VMCacheError("cache pointer fingerprint mismatch")
        for name in ("source_handle", "artifact_ref"):
            declared = value.get(name)
            if declared is not None and declared != getattr(result, name):
                raise VMCacheError("cache pointer reference alias disagrees with ref")
        return result


@dataclass(frozen=True, slots=True)
class VMCacheEntry:
    """One metadata record in one cache plane."""

    entry_id: str
    pointers: tuple[VMCachePointer, ...] = ()
    role: str = "opaque"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    similarity: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = VM_CACHE_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_id", _text(self.entry_id, "cache entry id"))
        pointers = tuple(
            pointer if isinstance(pointer, VMCachePointer) else VMCachePointer.from_dict(pointer)
            for pointer in self.pointers
        )
        if len({pointer.fingerprint for pointer in pointers}) != len(pointers):
            raise VMCacheError(f"cache entry {self.entry_id!r} repeats a pointer")
        object.__setattr__(
            self,
            "pointers",
            tuple(sorted(pointers, key=lambda item: item.fingerprint)),
        )
        object.__setattr__(self, "role", _text(self.role, "cache entry role"))
        object.__setattr__(self, "metadata", _metadata(self.metadata, "entry metadata"))
        object.__setattr__(
            self,
            "similarity",
            _safe_similarity(self.similarity, "entry similarity"),
        )
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    @property
    def refs(self) -> tuple[VMCachePointer, ...]:
        """Alias used by callers that call external identities ``refs``."""

        return self.pointers

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "entry_id": self.entry_id,
            "pointers": [pointer.to_dict() for pointer in self.pointers],
            "role": self.role,
            "metadata": dict(self.metadata),
            "similarity": dict(self.similarity),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VMCacheEntry:
        if not isinstance(value, Mapping):
            raise VMCacheError("cache entry must be an object")
        pointers = value.get("pointers", value.get("refs", ()))
        if pointers is None:
            pointers = ()
        if isinstance(pointers, (str, bytes, bytearray)) or not isinstance(pointers, Sequence):
            raise VMCacheError("cache entry pointers must be an array")
        result = cls(
            entry_id=value.get("entry_id", value.get("id")),
            pointers=tuple(VMCachePointer.from_dict(pointer) for pointer in pointers),
            role=value.get("role", "opaque"),
            metadata=value.get("metadata", {}),
            similarity=value.get("similarity", {}),
        )
        if value.get("schema", cls.schema) != cls.schema:
            raise VMCacheError("cache entry schema changed")
        if value.get("fingerprint") not in (None, result.fingerprint):
            raise VMCacheError("cache entry fingerprint mismatch")
        return result


def _lineage(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Mapping)):
        value = (value,)
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Sequence):
        raise VMCacheError("parent_lineage must be an array of references")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            result.append(
                {"ref": _reference_text(item, f"parent_lineage[{index}]", source_handle=True)}
            )
            continue
        if not isinstance(item, Mapping):
            raise VMCacheError(f"parent_lineage[{index}] must be an object or reference")
        normalized = _metadata(item, f"parent_lineage[{index}]")
        if normalized.get("ref") is not None:
            normalized["ref"] = _reference_text(
                normalized["ref"], f"parent_lineage[{index}].ref", source_handle=True
            )
        if normalized.get("manifest_ref") is not None:
            normalized["manifest_ref"] = _reference_text(
                normalized["manifest_ref"],
                f"parent_lineage[{index}].manifest_ref",
                source_handle=False,
            )
        if not any(
            normalized.get(key)
            for key in (
                "ref",
                "manifest_ref",
                "manifest_id",
                "manifest_sha256",
                "frame_id",
                "statecut_id",
            )
        ):
            raise VMCacheError(f"parent_lineage[{index}] has no parent identity")
        for key in ("manifest_sha256", "parent_manifest_sha256", "fingerprint"):
            if normalized.get(key) is not None:
                normalized[key] = _sha(normalized[key], f"parent_lineage[{index}].{key}")
        result.append(normalized)
    # Causal lineage is ordered; duplicate adjacent or non-adjacent refs are
    # almost always an accidental replay loop and make the view ambiguous.
    identities = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in result]
    if len(identities) != len(set(identities)):
        raise VMCacheError("parent_lineage cannot repeat the same reference")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class VMCacheManifest:
    """Content-addressed metadata manifest for one VM cache plane."""

    manifest_id: str
    plane: VMCachePlane | str
    model_identity: Mapping[str, Any] | str
    input_identity: Mapping[str, Any] | str
    abi_identity: Mapping[str, Any] | str
    parent_lineage: tuple[Any, ...] = ()
    entries: tuple[VMCacheEntry, ...] = ()
    similarity: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    claim_boundary: str = (
        "Metadata and exact external references only; similarity is not semantic equivalence "
        "and native-consumer closure remains authoritative."
    )
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = VM_CACHE_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_id", _text(self.manifest_id, "manifest id"))
        object.__setattr__(self, "plane", VMCachePlane.coerce(self.plane))
        object.__setattr__(self, "model_identity", _mapping(self.model_identity, "model_identity"))
        object.__setattr__(self, "input_identity", _mapping(self.input_identity, "input_identity"))
        object.__setattr__(self, "abi_identity", _mapping(self.abi_identity, "abi_identity"))
        object.__setattr__(self, "parent_lineage", _lineage(self.parent_lineage))
        entries = tuple(
            entry if isinstance(entry, VMCacheEntry) else VMCacheEntry.from_dict(entry)
            for entry in self.entries
        )
        ids = [entry.entry_id for entry in entries]
        if len(ids) != len(set(ids)):
            raise VMCacheError("cache entry IDs must be unique within a manifest")
        object.__setattr__(
            self,
            "entries",
            tuple(sorted(entries, key=lambda item: item.entry_id)),
        )
        object.__setattr__(
            self,
            "similarity",
            _safe_similarity(self.similarity, "manifest similarity"),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata, "manifest metadata"))
        object.__setattr__(self, "claim_boundary", _text(self.claim_boundary, "claim boundary"))
        body_bytes = canonical_bytes(self._body())
        if len(body_bytes) > VM_CACHE_MAX_METADATA_BYTES:
            raise VMCacheError("manifest exceeds the bounded metadata budget")
        object.__setattr__(self, "fingerprint", hashlib.sha256(body_bytes).hexdigest())

    @property
    def cache_id(self) -> str:
        """Compatibility alias for callers that use cache rather than manifest."""

        return self.manifest_id

    @property
    def parent_fingerprint(self) -> str | None:
        for row in reversed(self.parent_lineage):
            for key in ("manifest_sha256", "parent_manifest_sha256", "fingerprint"):
                value = row.get(key)
                if value:
                    return str(value)
            for key in ("ref", "manifest_ref"):
                value = row.get(key)
                if isinstance(value, str) and value.startswith(VM_CACHE_REF_PREFIX):
                    digest = value[len(VM_CACHE_REF_PREFIX) :]
                    if _SHA256_RE.fullmatch(digest):
                        return digest
        return None

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "manifest_id": self.manifest_id,
            "plane": self.plane,
            "model_identity": dict(self.model_identity),
            "input_identity": dict(self.input_identity),
            "abi_identity": dict(self.abi_identity),
            "parent_lineage": [dict(item) for item in self.parent_lineage],
            "entries": [entry.to_dict() for entry in self.entries],
            "similarity": dict(self.similarity),
            "metadata": dict(self.metadata),
            "claim_boundary": self.claim_boundary,
        }

    def to_dict(self, *, include_alias_hash: bool = True) -> dict[str, Any]:
        payload = {
            **self._body(),
            "fingerprint": self.fingerprint,
            "content_sha256": self.fingerprint,
        }
        if not include_alias_hash:
            payload.pop("content_sha256")
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VMCacheManifest:
        if not isinstance(value, Mapping):
            raise VMCacheError("cache manifest must be an object")
        result = cls(
            manifest_id=value.get("manifest_id", value.get("cache_id")),
            plane=value.get("plane"),
            model_identity=value.get("model_identity", value.get("model")),
            input_identity=value.get("input_identity", value.get("input")),
            abi_identity=value.get("abi_identity", value.get("abi")),
            parent_lineage=tuple(value.get("parent_lineage", value.get("lineage", ())) or ()),
            entries=tuple(value.get("entries", value.get("records", ())) or ()),
            similarity=value.get("similarity", {}),
            metadata=value.get("metadata", {}),
            claim_boundary=value.get(
                "claim_boundary",
                "Metadata and exact external references only; similarity is not semantic "
                "equivalence and native-consumer closure remains authoritative.",
            ),
        )
        if value.get("schema", cls.schema) != cls.schema:
            raise VMCacheError("cache manifest schema changed")
        for key in ("fingerprint", "content_sha256"):
            declared = value.get(key)
            if declared is not None and declared != result.fingerprint:
                raise VMCacheError(f"cache manifest {key} mismatch")
        return result

    @classmethod
    def create(
        cls,
        *,
        manifest_id: str,
        plane: VMCachePlane | str,
        model_identity: Mapping[str, Any] | str,
        input_identity: Mapping[str, Any] | str,
        abi_identity: Mapping[str, Any] | str,
        parent_lineage: Sequence[Any] = (),
        entries: Sequence[VMCacheEntry | Mapping[str, Any]] = (),
        similarity: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        claim_boundary: str | None = None,
    ) -> VMCacheManifest:
        return cls(
            manifest_id=manifest_id,
            plane=plane,
            model_identity=model_identity,
            input_identity=input_identity,
            abi_identity=abi_identity,
            parent_lineage=tuple(parent_lineage),
            entries=tuple(entries),
            similarity=similarity,
            metadata=metadata,
            claim_boundary=claim_boundary
            or "Metadata and exact external references only; similarity is not semantic "
            "equivalence and native-consumer closure remains authoritative.",
        )

    def entry(self, entry_id: str) -> VMCacheEntry:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        raise VMCacheError(f"manifest has no entry {entry_id!r}")

    def summary(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "manifest_id": self.manifest_id,
            "fingerprint": self.fingerprint,
            "plane": self.plane,
            "model_identity": dict(self.model_identity),
            "input_identity": dict(self.input_identity),
            "abi_identity": dict(self.abi_identity),
            "parent_fingerprint": self.parent_fingerprint,
            "parent_depth": len(self.parent_lineage),
            "entry_count": len(self.entries),
            "pointer_count": sum(len(entry.pointers) for entry in self.entries),
            "declared_bytes": sum(
                pointer.bytes or 0 for entry in self.entries for pointer in entry.pointers
            ),
            "semantic_equivalence": "not_assessed",
        }

    def child(
        self,
        *,
        manifest_id: str,
        entries: Sequence[VMCacheEntry | Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
        similarity: Mapping[str, Any] | None = None,
    ) -> VMCacheManifest:
        """Build a new manifest whose parent is this immutable manifest."""

        lineage = (
            *self.parent_lineage,
            {
                "ref": VM_CACHE_REF_PREFIX + self.fingerprint,
                "manifest_sha256": self.fingerprint,
            },
        )
        return VMCacheManifest.create(
            manifest_id=manifest_id,
            plane=self.plane,
            model_identity=self.model_identity,
            input_identity=self.input_identity,
            abi_identity=self.abi_identity,
            parent_lineage=lineage,
            entries=entries,
            metadata=metadata,
            similarity=similarity,
        )

    def diff(self, other: VMCacheManifest | Mapping[str, Any]) -> dict[str, Any]:
        return diff_vm_cache_manifests(self, other)


def _optional_digest(value: Any, label: str) -> str | None:
    if value in (None, ""):
        return None
    return _sha(value, label)


def _string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise VMCacheError(f"{label} must be an array of identifiers")
    result = tuple(_text(item, f"{label} item") for item in value)
    if len(result) != len(set(result)):
        raise VMCacheError(f"{label} contains duplicates")
    return result


@dataclass(frozen=True, slots=True)
class VMCacheExecutionBundle:
    """One pointer-only join of the four VM cache planes for an execution.

    The bundle is deliberately a manifest of manifest references.  It does
    not contain cell outputs, tensors, or a copy of an incremental request.
    ``dirty`` and ``admitted_plan`` retain only the bounded execution choices
    needed to explain which cache records were eligible for reuse.
    """

    run_id: str
    parent_identity: Mapping[str, Any] | str
    graph_fingerprint: str
    model_identity: Mapping[str, Any] | str
    input_identity: Mapping[str, Any] | str
    abi_identity: Mapping[str, Any] | str
    plane_references: Mapping[str, str]
    lineage: Mapping[str, Any] = field(default_factory=dict)
    dirty: Mapping[str, Any] = field(default_factory=dict)
    admitted_plan: Mapping[str, Any] = field(default_factory=dict)
    output_pointer: VMCachePointer | Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    claim_boundary: str = (
        "Execution metadata and exact external pointers only; cache records do not contain "
        "tensor payloads, and native-consumer parity remains authoritative."
    )
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = VM_CACHE_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "VM cache run id"))
        object.__setattr__(
            self,
            "parent_identity",
            _mapping(self.parent_identity, "VM cache parent identity"),
        )
        object.__setattr__(
            self,
            "graph_fingerprint",
            _sha(self.graph_fingerprint, "VM cache graph fingerprint"),
        )
        object.__setattr__(self, "model_identity", _mapping(self.model_identity, "model identity"))
        object.__setattr__(self, "input_identity", _mapping(self.input_identity, "input identity"))
        object.__setattr__(self, "abi_identity", _mapping(self.abi_identity, "ABI identity"))
        if not isinstance(self.plane_references, Mapping):
            raise VMCacheError("VM cache bundle planes must be an object")
        plane_references = _jsonable(self.plane_references, label="bundle planes")
        if not isinstance(plane_references, dict) or set(plane_references) != set(VM_CACHE_PLANES):
            raise VMCacheError(
                "VM cache bundle must reference exactly source, address, operator, and renderer"
            )
        normalized_planes = {
            plane: _reference_text(
                plane_references[plane], f"bundle {plane} manifest reference", source_handle=False
            )
            for plane in VM_CACHE_PLANES
        }
        object.__setattr__(self, "plane_references", normalized_planes)
        lineage = _metadata(self.lineage, "bundle lineage")
        dirty = _metadata(self.dirty, "bundle dirty summary")
        admitted_plan = _metadata(self.admitted_plan, "bundle admitted plan")
        object.__setattr__(self, "lineage", lineage)
        object.__setattr__(self, "dirty", dirty)
        object.__setattr__(self, "admitted_plan", admitted_plan)
        output = self.output_pointer
        if output is not None and not isinstance(output, VMCachePointer):
            output = VMCachePointer.from_dict(output)
        object.__setattr__(self, "output_pointer", output)
        object.__setattr__(self, "metadata", _metadata(self.metadata, "bundle metadata"))
        object.__setattr__(
            self, "claim_boundary", _text(self.claim_boundary, "bundle claim boundary")
        )
        body = self._body()
        if len(canonical_bytes(body)) > VM_CACHE_MAX_METADATA_BYTES:
            raise VMCacheError("VM cache execution bundle exceeds the bounded metadata budget")
        object.__setattr__(self, "fingerprint", content_digest(body))

    @property
    def reference(self) -> str:
        return VM_CACHE_BUNDLE_REF_PREFIX + self.fingerprint

    @property
    def parent_identity_fingerprint(self) -> str:
        return content_digest(self.parent_identity)

    @property
    def planes(self) -> Mapping[str, str]:
        """Alias for callers that refer to the four manifest refs as planes."""

        return self.plane_references

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "parent_identity": dict(self.parent_identity),
            "graph_fingerprint": self.graph_fingerprint,
            "model_identity": dict(self.model_identity),
            "input_identity": dict(self.input_identity),
            "abi_identity": dict(self.abi_identity),
            "planes": dict(self.plane_references),
            "lineage": dict(self.lineage),
            "dirty": dict(self.dirty),
            "admitted_plan": dict(self.admitted_plan),
            "output_pointer": (
                None if self.output_pointer is None else self.output_pointer.to_dict()
            ),
            "metadata": dict(self.metadata),
            "claim_boundary": self.claim_boundary,
        }

    def to_dict(self, *, include_alias_hash: bool = True) -> dict[str, Any]:
        payload = {**self._body(), "fingerprint": self.fingerprint}
        if include_alias_hash:
            payload["content_sha256"] = self.fingerprint
        return payload

    def summary(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "reference": self.reference,
            "fingerprint": self.fingerprint,
            "run_id": self.run_id,
            "parent_identity": dict(self.parent_identity),
            "parent_identity_fingerprint": self.parent_identity_fingerprint,
            "graph_fingerprint": self.graph_fingerprint,
            "planes": dict(self.plane_references),
            "dirty": dict(self.dirty),
            "admitted_plan": dict(self.admitted_plan),
            "output_pointer": (
                None if self.output_pointer is None else self.output_pointer.to_dict()
            ),
            "semantic_equivalence": "not_assessed",
        }

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        parent_identity: Mapping[str, Any] | str,
        graph_fingerprint: str,
        model_identity: Mapping[str, Any] | str,
        input_identity: Mapping[str, Any] | str,
        abi_identity: Mapping[str, Any] | str,
        plane_references: Mapping[str, str],
        lineage: Mapping[str, Any] | None = None,
        dirty: Mapping[str, Any] | None = None,
        admitted_plan: Mapping[str, Any] | None = None,
        output_pointer: VMCachePointer | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        claim_boundary: str | None = None,
    ) -> VMCacheExecutionBundle:
        return cls(
            run_id=run_id,
            parent_identity=parent_identity,
            graph_fingerprint=graph_fingerprint,
            model_identity=model_identity,
            input_identity=input_identity,
            abi_identity=abi_identity,
            plane_references=plane_references,
            lineage=lineage or {},
            dirty=dirty or {},
            admitted_plan=admitted_plan or {},
            output_pointer=output_pointer,
            metadata=metadata or {},
            claim_boundary=claim_boundary
            or "Execution metadata and exact external pointers only; cache records do not contain "
            "tensor payloads, and native-consumer parity remains authoritative.",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VMCacheExecutionBundle:
        if not isinstance(value, Mapping):
            raise VMCacheError("VM cache execution bundle must be an object")
        result = cls(
            run_id=value.get("run_id"),
            parent_identity=value.get("parent_identity"),
            graph_fingerprint=value.get("graph_fingerprint"),
            model_identity=value.get("model_identity"),
            input_identity=value.get("input_identity"),
            abi_identity=value.get("abi_identity"),
            plane_references=value.get("planes", value.get("plane_references", {})),
            lineage=value.get("lineage", {}),
            dirty=value.get("dirty", {}),
            admitted_plan=value.get("admitted_plan", {}),
            output_pointer=value.get("output_pointer"),
            metadata=value.get("metadata", {}),
            claim_boundary=value.get(
                "claim_boundary",
                "Execution metadata and exact external pointers only; cache records do not contain "
                "tensor payloads, and native-consumer parity remains authoritative.",
            ),
        )
        if value.get("schema", cls.schema) != cls.schema:
            raise VMCacheError("VM cache execution bundle schema changed")
        for key in ("fingerprint", "content_sha256"):
            declared = value.get(key)
            if declared is not None and declared != result.fingerprint:
                raise VMCacheError(f"VM cache execution bundle {key} mismatch")
        return result

    def diff(self, other: VMCacheExecutionBundle | Mapping[str, Any]) -> dict[str, Any]:
        return diff_vm_cache_bundles(self, other)


def build_vm_cache_manifest(
    *,
    manifest_id: str,
    plane: VMCachePlane | str,
    model_identity: Mapping[str, Any] | str,
    input_identity: Mapping[str, Any] | str,
    abi_identity: Mapping[str, Any] | str,
    parent_lineage: Sequence[Any] = (),
    entries: Sequence[VMCacheEntry | Mapping[str, Any]] = (),
    similarity: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    claim_boundary: str | None = None,
) -> VMCacheManifest:
    """Construct one of the four plane-specific manifests."""

    return VMCacheManifest.create(
        manifest_id=manifest_id,
        plane=plane,
        model_identity=model_identity,
        input_identity=input_identity,
        abi_identity=abi_identity,
        parent_lineage=parent_lineage,
        entries=entries,
        similarity=similarity,
        metadata=metadata,
        claim_boundary=claim_boundary,
    )


def build_source_cache_manifest(**kwargs: Any) -> VMCacheManifest:
    return build_vm_cache_manifest(plane=VMCachePlane.SOURCE, **kwargs)


def build_address_cache_manifest(**kwargs: Any) -> VMCacheManifest:
    return build_vm_cache_manifest(plane=VMCachePlane.ADDRESS, **kwargs)


def build_operator_cache_manifest(**kwargs: Any) -> VMCacheManifest:
    return build_vm_cache_manifest(plane=VMCachePlane.OPERATOR, **kwargs)


def build_renderer_cache_manifest(**kwargs: Any) -> VMCacheManifest:
    return build_vm_cache_manifest(plane=VMCachePlane.RENDERER, **kwargs)


def _load_manifest(value: VMCacheManifest | Mapping[str, Any] | str | Path) -> VMCacheManifest:
    if isinstance(value, VMCacheManifest):
        return value
    if isinstance(value, Mapping):
        return VMCacheManifest.from_dict(value)
    path = Path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VMCacheError(f"cannot read VM cache manifest: {path}") from exc
    return VMCacheManifest.from_dict(payload)


def _entry_diff(left: VMCacheEntry, right: VMCacheEntry) -> dict[str, Any]:
    left_pointers = {pointer.fingerprint: pointer for pointer in left.pointers}
    right_pointers = {pointer.fingerprint: pointer for pointer in right.pointers}
    left_metadata = dict(left.metadata)
    right_metadata = dict(right.metadata)
    return {
        "entry_id": left.entry_id,
        "left_fingerprint": left.fingerprint,
        "right_fingerprint": right.fingerprint,
        "role_changed": left.role != right.role,
        "pointer_added": sorted(set(right_pointers) - set(left_pointers)),
        "pointer_removed": sorted(set(left_pointers) - set(right_pointers)),
        "metadata_added": sorted(set(right_metadata) - set(left_metadata)),
        "metadata_removed": sorted(set(left_metadata) - set(right_metadata)),
        "metadata_changed": sorted(
            key
            for key in set(left_metadata) & set(right_metadata)
            if left_metadata[key] != right_metadata[key]
        ),
        "similarity_changed": dict(left.similarity) != dict(right.similarity),
    }


def diff_vm_cache_manifests(
    left: VMCacheManifest | Mapping[str, Any] | str | Path,
    right: VMCacheManifest | Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Return a bounded structural diff without semantic interpretation."""

    lhs = _load_manifest(left)
    rhs = _load_manifest(right)
    left_entries = {entry.entry_id: entry for entry in lhs.entries}
    right_entries = {entry.entry_id: entry for entry in rhs.entries}
    shared = sorted(set(left_entries) & set(right_entries))
    unchanged = [
        entry_id
        for entry_id in shared
        if left_entries[entry_id].fingerprint == right_entries[entry_id].fingerprint
    ]
    changed = [
        _entry_diff(left_entries[entry_id], right_entries[entry_id])
        for entry_id in shared
        if entry_id not in unchanged
    ]
    union_count = len(set(left_entries) | set(right_entries))
    left_pointer_fingerprints = {
        (entry_id, pointer.fingerprint)
        for entry_id in shared
        for pointer in left_entries[entry_id].pointers
    }
    right_pointer_fingerprints = {
        (entry_id, pointer.fingerprint)
        for entry_id in shared
        for pointer in right_entries[entry_id].pointers
    }
    pointer_union = left_pointer_fingerprints | right_pointer_fingerprints
    identity_changes = {
        name: {"left": getattr(lhs, name), "right": getattr(rhs, name)}
        for name in ("plane", "model_identity", "input_identity", "abi_identity", "parent_lineage")
        if getattr(lhs, name) != getattr(rhs, name)
    }
    return {
        "schema": VM_CACHE_DIFF_SCHEMA,
        "left": lhs.summary(),
        "right": rhs.summary(),
        "same_manifest": lhs.fingerprint == rhs.fingerprint,
        "identity_changes": identity_changes,
        "entries": {
            "added": sorted(set(right_entries) - set(left_entries)),
            "removed": sorted(set(left_entries) - set(right_entries)),
            "unchanged": unchanged,
            "changed": changed,
        },
        "structural_similarity": {
            "entry_id_jaccard": (len(shared) / union_count) if union_count else 1.0,
            "entry_fingerprint_fraction": (len(unchanged) / len(shared)) if shared else 1.0,
            "shared_pointer_jaccard": (
                len(left_pointer_fingerprints & right_pointer_fingerprints) / len(pointer_union)
                if pointer_union
                else 1.0
            ),
            "scope": "manifest metadata and pointer identities",
        },
        "similarity_metadata": {
            "left": dict(lhs.similarity),
            "right": dict(rhs.similarity),
        },
        "semantic_equivalence": "not_assessed",
        "claim_boundary": (
            "Structural metadata and exact pointer identity only; this diff does not establish "
            "semantic equivalence, consumer parity, or payload interchangeability."
        ),
    }


def _load_bundle(
    value: VMCacheExecutionBundle | Mapping[str, Any] | str | Path,
    *,
    store: VMCacheStore | None = None,
) -> VMCacheExecutionBundle:
    if isinstance(value, VMCacheExecutionBundle):
        return value
    if isinstance(value, Mapping):
        return VMCacheExecutionBundle.from_dict(value)
    if store is not None and str(value).startswith(VM_CACHE_BUNDLE_REF_PREFIX):
        return store.get_bundle(value)
    path = Path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VMCacheError(f"cannot read VM cache execution bundle: {path}") from exc
    return VMCacheExecutionBundle.from_dict(payload)


def diff_vm_cache_bundles(
    left: VMCacheExecutionBundle | Mapping[str, Any] | str | Path,
    right: VMCacheExecutionBundle | Mapping[str, Any] | str | Path,
    *,
    store: VMCacheStore | None = None,
) -> dict[str, Any]:
    """Diff one execution bundle while retaining an independent diff per plane.

    Without a store the result still exposes all four exact manifest
    references.  A store is needed only to descend into the referenced
    manifests; absence of that hydration is reported as a reference-only
    comparison rather than a semantic conclusion.
    """

    lhs = _load_bundle(left, store=store)
    rhs = _load_bundle(right, store=store)
    identity_changes = {
        name: {"left": getattr(lhs, name), "right": getattr(rhs, name)}
        for name in (
            "run_id",
            "parent_identity",
            "graph_fingerprint",
            "model_identity",
            "input_identity",
            "abi_identity",
            "lineage",
            "dirty",
            "admitted_plan",
        )
        if getattr(lhs, name) != getattr(rhs, name)
    }
    plane_diffs: dict[str, Any] = {}
    for plane in VM_CACHE_PLANES:
        left_ref = lhs.plane_references[plane]
        right_ref = rhs.plane_references[plane]
        if store is None:
            plane_diffs[plane] = {
                "plane": plane,
                "left_reference": left_ref,
                "right_reference": right_ref,
                "same_manifest": left_ref == right_ref,
                "status": "references-only",
                "semantic_equivalence": "not_assessed",
            }
        else:
            plane_diffs[plane] = diff_vm_cache_manifests(
                store.get(left_ref), store.get(right_ref)
            )
    return {
        "schema": VM_CACHE_BUNDLE_DIFF_SCHEMA,
        "left": lhs.summary(),
        "right": rhs.summary(),
        "same_bundle": lhs.fingerprint == rhs.fingerprint,
        "identity_changes": identity_changes,
        "planes": plane_diffs,
        "semantic_equivalence": "not_assessed",
        "claim_boundary": (
            "Execution lineage, dirty/admission metadata, and independent plane manifest diffs "
            "only; this comparison does not establish semantic equivalence or payload "
            "interchangeability."
        ),
    }


class VMCacheStore:
    """Small directory-backed CAS containing manifests, never tensor bytes."""

    schema: ClassVar[str] = VM_CACHE_STORE_SCHEMA

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects"
        self.bundles = self.root / "bundles"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.bundles.mkdir(parents=True, exist_ok=True)
        self._write_store_manifest()

    def _write_store_manifest(self) -> None:
        path = self.root / "manifest.json"
        body = {"schema": self.schema, "storage": "metadata-only-manifest-cas-v1"}
        payload = {**body, "content_sha256": content_digest(body)}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise VMCacheError(f"cannot read VM cache store manifest: {path}") from exc
            if existing != payload:
                raise VMCacheError("VM cache store manifest changed")
            return
        _atomic_publish(path, canonical_bytes(payload) + b"\n")

    @staticmethod
    def expected_ref(fingerprint: str) -> str:
        return VM_CACHE_REF_PREFIX + _sha(fingerprint, "manifest fingerprint")

    @staticmethod
    def _digest_from_ref(value: str | Path) -> str | None:
        if isinstance(value, Path):
            return None
        text = str(value)
        if text.startswith(VM_CACHE_REF_PREFIX):
            return _sha(text[len(VM_CACHE_REF_PREFIX) :], "manifest reference")
        if _SHA256_RE.fullmatch(text):
            return text
        return None

    def _path(self, manifest: VMCacheManifest) -> Path:
        return self.objects / manifest.plane / f"{manifest.fingerprint}.json"

    def put(self, manifest: VMCacheManifest | Mapping[str, Any]) -> str:
        value = _load_manifest(manifest)
        target = self._path(value)
        payload = canonical_bytes(value.to_dict()) + b"\n"
        if len(payload) > VM_CACHE_MAX_METADATA_BYTES:
            raise VMCacheError("manifest serialization exceeds metadata budget")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != payload:
                raise VMCacheError(f"content collision for {value.fingerprint}")
        else:
            _atomic_publish(target, payload)
        return self.expected_ref(value.fingerprint)

    publish = put
    put_manifest = put

    @staticmethod
    def expected_bundle_ref(fingerprint: str) -> str:
        return VM_CACHE_BUNDLE_REF_PREFIX + _sha(fingerprint, "bundle fingerprint")

    def _bundle_path(self, bundle: VMCacheExecutionBundle) -> Path:
        return self.bundles / f"{bundle.fingerprint}.json"

    def put_bundle(
        self, bundle: VMCacheExecutionBundle | Mapping[str, Any]
    ) -> str:
        """Publish one immutable bundle manifest; referenced bytes are untouched."""

        value = _load_bundle(bundle)
        target = self._bundle_path(value)
        payload = canonical_bytes(value.to_dict()) + b"\n"
        if len(payload) > VM_CACHE_MAX_METADATA_BYTES:
            raise VMCacheError("execution bundle serialization exceeds metadata budget")
        if target.exists():
            if target.read_bytes() != payload:
                raise VMCacheError(f"bundle content collision for {value.fingerprint}")
        else:
            _atomic_publish(target, payload)
        return self.expected_bundle_ref(value.fingerprint)

    publish_bundle = put_bundle

    def _find_bundle_path(self, value: str | Path) -> Path:
        if isinstance(value, Path) or (isinstance(value, str) and Path(value).is_file()):
            path = Path(value)
            if path.suffix.lower() != ".json":
                raise VMCacheError("bundle path must be a JSON file")
            return path
        text = str(value)
        if text.startswith(VM_CACHE_BUNDLE_REF_PREFIX):
            digest = _sha(text[len(VM_CACHE_BUNDLE_REF_PREFIX) :], "bundle reference")
        elif _SHA256_RE.fullmatch(text):
            digest = text
        else:
            raise VMCacheError("bundle lookup requires a bundle reference, digest, or JSON path")
        target = self.bundles / f"{digest}.json"
        if not target.exists():
            raise VMCacheError(f"execution bundle is not present in VM cache store: {digest}")
        return target

    def get_bundle(self, value: str | Path) -> VMCacheExecutionBundle:
        path = self._find_bundle_path(value)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VMCacheError(f"cannot read VM cache execution bundle: {path}") from exc
        result = VMCacheExecutionBundle.from_dict(payload)
        if path.parent == self.bundles and result.fingerprint != path.stem:
            raise VMCacheError("stored execution bundle filename does not match its content hash")
        return result

    load_bundle = get_bundle

    def list_bundles(self) -> tuple[dict[str, Any], ...]:
        summaries: list[dict[str, Any]] = []
        for path in sorted(self.bundles.glob("*.json")):
            summaries.append(self.get_bundle(path).summary())
        return tuple(summaries)

    def show_bundle(self, value: str | Path) -> dict[str, Any]:
        return self.get_bundle(value).to_dict()

    def diff_bundles(self, left: str | Path, right: str | Path) -> dict[str, Any]:
        return diff_vm_cache_bundles(
            self.get_bundle(left), self.get_bundle(right), store=self
        )

    def _find_path(self, value: str | Path) -> Path:
        if isinstance(value, Path) or (isinstance(value, str) and Path(value).is_file()):
            path = Path(value)
            if path.suffix.lower() != ".json":
                raise VMCacheError("manifest path must be a JSON file")
            return path
        digest = self._digest_from_ref(value)
        if digest is None:
            raise VMCacheError("manifest lookup requires a CAS reference, digest, or JSON path")
        candidates = list(self.objects.glob(f"*/{digest}.json"))
        if len(candidates) != 1:
            if not candidates:
                raise VMCacheError(f"manifest is not present in VM cache store: {digest}")
            raise VMCacheError(f"manifest digest is ambiguous across cache planes: {digest}")
        return candidates[0]

    def get(self, value: str | Path) -> VMCacheManifest:
        path = self._find_path(value)
        result = _load_manifest(path)
        if path.is_relative_to(self.objects):
            expected = path.stem
            if result.fingerprint != expected:
                raise VMCacheError("stored manifest filename does not match its content hash")
        return result

    load = get

    def list(self, plane: VMCachePlane | str | None = None) -> tuple[dict[str, Any], ...]:
        selected = None if plane is None else VMCachePlane.coerce(plane)
        summaries: list[dict[str, Any]] = []
        for path in sorted(self.objects.glob("*/*.json")):
            if selected is not None and path.parent.name != selected:
                continue
            summaries.append(self.get(path).summary())
        return tuple(summaries)

    list_manifests = list

    def show(self, value: str | Path) -> dict[str, Any]:
        if self._looks_like_bundle(value):
            return self.show_bundle(value)
        return self.get(value).to_dict()

    def diff(self, left: str | Path, right: str | Path) -> dict[str, Any]:
        left_bundle = self._looks_like_bundle(left)
        right_bundle = self._looks_like_bundle(right)
        if left_bundle or right_bundle:
            if not left_bundle or not right_bundle:
                raise VMCacheError("bundle diff requires two execution bundles")
            return self.diff_bundles(left, right)
        return diff_vm_cache_manifests(self.get(left), self.get(right))

    @staticmethod
    def _looks_like_bundle(value: str | Path) -> bool:
        if isinstance(value, Path):
            return value.parent.name == "bundles"
        text = str(value)
        return text.startswith(VM_CACHE_BUNDLE_REF_PREFIX)


class VMCacheEmitter:
    """Publish four metadata-only manifests and their execution bundle.

    ``entries`` is intentionally the only per-plane data input.  The emitter
    never accepts an output value or a tensor tree, and never calls a store's
    ``get`` method.  Callers provide external pointers in ``VMCacheEntry``
    records when a native runtime has durable bytes worth naming.
    """

    def __init__(self, store: VMCacheStore | str | Path) -> None:
        self.store = store if isinstance(store, VMCacheStore) else VMCacheStore(store)

    @staticmethod
    def _parent_lineage(
        *,
        run_id: str,
        parent_identity: Mapping[str, Any] | str,
        parent_lineage: Sequence[Any],
    ) -> tuple[Any, ...]:
        identity = _mapping(parent_identity, "VM cache parent identity")
        parent_fingerprint = content_digest(identity)
        parent_ref = f"ar://saturn/vm-cache-parent/{parent_fingerprint}"
        return (
            *tuple(parent_lineage),
            {
                "ref": parent_ref,
                "run_id": _text(run_id, "VM cache run id"),
                "parent_fingerprint": parent_fingerprint,
            },
        )

    @staticmethod
    def _plan_summary(plan: Mapping[str, Any] | None) -> dict[str, Any]:
        if plan is None:
            return {
                "status": "full-build",
                "pointer": None,
                "fingerprint": None,
                "execution_authorized": True,
                "authorizing_receipts": [],
            }
        if not isinstance(plan, Mapping):
            raise VMCacheError("execution plan metadata must be an object")
        fingerprint = _optional_digest(plan.get("fingerprint"), "plan fingerprint")
        pointer = plan.get("pointer")
        if pointer is None and fingerprint is not None:
            pointer = f"ar://saturn/circuit-plan/{fingerprint}"
        if pointer is not None:
            pointer = _reference_text(pointer, "plan pointer", source_handle=False)
        receipts = tuple(
            _sha(item, "authorizing plan receipt")
            for item in (plan.get("authorizing_receipts") or ())
        )
        result = {
            "status": (
                "admitted"
                if plan.get("execution_authorized") is True
                else "discovery-only"
            ),
            "pointer": pointer,
            "fingerprint": fingerprint,
            "execution_authorized": plan.get("execution_authorized") is True,
            "authorizing_receipts": list(receipts),
        }
        if plan.get("reason") is not None:
            result["reason"] = _text(plan["reason"], "plan reason")
        return result

    def emit(
        self,
        *,
        run_id: str,
        parent_identity: Mapping[str, Any] | str,
        graph_fingerprint: str,
        model_identity: Mapping[str, Any] | str,
        input_identity: Mapping[str, Any] | str,
        abi_identity: Mapping[str, Any] | str,
        entries: Mapping[str, Sequence[VMCacheEntry | Mapping[str, Any]]],
        parent_lineage: Sequence[Any] = (),
        dirty: Mapping[str, Any] | None = None,
        admitted_plan: Mapping[str, Any] | None = None,
        output_pointer: VMCachePointer | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        manifest_id_prefix: str | None = None,
    ) -> VMCacheExecutionBundle:
        """Emit all four planes and return the immutable bundle descriptor."""

        resolved_run_id = _text(run_id, "VM cache run id")
        resolved_parent = _mapping(parent_identity, "VM cache parent identity")
        resolved_graph = _sha(graph_fingerprint, "VM cache graph fingerprint")
        if not isinstance(entries, Mapping) or set(entries) != set(VM_CACHE_PLANES):
            raise VMCacheError(
                "VM cache emitter requires entries for source, address, operator, and renderer"
            )
        lineage_rows = self._parent_lineage(
            run_id=resolved_run_id,
            parent_identity=resolved_parent,
            parent_lineage=parent_lineage,
        )
        parent_fingerprint = content_digest(resolved_parent)
        prefix = (
            _text(manifest_id_prefix, "manifest id prefix")
            if manifest_id_prefix
            else resolved_run_id
        )
        references: dict[str, str] = {}
        for plane in VM_CACHE_PLANES:
            raw_entries = entries[plane]
            if isinstance(raw_entries, (str, bytes, bytearray)) or not isinstance(
                raw_entries, Sequence
            ):
                raise VMCacheError(f"{plane} entries must be an array")
            normalized_entries = tuple(
                item if isinstance(item, VMCacheEntry) else VMCacheEntry.from_dict(item)
                for item in raw_entries
            )
            manifest = build_vm_cache_manifest(
                manifest_id=f"{prefix}:{plane}",
                plane=plane,
                model_identity=model_identity,
                input_identity=input_identity,
                abi_identity=abi_identity,
                parent_lineage=lineage_rows,
                entries=normalized_entries,
                metadata={
                    "execution": {
                        "run_id": resolved_run_id,
                        "parent_identity_fingerprint": parent_fingerprint,
                        "graph_fingerprint": resolved_graph,
                    },
                    "plane": plane,
                },
            )
            references[plane] = self.store.put(manifest)
        bundle = VMCacheExecutionBundle.create(
            run_id=resolved_run_id,
            parent_identity=resolved_parent,
            graph_fingerprint=resolved_graph,
            model_identity=model_identity,
            input_identity=input_identity,
            abi_identity=abi_identity,
            plane_references=references,
            lineage={
                "run_id": resolved_run_id,
                "parent_identity_fingerprint": parent_fingerprint,
                "manifest_fingerprints": {
                    plane: references[plane][len(VM_CACHE_REF_PREFIX) :]
                    for plane in VM_CACHE_PLANES
                },
            },
            dirty=dirty,
            admitted_plan=self._plan_summary(admitted_plan),
            output_pointer=output_pointer,
            metadata=metadata,
        )
        self.store.put_bundle(bundle)
        return bundle

    emit_execution = emit


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
                raise VMCacheError(
                    f"immutable content path contains different bytes: {path}"
                ) from None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_vm_cache_manifest(
    manifest: VMCacheManifest | Mapping[str, Any], path: str | Path
) -> Path:
    """Write one metadata manifest for handoff; no external payload is copied."""

    value = _load_manifest(manifest)
    destination = Path(path).expanduser().resolve()
    payload = canonical_bytes(value.to_dict()) + b"\n"
    if len(payload) > VM_CACHE_MAX_METADATA_BYTES:
        raise VMCacheError("manifest serialization exceeds metadata budget")
    _atomic_publish(destination, payload)
    return destination


def list_vm_cache_manifests(
    root: str | Path, *, plane: VMCachePlane | str | None = None
) -> tuple[dict[str, Any], ...]:
    return VMCacheStore(root).list(plane)


def show_vm_cache_manifest(root: str | Path, reference: str | Path) -> dict[str, Any]:
    return VMCacheStore(root).show(reference)


def diff_vm_cache_store(root: str | Path, left: str | Path, right: str | Path) -> dict[str, Any]:
    return VMCacheStore(root).diff(left, right)


def list_vm_cache_bundles(root: str | Path) -> tuple[dict[str, Any], ...]:
    return VMCacheStore(root).list_bundles()


def show_vm_cache_bundle(root: str | Path, reference: str | Path) -> dict[str, Any]:
    return VMCacheStore(root).show_bundle(reference)


def diff_vm_cache_bundle_store(
    root: str | Path, left: str | Path, right: str | Path
) -> dict[str, Any]:
    return VMCacheStore(root).diff_bundles(left, right)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saturn-vm-cache",
        description="List, show, validate, and diff metadata-only Saturn VM cache manifests",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list lightweight manifest summaries")
    listing.add_argument("root", type=Path)
    listing.add_argument("--plane", choices=VM_CACHE_PLANES)
    listing.add_argument("--bundles", action="store_true", help="list execution bundle summaries")
    bundle_list = sub.add_parser("bundles", help="list execution bundle summaries")
    bundle_list.add_argument("root", type=Path)
    show = sub.add_parser("show", help="show one metadata-only manifest")
    show.add_argument("root", type=Path)
    show.add_argument("reference")
    bundle_show = sub.add_parser("bundle-show", help="show one metadata-only execution bundle")
    bundle_show.add_argument("root", type=Path)
    bundle_show.add_argument("reference")
    diff = sub.add_parser("diff", help="diff two manifests by metadata and pointer identity")
    diff.add_argument("root", type=Path)
    diff.add_argument("left")
    diff.add_argument("right")
    bundle_diff = sub.add_parser("bundle-diff", help="diff two execution bundles plane by plane")
    bundle_diff.add_argument("root", type=Path)
    bundle_diff.add_argument("left")
    bundle_diff.add_argument("right")
    validate = sub.add_parser("validate", help="validate the store or one manifest")
    validate.add_argument("root", type=Path)
    validate.add_argument("reference", nargs="?")
    put = sub.add_parser("put", help="publish one existing JSON manifest into the CAS")
    put.add_argument("root", type=Path)
    put.add_argument("manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        store = VMCacheStore(args.root)
        if args.command == "list":
            if args.bundles:
                payload = {
                    "schema": VM_CACHE_BUNDLE_LIST_SCHEMA,
                    "root": str(store.root),
                    "bundles": list(store.list_bundles()),
                }
            else:
                payload = {
                    "schema": VM_CACHE_LIST_SCHEMA,
                    "root": str(store.root),
                    "plane": args.plane,
                    "manifests": list(store.list(args.plane)),
                }
        elif args.command == "bundles":
            payload = {
                "schema": VM_CACHE_BUNDLE_LIST_SCHEMA,
                "root": str(store.root),
                "bundles": list(store.list_bundles()),
            }
        elif args.command == "show":
            payload = store.show(args.reference)
        elif args.command == "bundle-show":
            payload = store.show_bundle(args.reference)
        elif args.command == "diff":
            payload = store.diff(args.left, args.right)
        elif args.command == "bundle-diff":
            payload = store.diff_bundles(args.left, args.right)
        elif args.command == "put":
            manifest = _load_manifest(args.manifest)
            payload = {"reference": store.put(manifest), **manifest.summary()}
        else:
            manifest = store.get(args.reference) if args.reference else None
            if manifest is not None:
                payload = {"status": "valid", **manifest.summary()}
            else:
                summaries = store.list()
                payload = {
                    "schema": VM_CACHE_STORE_SCHEMA,
                    "status": "valid",
                    "manifest_count": len(summaries),
                    "manifests": list(summaries),
                    "bundle_count": len(store.list_bundles()),
                    "bundles": list(store.list_bundles()),
                    "semantic_equivalence": "not_assessed",
                }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (VMCacheError, OSError, json.JSONDecodeError) as exc:
        print(f"saturn-vm-cache: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VM_CACHE_BUNDLE_DIFF_SCHEMA",
    "VM_CACHE_BUNDLE_LIST_SCHEMA",
    "VM_CACHE_BUNDLE_REF_PREFIX",
    "VM_CACHE_BUNDLE_SCHEMA",
    "VM_CACHE_DIFF_SCHEMA",
    "VM_CACHE_ENTRY_SCHEMA",
    "VM_CACHE_LIST_SCHEMA",
    "VM_CACHE_MANIFEST_SCHEMA",
    "VM_CACHE_MAX_METADATA_BYTES",
    "VM_CACHE_PLANES",
    "VM_CACHE_POINTER_SCHEMA",
    "VM_CACHE_REF_PREFIX",
    "VM_CACHE_SCHEMA",
    "VM_CACHE_STORE_SCHEMA",
    "VMCacheEntry",
    "VMCacheError",
    "VMCacheExecutionBundle",
    "VMCacheEmitter",
    "VMCacheManifest",
    "VMCachePlane",
    "VMCachePointer",
    "VMCacheStore",
    "build_address_cache_manifest",
    "build_operator_cache_manifest",
    "build_renderer_cache_manifest",
    "build_source_cache_manifest",
    "build_vm_cache_manifest",
    "canonical_bytes",
    "content_digest",
    "diff_vm_cache_manifests",
    "diff_vm_cache_bundles",
    "diff_vm_cache_bundle_store",
    "diff_vm_cache_store",
    "list_vm_cache_bundles",
    "list_vm_cache_manifests",
    "main",
    "show_vm_cache_bundle",
    "show_vm_cache_manifest",
    "write_vm_cache_manifest",
]
