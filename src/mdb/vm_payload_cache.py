"""Physical content-addressed payloads for the Saturn VM.

``saturn.vm_cache`` intentionally stores metadata-only pointers.  This module
is the complementary, local data-plane seam: it stores immutable bytes,
rehydrates them with checksum/size/shape/dtype checks, and returns measured
receipts for every operation.  An optional bounded resident LRU tier reports
resident reuse separately from physical disk hydration.  The store owns byte
custody only.  It never claims that a rehydrated payload is semantically
equivalent to a native model value; a native consumer remains an explicit
external authority boundary.

The implementation is deliberately dependency-light.  Raw bytes work without
NumPy or torch.  ``PayloadHydration.as_numpy`` and ``as_torch`` are optional
convenience views and retain the descriptor checks before converting.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

PAYLOAD_CACHE_SCHEMA = "saturn-vm-payload-cache-v1"
PAYLOAD_DESCRIPTOR_SCHEMA = "saturn-vm-payload-descriptor-v1"
PAYLOAD_RECEIPT_SCHEMA = "saturn-vm-payload-receipt-v1"
PAYLOAD_REF_PREFIX = "cas://vm-payload/sha256/"
PAYLOAD_AUTHORITY_CACHE_ONLY = "cache-byte-integrity-only"
PAYLOAD_NATIVE_AUTHORITY_EXTERNAL = "external-native-consumer"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_ENCODINGS = frozenset({"raw"})
_KNOWN_ITEMSIZE = {
    "bool": 1,
    "boolean": 1,
    "uint8": 1,
    "int8": 1,
    "byte": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "half": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "single": 4,
    "uint64": 8,
    "int64": 8,
    "float64": 8,
    "double": 8,
    "complex64": 8,
    "complex128": 16,
}


class PayloadCacheError(ValueError):
    """Raised when a physical payload or its receipt is unsafe."""


class PayloadCacheMiss(PayloadCacheError):
    """Raised when a descriptor exists only as a pointer or is absent."""

    def __init__(self, message: str, *, receipt: PayloadCacheReceipt | None = None) -> None:
        super().__init__(message)
        self.receipt = receipt


class PayloadIntegrityError(PayloadCacheError):
    """Raised when stored bytes or their descriptor fail closed verification."""

    def __init__(self, message: str, *, receipt: PayloadCacheReceipt | None = None) -> None:
        super().__init__(message)
        self.receipt = receipt


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
        raise PayloadCacheError("value is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical(value))


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PayloadCacheError(f"{label} must be non-empty text")
    return value.strip()


def _digest(value: Any, label: str) -> str:
    text = _text(value, label).lower().removeprefix("sha256:")
    if _SHA256_RE.fullmatch(text) is None:
        raise PayloadCacheError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _optional_digest(value: Any, label: str) -> str | None:
    if value in (None, ""):
        return None
    return _digest(value, label)


def _shape(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PayloadCacheError("payload shape must be an integer sequence")
    result = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in result):
        raise PayloadCacheError("payload shape must contain non-negative integers")
    if len(result) > 64:
        raise PayloadCacheError("payload shape has too many dimensions")
    return result


def _dtype_name(value: Any) -> str | None:
    if value is None:
        return None
    text = _text(value, "payload dtype")
    if len(text.encode("utf-8")) > 256:
        raise PayloadCacheError("payload dtype is too long")
    return text


def _dtype_itemsize(dtype: str | None) -> int | None:
    if dtype is None:
        return None
    normalized = dtype.strip().lower()
    for prefix in ("torch.", "numpy.", "np."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    known = _KNOWN_ITEMSIZE.get(normalized)
    if known is not None:
        return known
    try:
        import numpy as np  # type: ignore

        itemsize = int(np.dtype(dtype).itemsize)
    except (ImportError, TypeError, ValueError):
        return None
    return itemsize if itemsize > 0 else None


def _shape_numel(shape: tuple[int, ...] | None) -> int | None:
    if shape is None:
        return None
    result = 1
    for item in shape:
        result *= item
    return result


def _validate_geometry(
    *, bytes_count: int, shape: tuple[int, ...] | None, dtype: str | None, label: str
) -> None:
    if bytes_count < 0:
        raise PayloadCacheError(f"{label} byte count must be non-negative")
    itemsize = _dtype_itemsize(dtype)
    numel = _shape_numel(shape)
    if itemsize is not None and numel is not None and bytes_count != itemsize * numel:
        raise PayloadCacheError(
            f"{label} byte count {bytes_count} disagrees with shape {shape} and "
            f"dtype {dtype!r} (expected {itemsize * numel})"
        )


def _metadata(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PayloadCacheError(f"{label} must be a JSON object")
    try:
        result = json.loads(_canonical(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PayloadCacheError(f"{label} must be strict JSON metadata") from exc
    if not isinstance(result, dict):  # pragma: no cover - guarded above
        raise PayloadCacheError(f"{label} must be a JSON object")
    return result


@dataclass(frozen=True, slots=True)
class PayloadRef:
    """Verified identity and geometry for one immutable physical payload."""

    sha256: str
    bytes: int
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    role: str = "opaque"
    encoding: str = "raw"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    authority_scope: str = PAYLOAD_AUTHORITY_CACHE_ONLY
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = PAYLOAD_DESCRIPTOR_SCHEMA

    def __post_init__(self) -> None:
        digest = _digest(self.sha256, "payload sha256")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
            raise PayloadCacheError("payload bytes must be a non-negative integer")
        resolved_shape = _shape(self.shape)
        resolved_dtype = _dtype_name(self.dtype)
        _validate_geometry(
            bytes_count=self.bytes,
            shape=resolved_shape,
            dtype=resolved_dtype,
            label="payload",
        )
        role = _text(self.role, "payload role")
        encoding = _text(self.encoding, "payload encoding")
        if encoding not in _ALLOWED_ENCODINGS:
            raise PayloadCacheError(f"unsupported payload encoding {encoding!r}")
        authority = _text(self.authority_scope, "payload authority scope")
        if authority != PAYLOAD_AUTHORITY_CACHE_ONLY:
            raise PayloadCacheError(
                "payload refs can only grant byte-integrity authority; native authority "
                "must remain external"
            )
        metadata = _metadata(self.metadata, "payload metadata")
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "shape", resolved_shape)
        object.__setattr__(self, "dtype", resolved_dtype)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "encoding", encoding)
        object.__setattr__(self, "authority_scope", authority)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "fingerprint", _sha256_json(self._body()))

    @property
    def ref(self) -> str:
        return PAYLOAD_REF_PREFIX + self.sha256

    @property
    def artifact_ref(self) -> str:
        return self.ref

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "shape": None if self.shape is None else list(self.shape),
            "dtype": self.dtype,
            "role": self.role,
            "encoding": self.encoding,
            "metadata": dict(self.metadata),
            "authority_scope": self.authority_scope,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "ref": self.ref, "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PayloadRef:
        if not isinstance(value, Mapping):
            raise PayloadCacheError("payload descriptor must be an object")
        result = cls(
            sha256=value.get("sha256"),
            bytes=value.get("bytes"),
            shape=value.get("shape"),
            dtype=value.get("dtype"),
            role=value.get("role", "opaque"),
            encoding=value.get("encoding", "raw"),
            metadata=value.get("metadata", {}),
            authority_scope=value.get("authority_scope", PAYLOAD_AUTHORITY_CACHE_ONLY),
        )
        if value.get("schema", cls.schema) != cls.schema:
            raise PayloadCacheError("payload descriptor schema changed")
        declared_ref = value.get("ref")
        if declared_ref not in (None, result.ref):
            raise PayloadCacheError("payload descriptor ref does not match its digest")
        if value.get("fingerprint") not in (None, result.fingerprint):
            raise PayloadCacheError("payload descriptor fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class PayloadCacheStats:
    """Cumulative measured physical cache counters."""

    puts: int = 0
    put_hits: int = 0
    hydrations: int = 0
    hits: int = 0
    misses: int = 0
    integrity_failures: int = 0
    bytes_written: int = 0
    bytes_read: int = 0
    resident_hits: int = 0
    resident_misses: int = 0
    resident_evictions: int = 0
    resident_bytes_read: int = 0
    resident_bytes: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "puts": self.puts,
            "put_hits": self.put_hits,
            "hydrations": self.hydrations,
            "hits": self.hits,
            "misses": self.misses,
            "integrity_failures": self.integrity_failures,
            "bytes_written": self.bytes_written,
            "bytes_read": self.bytes_read,
            "resident_hits": self.resident_hits,
            "resident_misses": self.resident_misses,
            "resident_evictions": self.resident_evictions,
            "resident_bytes_read": self.resident_bytes_read,
            "resident_bytes": self.resident_bytes,
        }


@dataclass(frozen=True, slots=True)
class PayloadCacheReceipt:
    """Measured result of a physical put or hydration operation.

    ``bytes_read`` counts bytes read from the durable object store.  A
    resident-tier hit reports those bytes as ``resident_bytes_read`` instead,
    making a memory reuse visibly different from disk hydration.
    """

    operation: str
    ref: str
    cache_hit: bool
    cache_miss: bool
    bytes_read: int
    bytes_written: int
    checksum_verified: bool
    size_verified: bool
    shape_verified: bool
    dtype_verified: bool
    authority_boundary: Mapping[str, Any]
    stats: Mapping[str, int]
    residency_source: str = "disk"
    resident_bytes_read: int = 0
    error: str | None = None
    elapsed_seconds: float = 0.0
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = PAYLOAD_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.operation not in {"put", "hydrate"}:
            raise PayloadCacheError("payload receipt operation must be put or hydrate")
        if self.residency_source not in {"put", "disk", "resident"}:
            raise PayloadCacheError("payload receipt has an unsupported residency source")
        _text(self.ref, "payload receipt ref")
        for name in ("bytes_read", "bytes_written", "resident_bytes_read"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PayloadCacheError(f"payload receipt {name} must be non-negative")
        if self.elapsed_seconds < 0 or not math.isfinite(float(self.elapsed_seconds)):
            raise PayloadCacheError("payload receipt elapsed_seconds must be finite")
        authority = _metadata(self.authority_boundary, "authority boundary")
        stats = _metadata(self.stats, "payload stats")
        object.__setattr__(self, "authority_boundary", authority)
        object.__setattr__(self, "stats", stats)
        object.__setattr__(self, "fingerprint", _sha256_json(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "operation": self.operation,
            "ref": self.ref,
            "cache_hit": self.cache_hit,
            "cache_miss": self.cache_miss,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "checksum_verified": self.checksum_verified,
            "size_verified": self.size_verified,
            "shape_verified": self.shape_verified,
            "dtype_verified": self.dtype_verified,
            "authority_boundary": dict(self.authority_boundary),
            "stats": dict(self.stats),
            "residency_source": self.residency_source,
            "resident_bytes_read": self.resident_bytes_read,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PayloadCacheReceipt:
        if not isinstance(value, Mapping):
            raise PayloadCacheError("payload receipt must be an object")
        result = cls(
            operation=value.get("operation"),
            ref=value.get("ref"),
            cache_hit=value.get("cache_hit", False),
            cache_miss=value.get("cache_miss", False),
            bytes_read=value.get("bytes_read", 0),
            bytes_written=value.get("bytes_written", 0),
            checksum_verified=value.get("checksum_verified", False),
            size_verified=value.get("size_verified", False),
            shape_verified=value.get("shape_verified", False),
            dtype_verified=value.get("dtype_verified", False),
            authority_boundary=value.get("authority_boundary", {}),
            stats=value.get("stats", {}),
            residency_source=value.get("residency_source", "disk"),
            resident_bytes_read=value.get("resident_bytes_read", 0),
            error=value.get("error"),
            elapsed_seconds=value.get("elapsed_seconds", 0.0),
        )
        if value.get("schema", cls.schema) != cls.schema:
            raise PayloadCacheError("payload receipt schema changed")
        declared_fingerprint = value.get("fingerprint")
        if declared_fingerprint not in (None, result.fingerprint):
            # Receipts are durable artifacts.  Accept a receipt emitted by an
            # earlier v1 writer that did not yet include the resident-byte
            # counter, while still checking the historical body digest.
            legacy_body = dict(value)
            legacy_body.pop("fingerprint", None)
            try:
                legacy_matches = (
                    "resident_bytes_read" not in value
                    and _sha256_json(legacy_body) == declared_fingerprint
                )
            except PayloadCacheError:
                legacy_matches = False
            if not legacy_matches:
                raise PayloadCacheError("payload receipt fingerprint mismatch")
        return result

    @property
    def hit(self) -> bool:
        return self.cache_hit

    @property
    def miss(self) -> bool:
        return self.cache_miss

    @property
    def resident(self) -> bool:
        return self.residency_source == "resident"


@dataclass(frozen=True, slots=True)
class PayloadPutResult:
    ref: PayloadRef
    receipt: PayloadCacheReceipt


@dataclass(frozen=True, slots=True)
class PayloadHydration:
    """Hydrated bytes plus its measured receipt.

    ``payload`` is immutable ``bytes``.  Conversion to an array/tensor is
    explicit so a cache hit cannot silently become native-consumer authority.
    """

    ref: PayloadRef
    payload: bytes
    receipt: PayloadCacheReceipt

    @property
    def bytes(self) -> bytes:
        return self.payload

    @property
    def authority_boundary(self) -> Mapping[str, Any]:
        return self.receipt.authority_boundary

    @property
    def source(self) -> str:
        return self.receipt.residency_source

    def as_numpy(self) -> Any:
        if self.ref.dtype is None or self.ref.shape is None:
            raise PayloadCacheError("array conversion requires declared dtype and shape")
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise PayloadCacheError("NumPy is required for as_numpy") from exc
        array = np.frombuffer(self.payload, dtype=np.dtype(self.ref.dtype))
        return array.reshape(self.ref.shape).copy()

    def as_torch(self, *, device: str = "cpu") -> Any:
        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise PayloadCacheError("torch is required for as_torch") from exc
        array = self.as_numpy()
        return torch.from_numpy(array).to(device=device)


def _bytes_value(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise PayloadCacheError("payload must be bytes-like; use put_array for tensor-like values")


def _array_bytes(value: Any) -> tuple[bytes, tuple[int, ...], str]:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    contiguous = getattr(candidate, "contiguous", None)
    if callable(contiguous):
        candidate = contiguous()
    shape = getattr(candidate, "shape", None)
    dtype = getattr(candidate, "dtype", None)
    tobytes = getattr(candidate, "tobytes", None)
    if not callable(tobytes):
        # Torch tensors expose NumPy conversion after the optional CPU copy.
        to_numpy = getattr(candidate, "numpy", None)
        if callable(to_numpy):
            candidate = to_numpy()
            shape = getattr(candidate, "shape", shape)
            dtype = getattr(candidate, "dtype", dtype)
            tobytes = getattr(candidate, "tobytes", None)
    if not callable(tobytes) or shape is None or dtype is None:
        raise PayloadCacheError("put_array requires a NumPy/torch-like contiguous array")
    resolved_shape = _shape(tuple(int(item) for item in shape))
    assert resolved_shape is not None
    resolved_dtype = _dtype_name(str(dtype))
    assert resolved_dtype is not None
    payload = bytes(tobytes(order="C"))
    return payload, resolved_shape, resolved_dtype


class PayloadStore:
    """Local immutable CAS for VM payload bytes and hydration receipts."""

    schema: ClassVar[str] = PAYLOAD_CACHE_SCHEMA

    def __init__(self, root: str | Path, *, resident_capacity_bytes: int = 0,
                 read_only: bool = False) -> None:
        if not isinstance(read_only, bool): raise PayloadCacheError("read_only must be boolean")
        if (
            isinstance(resident_capacity_bytes, bool)
            or not isinstance(resident_capacity_bytes, int)
            or resident_capacity_bytes < 0
        ):
            raise PayloadCacheError("resident_capacity_bytes must be a non-negative integer")
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects"
        self.descriptors = self.root / "descriptors"
        self.receipts = self.root / "receipts"
        self.read_only = read_only
        self._audit_rows: dict[str, dict[str, Any]] = {}
        self._audit_errors: list[dict[str, str]] = []
        self._audit_attempts = self._audit_bytes = self._audit_misses = self._audit_failures = 0
        self.resident_capacity_bytes = resident_capacity_bytes
        self._resident: OrderedDict[str, bytes] = OrderedDict()
        self._resident_pinned: set[str] = set()
        self._resident_bytes = 0
        for directory in (self.objects, self.descriptors, self.receipts):
            if read_only:
                if not directory.is_dir():
                    self._audit_failures += 1
                    error = PayloadCacheError("read-only payload store is incomplete")
                    error.audit_summary = self.audit_summary()
                    raise error
            else:
                directory.mkdir(parents=True, exist_ok=True)
        if read_only and not (self.root / "manifest.json").is_file():
            self._audit_failures += 1
            error = PayloadCacheError("read-only payload store manifest is absent")
            error.audit_summary = self.audit_summary()
            raise error
        self._stats = {
            "puts": 0,
            "put_hits": 0,
            "hydrations": 0,
            "hits": 0,
            "misses": 0,
            "integrity_failures": 0,
            "bytes_written": 0,
            "bytes_read": 0,
            "resident_hits": 0,
            "resident_misses": 0,
            "resident_evictions": 0,
            "resident_bytes_read": 0,
            "resident_bytes": 0,
        }
        try: self._write_store_manifest()
        except PayloadCacheError as exc:
            if read_only:
                self._audit_failures += 1
                exc.audit_summary = self.audit_summary()
            raise

    def _write_store_manifest(self) -> None:
        body = {
            "schema": self.schema,
            "storage": "immutable-payload-bytes-and-descriptor-cas-v1",
            "authority_scope": PAYLOAD_AUTHORITY_CACHE_ONLY,
        }
        payload = {**body, "content_sha256": _sha256_json(body)}
        target = self.root / "manifest.json"
        if target.exists():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PayloadCacheError(f"cannot read payload store manifest: {target}") from exc
            if existing != payload:
                # Resident memory is process-local and intentionally does not
                # participate in the persistent CAS identity.  Accept stores
                # written by the first resident-tier revision, whose manifest
                # recorded the configured capacity, when the immutable
                # storage contract itself is unchanged.
                legacy_body = {
                    key: value
                    for key, value in existing.items()
                    if key not in {"content_sha256", "resident_capacity_bytes"}
                }
                if legacy_body == body:
                    return
                raise PayloadCacheError("payload store manifest changed")
            return
        self._atomic_publish(target, _canonical(payload) + b"\n")

    @property
    def stats(self) -> PayloadCacheStats:
        return PayloadCacheStats(**self._stats)

    @property
    def metrics(self) -> dict[str, int]:
        return self.stats.to_dict()

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
                    raise PayloadCacheError(
                        f"immutable CAS path contains different bytes: {path}"
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

    @staticmethod
    def _digest_from_ref(value: PayloadRef | Mapping[str, Any] | str | Path) -> str:
        if isinstance(value, PayloadRef):
            return value.sha256
        if isinstance(value, Mapping):
            return PayloadRef.from_dict(value).sha256
        text = str(value)
        if text.startswith(PAYLOAD_REF_PREFIX):
            return _digest(text[len(PAYLOAD_REF_PREFIX) :], "payload reference")
        return _digest(text, "payload digest")

    def object_path(self, value: PayloadRef | Mapping[str, Any] | str | Path) -> Path:
        """Return the payload path for read-only inspection or integrity tests."""

        return self.objects / f"{self._digest_from_ref(value)}.bin"

    def _resident_put(self, ref: PayloadRef, payload: bytes) -> None:
        capacity = self.resident_capacity_bytes
        if capacity <= 0 or len(payload) > capacity:
            return
        digest = ref.sha256
        existing = self._resident.pop(digest, None)
        if existing is not None:
            self._resident_bytes -= len(existing)
        while self._resident and self._resident_bytes + len(payload) > capacity:
            evicted_digest, evicted = next(iter(self._resident.items()))
            if evicted_digest in self._resident_pinned:
                # A pinned entry is never silently discarded.  Search for the
                # oldest unpinned entry deterministically instead.
                candidate = next(
                    (
                        (item_digest, item)
                        for item_digest, item in self._resident.items()
                        if item_digest not in self._resident_pinned
                    ),
                    None,
                )
                if candidate is None:
                    break
                evicted_digest, evicted = candidate
            self._resident.pop(evicted_digest)
            self._resident_bytes -= len(evicted)
            self._resident_pinned.discard(evicted_digest)
            self._stats["resident_evictions"] += 1
        if self._resident_bytes + len(payload) <= capacity:
            self._resident[digest] = payload
            self._resident_bytes += len(payload)
            self._stats["resident_bytes"] = self._resident_bytes

    def _resident_get(self, ref: PayloadRef) -> bytes | None:
        payload = self._resident.get(ref.sha256)
        if payload is None:
            self._stats["resident_misses"] += 1
            return None
        self._resident.move_to_end(ref.sha256)
        self._stats["resident_hits"] += 1
        self._stats["resident_bytes_read"] += len(payload)
        return payload

    def pin(self, value: PayloadRef | Mapping[str, Any] | str | Path) -> bool:
        """Pin an already resident payload against deterministic LRU eviction."""

        digest = self._digest_from_ref(value)
        if digest not in self._resident:
            return False
        self._resident_pinned.add(digest)
        return True

    def unpin(self, value: PayloadRef | Mapping[str, Any] | str | Path) -> bool:
        digest = self._digest_from_ref(value)
        was_pinned = digest in self._resident_pinned
        self._resident_pinned.discard(digest)
        return was_pinned

    def evict(self, value: PayloadRef | Mapping[str, Any] | str | Path | None = None) -> int:
        """Evict one payload or all unpinned entries from resident memory."""

        if value is not None:
            digest = self._digest_from_ref(value)
            if digest in self._resident_pinned:
                raise PayloadCacheError("cannot evict a pinned resident payload")
            payload = self._resident.pop(digest, None)
            if payload is None:
                return 0
            self._resident_bytes -= len(payload)
            self._resident_pinned.discard(digest)
            self._stats["resident_evictions"] += 1
            self._stats["resident_bytes"] = self._resident_bytes
            return 1
        count = 0
        for digest in list(self._resident):
            if digest in self._resident_pinned:
                continue
            payload = self._resident.pop(digest)
            self._resident_bytes -= len(payload)
            self._stats["resident_evictions"] += 1
            count += 1
        self._stats["resident_bytes"] = self._resident_bytes
        return count

    def resident_snapshot(self) -> tuple[str, ...]:
        """Return resident digests from oldest to newest (LRU order)."""

        return tuple(self._resident)

    def descriptor_path(self, value: PayloadRef | Mapping[str, Any] | str | Path) -> Path:
        return self.descriptors / f"{self._digest_from_ref(value)}.json"

    def _write_receipt(self, receipt: PayloadCacheReceipt) -> str:
        if self.read_only: raise PayloadCacheError("read-only payload store cannot publish receipts")
        target = self.receipts / f"{receipt.fingerprint}.json"
        self._atomic_publish(target, _canonical(receipt.to_dict()) + b"\n")
        return f"{PAYLOAD_REF_PREFIX}{receipt.fingerprint}"

    def _descriptor_from_disk(self, digest: str) -> PayloadRef:
        path = self.descriptors / f"{digest}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PayloadCacheMiss(f"payload descriptor is not present: {digest}") from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PayloadIntegrityError(f"cannot read payload descriptor: {path}") from exc
        result = PayloadRef.from_dict(payload)
        if result.sha256 != digest:
            raise PayloadIntegrityError("payload descriptor filename does not match its digest")
        return result

    def descriptor(self, value: PayloadRef | Mapping[str, Any] | str | Path) -> PayloadRef:
        digest = self._digest_from_ref(value)
        result = self._descriptor_from_disk(digest)
        supplied: PayloadRef | None = None
        if isinstance(value, PayloadRef):
            supplied = value
        elif isinstance(value, Mapping):
            supplied = PayloadRef.from_dict(value)
        if supplied is not None and supplied.to_dict() != result.to_dict():
            raise PayloadIntegrityError(
                "supplied payload descriptor disagrees with the immutable stored descriptor"
            )
        return result

    def put_bytes(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        shape: Sequence[int] | None = None,
        dtype: str | None = None,
        role: str = "opaque",
        metadata: Mapping[str, Any] | None = None,
    ) -> PayloadRef:
        if self.read_only: raise PayloadCacheError("read-only payload store cannot put bytes")
        started = time.perf_counter()
        raw = _bytes_value(payload)
        resolved_shape = _shape(shape)
        resolved_dtype = _dtype_name(dtype)
        _validate_geometry(
            bytes_count=len(raw),
            shape=resolved_shape,
            dtype=resolved_dtype,
            label="payload",
        )
        digest = _sha256_bytes(raw)
        ref = PayloadRef(
            sha256=digest,
            bytes=len(raw),
            shape=resolved_shape,
            dtype=resolved_dtype,
            role=role,
            metadata=metadata or {},
        )
        self._stats["puts"] += 1
        object_path = self.object_path(ref)
        descriptor_path = self.descriptor_path(ref)
        object_existed = object_path.exists()
        descriptor_existed = descriptor_path.exists()
        complete_hit = object_existed and descriptor_existed
        if object_existed and object_path.read_bytes() != raw:
            raise PayloadIntegrityError("payload content collision at an existing digest")
        if descriptor_existed:
            existing = self._descriptor_from_disk(digest)
            if existing.to_dict() != ref.to_dict():
                raise PayloadIntegrityError("payload descriptor collision at an existing digest")
        if not object_existed:
            self._atomic_publish(object_path, raw)
            self._stats["bytes_written"] += len(raw)
        if not descriptor_existed:
            self._atomic_publish(descriptor_path, _canonical(ref.to_dict()) + b"\n")
        if complete_hit:
            self._stats["put_hits"] += 1
        receipt = self._make_receipt(
            operation="put",
            ref=ref,
            cache_hit=complete_hit,
            cache_miss=False,
            bytes_read=0,
            bytes_written=0 if object_existed else len(raw),
            checksum_verified=True,
            size_verified=True,
            shape_verified=True,
            dtype_verified=True,
            started=started,
            residency_source="put",
        )
        self._write_receipt(receipt)
        return ref

    def put_array(
        self,
        value: Any,
        *,
        role: str = "tensor",
        metadata: Mapping[str, Any] | None = None,
    ) -> PayloadRef:
        payload, shape, dtype = _array_bytes(value)
        return self.put_bytes(payload, shape=shape, dtype=dtype, role=role, metadata=metadata)

    def put_with_receipt(
        self, payload: bytes | bytearray | memoryview, **kwargs: Any
    ) -> PayloadPutResult:
        before = self.stats.puts
        ref = self.put_bytes(payload, **kwargs)
        # The receipt is deterministic for the just-completed operation except
        # for elapsed time; find the latest matching receipt by digest instead
        # of retaining mutable state in the public API.
        candidates = sorted(self.receipts.glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
        for path in reversed(candidates):
            try:
                payload_data = json.loads(path.read_text(encoding="utf-8"))
                if payload_data.get("operation") == "put" and payload_data.get("ref") == ref.ref:
                    return PayloadPutResult(
                        ref=ref,
                        receipt=PayloadCacheReceipt.from_dict(payload_data),
                    )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, PayloadCacheError):
                continue
        if self.stats.puts == before:  # pragma: no cover - defensive
            raise PayloadCacheError("put did not produce a receipt")
        raise PayloadCacheError("put receipt was not readable")

    def _make_receipt(
        self,
        *,
        operation: str,
        ref: PayloadRef,
        cache_hit: bool,
        cache_miss: bool,
        bytes_read: int,
        bytes_written: int,
        checksum_verified: bool,
        size_verified: bool,
        shape_verified: bool,
        dtype_verified: bool,
        started: float,
        error: str | None = None,
        native_consumer: str | None = None,
        residency_source: str = "disk",
        resident_bytes_read: int = 0,
    ) -> PayloadCacheReceipt:
        return PayloadCacheReceipt(
            operation=operation,
            ref=ref.ref,
            cache_hit=cache_hit,
            cache_miss=cache_miss,
            bytes_read=bytes_read,
            bytes_written=bytes_written,
            checksum_verified=checksum_verified,
            size_verified=size_verified,
            shape_verified=shape_verified,
            dtype_verified=dtype_verified,
            authority_boundary={
                "payload_store": PAYLOAD_AUTHORITY_CACHE_ONLY,
                "native_consumer": native_consumer,
                "native_consumer_authority": (
                    PAYLOAD_NATIVE_AUTHORITY_EXTERNAL if native_consumer else "unbound"
                ),
                "semantic_equivalence": "not_assessed",
            },
            stats=self.metrics,
            residency_source=residency_source,
            resident_bytes_read=resident_bytes_read,
            error=error,
            elapsed_seconds=max(0.0, time.perf_counter() - started),
        )

    @staticmethod
    def _expected(value: Any, label: str) -> Any:
        if value is None:
            return None
        return value

    def hydrate(
        self,
        value: PayloadRef | Mapping[str, Any] | str | Path,
        *,
        expected_shape: Sequence[int] | None = None,
        expected_dtype: str | None = None,
        expected_bytes: int | None = None,
        native_consumer: str | None = None,
        prefer_resident: bool = True,
    ) -> PayloadHydration:
        """Hydrate for a consumer, preserving durable per-operation receipts."""
        if self.read_only: raise PayloadCacheError("read-only payload store requires audit")
        return self._hydrate(value, expected_shape=expected_shape, expected_dtype=expected_dtype,
            expected_bytes=expected_bytes, native_consumer=native_consumer,
            prefer_resident=prefer_resident, persist_receipt=True)

    def audit(
        self,
        value: PayloadRef | Mapping[str, Any] | str | Path,
        *,
        expected_shape: Sequence[int] | None = None,
        expected_dtype: str | None = None,
        expected_bytes: int | None = None,
    ) -> PayloadHydration:
        """Verify disk bytes without publishing a native-consumer receipt.

        The returned measurement remains available to the aggregate audit. It
        has no native consumer authority and never trusts the resident cache.
        """
        self._audit_attempts += 1
        try:
            result = self._hydrate(value, expected_shape=expected_shape, expected_dtype=expected_dtype,
                expected_bytes=expected_bytes, native_consumer=None,
                prefer_resident=False, persist_receipt=False)
        except PayloadCacheError as exc:
            self._audit_failures += 1
            self._audit_misses += isinstance(exc, PayloadCacheMiss)
            receipt = getattr(exc, "receipt", None)
            if receipt: self._audit_bytes += receipt.bytes_read
            try: failed_ref = self._digest_from_ref(value)
            except PayloadCacheError: failed_ref = "invalid-reference"
            self._audit_errors.append({"payload_sha256": failed_ref, "error_type": type(exc).__name__})
            exc.audit_summary = self.audit_summary()
            raise
        self._audit_bytes += result.receipt.bytes_read
        self._audit_rows[result.ref.fingerprint] = {
            "descriptor_sha256": _sha256_json(result.ref.to_dict()),
            "payload_sha256": result.ref.sha256, "bytes": result.ref.bytes}
        return result

    def audit_summary(self) -> dict[str, Any]:
        """Deterministic aggregate; no elapsed times or receipt fingerprints."""
        return {"schema": "mdb-payload-audit-v1",
            "mechanics_status": "invalid" if self._audit_failures else "valid",
            "unique_payloads": len(self._audit_rows), "read_attempts": self._audit_attempts,
            "bytes_read": self._audit_bytes,
            "unique_verified_bytes": sum(r["bytes"] for r in self._audit_rows.values()),
            "descriptor_payload_sha256": _sha256_json(self._audit_rows),
            "failed_payloads": list(self._audit_errors),
            "checksum_verified": len(self._audit_rows), "geometry_verified": len(self._audit_rows),
            "misses": self._audit_misses, "failures": self._audit_failures,
            "authority": "disk-byte-integrity; native consumer not assessed"}

    def _hydrate(
        self,
        value: PayloadRef | Mapping[str, Any] | str | Path,
        *,
        expected_shape: Sequence[int] | None = None,
        expected_dtype: str | None = None,
        expected_bytes: int | None = None,
        native_consumer: str | None = None,
        prefer_resident: bool = True,
        persist_receipt: bool = True,
    ) -> PayloadHydration:
        """Read and verify one payload, returning measured bytes and receipt.

        A ``native_consumer`` label records the external boundary in the
        receipt; it does not elevate the cache to semantic authority.
        """

        started = time.perf_counter()
        digest = self._digest_from_ref(value)
        try:
            ref = self.descriptor(value)
        except PayloadCacheMiss as exc:
            self._stats["misses"] += 1
            placeholder = PayloadRef(sha256=digest, bytes=0)
            receipt = self._make_receipt(
                operation="hydrate",
                ref=placeholder,
                cache_hit=False,
                cache_miss=True,
                bytes_read=0,
                bytes_written=0,
                checksum_verified=False,
                size_verified=False,
                shape_verified=False,
                dtype_verified=False,
                started=started,
                error=str(exc),
                native_consumer=native_consumer,
            )
            if persist_receipt: self._write_receipt(receipt)
            raise PayloadCacheMiss(str(exc), receipt=receipt) from exc
        source = "disk"
        raw = self._resident_get(ref) if prefer_resident else None
        if raw is not None:
            source = "resident"
        else:
            path = self.object_path(ref)
            try:
                raw = path.read_bytes()
            except FileNotFoundError as exc:
                self._stats["misses"] += 1
                receipt = self._make_receipt(
                    operation="hydrate",
                    ref=ref,
                    cache_hit=False,
                    cache_miss=True,
                    bytes_read=0,
                    bytes_written=0,
                    checksum_verified=False,
                    size_verified=False,
                    shape_verified=False,
                    dtype_verified=False,
                    started=started,
                    error="payload bytes are not present",
                    native_consumer=native_consumer,
                    residency_source="disk",
                )
                if persist_receipt: self._write_receipt(receipt)
                raise PayloadCacheMiss(
                    f"payload bytes are not present: {ref.ref}", receipt=receipt
                ) from exc
            except OSError as exc:
                raise PayloadIntegrityError(f"cannot read payload bytes: {path}") from exc
            self._stats["bytes_read"] += len(raw)
        self._stats["hydrations"] += 1
        checksum_ok = _sha256_bytes(raw) == ref.sha256
        size_ok = len(raw) == ref.bytes
        shape = _shape(expected_shape)
        dtype = _dtype_name(expected_dtype)
        shape_ok = shape is None or shape == ref.shape
        dtype_ok = dtype is None or dtype == ref.dtype
        expected_byte_count_ok = expected_bytes is None or len(raw) == int(expected_bytes)
        geometry_ok = True
        try:
            _validate_geometry(
                bytes_count=len(raw),
                shape=ref.shape,
                dtype=ref.dtype,
                label="stored payload",
            )
        except PayloadCacheError:
            geometry_ok = False
        if not (
            checksum_ok
            and size_ok
            and shape_ok
            and dtype_ok
            and expected_byte_count_ok
            and geometry_ok
        ):
            self._stats["integrity_failures"] += 1
            error_parts = []
            if not checksum_ok:
                error_parts.append("checksum mismatch")
            if not size_ok:
                error_parts.append("byte-size mismatch")
            if not shape_ok:
                error_parts.append(f"shape mismatch: expected {shape}, stored {ref.shape}")
            if not dtype_ok:
                error_parts.append(f"dtype mismatch: expected {dtype!r}, stored {ref.dtype!r}")
            if not expected_byte_count_ok:
                error_parts.append(
                    f"expected byte count mismatch: expected {expected_bytes}, stored {len(raw)}"
                )
            if not geometry_ok:
                error_parts.append("stored shape/dtype geometry mismatch")
            receipt = self._make_receipt(
                operation="hydrate",
                ref=ref,
                cache_hit=False,
                cache_miss=False,
                bytes_read=len(raw) if source == "disk" else 0,
                bytes_written=0,
                checksum_verified=checksum_ok,
                size_verified=size_ok and expected_byte_count_ok,
                shape_verified=shape_ok,
                dtype_verified=dtype_ok,
                started=started,
                error="; ".join(error_parts),
                native_consumer=native_consumer,
                residency_source=source,
                resident_bytes_read=len(raw) if source == "resident" else 0,
            )
            if persist_receipt: self._write_receipt(receipt)
            raise PayloadIntegrityError(
                receipt.error or "payload integrity check failed", receipt=receipt
            )
        self._stats["hits"] += 1
        if source == "disk":
            # Do not admit unverified bytes into the resident tier.  A
            # tampered object must fail closed on every access, rather than
            # becoming a resident cache hit after its first failed read.
            self._resident_put(ref, raw)
        receipt = self._make_receipt(
            operation="hydrate",
            ref=ref,
            cache_hit=True,
            cache_miss=False,
            bytes_read=len(raw) if source == "disk" else 0,
            bytes_written=0,
            checksum_verified=True,
            size_verified=True,
            shape_verified=True,
            dtype_verified=True,
            started=started,
            native_consumer=native_consumer,
            residency_source=source,
            resident_bytes_read=len(raw) if source == "resident" else 0,
        )
        if persist_receipt: self._write_receipt(receipt)
        return PayloadHydration(ref=ref, payload=raw, receipt=receipt)

    load = hydrate
    get = hydrate

    def native_consumer_boundary(self, consumer: str) -> dict[str, Any]:
        """Describe the explicit handoff without claiming semantic parity."""

        return {
            "consumer": _text(consumer, "native consumer"),
            "payload_store_authority": PAYLOAD_AUTHORITY_CACHE_ONLY,
            "native_consumer_authority": PAYLOAD_NATIVE_AUTHORITY_EXTERNAL,
            "semantic_equivalence": "not_assessed",
            "boundary": "hydrated-bytes-to-external-native-consumer",
        }

    def receipt(self, fingerprint: str) -> PayloadCacheReceipt:
        digest = _digest(fingerprint, "receipt fingerprint")
        path = self.receipts / f"{digest}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PayloadCacheError(f"cannot read payload receipt: {path}") from exc
        return PayloadCacheReceipt.from_dict(payload)


PayloadDescriptor = PayloadRef
HydrationReceipt = PayloadCacheReceipt
PhysicalPayloadStore = PayloadStore
VMCachePayloadStore = PayloadStore


__all__ = [
    "PAYLOAD_AUTHORITY_CACHE_ONLY",
    "PAYLOAD_CACHE_SCHEMA",
    "PAYLOAD_DESCRIPTOR_SCHEMA",
    "PAYLOAD_NATIVE_AUTHORITY_EXTERNAL",
    "PAYLOAD_RECEIPT_SCHEMA",
    "PAYLOAD_REF_PREFIX",
    "PayloadCacheError",
    "PayloadCacheMiss",
    "PayloadCacheReceipt",
    "PayloadCacheStats",
    "PayloadDescriptor",
    "PayloadHydration",
    "PayloadIntegrityError",
    "PayloadPutResult",
    "PayloadRef",
    "PayloadStore",
    "PhysicalPayloadStore",
    "VMCachePayloadStore",
    "HydrationReceipt",
]
