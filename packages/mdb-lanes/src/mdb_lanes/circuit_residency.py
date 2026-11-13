"""Typed, model-free weight residency mechanics for circuit execution.

The objects here describe *which* resources an act demands and the lifetime of
their residency.  They deliberately do not know about tensors, devices, or a
particular store implementation.  A production adapter can therefore bind the
same contract to QStore, safetensors, or another content-addressed weight plane.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

WEIGHT_SET_SCHEMA = "saturn-circuit-weight-set-v1"
WEIGHT_LEASE_SCHEMA = "saturn-circuit-weight-lease-v1"
WEIGHT_RECEIPT_SCHEMA = "saturn-circuit-weight-receipt-v1"

DemandClass = Literal["static", "cyclic", "routed"]


class CircuitResidencyError(ValueError):
    """Raised when a residency declaration or operation fails closed."""


class MissingCircuitWeightError(CircuitResidencyError, KeyError):
    """Raised before acquisition when a demanded resource is unavailable."""


class CircuitWeightBudgetError(CircuitResidencyError):
    """Raised when demanded resources cannot coexist inside the byte budget."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CircuitResidencyError(f"{label} must be non-empty text")
    return value.strip()


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _FrozenList(tuple):
    """Tuple storage with list-compatible equality for receipt compatibility."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(other)
        return False

    __hash__ = tuple.__hash__


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return _FrozenList(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(child) for child in value]
    return value


def _snapshot_payload(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return _freeze_json(json.loads(encoded))
    except (TypeError, ValueError) as exc:
        raise CircuitResidencyError(
            "provider resources must be bytes-like or finite JSON values"
        ) from exc


def _payload_digest(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return hashlib.sha256(bytes(value)).hexdigest()
    try:
        return _canonical_digest(_thaw_json(value))
    except (TypeError, ValueError) as exc:
        raise CircuitResidencyError(
            "provider resources must be bytes-like or finite JSON values"
        ) from exc


def _resource_rows(value: Any) -> tuple[tuple[str, int], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    try:
        rows = tuple(items)
    except TypeError as exc:
        raise CircuitResidencyError("resources must map resource keys to byte sizes") from exc
    normalized: list[tuple[str, int]] = []
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise CircuitResidencyError("resources must contain (key, byte_size) pairs")
        key = _text(row[0], "resource key")
        byte_size = row[1]
        if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size <= 0:
            raise CircuitResidencyError(f"resource {key!r} byte size must be a positive integer")
        normalized.append((key, byte_size))
    if not normalized:
        raise CircuitResidencyError("a weight set must demand at least one resource")
    if len({key for key, _ in normalized}) != len(normalized):
        raise CircuitResidencyError("weight-set resource keys must be unique")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class CircuitWeightSet:
    """An immutable, content-addressed resource demand for one circuit act."""

    weight_set_id: str
    resources: Mapping[str, int] | tuple[tuple[str, int], ...]
    demand_class: DemandClass = "static"
    routed_demand_source: str | None = None
    numerical_contract: str = "native"
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = WEIGHT_SET_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight_set_id", _text(self.weight_set_id, "weight_set_id"))
        rows = _resource_rows(self.resources)
        object.__setattr__(self, "resources", rows)
        if self.demand_class not in ("static", "cyclic", "routed"):
            raise CircuitResidencyError("demand_class must be static, cyclic, or routed")
        source = self.routed_demand_source
        if self.demand_class == "routed":
            source = _text(source, "routed_demand_source")
        elif source is not None:
            raise CircuitResidencyError(
                "routed_demand_source is valid only for demand_class='routed'"
            )
        object.__setattr__(self, "routed_demand_source", source)
        object.__setattr__(
            self, "numerical_contract", _text(self.numerical_contract, "numerical_contract")
        )
        object.__setattr__(self, "fingerprint", _canonical_digest(self._body()))

    @property
    def resource_keys(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.resources)

    @property
    def total_bytes(self) -> int:
        return sum(byte_size for _, byte_size in self.resources)

    def byte_size(self, resource_key: str) -> int:
        for key, byte_size in self.resources:
            if key == resource_key:
                return byte_size
        raise KeyError(resource_key)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "weight_set_id": self.weight_set_id,
            "resources": [
                {"key": key, "byte_size": byte_size} for key, byte_size in self.resources
            ],
            "demand_class": self.demand_class,
            "routed_demand_source": self.routed_demand_source,
            "numerical_contract": self.numerical_contract,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CircuitWeightSet:
        rows = value.get("resources", ())
        resources = tuple((row.get("key"), row.get("byte_size")) for row in rows)
        result = cls(
            weight_set_id=value.get("weight_set_id"),
            resources=resources,
            demand_class=value.get("demand_class", "static"),
            routed_demand_source=value.get("routed_demand_source"),
            numerical_contract=value.get("numerical_contract", "native"),
        )
        if value.get("schema") != cls.schema or value.get("fingerprint") != result.fingerprint:
            raise CircuitResidencyError("weight-set schema or fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class CircuitWeightLease:
    """Resources pinned for the dynamic extent of one provider acquisition."""

    lease_id: str
    provider_fingerprint: str
    weight_set_fingerprint: str
    cell_id: str
    resources: Mapping[str, Any]
    resident_bytes: int
    _lifetime: _LeaseLifetime = field(init=False, repr=False, compare=False)

    schema: ClassVar[str] = WEIGHT_LEASE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        provider = _text(self.provider_fingerprint, "provider_fingerprint").casefold()
        if len(provider) != 64 or any(char not in "0123456789abcdef" for char in provider):
            raise CircuitResidencyError("provider_fingerprint must be a SHA-256 digest")
        object.__setattr__(self, "provider_fingerprint", provider)
        fingerprint = _text(self.weight_set_fingerprint, "weight_set_fingerprint").casefold()
        if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
            raise CircuitResidencyError("weight_set_fingerprint must be a SHA-256 digest")
        object.__setattr__(self, "weight_set_fingerprint", fingerprint)
        object.__setattr__(self, "cell_id", _text(self.cell_id, "cell_id"))
        if not isinstance(self.resources, Mapping) or not self.resources:
            raise CircuitResidencyError("a weight lease must expose demanded resources")
        lifetime = _LeaseLifetime()
        object.__setattr__(self, "_lifetime", lifetime)
        object.__setattr__(self, "resources", _LeaseResourceView(dict(self.resources), lifetime))
        if (
            isinstance(self.resident_bytes, bool)
            or not isinstance(self.resident_bytes, int)
            or self.resident_bytes <= 0
        ):
            raise CircuitResidencyError("resident_bytes must be a positive integer")

    def _release(self) -> None:
        self._lifetime.release()


class _LeaseLifetime:
    def __init__(self) -> None:
        self._active = True
        self._lock = RLock()

    def ensure_active(self) -> None:
        with self._lock:
            if not self._active:
                raise CircuitResidencyError("weight lease has been released")

    def release(self) -> None:
        with self._lock:
            self._active = False


class _LeaseResourceView(Mapping[str, Any]):
    def __init__(self, resources: Mapping[str, Any], lifetime: _LeaseLifetime) -> None:
        self._resources = MappingProxyType(dict(resources))
        self._lifetime = lifetime

    def __getitem__(self, key: str) -> Any:
        self._lifetime.ensure_active()
        return self._resources[key]

    def __iter__(self) -> Iterator[str]:
        self._lifetime.ensure_active()
        return iter(self._resources)

    def __len__(self) -> int:
        self._lifetime.ensure_active()
        return len(self._resources)


@runtime_checkable
class WeightProvider(Protocol):
    """Minimum runtime weight plane; prefetch is an optional provider extension."""

    def acquire(
        self, weight_set: CircuitWeightSet, *, cell_id: str
    ) -> AbstractContextManager[CircuitWeightLease]: ...


@runtime_checkable
class PrefetchingWeightProvider(WeightProvider, Protocol):
    def prefetch(self, weight_set: CircuitWeightSet, *, cell_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class CircuitWeightReceipt:
    """Deterministic provider telemetry with a content fingerprint."""

    provider_fingerprint: str
    resident_byte_budget: int
    events: tuple[Mapping[str, Any], ...]
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = WEIGHT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        provider = _text(self.provider_fingerprint, "provider_fingerprint").casefold()
        if len(provider) != 64 or any(char not in "0123456789abcdef" for char in provider):
            raise CircuitResidencyError("provider_fingerprint must be a SHA-256 digest")
        object.__setattr__(self, "provider_fingerprint", provider)
        if (
            isinstance(self.resident_byte_budget, bool)
            or not isinstance(self.resident_byte_budget, int)
            or self.resident_byte_budget <= 0
        ):
            raise CircuitResidencyError("resident_byte_budget must be a positive integer")
        frozen_events = tuple(_freeze_json(_thaw_json(event)) for event in self.events)
        object.__setattr__(self, "events", frozen_events)
        object.__setattr__(self, "fingerprint", _canonical_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "provider_fingerprint": self.provider_fingerprint,
            "resident_byte_budget": self.resident_byte_budget,
            "events": [_thaw_json(event) for event in self.events],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}


class InMemoryWeightProvider:
    """Strict byte-bounded provider for dependency-light mechanics tests.

    Resident entries are evicted in deterministic least-recent-use order with
    the resource key as a tie breaker.  Resources held by active leases are
    pinned and cannot be evicted.  This mirrors the two important QStore
    semantics without importing its tensor implementation: bounded bytes and
    explicit lifetime.
    """

    def __init__(
        self,
        resources: Mapping[str, Any],
        *,
        resident_byte_budget: int,
        resource_bytes: Mapping[str, int] | None = None,
    ) -> None:
        if not isinstance(resources, Mapping) or not resources:
            raise CircuitResidencyError("provider resources must be a non-empty mapping")
        if (
            isinstance(resident_byte_budget, bool)
            or not isinstance(resident_byte_budget, int)
            or resident_byte_budget <= 0
        ):
            raise CircuitResidencyError("resident_byte_budget must be a positive integer")
        original_resources = dict(resources)
        self._backing = {
            key: _snapshot_payload(payload) for key, payload in original_resources.items()
        }
        supplied_sizes = {} if resource_bytes is None else dict(resource_bytes)
        unknown_sizes = set(supplied_sizes) - set(self._backing)
        if unknown_sizes:
            raise CircuitResidencyError(
                f"resource byte sizes name unknown keys {sorted(unknown_sizes)}"
            )
        sizes: dict[str, int] = {}
        for key, payload in self._backing.items():
            _text(key, "provider resource key")
            size = supplied_sizes.get(key)
            if isinstance(payload, bytes) and size is not None and size != len(payload):
                raise CircuitResidencyError(
                    f"resource {key!r} declared byte size disagrees with bytes-like payload"
                )
            if size is None and isinstance(payload, (bytes, bytearray, memoryview)):
                size = len(payload)
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise CircuitResidencyError(
                    f"resource {key!r} needs a positive byte size or a bytes-like payload"
                )
            sizes[key] = size
        self._sizes = sizes
        self._content_digests = {
            key: _payload_digest(payload) for key, payload in self._backing.items()
        }
        self.resident_byte_budget = resident_byte_budget
        self._resident: dict[str, Any] = {}
        self._last_use: dict[str, int] = {}
        self._pins: dict[str, int] = {}
        self._events: list[dict[str, Any]] = []
        self._sequence = 0
        self._active_leases = 0
        self._lock = RLock()
        self.provider_fingerprint = _canonical_digest(
            {
                "schema": "saturn-in-memory-weight-provider-v1",
                "resources": [
                    {
                        "key": key,
                        "byte_size": self._sizes[key],
                        "content_sha256": self._content_digests[key],
                    }
                    for key in sorted(self._sizes)
                ],
                "resident_byte_budget": resident_byte_budget,
            }
        )

    @property
    def resident_keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._resident))

    @property
    def resident_bytes(self) -> int:
        with self._lock:
            return sum(self._sizes[key] for key in self._resident)

    @property
    def active_lease_count(self) -> int:
        with self._lock:
            return self._active_leases

    def _validate_demand(self, weight_set: CircuitWeightSet) -> None:
        if not isinstance(weight_set, CircuitWeightSet):
            raise TypeError("weight_set must be a CircuitWeightSet")
        missing = tuple(key for key in weight_set.resource_keys if key not in self._backing)
        if missing:
            raise MissingCircuitWeightError(f"missing demanded resources: {list(missing)}")
        mismatched = tuple(
            key for key, declared_size in weight_set.resources if self._sizes[key] != declared_size
        )
        if mismatched:
            raise CircuitResidencyError(
                f"declared byte sizes disagree with provider resources: {list(mismatched)}"
            )
        if weight_set.total_bytes > self.resident_byte_budget:
            raise CircuitWeightBudgetError("demanded weight set exceeds the resident byte budget")

    def _ensure_resident(
        self, weight_set: CircuitWeightSet
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        demanded = set(weight_set.resource_keys)
        needed = sum(self._sizes[key] for key in demanded if key not in self._resident)
        loaded: list[str] = []
        evicted: list[str] = []
        while self.resident_bytes + needed > self.resident_byte_budget:
            candidates = [
                key for key in self._resident if key not in demanded and self._pins.get(key, 0) == 0
            ]
            if not candidates:
                raise CircuitWeightBudgetError(
                    "resident budget is occupied by resources pinned by active leases"
                )
            victim = min(candidates, key=lambda key: (self._last_use.get(key, -1), key))
            del self._resident[victim]
            self._last_use.pop(victim, None)
            evicted.append(victim)
        for key in weight_set.resource_keys:
            if key not in self._resident:
                self._resident[key] = self._backing[key]
                loaded.append(key)
            self._sequence += 1
            self._last_use[key] = self._sequence
        return tuple(loaded), tuple(evicted)

    def _record(
        self,
        action: str,
        weight_set: CircuitWeightSet,
        cell_id: str,
        *,
        loaded: tuple[str, ...] = (),
        evicted: tuple[str, ...] = (),
    ) -> None:
        self._events.append(
            {
                "sequence": len(self._events),
                "action": action,
                "cell_id": cell_id,
                "weight_set_fingerprint": weight_set.fingerprint,
                "resource_keys": list(weight_set.resource_keys),
                "demanded_bytes": weight_set.total_bytes,
                "loaded_keys": list(loaded),
                "evicted_keys": list(evicted),
                "resident_keys": list(self.resident_keys),
                "resident_bytes": self.resident_bytes,
            }
        )

    def prefetch(self, weight_set: CircuitWeightSet, *, cell_id: str) -> None:
        cell_id = _text(cell_id, "cell_id")
        with self._lock:
            self._validate_demand(weight_set)
            loaded, evicted = self._ensure_resident(weight_set)
            self._record("prefetch", weight_set, cell_id, loaded=loaded, evicted=evicted)

    @contextmanager
    def acquire(
        self, weight_set: CircuitWeightSet, *, cell_id: str
    ) -> Iterator[CircuitWeightLease]:
        cell_id = _text(cell_id, "cell_id")
        with self._lock:
            self._validate_demand(weight_set)
            loaded, evicted = self._ensure_resident(weight_set)
            lease_id = _canonical_digest(
                {
                    "provider": self.provider_fingerprint,
                    "weight_set": weight_set.fingerprint,
                    "cell_id": cell_id,
                    "acquire_ordinal": sum(
                        1 for event in self._events if event["action"] == "acquire"
                    ),
                }
            )
            lease = CircuitWeightLease(
                lease_id=lease_id,
                provider_fingerprint=self.provider_fingerprint,
                weight_set_fingerprint=weight_set.fingerprint,
                cell_id=cell_id,
                resources={key: self._resident[key] for key in weight_set.resource_keys},
                resident_bytes=weight_set.total_bytes,
            )
            for key in weight_set.resource_keys:
                self._pins[key] = self._pins.get(key, 0) + 1
            self._active_leases += 1
            self._record("acquire", weight_set, cell_id, loaded=loaded, evicted=evicted)
        try:
            yield lease
        finally:
            with self._lock:
                lease._release()
                for key in weight_set.resource_keys:
                    count = self._pins[key] - 1
                    if count:
                        self._pins[key] = count
                    else:
                        del self._pins[key]
                self._active_leases -= 1
                self._record("release", weight_set, cell_id)

    def receipt(self) -> CircuitWeightReceipt:
        with self._lock:
            return CircuitWeightReceipt(
                provider_fingerprint=self.provider_fingerprint,
                resident_byte_budget=self.resident_byte_budget,
                events=tuple(self._events),
            )


__all__ = [
    "CircuitResidencyError",
    "CircuitWeightBudgetError",
    "CircuitWeightLease",
    "CircuitWeightReceipt",
    "CircuitWeightSet",
    "DemandClass",
    "InMemoryWeightProvider",
    "MissingCircuitWeightError",
    "PrefetchingWeightProvider",
    "WeightProvider",
]
