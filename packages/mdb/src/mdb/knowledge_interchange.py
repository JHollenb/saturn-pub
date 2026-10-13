"""Typed knowledge-bank/operator interchange mechanics.

This module is deliberately independent of the model-backed MDB runtime.  It
is a small, finite evidence harness for asking a narrower question:

``can a typed value bank and a recipient-local operator be crossed without
silently changing their source, ABI, payload, or continuation contract?``

The module does not implement semantic routing, SPF transport, neural
execution, or a scientific pass/fail gate.  Those boundaries are represented
as separate typed fields in the receipt so that a later backend can fill them
without turning a clean metadata match into a capability claim.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4


KNOWLEDGE_INTERCHANGE_SCHEMA = "mdb-knowledge-interchange-v1"
KNOWLEDGE_BANK_SCHEMA = "mdb-knowledge-bank-v1"
KNOWLEDGE_RECORD_SCHEMA = "mdb-knowledge-record-v1"
OPERATOR_CONTRACT_SCHEMA = "mdb-knowledge-operator-contract-v1"
PANEL_RECEIPT_SCHEMA = "mdb-knowledge-interchange-receipt-v1"

VALUE = "value"
CONTEXT = "context"
RECORD_KINDS = (VALUE, CONTEXT)

RESOLVED = "resolved"
OK = "ok"
MISSING = "missing"
AMBIGUOUS = "ambiguous"
WRONG_BANK = "wrong-bank"
WRONG_OPERATOR = "wrong-operator"
STALE = "stale"
CONTEXT_AS_VALUE = "context-as-value"
RECIPIENT_MISMATCH = "recipient-mismatch"
WRONG_ROUTE = "wrong-route"
STATE_LEAKAGE = "state-leakage"
RUNTIME_BLOCKED = "runtime-blocked"
CONSUMER_ERROR = "consumer-error"

_STATUSES = frozenset(
    {
        RESOLVED,
        OK,
        MISSING,
        AMBIGUOUS,
        WRONG_BANK,
        WRONG_OPERATOR,
        STALE,
        CONTEXT_AS_VALUE,
        RECIPIENT_MISMATCH,
        WRONG_ROUTE,
        STATE_LEAKAGE,
        RUNTIME_BLOCKED,
        CONSUMER_ERROR,
    }
)


class KnowledgeInterchangeError(ValueError):
    """Raised for malformed typed interchange declarations."""


class RuntimeBlockedError(KnowledgeInterchangeError):
    """Raised inside a lowerer when it asks for an undeclared runtime value."""

    status = RUNTIME_BLOCKED


class _GuardedState(Mapping[str, Any]):
    """Read-only state view which reports undeclared reads as runtime blocks."""

    def __init__(self, values: Mapping[str, Any], allowed: Iterable[str]):
        self._values = MappingProxyType(copy.deepcopy(dict(values)))
        self._allowed = frozenset(str(item) for item in allowed)

    def __getitem__(self, key: str) -> Any:
        if key not in self._allowed:
            raise RuntimeBlockedError(f"state read {key!r} is outside the operator contract")
        try:
            return _thaw(self._values[key])
        except KeyError as exc:
            raise RuntimeBlockedError(f"declared state read {key!r} is absent") from exc

    def __iter__(self):
        return iter(sorted(key for key in self._values if key in self._allowed))

    def __len__(self) -> int:
        return sum(1 for key in self._values if key in self._allowed)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._allowed:
            raise RuntimeBlockedError(f"state read {key!r} is outside the operator contract")
        if key not in self._values:
            return default
        return _thaw(self._values[key])


def _jsonable(value: Any, *, path: str = "value") -> Any:
    """Copy JSON-safe metadata while rejecting tensor-like payloads."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise KnowledgeInterchangeError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise KnowledgeInterchangeError(f"{path} contains binary payload")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise KnowledgeInterchangeError(f"{path} has a non-string or empty key")
            result[key] = _jsonable(child, path=f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [_jsonable(child, path=f"{path}[{index}]") for index, child in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        values = [_jsonable(child, path=f"{path}[]") for child in value]
        return sorted(values, key=lambda item: json.dumps(item, sort_keys=True))
    raise KnowledgeInterchangeError(
        f"{path} has unsupported type {type(value).__name__}; use JSON-safe metadata"
    )


def canonical_json(value: Any) -> bytes:
    """Return deterministic JSON bytes for a typed receipt or payload."""

    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def content_digest(value: Any) -> str:
    """Return the SHA-256 digest used by this module's content identities."""

    return hashlib.sha256(canonical_json(value)).hexdigest()


def _text(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise KnowledgeInterchangeError(f"{label} must be a non-empty string")
    return value if allow_empty else value.strip()


def _texts(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise KnowledgeInterchangeError(f"{label} must be a sequence of strings")
    result = tuple(_text(value, label=f"{label}[{index}]") for index, value in enumerate(values))
    if len(result) != len(set(result)):
        raise KnowledgeInterchangeError(f"{label} contains duplicate entries")
    return result


def _mapping(value: Mapping[str, Any] | None, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    normalized = _jsonable(value, path=label)
    if not isinstance(normalized, dict):
        raise KnowledgeInterchangeError(f"{label} must be a mapping")
    return normalized


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON-safe payloads held by frozen records.

    A frozen dataclass is not enough when a nested dict or list remains
    caller-owned.  Mapping proxies and tuples keep record identity stable for
    the lifetime of a bank and make content digests meaningful.
    """

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(child) for child in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(child) for child in value)
    return value


def _thaw(value: Any) -> Any:
    """Make a private mutable copy of an internally frozen JSON payload."""

    if isinstance(value, Mapping):
        return {str(key): _thaw(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return [_thaw(child) for child in value]
    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    """Source/model ABI/payload identity shared by a bank or operator."""

    source_id: str
    model_abi: str
    payload_digest: str
    revision: str = "current"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, label="source_id"))
        object.__setattr__(self, "model_abi", _text(self.model_abi, label="model_abi"))
        object.__setattr__(
            self, "payload_digest", _text(self.payload_digest, label="payload_digest")
        )
        object.__setattr__(self, "revision", _text(self.revision, label="revision"))

    @property
    def fingerprint(self) -> str:
        return content_digest(self.to_payload())

    def to_payload(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "model_abi": self.model_abi,
            "payload_digest": self.payload_digest,
            "revision": self.revision,
        }

    def differs(self, other: "ArtifactBinding") -> tuple[str, ...]:
        differences = []
        for name in ("source_id", "model_abi", "payload_digest", "revision"):
            if getattr(self, name) != getattr(other, name):
                differences.append(name)
        return tuple(differences)


SourceBinding = ArtifactBinding
ModelBinding = ArtifactBinding


@dataclass(frozen=True, slots=True)
class BindingExpectation:
    """An operator's explicit expected bank binding.

    ``None`` means that dimension is intentionally generic.  A generic
    operator can therefore cross banks A and B that share source/model ABI,
    while a strict operator can bind to one exact bank payload digest.
    """

    source_id: str | None = None
    model_abi: str | None = None
    payload_digest: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        for name in ("source_id", "model_abi", "payload_digest", "revision"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, label=name))

    @classmethod
    def from_binding(cls, binding: ArtifactBinding) -> "BindingExpectation":
        return cls(
            source_id=binding.source_id,
            model_abi=binding.model_abi,
            payload_digest=binding.payload_digest,
            revision=binding.revision,
        )

    def check(self, binding: ArtifactBinding) -> tuple[str, ...]:
        return tuple(
            name
            for name in ("source_id", "model_abi", "payload_digest", "revision")
            if getattr(self, name) is not None and getattr(self, name) != getattr(binding, name)
        )

    def to_payload(self) -> dict[str, str | None]:
        return {
            "source_id": self.source_id,
            "model_abi": self.model_abi,
            "payload_digest": self.payload_digest,
            "revision": self.revision,
        }


@dataclass(frozen=True, slots=True)
class LogicalTuple:
    """An exact subject/relation/object key.

    ``qualifiers`` are part of the key.  They allow two claims about the same
    names to remain distinct by scope, date, or other explicit context.
    """

    subject: str
    relation: str
    object: Any = None
    qualifiers: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", _text(self.subject, label="tuple.subject"))
        object.__setattr__(self, "relation", _text(self.relation, label="tuple.relation"))
        object.__setattr__(self, "object", _freeze(_jsonable(self.object, path="tuple.object")))
        object.__setattr__(
            self,
            "qualifiers",
            _freeze(_mapping(self.qualifiers, label="tuple.qualifiers")),
        )

    @property
    def predicate(self) -> str:
        """Alias used by callers that call the relation a predicate."""

        return self.relation

    @property
    def value(self) -> Any:
        return self.object

    @property
    def address(self) -> "LogicalAddress":
        return LogicalAddress(self.subject, self.relation, self.qualifiers)

    def to_payload(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "qualifiers": dict(self.qualifiers),
        }

    @property
    def fingerprint(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class LogicalAddress:
    """The lookup key: subject, relation, and scope/qualifiers only.

    The value/object is deliberately absent.  A query must never supply the
    answer it is intended to retrieve.  Multiple records with this address
    and different values therefore remain visible as an ``ambiguous`` result.
    """

    subject: str
    relation: str
    qualifiers: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", _text(self.subject, label="address.subject"))
        object.__setattr__(self, "relation", _text(self.relation, label="address.relation"))
        object.__setattr__(
            self,
            "qualifiers",
            _freeze(_mapping(self.qualifiers, label="address.qualifiers")),
        )

    @property
    def predicate(self) -> str:
        return self.relation

    def to_payload(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "qualifiers": dict(self.qualifiers),
        }

    @property
    def fingerprint(self) -> str:
        return content_digest(self.to_payload())

    def matches(self, logical_tuple: LogicalTuple) -> bool:
        return self == logical_tuple.address


@dataclass(frozen=True, slots=True)
class KnowledgeQuery:
    """A value-free typed lookup request."""

    address: LogicalAddress
    kind: Literal["value", "context"] = VALUE
    revision: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.address, LogicalAddress):
            if isinstance(self.address, Mapping):
                object.__setattr__(self, "address", LogicalAddress(**self.address))
            else:
                raise KnowledgeInterchangeError("query.address must be LogicalAddress")
        if self.kind not in RECORD_KINDS:
            raise KnowledgeInterchangeError(f"query kind must be one of {RECORD_KINDS}")
        if self.revision is not None:
            object.__setattr__(self, "revision", _text(self.revision, label="query.revision"))

    def to_payload(self) -> dict[str, Any]:
        return {"address": self.address.to_payload(), "kind": self.kind, "revision": self.revision}


@dataclass(frozen=True, slots=True)
class RecordProvenance:
    """Evidence binding carried by every knowledge record."""

    source_id: str
    model_abi: str
    payload_digest: str
    source_revision: str = "current"
    locator: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, label="provenance.source_id"))
        object.__setattr__(self, "model_abi", _text(self.model_abi, label="provenance.model_abi"))
        object.__setattr__(
            self,
            "payload_digest",
            _text(self.payload_digest, label="provenance.payload_digest"),
        )
        object.__setattr__(
            self, "source_revision", _text(self.source_revision, label="provenance.source_revision")
        )
        object.__setattr__(
            self, "locator", _text(self.locator, label="provenance.locator", allow_empty=True)
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze(_mapping(self.metadata, label="provenance.metadata")),
        )

    @property
    def binding(self) -> ArtifactBinding:
        return ArtifactBinding(
            source_id=self.source_id,
            model_abi=self.model_abi,
            payload_digest=self.payload_digest,
            revision=self.source_revision,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "model_abi": self.model_abi,
            "payload_digest": self.payload_digest,
            "source_revision": self.source_revision,
            "locator": self.locator,
            "metadata": dict(self.metadata),
        }


Provenance = RecordProvenance


@dataclass(frozen=True, slots=True)
class KnowledgeRecord:
    """One immutable typed value or context record."""

    logical_tuple: LogicalTuple
    kind: Literal["value", "context"]
    revision: str
    provenance: RecordProvenance
    record_id: str = ""
    mass: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    value_digest: str = field(default="", init=False, repr=False, compare=True)

    def __post_init__(self) -> None:
        if not isinstance(self.logical_tuple, LogicalTuple):
            if isinstance(self.logical_tuple, Mapping):
                object.__setattr__(self, "logical_tuple", LogicalTuple(**self.logical_tuple))
            else:
                raise KnowledgeInterchangeError("logical_tuple must be a LogicalTuple")
        if self.kind not in RECORD_KINDS:
            raise KnowledgeInterchangeError(f"record kind must be one of {RECORD_KINDS}")
        object.__setattr__(self, "revision", _text(self.revision, label="record.revision"))
        if not isinstance(self.provenance, RecordProvenance):
            if isinstance(self.provenance, Mapping):
                object.__setattr__(self, "provenance", RecordProvenance(**self.provenance))
            else:
                raise KnowledgeInterchangeError("provenance must be RecordProvenance")
        record_id = self.record_id or content_digest(
            {
                "tuple": self.logical_tuple.to_payload(),
                "kind": self.kind,
                "revision": self.revision,
                "provenance": self.provenance.to_payload(),
            }
        )
        object.__setattr__(self, "record_id", _text(record_id, label="record.record_id"))
        if isinstance(self.mass, bool) or not isinstance(self.mass, (int, float)):
            raise KnowledgeInterchangeError("record mass must be numeric")
        if not math.isfinite(float(self.mass)) or float(self.mass) < 0:
            raise KnowledgeInterchangeError("record mass must be finite and non-negative")
        object.__setattr__(self, "mass", float(self.mass))
        object.__setattr__(
            self,
            "value_digest",
            content_digest({"kind": self.kind, "value": self.logical_tuple.object}),
        )
        object.__setattr__(self, "metadata", _freeze(_mapping(self.metadata, label="record.metadata")))

    @property
    def key(self) -> str:
        return self.logical_tuple.fingerprint

    @property
    def payload_digest(self) -> str:
        return self.provenance.payload_digest

    @property
    def value(self) -> Any:
        """The immutable value/context payload carried by this record."""

        return self.logical_tuple.object

    @property
    def value_content_digest(self) -> str:
        """Digest of actual value/context content, separate from provenance."""

        return self.value_digest

    def validate_content(self) -> bool:
        """Recompute the value digest without trusting the source declaration."""

        return self.value_content_digest == content_digest({"kind": self.kind, "value": self.value})

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": KNOWLEDGE_RECORD_SCHEMA,
            "record_id": self.record_id,
            "logical_tuple": self.logical_tuple.to_payload(),
            "kind": self.kind,
            "revision": self.revision,
            "provenance": self.provenance.to_payload(),
            "mass": self.mass,
            "value_content_digest": self.value_content_digest,
            "metadata": dict(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class PartitionAccounting:
    """Query-local selected/omitted mass; no SPF or transport semantics."""

    selected_ids: tuple[str, ...] = ()
    omitted_ids: tuple[str, ...] = ()
    selected_mass: float = 0.0
    omitted_mass: float = 0.0
    basis: str = "record-mass"

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_ids", _texts(self.selected_ids, label="selected_ids"))
        object.__setattr__(self, "omitted_ids", _texts(self.omitted_ids, label="omitted_ids"))
        for name in ("selected_mass", "omitted_mass"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise KnowledgeInterchangeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise KnowledgeInterchangeError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))
        object.__setattr__(self, "basis", _text(self.basis, label="partition.basis"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "selected_ids": list(self.selected_ids),
            "omitted_ids": list(self.omitted_ids),
            "selected_mass": self.selected_mass,
            "omitted_mass": self.omitted_mass,
            "basis": self.basis,
        }


@dataclass(frozen=True, slots=True)
class QueryLocalMerge:
    """The query-local merge boundary, intentionally separate from SPF."""

    selected_ids: tuple[str, ...]
    omitted_ids: tuple[str, ...]
    partition: PartitionAccounting
    status: str = RESOLVED

    def to_payload(self) -> dict[str, Any]:
        return {
            "selected_ids": list(self.selected_ids),
            "omitted_ids": list(self.omitted_ids),
            "partition": self.partition.to_payload(),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class Resolution:
    status: str
    query: LogicalAddress
    record: KnowledgeRecord | None = None
    candidates: tuple[KnowledgeRecord, ...] = ()
    partition: PartitionAccounting = field(default_factory=PartitionAccounting)
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in _STATUSES and self.status != RESOLVED:
            raise KnowledgeInterchangeError(f"unknown resolution status {self.status!r}")
        object.__setattr__(self, "candidates", tuple(self.candidates))

    @property
    def merge(self) -> QueryLocalMerge:
        return QueryLocalMerge(
            selected_ids=self.partition.selected_ids,
            omitted_ids=self.partition.omitted_ids,
            partition=self.partition,
            status=self.status,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "query": self.query.to_payload(),
            "record": self.record.to_payload() if self.record else None,
            "candidate_ids": [item.record_id for item in self.candidates],
            "partition": self.partition.to_payload(),
            "reason": self.reason,
        }


class KnowledgeBank:
    """A revisioned, typed bank with immutable record payloads.

    ``add`` mutates only the bank's local catalogue.  ``fork`` makes a deep
    snapshot; no run can accidentally write through a sibling branch.
    """

    def __init__(
        self,
        bank_id: str,
        binding: ArtifactBinding | None = None,
        records: Iterable[KnowledgeRecord] = (),
        *,
        source_id: str | None = None,
        model_abi: str | None = None,
        payload_digest: str | None = None,
        revision: str = "current",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.bank_id = _text(bank_id, label="bank_id")
        if binding is None:
            if source_id is None or model_abi is None or payload_digest is None:
                raise KnowledgeInterchangeError(
                    "bank binding requires binding or source_id/model_abi/payload_digest"
                )
            binding = ArtifactBinding(source_id, model_abi, payload_digest, revision)
        self.binding = binding
        self.revision = _text(revision, label="bank.revision")
        self.metadata = _freeze(_mapping(metadata, label="bank.metadata"))
        self._records: list[KnowledgeRecord] = []
        for record in records:
            self.add(record)

    @property
    def records(self) -> tuple[KnowledgeRecord, ...]:
        return tuple(self._records)

    @property
    def bank_digest(self) -> str:
        return content_digest(
            {
                "schema": KNOWLEDGE_BANK_SCHEMA,
                "bank_id": self.bank_id,
                "binding": self.binding.to_payload(),
                "revision": self.revision,
                "records": [record.to_payload() for record in self._records],
            }
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": KNOWLEDGE_BANK_SCHEMA,
            "bank_id": self.bank_id,
            "binding": self.binding.to_payload(),
            "revision": self.revision,
            "records": [record.to_payload() for record in self._records],
            "bank_digest": self.bank_digest,
            "metadata": dict(self.metadata),
        }

    def add(self, record: KnowledgeRecord) -> KnowledgeRecord:
        if not isinstance(record, KnowledgeRecord):
            raise KnowledgeInterchangeError("bank records must be KnowledgeRecord instances")
        self._records.append(record)
        return record

    def with_record(self, record: KnowledgeRecord, *, bank_id: str | None = None) -> "KnowledgeBank":
        result = self.fork(bank_id=bank_id or self.bank_id)
        result.add(record)
        return result

    def fork(self, *, bank_id: str | None = None, revision: str | None = None) -> "KnowledgeBank":
        return KnowledgeBank(
            bank_id or self.bank_id,
            self.binding,
            self._records,
            revision=revision or self.revision,
            metadata=self.metadata,
        )

    def _matching(self, query: LogicalAddress) -> list[KnowledgeRecord]:
        return [record for record in self._records if query.matches(record.logical_tuple)]

    def _record_binding_differences(self, record: KnowledgeRecord) -> tuple[str, ...]:
        differences = record.provenance.binding.differs(self.binding)
        # ``current`` is the bank-level wildcard for records carrying their
        # own explicit source revision.  An explicit bank revision remains a
        # stale-cache check.
        if self.binding.revision == "current":
            differences = tuple(name for name in differences if name != "revision")
        return differences

    def resolve(
        self,
        query: LogicalAddress | KnowledgeQuery | LogicalTuple,
        *,
        kind: Literal["value", "context"] | None = None,
        revision: str | None = None,
    ) -> Resolution:
        if isinstance(query, KnowledgeQuery):
            if kind is None:
                kind = query.kind
            if revision is None:
                revision = query.revision
            query = query.address
        elif isinstance(query, LogicalTuple):
            # A full claim tuple is accepted for compatibility, but its object
            # is intentionally ignored.  A lookup cannot be allowed to supply
            # the answer it is meant to retrieve.
            query = query.address
        elif isinstance(query, Mapping):
            query = LogicalAddress(**query)
        if not isinstance(query, LogicalAddress):
            raise KnowledgeInterchangeError("query must be LogicalAddress or KnowledgeQuery")
        kind = kind or VALUE
        if kind not in RECORD_KINDS:
            raise KnowledgeInterchangeError(f"query kind must be one of {RECORD_KINDS}")
        all_matches = self._matching(query)
        exact_kind = [record for record in all_matches if record.kind == kind]
        omitted = tuple(record.record_id for record in self._records if record not in exact_kind)
        partition = PartitionAccounting(
            selected_ids=tuple(record.record_id for record in exact_kind),
            omitted_ids=omitted,
            selected_mass=sum(record.mass for record in exact_kind),
            omitted_mass=sum(record.mass for record in self._records if record not in exact_kind),
        )
        if not all_matches:
            return Resolution(MISSING, query, partition=partition, reason="no logical tuple matched")
        if not exact_kind:
            if kind == VALUE and any(record.kind == CONTEXT for record in all_matches):
                return Resolution(
                    CONTEXT_AS_VALUE,
                    query,
                    candidates=tuple(all_matches),
                    partition=partition,
                    reason="context records cannot satisfy a value read",
                )
            return Resolution(
                MISSING,
                query,
                candidates=tuple(all_matches),
                partition=partition,
                reason=f"no {kind} record matched",
            )
        for record in exact_kind:
            binding_differences = self._record_binding_differences(record)
            if binding_differences:
                status = (
                    WRONG_BANK
                    if any(name in binding_differences for name in ("source_id", "model_abi"))
                    else STALE
                )
                return Resolution(
                    status,
                    query,
                    candidates=tuple(exact_kind),
                    partition=partition,
                    reason="record provenance mismatch: " + ", ".join(binding_differences),
                )
        if revision is not None:
            revised = [record for record in exact_kind if record.revision == revision]
            if not revised:
                return Resolution(
                    STALE,
                    query,
                    candidates=tuple(exact_kind),
                    partition=partition,
                    reason=f"requested revision {revision!r} is absent",
                )
            exact_kind = revised
        if len(exact_kind) > 1:
            return Resolution(
                AMBIGUOUS,
                query,
                candidates=tuple(exact_kind),
                partition=partition,
                reason="more than one typed record is eligible",
            )
        return Resolution(RESOLVED, query, record=exact_kind[0], candidates=tuple(exact_kind), partition=partition)

    def resolve_candidates(
        self,
        query: LogicalAddress | KnowledgeQuery | LogicalTuple,
        candidate_record_ids: Sequence[str],
        *,
        kind: Literal["value", "context"] = VALUE,
        revision: str | None = None,
    ) -> Resolution:
        """Resolve an explicit route candidate after checking its exact address.

        A spectral/route hint may nominate record ids, but it cannot define
        meaning.  The candidate's full logical address is checked against the
        value-free query before the payload is admitted.  This is the bounded
        route-control seam; no phase score or route bucket is used as a
        semantic lookup key.
        """

        if isinstance(query, KnowledgeQuery):
            kind = query.kind
            revision = revision or query.revision
            query = query.address
        elif isinstance(query, LogicalTuple):
            query = query.address
        elif isinstance(query, Mapping):
            query = LogicalAddress(**query)
        if not isinstance(query, LogicalAddress):
            raise KnowledgeInterchangeError("candidate query must be LogicalAddress or KnowledgeQuery")
        ids = _texts(candidate_record_ids, label="candidate_record_ids")
        by_id = {record.record_id: record for record in self._records}
        selected = tuple(by_id[item] for item in ids if item in by_id)
        partition = PartitionAccounting(
            selected_ids=tuple(record.record_id for record in selected),
            omitted_ids=tuple(record.record_id for record in self._records if record.record_id not in ids),
            selected_mass=sum(record.mass for record in selected),
            omitted_mass=sum(record.mass for record in self._records if record.record_id not in ids),
            basis="explicit-route-candidates",
        )
        if len(selected) != len(ids):
            missing = [item for item in ids if item not in by_id]
            return Resolution(MISSING, query, candidates=selected, partition=partition,
                              reason="route candidate record is absent: " + ", ".join(missing))
        if any(not query.matches(record.logical_tuple) for record in selected):
            return Resolution(WRONG_ROUTE, query, candidates=selected, partition=partition,
                              reason="route candidate address does not equal the query address")
        for record in selected:
            binding_differences = self._record_binding_differences(record)
            if binding_differences:
                status = (
                    WRONG_BANK
                    if any(name in binding_differences for name in ("source_id", "model_abi"))
                    else STALE
                )
                return Resolution(
                    status,
                    query,
                    candidates=selected,
                    partition=partition,
                    reason="record provenance mismatch: " + ", ".join(binding_differences),
                )
        exact_kind = tuple(record for record in selected if record.kind == kind)
        if not exact_kind and kind == VALUE and any(record.kind == CONTEXT for record in selected):
            return Resolution(CONTEXT_AS_VALUE, query, candidates=selected, partition=partition,
                              reason="route nominated context for a value read")
        if revision is not None:
            exact_kind = tuple(record for record in exact_kind if record.revision == revision)
            if not exact_kind:
                return Resolution(STALE, query, candidates=selected, partition=partition,
                                  reason=f"requested revision {revision!r} is absent")
        if len(exact_kind) != 1:
            status = MISSING if not exact_kind else AMBIGUOUS
            return Resolution(status, query, candidates=selected, partition=partition,
                              reason="route candidates did not select exactly one record")
        return Resolution(RESOLVED, query, record=exact_kind[0], candidates=selected, partition=partition)


@dataclass(frozen=True, slots=True)
class SPFOriginTransport:
    """Metadata boundary for SPF origin/page transport.

    The finite harness records this object but never uses it to select a
    record.  That separation prevents a route bucket from becoming a semantic
    key by accident.
    """

    origin: int | str | None = None
    route: str | None = None
    page_id: str | None = None
    status: str = "not-provided"

    def __post_init__(self) -> None:
        if self.route is not None:
            object.__setattr__(self, "route", _text(self.route, label="transport.route"))
        if self.page_id is not None:
            object.__setattr__(self, "page_id", _text(self.page_id, label="transport.page_id"))
        object.__setattr__(self, "status", _text(self.status, label="transport.status"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "route": self.route,
            "page_id": self.page_id,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class NumericMixing:
    """Numeric mixing metadata, kept separate from an exact law declaration."""

    law: str
    terms: tuple[str, ...] = ()
    exact: bool = False
    status: str = "not-provided"

    def __post_init__(self) -> None:
        object.__setattr__(self, "law", _text(self.law, label="numeric_mixing.law"))
        object.__setattr__(self, "terms", _texts(self.terms, label="numeric_mixing.terms"))
        object.__setattr__(self, "status", _text(self.status, label="numeric_mixing.status"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "law": self.law,
            "terms": list(self.terms),
            "exact": self.exact,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ExactLaw:
    """A separately named exact operation; no numeric mixer is implied."""

    law_id: str
    version: str = "1"
    digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "law_id", _text(self.law_id, label="exact_law.law_id"))
        object.__setattr__(self, "version", _text(self.version, label="exact_law.version"))
        if self.digest is None:
            object.__setattr__(self, "digest", content_digest(self.to_payload(include_digest=False)))

    def to_payload(self, *, include_digest: bool = True) -> dict[str, str]:
        payload = {"law_id": self.law_id, "version": self.version}
        if include_digest:
            payload["digest"] = str(self.digest)
        return payload


@dataclass(frozen=True, slots=True)
class ArmMetadata:
    """Per-arm address/page/operator/commit/consumer identity."""

    address: str
    page_id: str
    operator_id: str
    commit_id: str
    consumer_id: str

    def __post_init__(self) -> None:
        for name in ("address", "page_id", "operator_id", "commit_id", "consumer_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"arm.{name}"))

    def to_payload(self) -> dict[str, str]:
        return {
            "address": self.address,
            "page_id": self.page_id,
            "operator_id": self.operator_id,
            "commit_id": self.commit_id,
            "consumer_id": self.consumer_id,
        }


@dataclass(frozen=True, slots=True)
class RecipientLowering:
    """Declarative recipient-local lowering contract."""

    recipient_id: str
    lowerer_id: str
    reads: tuple[str, ...] = ("value",)
    writes: tuple[str, ...] = ("output",)
    output_kind: str = "native-state"

    def __post_init__(self) -> None:
        object.__setattr__(self, "recipient_id", _text(self.recipient_id, label="recipient_id"))
        object.__setattr__(self, "lowerer_id", _text(self.lowerer_id, label="lowerer_id"))
        object.__setattr__(self, "reads", _texts(self.reads, label="lowerer.reads"))
        object.__setattr__(self, "writes", _texts(self.writes, label="lowerer.writes"))
        object.__setattr__(self, "output_kind", _text(self.output_kind, label="lowerer.output_kind"))

    @property
    def fingerprint(self) -> str:
        return content_digest(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "recipient_id": self.recipient_id,
            "lowerer_id": self.lowerer_id,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "output_kind": self.output_kind,
        }


class LoweringRequest:
    """The only object a recipient lowerer receives at runtime."""

    __slots__ = (
        "record",
        "query",
        "bank_id",
        "recipient_id",
        "state",
        "partition",
        "transport",
    )

    def __init__(
        self,
        *,
        record: KnowledgeRecord,
        query: LogicalAddress,
        bank_id: str,
        recipient_id: str,
        state: Mapping[str, Any],
        allowed_state: Iterable[str] = (),
        partition: PartitionAccounting,
        transport: SPFOriginTransport,
    ) -> None:
        self.record = record
        self.query = query
        self.bank_id = bank_id
        self.recipient_id = recipient_id
        self.state = _GuardedState(state, allowed_state)
        self.partition = partition
        self.transport = transport

    @property
    def value(self) -> Any:
        return _thaw(self.record.logical_tuple.object)

    @property
    def logical_tuple(self) -> LogicalTuple:
        return self.record.logical_tuple

    @property
    def provenance(self) -> RecordProvenance:
        return self.record.provenance

    def __getattr__(self, name: str) -> Any:
        if name in {"bank", "all_records", "full_state", "weights", "model"}:
            raise RuntimeBlockedError(f"lowerer access to {name!r} is runtime-blocked")
        raise AttributeError(name)


class RecipientLowerer:
    """Callable implementation paired with a :class:`RecipientLowering`."""

    def __init__(self, contract: RecipientLowering, function: Callable[[LoweringRequest], Any]):
        if not isinstance(contract, RecipientLowering):
            raise KnowledgeInterchangeError("lowerer contract must be RecipientLowering")
        if not callable(function):
            raise KnowledgeInterchangeError("recipient lowerer function must be callable")
        self.contract = contract
        self.function = function

    def lower(self, request: LoweringRequest) -> Any:
        return self.function(request)


@dataclass(frozen=True, slots=True)
class OperatorContract:
    """Explicit operator ABI and recipient access contract."""

    operator_id: str
    mode: Literal["native", "compiled"]
    binding: ArtifactBinding
    recipient_id: str
    bank_binding: BindingExpectation | None = None
    input_kinds: tuple[str, ...] = (VALUE,)
    state_reads: tuple[str, ...] = ()
    state_writes: tuple[str, ...] = ("output",)
    lowerer_id: str = "lowerer"
    consumer_id: str = "consumer"
    continuation_horizon: int = 0
    address: str = "knowledge/value"
    page_id: str = "page-0"
    numeric_mixing: NumericMixing | None = None
    exact_law: ExactLaw | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator_id", _text(self.operator_id, label="operator_id"))
        if self.mode not in ("native", "compiled"):
            raise KnowledgeInterchangeError("operator mode must be 'native' or 'compiled'")
        if not isinstance(self.binding, ArtifactBinding):
            raise KnowledgeInterchangeError("operator binding must be ArtifactBinding")
        object.__setattr__(self, "recipient_id", _text(self.recipient_id, label="recipient_id"))
        if self.bank_binding is not None and not isinstance(self.bank_binding, BindingExpectation):
            if isinstance(self.bank_binding, ArtifactBinding):
                object.__setattr__(self, "bank_binding", BindingExpectation.from_binding(self.bank_binding))
            else:
                raise KnowledgeInterchangeError("bank_binding must be BindingExpectation")
        kinds = _texts(self.input_kinds, label="operator.input_kinds")
        if not kinds or any(kind not in RECORD_KINDS for kind in kinds):
            raise KnowledgeInterchangeError("operator input_kinds must contain value/context")
        object.__setattr__(self, "input_kinds", kinds)
        object.__setattr__(self, "state_reads", _texts(self.state_reads, label="operator.state_reads"))
        object.__setattr__(self, "state_writes", _texts(self.state_writes, label="operator.state_writes"))
        object.__setattr__(self, "lowerer_id", _text(self.lowerer_id, label="lowerer_id"))
        object.__setattr__(self, "consumer_id", _text(self.consumer_id, label="consumer_id"))
        if not isinstance(self.continuation_horizon, int) or self.continuation_horizon < 0:
            raise KnowledgeInterchangeError("continuation_horizon must be non-negative")
        object.__setattr__(self, "address", _text(self.address, label="operator.address"))
        object.__setattr__(self, "page_id", _text(self.page_id, label="operator.page_id"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, label="operator.metadata"))

    @property
    def fingerprint(self) -> str:
        return content_digest(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": OPERATOR_CONTRACT_SCHEMA,
            "operator_id": self.operator_id,
            "mode": self.mode,
            "binding": self.binding.to_payload(),
            "recipient_id": self.recipient_id,
            "bank_binding": self.bank_binding.to_payload() if self.bank_binding else None,
            "input_kinds": list(self.input_kinds),
            "state_reads": list(self.state_reads),
            "state_writes": list(self.state_writes),
            "lowerer_id": self.lowerer_id,
            "consumer_id": self.consumer_id,
            "continuation_horizon": self.continuation_horizon,
            "address": self.address,
            "page_id": self.page_id,
            "numeric_mixing": self.numeric_mixing.to_payload() if self.numeric_mixing else None,
            "exact_law": self.exact_law.to_payload() if self.exact_law else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class LoweredResult:
    """Output from a recipient lowerer, with explicit state/trace channels."""

    output: Any
    state_updates: Mapping[str, Any] = field(default_factory=dict)
    own_predicted_state: Any | None = None
    continuation_trace: tuple[Mapping[str, Any], ...] = ()
    claims: tuple[Mapping[str, Any], ...] = ()
    numeric_mixing: NumericMixing | None = None
    exact_law: ExactLaw | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_updates", _mapping(self.state_updates, label="state_updates"))
        object.__setattr__(
            self,
            "continuation_trace",
            tuple(_mapping(item, label="continuation_trace") for item in self.continuation_trace),
        )
        object.__setattr__(self, "claims", tuple(_mapping(item, label="claims") for item in self.claims))


@dataclass(frozen=True, slots=True)
class ConsumerInput:
    record: KnowledgeRecord
    lowered_output: Any
    execution_state: Any
    execution_state_source: str
    continuation_trace: tuple[Mapping[str, Any], ...]
    bank_id: str
    operator_id: str
    query: LogicalAddress

    def to_payload(self) -> dict[str, Any]:
        return {
            "record_id": self.record.record_id,
            "lowered_output": self.lowered_output,
            "execution_state": self.execution_state,
            "execution_state_source": self.execution_state_source,
            "continuation_trace": [dict(item) for item in self.continuation_trace],
            "bank_id": self.bank_id,
            "operator_id": self.operator_id,
            "query": self.query.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class SignedConsumerResult:
    """A deterministic content signature over one consumer observation."""

    signer: str
    payload: Mapping[str, Any]
    signature: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "signer", _text(self.signer, label="consumer_result.signer"))
        object.__setattr__(
            self,
            "payload",
            _freeze(_mapping(self.payload, label="consumer_result.payload")),
        )
        object.__setattr__(self, "signature", _text(self.signature, label="consumer_result.signature"))

    @property
    def content_sha256(self) -> str:
        """Alias naming the deterministic content checksum explicitly."""

        return self.signature

    @classmethod
    def sign(cls, payload: Mapping[str, Any], *, signer: str) -> "SignedConsumerResult":
        normalized = _mapping(payload, label="consumer_result.payload")
        signer = _text(signer, label="consumer_result.signer")
        signature = content_digest({"signer": signer, "payload": normalized})
        return cls(signer, normalized, signature)

    def to_payload(self) -> dict[str, Any]:
        return _jsonable({"signer": self.signer, "payload": dict(self.payload), "signature": self.signature})


# The descriptive name avoids implying authenticity or authorization.  Keep
# the shorter alias for callers of the first mechanics draft.
ContentSignedConsumerResult = SignedConsumerResult


@dataclass(frozen=True, slots=True)
class OperatorExecution:
    status: str
    resolution: Resolution
    output: Any | None = None
    reason: str = ""
    own_predicted_state: Any | None = None
    supplied_execution: Any | None = None
    effective_state: Any | None = None
    execution_state_source: str = "none"
    continuation_trace: tuple[Mapping[str, Any], ...] = ()
    claims: tuple[Mapping[str, Any], ...] = ()
    signed_consumer_result: SignedConsumerResult | None = None
    arm: ArmMetadata | None = None
    query_local_merge: QueryLocalMerge | None = None
    spf_origin_transport: SPFOriginTransport | None = None
    numeric_mixing: NumericMixing | None = None
    exact_law: ExactLaw | None = None
    commit: str = "retained"

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise KnowledgeInterchangeError(f"unknown execution status {self.status!r}")
        object.__setattr__(self, "continuation_trace", tuple(self.continuation_trace))
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "commit", _text(self.commit, label="commit"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "resolution": self.resolution.to_payload(),
            "output": self.output,
            "reason": self.reason,
            "own_predicted_state": self.own_predicted_state,
            "supplied_execution": self.supplied_execution,
            "effective_state": self.effective_state,
            "execution_state_source": self.execution_state_source,
            "continuation_trace": [dict(item) for item in self.continuation_trace],
            "claims": [dict(item) for item in self.claims],
            "signed_consumer_result": (
                self.signed_consumer_result.to_payload() if self.signed_consumer_result else None
            ),
            "arm": self.arm.to_payload() if self.arm else None,
            "query_local_merge": self.query_local_merge.to_payload() if self.query_local_merge else None,
            "spf_origin_transport": (
                self.spf_origin_transport.to_payload() if self.spf_origin_transport else None
            ),
            "numeric_mixing": self.numeric_mixing.to_payload() if self.numeric_mixing else None,
            "exact_law": self.exact_law.to_payload() if self.exact_law else None,
            "commit": self.commit,
        }


_MISSING_ARGUMENT = object()


def _call(fn: Callable[..., Any], value: Any, second: Any = _MISSING_ARGUMENT) -> Any:
    """Call one of the deliberately small one-argument callbacks.

    A two-argument callback is accepted for ergonomic finite fixtures when the
    caller supplies ``second``.  We do not catch callback exceptions here
    because a lowerer exception is evidence for the row.
    """

    try:
        signature = inspect.signature(fn)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) == 0:
            return fn()
        if len(positional) >= 2 and second is not _MISSING_ARGUMENT:
            return fn(value, second)
    except (TypeError, ValueError):
        pass
    return fn(value)


class KnowledgeOperator:
    """Executable recipient-local operator with a declared contract."""

    def __init__(
        self,
        contract: OperatorContract,
        lowerer: RecipientLowerer | Callable[[LoweringRequest], Any],
        *,
        consumer: Callable[[ConsumerInput], Any] | None = None,
        continuation: Callable[[ConsumerInput], Any] | None = None,
        claim_extractor: Callable[[Any], Iterable[Mapping[str, Any]]] | None = None,
    ) -> None:
        if not isinstance(contract, OperatorContract):
            raise KnowledgeInterchangeError("operator contract must be OperatorContract")
        if isinstance(lowerer, RecipientLowerer):
            wrapped = lowerer
        else:
            wrapped = RecipientLowerer(
                RecipientLowering(
                    recipient_id=contract.recipient_id,
                    lowerer_id=contract.lowerer_id,
                    reads=("value", *contract.state_reads),
                    writes=contract.state_writes,
                ),
                lowerer,
            )
        self.contract = contract
        self.lowerer = wrapped
        self.consumer = consumer
        self.continuation = continuation
        self.claim_extractor = claim_extractor

    @property
    def operator_id(self) -> str:
        return self.contract.operator_id

    def _blocked(
        self,
        status: str,
        resolution: Resolution,
        *,
        reason: str,
        state: Mapping[str, Any],
        supplied_execution: Any,
        commit: str,
        transport: SPFOriginTransport,
    ) -> OperatorExecution:
        return OperatorExecution(
            status=status,
            resolution=resolution,
            reason=reason,
            supplied_execution=copy.deepcopy(supplied_execution),
            effective_state=copy.deepcopy(dict(state)),
            query_local_merge=resolution.merge,
            spf_origin_transport=transport,
            arm=ArmMetadata(
                address=self.contract.address,
                page_id=self.contract.page_id,
                operator_id=self.contract.operator_id,
                commit_id=commit,
                consumer_id=self.contract.consumer_id,
            ),
            commit=commit,
        )

    def execute(
        self,
        bank: KnowledgeBank,
        query: LogicalAddress | KnowledgeQuery | LogicalTuple,
        *,
        state: Mapping[str, Any] | None = None,
        expected_operator_id: str | None = None,
        expected_recipient: str | None = None,
        expected_mode: str | None = None,
        revision: str | None = None,
        candidate_record_ids: Sequence[str] | None = None,
        supplied_execution: Any | None = None,
        transport: SPFOriginTransport | None = None,
        commit: bool = False,
        consumer: Callable[[ConsumerInput], Any] | None = None,
        claim_extractor: Callable[[Any], Iterable[Mapping[str, Any]]] | None = None,
    ) -> OperatorExecution:
        if not isinstance(bank, KnowledgeBank):
            raise KnowledgeInterchangeError("operator requires KnowledgeBank")
        if isinstance(query, KnowledgeQuery):
            if revision is None:
                revision = query.revision
            query = query.address
        elif isinstance(query, LogicalTuple):
            query = query.address
        elif isinstance(query, Mapping):
            query = LogicalAddress(**query)
        if not isinstance(query, LogicalAddress):
            raise KnowledgeInterchangeError("operator query must be LogicalAddress or KnowledgeQuery")
        state = _mapping(state, label="execution.state")
        transport = transport or SPFOriginTransport()
        commit_label = "committed" if commit else "retained"
        unresolved = Resolution(MISSING, query, reason="operator preflight did not resolve a query")
        if expected_operator_id is not None and expected_operator_id != self.contract.operator_id:
            return self._blocked(
                WRONG_OPERATOR,
                unresolved,
                reason="operator id does not match the requested arm",
                state=state,
                supplied_execution=supplied_execution,
                commit=commit_label,
                transport=transport,
            )
        if expected_mode is not None and expected_mode != self.contract.mode:
            return self._blocked(
                WRONG_OPERATOR,
                unresolved,
                reason="operator mode does not match the requested arm",
                state=state,
                supplied_execution=supplied_execution,
                commit=commit_label,
                transport=transport,
            )
        if expected_recipient is not None and expected_recipient != self.contract.recipient_id:
            return self._blocked(
                RECIPIENT_MISMATCH,
                unresolved,
                reason="recipient id does not match the requested consumer",
                state=state,
                supplied_execution=supplied_execution,
                commit=commit_label,
                transport=transport,
            )
        if self.lowerer.contract.recipient_id != self.contract.recipient_id:
            return self._blocked(
                RECIPIENT_MISMATCH,
                unresolved,
                reason="lowerer recipient differs from operator recipient",
                state=state,
                supplied_execution=supplied_execution,
                commit=commit_label,
                transport=transport,
            )
        if self.lowerer.contract.lowerer_id != self.contract.lowerer_id:
            return self._blocked(
                WRONG_OPERATOR,
                unresolved,
                reason="lowerer implementation id is not bound to the contract",
                state=state,
                supplied_execution=supplied_execution,
                commit=commit_label,
                transport=transport,
            )
        if self.contract.bank_binding is not None:
            differences = self.contract.bank_binding.check(bank.binding)
            if differences:
                status = WRONG_BANK if any(name in differences for name in ("source_id", "model_abi")) else STALE
                return self._blocked(
                    status,
                    unresolved,
                    reason="bank binding mismatch: " + ", ".join(differences),
                    state=state,
                    supplied_execution=supplied_execution,
                    commit=commit_label,
                    transport=transport,
                )
        input_kind = VALUE if VALUE in self.contract.input_kinds else self.contract.input_kinds[0]
        resolution = (
            bank.resolve_candidates(
                query,
                candidate_record_ids,
                kind=input_kind,
                revision=revision,
            )
            if candidate_record_ids is not None
            else bank.resolve(query, kind=input_kind, revision=revision)
        )
        if resolution.status != RESOLVED:
            return self._blocked(
                resolution.status,
                resolution,
                reason=resolution.reason,
                state=state,
                supplied_execution=supplied_execution,
                commit=commit_label,
                transport=transport,
            )
        assert resolution.record is not None
        if resolution.record.kind not in self.contract.input_kinds:
            status = CONTEXT_AS_VALUE if resolution.record.kind == CONTEXT else MISSING
            return self._blocked(
                status,
                resolution,
                reason="record kind is outside operator input_kinds",
                state=state,
                supplied_execution=supplied_execution,
                commit=commit_label,
                transport=transport,
            )

        request = LoweringRequest(
            record=resolution.record,
            query=query,
            bank_id=bank.bank_id,
            recipient_id=self.contract.recipient_id,
            state=state,
            allowed_state=self.contract.state_reads,
            partition=resolution.partition,
            transport=transport,
        )
        before_state = canonical_json(state)
        try:
            raw = self.lowerer.lower(request)
        except RuntimeBlockedError as exc:
            return self._blocked(
                RUNTIME_BLOCKED,
                resolution,
                reason=str(exc),
                state=state,
                supplied_execution=supplied_execution,
                commit=commit_label,
                transport=transport,
            )
        except Exception as exc:  # callback failure belongs to this row
            return self._blocked(
                RUNTIME_BLOCKED,
                resolution,
                reason=f"lowerer raised {type(exc).__name__}: {exc}",
                state=state,
                supplied_execution=supplied_execution,
                commit=commit_label,
                transport=transport,
            )
        if canonical_json(state) != before_state:
            return self._blocked(
                STATE_LEAKAGE,
                resolution,
                reason="lowerer mutated caller-owned state",
                state=state,
                supplied_execution=supplied_execution,
                commit=commit_label,
                transport=transport,
            )
        if isinstance(raw, LoweredResult):
            lowered = raw
        else:
            lowered = LoweredResult(output=raw)
        undeclared = set(lowered.state_updates) - set(self.contract.state_writes)
        if undeclared:
            return self._blocked(
                STATE_LEAKAGE,
                resolution,
                reason="lowerer wrote undeclared state: " + ", ".join(sorted(undeclared)),
                state=state,
                supplied_execution=supplied_execution,
                commit=commit_label,
                transport=transport,
            )

        trace = list(lowered.continuation_trace)
        own_predicted_state = copy.deepcopy(lowered.own_predicted_state)
        if self.continuation is not None and self.contract.continuation_horizon:
            continuation_input = ConsumerInput(
                record=resolution.record,
                lowered_output=lowered.output,
                execution_state=own_predicted_state,
                execution_state_source="own-predicted-state",
                continuation_trace=tuple(trace),
                bank_id=bank.bank_id,
                operator_id=self.contract.operator_id,
                query=query,
            )
            try:
                continued = _call(self.continuation, continuation_input)
                if isinstance(continued, LoweredResult):
                    own_predicted_state = continued.own_predicted_state
                    trace.extend(continued.continuation_trace)
                    if continued.output is not None:
                        lowered = LoweredResult(
                            output=continued.output,
                            state_updates=lowered.state_updates,
                            own_predicted_state=own_predicted_state,
                            continuation_trace=tuple(trace),
                            claims=tuple((*lowered.claims, *continued.claims)),
                            numeric_mixing=continued.numeric_mixing or lowered.numeric_mixing,
                            exact_law=continued.exact_law or lowered.exact_law,
                        )
                elif continued is not None:
                    own_predicted_state = copy.deepcopy(continued)
                    trace.append({"source": "operator-continuation", "state": own_predicted_state})
            except RuntimeBlockedError as exc:
                return self._blocked(
                    RUNTIME_BLOCKED,
                    resolution,
                    reason=str(exc),
                    state=state,
                    supplied_execution=supplied_execution,
                    commit=commit_label,
                    transport=transport,
                )
            except Exception as exc:
                return self._blocked(
                    RUNTIME_BLOCKED,
                    resolution,
                    reason=f"continuation raised {type(exc).__name__}: {exc}",
                    state=state,
                    supplied_execution=supplied_execution,
                    commit=commit_label,
                    transport=transport,
                )
        if supplied_execution is not None:
            effective_state = copy.deepcopy(supplied_execution)
            source = "supplied-execution"
        elif own_predicted_state is not None:
            effective_state = copy.deepcopy(own_predicted_state)
            source = "own-predicted-state"
        else:
            effective_state = copy.deepcopy(state)
            source = "base-state"
        consumer_input = ConsumerInput(
            record=resolution.record,
            lowered_output=lowered.output,
            execution_state=effective_state,
            execution_state_source=source,
            continuation_trace=tuple(trace),
            bank_id=bank.bank_id,
            operator_id=self.contract.operator_id,
            query=query,
        )
        active_consumer = consumer or self.consumer
        try:
            observation = (
                _call(active_consumer, consumer_input) if active_consumer is not None else lowered.output
            )
            observation = _jsonable(observation, path="consumer.observation")
        except RuntimeBlockedError as exc:
            return self._blocked(
                RUNTIME_BLOCKED,
                resolution,
                reason=str(exc),
                state=state,
                supplied_execution=supplied_execution,
                commit=commit_label,
                transport=transport,
            )
        except Exception as exc:
            return self._blocked(
                CONSUMER_ERROR,
                resolution,
                reason=f"consumer raised {type(exc).__name__}: {exc}",
                state=state,
                supplied_execution=supplied_execution,
                commit=commit_label,
                transport=transport,
            )
        extract = claim_extractor or self.claim_extractor
        claims = list(lowered.claims)
        if extract is not None:
            try:
                claims.extend(_mapping(item, label="claim") for item in extract(observation))
            except Exception as exc:
                return self._blocked(
                    CONSUMER_ERROR,
                    resolution,
                    reason=f"claim extractor raised {type(exc).__name__}: {exc}",
                    state=state,
                    supplied_execution=supplied_execution,
                    commit=commit_label,
                    transport=transport,
                )
        signed = SignedConsumerResult.sign(
            {
                "observation": observation,
                "record_id": resolution.record.record_id,
                "bank_id": bank.bank_id,
                "operator_id": self.contract.operator_id,
                "execution_state_source": source,
                "execution_state": effective_state,
                "continuation_trace": trace,
            },
            signer=self.contract.consumer_id,
        )
        return OperatorExecution(
            status=OK,
            resolution=resolution,
            output=observation,
            own_predicted_state=own_predicted_state,
            supplied_execution=copy.deepcopy(supplied_execution),
            effective_state=effective_state,
            execution_state_source=source,
            continuation_trace=tuple(trace),
            claims=tuple(claims),
            signed_consumer_result=signed,
            arm=ArmMetadata(
                address=self.contract.address,
                page_id=self.contract.page_id,
                operator_id=self.contract.operator_id,
                commit_id=("commit" if commit else "retain"),
                consumer_id=self.contract.consumer_id,
            ),
            query_local_merge=resolution.merge,
            spf_origin_transport=transport,
            numeric_mixing=lowered.numeric_mixing or self.contract.numeric_mixing,
            exact_law=lowered.exact_law or self.contract.exact_law,
            commit=commit_label,
        )


@dataclass(frozen=True, slots=True)
class BranchCut:
    name: str
    parent: str | None
    state: Mapping[str, Any]
    digest: str
    source: str = "standalone"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="cut.name"))
        if self.parent is not None:
            object.__setattr__(self, "parent", _text(self.parent, label="cut.parent"))
        object.__setattr__(self, "state", _freeze(_mapping(self.state, label="cut.state")))
        object.__setattr__(self, "digest", _text(self.digest, label="cut.digest"))
        object.__setattr__(self, "source", _text(self.source, label="cut.source"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent": self.parent,
            "state": dict(self.state),
            "digest": self.digest,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class RowReceipt:
    row_id: str
    bank_id: str
    operator_id: str
    operator_mode: str
    query: LogicalAddress
    execution: OperatorExecution

    @property
    def status(self) -> str:
        return self.execution.status

    def to_payload(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "bank_id": self.bank_id,
            "operator_id": self.operator_id,
            "operator_mode": self.operator_mode,
            "query": self.query.to_payload(),
            "execution": self.execution.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class PanelReceipt:
    """Finite raw rows plus orthogonal evidence statuses."""

    panel_id: str
    rows: tuple[RowReceipt, ...]
    mechanics_status: str
    instrument_status: str
    trend_status: str
    terminal_status: str
    branch: BranchCut
    limitations: tuple[str, ...] = (
        "model-free finite harness",
        "SPF origin transport is metadata-only",
        "numeric mixing and exact laws are separate optional declarations",
    )
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.receipt_sha256:
            object.__setattr__(
                self,
                "receipt_sha256",
                content_digest(self.to_payload(include_hash=False)),
            )

    @property
    def status_counts(self) -> dict[str, int]:
        return dict(Counter(row.status for row in self.rows))

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": PANEL_RECEIPT_SCHEMA,
            "panel_id": self.panel_id,
            "rows": [row.to_payload() for row in self.rows],
            "status_counts": self.status_counts,
            "mechanics_status": self.mechanics_status,
            "instrument_status": self.instrument_status,
            "trend_status": self.trend_status,
            "terminal_status": self.terminal_status,
            "branch": self.branch.to_payload(),
            "limitations": list(self.limitations),
        }
        if include_hash:
            payload["receipt_sha256"] = self.receipt_sha256
        return _jsonable(payload)


@dataclass(frozen=True, slots=True)
class ChainStepReceipt:
    """One executed read/consumer step in a bounded own-state chain."""

    step_index: int
    query: LogicalAddress
    row: RowReceipt
    next_query: LogicalAddress | None = None
    stop_reason: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return _jsonable({
            "step_index": self.step_index,
            "query": self.query.to_payload(),
            "row": self.row.to_payload(),
            "next_query": self.next_query.to_payload() if self.next_query else None,
            "stop_reason": self.stop_reason,
        })


@dataclass(frozen=True, slots=True)
class ChainReceipt:
    """Per-step chain evidence; no closed-loop capability verdict."""

    panel_id: str
    steps: tuple[ChainStepReceipt, ...]
    status: str
    stop_reason: str
    mechanics_status: str
    instrument_status: str
    trend_status: str
    terminal_status: str = "not-requested"
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.receipt_sha256:
            object.__setattr__(
                self,
                "receipt_sha256",
                content_digest(self.to_payload(include_hash=False)),
            )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": PANEL_RECEIPT_SCHEMA,
            "panel_id": self.panel_id,
            "steps": [step.to_payload() for step in self.steps],
            "status": self.status,
            "stop_reason": self.stop_reason,
            "mechanics_status": self.mechanics_status,
            "instrument_status": self.instrument_status,
            "trend_status": self.trend_status,
            "terminal_status": self.terminal_status,
        }
        if include_hash:
            payload["receipt_sha256"] = self.receipt_sha256
        return _jsonable(payload)


class NativeResearchSessionBridge:
    """Optional duck-typed bridge for a caller-owned NativeResearchSession.

    The import is intentionally absent.  A bridge is useful only when the
    caller already owns a session and wants its native tensor cut to be the
    parent/branch authority; the model-free panel remains usable without it.
    """

    def __init__(self, session: Any):
        required = ("capture", "restore")
        if any(not callable(getattr(session, name, None)) for name in required):
            raise KnowledgeInterchangeError("session bridge requires capture and restore methods")
        try:
            capture_parameters = inspect.signature(session.capture).parameters
            restore_parameters = inspect.signature(session.restore).parameters
        except (TypeError, ValueError) as exc:
            raise KnowledgeInterchangeError("session bridge could not inspect public session methods") from exc
        if not {"name", "state", "parent", "metadata"}.issubset(capture_parameters):
            raise KnowledgeInterchangeError(
                "session capture must expose NativeResearchSession(name, state, parent, metadata)"
            )
        if "name" not in restore_parameters:
            raise KnowledgeInterchangeError("session restore must expose NativeResearchSession(name)")
        self.session = session

    def capture(self, name: str, state: Mapping[str, Any], *, parent: str = "parent") -> Any:
        return self.session.capture(name, state, parent=parent, metadata={"interchange": True})

    def restore(self, name: str) -> Any:
        return self.session.restore(name)


class InterchangePanel:
    """Generic finite bank/operator crossing with explicit branch custody."""

    def __init__(
        self,
        banks: Mapping[str, KnowledgeBank] | Sequence[KnowledgeBank],
        operators: Mapping[str, KnowledgeOperator] | Sequence[KnowledgeOperator],
        *,
        initial_state: Mapping[str, Any] | None = None,
        panel_id: str | None = None,
        session: NativeResearchSessionBridge | None = None,
    ) -> None:
        if isinstance(banks, Mapping):
            bank_items = list(banks.items())
        else:
            bank_items = [(bank.bank_id, bank) for bank in banks]
        if isinstance(operators, Mapping):
            operator_items = list(operators.items())
        else:
            operator_items = [(operator.operator_id, operator) for operator in operators]
        if not bank_items or not operator_items:
            raise KnowledgeInterchangeError("panel requires at least one bank and one operator")
        self.banks = dict(bank_items)
        self.operators = dict(operator_items)
        if len(self.banks) != len(bank_items) or len(self.operators) != len(operator_items):
            raise KnowledgeInterchangeError("bank and operator ids must be unique")
        if any(not isinstance(bank, KnowledgeBank) for bank in self.banks.values()):
            raise KnowledgeInterchangeError("panel banks must be KnowledgeBank instances")
        if any(not isinstance(operator, KnowledgeOperator) for operator in self.operators.values()):
            raise KnowledgeInterchangeError("panel operators must be KnowledgeOperator instances")
        self.panel_id = panel_id or "panel-" + uuid4().hex
        self.session = session
        state = _mapping(initial_state, label="panel.initial_state")
        self._cuts: dict[str, BranchCut] = {
            "parent": BranchCut(
                name="parent",
                parent=None,
                state=state,
                digest=content_digest(state),
                source="native-research-session" if session else "standalone",
            )
        }
        self._current = "parent"

    @property
    def cuts(self) -> Mapping[str, BranchCut]:
        return MappingProxyType(dict(self._cuts))

    def common_parent(self) -> BranchCut:
        return self._cuts["parent"]

    def fork(
        self,
        name: str,
        *,
        parent: str = "parent",
        updates: Mapping[str, Any] | None = None,
    ) -> BranchCut:
        name = _text(name, label="cut.name")
        if name in self._cuts:
            raise KnowledgeInterchangeError(f"cut {name!r} already exists")
        if parent not in self._cuts:
            raise KnowledgeInterchangeError(f"unknown parent cut {parent!r}")
        state = dict(self._cuts[parent].state)
        state.update(_mapping(updates, label="cut.updates"))
        if self.session is not None:
            self.session.capture(name, state, parent=parent)
        cut = BranchCut(
            name=name,
            parent=parent,
            state=state,
            digest=content_digest(state),
            source="native-research-session" if self.session else "standalone",
        )
        self._cuts[name] = cut
        return cut

    def restore(self, name: str) -> BranchCut:
        if name not in self._cuts:
            raise KnowledgeInterchangeError(f"unknown cut {name!r}")
        cut = self._cuts[name]
        if self.session is not None:
            self.session.restore(name)
        self._current = name
        return cut

    def run_matrix(
        self,
        query: LogicalAddress | KnowledgeQuery | LogicalTuple,
        *,
        cut: str = "parent",
        recipient: str | None = None,
        operator_ids: Sequence[str] | None = None,
        bank_ids: Sequence[str] | None = None,
        supplied_execution: Any | None = None,
        transport: SPFOriginTransport | None = None,
        commit: bool = False,
        revision: str | None = None,
        candidate_record_ids: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
        consumer: Callable[[ConsumerInput], Any] | None = None,
        claim_extractor: Callable[[Any], Iterable[Mapping[str, Any]]] | None = None,
    ) -> PanelReceipt:
        if cut not in self._cuts:
            raise KnowledgeInterchangeError(f"unknown cut {cut!r}")
        if isinstance(query, KnowledgeQuery):
            query = query.address
        elif isinstance(query, LogicalTuple):
            query = query.address
        elif isinstance(query, Mapping):
            query = LogicalAddress(**query)
        if not isinstance(query, LogicalAddress):
            raise KnowledgeInterchangeError("panel query must be LogicalAddress or KnowledgeQuery")
        selected_banks = [self.banks[name] for name in (bank_ids or tuple(self.banks))]
        selected_operators = [self.operators[name] for name in (operator_ids or tuple(self.operators))]
        rows: list[RowReceipt] = []
        row_number = 0
        branch = self._cuts[cut]
        for bank in selected_banks:
            for operator in selected_operators:
                row_candidates = (
                    candidate_record_ids.get(bank.bank_id)
                    if isinstance(candidate_record_ids, Mapping)
                    else candidate_record_ids
                )
                execution = operator.execute(
                    bank,
                    query,
                    state=branch.state,
                    expected_operator_id=operator.operator_id,
                    expected_recipient=recipient,
                    supplied_execution=supplied_execution,
                    transport=transport,
                    commit=commit,
                    revision=revision,
                    candidate_record_ids=row_candidates,
                    consumer=consumer,
                    claim_extractor=claim_extractor,
                )
                rows.append(
                    RowReceipt(
                        row_id=f"row-{row_number:04d}",
                        bank_id=bank.bank_id,
                        operator_id=operator.operator_id,
                        operator_mode=operator.contract.mode,
                        query=query,
                        execution=execution,
                    )
                )
                row_number += 1
        # These labels describe evidence collection only.  They do not reduce
        # heterogeneous rows to a scientific boolean verdict.
        mechanics = "valid" if rows else "incomplete"
        observed = sum(1 for row in rows if row.execution.signed_consumer_result is not None)
        blocked = sum(1 for row in rows if row.status in {RUNTIME_BLOCKED, STATE_LEAKAGE})
        instrument = "observed" if observed else "inapplicable"
        if blocked and observed:
            instrument = "mixed"
        trend = "observed" if observed else "no-observation"
        terminal = "not-requested"
        return PanelReceipt(
            panel_id=self.panel_id,
            rows=tuple(rows),
            mechanics_status=mechanics,
            instrument_status=instrument,
            trend_status=trend,
            terminal_status=terminal,
            branch=branch,
        )

    def run_chain(
        self,
        start_query: LogicalAddress | KnowledgeQuery | LogicalTuple,
        *,
        bank_id: str | None = None,
        operator_ids: Sequence[str] | None = None,
        next_query: Callable[..., LogicalAddress | KnowledgeQuery | LogicalTuple | None],
        max_steps: int = 2,
        cut: str = "parent",
        recipient: str | None = None,
        supplied_executions: Sequence[Any] | None = None,
        transport: SPFOriginTransport | None = None,
        commit: bool = False,
        revision: str | None = None,
        candidate_record_ids: Sequence[Sequence[str] | None] | None = None,
        consumer: Callable[[ConsumerInput], Any] | None = None,
        claim_extractor: Callable[[Any], Iterable[Mapping[str, Any]]] | None = None,
    ) -> ChainReceipt:
        """Run a bounded read/own-state/next-address loop.

        ``next_query`` receives only the previous operator's own predicted
        state (and optionally its completed execution).  A supplied execution
        can affect the current consumer, but it is never used to invent the
        next address.  Every step keeps its full resolution and consumer
        receipt.  A missing or ambiguous next read is retained as the terminal
        step rather than collapsed into a chain-level success flag.
        """

        if max_steps < 1:
            raise KnowledgeInterchangeError("max_steps must be positive")
        if cut not in self._cuts:
            raise KnowledgeInterchangeError(f"unknown cut {cut!r}")
        if bank_id is None:
            bank_id = next(iter(self.banks))
        if bank_id not in self.banks:
            raise KnowledgeInterchangeError(f"unknown chain bank {bank_id!r}")
        if not callable(next_query):
            raise KnowledgeInterchangeError("next_query must be callable")
        if isinstance(start_query, KnowledgeQuery):
            query: LogicalAddress | None = start_query.address
            if revision is None:
                revision = start_query.revision
        elif isinstance(start_query, LogicalTuple):
            query = start_query.address
        elif isinstance(start_query, LogicalAddress):
            query = start_query
        elif isinstance(start_query, Mapping):
            query = LogicalAddress(**start_query)
        else:
            raise KnowledgeInterchangeError("start_query must be LogicalAddress or KnowledgeQuery")

        operator_names = tuple(operator_ids or (next(iter(self.operators)),))
        for name in operator_names:
            if name not in self.operators:
                raise KnowledgeInterchangeError(f"unknown chain operator {name!r}")
        branch = self._cuts[cut]
        current_state: Any = dict(branch.state)
        steps: list[ChainStepReceipt] = []
        chain_status = OK
        stop_reason = "horizon-exhausted"
        for index in range(max_steps):
            if query is None:
                chain_status = MISSING
                stop_reason = "no-next-address"
                break
            operator_name = operator_names[index] if index < len(operator_names) else operator_names[-1]
            operator = self.operators[operator_name]
            supplied = (
                supplied_executions[index]
                if supplied_executions is not None and index < len(supplied_executions)
                else None
            )
            candidate_ids = (
                candidate_record_ids[index]
                if candidate_record_ids is not None and index < len(candidate_record_ids)
                else None
            )
            execution = operator.execute(
                self.banks[bank_id],
                query,
                state=current_state if isinstance(current_state, Mapping) else {"own_state": current_state},
                expected_operator_id=operator.operator_id,
                expected_recipient=recipient,
                supplied_execution=supplied,
                transport=transport,
                commit=commit,
                revision=revision,
                candidate_record_ids=candidate_ids,
                consumer=consumer,
                claim_extractor=claim_extractor,
            )
            row = RowReceipt(
                row_id=f"chain-{index:04d}",
                bank_id=bank_id,
                operator_id=operator.operator_id,
                operator_mode=operator.contract.mode,
                query=query,
                execution=execution,
            )
            if execution.status != OK:
                stop_reason = execution.reason or execution.status
                chain_status = execution.status
                steps.append(ChainStepReceipt(index, query, row, stop_reason=stop_reason))
                break
            own_state = execution.own_predicted_state
            if own_state is None:
                stop_reason = "no-own-predicted-state"
                chain_status = RUNTIME_BLOCKED
                steps.append(ChainStepReceipt(index, query, row, stop_reason=stop_reason))
                break
            steps.append(ChainStepReceipt(index, query, row))
            try:
                signature = inspect.signature(next_query)
                positional = [
                    parameter
                    for parameter in signature.parameters.values()
                    if parameter.kind
                    in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                if len(positional) == 0:
                    candidate_query = next_query()
                elif len(positional) == 1:
                    candidate_query = next_query(copy.deepcopy(own_state))
                else:
                    candidate_query = next_query(copy.deepcopy(own_state), execution)
            except RuntimeBlockedError as exc:
                chain_status = RUNTIME_BLOCKED
                stop_reason = str(exc)
                steps[-1] = ChainStepReceipt(index, query, row, stop_reason=stop_reason)
                break
            except Exception as exc:
                chain_status = RUNTIME_BLOCKED
                stop_reason = f"next_query raised {type(exc).__name__}: {exc}"
                steps[-1] = ChainStepReceipt(index, query, row, stop_reason=stop_reason)
                break
            if candidate_query is None:
                stop_reason = "no-next-address"
                chain_status = MISSING
                steps[-1] = ChainStepReceipt(index, query, row, stop_reason=stop_reason)
                break
            if isinstance(candidate_query, KnowledgeQuery):
                query = candidate_query.address
            elif isinstance(candidate_query, LogicalTuple):
                query = candidate_query.address
            elif isinstance(candidate_query, LogicalAddress):
                query = candidate_query
            elif isinstance(candidate_query, Mapping):
                query = LogicalAddress(**candidate_query)
            else:
                chain_status = RUNTIME_BLOCKED
                stop_reason = "next_query returned an unsupported address"
                steps[-1] = ChainStepReceipt(index, query, row, stop_reason=stop_reason)
                break
            steps[-1] = ChainStepReceipt(index, steps[-1].query, row, next_query=query)
            current_state = own_state
        observed = sum(1 for step in steps if step.row.execution.signed_consumer_result is not None)
        instrument = "observed" if observed else "inapplicable"
        mechanics = "valid" if steps else "incomplete"
        trend = "observed" if observed else "no-observation"
        return ChainReceipt(
            panel_id=self.panel_id,
            steps=tuple(steps),
            status=chain_status,
            stop_reason=stop_reason,
            mechanics_status=mechanics,
            instrument_status=instrument,
            trend_status=trend,
        )

    run = run_matrix


def build_finite_panel(
    *,
    bank_a: KnowledgeBank,
    bank_b: KnowledgeBank,
    native_operator: KnowledgeOperator,
    compiled_operator: KnowledgeOperator,
    initial_state: Mapping[str, Any] | None = None,
    panel_id: str | None = None,
) -> InterchangePanel:
    """Build the canonical Bank-A/Bank-B × native/compiled finite panel."""

    return InterchangePanel(
        {"bankA": bank_a, "bankB": bank_b},
        {"native": native_operator, "compiled": compiled_operator},
        initial_state=initial_state,
        panel_id=panel_id,
    )


__all__ = [
    "AMBIGUOUS",
    "ArmMetadata",
    "ArtifactBinding",
    "BindingExpectation",
    "BranchCut",
    "ChainReceipt",
    "ChainStepReceipt",
    "CONSUMER_ERROR",
    "CONTEXT",
    "CONTEXT_AS_VALUE",
    "ContentSignedConsumerResult",
    "ConsumerInput",
    "ExactLaw",
    "InterchangePanel",
    "KNOWLEDGE_BANK_SCHEMA",
    "KNOWLEDGE_INTERCHANGE_SCHEMA",
    "KNOWLEDGE_RECORD_SCHEMA",
    "KnowledgeBank",
    "KnowledgeInterchangeError",
    "KnowledgeOperator",
    "KnowledgeRecord",
    "KnowledgeQuery",
    "LogicalAddress",
    "LogicalTuple",
    "LoweredResult",
    "LoweringRequest",
    "MISSING",
    "ModelBinding",
    "NativeResearchSessionBridge",
    "NumericMixing",
    "OK",
    "OperatorContract",
    "OperatorExecution",
    "OPERATOR_CONTRACT_SCHEMA",
    "PANEL_RECEIPT_SCHEMA",
    "PanelReceipt",
    "PartitionAccounting",
    "Provenance",
    "QueryLocalMerge",
    "RECIPIENT_MISMATCH",
    "RECORD_KINDS",
    "RESOLVED",
    "RUNTIME_BLOCKED",
    "RecordProvenance",
    "RecipientLowerer",
    "RecipientLowering",
    "Resolution",
    "RuntimeBlockedError",
    "STATE_LEAKAGE",
    "STALE",
    "SPFOriginTransport",
    "SignedConsumerResult",
    "SourceBinding",
    "VALUE",
    "WRONG_BANK",
    "WRONG_OPERATOR",
    "WRONG_ROUTE",
    "build_finite_panel",
    "canonical_json",
    "content_digest",
]
