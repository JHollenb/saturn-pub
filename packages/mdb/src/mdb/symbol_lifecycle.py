"""Small, sealed evidence records for the life of one debug symbol.

This module is deliberately a control-plane record format.  It joins the
existing public :class:`saturn_pub.symbols.SymbolBinding` to observations made
at native cuts, optional spectral addresses, VM cache manifests, CoT readout
references, and native-consumer uses.  It does not resolve a model, hydrate a
cache, infer a semantic role, or grant write authority.

The records use four separate kinds of qualification:

``ClockWitness``
    A typed clock observation and the evidence that supports its validity.
``SymbolOccurrence``
    A context-qualified location set at one source cut/branch.
``SymbolUse``
    A consumer-side observation tied to a receipt/cut and signed evidence.
``SymbolLifecycle``
    A sealed sidecar joining the binding, definitions, uses, and scoped
    invalidations.

An absent or unknown clock remains unknown.  In particular, a token position
is never silently reinterpreted as a local-age clock for a different runtime.
The fixed ``authority`` fields make the read-only and non-causal boundary
visible in every serialized record.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

from saturn_pub.contracts import StateAddress
from saturn_pub.symbols import SymbolBinding

SYMBOL_LIFECYCLE_SCHEMA = "mdb-symbol-lifecycle-v1"
CLOCK_WITNESS_SCHEMA = "mdb-symbol-clock-witness-v1"
SYMBOL_OCCURRENCE_SCHEMA = "mdb-symbol-occurrence-v1"
SYMBOL_USE_SCHEMA = "mdb-symbol-use-v1"
SPECTRAL_ADDRESS_SCHEMA = "saturn-spectral-address-abi-v1"

_SHA256 = set("0123456789abcdef")
_STATUSES = frozenset(
    {"candidate", "supplied", "observed", "validated", "abstained", "unknown"}
)
_VALIDITY = frozenset({"supported", "partial", "unknown", "invalidated"})
SYMBOL_MAX_JSON_DEPTH = 12
SYMBOL_MAX_SEQUENCE_ITEMS = 512
SYMBOL_MAX_MAPPING_ITEMS = 512
SYMBOL_MAX_STRING_BYTES = 16 * 1024
SYMBOL_MAX_JSON_BYTES = 256 * 1024
_CLOCK_KINDS = frozenset(
    {
        "unknown",
        "token-position",
        "token",
        "depth",
        "decoder-depth",
        "layer",
        "execution-point",
        "local-dt",
        "vector-dt",
        "phase",
        "channel",
        # These are retained as explicitly non-symbol clock domains.  They
        # may be recorded as unknown context, never as an observed symbol age.
        "wall-clock",
        "vm-lru",
    }
)
_LOCAL_CLOCK_KINDS = frozenset({"local-dt", "vector-dt", "phase"})
_NON_SYMBOL_CLOCK_KINDS = frozenset({"wall-clock", "vm-lru"})
_LIFECYCLE_STATUSES = frozenset(
    {
        "valid",
        "invalid",
        "unknown",
        "not-assessed",
        "not-applicable",
        "evidence-incomplete",
        "instrument-broken",
        "observations-only",
        "mixed",
        "directional",
        "convergent",
        "not-established",
    }
)


class SymbolLifecycleError(ValueError):
    """Raised when a symbol lifecycle record cannot be safely represented."""


def _jsonable(value: Any, *, label: str = "value", depth: int = 0) -> Any:
    """Normalize bounded JSON without converting tensor-like payloads."""

    if depth > SYMBOL_MAX_JSON_DEPTH:
        raise SymbolLifecycleError(f"{label} exceeds metadata nesting limit")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value.encode("utf-8")) > SYMBOL_MAX_STRING_BYTES:
            raise SymbolLifecycleError(f"{label} string is too large for metadata")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SymbolLifecycleError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise SymbolLifecycleError(f"{label} cannot contain raw byte payloads")
    if any(hasattr(value, attribute) for attribute in ("detach", "numpy", "ndim", "shape")):
        raise SymbolLifecycleError(f"{label} contains a tensor-like payload")
    if isinstance(value, Mapping):
        if len(value) > SYMBOL_MAX_MAPPING_ITEMS:
            raise SymbolLifecycleError(f"{label} has too many metadata keys")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise SymbolLifecycleError(f"{label} contains a non-text key")
            result[key] = _jsonable(item, label=f"{label}.{key}", depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > SYMBOL_MAX_SEQUENCE_ITEMS:
            raise SymbolLifecycleError(f"{label} has too many metadata items")
        return [
            _jsonable(item, label=f"{label}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    raise SymbolLifecycleError(f"{label} contains unsupported {type(value).__name__}")


def _canonical(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > SYMBOL_MAX_JSON_BYTES:
            raise SymbolLifecycleError("record exceeds the bounded metadata budget")
        return encoded
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SymbolLifecycleError):
            raise
        raise SymbolLifecycleError("record is not canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise SymbolLifecycleError(f"{label} must be non-empty text")
    result = value.strip()
    if len(result.encode("utf-8")) > SYMBOL_MAX_STRING_BYTES:
        raise SymbolLifecycleError(f"{label} is too large for metadata")
    return result


def _optional_text(value: Any, label: str) -> str | None:
    if value in (None, ""):
        return None
    return _text(value, label)


def _refs(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Sequence):
        raise SymbolLifecycleError(f"{label} must be an array of references")
    if len(value) > SYMBOL_MAX_SEQUENCE_ITEMS:
        raise SymbolLifecycleError(f"{label} has too many references")
    result = tuple(_text(item, f"{label}[]") for item in value)
    if len(result) != len(set(result)):
        raise SymbolLifecycleError(f"{label} contains duplicates")
    return result


def _status(value: Any, label: str) -> str:
    result = _text(value, label)
    if result not in _STATUSES:
        raise SymbolLifecycleError(f"{label} has unsupported status {result!r}")
    return result


def _lifecycle_status(value: Any, label: str) -> str:
    result = _text(value, label)
    if result not in _LIFECYCLE_STATUSES:
        raise SymbolLifecycleError(f"{label} has unsupported status {result!r}")
    return result


def _sha(value: Any, label: str) -> str:
    result = _text(value, label).lower().removeprefix("sha256:")
    if len(result) != 64 or any(char not in _SHA256 for char in result):
        raise SymbolLifecycleError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SymbolLifecycleError(f"{label} must be an object")
    result = _jsonable(value, label=label)
    if not isinstance(result, dict):  # pragma: no cover - guarded by Mapping above
        raise SymbolLifecycleError(f"{label} must be an object")
    return result


def _freeze(value: Any) -> Any:
    """Recursively freeze normalized JSON so callers cannot mutate a seal."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return a fresh JSON-shaped copy for callers and serialization."""

    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _frozen_mapping(value: Any, label: str) -> Mapping[str, Any]:
    return _freeze(_mapping(value, label))


def _evidence_rows(value: Any, label: str) -> tuple[Any, ...]:
    """Keep existing signed artifacts opaque while making them JSON-safe."""

    if value is None:
        return ()
    if isinstance(value, (str, Mapping)):
        value = (value,)
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Sequence):
        raise SymbolLifecycleError(f"{label} must be references or JSON objects")
    if len(value) > SYMBOL_MAX_SEQUENCE_ITEMS:
        raise SymbolLifecycleError(f"{label} has too many evidence rows")
    result = tuple(_freeze(_jsonable(item, label=f"{label}[]")) for item in value)
    if len(result) != len({_digest(item) for item in result}):
        raise SymbolLifecycleError(f"{label} contains duplicate evidence")
    return result


def _has_evidence(*values: Any) -> bool:
    return any(value not in (None, "", (), [], {}) for value in values)


def _require_status_evidence(status: str, label: str, *values: Any) -> None:
    if status in {"observed", "validated"} and not _has_evidence(*values):
        raise SymbolLifecycleError(f"{label} status {status!r} requires receipt or evidence")


def _state_address(value: StateAddress | Mapping[str, Any]) -> StateAddress:
    if isinstance(value, StateAddress):
        return value
    if not isinstance(value, Mapping):
        raise SymbolLifecycleError("location must be a StateAddress or object")
    try:
        return StateAddress(
            str(value.get("slot") or ""),
            tuple(value.get("selector") or ()),
            str(value.get("role") or ""),
            value.get("execution_point"),
            value.get("state_schema"),
        )
    except (TypeError, ValueError) as exc:
        raise SymbolLifecycleError("invalid StateAddress location") from exc


def _locations(value: Any, label: str) -> tuple[StateAddress, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise SymbolLifecycleError(f"{label} must be an array of locations")
    if len(value) > SYMBOL_MAX_SEQUENCE_ITEMS:
        raise SymbolLifecycleError(f"{label} has too many locations")
    result = tuple(_state_address(item) for item in value)
    fingerprints = [_digest(item.to_dict()) for item in result]
    if len(fingerprints) != len(set(fingerprints)):
        raise SymbolLifecycleError(f"{label} contains duplicate locations")
    return result


def _location_fingerprint(value: StateAddress) -> str:
    return _digest(value.to_dict())


def _location_sets(value: Any, label: str) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SymbolLifecycleError(f"{label} must be an object of location groups")
    if len(value) > SYMBOL_MAX_MAPPING_ITEMS:
        raise SymbolLifecycleError(f"{label} has too many groups")
    result: dict[str, tuple[str, ...]] = {}
    for name, raw_group in value.items():
        group_name = _text(name, f"{label} name")
        if isinstance(raw_group, (str, bytes, bytearray)) or not isinstance(raw_group, Sequence):
            raise SymbolLifecycleError(f"{label}.{group_name} must be an array")
        if len(raw_group) > SYMBOL_MAX_SEQUENCE_ITEMS:
            raise SymbolLifecycleError(f"{label}.{group_name} has too many locations")
        members: list[str] = []
        for item in raw_group:
            if isinstance(item, StateAddress) or isinstance(item, Mapping):
                members.append(_location_fingerprint(_state_address(item)))
            else:
                members.append(_text(item, f"{label}.{group_name}[]"))
        if len(members) != len(set(members)):
            raise SymbolLifecycleError(f"{label}.{group_name} contains duplicates")
        result[group_name] = tuple(members)
    return result


def _clock(value: Any) -> ClockWitness | None:
    if value is None:
        return None
    if isinstance(value, ClockWitness):
        return value
    if isinstance(value, Mapping):
        return ClockWitness.from_dict(value) if value.get("fingerprint") else ClockWitness(
            clock_id=value.get("clock_id"),
            kind=value.get("kind"),
            domain=value.get("domain"),
            reading=value.get("reading"),
            provenance=tuple(value.get("provenance") or ()),
            validity_status=value.get("validity_status", "unknown"),
            validity_support=tuple(value.get("validity_support") or ()),
            lifetime=value.get("lifetime", {}),
            status=value.get("status", "unknown"),
            metadata=value.get("metadata", {}),
        )
    raise SymbolLifecycleError("clock must be a ClockWitness or object")


def _spectral_typed_payload(value: Mapping[str, Any]) -> Any:
    """Reconstruct the public logical address and verify its canonical digest."""

    from .spectral_address import (
        CanonicalSymbolKey,
        SpectralFrame,
        SpectralInstanceKey,
        SpectralLogicalAddress,
        SpectralRoleTransformSpec,
    )

    allowed = {"schema", "canonical_key", "instance_key", "role_transform", "frame", "fingerprint"}
    if set(value) != allowed or value.get("schema") != SPECTRAL_ADDRESS_SCHEMA:
        raise SymbolLifecycleError("spectral address payload is not a typed logical address")
    try:
        canonical = value["canonical_key"]
        instance = value["instance_key"]
        role = value["role_transform"]
        frame = value["frame"]
        if not all(isinstance(item, Mapping) for item in (canonical, instance, role, frame)):
            raise TypeError("spectral address components must be objects")
        address = SpectralLogicalAddress(
            CanonicalSymbolKey(
                canonical["symbol_id"], tuple(canonical["identity_phases"]),
                canonical["namespace"], canonical["symbol_type"], canonical["address_book_id"],
            ),
            SpectralInstanceKey(
                instance["scene_id"], instance["instance_id"],
                tuple(instance["instance_phases"]), instance.get("edge_id"),
                instance.get("execution_scope", "scene"),
            ),
            SpectralRoleTransformSpec(
                role["role_id"], tuple(role["phase_offsets"]), role["arity"],
                role["argument_index"], role["input_type"], role["output_type"],
                role.get("commutative", False), role.get("inverse_role_id"),
                role.get("exact_unbinding", True), role.get("cleanup_required", False),
                tuple(role.get("frequency_bands", ())), role.get("spf_conformance", "unverified"),
            ),
            SpectralFrame(frame["frame_id"], frame.get("parent_frame_id")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SymbolLifecycleError("spectral address typed payload is invalid") from exc
    if _sha(value.get("fingerprint"), "spectral address fingerprint") != address.fingerprint:
        raise SymbolLifecycleError("spectral address fingerprint mismatch")
    return address


def _unjoined_reference(value: Any, label: str) -> dict[str, Any]:
    return {
        "reference": _text(value, label),
        "joined": False,
        "verification": "unverified",
    }


def _spectral(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _unjoined_reference(value, "spectral address reference")
    if isinstance(value, Mapping):
        result = _mapping(value, "spectral_address")
        if result.get("reference") is not None and result.get("fingerprint") is None:
            if set(result) - {"reference", "joined", "verification"}:
                raise SymbolLifecycleError("unjoined spectral address has unsupported fields")
            return _unjoined_reference(result["reference"], "spectral address reference")
        _spectral_typed_payload(result)
        result["fingerprint"] = _sha(result["fingerprint"], "spectral address fingerprint")
        return result
    from .spectral_address import SpectralLogicalAddress

    if not isinstance(value, SpectralLogicalAddress):
        raise SymbolLifecycleError("spectral address must be a SpectralLogicalAddress")
    payload: Any = dataclasses.asdict(value)
    result = _mapping(payload, "spectral_address")
    result["schema"] = SPECTRAL_ADDRESS_SCHEMA
    result["fingerprint"] = _sha(value.fingerprint, "spectral address fingerprint")
    # Verify the exact payload we are about to retain, including the typed
    # object fields and its canonical identity.
    _spectral_typed_payload({key: result[key] for key in (
        "schema", "canonical_key", "instance_key", "role_transform", "frame", "fingerprint"
    )})
    return result


def _vm_bundle(value: Any) -> dict[str, Any] | None:
    """Normalize a VM bundle while retaining only its metadata references."""

    if value is None:
        return None
    if isinstance(value, str):
        return _unjoined_reference(value, "VM bundle reference")
    if isinstance(value, Mapping):
        result = _mapping(value, "vm_bundle")
    elif hasattr(value, "to_dict"):
        result = _mapping(value.to_dict(), "vm_bundle")
    else:
        raise SymbolLifecycleError("vm_bundle must be a bundle, mapping, or reference")
    schema = result.get("schema")
    if schema is None:
        if "reference" not in result or set(result) - {"reference", "joined", "verification"}:
            raise SymbolLifecycleError("VM bundle must be a sealed bundle or explicit reference")
        return _unjoined_reference(result["reference"], "VM bundle reference")
    if schema == "saturn-vm-cache-execution-bundle-v1":
        # Import lazily so importing this metadata module never imports a model.
        from .vm_cache import VMCacheExecutionBundle

        if result.get("fingerprint") is None and result.get("content_sha256") is None:
            raise SymbolLifecycleError("VM bundle must include its original seal")
        try:
            checked = VMCacheExecutionBundle.from_dict(result)
        except Exception as exc:  # normalize the public VM error at this boundary
            raise SymbolLifecycleError("VM bundle seal or schema is invalid") from exc
        checked_payload = checked.to_dict()
        if set(result) - set(checked_payload):
            raise SymbolLifecycleError("VM bundle contains unsupported fields")
        result = checked_payload
    else:
        raise SymbolLifecycleError("unsupported VM bundle schema")
    return result


def _binding(value: SymbolBinding | Mapping[str, Any]) -> SymbolBinding:
    if isinstance(value, SymbolBinding):
        return value
    if not isinstance(value, Mapping):
        raise SymbolLifecycleError("binding must be a SymbolBinding or sealed object")
    try:
        return SymbolBinding.from_dict(dict(value))
    except (TypeError, KeyError, ValueError) as exc:
        raise SymbolLifecycleError("symbol binding seal or schema is invalid") from exc


@dataclass(frozen=True, slots=True)
class ClockWitness:
    """A typed local clock reading plus explicit support for its validity."""

    clock_id: str
    kind: str
    domain: str
    reading: Any = None
    provenance: tuple[str, ...] = ()
    validity_status: str = "unknown"
    validity_support: tuple[str, ...] = ()
    lifetime: Mapping[str, Any] = field(default_factory=dict)
    status: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    schema: ClassVar[str] = CLOCK_WITNESS_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "clock_id", _text(self.clock_id, "clock_id"))
        kind = _text(self.kind, "clock kind").lower()
        if kind not in _CLOCK_KINDS:
            raise SymbolLifecycleError(
                f"unsupported clock kind {kind!r}; use an explicit 'unknown' kind"
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "domain", _text(self.domain, "clock domain"))
        object.__setattr__(self, "reading", _freeze(_jsonable(self.reading, label="clock reading")))
        object.__setattr__(self, "provenance", _refs(self.provenance, "clock provenance"))
        validity = _text(self.validity_status, "clock validity status")
        if validity not in _VALIDITY:
            raise SymbolLifecycleError(f"unsupported clock validity status {validity!r}")
        support = _refs(self.validity_support, "clock validity support")
        if validity in {"supported", "invalidated"} and not support:
            raise SymbolLifecycleError("supported or invalidated clocks require validity evidence")
        object.__setattr__(self, "validity_status", validity)
        object.__setattr__(self, "validity_support", support)
        lifetime = _mapping(self.lifetime, "clock lifetime")
        object.__setattr__(self, "lifetime", _freeze(lifetime))
        status = _status(self.status, "clock status")
        if status != "unknown" and self.reading is None:
            raise SymbolLifecycleError("a non-unknown clock needs a reading")
        if self.reading is not None and not self.provenance:
            raise SymbolLifecycleError("a clock reading needs source provenance")
        metadata = _mapping(self.metadata, "clock metadata")
        if kind in _NON_SYMBOL_CLOCK_KINDS and status != "unknown":
            raise SymbolLifecycleError(f"{kind} cannot be an observed symbol clock")
        if kind in _LOCAL_CLOCK_KINDS and status in {"observed", "validated"}:
            required = ("source_cut", "model_execution", "consumer", "reset")
            missing = [key for key in required if not _has_evidence(metadata.get(key))]
            if kind == "phase" and not (
                _has_evidence(metadata.get("winding")) or _has_evidence(metadata.get("age"))
            ):
                missing.append("winding-or-age")
            if missing:
                raise SymbolLifecycleError(
                    f"{kind} clock requires source/model/consumer/reset evidence: {missing}"
                )
            if kind in {"local-dt", "vector-dt"} and (
                isinstance(self.reading, (str, bytes, bytearray))
                or not isinstance(self.reading, Sequence)
                or not self.reading
            ):
                raise SymbolLifecycleError(f"{kind} clock reading must be a non-empty vector")
        if status == "unknown" and self.reading is not None:
            # Keep an explicitly supplied value usable, but do not upgrade the
            # evidence state merely because a value happens to be present.
            pass
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metadata", _freeze(metadata))
        _canonical(self._body())

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "clock_id": self.clock_id,
            "kind": self.kind,
            "domain": self.domain,
            "reading": _thaw(self.reading),
            "provenance": list(self.provenance),
            "validity_status": self.validity_status,
            "validity_support": list(self.validity_support),
            "lifetime": _thaw(self.lifetime),
            "status": self.status,
            "metadata": _thaw(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        return _digest(self._body())

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ClockWitness:
        if not isinstance(value, Mapping) or value.get("schema") != cls.schema:
            raise SymbolLifecycleError("unsupported clock witness schema")
        declared = value.get("fingerprint")
        body = {key: item for key, item in value.items() if key != "fingerprint"}
        if declared != _digest(body):
            raise SymbolLifecycleError("clock witness fingerprint mismatch")
        result = cls(
            clock_id=value.get("clock_id"), kind=value.get("kind"), domain=value.get("domain"),
            reading=value.get("reading"), provenance=tuple(value.get("provenance") or ()),
            validity_status=value.get("validity_status", "unknown"),
            validity_support=tuple(value.get("validity_support") or ()),
            lifetime=value.get("lifetime", {}), status=value.get("status", "unknown"),
            metadata=value.get("metadata", {}),
        )
        if result.to_dict() != dict(value):
            raise SymbolLifecycleError("clock witness contains unsupported fields")
        return result


@dataclass(frozen=True, slots=True)
class SymbolOccurrence:
    """One definition/carrier location set at a qualified native boundary."""

    occurrence_id: str
    binding_fingerprint: str
    session_id: str
    source_cut: str
    locations: tuple[StateAddress, ...]
    # The lifecycle normally carries the binding once at its root.  A caller
    # may also embed the sealed public binding here when an occurrence is
    # transported on its own; both forms are checked against the fingerprint.
    binding: SymbolBinding | Mapping[str, Any] | None = None
    branch: str | None = None
    result_cut: str | None = None
    execution: str | None = None
    receipt: str | None = None
    coalition: tuple[str, ...] = ()
    location_sets: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    clock: ClockWitness | None = None
    spectral_address: Any = None
    vm_bundle: Any = None
    cot_refs: tuple[Any, ...] = ()
    definition_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    status: str = "unknown"
    validity_status: str = "unknown"
    validity_support: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    schema: ClassVar[str] = SYMBOL_OCCURRENCE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "occurrence_id", _text(self.occurrence_id, "occurrence_id"))
        object.__setattr__(self, "binding_fingerprint", _sha(self.binding_fingerprint, "binding_fingerprint"))
        object.__setattr__(self, "session_id", _text(self.session_id, "session_id"))
        object.__setattr__(self, "source_cut", _text(self.source_cut, "source_cut"))
        object.__setattr__(self, "locations", _locations(self.locations, "locations"))
        binding = None if self.binding is None else _binding(self.binding)
        if binding is not None and binding.to_dict()["fingerprint"] != self.binding_fingerprint:
            raise SymbolLifecycleError("occurrence binding does not match binding_fingerprint")
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "branch", _optional_text(self.branch, "branch"))
        object.__setattr__(self, "result_cut", _optional_text(self.result_cut, "result_cut"))
        object.__setattr__(self, "execution", _optional_text(self.execution, "execution"))
        object.__setattr__(self, "receipt", _optional_text(self.receipt, "receipt"))
        coalition = _refs(self.coalition, "coalition")
        object.__setattr__(self, "coalition", coalition)
        object.__setattr__(
            self, "location_sets", _freeze(_location_sets(self.location_sets, "location_sets"))
        )
        object.__setattr__(self, "clock", _clock(self.clock))
        spectral = _spectral(self.spectral_address)
        vm_bundle = _vm_bundle(self.vm_bundle)
        object.__setattr__(self, "spectral_address", _freeze(spectral))
        object.__setattr__(self, "vm_bundle", _freeze(vm_bundle))
        object.__setattr__(self, "cot_refs", _evidence_rows(self.cot_refs, "cot_refs"))
        object.__setattr__(self, "definition_refs", _refs(self.definition_refs, "definition_refs"))
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs, "evidence_refs"))
        status = _status(self.status, "occurrence status")
        object.__setattr__(self, "status", status)
        validity = _text(self.validity_status, "occurrence validity status")
        if validity not in _VALIDITY:
            raise SymbolLifecycleError(f"unsupported occurrence validity status {validity!r}")
        support = _refs(self.validity_support, "occurrence validity support")
        if validity in {"supported", "invalidated"} and not support:
            raise SymbolLifecycleError("occurrence validity requires evidence")
        _require_status_evidence(
            status, "occurrence", self.receipt, self.definition_refs,
            self.evidence_refs, self.cot_refs,
        )
        object.__setattr__(self, "validity_status", validity)
        object.__setattr__(self, "validity_support", support)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, "occurrence metadata"))
        _canonical(self._body())

    @property
    def location_set(self) -> tuple[str, ...]:
        return tuple(_location_fingerprint(item) for item in self.locations)

    @property
    def branch_id(self) -> str | None:
        return self.branch

    @property
    def cut_id(self) -> str:
        return self.source_cut

    @property
    def binding_record(self) -> SymbolBinding | None:
        return self.binding

    @classmethod
    def for_binding(cls, binding: SymbolBinding | Mapping[str, Any], **kwargs: Any) -> SymbolOccurrence:
        """Construct an occurrence with a sealed binding payload and identity."""

        resolved = _binding(binding)
        kwargs.setdefault("binding_fingerprint", resolved.to_dict()["fingerprint"])
        kwargs.setdefault("binding", resolved)
        return cls(**kwargs)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "occurrence_id": self.occurrence_id,
            "binding_fingerprint": self.binding_fingerprint,
            "binding": None if self.binding is None else self.binding.to_dict(),
            "session_id": self.session_id,
            "source_cut": self.source_cut,
            "result_cut": self.result_cut,
            "branch": self.branch,
            "execution": self.execution,
            "receipt": self.receipt,
            "locations": [item.to_dict() for item in self.locations],
            "location_set": list(self.location_set),
            "coalition": list(self.coalition),
            "location_sets": _thaw(self.location_sets),
            "clock": None if self.clock is None else self.clock.to_dict(),
            "spectral_address": _thaw(self.spectral_address),
            "vm_bundle": _thaw(self.vm_bundle),
            "cot_refs": _thaw(self.cot_refs),
            "definition_refs": list(self.definition_refs),
            "evidence_refs": list(self.evidence_refs),
            "status": self.status,
            "validity_status": self.validity_status,
            "validity_support": list(self.validity_support),
            "authority": {"write": "native-only", "causal_effect": "not-established"},
            "metadata": _thaw(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        return _digest(self._body())

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SymbolOccurrence:
        if not isinstance(value, Mapping) or value.get("schema") != cls.schema:
            raise SymbolLifecycleError("unsupported symbol occurrence schema")
        declared = value.get("fingerprint")
        body = {key: item for key, item in value.items() if key != "fingerprint"}
        if declared != _digest(body):
            raise SymbolLifecycleError("symbol occurrence fingerprint mismatch")
        result = cls(
            occurrence_id=value.get("occurrence_id"),
            binding_fingerprint=value.get("binding_fingerprint"),
            session_id=value.get("session_id"),
            source_cut=value.get("source_cut"),
            result_cut=value.get("result_cut"),
            branch=value.get("branch"),
            execution=value.get("execution"),
            receipt=value.get("receipt"),
            locations=tuple(value.get("locations") or ()),
            binding=value.get("binding"),
            coalition=tuple(value.get("coalition") or ()),
            location_sets=value.get("location_sets") or {},
            clock=value.get("clock"),
            spectral_address=value.get("spectral_address"),
            vm_bundle=value.get("vm_bundle"),
            cot_refs=tuple(value.get("cot_refs") or ()),
            definition_refs=tuple(value.get("definition_refs") or ()),
            evidence_refs=tuple(value.get("evidence_refs") or ()),
            status=value.get("status", "unknown"),
            validity_status=value.get("validity_status", "unknown"),
            validity_support=tuple(value.get("validity_support") or ()),
            metadata=value.get("metadata", {}),
        )
        if result.to_dict() != dict(value):
            raise SymbolLifecycleError("symbol occurrence contains unsupported fields")
        return result


@dataclass(frozen=True, slots=True)
class SymbolUse:
    """One consumer-side use with signed observations kept separate from claims."""

    use_id: str
    binding_fingerprint: str
    session_id: str
    consumer: str
    source_cut: str | None = None
    parent_cut: str | None = None
    result_cut: str | None = None
    branch: str | None = None
    execution: str | None = None
    receipt: str | None = None
    locations: tuple[StateAddress, ...] = ()
    clock: ClockWitness | None = None
    signed_observations: tuple[Any, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    status: str = "unknown"
    validity_status: str = "unknown"
    validity_support: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    schema: ClassVar[str] = SYMBOL_USE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "use_id", _text(self.use_id, "use_id"))
        object.__setattr__(self, "binding_fingerprint", _sha(self.binding_fingerprint, "binding_fingerprint"))
        object.__setattr__(self, "session_id", _text(self.session_id, "session_id"))
        object.__setattr__(self, "consumer", _text(self.consumer, "consumer"))
        for name in ("source_cut", "parent_cut", "result_cut", "branch", "execution", "receipt"):
            object.__setattr__(self, name, _optional_text(getattr(self, name), name))
        object.__setattr__(self, "locations", _locations(self.locations, "use locations"))
        object.__setattr__(self, "clock", _clock(self.clock))
        object.__setattr__(self, "signed_observations", _evidence_rows(self.signed_observations, "signed_observations"))
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs, "use evidence_refs"))
        status = _status(self.status, "use status")
        object.__setattr__(self, "status", status)
        validity = _text(self.validity_status, "use validity status")
        if validity not in _VALIDITY:
            raise SymbolLifecycleError(f"unsupported use validity status {validity!r}")
        support = _refs(self.validity_support, "use validity support")
        if validity in {"supported", "invalidated"} and not support:
            raise SymbolLifecycleError("use validity requires evidence")
        _require_status_evidence(
            status, "use", self.receipt, self.evidence_refs, self.signed_observations,
        )
        object.__setattr__(self, "validity_status", validity)
        object.__setattr__(self, "validity_support", support)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, "use metadata"))
        _canonical(self._body())

    @property
    def observations(self) -> tuple[Any, ...]:
        return self.signed_observations

    @property
    def branch_id(self) -> str | None:
        return self.branch

    @property
    def cuts(self) -> dict[str, str]:
        return {
            key: value
            for key, value in (
                ("source", self.source_cut), ("parent", self.parent_cut), ("result", self.result_cut)
            )
            if value is not None
        }

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "use_id": self.use_id,
            "binding_fingerprint": self.binding_fingerprint,
            "session_id": self.session_id,
            "consumer": self.consumer,
            "source_cut": self.source_cut,
            "parent_cut": self.parent_cut,
            "result_cut": self.result_cut,
            "branch": self.branch,
            "execution": self.execution,
            "receipt": self.receipt,
            "locations": [item.to_dict() for item in self.locations],
            "clock": None if self.clock is None else self.clock.to_dict(),
            "signed_observations": _thaw(self.signed_observations),
            "evidence_refs": list(self.evidence_refs),
            "status": self.status,
            "validity_status": self.validity_status,
            "validity_support": list(self.validity_support),
            "authority": {"write": "native-only", "causal_effect": "not-established"},
            "metadata": _thaw(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        return _digest(self._body())

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SymbolUse:
        if not isinstance(value, Mapping) or value.get("schema") != cls.schema:
            raise SymbolLifecycleError("unsupported symbol use schema")
        declared = value.get("fingerprint")
        body = {key: item for key, item in value.items() if key != "fingerprint"}
        if declared != _digest(body):
            raise SymbolLifecycleError("symbol use fingerprint mismatch")
        result = cls(
            use_id=value.get("use_id"), binding_fingerprint=value.get("binding_fingerprint"),
            session_id=value.get("session_id"), consumer=value.get("consumer"),
            source_cut=value.get("source_cut"), parent_cut=value.get("parent_cut"),
            result_cut=value.get("result_cut"), branch=value.get("branch"),
            execution=value.get("execution"), receipt=value.get("receipt"),
            locations=tuple(value.get("locations") or ()), clock=value.get("clock"),
            signed_observations=tuple(value.get("signed_observations") or ()),
            evidence_refs=tuple(value.get("evidence_refs") or ()), status=value.get("status", "unknown"),
            validity_status=value.get("validity_status", "unknown"),
            validity_support=tuple(value.get("validity_support") or ()), metadata=value.get("metadata", {}),
        )
        if result.to_dict() != dict(value):
            raise SymbolLifecycleError("symbol use contains unsupported fields")
        return result


def _invalidation(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SymbolLifecycleError("invalidation must be an object")
    scope = _text(value.get("scope"), "invalidation scope")
    reason = _text(value.get("reason"), "invalidation reason")
    refs = _refs(value.get("evidence_refs") or (), "invalidation evidence")
    clock = _clock(value.get("clock"))
    status_value = value.get("status")
    status = _status(
        status_value if status_value is not None else ("observed" if _has_evidence(refs, clock) else "unknown"),
        "invalidation status",
    )
    _require_status_evidence(status, "invalidation", refs, clock)
    return {
        "scope": scope,
        "reason": reason,
        "status": status,
        "evidence_refs": list(refs),
        "clock": None if clock is None else clock.to_dict(),
        "metadata": _mapping(value.get("metadata", {}), "invalidation metadata"),
    }


@dataclass(frozen=True, slots=True)
class SymbolLifecycle:
    """A sealed, append-only evidence sidecar for one public symbol binding."""

    symbol: str
    binding: SymbolBinding | Mapping[str, Any]
    definitions: tuple[SymbolOccurrence, ...] = ()
    uses: tuple[SymbolUse, ...] = ()
    mechanics_status: str = "unknown"
    instrument_status: str = "unknown"
    trend_status: str = "unknown"
    terminal_status: str = "unknown"
    invalidations: tuple[Mapping[str, Any], ...] = ()
    claim_boundary: str = (
        "This sidecar joins qualified debug evidence. It does not establish semantic "
        "equivalence, causal necessity, or new write authority."
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = SYMBOL_LIFECYCLE_SCHEMA

    def __post_init__(self) -> None:
        symbol = _text(self.symbol, "symbol")
        binding = _binding(self.binding)
        if binding.symbol != symbol:
            raise SymbolLifecycleError("lifecycle symbol differs from binding symbol")
        definitions = tuple(
            item if isinstance(item, SymbolOccurrence) else SymbolOccurrence.from_dict(item)
            for item in self.definitions
        )
        uses = tuple(item if isinstance(item, SymbolUse) else SymbolUse.from_dict(item) for item in self.uses)
        records = [item.occurrence_id for item in definitions] + [item.use_id for item in uses]
        if len(records) != len(set(records)):
            raise SymbolLifecycleError("lifecycle occurrence/use IDs must be unique")
        root_locations = {_location_fingerprint(location) for location in binding.locations}
        for item in (*definitions, *uses):
            if item.binding_fingerprint != binding.to_dict()["fingerprint"]:
                raise SymbolLifecycleError("lifecycle record belongs to another binding")
            execution = getattr(item, "execution", None)
            if execution is not None and execution != binding.execution:
                raise SymbolLifecycleError("lifecycle record belongs to another execution")
            item_locations = {_location_fingerprint(location) for location in item.locations}
            if not item_locations.issubset(root_locations):
                raise SymbolLifecycleError("lifecycle record locations exceed root binding locations")
        invalidations = tuple(_invalidation(item) for item in self.invalidations)
        for item in invalidations:
            scope = item["scope"]
            if scope.startswith("occurrence:") and scope.removeprefix("occurrence:") not in {
                row.occurrence_id for row in definitions
            }:
                raise SymbolLifecycleError("occurrence invalidation names an unknown occurrence")
            if scope.startswith("use:") and scope.removeprefix("use:") not in {
                row.use_id for row in uses
            }:
                raise SymbolLifecycleError("use invalidation names an unknown use")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "definitions", definitions)
        object.__setattr__(self, "uses", uses)
        for name in ("mechanics_status", "instrument_status", "trend_status", "terminal_status"):
            object.__setattr__(self, name, _lifecycle_status(getattr(self, name), name))
        object.__setattr__(self, "invalidations", tuple(_freeze(item) for item in invalidations))
        object.__setattr__(self, "claim_boundary", _text(self.claim_boundary, "claim_boundary"))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, "lifecycle metadata"))
        object.__setattr__(self, "fingerprint", _digest(self._body()))

    @classmethod
    def define(cls, binding: SymbolBinding | Mapping[str, Any], **kwargs: Any) -> SymbolLifecycle:
        resolved = _binding(binding)
        return cls(symbol=resolved.symbol, binding=resolved, **kwargs)

    create = define

    @property
    def binding_fingerprint(self) -> str:
        return self.binding.to_dict()["fingerprint"]

    @property
    def authority(self) -> dict[str, str]:
        return {"write": "native-only", "causal_effect": "not-established"}

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "symbol": self.symbol,
            "binding": self.binding.to_dict(),
            "binding_fingerprint": self.binding_fingerprint,
            "definitions": [item.to_dict() for item in self.definitions],
            "uses": [item.to_dict() for item in self.uses],
            "statuses": {
                "mechanics": self.mechanics_status,
                "instrument": self.instrument_status,
                "trend": self.trend_status,
                "terminal": self.terminal_status,
            },
            "invalidations": _thaw(self.invalidations),
            "claim_boundary": self.claim_boundary,
            "authority": self.authority,
            "metadata": _thaw(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SymbolLifecycle:
        if not isinstance(value, Mapping) or value.get("schema") != cls.schema:
            raise SymbolLifecycleError("unsupported symbol lifecycle schema")
        declared = value.get("fingerprint")
        body = {key: item for key, item in value.items() if key != "fingerprint"}
        if declared != _digest(body):
            raise SymbolLifecycleError("symbol lifecycle fingerprint mismatch")
        statuses = value.get("statuses") or {}
        if not isinstance(statuses, Mapping):
            raise SymbolLifecycleError("lifecycle statuses must be an object")
        result = cls(
            symbol=value.get("symbol"), binding=value.get("binding"),
            definitions=tuple(value.get("definitions") or ()), uses=tuple(value.get("uses") or ()),
            mechanics_status=statuses.get("mechanics", "unknown"),
            instrument_status=statuses.get("instrument", "unknown"),
            trend_status=statuses.get("trend", "unknown"),
            terminal_status=statuses.get("terminal", "unknown"),
            invalidations=tuple(value.get("invalidations") or ()),
            claim_boundary=value.get("claim_boundary"), metadata=value.get("metadata", {}),
        )
        if value.get("binding_fingerprint") != result.binding_fingerprint:
            raise SymbolLifecycleError("lifecycle binding fingerprint mismatch")
        if result.to_dict() != dict(value):
            raise SymbolLifecycleError("symbol lifecycle contains unsupported fields")
        return result

    def verify(self) -> dict[str, Any]:
        """Recheck the seal and child identities without executing anything."""

        if self.fingerprint != _digest(self._body()):
            raise SymbolLifecycleError("symbol lifecycle fingerprint mismatch")
        return {
            "schema": self.schema,
            "fingerprint": self.fingerprint,
            "binding_fingerprint": self.binding_fingerprint,
            "definitions": len(self.definitions),
            "uses": len(self.uses),
            "invalidations": len(self.invalidations),
            "passed": True,
        }

    def add_occurrence(self, occurrence: SymbolOccurrence | Mapping[str, Any]) -> SymbolLifecycle:
        row = occurrence if isinstance(occurrence, SymbolOccurrence) else SymbolOccurrence.from_dict(occurrence)
        if row.binding_fingerprint != self.binding_fingerprint:
            raise SymbolLifecycleError("occurrence belongs to another binding")
        if row.occurrence_id in {item.occurrence_id for item in self.definitions} | {item.use_id for item in self.uses}:
            raise SymbolLifecycleError("lifecycle occurrence/use ID already exists")
        return type(self)(
            symbol=self.symbol, binding=self.binding, definitions=(*self.definitions, row), uses=self.uses,
            mechanics_status=self.mechanics_status, instrument_status=self.instrument_status,
            trend_status=self.trend_status, terminal_status=self.terminal_status,
            invalidations=self.invalidations, claim_boundary=self.claim_boundary, metadata=self.metadata,
        )

    def add_use(self, use: SymbolUse | Mapping[str, Any]) -> SymbolLifecycle:
        row = use if isinstance(use, SymbolUse) else SymbolUse.from_dict(use)
        if row.binding_fingerprint != self.binding_fingerprint:
            raise SymbolLifecycleError("use belongs to another binding")
        if row.use_id in {item.occurrence_id for item in self.definitions} | {item.use_id for item in self.uses}:
            raise SymbolLifecycleError("lifecycle occurrence/use ID already exists")
        return type(self)(
            symbol=self.symbol, binding=self.binding, definitions=self.definitions, uses=(*self.uses, row),
            mechanics_status=self.mechanics_status, instrument_status=self.instrument_status,
            trend_status=self.trend_status, terminal_status=self.terminal_status,
            invalidations=self.invalidations, claim_boundary=self.claim_boundary, metadata=self.metadata,
        )

    def invalidate(
        self,
        *,
        scope: str,
        reason: str,
        evidence_refs: Sequence[str] = (),
        clock: ClockWitness | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SymbolLifecycle:
        """Add a scoped invalidation while retaining all prior evidence."""

        row = _invalidation(
            {
                "scope": scope,
                "reason": reason,
                "evidence_refs": list(evidence_refs),
                "clock": None if clock is None else (_clock(clock).to_dict() if _clock(clock) else None),
                "metadata": metadata or {},
            }
        )
        return type(self)(
            symbol=self.symbol, binding=self.binding, definitions=self.definitions, uses=self.uses,
            mechanics_status=self.mechanics_status, instrument_status=self.instrument_status,
            trend_status=self.trend_status, terminal_status=self.terminal_status,
            invalidations=(*self.invalidations, row), claim_boundary=self.claim_boundary, metadata=self.metadata,
        )

    def occurrences(
        self, *, session_id: str | None = None, branch: str | None = None,
        source_cut: str | None = None, execution: str | None = None,
    ) -> tuple[SymbolOccurrence, ...]:
        """Select definitions by exact supplied identity; unknown filters are not inferred."""

        rows = self.definitions
        if session_id is not None:
            rows = tuple(item for item in rows if item.session_id == session_id)
        if branch is not None:
            rows = tuple(item for item in rows if item.branch == branch)
        if source_cut is not None:
            rows = tuple(item for item in rows if item.source_cut == source_cut)
        if execution is not None:
            rows = tuple(item for item in rows if item.execution == execution)
        return rows

    def save(self, path: str | Path) -> None:
        self.verify()
        Path(path).write_text(json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SymbolLifecycle:
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SymbolLifecycleError(f"cannot read symbol lifecycle: {path}") from exc
        return cls.from_dict(value)


def verify_symbol_lifecycle(value: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a serialized lifecycle and return compact proof metadata."""

    return SymbolLifecycle.from_dict(value).verify()


__all__ = [
    "CLOCK_WITNESS_SCHEMA",
    "SPECTRAL_ADDRESS_SCHEMA",
    "SYMBOL_LIFECYCLE_SCHEMA",
    "SYMBOL_OCCURRENCE_SCHEMA",
    "SYMBOL_USE_SCHEMA",
    "ClockWitness",
    "SymbolLifecycle",
    "SymbolLifecycleError",
    "SymbolOccurrence",
    "SymbolUse",
    "verify_symbol_lifecycle",
]
