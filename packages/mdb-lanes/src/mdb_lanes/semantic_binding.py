"""Conditioner-aware debug locations for semantic symbols.

An object manifest names a semantic role and its measured local coordinates.
This module binds that role to one exact model/conditioner/input context and a
typed location list.  The binding is metadata only: native adapters continue
to own tensor resolution, mutation, invalidation, and continuation.

The contract deliberately behaves like compiler debug information.  A stable
symbol may lower to different physical support at different typed boundaries,
and changing the model, tokenizer, conditioner, input, state schema, or seed
invalidates the binding rather than silently reusing an old row.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .object_manifest import validate_object_manifest

SEMANTIC_BINDING_SCHEMA = "saturn-semantic-binding-v1"

_DEFAULT_CLAIM_BOUNDARY = (
    "This binding is a model-, conditioner-, tokenizer-, input-, state-schema-, "
    "and seed-qualified debug location list. It is not a universal concept "
    "address, a minimal circuit, or evidence that the named field is causally "
    "complete without its referenced native-consumer controls."
)
_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTEXT_KEYS = (
    "model_id",
    "model_fingerprint",
    "conditioner_id",
    "conditioner_family",
    "tokenizer_fingerprint",
    "input_fingerprint",
    "state_schema_id",
    "seed",
)
_LOCATION_KINDS = {
    "conditioner-row-set",
    "conditioner-span",
    "pooled-conditioner",
    "latent-id-set",
    "custom",
}
_BINDING_STATUSES = {"candidate", "observed", "validated", "abstained"}
_REPLICATION_POLICIES = {"all-listed-sites", "native-resolver"}


class SemanticBindingError(ValueError):
    """Raised when a semantic symbol cannot be bound without ambiguity."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, *, label: str, safe: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticBindingError(f"{label} must be a non-empty string")
    result = value.strip()
    if safe and not _SAFE_NAME.fullmatch(result):
        raise SemanticBindingError(f"{label} is not a safe identifier: {result!r}")
    return result


def _json_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticBindingError(f"{label} must be an object")
    try:
        normalized = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SemanticBindingError(f"{label} must contain JSON data") from exc
    if not isinstance(normalized, dict):
        raise SemanticBindingError(f"{label} must be an object")
    return normalized


def _strings(value: Sequence[Any], *, label: str, safe: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise SemanticBindingError(f"{label} must be a sequence")
    result = tuple(_text(item, label=label, safe=safe) for item in value)
    if len(result) != len(set(result)):
        raise SemanticBindingError(f"{label} contains duplicates")
    return result


def _indices(value: Sequence[Any], *, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise SemanticBindingError(f"{label} must be a sequence of integers")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise SemanticBindingError(f"{label} must contain non-negative plain integers")
        result.append(int(item))
    if not result:
        raise SemanticBindingError(f"{label} must not be empty")
    if len(result) != len(set(result)):
        raise SemanticBindingError(f"{label} contains duplicates")
    return tuple(result)


def _probability(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticBindingError(f"{label} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise SemanticBindingError(f"{label} must be in [0, 1]")
    return result


def _context(value: Mapping[str, Any]) -> dict[str, Any]:
    context = _json_mapping(value, label="binding context")
    missing = [key for key in _CONTEXT_KEYS if context.get(key) in {None, ""}]
    if missing:
        raise SemanticBindingError(f"binding context is missing required keys: {missing}")
    for key in ("model_id", "conditioner_id", "conditioner_family", "state_schema_id"):
        _text(context[key], label=f"binding context {key}")
    for key in ("model_fingerprint", "tokenizer_fingerprint", "input_fingerprint"):
        fingerprint = str(context[key])
        if not _SHA256.fullmatch(fingerprint):
            raise SemanticBindingError(
                f"binding context {key} must be a lowercase SHA-256"
            )
    return context


def _support_indices(support: Mapping[str, Any], key: str, *, label: str) -> tuple[int, ...]:
    raw = support.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SemanticBindingError(f"{label} needs support.{key}")
    return _indices(raw, label=f"{label} support.{key}")


@dataclass(frozen=True, slots=True)
class SemanticLocation:
    """One physical component of a semantic symbol's typed location list.

    Multiple locations may be active at the same boundary/index.  They are a
    joint address set (for example a T5 span plus a pooled conditioner
    component), not competing aliases.
    """

    location_id: str
    kind: str
    boundary: str
    indices: tuple[int, ...]
    carrier: str
    support: Mapping[str, Any]
    resolver: str
    native_consumer: str
    stream: str | None = None
    sites: tuple[str, ...] = ()
    replication_policy: str = "all-listed-sites"
    writer: str | None = None
    invalidates: tuple[str, ...] = ()
    numerical_contract: str = "bounded"
    confidence: float = 0.0
    evidence_refs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        location_id = _text(self.location_id, label="location_id", safe=True)
        kind = _text(self.kind, label="location kind", safe=True)
        if kind not in _LOCATION_KINDS:
            raise SemanticBindingError(f"unsupported semantic location kind {kind!r}")
        boundary = _text(self.boundary, label="location boundary", safe=True)
        indices = _indices(self.indices, label="location indices")
        carrier = _text(self.carrier, label="location carrier", safe=True)
        resolver = _text(self.resolver, label="location resolver", safe=True)
        consumer = _text(self.native_consumer, label="native consumer", safe=True)
        stream = None
        if self.stream is not None:
            stream = _text(self.stream, label="location stream", safe=True)
        sites = _strings(self.sites, label="location sites", safe=True)
        replication_policy = _text(
            self.replication_policy,
            label="location replication policy",
            safe=True,
        )
        if replication_policy not in _REPLICATION_POLICIES:
            raise SemanticBindingError(
                f"unsupported location replication policy {replication_policy!r}"
            )
        writer = None
        if self.writer is not None:
            writer = _text(self.writer, label="location writer", safe=True)
        invalidates = _strings(self.invalidates, label="location invalidations", safe=True)
        numerical_contract = _text(
            self.numerical_contract,
            label="location numerical contract",
            safe=True,
        )
        confidence = _probability(self.confidence, label="location confidence")
        evidence_refs = _strings(self.evidence_refs, label="location evidence references")
        support = _json_mapping(self.support, label="location support")
        metadata = _json_mapping(self.metadata, label="location metadata")

        if kind in {"conditioner-row-set", "conditioner-span"}:
            if stream is None or not sites:
                raise SemanticBindingError(
                    f"{kind} requires one stream and at least one route site"
                )
            rows = _support_indices(support, "rows", label=kind)
            support["rows"] = list(rows)
            if kind == "conditioner-span":
                expected = tuple(range(rows[0], rows[-1] + 1))
                if rows != expected:
                    raise SemanticBindingError(
                        "conditioner-span support.rows must be ordered and contiguous"
                    )
        elif kind == "pooled-conditioner":
            support["slot"] = _text(
                support.get("slot"), label="pooled-conditioner support.slot", safe=True
            )
        elif kind == "latent-id-set":
            latent_ids = _support_indices(support, "latent_ids", label=kind)
            support["latent_ids"] = list(latent_ids)

        object.__setattr__(self, "location_id", location_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "carrier", carrier)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "resolver", resolver)
        object.__setattr__(self, "native_consumer", consumer)
        object.__setattr__(self, "stream", stream)
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "replication_policy", replication_policy)
        object.__setattr__(self, "writer", writer)
        object.__setattr__(self, "invalidates", invalidates)
        object.__setattr__(self, "numerical_contract", numerical_contract)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "metadata", metadata)

    def matches(self, boundary: str, index: int) -> bool:
        return self.boundary == str(boundary) and int(index) in self.indices

    def to_dict(self) -> dict[str, Any]:
        return {
            "location_id": self.location_id,
            "kind": self.kind,
            "boundary": self.boundary,
            "indices": list(self.indices),
            "carrier": self.carrier,
            "support": dict(self.support),
            "resolver": self.resolver,
            "native_consumer": self.native_consumer,
            "stream": self.stream,
            "sites": list(self.sites),
            "replication_policy": self.replication_policy,
            "writer": self.writer,
            "invalidates": list(self.invalidates),
            "numerical_contract": self.numerical_contract,
            "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticLocation:
        if not isinstance(value, Mapping):
            raise SemanticBindingError("semantic location must be an object")
        return cls(
            location_id=str(value.get("location_id") or ""),
            kind=str(value.get("kind") or ""),
            boundary=str(value.get("boundary") or ""),
            indices=tuple(value.get("indices") or ()),
            carrier=str(value.get("carrier") or ""),
            support=dict(value.get("support") or {}),
            resolver=str(value.get("resolver") or ""),
            native_consumer=str(value.get("native_consumer") or ""),
            stream=value.get("stream"),
            sites=tuple(value.get("sites") or ()),
            replication_policy=str(
                value.get("replication_policy") or "all-listed-sites"
            ),
            writer=value.get("writer"),
            invalidates=tuple(value.get("invalidates") or ()),
            numerical_contract=str(value.get("numerical_contract") or "bounded"),
            confidence=value.get("confidence", 0.0),
            evidence_refs=tuple(value.get("evidence_refs") or ()),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class SemanticBinding:
    """A fingerprinted semantic symbol plus context-local physical locations."""

    symbol: str
    semantic_type: str
    manifest_fingerprint: str
    context: Mapping[str, Any]
    locations: tuple[SemanticLocation, ...]
    confidence: float
    status: str = "candidate"
    abstain: bool = False
    abstain_reason: str | None = None
    evidence_refs: tuple[str, ...] = ()
    claim_boundary: str = _DEFAULT_CLAIM_BOUNDARY
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        symbol = _text(self.symbol, label="semantic symbol", safe=True)
        semantic_type = _text(self.semantic_type, label="semantic type", safe=True)
        manifest_fingerprint = str(self.manifest_fingerprint)
        if not _SHA256.fullmatch(manifest_fingerprint):
            raise SemanticBindingError("manifest_fingerprint must be a lowercase SHA-256")
        context = _context(self.context)
        locations = tuple(self.locations)
        if any(not isinstance(item, SemanticLocation) for item in locations):
            raise SemanticBindingError("locations must contain SemanticLocation objects")
        location_ids = [item.location_id for item in locations]
        if len(location_ids) != len(set(location_ids)):
            raise SemanticBindingError("semantic binding location IDs must be unique")
        physical_locations = [
            _canonical(
                {
                    "kind": item.kind,
                    "boundary": item.boundary,
                    "indices": list(item.indices),
                    "carrier": item.carrier,
                    "support": dict(item.support),
                    "stream": item.stream,
                    "sites": list(item.sites),
                    "replication_policy": item.replication_policy,
                }
            )
            for item in locations
        ]
        if len(physical_locations) != len(set(physical_locations)):
            raise SemanticBindingError(
                "semantic binding contains duplicate physical locations"
            )
        confidence = _probability(self.confidence, label="binding confidence")
        status = _text(self.status, label="binding status", safe=True)
        if status not in _BINDING_STATUSES:
            raise SemanticBindingError(f"unsupported semantic binding status {status!r}")
        abstain = bool(self.abstain)
        abstain_reason = self.abstain_reason
        if abstain:
            if status != "abstained":
                raise SemanticBindingError("abstain=true requires status='abstained'")
            abstain_reason = _text(abstain_reason, label="abstain_reason")
            if any(item.writer is not None for item in locations):
                raise SemanticBindingError("an abstaining binding cannot advertise a writer")
        elif not locations:
            raise SemanticBindingError("a non-abstaining binding requires a location list")
        elif abstain_reason is not None:
            raise SemanticBindingError("abstain_reason requires abstain=true")
        elif status == "abstained":
            raise SemanticBindingError("status='abstained' requires abstain=true")
        evidence_refs = _strings(self.evidence_refs, label="binding evidence references")
        if status == "validated" and not evidence_refs:
            raise SemanticBindingError("a validated binding requires evidence references")
        claim_boundary = _text(self.claim_boundary, label="binding claim boundary")
        metadata = _json_mapping(self.metadata, label="binding metadata")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "semantic_type", semantic_type)
        object.__setattr__(self, "manifest_fingerprint", manifest_fingerprint)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "abstain", abstain)
        object.__setattr__(self, "abstain_reason", abstain_reason)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "claim_boundary", claim_boundary)
        object.__setattr__(self, "metadata", metadata)

    @property
    def context_fingerprint(self) -> str:
        return _digest(self.context)

    @property
    def writable(self) -> bool:
        return bool(
            not self.abstain
            and self.locations
            and all(
                location.writer is not None
                and location.invalidates
                and location.native_consumer
                and location.resolver
                and location.numerical_contract
                for location in self.locations
            )
        )

    @property
    def boundaries(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(location.boundary for location in self.locations))

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(
            sorted({index for location in self.locations for index in location.indices})
        )

    @property
    def invalidates(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                value for location in self.locations for value in location.invalidates
            )
        )

    @property
    def numerical_contracts(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(location.numerical_contract for location in self.locations)
        )

    def resolve(self, boundary: str, index: int) -> tuple[SemanticLocation, ...]:
        """Return the joint physical address set active at one typed boundary."""

        if self.abstain:
            raise SemanticBindingError(
                f"semantic binding {self.symbol!r} abstained: {self.abstain_reason}"
            )
        matches = tuple(
            location for location in self.locations if location.matches(boundary, index)
        )
        if not matches:
            raise SemanticBindingError(
                f"semantic binding {self.symbol!r} has no location at "
                f"{boundary}[{int(index)}]"
            )
        return matches

    def validate_context(self, active_context: Mapping[str, Any]) -> dict[str, Any]:
        """Fail closed when a binding is reused under a different runtime context."""

        active = _context(active_context)
        if _digest(active) != self.context_fingerprint:
            keys = sorted(
                key
                for key in set(self.context) | set(active)
                if self.context.get(key) != active.get(key)
            )
            raise SemanticBindingError(
                f"semantic binding context mismatch for {self.symbol!r}: {keys}"
            )
        return active

    def validate_manifest(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_object_manifest(manifest)
        if validated["fingerprint"] != self.manifest_fingerprint:
            raise SemanticBindingError(
                f"semantic binding {self.symbol!r} belongs to another object manifest"
            )
        return validated

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": SEMANTIC_BINDING_SCHEMA,
            "symbol": self.symbol,
            "semantic_type": self.semantic_type,
            "manifest_fingerprint": self.manifest_fingerprint,
            "context": dict(self.context),
            "context_fingerprint": self.context_fingerprint,
            "locations": [location.to_dict() for location in self.locations],
            "confidence": self.confidence,
            "status": self.status,
            "abstain": self.abstain,
            "abstain_reason": self.abstain_reason,
            "evidence_refs": list(self.evidence_refs),
            "claim_boundary": self.claim_boundary,
            "metadata": dict(self.metadata),
        }
        if include_fingerprint:
            payload["fingerprint"] = _digest(payload)
        return payload

    @property
    def fingerprint(self) -> str:
        return self.to_dict()["fingerprint"]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticBinding:
        if not isinstance(value, Mapping) or value.get("schema") != SEMANTIC_BINDING_SCHEMA:
            raise SemanticBindingError("unsupported semantic binding schema")
        declared = str(value.get("fingerprint") or "")
        body = {str(key): item for key, item in value.items() if key != "fingerprint"}
        if not _SHA256.fullmatch(declared) or _digest(body) != declared:
            raise SemanticBindingError("semantic binding fingerprint mismatch")
        binding = cls(
            symbol=str(body.get("symbol") or ""),
            semantic_type=str(body.get("semantic_type") or ""),
            manifest_fingerprint=str(body.get("manifest_fingerprint") or ""),
            context=dict(body.get("context") or {}),
            locations=tuple(
                SemanticLocation.from_dict(item) for item in body.get("locations") or ()
            ),
            confidence=body.get("confidence", 0.0),
            status=str(body.get("status") or "candidate"),
            abstain=bool(body.get("abstain", False)),
            abstain_reason=body.get("abstain_reason"),
            evidence_refs=tuple(body.get("evidence_refs") or ()),
            claim_boundary=str(body.get("claim_boundary") or _DEFAULT_CLAIM_BOUNDARY),
            metadata=dict(body.get("metadata") or {}),
        )
        if body.get("context_fingerprint") != binding.context_fingerprint:
            raise SemanticBindingError("semantic binding context fingerprint mismatch")
        return binding


def _symbol_part(value: str) -> str:
    part = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    if not part:
        raise SemanticBindingError("semantic symbol component is empty")
    if not part[0].isalpha():
        part = "v_" + part
    return part


def bind_object_attribute(
    manifest: Mapping[str, Any],
    *,
    object_name: str,
    attribute: str,
    context: Mapping[str, Any],
    boundary: str,
    indices: Sequence[int],
    stream: str,
    resolver: str,
    writer: str | None,
    native_consumer: str,
    address_grain: str = "row",
    support_rows: Sequence[int] | None = None,
    sites: Sequence[str] | None = None,
    invalidates: Sequence[str] = (),
    numerical_contract: str = "bounded",
    confidence: float = 0.0,
    status: str = "candidate",
    evidence_refs: Sequence[str] = (),
    claim_boundary: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SemanticBinding:
    """Bind one manifest object attribute to an explicit conditioner location.

    ``address_grain='row'`` uses the manifest's typed attribute rows unless
    ``support_rows`` is supplied.  ``address_grain='span'`` requires explicit
    contiguous support; Saturn does not invent a T5 window from a Qwen row.
    """

    validated = validate_object_manifest(manifest)
    objects = validated["objects"]
    if object_name not in objects or not isinstance(objects[object_name], Mapping):
        raise SemanticBindingError(f"object manifest has no object {object_name!r}")
    entry = objects[object_name]
    lexical = entry.get("lexical")
    if not isinstance(lexical, Mapping):
        raise SemanticBindingError(f"object {object_name!r} has no lexical address")
    attributes = lexical.get("attribute_rows")
    if not isinstance(attributes, Mapping) or attribute not in attributes:
        raise SemanticBindingError(
            f"object {object_name!r} has no attribute rows for {attribute!r}"
        )
    if address_grain not in {"row", "span"}:
        raise SemanticBindingError("object attribute address_grain must be 'row' or 'span'")
    if address_grain == "span" and support_rows is None:
        raise SemanticBindingError(
            "span lowering requires explicit support_rows; it is never inferred from one row"
        )
    raw_rows = support_rows if support_rows is not None else attributes[attribute]
    rows = _indices(raw_rows, label="object attribute support rows")
    text_positions = int(validated.get("text_positions") or 0)
    if text_positions <= 0 or any(row >= text_positions for row in rows):
        raise SemanticBindingError("object attribute support is outside the text geometry")
    route = tuple(sites) if sites is not None else tuple(entry.get("route") or ())
    if not route:
        raise SemanticBindingError("object attribute binding requires a native route")

    object_part = _symbol_part(object_name)
    attribute_part = _symbol_part(attribute)
    object_collisions = sorted(
        str(name)
        for name in objects
        if str(name) != object_name and _symbol_part(str(name)) == object_part
    )
    if object_collisions:
        raise SemanticBindingError(
            f"object name {object_name!r} has an ambiguous semantic symbol with "
            f"{object_collisions}"
        )
    attribute_collisions = sorted(
        str(name)
        for name in attributes
        if str(name) != attribute and _symbol_part(str(name)) == attribute_part
    )
    if attribute_collisions:
        raise SemanticBindingError(
            f"attribute {attribute!r} has an ambiguous semantic symbol with "
            f"{attribute_collisions}"
        )
    supplied_metadata = dict(metadata or {})
    reserved_metadata = {"object", "attribute", "address_grain"}
    shadowed = sorted(reserved_metadata & set(supplied_metadata))
    if shadowed:
        raise SemanticBindingError(
            f"binding metadata cannot override reserved keys: {shadowed}"
        )
    location = SemanticLocation(
        location_id=f"{object_part}.{attribute_part}.{address_grain}",
        kind="conditioner-row-set" if address_grain == "row" else "conditioner-span",
        boundary=boundary,
        indices=tuple(indices),
        carrier="conditioner",
        support={"rows": list(rows), "address_grain": address_grain},
        resolver=resolver,
        writer=writer,
        native_consumer=native_consumer,
        stream=stream,
        sites=route,
        invalidates=tuple(invalidates),
        numerical_contract=numerical_contract,
        confidence=confidence,
        evidence_refs=tuple(evidence_refs),
        metadata={
            "row_source": (
                "explicit-support_rows"
                if support_rows is not None
                else "object-manifest.attribute_rows"
            )
        },
    )
    return SemanticBinding(
        symbol=f"object.{object_part}.{attribute_part}",
        semantic_type="object.attribute",
        manifest_fingerprint=validated["fingerprint"],
        context=context,
        locations=(location,),
        confidence=confidence,
        status=status,
        evidence_refs=tuple(evidence_refs),
        claim_boundary=claim_boundary or _DEFAULT_CLAIM_BOUNDARY,
        metadata={
            "object": object_name,
            "attribute": attribute,
            "address_grain": address_grain,
            **supplied_metadata,
        },
    )


def binding_diff(
    before: SemanticBinding | Mapping[str, Any],
    after: SemanticBinding | Mapping[str, Any],
) -> dict[str, Any]:
    """Describe a semantic re-lowering without compiling an executable write."""

    left = before if isinstance(before, SemanticBinding) else SemanticBinding.from_dict(before)
    right = after if isinstance(after, SemanticBinding) else SemanticBinding.from_dict(after)
    context_keys = sorted(set(left.context) | set(right.context))
    context_changes = {
        key: {"before": left.context.get(key), "after": right.context.get(key)}
        for key in context_keys
        if left.context.get(key) != right.context.get(key)
    }
    left_locations = [location.to_dict() for location in left.locations]
    right_locations = [location.to_dict() for location in right.locations]
    payload: dict[str, Any] = {
        "schema": "saturn-semantic-binding-diff-v1",
        "before_fingerprint": left.fingerprint,
        "after_fingerprint": right.fingerprint,
        "symbol_before": left.symbol,
        "symbol_after": right.symbol,
        "same_symbol": left.symbol == right.symbol,
        "context_changes": context_changes,
        "location_changes": {
            "before": left_locations,
            "after": right_locations,
            "changed": left_locations != right_locations,
        },
        "status_change": {"before": left.status, "after": right.status},
        "writable_change": {"before": left.writable, "after": right.writable},
        "diagnostic_only": True,
        "compiles_write": False,
    }
    payload["fingerprint"] = _digest(payload)
    return payload


__all__ = [
    "SEMANTIC_BINDING_SCHEMA",
    "SemanticBinding",
    "SemanticBindingError",
    "SemanticLocation",
    "binding_diff",
    "bind_object_attribute",
]
