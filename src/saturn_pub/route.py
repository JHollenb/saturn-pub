"""Metadata-plus-delta route: choose a measured intervention before paying for its bytes.

Saturn-pub already applies a measured intervention to a session and rolls it back
exactly (:meth:`Session.apply` / :meth:`Session.restore`) and binds an ordered,
recipient-locked schedule of measured Acts to one parent
(:mod:`saturn_pub.program`). What it did not have is a *payload-free catalog* of
candidate deltas you can query and choose among before hydrating any heavy
tensor, plus the custody accounting that proves the compact route lost nothing
and that the applied delta rolled back bit-exactly.

This module is that thin layer. A :class:`DeltaCard` is a content-addressed,
tensor-free description of one measured delta: the recipient identity it was
measured against, free-form selection ``tags``, the registered operation that
applies it, a content address (sha256 + shape + dtype + where to fetch the
bytes) for the heavy delta tensor, and the measured effect. A
:class:`MetadataDeltaRoute` indexes many cards and answers :meth:`select`
(``**tags``) purely from metadata; only :meth:`MetadataDeltaRoute.hydrate`
touches bytes, through a caller-owned resolver, deduplicated and byte-accounted.
:func:`apply_delta` then composes the existing ``Session`` lifecycle -- fork a
candidate and a native branch from one parent, apply, continue, compare, and
restore -- into one route-execution receipt whose ``rollback.verified_exact``
and per-element ``restored_elements`` are the public analog of the privately
demonstrated exact rollback.

Adapter-neutral: the same card drives a decoder (add to ``hidden``) and a FLUX
block suffix (add to ``image`` / ``text``). Nothing here is diffusion-specific
and no private vocabulary is carried over.

Relationship to :mod:`saturn_pub.program`: a ``Program`` binds *one* measured
arm's ordered Acts to *one* parent and replays it. This module does not repeat
that -- it reuses ``Session``/``Act``/``Receipt`` and adds only what Program
lacks: a payload-free catalog of *many* candidate deltas, selection before
hydration, a deduplicated resolver seam with byte accounting, and a
native-comparison-plus-exact-rollback receipt. Hydrated card Acts can be fed to
``Program`` or to :func:`apply_delta` interchangeably.

Claim boundary: this is controlled mechanics. A route selects and applies a
measured delta and proves exact custody and exact rollback; it does not assert
that the delta is semantically meaningful or that two model families are
equivalent.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import Act, Session, StateCut
from .values import describe, digest

DELTA_CARD_SCHEMA = "saturn-pub-delta-card-v1"
METADATA_DELTA_ROUTE_SCHEMA = "saturn-pub-metadata-delta-route-v1"
ROUTE_EXECUTION_SCHEMA = "saturn-pub-route-execution-v1"

DEFAULT_CLAIM_BOUNDARY = (
    "controlled mechanics: a selected, content-verified delta applied and rolled "
    "back exactly; not a claim of semantic meaning or cross-family equivalence"
)
_BUILTIN_OPERATIONS = {
    "add": "saturn.builtin.add",
    "replace": "saturn.builtin.replace",
    "zero": "saturn.builtin.zero",
}
_VALUE_OPERATIONS = ("add", "replace")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Selection coordinate keys promoted to keyword arguments on ``select``. Any other
# string-keyed tag is still selectable through ``select(tags={...})``; these are
# only the conventional, adapter-neutral names.
SELECTION_KEYS = ("role", "family", "site", "stream", "step", "consumer")


class RouteError(ValueError):
    """Raised when a metadata-plus-delta route cannot be safely built or used."""


# --- payload-free value discipline (no tensor bytes ever enter the metadata plane) ------


def _payload_free(value: Any, label: str = "value") -> Any:
    """Return a JSON-safe copy, refusing tensor-like payloads and non-finite floats."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RouteError(f"{label} cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        return {str(key): _payload_free(item, f"{label}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_payload_free(item, label) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_payload_free(item, label) for item in value), key=repr)
    if any(hasattr(value, marker) for marker in ("detach", "numpy", "ndim", "shape")):
        raise RouteError(f"raw tensor-like {type(value).__name__} is not allowed in {label}")
    raise RouteError(f"{label} is not JSON-safe: {type(value).__name__}")


def _fingerprint(value: Any) -> str:
    return digest(_payload_free(value))


def _text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise RouteError(f"{label} must be non-empty")
    return result


def _item_size(dtype: Any) -> int:
    name = str(dtype or "").lower()
    if any(token in name for token in ("float64", "double", "int64", "complex64")):
        return 8
    if any(token in name for token in ("float16", "bfloat16", "half", "int16")):
        return 2
    if any(token in name for token in ("uint8", "int8", "bool")):
        return 1
    return 4


def _numel(shape: Any) -> int | None:
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes, bytearray)):
        return None
    total = 1
    try:
        for dimension in shape:
            dimension = int(dimension)
            if dimension < 0:
                return None
            total *= dimension
    except (TypeError, ValueError, OverflowError):
        return None
    return total


def _declared_bytes(ref: Mapping[str, Any]) -> int | None:
    if ref.get("bytes") is not None:
        try:
            value = int(ref["bytes"])
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None
    count = _numel(ref.get("shape"))
    if count is None:
        return None
    return count * _item_size(ref.get("dtype"))


def _reference(ref: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(ref, Mapping):
        raise RouteError(f"{label} must be an object")
    value = _payload_free(ref, label)
    role = _text(value.get("role"), f"{label}.role")
    source_handle = _text(value.get("source_handle"), f"{label}.source_handle")
    sha256 = str(value.get("sha256") or "").removeprefix("sha256:")
    if _SHA256_RE.fullmatch(sha256) is None:
        raise RouteError(f"{label}.sha256 must be a lowercase SHA-256 digest")
    shape = value.get("shape")
    if shape is not None and _numel(shape) is None:
        raise RouteError(f"{label}.shape must contain non-negative dimensions")
    return {**value, "role": role, "source_handle": source_handle, "sha256": sha256}


def _artifact_key(ref: Mapping[str, Any]) -> str:
    return "|".join(
        (str(ref.get("source_handle") or ""), str(ref.get("role") or ""), str(ref["sha256"]))
    )


def _operation_spec(operation: Any) -> dict[str, Any]:
    if not isinstance(operation, Mapping):
        raise RouteError("operation must be an object")
    value = _payload_free(operation, "operation")
    kind = _text(value.get("kind"), "operation.kind")
    if kind not in _BUILTIN_OPERATIONS and "operation_id" not in value:
        raise RouteError(
            "operation.kind must be add, replace, or zero, or declare operation_id/version "
            "and be hydrated with an explicit builder"
        )
    address = _text(value.get("address"), "operation.address")
    spec: dict[str, Any] = {"kind": kind, "address": address}
    if kind in _VALUE_OPERATIONS:
        dose = value.get("dose", 1)
        if isinstance(dose, bool) or not isinstance(dose, (int, float)):
            raise RouteError("operation.dose must be numeric")
        if not math.isfinite(float(dose)):
            raise RouteError("operation.dose must be finite")
        spec["dose"] = dose
    for carry in ("operation_id", "operation_version"):
        if value.get(carry) is not None:
            spec[carry] = str(value[carry])
    return spec


# --- the content-addressed metadata record for one measured delta -----------------------


@dataclass(frozen=True)
class DeltaCard:
    """A tensor-free, content-addressed description of one measured intervention delta."""

    card_id: str
    recipient: Mapping[str, Any]
    tags: Mapping[str, Any]
    operation: Mapping[str, Any]
    artifact_refs: tuple[Mapping[str, Any], ...] = ()
    act_manifest: Mapping[str, Any] | None = None
    effect: Mapping[str, Any] = field(default_factory=dict)
    claim_boundary: str = DEFAULT_CLAIM_BOUNDARY
    limitations: tuple[str, ...] = ()
    schema: str = DELTA_CARD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DELTA_CARD_SCHEMA:
            raise RouteError(f"unsupported delta-card schema {self.schema!r}")
        object.__setattr__(self, "card_id", _text(self.card_id, "card_id"))
        recipient = _payload_free(self.recipient, "recipient")
        for required in ("model_identity", "execution", "boundary", "parent_fingerprint"):
            if required not in recipient:
                raise RouteError(f"recipient must declare {required}")
        object.__setattr__(self, "recipient", recipient)
        tags = _payload_free(self.tags, "tags")
        if not isinstance(tags, Mapping) or not tags:
            raise RouteError("tags must be a non-empty object of selectable scalars")
        for key, item in tags.items():
            if isinstance(item, bool) or not isinstance(item, (str, int)):
                raise RouteError(f"tag {key!r} must be a string or integer")
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "operation", _operation_spec(self.operation))
        refs = tuple(_reference(ref, label=f"artifact_refs[{i}]")
                     for i, ref in enumerate(self.artifact_refs))
        roles = [ref["role"] for ref in refs]
        if len(set(roles)) != len(roles):
            raise RouteError("artifact_refs roles must be unique")
        kind = self.operation["kind"]
        if kind in _VALUE_OPERATIONS and "value" not in roles:
            raise RouteError(f"{kind} operation requires an artifact_ref with role 'value'")
        if kind == "zero" and refs:
            raise RouteError("zero operation does not reference any artifact")
        object.__setattr__(self, "artifact_refs", refs)
        if self.act_manifest is not None:
            object.__setattr__(self, "act_manifest", _payload_free(self.act_manifest, "act_manifest"))
        object.__setattr__(self, "effect", _payload_free(self.effect or {}, "effect"))
        object.__setattr__(self, "claim_boundary", _text(self.claim_boundary, "claim_boundary"))
        limitations = tuple(_text(item, "limitations") for item in self.limitations)
        object.__setattr__(self, "limitations", limitations)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def ref(self, role: str = "value") -> Mapping[str, Any]:
        for candidate in self.artifact_refs:
            if candidate["role"] == role:
                return candidate
        raise RouteError(f"delta card has no artifact ref for role {role!r}")

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "card_id": self.card_id,
            "recipient": dict(self.recipient),
            "tags": dict(self.tags),
            "operation": dict(self.operation),
            "artifact_refs": [dict(ref) for ref in self.artifact_refs],
            "act_manifest": None if self.act_manifest is None else dict(self.act_manifest),
            "effect": dict(self.effect),
            "claim_boundary": self.claim_boundary,
            "limitations": list(self.limitations),
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeltaCard:
        if not isinstance(payload, Mapping):
            raise RouteError("delta card payload must be an object")
        card = cls(
            card_id=str(payload.get("card_id") or ""),
            recipient=payload.get("recipient") or {},
            tags=payload.get("tags") or {},
            operation=payload.get("operation") or {},
            artifact_refs=tuple(payload.get("artifact_refs") or ()),
            act_manifest=payload.get("act_manifest"),
            effect=payload.get("effect") or {},
            claim_boundary=str(payload.get("claim_boundary") or DEFAULT_CLAIM_BOUNDARY),
            limitations=tuple(payload.get("limitations") or ()),
            schema=str(payload.get("schema") or DELTA_CARD_SCHEMA),
        )
        supplied = payload.get("fingerprint")
        if supplied and str(supplied) != card.fingerprint:
            raise RouteError("delta card fingerprint does not match its content")
        return card

    @classmethod
    def from_act(
        cls,
        *,
        card_id: str,
        parent: StateCut,
        act: Act,
        tags: Mapping[str, Any],
        source_handle: str,
        effect: Mapping[str, Any] | None = None,
        limitations: Sequence[str] = (),
    ) -> DeltaCard:
        """Save a measured builtin-operation ``Act`` (bound to ``parent``) as a card.

        The heavy delta tensor is never embedded: its content address (sha256,
        shape, dtype) is read from the Act's own sealed parameters and recorded
        as an artifact reference that a resolver fetches at hydration time.
        """
        operation_id = act.operation_id
        kind = next((name for name, oid in _BUILTIN_OPERATIONS.items() if oid == operation_id), None)
        if kind is None:
            raise RouteError(
                "from_act supports the builtin add/replace/zero operations; for a custom "
                "operation, build DeltaCard directly with an explicit operation spec"
            )
        if len(act.writes) != 1:
            raise RouteError("delta operation must write exactly one address")
        address = act.writes[0]
        manifest = act.manifest()
        operation: dict[str, Any] = {"kind": kind, "address": address}
        refs: list[dict[str, Any]] = []
        if kind in _VALUE_OPERATIONS:
            described = describe(act.parameters["value"])
            if not isinstance(described, Mapping) or described.get("kind") != "tensor":
                raise RouteError(f"{kind} delta requires a tensor value")
            operation["dose"] = act.parameters.get("dose", 1)
            refs.append(
                {
                    "role": "value",
                    "sha256": described["sha256"],
                    "shape": list(described["shape"]),
                    "dtype": str(described["dtype"]),
                    "source_handle": source_handle,
                    "bytes": _declared_bytes({"shape": described["shape"], "dtype": described["dtype"]}),
                }
            )
        return cls(
            card_id=card_id,
            recipient={
                "model_identity": parent.model_identity,
                "execution": dict(parent.execution),
                "boundary": parent.boundary,
                "parent_fingerprint": parent.fingerprint,
            },
            tags=tags,
            operation=operation,
            artifact_refs=tuple(refs),
            act_manifest=manifest,
            effect=effect or {},
            limitations=tuple(limitations),
        )


# --- the payload-free catalog with lazy, deduplicated, byte-accounted hydration ---------

Resolver = Callable[[Mapping[str, Any]], Any]
Builder = Callable[["DeltaCard", Mapping[str, Any]], Act]


class MetadataDeltaRoute:
    """A metadata index over many :class:`DeltaCard`s with lazy, accounted hydration."""

    def __init__(
        self,
        cards: Sequence[DeltaCard | Mapping[str, Any]],
        *,
        transport: Mapping[str, Any] | None = None,
    ) -> None:
        parsed: list[DeltaCard] = []
        seen: set[str] = set()
        for card in cards:
            card = card if isinstance(card, DeltaCard) else DeltaCard.from_dict(card)
            if card.card_id in seen:
                raise RouteError(f"duplicate card_id {card.card_id!r}")
            seen.add(card.card_id)
            parsed.append(card)
        self.cards: tuple[DeltaCard, ...] = tuple(parsed)
        self._by_id = {card.card_id: card for card in self.cards}
        self._index: dict[str, dict[str, set[str]]] = {}
        for card in self.cards:
            for key, value in card.tags.items():
                self._index.setdefault(key, {}).setdefault(str(value), set()).add(card.card_id)
        self.transport = _payload_free(transport or {}, "transport")
        self._cache: dict[str, Any] = {}
        self._metrics = {
            "metadata_queries": 0,
            "metadata_rows_returned": 0,
            "artifact_requests": 0,
            "artifact_hydrations": 0,
            "artifact_cache_hits": 0,
            "declared_bytes_requested": 0,
            "hydrated_bytes": 0,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "schema": METADATA_DELTA_ROUTE_SCHEMA,
                "cards": [card.fingerprint for card in self.cards],
                "transport": self.transport,
            }
        )

    @property
    def declared_total_bytes(self) -> int:
        total = 0
        for card in self.cards:
            for ref in card.artifact_refs:
                declared = _declared_bytes(ref)
                if declared is not None:
                    total += declared
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": METADATA_DELTA_ROUTE_SCHEMA,
            "fingerprint": self.fingerprint,
            "transport": dict(self.transport),
            "cards": [card.to_dict() for card in self.cards],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MetadataDeltaRoute:
        if not isinstance(payload, Mapping) or payload.get("schema") != METADATA_DELTA_ROUTE_SCHEMA:
            raise RouteError("unsupported metadata-delta-route payload")
        route = cls(payload.get("cards") or (), transport=payload.get("transport") or {})
        supplied = payload.get("fingerprint")
        if supplied and str(supplied) != route.fingerprint:
            raise RouteError("metadata-delta-route fingerprint does not match its content")
        return route

    def select(
        self,
        *,
        card_id: str | None = None,
        tags: Mapping[str, Any] | None = None,
        **keyword_tags: Any,
    ) -> tuple[DeltaCard, ...]:
        """Return matching cards from metadata only -- no artifact bytes are touched."""
        query: dict[str, Any] = {}
        for source in (tags or {}, keyword_tags):
            for key, value in source.items():
                if value is None:
                    continue
                if key in query and query[key] != value:
                    raise RouteError(f"conflicting selection for tag {key!r}")
                query[key] = value
        candidate_ids: set[str] | None = None
        if card_id is not None:
            candidate_ids = {card_id} & set(self._by_id)
        for key, value in query.items():
            matches = set(self._index.get(key, {}).get(str(value), ()))
            candidate_ids = matches if candidate_ids is None else candidate_ids & matches
        ids = set(self._by_id) if candidate_ids is None else candidate_ids
        rows = sorted((self._by_id[i] for i in ids), key=lambda card: card.card_id)
        self._metrics["metadata_queries"] += 1
        self._metrics["metadata_rows_returned"] += len(rows)
        return tuple(rows)

    def project(
        self, cards: Sequence[DeltaCard] | None = None, *, fields: Sequence[str] = ()
    ) -> list[dict[str, Any]]:
        """A deterministic, payload-free projection of cards for parity checks."""
        rows = self.cards if cards is None else tuple(cards)
        fields = tuple(fields) or ("card_id", "tags", "operation")
        projected = []
        for card in sorted(rows, key=lambda card: card.card_id):
            payload = card.to_dict()
            projected.append({field: payload.get(field) for field in fields})
        return projected

    def hydrate(
        self,
        card: DeltaCard | str,
        resolver: Resolver,
        *,
        builder: Builder | None = None,
    ) -> Act:
        """Resolve a card's referenced bytes (lazily, deduplicated) and rebuild its Act.

        ``resolver`` is caller-owned so the catalog works against an in-memory
        store, a local content-addressed directory, or a remote object store
        without duplicating custody. Each resolved value is content-verified
        against its reference (sha256, shape, dtype) before it is admitted.
        """
        card = card if isinstance(card, DeltaCard) else self._by_id[card]
        values: dict[str, Any] = {}
        for ref in card.artifact_refs:
            key = _artifact_key(ref)
            self._metrics["artifact_requests"] += 1
            declared = _declared_bytes(ref)
            if declared is not None:
                self._metrics["declared_bytes_requested"] += declared
            if key in self._cache:
                value = self._cache[key]
                self._metrics["artifact_cache_hits"] += 1
            else:
                value = resolver(ref)
                self._verify(ref, value)
                self._cache[key] = value
                self._metrics["artifact_hydrations"] += 1
                if declared is not None:
                    self._metrics["hydrated_bytes"] += declared
            values[ref["role"]] = value
        act = self._build(card, values) if builder is None else builder(card, values)
        if not isinstance(act, Act):
            raise RouteError("builder did not produce an Act")
        act.verify_implementation()
        if card.act_manifest is not None and act.manifest() != dict(card.act_manifest):
            raise RouteError("hydrated Act does not match the card's measured operation")
        return act

    @staticmethod
    def _verify(ref: Mapping[str, Any], value: Any) -> None:
        summary = describe(value)
        if not isinstance(summary, Mapping) or summary.get("kind") != "tensor":
            raise RouteError(f"resolved artifact {ref['role']!r} is not a tensor")
        if summary["sha256"] != ref["sha256"]:
            raise RouteError(f"resolved artifact {ref['role']!r} content does not match its address")
        if ref.get("shape") is not None and list(summary["shape"]) != list(ref["shape"]):
            raise RouteError(f"resolved artifact {ref['role']!r} shape does not match its reference")
        if ref.get("dtype") and str(summary["dtype"]) != str(ref["dtype"]):
            raise RouteError(f"resolved artifact {ref['role']!r} dtype does not match its reference")

    @staticmethod
    def _build(card: DeltaCard, values: Mapping[str, Any]) -> Act:
        operation = card.operation
        kind = operation["kind"]
        address = operation["address"]
        if kind == "zero":
            return Act.zero(address)
        value = values["value"]
        if kind == "add":
            return Act.add(address, value, dose=operation.get("dose", 1))
        if kind == "replace":
            return Act.replace(address, value)
        raise RouteError(
            f"operation kind {kind!r} needs an explicit builder; no builtin constructor"
        )

    def receipt(self) -> dict[str, Any]:
        """Compact custody accounting -- never exposes a cached payload."""
        declared_total = self.declared_total_bytes
        hydrated = self._metrics["hydrated_bytes"]
        selected = self.transport.get("selected_bytes_estimate", hydrated)
        dense = self.transport.get("dense_bytes_estimate", declared_total)
        reduction = (1.0 - selected / dense) if dense and selected is not None else None
        return {
            "schema": ROUTE_EXECUTION_SCHEMA,
            "route_fingerprint": self.fingerprint,
            "raw_payloads_embedded": False,
            "card_count": len(self.cards),
            "declared_total_bytes": declared_total,
            "transport": {
                "policy": "metadata_plus_selected_delta",
                "dense_bytes_estimate": dense,
                "selected_bytes_estimate": selected,
                "estimated_reduction": reduction,
            },
            "cache": dict(self._metrics),
        }


# --- the thin lifecycle bridge: apply a selected delta, compare to native, roll back -----


def apply_delta(
    session: Session,
    acts: Act | Sequence[Act],
    *,
    continuation_steps: int = 0,
    evaluator: Callable[[Session], Mapping[str, Any]] | None = None,
    parent: StateCut | None = None,
) -> dict[str, Any]:
    """Fork a candidate and a native branch from one parent, apply the delta, compare, roll back.

    Returns the two branches plus a sealed route-execution receipt. The receipt's
    ``rollback.verified_exact`` is True and ``rollback.restored_elements`` counts
    every restored tensor element -- the public analog of an exact rollback. This
    reuses ``Session.apply`` / ``continue_`` / ``compare`` / ``restore``; it does
    not reimplement them.
    """
    acts = (acts,) if isinstance(acts, Act) else tuple(acts)
    if not acts:
        raise RouteError("apply_delta requires at least one Act")
    parent = parent or session.capture()
    if continuation_steps < 0:
        raise RouteError("continuation_steps must be non-negative")

    native = session.fork(parent)
    candidate = session.fork(parent)
    apply_receipts = [candidate.apply(act).to_dict() for act in acts]
    if continuation_steps:
        native.continue_(continuation_steps)
        candidate.continue_(continuation_steps)
    comparison = candidate.compare(native, evaluator)
    rollback = candidate.restore(parent)
    rollback_row = rollback.to_dict()

    restored_slots, restored_elements = _count_elements(parent)
    verified = bool(rollback_row.get("verified_exact")) and (
        rollback_row.get("result") == parent.fingerprint
    )
    receipt = {
        "schema": ROUTE_EXECUTION_SCHEMA,
        "model_identity": session.adapter.model_identity,
        "parent": parent.fingerprint,
        "entry_boundary": parent.boundary,
        "continuation_steps": continuation_steps,
        "apply": apply_receipts,
        "effect": {
            "equal_payload": comparison["equal_payload"],
            "changed_vs_native": not comparison["equal_payload"],
            **({"metrics": comparison["metrics"]} if "metrics" in comparison else {}),
        },
        "rollback": {
            "verified_exact": verified,
            "restored": rollback_row.get("result"),
            "matches_parent": rollback_row.get("result") == parent.fingerprint,
            "restored_slots": restored_slots,
            "restored_elements": restored_elements,
            "receipt": rollback_row,
        },
        "raw_payloads_embedded": False,
    }
    receipt["fingerprint"] = _fingerprint(receipt)
    return {
        "native": native,
        "candidate": candidate,
        "parent": parent,
        "comparison": comparison,
        "rollback": rollback,
        "receipt": receipt,
    }


def _count_elements(cut: StateCut) -> tuple[int, int]:
    """Count durable tensor slots and total tensor elements in a cut's payload."""

    def walk(value: Any) -> tuple[int, int]:
        summary = describe(value)
        return _walk_described(summary)

    def _walk_described(summary: Any) -> tuple[int, int]:
        if isinstance(summary, Mapping) and summary.get("kind") == "tensor":
            count = _numel(summary.get("shape")) or 0
            return (1, count)
        slots = elements = 0
        if isinstance(summary, Mapping) and summary.get("kind") == "mapping":
            for item in summary.get("items", ()):
                s, e = _walk_described(item.get("value"))
                slots += s
                elements += e
        elif isinstance(summary, Mapping) and summary.get("kind") in ("list", "tuple"):
            for item in summary.get("items", ()):
                s, e = _walk_described(item)
                slots += s
                elements += e
        return (slots, elements)

    total_slots = total_elements = 0
    for value in cut.payload.values():
        slots, elements = walk(value)
        total_slots += slots
        total_elements += elements
    return total_slots, total_elements


# --- stdlib-only CLI (torch-free): inspect a saved route without touching any bytes ------


def _load_route(path: str) -> MetadataDeltaRoute:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RouteError(f"cannot read route JSON {path}") from exc
    return MetadataDeltaRoute.from_dict(payload)


def _parse_tags(pairs: Sequence[str]) -> dict[str, Any]:
    tags: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise RouteError(f"--tag expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        raw = raw.strip()
        tags[key] = int(raw) if re.fullmatch(r"-?\d+", raw) else raw
    return tags


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a saved metadata-plus-delta route without hydrating any bytes"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="load a route JSON and print its custody receipt")
    validate.add_argument("route")
    select = sub.add_parser("select", help="select cards by tag from metadata only")
    select.add_argument("route")
    select.add_argument("--tag", action="append", default=[], help="key=value (repeatable)")
    select.add_argument("--card-id")
    show = sub.add_parser("show", help="print one card by id")
    show.add_argument("route")
    show.add_argument("card_id")
    args = parser.parse_args(argv)
    route = _load_route(args.route)
    if args.command == "validate":
        print(json.dumps(route.receipt(), indent=2, sort_keys=True))
        return 0
    if args.command == "select":
        cards = route.select(card_id=args.card_id, tags=_parse_tags(args.tag))
        print(
            json.dumps(
                {
                    "schema": ROUTE_EXECUTION_SCHEMA,
                    "route_fingerprint": route.fingerprint,
                    "selected": [card.card_id for card in cards],
                    "projection": route.project(cards),
                    "cache": route.receipt()["cache"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    matches = [item for item in route.cards if item.card_id == args.card_id]
    if not matches:
        raise RouteError(f"unknown card_id {args.card_id!r}")
    print(json.dumps(matches[0].to_dict(), indent=2, sort_keys=True))
    return 0


__all__ = [
    "DELTA_CARD_SCHEMA",
    "DEFAULT_CLAIM_BOUNDARY",
    "METADATA_DELTA_ROUTE_SCHEMA",
    "ROUTE_EXECUTION_SCHEMA",
    "SELECTION_KEYS",
    "DeltaCard",
    "MetadataDeltaRoute",
    "RouteError",
    "apply_delta",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
