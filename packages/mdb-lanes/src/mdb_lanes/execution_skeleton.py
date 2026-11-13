"""Immutable, descriptive execution-skeleton records.

The execution skeleton is the stable route grammar recovered by Rosetta
cuts.  It names a specimen, ordered typed route nodes, and the edges between
them.  It intentionally does *not* contain tensor payloads, prompt-local
coefficients, or an executable implementation.  Those belong to dynamic
circuit records, StateCuts, and typed Acts respectively.

The records in this module are safe to keep beside a :class:`Frame` or a
StateCut as a content-addressed reference.  A Frame/StateCut binding stores
only identity and scope; it never embeds the graph in ``Frame.metadata``.
Validation is fail-closed: malformed JSON, duplicate identifiers, dangling
edges, and specimen/fingerprint mismatches raise
``ExecutionSkeletonError`` before a record can be consumed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

EXECUTION_SKELETON_SCHEMA = "saturn-execution-skeleton-v1"
SKELETON_SCHEMA = EXECUTION_SKELETON_SCHEMA
SPECIMEN_CONTRACT_SCHEMA = "saturn-specimen-contract-v1"
SPECIMEN_SCHEMA = SPECIMEN_CONTRACT_SCHEMA
SKELETON_NODE_SCHEMA = "saturn-execution-skeleton-node-v1"
SKELETON_EDGE_SCHEMA = "saturn-execution-skeleton-edge-v1"
FRAME_SKELETON_BINDING_SCHEMA = "saturn-frame-skeleton-binding-v1"
STATECUT_SKELETON_BINDING_SCHEMA = "saturn-statecut-skeleton-binding-v1"
CIRCUIT_ATLAS_REF_SCHEMA = "saturn-circuit-atlas-ref-v1"


class ExecutionSkeletonError(ValueError):
    """Raised when an immutable skeleton record fails closed validation."""


def _jsonable(value: Any, *, label: str = "value") -> Any:
    """Detach a value into the narrow JSON vocabulary used by custody.

    Falling back to ``repr`` would make content identity depend on a process
    or object implementation.  We therefore reject non-JSON values and all
    non-finite floats.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExecutionSkeletonError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExecutionSkeletonError(f"{label} has a non-string key")
            result[key] = _jsonable(item, label=f"{label}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    raise ExecutionSkeletonError(
        f"{label} must contain only JSON scalars, arrays, and objects; got {type(value).__name__}"
    )


def _freeze(value: Any, *, label: str = "value") -> Any:
    """Recursively freeze a JSON-like value."""

    value = _jsonable(value, label=label)
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze(item, label=f"{label}.{key}") for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item, label=f"{label}[{index}]") for index, item in enumerate(value))
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for fingerprints."""

    return json.dumps(
        _jsonable(value, label="canonical content"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 bytes for a JSON-like value."""

    return canonical_json(value).encode("utf-8")


def content_digest(value: Any) -> str:
    """Return a SHA-256 digest over canonical JSON content."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _text(value: Any, label: str, *, default: str | None = None) -> str:
    if value is None and default is not None:
        value = default
    if isinstance(value, bool) or not isinstance(value, str):
        raise ExecutionSkeletonError(f"{label} must be non-empty text")
    result = value.strip()
    if not result:
        raise ExecutionSkeletonError(f"{label} must be non-empty text")
    return result


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecutionSkeletonError(f"{label} must be a JSON object")
    return _freeze(value, label=label)


def _strings(value: Any, label: str, *, sort: bool = False) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    if value is None:
        value = ()
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Sequence):
        raise ExecutionSkeletonError(f"{label} must be an array of strings")
    result = tuple(_text(item, f"{label}[{index}]") for index, item in enumerate(value))
    if sort:
        if len(set(result)) != len(result):
            raise ExecutionSkeletonError(f"{label} contains duplicate identifiers")
        result = tuple(sorted(result))
    return result


def _integers(value: Any, label: str) -> tuple[int, ...]:
    if value is None:
        value = ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ExecutionSkeletonError(f"{label} must be an array of integers")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ExecutionSkeletonError(f"{label}[{index}] must be an integer")
        if item < 0:
            raise ExecutionSkeletonError(f"{label}[{index}] must be non-negative")
        result.append(item)
    return tuple(result)


def _order(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExecutionSkeletonError(f"{label} must be an integer")
    if value < 0:
        raise ExecutionSkeletonError(f"{label} must be non-negative")
    return value


def _verify_fingerprint(raw: Mapping[str, Any], actual: str, label: str) -> None:
    supplied = raw.get("fingerprint")
    if not isinstance(supplied, str) or not supplied:
        raise ExecutionSkeletonError(f"{label} requires a fingerprint")
    if supplied != actual:
        raise ExecutionSkeletonError(f"{label} fingerprint mismatch")


def _verify_schema(raw: Mapping[str, Any], expected: str, label: str) -> None:
    if raw.get("schema") != expected:
        raise ExecutionSkeletonError(
            f"{label} requires schema {expected!r}; got {raw.get('schema')!r}"
        )


def _record(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecutionSkeletonError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class SpecimenContract:
    """The model/runtime boundary to which a skeleton is valid.

    This is an identity contract, not a model loader.  Resolution, latent
    geometry, schedule, and native consumer are descriptive fields used to
    prevent accidental joins across incompatible specimens.
    """

    specimen_id: str
    model_family: str
    model_revision: str = "unknown"
    backend: str = "unknown"
    dtype: str = "unknown"
    resolution: tuple[int, ...] = ()
    latent_geometry: Mapping[str, Any] = field(default_factory=dict)
    schedule: Mapping[str, Any] = field(default_factory=dict)
    native_consumer: tuple[str, ...] = ()
    continuation_horizon: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = SPECIMEN_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "specimen_id", _text(self.specimen_id, "specimen_id"))
        object.__setattr__(self, "model_family", _text(self.model_family, "model_family"))
        object.__setattr__(self, "model_revision", _text(self.model_revision, "model_revision"))
        object.__setattr__(self, "backend", _text(self.backend, "backend"))
        object.__setattr__(self, "dtype", _text(self.dtype, "dtype"))
        object.__setattr__(self, "resolution", _integers(self.resolution, "resolution"))
        object.__setattr__(
            self, "latent_geometry", _mapping(self.latent_geometry, "latent_geometry")
        )
        object.__setattr__(self, "schedule", _mapping(self.schedule, "schedule"))
        object.__setattr__(
            self, "native_consumer", _strings(self.native_consumer, "native_consumer")
        )
        object.__setattr__(
            self,
            "continuation_horizon",
            _mapping(self.continuation_horizon, "continuation_horizon"),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": SPECIMEN_CONTRACT_SCHEMA,
            "specimen_id": self.specimen_id,
            "model_family": self.model_family,
            "model_revision": self.model_revision,
            "backend": self.backend,
            "dtype": self.dtype,
            "resolution": list(self.resolution),
            "latent_geometry": _thaw(self.latent_geometry),
            "schedule": _thaw(self.schedule),
            "native_consumer": list(self.native_consumer),
            "continuation_horizon": _thaw(self.continuation_horizon),
            "metadata": _thaw(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SpecimenContract:
        raw = _record(value, "specimen contract")
        _verify_schema(raw, SPECIMEN_CONTRACT_SCHEMA, "specimen contract")
        result = cls(
            specimen_id=raw.get("specimen_id"),
            model_family=raw.get("model_family"),
            model_revision=raw.get("model_revision", "unknown"),
            backend=raw.get("backend", "unknown"),
            dtype=raw.get("dtype", "unknown"),
            resolution=raw.get("resolution", ()),
            latent_geometry=raw.get("latent_geometry", {}),
            schedule=raw.get("schedule", {}),
            native_consumer=raw.get("native_consumer", ()),
            continuation_horizon=raw.get("continuation_horizon", {}),
            metadata=raw.get("metadata", {}),
        )
        _verify_fingerprint(raw, result.fingerprint, "specimen contract")
        return result

    @property
    def checkpoint(self) -> str:
        """Alias used by older Saturn evidence records."""

        return self.model_revision

    @property
    def consumer(self) -> tuple[str, ...]:
        return self.native_consumer


@dataclass(frozen=True, slots=True)
class SkeletonNode:
    """One ordered, typed physical route node in a skeleton."""

    node_id: str
    kind: str
    label: str | None = None
    coordinate: Mapping[str, Any] = field(default_factory=dict)
    order: int = 0
    phase: str = "unknown"
    abi: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = SKELETON_NODE_SCHEMA

    def __post_init__(self) -> None:
        node_id = _text(self.node_id, "node_id")
        kind = _text(self.kind, "node.kind")
        label = _text(self.label, "node.label", default=node_id)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "coordinate", _mapping(self.coordinate, "node.coordinate"))
        object.__setattr__(self, "order", _order(self.order, "node.order"))
        object.__setattr__(self, "phase", _text(self.phase, "node.phase"))
        object.__setattr__(self, "abi", _mapping(self.abi, "node.abi"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "node.metadata"))
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": SKELETON_NODE_SCHEMA,
            "node_id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "coordinate": _thaw(self.coordinate),
            "order": self.order,
            "phase": self.phase,
            "abi": _thaw(self.abi),
            "metadata": _thaw(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SkeletonNode:
        raw = _record(value, "skeleton node")
        _verify_schema(raw, SKELETON_NODE_SCHEMA, "skeleton node")
        result = cls(
            node_id=raw.get("node_id"),
            kind=raw.get("kind"),
            label=raw.get("label"),
            coordinate=raw.get("coordinate", {}),
            order=raw.get("order", 0),
            phase=raw.get("phase", "unknown"),
            abi=raw.get("abi", {}),
            metadata=raw.get("metadata", {}),
        )
        _verify_fingerprint(raw, result.fingerprint, "skeleton node")
        return result

    @property
    def id(self) -> str:
        return self.node_id

    @property
    def content_hash(self) -> str:
        return self.fingerprint


@dataclass(frozen=True, slots=True)
class SkeletonEdge:
    """One directed, typed route edge between two skeleton nodes."""

    edge_id: str
    source_node_id: str
    target_node_id: str
    relation: str
    order: int = 0
    phase: str = "unknown"
    directed: bool = True
    abi: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = SKELETON_EDGE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_id", _text(self.edge_id, "edge_id"))
        object.__setattr__(
            self, "source_node_id", _text(self.source_node_id, "edge.source_node_id")
        )
        object.__setattr__(
            self, "target_node_id", _text(self.target_node_id, "edge.target_node_id")
        )
        object.__setattr__(self, "relation", _text(self.relation, "edge.relation"))
        object.__setattr__(self, "order", _order(self.order, "edge.order"))
        object.__setattr__(self, "phase", _text(self.phase, "edge.phase"))
        if not isinstance(self.directed, bool):
            raise ExecutionSkeletonError("edge.directed must be boolean")
        object.__setattr__(self, "abi", _mapping(self.abi, "edge.abi"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "edge.metadata"))
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": SKELETON_EDGE_SCHEMA,
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "relation": self.relation,
            "order": self.order,
            "phase": self.phase,
            "directed": self.directed,
            "abi": _thaw(self.abi),
            "metadata": _thaw(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SkeletonEdge:
        raw = _record(value, "skeleton edge")
        _verify_schema(raw, SKELETON_EDGE_SCHEMA, "skeleton edge")
        result = cls(
            edge_id=raw.get("edge_id"),
            source_node_id=raw.get("source_node_id"),
            target_node_id=raw.get("target_node_id"),
            relation=raw.get("relation"),
            order=raw.get("order", 0),
            phase=raw.get("phase", "unknown"),
            directed=raw.get("directed", True),
            abi=raw.get("abi", {}),
            metadata=raw.get("metadata", {}),
        )
        _verify_fingerprint(raw, result.fingerprint, "skeleton edge")
        return result

    @property
    def id(self) -> str:
        return self.edge_id

    @property
    def source(self) -> str:
        return self.source_node_id

    @property
    def target(self) -> str:
        return self.target_node_id

    @property
    def content_hash(self) -> str:
        return self.fingerprint


@dataclass(frozen=True, slots=True)
class CircuitAtlasRef:
    """A non-owning reference from a skeleton to an epistemic atlas."""

    atlas_fingerprint: str
    skeleton_id: str
    skeleton_fingerprint: str
    dynamic_record_ids: tuple[str, ...] = ()
    evidence_roots: tuple[str, ...] = ()
    atlas_schema: str = "saturn-epistemic-circuit-atlas-v1"
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = CIRCUIT_ATLAS_REF_SCHEMA

    authority: ClassVar[str] = "descriptive"
    execution_authority: ClassVar[bool] = False
    writer_authority: ClassVar[bool] = False
    promotion_authority: ClassVar[bool] = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "atlas_fingerprint", _text(self.atlas_fingerprint, "atlas_fingerprint")
        )
        object.__setattr__(self, "skeleton_id", _text(self.skeleton_id, "skeleton_id"))
        object.__setattr__(
            self, "skeleton_fingerprint", _text(self.skeleton_fingerprint, "skeleton_fingerprint")
        )
        object.__setattr__(
            self,
            "dynamic_record_ids",
            _strings(self.dynamic_record_ids, "dynamic_record_ids", sort=True),
        )
        object.__setattr__(
            self, "evidence_roots", _strings(self.evidence_roots, "evidence_roots", sort=True)
        )
        object.__setattr__(self, "atlas_schema", _text(self.atlas_schema, "atlas_schema"))
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": CIRCUIT_ATLAS_REF_SCHEMA,
            "atlas_fingerprint": self.atlas_fingerprint,
            "skeleton_id": self.skeleton_id,
            "skeleton_fingerprint": self.skeleton_fingerprint,
            "dynamic_record_ids": list(self.dynamic_record_ids),
            "evidence_roots": list(self.evidence_roots),
            "atlas_schema": self.atlas_schema,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CircuitAtlasRef:
        raw = _record(value, "circuit atlas reference")
        _verify_schema(raw, CIRCUIT_ATLAS_REF_SCHEMA, "circuit atlas reference")
        result = cls(
            atlas_fingerprint=raw.get("atlas_fingerprint"),
            skeleton_id=raw.get("skeleton_id"),
            skeleton_fingerprint=raw.get("skeleton_fingerprint"),
            dynamic_record_ids=raw.get("dynamic_record_ids", ()),
            evidence_roots=raw.get("evidence_roots", ()),
            atlas_schema=raw.get("atlas_schema", "saturn-epistemic-circuit-atlas-v1"),
        )
        _verify_fingerprint(raw, result.fingerprint, "circuit atlas reference")
        return result

    def validate_against(self, skeleton: ExecutionSkeleton) -> CircuitAtlasRef:
        _validate_skeleton_ref(
            skeleton,
            self.skeleton_id,
            self.skeleton_fingerprint,
            label="circuit atlas reference",
        )
        return self


@dataclass(frozen=True, slots=True)
class FrameSkeletonBinding:
    """An external, non-owning skeleton reference for one Saturn Frame."""

    frame_id: str
    skeleton_id: str
    skeleton_fingerprint: str
    specimen_fingerprint: str
    coordinate_scope: Mapping[str, Any] = field(default_factory=dict)
    state_slots: tuple[str, ...] = ()
    binding_role: str = "observed"
    frame_fingerprint: str | None = None
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = FRAME_SKELETON_BINDING_SCHEMA

    authority: ClassVar[str] = "descriptive"
    execution_authority: ClassVar[bool] = False
    writer_authority: ClassVar[bool] = False
    promotion_authority: ClassVar[bool] = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _text(self.frame_id, "frame_id"))
        object.__setattr__(self, "skeleton_id", _text(self.skeleton_id, "skeleton_id"))
        object.__setattr__(
            self, "skeleton_fingerprint", _text(self.skeleton_fingerprint, "skeleton_fingerprint")
        )
        object.__setattr__(
            self, "specimen_fingerprint", _text(self.specimen_fingerprint, "specimen_fingerprint")
        )
        object.__setattr__(
            self, "coordinate_scope", _mapping(self.coordinate_scope, "coordinate_scope")
        )
        object.__setattr__(
            self, "state_slots", _strings(self.state_slots, "state_slots", sort=True)
        )
        object.__setattr__(self, "binding_role", _text(self.binding_role, "binding_role"))
        object.__setattr__(
            self, "frame_fingerprint", _optional_text(self.frame_fingerprint, "frame_fingerprint")
        )
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": FRAME_SKELETON_BINDING_SCHEMA,
            "frame_id": self.frame_id,
            "skeleton_id": self.skeleton_id,
            "skeleton_fingerprint": self.skeleton_fingerprint,
            "specimen_fingerprint": self.specimen_fingerprint,
            "coordinate_scope": _thaw(self.coordinate_scope),
            "state_slots": list(self.state_slots),
            "binding_role": self.binding_role,
            "frame_fingerprint": self.frame_fingerprint,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_skeleton(
        cls,
        frame_id: str,
        skeleton: ExecutionSkeleton,
        *,
        coordinate_scope: Mapping[str, Any] | None = None,
        state_slots: Sequence[str] = (),
        binding_role: str = "observed",
        frame_fingerprint: str | None = None,
    ) -> FrameSkeletonBinding:
        if not isinstance(skeleton, ExecutionSkeleton):
            raise ExecutionSkeletonError("frame binding requires an ExecutionSkeleton")
        return cls(
            frame_id=frame_id,
            skeleton_id=skeleton.skeleton_id,
            skeleton_fingerprint=skeleton.fingerprint,
            specimen_fingerprint=skeleton.specimen.fingerprint,
            coordinate_scope=coordinate_scope or {},
            state_slots=state_slots,
            binding_role=binding_role,
            frame_fingerprint=frame_fingerprint,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FrameSkeletonBinding:
        raw = _record(value, "frame skeleton binding")
        _verify_schema(raw, FRAME_SKELETON_BINDING_SCHEMA, "frame skeleton binding")
        result = cls(
            frame_id=raw.get("frame_id"),
            skeleton_id=raw.get("skeleton_id"),
            skeleton_fingerprint=raw.get("skeleton_fingerprint"),
            specimen_fingerprint=raw.get("specimen_fingerprint"),
            coordinate_scope=raw.get("coordinate_scope", {}),
            state_slots=raw.get("state_slots", ()),
            binding_role=raw.get("binding_role", "observed"),
            frame_fingerprint=raw.get("frame_fingerprint"),
        )
        _verify_fingerprint(raw, result.fingerprint, "frame skeleton binding")
        return result

    def validate_against(self, skeleton: ExecutionSkeleton) -> FrameSkeletonBinding:
        _validate_skeleton_ref(
            skeleton,
            self.skeleton_id,
            self.skeleton_fingerprint,
            label="frame skeleton binding",
        )
        if self.specimen_fingerprint != skeleton.specimen.fingerprint:
            raise ExecutionSkeletonError("frame skeleton binding specimen mismatch")
        return self

    @property
    def binding_fingerprint(self) -> str:
        return self.fingerprint


@dataclass(frozen=True, slots=True)
class StateCutSkeletonBinding:
    """An external, non-owning skeleton reference for one Saturn StateCut."""

    statecut_id: str
    skeleton_id: str
    skeleton_fingerprint: str
    specimen_fingerprint: str
    boundary: Mapping[str, Any] = field(default_factory=dict)
    coordinate_scope: Mapping[str, Any] = field(default_factory=dict)
    state_schema_id: str | None = None
    slot_names: tuple[str, ...] = ()
    parent_frame_id: str | None = None
    binding_role: str = "observed"
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = STATECUT_SKELETON_BINDING_SCHEMA

    authority: ClassVar[str] = "descriptive"
    execution_authority: ClassVar[bool] = False
    writer_authority: ClassVar[bool] = False
    promotion_authority: ClassVar[bool] = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "statecut_id", _text(self.statecut_id, "statecut_id"))
        object.__setattr__(self, "skeleton_id", _text(self.skeleton_id, "skeleton_id"))
        object.__setattr__(
            self, "skeleton_fingerprint", _text(self.skeleton_fingerprint, "skeleton_fingerprint")
        )
        object.__setattr__(
            self, "specimen_fingerprint", _text(self.specimen_fingerprint, "specimen_fingerprint")
        )
        object.__setattr__(self, "boundary", _mapping(self.boundary, "boundary"))
        object.__setattr__(
            self, "coordinate_scope", _mapping(self.coordinate_scope, "coordinate_scope")
        )
        object.__setattr__(
            self, "state_schema_id", _optional_text(self.state_schema_id, "state_schema_id")
        )
        object.__setattr__(self, "slot_names", _strings(self.slot_names, "slot_names", sort=True))
        object.__setattr__(
            self, "parent_frame_id", _optional_text(self.parent_frame_id, "parent_frame_id")
        )
        object.__setattr__(self, "binding_role", _text(self.binding_role, "binding_role"))
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": STATECUT_SKELETON_BINDING_SCHEMA,
            "statecut_id": self.statecut_id,
            "skeleton_id": self.skeleton_id,
            "skeleton_fingerprint": self.skeleton_fingerprint,
            "specimen_fingerprint": self.specimen_fingerprint,
            "boundary": _thaw(self.boundary),
            "coordinate_scope": _thaw(self.coordinate_scope),
            "state_schema_id": self.state_schema_id,
            "slot_names": list(self.slot_names),
            "parent_frame_id": self.parent_frame_id,
            "binding_role": self.binding_role,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_skeleton(
        cls,
        statecut_id: str,
        skeleton: ExecutionSkeleton,
        *,
        boundary: Mapping[str, Any] | None = None,
        coordinate_scope: Mapping[str, Any] | None = None,
        state_schema_id: str | None = None,
        slot_names: Sequence[str] = (),
        parent_frame_id: str | None = None,
        binding_role: str = "observed",
    ) -> StateCutSkeletonBinding:
        if not isinstance(skeleton, ExecutionSkeleton):
            raise ExecutionSkeletonError("state-cut binding requires an ExecutionSkeleton")
        return cls(
            statecut_id=statecut_id,
            skeleton_id=skeleton.skeleton_id,
            skeleton_fingerprint=skeleton.fingerprint,
            specimen_fingerprint=skeleton.specimen.fingerprint,
            boundary=boundary or {},
            coordinate_scope=coordinate_scope or {},
            state_schema_id=state_schema_id,
            slot_names=slot_names,
            parent_frame_id=parent_frame_id,
            binding_role=binding_role,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StateCutSkeletonBinding:
        raw = _record(value, "state-cut skeleton binding")
        _verify_schema(raw, STATECUT_SKELETON_BINDING_SCHEMA, "state-cut skeleton binding")
        result = cls(
            statecut_id=raw.get("statecut_id"),
            skeleton_id=raw.get("skeleton_id"),
            skeleton_fingerprint=raw.get("skeleton_fingerprint"),
            specimen_fingerprint=raw.get("specimen_fingerprint"),
            boundary=raw.get("boundary", {}),
            coordinate_scope=raw.get("coordinate_scope", {}),
            state_schema_id=raw.get("state_schema_id"),
            slot_names=raw.get("slot_names", ()),
            parent_frame_id=raw.get("parent_frame_id"),
            binding_role=raw.get("binding_role", "observed"),
        )
        _verify_fingerprint(raw, result.fingerprint, "state-cut skeleton binding")
        return result

    def validate_against(self, skeleton: ExecutionSkeleton) -> StateCutSkeletonBinding:
        _validate_skeleton_ref(
            skeleton,
            self.skeleton_id,
            self.skeleton_fingerprint,
            label="state-cut skeleton binding",
        )
        if self.specimen_fingerprint != skeleton.specimen.fingerprint:
            raise ExecutionSkeletonError("state-cut skeleton binding specimen mismatch")
        return self

    @property
    def cut_id(self) -> str:
        return self.statecut_id

    @property
    def binding_fingerprint(self) -> str:
        return self.fingerprint


def _validate_skeleton_ref(
    skeleton: ExecutionSkeleton,
    skeleton_id: str,
    skeleton_fingerprint: str,
    *,
    label: str,
) -> None:
    if not isinstance(skeleton, ExecutionSkeleton):
        raise ExecutionSkeletonError(f"{label} requires an ExecutionSkeleton")
    if skeleton_id != skeleton.skeleton_id:
        raise ExecutionSkeletonError(f"{label} skeleton ID mismatch")
    if skeleton_fingerprint != skeleton.fingerprint:
        raise ExecutionSkeletonError(f"{label} skeleton fingerprint mismatch")


@dataclass(frozen=True, slots=True)
class ExecutionSkeleton:
    """Immutable route grammar shared by a bounded specimen family.

    Node/edge input order is not identity-bearing: nodes are normalized by
    their explicit route ``order`` and then ID, and edges by edge ``order``
    and then ID.  The explicit order fields remain in each content digest, so
    changing route order changes the skeleton even though reordering a JSON
    array does not.

    The object is deliberately passive.  It has no model executor, writer,
    promotion, or Act-registration method.  ``bind_frame`` and
    ``bind_statecut`` only create validated, non-owning references.
    """

    skeleton_id: str
    specimen: SpecimenContract
    nodes: tuple[SkeletonNode, ...]
    edges: tuple[SkeletonEdge, ...]
    coordinate_contract: Mapping[str, Any] = field(default_factory=dict)
    native_consumer: tuple[str, ...] = ()
    continuation_horizon: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    atlas_ref: CircuitAtlasRef | None = None
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = EXECUTION_SKELETON_SCHEMA

    authority: ClassVar[str] = "descriptive"
    execution_authority: ClassVar[bool] = False
    writer_authority: ClassVar[bool] = False
    promotion_authority: ClassVar[bool] = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "skeleton_id", _text(self.skeleton_id, "skeleton_id"))
        if not isinstance(self.specimen, SpecimenContract):
            raise ExecutionSkeletonError("skeleton.specimen must be a SpecimenContract")
        if isinstance(self.nodes, (str, bytes, bytearray)) or not isinstance(self.nodes, Sequence):
            raise ExecutionSkeletonError("skeleton.nodes must be an array")
        if isinstance(self.edges, (str, bytes, bytearray)) or not isinstance(self.edges, Sequence):
            raise ExecutionSkeletonError("skeleton.edges must be an array")

        nodes: list[SkeletonNode] = []
        node_ids: set[str] = set()
        for index, node in enumerate(self.nodes):
            if not isinstance(node, SkeletonNode):
                raise ExecutionSkeletonError(f"skeleton.nodes[{index}] must be a SkeletonNode")
            if node.node_id in node_ids:
                raise ExecutionSkeletonError(f"duplicate skeleton node ID {node.node_id!r}")
            node_ids.add(node.node_id)
            nodes.append(node)

        edges: list[SkeletonEdge] = []
        edge_ids: set[str] = set()
        for index, edge in enumerate(self.edges):
            if not isinstance(edge, SkeletonEdge):
                raise ExecutionSkeletonError(f"skeleton.edges[{index}] must be a SkeletonEdge")
            if edge.edge_id in edge_ids:
                raise ExecutionSkeletonError(f"duplicate skeleton edge ID {edge.edge_id!r}")
            if edge.source_node_id not in node_ids:
                raise ExecutionSkeletonError(
                    f"skeleton edge {edge.edge_id!r} has dangling source {edge.source_node_id!r}"
                )
            if edge.target_node_id not in node_ids:
                raise ExecutionSkeletonError(
                    f"skeleton edge {edge.edge_id!r} has dangling target {edge.target_node_id!r}"
                )
            edge_ids.add(edge.edge_id)
            edges.append(edge)

        object.__setattr__(
            self, "nodes", tuple(sorted(nodes, key=lambda item: (item.order, item.node_id)))
        )
        object.__setattr__(
            self, "edges", tuple(sorted(edges, key=lambda item: (item.order, item.edge_id)))
        )
        object.__setattr__(
            self, "coordinate_contract", _mapping(self.coordinate_contract, "coordinate_contract")
        )
        object.__setattr__(
            self, "native_consumer", _strings(self.native_consumer, "native_consumer")
        )
        object.__setattr__(
            self,
            "continuation_horizon",
            _mapping(self.continuation_horizon, "continuation_horizon"),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        if self.atlas_ref is not None and not isinstance(self.atlas_ref, CircuitAtlasRef):
            raise ExecutionSkeletonError("skeleton.atlas_ref must be a CircuitAtlasRef or None")
        object.__setattr__(self, "fingerprint", content_digest(self._body()))
        if self.atlas_ref is not None:
            self.atlas_ref.validate_against(self)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": EXECUTION_SKELETON_SCHEMA,
            "skeleton_id": self.skeleton_id,
            "specimen": self.specimen.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "coordinate_contract": _thaw(self.coordinate_contract),
            "native_consumer": list(self.native_consumer),
            "continuation_horizon": _thaw(self.continuation_horizon),
            "metadata": _thaw(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        result = {**self._body(), "fingerprint": self.fingerprint}
        if self.atlas_ref is not None:
            result["atlas_ref"] = self.atlas_ref.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExecutionSkeleton:
        raw = _record(value, "execution skeleton")
        _verify_schema(raw, EXECUTION_SKELETON_SCHEMA, "execution skeleton")
        specimen_raw = _record(raw.get("specimen"), "skeleton specimen")
        nodes_raw = raw.get("nodes")
        edges_raw = raw.get("edges")
        if isinstance(nodes_raw, (str, bytes, bytearray)) or not isinstance(nodes_raw, Sequence):
            raise ExecutionSkeletonError("execution skeleton nodes must be an array")
        if isinstance(edges_raw, (str, bytes, bytearray)) or not isinstance(edges_raw, Sequence):
            raise ExecutionSkeletonError("execution skeleton edges must be an array")
        result = cls(
            skeleton_id=raw.get("skeleton_id"),
            specimen=SpecimenContract.from_dict(specimen_raw),
            nodes=tuple(
                SkeletonNode.from_dict(_record(item, "skeleton node")) for item in nodes_raw
            ),
            edges=tuple(
                SkeletonEdge.from_dict(_record(item, "skeleton edge")) for item in edges_raw
            ),
            coordinate_contract=raw.get("coordinate_contract", {}),
            native_consumer=raw.get("native_consumer", ()),
            continuation_horizon=raw.get("continuation_horizon", {}),
            metadata=raw.get("metadata", {}),
            atlas_ref=(
                CircuitAtlasRef.from_dict(_record(raw["atlas_ref"], "circuit atlas reference"))
                if raw.get("atlas_ref") is not None
                else None
            ),
        )
        _verify_fingerprint(raw, result.fingerprint, "execution skeleton")
        return result

    def bind_frame(
        self,
        frame_id: str,
        *,
        coordinate_scope: Mapping[str, Any] | None = None,
        state_slots: Sequence[str] = (),
        binding_role: str = "observed",
        frame_fingerprint: str | None = None,
    ) -> FrameSkeletonBinding:
        return FrameSkeletonBinding.from_skeleton(
            frame_id,
            self,
            coordinate_scope=coordinate_scope,
            state_slots=state_slots,
            binding_role=binding_role,
            frame_fingerprint=frame_fingerprint,
        )

    def bind_statecut(
        self,
        statecut_id: str,
        *,
        boundary: Mapping[str, Any] | None = None,
        coordinate_scope: Mapping[str, Any] | None = None,
        state_schema_id: str | None = None,
        slot_names: Sequence[str] = (),
        parent_frame_id: str | None = None,
        binding_role: str = "observed",
    ) -> StateCutSkeletonBinding:
        return StateCutSkeletonBinding.from_skeleton(
            statecut_id,
            self,
            boundary=boundary,
            coordinate_scope=coordinate_scope,
            state_schema_id=state_schema_id,
            slot_names=slot_names,
            parent_frame_id=parent_frame_id,
            binding_role=binding_role,
        )

    def node(self, node_id: str) -> SkeletonNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def edge(self, edge_id: str) -> SkeletonEdge:
        for edge in self.edges:
            if edge.edge_id == edge_id:
                return edge
        raise KeyError(edge_id)

    @property
    def skeleton_fingerprint(self) -> str:
        return self.fingerprint

    @property
    def content_hash(self) -> str:
        return self.fingerprint


# Descriptive aliases keep the vocabulary explicit for callers that prefer
# the schema name over the shorter record names.
SkeletonSpecimenContract = SpecimenContract
ExecutionSkeletonNode = SkeletonNode
ExecutionSkeletonEdge = SkeletonEdge


__all__ = [
    "CIRCUIT_ATLAS_REF_SCHEMA",
    "CircuitAtlasRef",
    "EXECUTION_SKELETON_SCHEMA",
    "ExecutionSkeleton",
    "ExecutionSkeletonEdge",
    "ExecutionSkeletonError",
    "ExecutionSkeletonNode",
    "FRAME_SKELETON_BINDING_SCHEMA",
    "FrameSkeletonBinding",
    "SKELETON_EDGE_SCHEMA",
    "SKELETON_NODE_SCHEMA",
    "SKELETON_SCHEMA",
    "SPECIMEN_CONTRACT_SCHEMA",
    "SPECIMEN_SCHEMA",
    "STATECUT_SKELETON_BINDING_SCHEMA",
    "SkeletonEdge",
    "SkeletonNode",
    "SkeletonSpecimenContract",
    "SpecimenContract",
    "StateCutSkeletonBinding",
    "canonical_bytes",
    "canonical_json",
    "content_digest",
]
