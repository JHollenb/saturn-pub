"""Typed spectral address ABI for semantic and spatial execution.

This module is a model-independent reference implementation.  It defines how
symbols, instances, roles, frames, live queries, and opaque payload references
are represented and compared.  It deliberately grants no native-consumer or
renderer authority.

Canonical identity is carried by unit phasors.  Similarity-bearing semantic
descriptors, runtime weights, uncertainty, and evidence provenance are separate
objects and do not participate in canonical address equality.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Literal


SPECTRAL_ADDRESS_ABI_SCHEMA = "saturn-spectral-address-abi-v1"
SPECTRAL_READ_TRACE_SCHEMA = "saturn-spectral-read-trace-v1"

ReadMetric = Literal["phase_concordance", "cartesian_dot", "exact_identity"]


class SpectralAddressError(ValueError):
    """An address, query, transform, or read violates the declared ABI."""


@dataclass(frozen=True, slots=True)
class SpectralQuantizationProfile:
    profile_id: str
    phase_bits: int = 16
    magnitude_range: tuple[float, float] = (-16.0, 16.0)
    saturation_policy: Literal["clip", "error"] = "clip"
    frequency_layout: str = "flat-v1"
    profile_version: str = "1"

    def __post_init__(self) -> None:
        for name in ("profile_id", "frequency_layout", "profile_version"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if not 1 <= self.phase_bits <= 32:
            raise SpectralAddressError("phase_bits must lie in [1, 32]")
        low, high = self.magnitude_range
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise SpectralAddressError("magnitude range must be finite and increasing")


@dataclass(frozen=True, slots=True)
class SpectralAddressSpaceSpec:
    address_space_id: str
    dimensions: int
    namespaces: tuple[str, ...]
    supported_operations: tuple[str, ...]
    quantization_profile: SpectralQuantizationProfile
    read_kernel_ids: tuple[str, ...] = ()
    frame_ids: tuple[str, ...] = ("world",)
    gauge_policy: str = "consistent-global-phase"
    boundary_conditions: str = "toroidal-core"

    schema: ClassVar[str] = SPECTRAL_ADDRESS_ABI_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "address_space_id", _text(self.address_space_id, "address_space_id")
        )
        if self.dimensions <= 0:
            raise SpectralAddressError("dimensions must be positive")
        for name in ("namespaces", "supported_operations", "frame_ids"):
            values = tuple(_text(value, name) for value in getattr(self, name))
            if not values:
                raise SpectralAddressError(f"{name} must not be empty")
            object.__setattr__(self, name, values)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpectralAddressError(f"{label} must be non-empty text")
    return value.strip()


def _finite_tuple(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise SpectralAddressError(f"{label} must contain finite values")
    return result


def _phases(values: Sequence[float], label: str = "phases") -> tuple[float, ...]:
    return tuple(value % 1.0 for value in _finite_tuple(values, label))


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_phases(label: str, dimensions: int) -> tuple[float, ...]:
    if dimensions <= 0:
        raise SpectralAddressError("dimensions must be positive")
    phases: list[float] = []
    for index in range(dimensions):
        raw = hashlib.sha256(f"{label}\0{index}".encode()).digest()[:8]
        phases.append(int.from_bytes(raw, "big") / float(1 << 64))
    return tuple(phases)


def phase_concordance(left: Sequence[float], right: Sequence[float]) -> float:
    """Mean unit-phasor concordance in ``[-1, 1]``."""

    import numpy as np
    lhs = np.asarray(_phases(left, "left phases"), dtype=np.float64)
    rhs = np.asarray(_phases(right, "right phases"), dtype=np.float64)
    if lhs.shape != rhs.shape:
        raise SpectralAddressError("phase vectors must have equal shape")
    return float(np.cos(2.0 * np.pi * (lhs - rhs)).mean())


def cartesian_phase_score(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine score after expanding each phase to real/imaginary coordinates."""

    import numpy as np
    lhs = np.asarray(_phases(left, "left phases"), dtype=np.float64)
    rhs = np.asarray(_phases(right, "right phases"), dtype=np.float64)
    if lhs.shape != rhs.shape:
        raise SpectralAddressError("phase vectors must have equal shape")
    lhs_xy = np.concatenate((np.cos(2 * np.pi * lhs), np.sin(2 * np.pi * lhs)))
    rhs_xy = np.concatenate((np.cos(2 * np.pi * rhs), np.sin(2 * np.pi * rhs)))
    return float(np.dot(lhs_xy, rhs_xy) / (np.linalg.norm(lhs_xy) * np.linalg.norm(rhs_xy)))


def pack_phases(phases: Sequence[float], profile: SpectralQuantizationProfile) -> tuple[int, ...]:
    """Quantize turns to the declared unsigned packed-phase lattice."""

    levels = 1 << profile.phase_bits
    return tuple(int(math.floor(phase * levels + 0.5)) % levels for phase in _phases(phases))


def unpack_phases(packed: Sequence[int], profile: SpectralQuantizationProfile) -> tuple[float, ...]:
    levels = 1 << profile.phase_bits
    values = tuple(int(value) for value in packed)
    if not values or any(value < 0 or value >= levels for value in values):
        raise SpectralAddressError("packed phases lie outside the numerical profile")
    return tuple(value / levels for value in values)


@dataclass(frozen=True, slots=True)
class SemanticDescriptor:
    """Similarity-bearing parser output; never a canonical primary key."""

    values: tuple[float, ...]
    resolver_id: str
    provenance: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _finite_tuple(self.values, "descriptor"))
        object.__setattr__(self, "resolver_id", _text(self.resolver_id, "resolver_id"))


@dataclass(frozen=True, slots=True)
class CanonicalSymbolKey:
    """Stable exact identity backed only by unit-phase coordinates."""

    symbol_id: str
    identity_phases: tuple[float, ...]
    namespace: str
    symbol_type: str
    address_book_id: str

    def __post_init__(self) -> None:
        for name in ("symbol_id", "namespace", "symbol_type", "address_book_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "identity_phases", _phases(self.identity_phases))

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "identity_phases": list(self.identity_phases),
            "namespace": self.namespace,
            "symbol_type": self.symbol_type,
            "address_book_id": self.address_book_id,
        }


@dataclass(frozen=True, slots=True)
class AddressAlias:
    alias: str
    symbol_id: str
    namespace: str

    def __post_init__(self) -> None:
        for name in ("alias", "symbol_id", "namespace"):
            object.__setattr__(self, name, _text(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class SpectralInstanceKey:
    """Scene-local identity distinct from canonical semantic identity."""

    scene_id: str
    instance_id: str
    instance_phases: tuple[float, ...]
    edge_id: str | None = None
    execution_scope: str = "scene"

    def __post_init__(self) -> None:
        for name in ("scene_id", "instance_id", "execution_scope"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.edge_id is not None:
            object.__setattr__(self, "edge_id", _text(self.edge_id, "edge_id"))
        object.__setattr__(
            self, "instance_phases", _phases(self.instance_phases, "instance phases")
        )

    @classmethod
    def deterministic(
        cls, scene_id: str, instance_id: str, dimensions: int, *, edge_id: str | None = None
    ) -> SpectralInstanceKey:
        label = f"instance:{scene_id}:{instance_id}:{edge_id or ''}"
        return cls(scene_id, instance_id, _hash_phases(label, dimensions), edge_id=edge_id)


@dataclass(frozen=True, slots=True)
class SpectralRoleTransformSpec:
    """Typed, ordered role operator applied before any superposition."""

    role_id: str
    phase_offsets: tuple[float, ...]
    arity: int
    argument_index: int
    input_type: str
    output_type: str
    commutative: bool = False
    inverse_role_id: str | None = None
    exact_unbinding: bool = True
    cleanup_required: bool = False
    frequency_bands: tuple[int, ...] = ()
    spf_conformance: str = "unverified"

    def __post_init__(self) -> None:
        for name in ("role_id", "input_type", "output_type", "spf_conformance"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.arity <= 0 or not 0 <= self.argument_index < self.arity:
            raise SpectralAddressError("argument_index must be within the declared arity")
        if self.inverse_role_id is not None:
            object.__setattr__(
                self, "inverse_role_id", _text(self.inverse_role_id, "inverse_role_id")
            )
        object.__setattr__(self, "phase_offsets", _phases(self.phase_offsets, "role phases"))
        bands = tuple(int(index) for index in self.frequency_bands)
        if bands and (min(bands) < 0 or max(bands) >= len(self.phase_offsets)):
            raise SpectralAddressError("role frequency band lies outside phase layout")
        object.__setattr__(self, "frequency_bands", bands)


@dataclass(frozen=True, slots=True)
class SpectralFrame:
    frame_id: str
    parent_frame_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _text(self.frame_id, "frame_id"))
        if self.parent_frame_id is not None:
            object.__setattr__(
                self, "parent_frame_id", _text(self.parent_frame_id, "parent_frame_id")
            )


@dataclass(frozen=True, slots=True)
class SpectralLogicalAddress:
    canonical_key: CanonicalSymbolKey
    instance_key: SpectralInstanceKey
    role_transform: SpectralRoleTransformSpec
    frame: SpectralFrame

    def __post_init__(self) -> None:
        dimensions = {
            len(self.canonical_key.identity_phases),
            len(self.instance_key.instance_phases),
            len(self.role_transform.phase_offsets),
        }
        if len(dimensions) != 1:
            raise SpectralAddressError("identity, instance, and role phases must share dimensions")

    @property
    def phases(self) -> tuple[float, ...]:
        return tuple(
            (identity + instance + role) % 1.0
            for identity, instance, role in zip(
                self.canonical_key.identity_phases,
                self.instance_key.instance_phases,
                self.role_transform.phase_offsets,
                strict=True,
            )
        )

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "schema": SPECTRAL_ADDRESS_ABI_SCHEMA,
                "canonical": self.canonical_key.to_dict(),
                "scene": self.instance_key.scene_id,
                "instance": self.instance_key.instance_id,
                "edge": self.instance_key.edge_id,
                "role": self.role_transform.role_id,
                "argument_index": self.role_transform.argument_index,
                "frame": self.frame.frame_id,
                "phases": self.phases,
            }
        )


@dataclass(frozen=True, slots=True)
class SpectralSpatialAddress:
    """A logical address plus an explicit spatial group action and frame crossing."""

    logical_address: SpectralLogicalAddress
    source_frame: str
    target_frame: str
    group_action: str
    parameters: tuple[float, ...]
    interpolation: str = "fourier-exact"
    boundary: str = "toroidal"

    def __post_init__(self) -> None:
        for name in ("source_frame", "target_frame", "group_action", "interpolation", "boundary"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "parameters", _finite_tuple(self.parameters, "parameters"))
        if self.group_action != "transform.spatial.translate":
            raise SpectralAddressError(f"unsupported V1 group action {self.group_action!r}")
        if len(self.parameters) != 2:
            raise SpectralAddressError("translation requires x and y parameters")

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "schema": SPECTRAL_ADDRESS_ABI_SCHEMA,
                "logical_address": self.logical_address.fingerprint,
                "source_frame": self.source_frame,
                "target_frame": self.target_frame,
                "group_action": self.group_action,
                "parameters": self.parameters,
                "interpolation": self.interpolation,
                "boundary": self.boundary,
            }
        )


@dataclass(frozen=True, slots=True)
class SpectralWeight:
    """Retrieval salience or operator dose; never address identity."""

    log_magnitude: float = 0.0
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.log_magnitude):
            raise SpectralAddressError("log magnitude must be finite")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise SpectralAddressError("confidence must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class SpectralBinding:
    edge_id: str
    operator_id: str
    arguments: tuple[tuple[str, SpectralLogicalAddress], ...]
    payload_reference: str
    frame: SpectralFrame
    weight: SpectralWeight = SpectralWeight()
    validity_interval: tuple[int, int | None] = (0, None)

    def __post_init__(self) -> None:
        for name in ("edge_id", "operator_id", "payload_reference"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if not self.arguments:
            raise SpectralAddressError("binding requires ordered arguments")
        roles = tuple(role for role, _ in self.arguments)
        if len(roles) != len(set(roles)):
            raise SpectralAddressError("binding argument roles must be unique")
        arities = {address.role_transform.arity for _, address in self.arguments}
        indices = tuple(address.role_transform.argument_index for _, address in self.arguments)
        if len(arities) != 1 or arities != {len(self.arguments)}:
            raise SpectralAddressError("binding arguments must fill one declared arity")
        if indices != tuple(range(len(self.arguments))):
            raise SpectralAddressError("binding arguments must follow declared argument order")
        if any(role != address.role_transform.role_id for role, address in self.arguments):
            raise SpectralAddressError("binding role labels must match their role transforms")
        if self.validity_interval[0] < 0:
            raise SpectralAddressError("validity interval must start at a non-negative step")
        stop = self.validity_interval[1]
        if stop is not None and stop < self.validity_interval[0]:
            raise SpectralAddressError("validity interval stop precedes start")


@dataclass(frozen=True, slots=True)
class SpectralQuery:
    query_id: str
    phases: tuple[float, ...]
    namespace: str
    symbol_type: str
    frame_id: str
    control_epoch: int = 0
    logical_step: int = 0
    transaction_id: str = "read"
    exact_symbol_id: str | None = None
    exact_scene_id: str | None = None
    exact_instance_id: str | None = None
    exact_role_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("query_id", "namespace", "symbol_type", "frame_id", "transaction_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.control_epoch < 0 or self.logical_step < 0:
            raise SpectralAddressError("control epoch and logical step must be non-negative")
        for name in (
            "exact_symbol_id",
            "exact_scene_id",
            "exact_instance_id",
            "exact_role_id",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        object.__setattr__(self, "phases", _phases(self.phases, "query phases"))


@dataclass(frozen=True, slots=True)
class SpectralReadKernelSpec:
    kernel_id: str
    metric: ReadMetric = "phase_concordance"
    minimum_score: float = 0.0
    calibration_id: str = "uncalibrated-mechanics-only"
    cleanup: Literal["top1", "none"] = "top1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernel_id", _text(self.kernel_id, "kernel_id"))
        object.__setattr__(self, "calibration_id", _text(self.calibration_id, "calibration_id"))
        if self.metric not in {"phase_concordance", "cartesian_dot", "exact_identity"}:
            raise SpectralAddressError(f"unsupported read metric {self.metric!r}")
        if not -1.0 <= self.minimum_score <= 1.0:
            raise SpectralAddressError("minimum score must lie in [-1, 1]")


@dataclass(frozen=True, slots=True)
class SupportResult:
    symbol_registered: bool
    semantic_descriptor_in_support: bool
    role_signature_valid: bool
    instance_scope_valid: bool
    operator_available: bool
    spatial_transform_supported: bool
    coordinate_frame_supported: bool
    backend_lowering_supported: bool
    numerical_profile_supported: bool
    consumer_certified: bool = False

    @property
    def address_supported(self) -> bool:
        return all(
            (
                self.symbol_registered,
                self.semantic_descriptor_in_support,
                self.role_signature_valid,
                self.instance_scope_valid,
                self.coordinate_frame_supported,
                self.numerical_profile_supported,
            )
        )

    @property
    def execution_supported(self) -> bool:
        return self.address_supported and all(
            (
                self.operator_available,
                self.spatial_transform_supported,
                self.backend_lowering_supported,
            )
        )

    @property
    def terminal_supported(self) -> bool:
        return self.execution_supported and self.consumer_certified


@dataclass(frozen=True, slots=True)
class SpectralReadTrace:
    query_id: str
    kernel_id: str
    ranked: tuple[tuple[str, float], ...]
    selected_address_fingerprint: str | None
    selected_payload_hash: str | None
    control_epoch: int
    logical_step: int
    transaction_id: str

    schema: ClassVar[str] = SPECTRAL_READ_TRACE_SCHEMA


@dataclass(frozen=True, slots=True)
class SpectralReadResult:
    selected_binding: SpectralBinding | None
    score: float | None
    support: SupportResult
    trace: SpectralReadTrace
    abstention_reason: str | None = None

    @property
    def abstained(self) -> bool:
        return self.selected_binding is None


@dataclass(frozen=True, slots=True)
class SpectralResolution:
    """Descriptor-to-primary-key resolution event with explicit abstention."""

    ranked_symbol_ids: tuple[tuple[str, float], ...]
    selected_key: CanonicalSymbolKey | None
    support: SupportResult
    resolver_id: str
    calibration_id: str
    uncertainty: float | None
    abstention_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SpectralGaugeSpec:
    gauge_id: str
    phase_shift: tuple[float, ...]
    admitted: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "gauge_id", _text(self.gauge_id, "gauge_id"))
        object.__setattr__(self, "phase_shift", _phases(self.phase_shift, "gauge shift"))


@dataclass(frozen=True, slots=True)
class SpectralGaugeMap:
    spec: SpectralGaugeSpec

    def query(self, query: SpectralQuery) -> SpectralQuery:
        if len(query.phases) != len(self.spec.phase_shift):
            raise SpectralAddressError("gauge and query dimensions differ")
        return replace(
            query,
            phases=tuple(
                (phase + shift) % 1.0
                for phase, shift in zip(query.phases, self.spec.phase_shift, strict=True)
            ),
        )

    def address(self, address: SpectralLogicalAddress) -> SpectralLogicalAddress:
        if len(address.phases) != len(self.spec.phase_shift):
            raise SpectralAddressError("gauge and address dimensions differ")
        key = replace(
            address.canonical_key,
            identity_phases=tuple(
                (phase + shift) % 1.0
                for phase, shift in zip(
                    address.canonical_key.identity_phases,
                    self.spec.phase_shift,
                    strict=True,
                )
            ),
        )
        return replace(address, canonical_key=key)


@dataclass(frozen=True, slots=True)
class AddressEquivalenceResult:
    gauge_id: str
    left_score: float
    transformed_score: float
    absolute_error: float
    equivalent: bool


@dataclass(frozen=True, slots=True)
class AddressCollisionReport:
    dimensions: int
    key_count: int
    maximum_pair_score: float
    colliding_pairs: tuple[tuple[str, str, float], ...]
    threshold: float


@dataclass(frozen=True, slots=True)
class SpectralCapacitySpec:
    dimensions: int
    loads: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    phase_bits: int | None = None
    collision_threshold: float = 0.8

    def __post_init__(self) -> None:
        if self.dimensions <= 0 or not self.loads or min(self.loads) <= 0:
            raise SpectralAddressError("capacity dimensions and loads must be positive")
        if self.phase_bits is not None and self.phase_bits <= 0:
            raise SpectralAddressError("phase bits must be positive")


@dataclass(frozen=True, slots=True)
class AddressCapacityRow:
    load: int
    correct: int
    total: int
    accuracy: float
    maximum_pair_score: float


@dataclass(frozen=True, slots=True)
class AddressCapacityReport:
    spec: SpectralCapacitySpec
    rows: tuple[AddressCapacityRow, ...]
    mechanics_status: Literal["valid", "invalid"]
    trend_status: str
    terminal_status: Literal["not-evaluated"] = "not-evaluated"


class AddressAllocator:
    """Deterministic allocator with explicit finite-capacity collision checks."""

    def __init__(self, address_book_id: str, dimensions: int, collision_threshold: float = 0.8):
        self.address_book_id = _text(address_book_id, "address_book_id")
        if dimensions <= 0:
            raise SpectralAddressError("dimensions must be positive")
        self.dimensions = dimensions
        self.collision_threshold = float(collision_threshold)
        if not -1.0 <= self.collision_threshold <= 1.0:
            raise SpectralAddressError("collision threshold must lie in [-1, 1]")
        self._keys: dict[str, CanonicalSymbolKey] = {}

    def allocate(self, symbol_id: str, namespace: str, symbol_type: str) -> CanonicalSymbolKey:
        identity = f"{namespace}:{symbol_id}"
        if identity in self._keys:
            return self._keys[identity]
        phases = _hash_phases(f"{self.address_book_id}:{identity}", self.dimensions)
        for existing in self._keys.values():
            if phase_concordance(phases, existing.identity_phases) >= self.collision_threshold:
                raise SpectralAddressError(f"candidate key collides with {existing.symbol_id!r}")
        key = CanonicalSymbolKey(
            symbol_id=symbol_id,
            identity_phases=phases,
            namespace=namespace,
            symbol_type=symbol_type,
            address_book_id=self.address_book_id,
        )
        self._keys[identity] = key
        return key

    def collision_report(self) -> AddressCollisionReport:
        collisions: list[tuple[str, str, float]] = []
        maximum = -1.0
        keys = tuple(self._keys.values())
        for index, left in enumerate(keys):
            for right in keys[index + 1 :]:
                score = phase_concordance(left.identity_phases, right.identity_phases)
                maximum = max(maximum, score)
                if score >= self.collision_threshold:
                    collisions.append((left.symbol_id, right.symbol_id, score))
        return AddressCollisionReport(
            self.dimensions,
            len(keys),
            maximum if len(keys) > 1 else -1.0,
            tuple(collisions),
            self.collision_threshold,
        )


def _score(query: SpectralQuery, address: SpectralLogicalAddress, metric: ReadMetric) -> float:
    if query.namespace != address.canonical_key.namespace:
        return -1.0
    if query.symbol_type != address.canonical_key.symbol_type:
        return -1.0
    if query.frame_id != address.frame.frame_id:
        return -1.0
    if metric == "exact_identity":
        if (
            query.exact_symbol_id is None
            or query.exact_scene_id is None
            or query.exact_instance_id is None
        ):
            raise SpectralAddressError(
                "exact identity reads require symbol, scene, and scene-instance identities"
            )
        return float(
            query.exact_symbol_id == address.canonical_key.symbol_id
            and query.exact_scene_id == address.instance_key.scene_id
            and query.exact_instance_id == address.instance_key.instance_id
            and (
                query.exact_role_id is None or query.exact_role_id == address.role_transform.role_id
            )
        )
    if metric == "cartesian_dot":
        return cartesian_phase_score(query.phases, address.phases)
    return phase_concordance(query.phases, address.phases)


def resolve_descriptor(
    descriptor: SemanticDescriptor,
    registry: Mapping[CanonicalSymbolKey, SemanticDescriptor],
    support: SupportResult,
    *,
    minimum_cosine: float,
    calibration_id: str,
) -> SpectralResolution:
    """Resolve semantic similarity to an exact key; never mint implicitly."""
    import numpy as np
    if not -1.0 <= minimum_cosine <= 1.0:
        raise SpectralAddressError("minimum cosine must lie in [-1, 1]")
    ranked: list[tuple[CanonicalSymbolKey, float]] = []
    query_values = np.asarray(descriptor.values, dtype=np.float64)
    for key, candidate in registry.items():
        values = np.asarray(candidate.values, dtype=np.float64)
        if values.shape != query_values.shape:
            raise SpectralAddressError("semantic descriptor dimensions differ")
        denominator = float(np.linalg.norm(query_values) * np.linalg.norm(values))
        score = -1.0 if denominator == 0.0 else float(np.dot(query_values, values) / denominator)
        ranked.append((key, score))
    ranked.sort(key=lambda row: (-row[1], row[0].symbol_id))
    top = ranked[0] if ranked else None
    reason: str | None = None
    if not support.symbol_registered or not support.semantic_descriptor_in_support:
        reason = "semantic-support-failed"
    elif top is None:
        reason = "empty-symbol-registry"
    elif top[1] < minimum_cosine:
        reason = "descriptor-below-calibrated-support"
    selected = None if reason else top[0]
    uncertainty = None if top is None else 1.0 - max(-1.0, min(1.0, top[1]))
    return SpectralResolution(
        tuple((key.symbol_id, score) for key, score in ranked),
        selected,
        support,
        descriptor.resolver_id,
        _text(calibration_id, "calibration_id"),
        uncertainty,
        reason,
    )


def read_bindings(
    query: SpectralQuery,
    bindings: Sequence[SpectralBinding],
    kernel: SpectralReadKernelSpec,
    support: SupportResult,
) -> SpectralReadResult:
    """Resolve a live query and return an opaque payload binding or abstain."""

    candidates: list[tuple[float, SpectralBinding, SpectralLogicalAddress]] = []
    for binding in bindings:
        start, stop = binding.validity_interval
        if query.logical_step < start or (stop is not None and query.logical_step > stop):
            continue
        for _, address in binding.arguments:
            candidates.append((_score(query, address, kernel.metric), binding, address))
    candidates.sort(key=lambda row: (-row[0], row[2].fingerprint))
    ranked = tuple((address.fingerprint, score) for score, _, address in candidates)
    selected = candidates[0] if candidates else None
    reason: str | None = None
    if not support.address_supported:
        reason = "address-support-failed"
    elif selected is None:
        reason = "no-candidate"
    elif kernel.metric == "exact_identity" and selected[0] != 1.0:
        # Exact identity is a discrete contract, not a calibrated similarity.
        # The default score floor is zero and must not admit a zero-score miss.
        reason = "exact-identity-mismatch"
    elif selected[0] < kernel.minimum_score:
        reason = "score-below-calibrated-support"
    elif len({candidate[1].payload_reference for candidate in candidates
              if candidate[0] == selected[0]}) > 1:
        # A stable sort is not an ambiguity resolver. In particular, duplicated
        # addresses with conflicting payloads must not depend on table order.
        reason = "ambiguous-payload-binding"
    binding = None if reason else selected[1]
    address = None if reason else selected[2]
    payload_hash = None if binding is None else _digest(binding.payload_reference)
    trace = SpectralReadTrace(
        query.query_id,
        kernel.kernel_id,
        ranked,
        None if address is None else address.fingerprint,
        payload_hash,
        query.control_epoch,
        query.logical_step,
        query.transaction_id,
    )
    return SpectralReadResult(
        binding,
        None if selected is None else selected[0],
        support,
        trace,
        reason,
    )


def verify_gauge_equivalence(
    query: SpectralQuery,
    address: SpectralLogicalAddress,
    gauge: SpectralGaugeMap,
    *,
    tolerance: float = 1e-12,
) -> AddressEquivalenceResult:
    if not gauge.spec.admitted:
        raise SpectralAddressError("gauge is not admitted by the ABI")
    left = phase_concordance(query.phases, address.phases)
    transformed_query = gauge.query(query)
    transformed_address = gauge.address(address)
    right = phase_concordance(transformed_query.phases, transformed_address.phases)
    error = abs(left - right)
    return AddressEquivalenceResult(gauge.spec.gauge_id, left, right, error, error <= tolerance)


def characterize_capacity(spec: SpectralCapacitySpec) -> AddressCapacityReport:
    """Measure noiseless cleanup and collision pressure across declared loads."""

    rows: list[AddressCapacityRow] = []
    for load in spec.loads:
        phases = [
            _hash_phases(f"capacity:{spec.dimensions}:{index}", spec.dimensions)
            for index in range(load)
        ]
        if spec.phase_bits is not None:
            profile = SpectralQuantizationProfile("capacity", phase_bits=spec.phase_bits)
            phases = [unpack_phases(pack_phases(row, profile), profile) for row in phases]
        correct = 0
        maximum = -1.0
        for index, query in enumerate(phases):
            scores = [phase_concordance(query, candidate) for candidate in phases]
            selected = max(range(load), key=lambda candidate: (scores[candidate], -candidate))
            correct += int(selected == index)
            for other in range(index + 1, load):
                maximum = max(maximum, scores[other])
        rows.append(
            AddressCapacityRow(
                load,
                correct,
                load,
                correct / load,
                maximum if load > 1 else -1.0,
            )
        )
    collision_seen = any(row.maximum_pair_score >= spec.collision_threshold for row in rows)
    return AddressCapacityReport(
        spec,
        tuple(rows),
        "valid",
        "collision-pressure-observed" if collision_seen else "noiseless-cleanup-stable",
    )


def lowered_execution_layout_hash(events: Sequence[Mapping[str, Any]]) -> str:
    """Order-sensitive runtime fingerprint, unlike semantic graph identity."""

    return _digest({"schema": "saturn-spectral-execution-layout-v1", "events": list(events)})


__all__ = [
    "SPECTRAL_ADDRESS_ABI_SCHEMA",
    "SPECTRAL_READ_TRACE_SCHEMA",
    "AddressAlias",
    "AddressAllocator",
    "AddressCapacityReport",
    "AddressCapacityRow",
    "AddressCollisionReport",
    "AddressEquivalenceResult",
    "CanonicalSymbolKey",
    "SemanticDescriptor",
    "SpectralAddressError",
    "SpectralAddressSpaceSpec",
    "SpectralBinding",
    "SpectralCapacitySpec",
    "SpectralFrame",
    "SpectralGaugeMap",
    "SpectralGaugeSpec",
    "SpectralInstanceKey",
    "SpectralLogicalAddress",
    "SpectralQuantizationProfile",
    "SpectralQuery",
    "SpectralReadKernelSpec",
    "SpectralReadResult",
    "SpectralReadTrace",
    "SpectralResolution",
    "SpectralRoleTransformSpec",
    "SpectralWeight",
    "SpectralSpatialAddress",
    "SupportResult",
    "cartesian_phase_score",
    "characterize_capacity",
    "lowered_execution_layout_hash",
    "phase_concordance",
    "pack_phases",
    "read_bindings",
    "resolve_descriptor",
    "unpack_phases",
    "verify_gauge_equivalence",
]
