"""Versioned teacher-transition records for Saturn's AR training lane.

The record layer deliberately separates trajectory data from transition rows.
One trajectory stores its token sequence once; each transition addresses a
prefix by ``(trajectory_id, prefix_length)``.  This avoids the quadratic prefix
duplication that a naive JSONL export would create while keeping every training
row independently traceable to Frame/Act evidence and retained tensor objects.

These records are evidence indexes, not live tensor containers.  Tensor bytes
remain in a named artifact and are referenced by checksum.  A tensor that was
not retained is represented honestly as fingerprint-only evidence.  Source
receipts resolve into a separately named, content-validated observation archive
instead of becoming orphan hashes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

AR_TEACHER_TRAJECTORY_SCHEMA = "saturn-ar-teacher-trajectory-v1"
AR_TEACHER_TRANSITION_SCHEMA = "saturn-ar-teacher-transition-v1"
AR_TRANSITION_CORPUS_SCHEMA = "saturn-ar-transition-corpus-v1"
AR_TENSOR_REFERENCE_SCHEMA = "saturn-ar-training-tensor-reference-v1"
AR_SUPERSTEP_ADDRESS_SCHEMA = "saturn-ar-superstep-address-v1"
AR_TEACHER_OBSERVATION_ARCHIVE_SCHEMA = "saturn-ar-teacher-observation-archive-v1"

AR_DATA_SPLITS = ("train", "validation", "test")
EXACT_SCALAR_NUMERICAL_LANE = "exact_scalar"
LAST_ROW_NUMERICAL_LANE = "last_row"
ANCESTOR_ROW_NUMERICAL_LANE = "ancestor_row"
AR_NUMERICAL_LANES = (
    EXACT_SCALAR_NUMERICAL_LANE,
    LAST_ROW_NUMERICAL_LANE,
    ANCESTOR_ROW_NUMERICAL_LANE,
    "batch_dependent",
    "learned_surrogate",
)


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _FrozenJsonMapping(dict[str, Any]):
    """A detached JSON object with dict compatibility and no public mutation API."""

    def __init__(self, values: Any = ()) -> None:
        dict.__init__(self, values)

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("immutable JSON mapping does not support mutation")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __reduce_ex__(self, _protocol: int) -> tuple[Any, tuple[Any, ...]]:
        return type(self), (tuple(self.items()),)


class _FrozenJsonSequence(list[Any]):
    """A detached JSON array retaining list equality, pickle, and JSON behavior."""

    def __init__(self, values: Any = ()) -> None:
        list.__init__(self, values)

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("immutable JSON sequence does not support mutation")

    __setitem__ = _immutable
    __delitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable

    def __reduce_ex__(self, _protocol: int) -> tuple[Any, tuple[Any, ...]]:
        return type(self), (tuple(self),)


class _FingerprintCache:
    """Non-field slot so dataclass equality/asdict retain their historical shape."""

    __slots__ = ("_fingerprint",)


def _cached_record_state(value: Any) -> tuple[tuple[Any, ...], str]:
    return tuple(getattr(value, item.name) for item in fields(value)), value._fingerprint


def _restore_cached_record_state(
    value: Any,
    state: tuple[tuple[Any, ...], str],
) -> None:
    record_values, fingerprint = state
    record_fields = fields(value)
    if len(record_values) != len(record_fields):
        raise ValueError("cached record pickle field count changed")
    for item, restored in zip(record_fields, record_values, strict=True):
        object.__setattr__(value, item.name, restored)
    object.__setattr__(value, "_fingerprint", fingerprint)


def _preserve_cached_pickle(record_type: Any) -> Any:
    """Restore cache-aware pickle hooks after ``dataclass(slots=True)`` rewrites them."""

    record_type.__getstate__ = _cached_record_state
    record_type.__setstate__ = _restore_cached_record_state
    return record_type


def _freeze_json(value: Any) -> Any:
    """Detach and recursively freeze one already-JSON-compatible value.

    ``frozen=True`` only protects dataclass attribute assignment.  A plain
    nested ``dict`` or ``list`` would still let a caller mutate record content
    after its identity had been validated (and, now, cached).  The frozen JSON
    containers retain dict/list equality, pickle, ``dataclasses.asdict``, and
    JSON compatibility while refusing ordinary mutation.  :func:`_thaw_json`
    returns independent builtins at explicit serialization boundaries.
    """

    if isinstance(value, (_FrozenJsonMapping, _FrozenJsonSequence)):
        return value
    if isinstance(value, Mapping):
        return _FrozenJsonMapping((key, _freeze_json(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return _FrozenJsonSequence(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Return plain JSON containers without exposing mutable record storage."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw_json(item) for item in value]
    return value


def _require_digest(value: str, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _require_name(value: str, *, label: str) -> str:
    result = str(value)
    if not result or result.strip() != result:
        raise ValueError(f"{label} must be a non-empty, trimmed string")
    return result


@dataclass(frozen=True, slots=True)
class TrainingTensorReference:
    """Checksum-bound reference to one tensor in a corpus artifact."""

    role: str
    sha256: str
    shape: tuple[int, ...]
    dtype: str
    producer: str
    consumer: str
    abi: str
    retained: bool
    artifact: str | None = None
    key: str | None = None

    def __post_init__(self) -> None:
        for name in ("role", "dtype", "producer", "consumer", "abi"):
            object.__setattr__(self, name, _require_name(getattr(self, name), label=name))
        object.__setattr__(self, "sha256", _require_digest(self.sha256, label="tensor sha256"))
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        if not self.shape or any(value < 0 for value in self.shape):
            raise ValueError("tensor shape must contain non-negative dimensions")
        if self.retained:
            if not self.artifact or not self.key:
                raise ValueError("retained tensor references require artifact and key")
        elif self.artifact is not None or self.key is not None:
            raise ValueError("fingerprint-only tensor references cannot name artifact storage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": AR_TENSOR_REFERENCE_SCHEMA,
            "role": self.role,
            "sha256": self.sha256,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "producer": self.producer,
            "consumer": self.consumer,
            "abi": self.abi,
            "retained": self.retained,
            "artifact": self.artifact,
            "key": self.key,
        }


@dataclass(frozen=True, slots=True)
class SuperstepAddress:
    """Separate a logical transition/KV address from physical weight ownership."""

    logical_step_id: int
    weight_bank_id: str
    teacher_layer_start: int
    teacher_layer_end: int
    kv_slot_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.logical_step_id, bool) or self.logical_step_id < 0:
            raise ValueError("logical_step_id must be a non-negative integer")
        object.__setattr__(
            self,
            "weight_bank_id",
            _require_name(self.weight_bank_id, label="weight_bank_id"),
        )
        if (
            isinstance(self.teacher_layer_start, bool)
            or isinstance(self.teacher_layer_end, bool)
            or self.teacher_layer_start < 0
            or self.teacher_layer_end < self.teacher_layer_start
        ):
            raise ValueError("teacher layer range is invalid")
        slots = tuple(int(value) for value in self.kv_slot_ids)
        if len(slots) != len(set(slots)) or any(value < 0 for value in slots):
            raise ValueError("kv_slot_ids must be unique non-negative integers")
        expected = tuple(range(self.teacher_layer_start, self.teacher_layer_end + 1))
        if slots != expected:
            raise ValueError("a grouped superstep must name every teacher KV slot in order")
        object.__setattr__(self, "kv_slot_ids", slots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": AR_SUPERSTEP_ADDRESS_SCHEMA,
            "logical_step_id": self.logical_step_id,
            "weight_bank_id": self.weight_bank_id,
            "teacher_layers": [self.teacher_layer_start, self.teacher_layer_end],
            "kv_slot_ids": list(self.kv_slot_ids),
        }


def validate_superstep_plan(
    steps: Sequence[SuperstepAddress],
    *,
    teacher_layer_count: int,
) -> tuple[SuperstepAddress, ...]:
    """Validate complete logical coverage without conflating it with weight banks."""

    resolved = tuple(steps)
    if teacher_layer_count < 1 or not resolved:
        raise ValueError("a superstep plan requires a positive teacher depth and at least one step")
    logical = [step.logical_step_id for step in resolved]
    if logical != list(range(len(resolved))):
        raise ValueError("logical superstep IDs must be contiguous and ordered from zero")
    covered = tuple(slot for step in resolved for slot in step.kv_slot_ids)
    if covered != tuple(range(teacher_layer_count)):
        raise ValueError("superstep KV slots must cover every teacher layer exactly once")
    # Repeated weight_bank_id values are intentionally legal: physical sharing
    # does not alias the logical transition or its teacher KV slots.
    return resolved


@_preserve_cached_pickle
@dataclass(frozen=True, slots=True)
class TeacherObservationArchive(_FingerprintCache):
    """Content-addressed Act/ledger/retention evidence behind source receipts."""

    model_identity: Mapping[str, Any]
    capture_policy: Mapping[str, Any]
    observations: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        model_identity = _freeze_json(self.model_identity)
        capture_policy = _freeze_json(self.capture_policy)
        observations: dict[str, Mapping[str, Any]] = {}
        for identity, payload in self.observations.items():
            receipt = _require_digest(identity, label="observation receipt")
            resolved = _freeze_json(payload)
            if _canonical_digest(_thaw_json(resolved)) != receipt:
                raise ValueError("observation receipt does not match its content")
            observations[receipt] = resolved
        if not observations:
            raise ValueError("an observation archive cannot be empty")
        object.__setattr__(self, "model_identity", model_identity)
        object.__setattr__(self, "capture_policy", capture_policy)
        object.__setattr__(self, "observations", _freeze_json(observations))
        object.__setattr__(self, "_fingerprint", _canonical_digest(self._payload()))

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _payload(self) -> dict[str, Any]:
        return {
            "model_identity": _thaw_json(self.model_identity),
            "capture_policy": _thaw_json(self.capture_policy),
            "observations": _thaw_json(self.observations),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": AR_TEACHER_OBSERVATION_ARCHIVE_SCHEMA,
            "fingerprint": self.fingerprint,
            **self._payload(),
        }

    def validate_receipts(self, receipts: Sequence[str]) -> None:
        missing = {
            _require_digest(value, label="source receipt")
            for value in receipts
            if value not in self.observations
        }
        if missing:
            raise ValueError(f"observation archive is missing {len(missing)} source receipts")


@dataclass(frozen=True, slots=True)
class TeacherTrajectory:
    """One prompt and teacher continuation, stored once for many transition rows."""

    trajectory_id: str
    prompt_sha256: str
    prompt: str
    split: str
    prompt_length: int
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trajectory_id",
            _require_digest(self.trajectory_id, label="trajectory_id"),
        )
        object.__setattr__(
            self,
            "prompt_sha256",
            _require_digest(self.prompt_sha256, label="prompt_sha256"),
        )
        if self.split not in AR_DATA_SPLITS:
            raise ValueError(f"unsupported split {self.split!r}")
        tokens = tuple(int(value) for value in self.token_ids)
        if not tokens or any(value < 0 for value in tokens):
            raise ValueError("trajectory token IDs must be non-empty and non-negative")
        if self.prompt_length < 1 or self.prompt_length > len(tokens):
            raise ValueError("prompt_length must fall within the trajectory")
        object.__setattr__(self, "token_ids", tokens)

    @classmethod
    def create(
        cls,
        *,
        prompt: str,
        split: str,
        prompt_length: int,
        token_ids: Sequence[int],
        model_fingerprint: str,
    ) -> TeacherTrajectory:
        prompt_bytes = str(prompt).encode("utf-8")
        prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        payload = {
            "prompt_sha256": prompt_sha256,
            "split": split,
            "prompt_length": int(prompt_length),
            "token_ids": [int(value) for value in token_ids],
            "model_fingerprint": _require_digest(
                model_fingerprint,
                label="model_fingerprint",
            ),
        }
        return cls(
            trajectory_id=_canonical_digest(payload),
            prompt_sha256=prompt_sha256,
            prompt=str(prompt),
            split=split,
            prompt_length=int(prompt_length),
            token_ids=tuple(int(value) for value in token_ids),
        )

    def prefix(self, length: int) -> tuple[int, ...]:
        if length < self.prompt_length or length >= len(self.token_ids):
            raise ValueError("transition prefix length is outside the trajectory")
        return self.token_ids[:length]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": AR_TEACHER_TRAJECTORY_SCHEMA,
            "trajectory_id": self.trajectory_id,
            "prompt_sha256": self.prompt_sha256,
            "prompt": self.prompt,
            "split": self.split,
            "prompt_length": self.prompt_length,
            "token_ids": list(self.token_ids),
        }


@dataclass(frozen=True, slots=True)
class TeacherTransition:
    """A replay-addressable target for a grouped AR student transition."""

    example_id: str
    trajectory_id: str
    split: str
    prefix_length: int
    cursor: int
    phase: str
    parent_frame_id: str
    child_frame_id: str
    target_act: str
    numerical_lane: str
    teacher_token_id: int
    teacher_runner_up_id: int
    teacher_margin: float
    supersteps: tuple[SuperstepAddress, ...]
    input_refs: tuple[TrainingTensorReference, ...] = ()
    target_refs: tuple[TrainingTensorReference, ...] = ()
    consumer_refs: tuple[TrainingTensorReference, ...] = ()
    source_receipts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _require_digest(self.example_id, label="example_id"))
        object.__setattr__(
            self,
            "trajectory_id",
            _require_digest(self.trajectory_id, label="trajectory_id"),
        )
        if self.split not in AR_DATA_SPLITS:
            raise ValueError(f"unsupported split {self.split!r}")
        if self.prefix_length < 1 or self.cursor < 0:
            raise ValueError("prefix_length must be positive and cursor non-negative")
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("transition phase must be prefill or decode")
        for name in ("parent_frame_id", "child_frame_id", "target_act"):
            object.__setattr__(self, name, _require_name(getattr(self, name), label=name))
        if self.numerical_lane not in AR_NUMERICAL_LANES:
            raise ValueError(f"unsupported numerical lane {self.numerical_lane!r}")
        if min(self.teacher_token_id, self.teacher_runner_up_id) < 0:
            raise ValueError("teacher token IDs must be non-negative")
        if self.teacher_token_id == self.teacher_runner_up_id:
            raise ValueError("teacher winner and runner-up must differ")
        if not self.supersteps:
            raise ValueError("transition requires at least one superstep address")
        object.__setattr__(self, "supersteps", tuple(self.supersteps))
        object.__setattr__(self, "input_refs", tuple(self.input_refs))
        object.__setattr__(self, "target_refs", tuple(self.target_refs))
        object.__setattr__(self, "consumer_refs", tuple(self.consumer_refs))
        receipts = tuple(
            _require_digest(value, label="source receipt") for value in self.source_receipts
        )
        if not receipts:
            raise ValueError("transition requires immutable source receipt identities")
        object.__setattr__(self, "source_receipts", receipts)
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))
        if _canonical_digest(self._identity_payload()) != self.example_id:
            raise ValueError("example_id does not match transition content")

    @classmethod
    def create(cls, **values: Any) -> TeacherTransition:
        payload = dict(values)
        payload["supersteps"] = tuple(payload.get("supersteps", ()))
        payload["input_refs"] = tuple(payload.get("input_refs", ()))
        payload["target_refs"] = tuple(payload.get("target_refs", ()))
        payload["consumer_refs"] = tuple(payload.get("consumer_refs", ()))
        payload["source_receipts"] = tuple(payload.get("source_receipts", ()))
        payload["metadata"] = dict(payload.get("metadata", {}))
        temporary = cls.__new__(cls)
        for name, value in payload.items():
            object.__setattr__(temporary, name, value)
        identity = temporary._identity_payload()
        return cls(example_id=_canonical_digest(identity), **payload)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "split": self.split,
            "prefix_length": int(self.prefix_length),
            "cursor": int(self.cursor),
            "phase": self.phase,
            "parent_frame_id": self.parent_frame_id,
            "child_frame_id": self.child_frame_id,
            "target_act": self.target_act,
            "numerical_lane": self.numerical_lane,
            "teacher_token_id": int(self.teacher_token_id),
            "teacher_runner_up_id": int(self.teacher_runner_up_id),
            "teacher_margin": float(self.teacher_margin),
            "supersteps": [item.to_dict() for item in self.supersteps],
            "input_refs": [item.to_dict() for item in self.input_refs],
            "target_refs": [item.to_dict() for item in self.target_refs],
            "consumer_refs": [item.to_dict() for item in self.consumer_refs],
            "source_receipts": list(self.source_receipts),
            "metadata": _thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": AR_TEACHER_TRANSITION_SCHEMA,
            "example_id": self.example_id,
            **self._identity_payload(),
        }


@_preserve_cached_pickle
@dataclass(frozen=True, slots=True)
class TransitionCorpus(_FingerprintCache):
    """Validated corpus manifest with prompt-disjoint split enforcement."""

    model_identity: Mapping[str, Any]
    tensor_artifact: str
    observation_artifact: str
    observation_archive_fingerprint: str
    trajectories: tuple[TeacherTrajectory, ...]
    transitions: tuple[TeacherTransition, ...]
    superstep_plan: tuple[SuperstepAddress, ...]
    teacher_layer_count: int
    capture_policy: Mapping[str, Any]
    split_policy: str = "prompt_disjoint"

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_identity", _freeze_json(self.model_identity))
        object.__setattr__(
            self,
            "tensor_artifact",
            _require_name(self.tensor_artifact, label="tensor_artifact"),
        )
        object.__setattr__(
            self,
            "observation_artifact",
            _require_name(self.observation_artifact, label="observation_artifact"),
        )
        object.__setattr__(
            self,
            "observation_archive_fingerprint",
            _require_digest(
                self.observation_archive_fingerprint,
                label="observation_archive_fingerprint",
            ),
        )
        if self.observation_artifact == self.tensor_artifact:
            raise ValueError("tensor and observation artifacts must be separate objects")
        trajectories = tuple(self.trajectories)
        transitions = tuple(self.transitions)
        plan = validate_superstep_plan(
            self.superstep_plan,
            teacher_layer_count=self.teacher_layer_count,
        )
        if not trajectories or not transitions:
            raise ValueError("a transition corpus requires trajectories and transitions")
        if self.split_policy != "prompt_disjoint":
            raise ValueError("AR v1 training corpus requires prompt_disjoint splitting")
        by_id = {item.trajectory_id: item for item in trajectories}
        if len(by_id) != len(trajectories):
            raise ValueError("trajectory IDs must be unique")
        prompt_splits: dict[str, str] = {}
        for trajectory in trajectories:
            previous = prompt_splits.setdefault(trajectory.prompt_sha256, trajectory.split)
            if previous != trajectory.split:
                raise ValueError("one prompt identity cannot appear in multiple splits")
        seen_examples: set[str] = set()
        for transition in transitions:
            trajectory = by_id.get(transition.trajectory_id)
            if trajectory is None:
                raise ValueError("transition refers to an unknown trajectory")
            if transition.split != trajectory.split:
                raise ValueError("transition split disagrees with its trajectory")
            trajectory.prefix(transition.prefix_length)
            expected = trajectory.token_ids[transition.prefix_length]
            if expected != transition.teacher_token_id:
                raise ValueError("transition target disagrees with the deduplicated trajectory")
            if transition.supersteps != plan:
                raise ValueError("transition superstep plan disagrees with corpus plan")
            if transition.example_id in seen_examples:
                raise ValueError("transition example IDs must be unique")
            seen_examples.add(transition.example_id)
            references = (
                *transition.input_refs,
                *transition.target_refs,
                *transition.consumer_refs,
            )
            for reference in references:
                if reference.retained and reference.artifact != self.tensor_artifact:
                    raise ValueError("retained tensor refers outside the corpus artifact")
        object.__setattr__(self, "trajectories", trajectories)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "superstep_plan", plan)
        object.__setattr__(self, "capture_policy", _freeze_json(self.capture_policy))
        object.__setattr__(self, "_fingerprint", _canonical_digest(self._payload()))

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _payload(self) -> dict[str, Any]:
        return {
            "model_identity": _thaw_json(self.model_identity),
            "tensor_artifact": self.tensor_artifact,
            "observation_artifact": self.observation_artifact,
            "observation_archive_fingerprint": self.observation_archive_fingerprint,
            "trajectories": [item.to_dict() for item in self.trajectories],
            "transitions": [item.to_dict() for item in self.transitions],
            "superstep_plan": [item.to_dict() for item in self.superstep_plan],
            "teacher_layer_count": self.teacher_layer_count,
            "capture_policy": _thaw_json(self.capture_policy),
            "split_policy": self.split_policy,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": AR_TRANSITION_CORPUS_SCHEMA,
            "fingerprint": self.fingerprint,
            **self._payload(),
        }


__all__ = [
    "AR_DATA_SPLITS",
    "AR_NUMERICAL_LANES",
    "ANCESTOR_ROW_NUMERICAL_LANE",
    "EXACT_SCALAR_NUMERICAL_LANE",
    "LAST_ROW_NUMERICAL_LANE",
    "AR_SUPERSTEP_ADDRESS_SCHEMA",
    "AR_TEACHER_OBSERVATION_ARCHIVE_SCHEMA",
    "AR_TEACHER_TRAJECTORY_SCHEMA",
    "AR_TEACHER_TRANSITION_SCHEMA",
    "AR_TENSOR_REFERENCE_SCHEMA",
    "AR_TRANSITION_CORPUS_SCHEMA",
    "SuperstepAddress",
    "TeacherObservationArchive",
    "TeacherTrajectory",
    "TeacherTransition",
    "TrainingTensorReference",
    "TransitionCorpus",
    "validate_superstep_plan",
]
