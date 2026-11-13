"""Model-family-neutral state-cut checkpoints.

The state-cut object is deliberately independent of any executor.  A family
declares a table of slots, then supplies tensor or JSON-like values at one
execution boundary.  The MinIO store serializes tensor slots with
``safetensors`` and keeps JSON slots in the manifest.

Model weights and recomputable derived state do not belong in this object.
They are represented by ``model_identity`` and ``state_schema_id`` and must be
validated by a caller before restore.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from types import MappingProxyType
from typing import Any

STATECUT_CHECKPOINT_SCHEMA = "saturn-statecut-checkpoint-v1"
STATECUT_CHECKPOINT_KIND = "state_cut"
STATECUT_BOUNDARY_KINDS = frozenset(
    {
        "token_cut",
        "layer_cut",
        "step_cut",
        "sequence_cut",
        "expert_cut",
    }
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_CHECKPOINT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class StateCutCheckpointError(ValueError):
    """Raised when a state cut is not portable or does not match its schema."""


def _identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise StateCutCheckpointError(f"{field} must be a string identifier")
    resolved = value
    if not _IDENTIFIER_RE.fullmatch(resolved):
        raise StateCutCheckpointError(
            f"{field} must be a non-empty path-safe identifier"
        )
    return resolved


def _checkpoint_id(value: Any) -> str:
    if not isinstance(value, str):
        raise StateCutCheckpointError("checkpoint_id must be a string")
    resolved = value
    if not _CHECKPOINT_ID_RE.fullmatch(resolved):
        raise StateCutCheckpointError(
            "checkpoint_id must contain only letters, digits, '.', '_' or '-'"
        )
    return resolved


def normalize_json(value: Any, *, field: str = "value") -> Any:
    """Return a detached, canonical JSON value or fail closed.

    Checkpoint identity may never fall back to ``repr``: that would make a
    restore dependent on process-local object formatting.  Mapping keys are
    consequently required to be strings and NaN/Infinity are refused.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StateCutCheckpointError(f"{field} contains a non-finite float")
        return value
    if isinstance(value, (tuple, list)):
        return [normalize_json(item, field=f"{field}[]") for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StateCutCheckpointError(f"{field} contains a non-string key")
            normalized[key] = normalize_json(item, field=f"{field}.{key}")
        return normalized
    raise StateCutCheckpointError(
        f"{field} must contain only JSON scalars, arrays, and objects"
    )


def _freeze_json(value: Any) -> Any:
    """Recursively freeze an already normalized JSON value."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        normalize_json(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_tensor(value: Any) -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is required on execution hosts
        return False
    return isinstance(value, torch.Tensor)


def _copy_tensor(value: Any, *, name: str) -> Any:
    try:
        return value.detach().to(device="cpu").contiguous().clone()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise StateCutCheckpointError(
            f"tensor slot {name!r} cannot be copied to a dense CPU tensor"
        ) from exc


def _tensor_digest(value: Any) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - package dependency
        raise StateCutCheckpointError("tensor fingerprinting requires torch") from exc
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(item) for item in tensor.shape)).encode("ascii"))
    digest.update(b"\x00")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\x00")
    # Scalars cannot be viewed as a different dtype directly.  Flattening also
    # gives empty tensors one stable byte representation without special cases.
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class StateCutBoundary:
    """One durable execution boundary in a model program."""

    kind: str
    index: int
    total: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise StateCutCheckpointError("boundary kind must be a string")
        kind = self.kind
        if kind not in STATECUT_BOUNDARY_KINDS:
            raise StateCutCheckpointError(f"unsupported state-cut boundary kind {kind!r}")
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or isinstance(self.total, bool)
            or not isinstance(self.total, int)
        ):
            raise StateCutCheckpointError("boundary index and total must be integers")
        index = self.index
        total = self.total
        if total <= 0 or not 0 <= index <= total:
            raise StateCutCheckpointError(
                "boundary requires total > 0 and 0 <= index <= total"
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "total", total)

    @classmethod
    def from_value(cls, value: StateCutBoundary | Mapping[str, Any]) -> StateCutBoundary:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise StateCutCheckpointError("boundary must be a StateCutBoundary or object")
        return cls(
            kind=value.get("kind"),
            index=value.get("index"),
            total=value.get("total"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "index": self.index, "total": self.total}


@dataclass(frozen=True, slots=True)
class StateCutIdentity:
    """The exact live model binding required to reopen a state cut."""

    family: str
    state_schema_id: str
    model_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        family = _identifier(self.family, field="family")
        state_schema_id = _identifier(self.state_schema_id, field="state_schema_id")
        if not isinstance(self.model_identity, Mapping) or not self.model_identity:
            raise StateCutCheckpointError("model_identity must be a non-empty JSON object")
        model_identity = _freeze_json(
            normalize_json(self.model_identity, field="model_identity")
        )
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "state_schema_id", state_schema_id)
        object.__setattr__(self, "model_identity", model_identity)

    @classmethod
    def from_value(cls, value: StateCutIdentity | Mapping[str, Any]) -> StateCutIdentity:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise StateCutCheckpointError(
                "expected identity must be StateCutIdentity or an object"
            )
        return cls(
            family=value.get("family"),
            state_schema_id=value.get("state_schema_id"),
            model_identity=value.get("model_identity") or {},
        )

    @property
    def fingerprint(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "state_schema_id": self.state_schema_id,
            "model_identity": normalize_json(self.model_identity),
        }


@dataclass(frozen=True, slots=True)
class StateCutSlotSpec:
    """Declarative type row for one tensor or JSON state slot."""

    kind: str
    required: bool = True
    shape_class: str = "opaque"
    dtype: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise StateCutCheckpointError("slot kind must be a string")
        kind = self.kind
        if kind not in {"tensor", "json"}:
            raise StateCutCheckpointError(f"unsupported state-cut slot kind {kind!r}")
        if not isinstance(self.required, bool):
            raise StateCutCheckpointError("slot required must be boolean")
        shape_class = _identifier(self.shape_class, field="slot shape_class")
        dtype = None if self.dtype is None else str(self.dtype)
        if dtype == "":
            raise StateCutCheckpointError("slot dtype must be non-empty when declared")
        if kind == "json":
            if dtype not in {None, "json"}:
                raise StateCutCheckpointError("JSON slots must use dtype 'json'")
            dtype = "json"
        elif dtype == "json":
            raise StateCutCheckpointError("tensor slots cannot use dtype 'json'")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "shape_class", shape_class)
        object.__setattr__(self, "dtype", dtype)

    @classmethod
    def from_value(cls, value: StateCutSlotSpec | Mapping[str, Any]) -> StateCutSlotSpec:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise StateCutCheckpointError("slot specs must be StateCutSlotSpec or objects")
        return cls(
            kind=value.get("kind"),
            required=value.get("required", True),
            shape_class=str(value.get("shape_class") or "opaque"),
            dtype=value.get("dtype"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "required": self.required,
            "shape_class": self.shape_class,
            "dtype": self.dtype,
        }


def _inferred_slot_spec(value: Any) -> StateCutSlotSpec:
    if _is_tensor(value):
        rank = len(getattr(value, "shape", ()))
        return StateCutSlotSpec(
            kind="tensor",
            required=True,
            shape_class=f"rank_{rank}",
            dtype=str(value.dtype),
        )
    shape_class = "json_scalar" if value is None or isinstance(
        value, (str, bool, int, float)
    ) else "json_structure"
    return StateCutSlotSpec(kind="json", required=True, shape_class=shape_class)


@dataclass(frozen=True, slots=True)
class StateCutCheckpoint:
    """A content-addressable, family-neutral state boundary.

    ``slots`` accepts both tensors and JSON-like values.  ``slot_specs`` is
    optional for convenience; omitted rows are inferred as required rows.
    Family plugins should supply the full table so optional slots and semantic
    shape classes are explicit in the durable manifest.
    """

    checkpoint_id: str
    family: str
    state_schema_id: str
    model_identity: Mapping[str, Any]
    boundary: StateCutBoundary | Mapping[str, Any]
    slots: Mapping[str, Any]
    slot_specs: Mapping[str, StateCutSlotSpec | Mapping[str, Any]] | None = None
    metadata: Mapping[str, Any] | None = None
    schema: str = STATECUT_CHECKPOINT_SCHEMA
    _fingerprint: str = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema != STATECUT_CHECKPOINT_SCHEMA:
            raise StateCutCheckpointError(
                f"unsupported state-cut checkpoint schema {self.schema!r}"
            )
        checkpoint_id = _checkpoint_id(self.checkpoint_id)
        identity = StateCutIdentity(
            family=self.family,
            state_schema_id=self.state_schema_id,
            model_identity=self.model_identity,
        )
        boundary = StateCutBoundary.from_value(self.boundary)
        if not isinstance(self.slots, Mapping) or not self.slots:
            raise StateCutCheckpointError("state-cut checkpoint requires at least one slot")
        if self.slot_specs is not None and not isinstance(self.slot_specs, Mapping):
            raise StateCutCheckpointError("slot_specs must be an object keyed by slot name")

        specs: dict[str, StateCutSlotSpec] = {}
        for raw_name, raw_spec in (self.slot_specs or {}).items():
            name = _identifier(raw_name, field="slot name")
            if name in specs:
                raise StateCutCheckpointError(f"duplicate state-cut slot spec {name!r}")
            specs[name] = StateCutSlotSpec.from_value(raw_spec)

        slots: dict[str, Any] = {}
        for raw_name, raw_value in self.slots.items():
            name = _identifier(raw_name, field="slot name")
            if name in slots:
                raise StateCutCheckpointError(f"duplicate state-cut slot {name!r}")
            spec = specs.get(name) or _inferred_slot_spec(raw_value)
            if self.slot_specs is not None and name not in specs:
                raise StateCutCheckpointError(
                    f"state-cut slot {name!r} is not declared by slot_specs"
                )
            if _is_tensor(raw_value):
                if spec.kind != "tensor":
                    raise StateCutCheckpointError(
                        f"state-cut slot {name!r} is a tensor but its spec is {spec.kind!r}"
                    )
                value = _copy_tensor(raw_value, name=name)
                actual_dtype = str(value.dtype)
                if spec.dtype is not None and spec.dtype != actual_dtype:
                    raise StateCutCheckpointError(
                        f"state-cut slot {name!r} dtype {actual_dtype!r} does not match "
                        f"declared {spec.dtype!r}"
                    )
                if spec.dtype is None:
                    spec = replace(spec, dtype=actual_dtype)
                slots[name] = value
            else:
                if spec.kind != "json":
                    raise StateCutCheckpointError(
                        f"state-cut slot {name!r} is JSON but its spec is {spec.kind!r}"
                    )
                slots[name] = _freeze_json(
                    normalize_json(raw_value, field=f"slots.{name}")
                )
            specs[name] = spec

        for name, spec in specs.items():
            if spec.required and name not in slots:
                raise StateCutCheckpointError(
                    f"state-cut checkpoint is missing required slot {name!r}"
                )
            if spec.kind == "tensor" and spec.dtype is None:
                raise StateCutCheckpointError(
                    f"absent tensor slot {name!r} must declare its dtype"
                )

        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise StateCutCheckpointError("state-cut metadata must be a JSON object")
        metadata = _freeze_json(normalize_json(self.metadata or {}, field="metadata"))
        object.__setattr__(self, "checkpoint_id", checkpoint_id)
        object.__setattr__(self, "family", identity.family)
        object.__setattr__(self, "state_schema_id", identity.state_schema_id)
        object.__setattr__(self, "model_identity", identity.model_identity)
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "slots", MappingProxyType(slots))
        object.__setattr__(self, "slot_specs", MappingProxyType(specs))
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "_fingerprint", self._compute_fingerprint())

    @property
    def identity(self) -> StateCutIdentity:
        return StateCutIdentity(
            family=self.family,
            state_schema_id=self.state_schema_id,
            model_identity=self.model_identity,
        )

    @property
    def tensor_slots(self) -> dict[str, Any]:
        return {
            name: _copy_tensor(self.slots[name], name=name)
            for name, spec in self.slot_specs.items()
            if name in self.slots and spec.kind == "tensor"
        }

    @property
    def json_slots(self) -> dict[str, Any]:
        return {
            name: normalize_json(self.slots[name])
            for name, spec in self.slot_specs.items()
            if name in self.slots and spec.kind == "json"
        }

    @property
    def fingerprint(self) -> str:
        """Return the construction-time identity of this immutable snapshot."""

        return self._fingerprint

    def _compute_fingerprint(self) -> str:
        slot_records: dict[str, Any] = {}
        for name in sorted(self.slot_specs):
            spec = self.slot_specs[name]
            record = {**spec.to_dict(), "present": name in self.slots}
            if name in self.slots:
                value = self.slots[name]
                if spec.kind == "tensor":
                    record.update(
                        {
                            "shape": [int(item) for item in value.shape],
                            "numel": int(value.numel()),
                            "sha256": _tensor_digest(value),
                        }
                    )
                else:
                    record["value"] = normalize_json(value)
            slot_records[name] = record
        payload = {
            "schema": self.schema,
            "checkpoint_id": self.checkpoint_id,
            **self.identity.to_dict(),
            "boundary": self.boundary.to_dict(),
            "slots": slot_records,
            "metadata": normalize_json(self.metadata),
        }
        return _sha256(_canonical_bytes(payload))

    def metadata_only(self) -> dict[str, Any]:
        slots: dict[str, Any] = {}
        for name, spec in self.slot_specs.items():
            record = {**spec.to_dict(), "present": name in self.slots}
            if name in self.slots and spec.kind == "tensor":
                value = self.slots[name]
                record.update(
                    {
                        "shape": [int(item) for item in value.shape],
                        "numel": int(value.numel()),
                    }
                )
            elif name in self.slots:
                record["value"] = normalize_json(self.slots[name])
            slots[name] = record
        return {
            "schema": self.schema,
            "checkpoint_id": self.checkpoint_id,
            "fingerprint": self.fingerprint,
            **self.identity.to_dict(),
            "identity_fingerprint": self.identity.fingerprint,
            "boundary": self.boundary.to_dict(),
            "slots": slots,
            "metadata": normalize_json(self.metadata),
        }


__all__ = [
    "STATECUT_BOUNDARY_KINDS",
    "STATECUT_CHECKPOINT_KIND",
    "STATECUT_CHECKPOINT_SCHEMA",
    "StateCutBoundary",
    "StateCutCheckpoint",
    "StateCutCheckpointError",
    "StateCutIdentity",
    "StateCutSlotSpec",
    "normalize_json",
]
