"""Source-bound intent records for StateCut interventions and reconstruction.

StateCut storage answers *which state existed at a boundary*.  It does not by
itself say whether an edit is meant to affect only the state visible at that
boundary or to reconstruct a counterfactual history from an earlier cut.  This
module makes that distinction explicit without changing any StateCut schema or
executing a model.

The planner consumes an authoritative dependency declaration and records its
transitive dirty closure.  That closure is declared scope: it cannot establish
that an unknown dependency is absent.  Historical intents also carry an
explicit replay schedule.  Parent links become receipt-bound scope only when a
caller supplies authenticated receipt fingerprints for those links; this API
does not turn a bare receipt reference into proof of ancestry.  Restore parity and
native-future replay parity remain independent receipt fields.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

from .authoritative_dependency import DependencyContract, compute_dirty_closure
from .execution_skeleton import content_digest

STATECUT_INTENT_SCHEMA = "saturn-statecut-intent-v1"
STATECUT_INTENT_PLAN_SCHEMA = "saturn-statecut-intent-plan-v1"
STATECUT_INTENT_RECEIPT_SCHEMA = "saturn-statecut-intent-receipt-v1"
STATECUT_REF_SCHEMA = "saturn-statecut-intent-ref-v1"
STATECUT_REPLAY_ITEM_SCHEMA = "saturn-statecut-replay-item-v1"
STATECUT_EXECUTION_EVIDENCE_SCHEMA = "saturn-statecut-execution-evidence-v1"
STATECUT_PARITY_EVIDENCE_SCHEMA = "saturn-statecut-parity-evidence-v1"

PRESENT_STATE_INTERVENTION = "present-state-intervention"
HISTORICAL_COUNTERFACTUAL_RECONSTRUCTION = "historical-counterfactual-reconstruction"
INTENT_KINDS = {
    PRESENT_STATE_INTERVENTION,
    HISTORICAL_COUNTERFACTUAL_RECONSTRUCTION,
}
INPUT_AUTHORITIES = {"caller-supplied", "model-generated"}
PARITY_KINDS = {"restore", "replay"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StateCutIntentError(ValueError):
    """Raised when a StateCut intent or its evidence fails closed."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StateCutIntentError(f"{label} must be non-empty text")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    value = _text(value, label).lower().removeprefix("sha256:")
    if not _SHA256.fullmatch(value):
        raise StateCutIntentError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _optional_sha(value: Any, label: str) -> str | None:
    return None if value is None else _sha(value, label)


def _json(value: Any, label: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StateCutIntentError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StateCutIntentError(f"{label} has a non-string key")
            result[key] = _json(item, f"{label}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json(item, f"{label}[{index}]") for index, item in enumerate(value)]
    raise StateCutIntentError(f"{label} contains unsupported {type(value).__name__}")


def _freeze(value: Any, label: str = "value") -> Any:
    value = _json(value, label)
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze(item, f"{label}.{key}") for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item, f"{label}[]") for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _strings(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise StateCutIntentError(f"{label} must be a sequence of strings")
    result = tuple(_text(item, f"{label}[{index}]") for index, item in enumerate(values))
    if len(set(result)) != len(result):
        raise StateCutIntentError(f"{label} contains duplicates")
    return result


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _json(body, "record")
    normalized["fingerprint"] = content_digest(normalized)
    return normalized


def _verify_seal(value: Mapping[str, Any], schema: str, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StateCutIntentError(f"{label} must be an object")
    body = _json(dict(value), label)
    declared = body.pop("fingerprint", None)
    if body.get("schema") != schema or _sha(declared, f"{label}.fingerprint") != content_digest(
        body
    ):
        raise StateCutIntentError(f"{label} schema or fingerprint mismatch")
    return body


def _validate_plan(value: Mapping[str, Any]) -> tuple[dict[str, Any], "StateCutIntent"]:
    """Hydrate a plan and reject re-sealed disagreement with its embedded intent."""

    body = _verify_seal(value, STATECUT_INTENT_PLAN_SCHEMA, "StateCut intent plan")
    intent = StateCutIntent.from_dict(body.get("intent", {}))
    duplicated = {
        "intent_fingerprint": intent.fingerprint,
        "intent_kind": intent.intent_kind,
        "edit_application_cut": intent.edit_application_cut.to_dict(),
        "rewind_cut": None if intent.rewind_cut is None else intent.rewind_cut.to_dict(),
        "continuation_start_cut": intent.edit_application_cut.to_dict(),
        "observation_cut": intent.observation_cut.to_dict(),
        "execution_history_fingerprint": intent.execution_history_fingerprint,
        "intervention_lineage": list(intent.intervention_lineage),
        "provenance": _thaw(intent.provenance),
        "numerical_contract": intent.numerical_contract,
        "model_identity_fingerprint": intent.model_identity_fingerprint,
        "state_schema_fingerprint": intent.state_schema_fingerprint,
        "dependency_contract_fingerprint": intent.dependency_contract_fingerprint,
        "replay_schedule": [item.to_dict() for item in intent.replay_schedule],
        "native_consumer": intent.native_consumer,
        "continuation_horizon": intent.continuation_horizon,
    }
    for field_name, expected in duplicated.items():
        if body.get(field_name) != expected:
            raise StateCutIntentError(
                f"StateCut intent plan disagrees with embedded intent field {field_name}"
            )
    closure = body.get("declared_dependency_closure")
    if not isinstance(closure, Mapping):
        raise StateCutIntentError("StateCut intent plan lacks a declared dependency closure")
    closure_body = dict(closure)
    closure_fingerprint = closure_body.pop("fingerprint", None)
    if (
        closure_body.get("schema") != "saturn-dirty-closure-v1"
        or _sha(closure_fingerprint, "dirty closure fingerprint") != content_digest(closure_body)
        or closure_body.get("contract_fingerprint") != intent.dependency_contract_fingerprint
        or tuple(closure_body.get("changed_cells", ())) != intent.changed_cells
        or tuple(closure_body.get("changed_addresses", ())) != intent.changed_addresses
        or body.get("execution_cells") != closure_body.get("dirty_cells")
        or body.get("reused_cells") != closure_body.get("reused_cells")
        or body.get("invalidated_addresses") != closure_body.get("invalidated_addresses")
    ):
        raise StateCutIntentError("StateCut intent plan dependency closure binding mismatch")
    return body, intent


@dataclass(frozen=True, slots=True)
class StateCutRef:
    """Opaque StateCut reference plus the digest used by intent identity."""

    cut_ref: str
    cut_fingerprint: str
    parent_cut_ref: str | None = None
    parent_cut_fingerprint: str | None = None
    parent_receipt_fingerprint: str | None = None
    boundary: Mapping[str, Any] = field(default_factory=dict)

    schema: ClassVar[str] = STATECUT_REF_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "cut_ref", _text(self.cut_ref, "cut_ref"))
        object.__setattr__(self, "cut_fingerprint", _sha(self.cut_fingerprint, "cut_fingerprint"))
        parent_ref = (
            None if self.parent_cut_ref is None else _text(self.parent_cut_ref, "parent_cut_ref")
        )
        parent_fp = _optional_sha(self.parent_cut_fingerprint, "parent_cut_fingerprint")
        if (parent_ref is None) != (parent_fp is None):
            raise StateCutIntentError("parent cut ref and fingerprint must be supplied together")
        receipt = _optional_sha(self.parent_receipt_fingerprint, "parent_receipt_fingerprint")
        if receipt is not None and parent_fp is None:
            raise StateCutIntentError("a parent receipt requires a parent cut")
        object.__setattr__(self, "parent_cut_ref", parent_ref)
        object.__setattr__(self, "parent_cut_fingerprint", parent_fp)
        object.__setattr__(self, "parent_receipt_fingerprint", receipt)
        object.__setattr__(self, "boundary", _freeze(self.boundary, "boundary"))

    @property
    def parent_link_receipt_bound(self) -> bool:
        return self.parent_receipt_fingerprint is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cut_ref": self.cut_ref,
            "cut_fingerprint": self.cut_fingerprint,
            "parent_cut_ref": self.parent_cut_ref,
            "parent_cut_fingerprint": self.parent_cut_fingerprint,
            "parent_receipt_fingerprint": self.parent_receipt_fingerprint,
            "boundary": _thaw(self.boundary),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateCutRef":
        if not isinstance(value, Mapping) or value.get("schema") != cls.schema:
            raise StateCutIntentError("StateCut ref schema mismatch")
        return cls(
            cut_ref=value.get("cut_ref"),
            cut_fingerprint=value.get("cut_fingerprint"),
            parent_cut_ref=value.get("parent_cut_ref"),
            parent_cut_fingerprint=value.get("parent_cut_fingerprint"),
            parent_receipt_fingerprint=value.get("parent_receipt_fingerprint"),
            boundary=value.get("boundary", {}),
        )


@dataclass(frozen=True, slots=True)
class ReplayScheduleItem:
    """One declared candidate transition whose input policy is explicit."""

    step_id: str
    sequence: int
    input_authority: str
    input_fingerprint: str
    parent_cut: StateCutRef
    result_cut: StateCutRef
    dependency_cells: Sequence[str] = field(default_factory=tuple)
    input_metadata: Mapping[str, Any] = field(default_factory=dict)

    schema: ClassVar[str] = STATECUT_REPLAY_ITEM_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _text(self.step_id, "replay step_id"))
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise StateCutIntentError("replay sequence must be a non-negative integer")
        authority = _text(self.input_authority, "input_authority")
        if authority not in INPUT_AUTHORITIES:
            raise StateCutIntentError(f"unsupported input authority {authority!r}")
        if not isinstance(self.parent_cut, StateCutRef) or not isinstance(
            self.result_cut, StateCutRef
        ):
            raise StateCutIntentError("replay items require typed parent and result cuts")
        if (
            self.result_cut.parent_cut_ref != self.parent_cut.cut_ref
            or self.result_cut.parent_cut_fingerprint != self.parent_cut.cut_fingerprint
        ):
            raise StateCutIntentError("replay result does not declare its supplied parent")
        object.__setattr__(self, "input_authority", authority)
        object.__setattr__(
            self, "input_fingerprint", _sha(self.input_fingerprint, "input_fingerprint")
        )
        object.__setattr__(
            self, "dependency_cells", _strings(self.dependency_cells, "dependency_cells")
        )
        object.__setattr__(self, "input_metadata", _freeze(self.input_metadata, "input_metadata"))

    @property
    def parent_link_receipt_bound(self) -> bool:
        return self.result_cut.parent_link_receipt_bound

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "step_id": self.step_id,
            "sequence": self.sequence,
            "input_authority": self.input_authority,
            "input_fingerprint": self.input_fingerprint,
            "parent_cut": self.parent_cut.to_dict(),
            "result_cut": self.result_cut.to_dict(),
            "dependency_cells": list(self.dependency_cells),
            "input_metadata": _thaw(self.input_metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplayScheduleItem":
        if not isinstance(value, Mapping) or value.get("schema") != cls.schema:
            raise StateCutIntentError("replay schedule item schema mismatch")
        return cls(
            step_id=value.get("step_id"),
            sequence=value.get("sequence"),
            input_authority=value.get("input_authority"),
            input_fingerprint=value.get("input_fingerprint"),
            parent_cut=StateCutRef.from_dict(value.get("parent_cut", {})),
            result_cut=StateCutRef.from_dict(value.get("result_cut", {})),
            dependency_cells=tuple(value.get("dependency_cells", ())),
            input_metadata=value.get("input_metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class StateCutIntent:
    """An immutable statement of what a StateCut edit is meant to mean."""

    intent_id: str
    intent_kind: str
    edit_application_cut: StateCutRef
    observation_cut: StateCutRef
    model_identity_fingerprint: str
    state_schema_fingerprint: str
    numerical_contract: str
    dependency_contract_fingerprint: str
    changed_cells: Sequence[str] = field(default_factory=tuple)
    changed_addresses: Sequence[str] = field(default_factory=tuple)
    execution_history_fingerprint: str = ""
    intervention_lineage: Sequence[str] = field(default_factory=tuple)
    rewind_cut: StateCutRef | None = None
    replay_schedule: Sequence[ReplayScheduleItem] = field(default_factory=tuple)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    native_consumer: str = "native"
    continuation_horizon: int = 0
    skeleton_fingerprint: str | None = None
    specimen_fingerprint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = STATECUT_INTENT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_id", _text(self.intent_id, "intent_id"))
        kind = _text(self.intent_kind, "intent_kind")
        if kind not in INTENT_KINDS:
            raise StateCutIntentError(f"unsupported StateCut intent kind {kind!r}")
        if not isinstance(self.edit_application_cut, StateCutRef) or not isinstance(
            self.observation_cut, StateCutRef
        ):
            raise StateCutIntentError("intent cuts must be StateCutRef values")
        model = _sha(self.model_identity_fingerprint, "model_identity_fingerprint")
        state_schema = _sha(self.state_schema_fingerprint, "state_schema_fingerprint")
        dependency = _sha(self.dependency_contract_fingerprint, "dependency_contract_fingerprint")
        changed_cells = _strings(self.changed_cells, "changed_cells")
        changed_addresses = _strings(self.changed_addresses, "changed_addresses")
        if not changed_cells and not changed_addresses:
            raise StateCutIntentError("intent requires at least one changed cell or address")
        history = _sha(self.execution_history_fingerprint, "execution_history_fingerprint")
        lineage = _strings(self.intervention_lineage, "intervention_lineage")
        schedule = tuple(self.replay_schedule)
        if any(not isinstance(item, ReplayScheduleItem) for item in schedule):
            raise StateCutIntentError("replay_schedule must contain ReplayScheduleItem values")
        ids = tuple(item.step_id for item in schedule)
        if len(ids) != len(set(ids)) or tuple(item.sequence for item in schedule) != tuple(
            range(len(schedule))
        ):
            raise StateCutIntentError(
                "replay schedule IDs must be unique and sequence must start at zero"
            )
        if kind == PRESENT_STATE_INTERVENTION:
            if self.edit_application_cut.cut_fingerprint != self.observation_cut.cut_fingerprint:
                raise StateCutIntentError("present-state intent must edit the observation cut")
            if self.rewind_cut is not None or schedule:
                raise StateCutIntentError("present-state intent cannot declare rewind or replay")
        else:
            if not isinstance(self.rewind_cut, StateCutRef):
                raise StateCutIntentError("historical reconstruction requires a rewind cut")
            if not schedule:
                raise StateCutIntentError(
                    "historical reconstruction requires an explicit replay schedule"
                )
            if schedule[0].parent_cut.cut_fingerprint != self.rewind_cut.cut_fingerprint:
                raise StateCutIntentError("replay schedule does not begin at the rewind cut")
            for previous, current in zip(schedule, schedule[1:]):
                if (
                    current.parent_cut.cut_ref != previous.result_cut.cut_ref
                    or current.parent_cut.cut_fingerprint != previous.result_cut.cut_fingerprint
                ):
                    raise StateCutIntentError(
                        "replay schedule is not a contiguous declared lineage"
                    )
            if schedule[-1].result_cut.cut_fingerprint != self.observation_cut.cut_fingerprint:
                raise StateCutIntentError("replay schedule does not end at the observation cut")
            path = [schedule[0].parent_cut.cut_fingerprint] + [
                item.result_cut.cut_fingerprint for item in schedule
            ]
            if self.edit_application_cut.cut_fingerprint not in path:
                raise StateCutIntentError("edit application cut is not on the declared rewind path")
        if (
            isinstance(self.continuation_horizon, bool)
            or not isinstance(self.continuation_horizon, int)
            or self.continuation_horizon < 0
        ):
            raise StateCutIntentError("continuation_horizon must be a non-negative integer")
        object.__setattr__(self, "intent_kind", kind)
        object.__setattr__(self, "model_identity_fingerprint", model)
        object.__setattr__(self, "state_schema_fingerprint", state_schema)
        object.__setattr__(
            self, "numerical_contract", _text(self.numerical_contract, "numerical_contract")
        )
        object.__setattr__(self, "dependency_contract_fingerprint", dependency)
        object.__setattr__(self, "changed_cells", changed_cells)
        object.__setattr__(self, "changed_addresses", changed_addresses)
        object.__setattr__(self, "execution_history_fingerprint", history)
        object.__setattr__(self, "intervention_lineage", lineage)
        object.__setattr__(self, "replay_schedule", schedule)
        object.__setattr__(self, "provenance", _freeze(self.provenance, "provenance"))
        object.__setattr__(self, "native_consumer", _text(self.native_consumer, "native_consumer"))
        object.__setattr__(
            self,
            "skeleton_fingerprint",
            _optional_sha(self.skeleton_fingerprint, "skeleton_fingerprint"),
        )
        object.__setattr__(
            self,
            "specimen_fingerprint",
            _optional_sha(self.specimen_fingerprint, "specimen_fingerprint"),
        )
        object.__setattr__(self, "metadata", _freeze(self.metadata, "metadata"))
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "intent_id": self.intent_id,
            "intent_kind": self.intent_kind,
            "edit_application_cut": self.edit_application_cut.to_dict(),
            "observation_cut": self.observation_cut.to_dict(),
            "rewind_cut": None if self.rewind_cut is None else self.rewind_cut.to_dict(),
            "model_identity_fingerprint": self.model_identity_fingerprint,
            "state_schema_fingerprint": self.state_schema_fingerprint,
            "numerical_contract": self.numerical_contract,
            "dependency_contract_fingerprint": self.dependency_contract_fingerprint,
            "skeleton_fingerprint": self.skeleton_fingerprint,
            "specimen_fingerprint": self.specimen_fingerprint,
            "changed_cells": list(self.changed_cells),
            "changed_addresses": list(self.changed_addresses),
            "execution_history_fingerprint": self.execution_history_fingerprint,
            "intervention_lineage": list(self.intervention_lineage),
            "replay_schedule": [item.to_dict() for item in self.replay_schedule],
            "provenance": _thaw(self.provenance),
            "native_consumer": self.native_consumer,
            "continuation_horizon": self.continuation_horizon,
            "metadata": _thaw(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateCutIntent":
        body = _verify_seal(value, cls.schema, "StateCut intent")
        result = cls(
            intent_id=body.get("intent_id"),
            intent_kind=body.get("intent_kind"),
            edit_application_cut=StateCutRef.from_dict(body.get("edit_application_cut", {})),
            observation_cut=StateCutRef.from_dict(body.get("observation_cut", {})),
            rewind_cut=(
                StateCutRef.from_dict(body["rewind_cut"]) if body.get("rewind_cut") else None
            ),
            model_identity_fingerprint=body.get("model_identity_fingerprint"),
            state_schema_fingerprint=body.get("state_schema_fingerprint"),
            numerical_contract=body.get("numerical_contract"),
            dependency_contract_fingerprint=body.get("dependency_contract_fingerprint"),
            skeleton_fingerprint=body.get("skeleton_fingerprint"),
            specimen_fingerprint=body.get("specimen_fingerprint"),
            changed_cells=tuple(body.get("changed_cells", ())),
            changed_addresses=tuple(body.get("changed_addresses", ())),
            execution_history_fingerprint=body.get("execution_history_fingerprint"),
            intervention_lineage=tuple(body.get("intervention_lineage", ())),
            replay_schedule=tuple(
                ReplayScheduleItem.from_dict(item) for item in body.get("replay_schedule", ())
            ),
            provenance=body.get("provenance", {}),
            native_consumer=body.get("native_consumer"),
            continuation_horizon=body.get("continuation_horizon", 0),
            metadata=body.get("metadata", {}),
        )
        if result.fingerprint != value.get("fingerprint"):
            raise StateCutIntentError("StateCut intent fingerprint changed during hydration")
        return result


def plan_statecut_intent(intent: StateCutIntent, contract: DependencyContract) -> dict[str, Any]:
    """Compile declared dependency work without claiming graph completeness."""

    if not isinstance(intent, StateCutIntent):
        raise StateCutIntentError("planner requires a StateCutIntent")
    if not isinstance(contract, DependencyContract):
        raise StateCutIntentError("planner requires a DependencyContract")
    if contract.fingerprint != intent.dependency_contract_fingerprint:
        raise StateCutIntentError("intent and dependency contract fingerprints differ")
    if (
        intent.skeleton_fingerprint is not None
        and contract.skeleton_fingerprint != intent.skeleton_fingerprint
    ):
        raise StateCutIntentError("intent and dependency contract skeletons differ")
    if (
        intent.specimen_fingerprint is not None
        and contract.specimen_fingerprint != intent.specimen_fingerprint
    ):
        raise StateCutIntentError("intent and dependency contract specimens differ")
    unknown_schedule_cells = sorted(
        {
            cell_id
            for item in intent.replay_schedule
            for cell_id in item.dependency_cells
            if cell_id not in contract.node_ids
        }
    )
    if unknown_schedule_cells:
        raise StateCutIntentError(
            f"replay schedule names cells outside the dependency contract: {unknown_schedule_cells}"
        )

    closure = compute_dirty_closure(
        contract,
        changed_cells=intent.changed_cells,
        changed_addresses=intent.changed_addresses,
    )
    receipt_bound_steps = [
        item.step_id for item in intent.replay_schedule if item.parent_link_receipt_bound
    ]
    ancestry_status = (
        "not-applicable"
        if intent.intent_kind == PRESENT_STATE_INTERVENTION
        else "receipt-bound-parent-links"
        if len(receipt_bound_steps) == len(intent.replay_schedule)
        else "declared-only"
    )
    semantics = (
        "edit-current-state-and-run-only-the-declared-future-closure"
        if intent.intent_kind == PRESENT_STATE_INTERVENTION
        else "restore-rewind-cut-apply-edit-at-source-and-replay-the-explicit-schedule"
    )
    body = {
        "schema": STATECUT_INTENT_PLAN_SCHEMA,
        "intent": intent.to_dict(),
        "intent_fingerprint": intent.fingerprint,
        "intent_kind": intent.intent_kind,
        "edit_application_cut": intent.edit_application_cut.to_dict(),
        "rewind_cut": None if intent.rewind_cut is None else intent.rewind_cut.to_dict(),
        "continuation_start_cut": intent.edit_application_cut.to_dict(),
        "observation_cut": intent.observation_cut.to_dict(),
        "execution_history_fingerprint": intent.execution_history_fingerprint,
        "intervention_lineage": list(intent.intervention_lineage),
        "provenance": _thaw(intent.provenance),
        "numerical_contract": intent.numerical_contract,
        "model_identity_fingerprint": intent.model_identity_fingerprint,
        "state_schema_fingerprint": intent.state_schema_fingerprint,
        "dependency_contract_fingerprint": contract.fingerprint,
        "declared_dependency_closure": closure.to_dict(),
        "execution_cells": list(closure.dirty_cells),
        "reused_cells": list(closure.reused_cells),
        "invalidated_addresses": list(closure.invalidated_addresses),
        "replay_schedule": [item.to_dict() for item in intent.replay_schedule],
        "declared_parent_linked_closure": [
            {
                "step_id": item.step_id,
                "parent_cut_ref": item.parent_cut.cut_ref,
                "parent_cut_fingerprint": item.parent_cut.cut_fingerprint,
                "result_cut_ref": item.result_cut.cut_ref,
                "result_cut_fingerprint": item.result_cut.cut_fingerprint,
            }
            for item in intent.replay_schedule
        ],
        "receipt_bound_parent_linked_closure": [
            {
                "step_id": item.step_id,
                "result_cut_fingerprint": item.result_cut.cut_fingerprint,
                "parent_receipt_fingerprint": item.result_cut.parent_receipt_fingerprint,
            }
            for item in intent.replay_schedule
            if item.parent_link_receipt_bound
        ],
        "declared_scope": {
            "dependency_graph": contract.fingerprint,
            "source_cells": list(intent.changed_cells),
            "source_addresses": list(intent.changed_addresses),
            "descendant_cells": list(closure.dirty_cells),
            "replay_steps": [item.step_id for item in intent.replay_schedule],
            "semantics": semantics,
        },
        "verified_scope": {
            "identity_and_numerical_contract_bound": True,
            "dependency_contract_digest_bound": True,
            "ancestry_status": ancestry_status,
            "receipt_bound_parent_steps": receipt_bound_steps,
            "dependency_completeness": "not-established",
            "native_restore_parity": "not-measured",
            "native_replay_parity": "not-measured",
        },
        "native_consumer": intent.native_consumer,
        "continuation_horizon": intent.continuation_horizon,
        "native_consumer_required": True,
        "statuses": {
            "mechanics_status": "valid-plan",
            "instrument_status": "planning-only",
            "trend_status": "not-assessed",
            "terminal_status": "not-established",
        },
        "claims": {
            "exact_reconstruction": False,
            "dependency_graph_complete": False,
            "speedup": "not-claimed",
        },
    }
    return _sealed(body)


@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    """One externally executed cell, identity-bound but not dependency-complete."""

    cell_id: str
    receipt_fingerprint: str
    model_identity_fingerprint: str
    state_schema_fingerprint: str
    numerical_contract: str

    schema: ClassVar[str] = STATECUT_EXECUTION_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_id", _text(self.cell_id, "cell_id"))
        object.__setattr__(
            self, "receipt_fingerprint", _sha(self.receipt_fingerprint, "receipt_fingerprint")
        )
        object.__setattr__(
            self,
            "model_identity_fingerprint",
            _sha(self.model_identity_fingerprint, "model_identity_fingerprint"),
        )
        object.__setattr__(
            self,
            "state_schema_fingerprint",
            _sha(self.state_schema_fingerprint, "state_schema_fingerprint"),
        )
        object.__setattr__(
            self, "numerical_contract", _text(self.numerical_contract, "numerical_contract")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cell_id": self.cell_id,
            "receipt_fingerprint": self.receipt_fingerprint,
            "model_identity_fingerprint": self.model_identity_fingerprint,
            "state_schema_fingerprint": self.state_schema_fingerprint,
            "numerical_contract": self.numerical_contract,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionEvidence":
        if not isinstance(value, Mapping) or value.get("schema") != cls.schema:
            raise StateCutIntentError("execution evidence schema mismatch")
        return cls(
            cell_id=value.get("cell_id"),
            receipt_fingerprint=value.get("receipt_fingerprint"),
            model_identity_fingerprint=value.get("model_identity_fingerprint"),
            state_schema_fingerprint=value.get("state_schema_fingerprint"),
            numerical_contract=value.get("numerical_contract"),
        )


@dataclass(frozen=True, slots=True)
class ParityEvidence:
    """Digest comparison for restore or native-future replay mechanics."""

    parity_kind: str
    reference_sha256: str
    candidate_sha256: str
    receipt_fingerprint: str
    scope: Mapping[str, Any]

    schema: ClassVar[str] = STATECUT_PARITY_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        kind = _text(self.parity_kind, "parity_kind")
        if kind not in PARITY_KINDS:
            raise StateCutIntentError(f"unsupported parity kind {kind!r}")
        object.__setattr__(self, "parity_kind", kind)
        object.__setattr__(
            self, "reference_sha256", _sha(self.reference_sha256, "reference_sha256")
        )
        object.__setattr__(
            self, "candidate_sha256", _sha(self.candidate_sha256, "candidate_sha256")
        )
        object.__setattr__(
            self, "receipt_fingerprint", _sha(self.receipt_fingerprint, "receipt_fingerprint")
        )
        object.__setattr__(self, "scope", _freeze(self.scope, "parity scope"))

    @property
    def exact(self) -> bool:
        return self.reference_sha256 == self.candidate_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "parity_kind": self.parity_kind,
            "reference_sha256": self.reference_sha256,
            "candidate_sha256": self.candidate_sha256,
            "receipt_fingerprint": self.receipt_fingerprint,
            "scope": _thaw(self.scope),
            "exact": self.exact,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParityEvidence":
        if not isinstance(value, Mapping) or value.get("schema") != cls.schema:
            raise StateCutIntentError("parity evidence schema mismatch")
        result = cls(
            parity_kind=value.get("parity_kind"),
            reference_sha256=value.get("reference_sha256"),
            candidate_sha256=value.get("candidate_sha256"),
            receipt_fingerprint=value.get("receipt_fingerprint"),
            scope=value.get("scope", {}),
        )
        if value.get("exact") is not result.exact:
            raise StateCutIntentError("parity exact flag does not match its digests")
        return result


def record_statecut_intent(
    plan: Mapping[str, Any],
    *,
    result_cut: StateCutRef,
    execution_evidence: Sequence[ExecutionEvidence],
    native_consumer_receipt_fingerprint: str,
    replay_step_receipts: Mapping[str, str] | None = None,
    restore_parity: ParityEvidence | None = None,
    replay_parity: ParityEvidence | None = None,
) -> dict[str, Any]:
    """Seal execution evidence for one plan without widening its claims."""

    body, _ = _validate_plan(plan)
    if not isinstance(result_cut, StateCutRef):
        raise StateCutIntentError("result_cut must be a StateCutRef")
    evidence = tuple(execution_evidence)
    if any(not isinstance(item, ExecutionEvidence) for item in evidence):
        raise StateCutIntentError("execution_evidence must contain ExecutionEvidence values")
    expected_cells = tuple(body.get("execution_cells", ()))
    observed_cells = tuple(item.cell_id for item in evidence)
    if observed_cells != expected_cells:
        raise StateCutIntentError("execution evidence must cover planned cells in exact order")
    for item in evidence:
        if (
            item.model_identity_fingerprint != body["model_identity_fingerprint"]
            or item.state_schema_fingerprint != body["state_schema_fingerprint"]
            or item.numerical_contract != body["numerical_contract"]
        ):
            raise StateCutIntentError("execution evidence identity or numerical contract mismatch")

    expected_steps = tuple(item["step_id"] for item in body.get("replay_schedule", ()))
    step_receipts = dict(replay_step_receipts or {})
    if set(step_receipts) != set(expected_steps):
        raise StateCutIntentError("replay step receipts must cover the explicit schedule exactly")
    ordered_step_receipts = {
        step_id: _sha(step_receipts[step_id], f"replay receipt {step_id}")
        for step_id in expected_steps
    }
    for item in body.get("replay_schedule", ()):
        bound_parent_receipt = item["result_cut"].get("parent_receipt_fingerprint")
        if (
            bound_parent_receipt is not None
            and ordered_step_receipts[item["step_id"]] != bound_parent_receipt
        ):
            raise StateCutIntentError(
                "replay execution receipt differs from the schedule parent-link receipt"
            )
    if restore_parity is not None and (
        not isinstance(restore_parity, ParityEvidence) or restore_parity.parity_kind != "restore"
    ):
        raise StateCutIntentError("restore_parity must be restore ParityEvidence")
    if replay_parity is not None and (
        not isinstance(replay_parity, ParityEvidence) or replay_parity.parity_kind != "replay"
    ):
        raise StateCutIntentError("replay_parity must be replay ParityEvidence")
    native_receipt = _sha(
        native_consumer_receipt_fingerprint, "native_consumer_receipt_fingerprint"
    )
    ancestry_status = body["verified_scope"]["ancestry_status"]
    receipt_body = {
        "schema": STATECUT_INTENT_RECEIPT_SCHEMA,
        "plan_fingerprint": plan["fingerprint"],
        "intent_fingerprint": body["intent_fingerprint"],
        "intent_kind": body["intent_kind"],
        "edit_application_cut": body["edit_application_cut"],
        "rewind_cut": body["rewind_cut"],
        "continuation_start_cut": body["continuation_start_cut"],
        "observation_cut": body["observation_cut"],
        "result_cut": result_cut.to_dict(),
        "execution_history_fingerprint": body["execution_history_fingerprint"],
        "intervention_lineage": body["intervention_lineage"],
        "provenance": body["provenance"],
        "numerical_contract": body["numerical_contract"],
        "model_identity_fingerprint": body["model_identity_fingerprint"],
        "state_schema_fingerprint": body["state_schema_fingerprint"],
        "dependency_contract_fingerprint": body["dependency_contract_fingerprint"],
        "declared_dependency_closure": body["declared_dependency_closure"],
        "declared_parent_linked_closure": body["declared_parent_linked_closure"],
        "receipt_bound_parent_linked_closure": body["receipt_bound_parent_linked_closure"],
        "execution_evidence": [item.to_dict() for item in evidence],
        "replay_step_receipts": ordered_step_receipts,
        "native_consumer_receipt_fingerprint": native_receipt,
        "restore_parity": None if restore_parity is None else restore_parity.to_dict(),
        "replay_parity": None if replay_parity is None else replay_parity.to_dict(),
        "parity_scope": {
            "restore": None if restore_parity is None else _thaw(restore_parity.scope),
            "replay": None if replay_parity is None else _thaw(replay_parity.scope),
        },
        "verified_scope": {
            "identity_and_numerical_contract_bound": True,
            "declared_cells_receipt_bound": True,
            "replay_schedule_receipt_bound": bool(expected_steps),
            "ancestry_status": ancestry_status,
            "dependency_completeness": "not-established",
            "native_restore_parity": (
                "not-measured"
                if restore_parity is None
                else "exact"
                if restore_parity.exact
                else "different"
            ),
            "native_replay_parity": (
                "not-measured"
                if replay_parity is None
                else "exact"
                if replay_parity.exact
                else "different"
            ),
        },
        "statuses": {
            "mechanics_status": "recorded",
            "instrument_status": "receipt-bound",
            "trend_status": "not-assessed",
            "terminal_status": "not-established",
        },
        "claims": {
            "dependency_graph_complete": False,
            "semantic_equivalence": "not-assessed",
            "speedup": "not-claimed",
        },
    }
    return _sealed(receipt_body)


def verify_statecut_intent_receipt(receipt: Mapping[str, Any], plan: Mapping[str, Any]) -> bool:
    """Verify receipt identity and all bindings that can be checked offline."""

    plan_body, _ = _validate_plan(plan)
    body = _verify_seal(receipt, STATECUT_INTENT_RECEIPT_SCHEMA, "StateCut intent receipt")
    if body.get("plan_fingerprint") != plan.get("fingerprint"):
        raise StateCutIntentError("receipt belongs to another plan")
    for field_name in (
        "intent_fingerprint",
        "intent_kind",
        "edit_application_cut",
        "rewind_cut",
        "continuation_start_cut",
        "observation_cut",
        "execution_history_fingerprint",
        "intervention_lineage",
        "provenance",
        "numerical_contract",
        "model_identity_fingerprint",
        "state_schema_fingerprint",
        "dependency_contract_fingerprint",
        "declared_dependency_closure",
        "declared_parent_linked_closure",
        "receipt_bound_parent_linked_closure",
    ):
        if body.get(field_name) != plan_body.get(field_name):
            raise StateCutIntentError(f"receipt changed plan-bound field {field_name}")
    expected_cells = tuple(plan_body.get("execution_cells", ()))
    rows = body.get("execution_evidence", ())
    if tuple(row.get("cell_id") for row in rows) != expected_cells:
        raise StateCutIntentError("receipt execution evidence changed planned cell order")
    for row in rows:
        evidence = ExecutionEvidence.from_dict(row)
        if (
            evidence.model_identity_fingerprint != plan_body["model_identity_fingerprint"]
            or evidence.state_schema_fingerprint != plan_body["state_schema_fingerprint"]
            or evidence.numerical_contract != plan_body["numerical_contract"]
        ):
            raise StateCutIntentError("receipt execution binding changed")
    schedule = tuple(plan_body.get("replay_schedule", ()))
    expected_steps = {item["step_id"] for item in schedule}
    step_receipts = body.get("replay_step_receipts", {})
    if not isinstance(step_receipts, Mapping) or set(step_receipts) != expected_steps:
        raise StateCutIntentError("receipt replay schedule coverage changed")
    for item in schedule:
        step_id = item["step_id"]
        observed_receipt = _sha(step_receipts[step_id], f"replay receipt {step_id}")
        bound_parent_receipt = item["result_cut"].get("parent_receipt_fingerprint")
        if bound_parent_receipt is not None and observed_receipt != bound_parent_receipt:
            raise StateCutIntentError(
                "replay execution receipt differs from the schedule parent-link receipt"
            )
    StateCutRef.from_dict(body.get("result_cut", {}))
    _sha(body.get("native_consumer_receipt_fingerprint"), "native consumer receipt")
    for key, kind in (("restore_parity", "restore"), ("replay_parity", "replay")):
        raw = body.get(key)
        if raw is not None and ParityEvidence.from_dict(raw).parity_kind != kind:
            raise StateCutIntentError(f"receipt {key} kind changed")
    return True


__all__ = [
    "ExecutionEvidence",
    "HISTORICAL_COUNTERFACTUAL_RECONSTRUCTION",
    "INPUT_AUTHORITIES",
    "INTENT_KINDS",
    "PRESENT_STATE_INTERVENTION",
    "ParityEvidence",
    "ReplayScheduleItem",
    "STATECUT_INTENT_PLAN_SCHEMA",
    "STATECUT_INTENT_RECEIPT_SCHEMA",
    "STATECUT_INTENT_SCHEMA",
    "StateCutIntent",
    "StateCutIntentError",
    "StateCutRef",
    "plan_statecut_intent",
    "record_statecut_intent",
    "verify_statecut_intent_receipt",
]
