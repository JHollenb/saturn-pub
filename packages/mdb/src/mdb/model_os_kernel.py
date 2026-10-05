"""A small, reversible Model OS authority kernel.

This module joins the portable Model OS contracts to one explicit lifecycle:

    capture -> plan -> preview -> continue -> verify -> commit/restore

The kernel owns admission and lifecycle semantics.  A backend owns model
execution and may bind these calls to Saturn StateCuts, a native consumer, or
an mrun-resident runtime.  ``InMemoryModelBackend`` is intentionally limited to
JSON-safe state; it is a mechanics oracle for the protocol, not a claim that a
native transformer consumer is already integrated.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
import math
from typing import Any, Protocol, runtime_checkable

from mdb.model_os_contracts import (
    ConsumerContract,
    ModelABI,
    ModelOSContractError,
    ModelPackage,
    ModelProgram,
    StateSchema,
    canonical_json_bytes,
    content_digest,
)
from mdb.titan import Frame


MODEL_OS_KERNEL_SCHEMA = "saturn-model-os-kernel-v1"
MODEL_OS_STATE_CUT_SCHEMA = "saturn-model-os-state-cut-v1"
MODEL_OS_EXECUTION_PLAN_SCHEMA = "saturn-model-os-execution-plan-v1"
MODEL_OS_EXECUTION_RECEIPT_SCHEMA = "saturn-model-os-execution-receipt-v1"
MODEL_OS_ATOMIC_BUNDLE_EXECUTION_SCHEMA = "saturn-model-os-atomic-bundle-execution-v1"


class ModelOSKernelError(RuntimeError):
    """Base error for Model OS admission and lifecycle failures."""


class ModelOSAdmissionError(ModelOSKernelError):
    """Raised when a package, backend, or plan is not admitted."""


class ModelOSExecutionError(ModelOSKernelError):
    """Raised when a backend cannot complete an admitted lifecycle step."""


def _safe_object(value: Any, *, name: str) -> dict[str, Any]:
    try:
        normalized = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ModelOSKernelError(f"{name} must be JSON-safe") from exc
    if not isinstance(normalized, dict):
        raise ModelOSKernelError(f"{name} must be an object")
    return normalized


def _safe_value(value: Any, *, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ModelOSKernelError(f"{name} must be JSON-safe") from exc


@dataclass(frozen=True, slots=True)
class ModelOSStateCut:
    """An immutable public reference to a backend-owned state snapshot."""

    handle: str
    model_abi_fingerprint: str
    state_schema_fingerprint: str
    state_fingerprint: str
    parent_handle: str | None = None
    status: str = "captured"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "handle",
            "model_abi_fingerprint",
            "state_schema_fingerprint",
            "state_fingerprint",
            "status",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ModelOSKernelError(f"{name} must be a non-empty string")
        if self.parent_handle is not None and not self.parent_handle.strip():
            raise ModelOSKernelError("parent_handle must be non-empty when supplied")
        object.__setattr__(
            self,
            "metadata",
            _safe_object(self.metadata, name="state cut metadata"),
        )

    def to_payload(self) -> dict[str, Any]:
        body = {
            "schema": MODEL_OS_STATE_CUT_SCHEMA,
            "handle": self.handle,
            "model_abi_fingerprint": self.model_abi_fingerprint,
            "state_schema_fingerprint": self.state_schema_fingerprint,
            "state_fingerprint": self.state_fingerprint,
            "parent_handle": self.parent_handle,
            "status": self.status,
            "metadata": dict(self.metadata),
        }
        body["content_sha256"] = content_digest(body)
        return body

    @property
    def fingerprint(self) -> str:
        return self.to_payload()["content_sha256"]


@dataclass(frozen=True, slots=True)
class ModelOSExecutionPlan:
    """The admitted read/write and consumer surface for one execution."""

    plan_id: str
    package_fingerprint: str
    program_fingerprint: str
    parent_handle: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    consumer_id: str
    authority_subject: str
    dose: float
    horizon: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "plan_id",
            "package_fingerprint",
            "program_fingerprint",
            "parent_handle",
            "consumer_id",
            "authority_subject",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ModelOSKernelError(f"{name} must be a non-empty string")
        object.__setattr__(self, "reads", tuple(dict.fromkeys(self.reads)))
        object.__setattr__(self, "writes", tuple(dict.fromkeys(self.writes)))
        if not isinstance(self.dose, (int, float)) or not math.isfinite(self.dose):
            raise ModelOSKernelError("dose must be finite")
        if self.dose < 0:
            raise ModelOSKernelError("dose must be non-negative")
        if not isinstance(self.horizon, int) or self.horizon < 0:
            raise ModelOSKernelError("horizon must be a non-negative integer")
        object.__setattr__(
            self,
            "metadata",
            _safe_object(self.metadata, name="execution plan metadata"),
        )

    def to_payload(self) -> dict[str, Any]:
        body = {
            "schema": MODEL_OS_EXECUTION_PLAN_SCHEMA,
            "plan_id": self.plan_id,
            "package_fingerprint": self.package_fingerprint,
            "program_fingerprint": self.program_fingerprint,
            "parent_handle": self.parent_handle,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "consumer_id": self.consumer_id,
            "authority_subject": self.authority_subject,
            "dose": self.dose,
            "horizon": self.horizon,
            "metadata": dict(self.metadata),
        }
        body["content_sha256"] = content_digest(body)
        return body

    @property
    def fingerprint(self) -> str:
        return self.to_payload()["content_sha256"]


@dataclass(frozen=True, slots=True)
class ModelOSExecutionReceipt:
    """A portable record of a committed or restored lifecycle."""

    status: str
    decision: str
    package_fingerprint: str
    model_abi_fingerprint: str
    state_schema_fingerprint: str
    program_fingerprint: str
    plan_fingerprint: str
    parent_handle: str
    child_handle: str
    consumer_id: str
    continuation: Mapping[str, Any]
    verification: Mapping[str, Any]
    backend: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "status",
            "decision",
            "package_fingerprint",
            "model_abi_fingerprint",
            "state_schema_fingerprint",
            "program_fingerprint",
            "plan_fingerprint",
            "parent_handle",
            "child_handle",
            "consumer_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ModelOSKernelError(f"receipt {name} must be non-empty")
        for name in ("continuation", "verification", "backend", "metadata"):
            object.__setattr__(
                self,
                name,
                _safe_object(getattr(self, name), name=f"receipt {name}"),
            )

    def to_payload(self) -> dict[str, Any]:
        body = {
            "schema": MODEL_OS_EXECUTION_RECEIPT_SCHEMA,
            "kernel_schema": MODEL_OS_KERNEL_SCHEMA,
            "status": self.status,
            "decision": self.decision,
            "package_fingerprint": self.package_fingerprint,
            "model_abi_fingerprint": self.model_abi_fingerprint,
            "state_schema_fingerprint": self.state_schema_fingerprint,
            "program_fingerprint": self.program_fingerprint,
            "plan_fingerprint": self.plan_fingerprint,
            "parent_handle": self.parent_handle,
            "child_handle": self.child_handle,
            "consumer_id": self.consumer_id,
            "continuation": dict(self.continuation),
            "verification": dict(self.verification),
            "backend": dict(self.backend),
            "metadata": dict(self.metadata),
        }
        body["content_sha256"] = content_digest(body)
        return body

    @property
    def fingerprint(self) -> str:
        return self.to_payload()["content_sha256"]


@dataclass(frozen=True, slots=True)
class ModelOSExecutionResult:
    """The inspectable result returned by :meth:`ModelOSKernel.execute`."""

    plan: ModelOSExecutionPlan
    parent: ModelOSStateCut
    candidate: ModelOSStateCut
    observation: Mapping[str, Any]
    receipt: ModelOSExecutionReceipt

    @property
    def status(self) -> str:
        return self.receipt.status


@runtime_checkable
class ModelBackend(Protocol):
    """Backend seam used by the kernel to bind a real model runtime."""

    model_abi: ModelABI
    state_schema: StateSchema

    @property
    def current_statecut(self) -> ModelOSStateCut: ...

    def capture_statecut(
        self,
        parent: ModelOSStateCut | None,
        *,
        metadata: Mapping[str, Any],
    ) -> ModelOSStateCut: ...

    def preview(
        self,
        parent: ModelOSStateCut,
        plan: ModelOSExecutionPlan,
        updates: Mapping[str, Any],
    ) -> tuple[ModelOSStateCut, Mapping[str, Any]]: ...

    def continue_execution(
        self,
        candidate: ModelOSStateCut,
        *,
        consumer: ConsumerContract,
        horizon: int,
    ) -> Mapping[str, Any]: ...

    def commit(self, candidate: ModelOSStateCut) -> Mapping[str, Any]: ...

    def abort(self, candidate: ModelOSStateCut, *, reason: str) -> Mapping[str, Any]: ...

    def restore(self, target: ModelOSStateCut) -> Mapping[str, Any]: ...

    def read_state(self, cut: ModelOSStateCut, address: str) -> Any: ...


class InMemoryModelBackend:
    """JSON-only backend for testing admission and reversible state mechanics."""

    def __init__(
        self,
        *,
        model_abi: ModelABI,
        state_schema: StateSchema,
        initial_state: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_abi = model_abi
        self.state_schema = state_schema
        initial = _safe_object(initial_state or {}, name="initial state")
        unknown = set(initial) - set(state_schema.addresses)
        if unknown:
            raise ModelOSAdmissionError(f"initial state has unknown addresses: {sorted(unknown)}")
        self._states: dict[str, dict[str, Any]] = {}
        self._cuts: dict[str, ModelOSStateCut] = {}
        self._counter = 0
        self._current_handle: str | None = None
        genesis = self._new_cut(
            parent_handle=None,
            state=initial,
            status="committed",
            metadata={"kind": "genesis"},
        )
        self._current_handle = genesis.handle

    @property
    def current_statecut(self) -> ModelOSStateCut:
        if self._current_handle is None:
            raise ModelOSExecutionError("backend has no current state cut")
        return self._cuts[self._current_handle]

    def _new_cut(
        self,
        *,
        parent_handle: str | None,
        state: Mapping[str, Any],
        status: str,
        metadata: Mapping[str, Any],
    ) -> ModelOSStateCut:
        normalized = _safe_object(state, name="backend state")
        state_fingerprint = content_digest(
            {
                "state_schema": self.state_schema.fingerprint,
                "state": normalized,
            }
        )
        self._counter += 1
        handle = f"model-os://statecut/{self._counter:08d}-{state_fingerprint[:24]}"
        cut = ModelOSStateCut(
            handle=handle,
            model_abi_fingerprint=self.model_abi.fingerprint,
            state_schema_fingerprint=self.state_schema.fingerprint,
            state_fingerprint=state_fingerprint,
            parent_handle=parent_handle,
            status=status,
            metadata=metadata,
        )
        self._states[handle] = deepcopy(normalized)
        self._cuts[handle] = cut
        return cut

    def _require_cut(self, cut: ModelOSStateCut) -> ModelOSStateCut:
        stored = self._cuts.get(cut.handle)
        if stored is None:
            raise ModelOSExecutionError(f"unknown state cut {cut.handle!r}")
        if stored.model_abi_fingerprint != self.model_abi.fingerprint:
            raise ModelOSExecutionError("state cut belongs to a different model ABI")
        return stored

    def capture_statecut(
        self,
        parent: ModelOSStateCut | None,
        *,
        metadata: Mapping[str, Any],
    ) -> ModelOSStateCut:
        base = self._require_cut(parent) if parent is not None else self.current_statecut
        if base.status in {"restored", "aborted"}:
            raise ModelOSExecutionError("cannot branch from a restored state cut")
        return self._new_cut(
            parent_handle=base.handle,
            state=self._states[base.handle],
            status="captured",
            metadata=metadata,
        )

    def preview(
        self,
        parent: ModelOSStateCut,
        plan: ModelOSExecutionPlan,
        updates: Mapping[str, Any],
    ) -> tuple[ModelOSStateCut, Mapping[str, Any]]:
        stored_parent = self._require_cut(parent)
        if stored_parent.status in {"restored", "aborted"}:
            raise ModelOSExecutionError("cannot preview from a restored state cut")
        normalized_updates = _safe_object(updates, name="execution updates")
        undeclared = set(normalized_updates) - set(plan.writes)
        if undeclared:
            raise ModelOSAdmissionError(
                f"updates exceed declared writes: {sorted(undeclared)}"
            )
        state = deepcopy(self._states[stored_parent.handle])
        state.update(normalized_updates)
        candidate = self._new_cut(
            parent_handle=stored_parent.handle,
            state=state,
            status="candidate",
            metadata={"plan_fingerprint": plan.fingerprint},
        )
        return candidate, {
            "accepted": True,
            "execution_mode": "in-memory-mechanics",
            "updated_addresses": sorted(normalized_updates),
            "state_fingerprint": candidate.state_fingerprint,
        }

    def continue_execution(
        self,
        candidate: ModelOSStateCut,
        *,
        consumer: ConsumerContract,
        horizon: int,
    ) -> Mapping[str, Any]:
        stored = self._require_cut(candidate)
        if stored.status != "candidate":
            raise ModelOSExecutionError("only a candidate state cut can continue")
        if not isinstance(horizon, int) or horizon < 0:
            raise ModelOSExecutionError("horizon must be a non-negative integer")
        return {
            "accepted": True,
            "execution_mode": "in-memory-mechanics",
            "consumer_id": consumer.consumer_id,
            "consumer_native": consumer.native,
            "horizon": horizon,
            "output_kind": consumer.output_kind,
            "output_summary": candidate.state_fingerprint,
        }

    def commit(self, candidate: ModelOSStateCut) -> Mapping[str, Any]:
        stored = self._require_cut(candidate)
        if stored.status != "candidate":
            raise ModelOSExecutionError("only a candidate state cut can commit")
        committed = replace(stored, status="committed")
        self._cuts[stored.handle] = committed
        self._current_handle = stored.handle
        return {
            "status": "committed",
            "current_handle": stored.handle,
            "state_fingerprint": stored.state_fingerprint,
            "output_visible": True,
        }

    def abort(self, candidate: ModelOSStateCut, *, reason: str) -> Mapping[str, Any]:
        stored = self._require_cut(candidate)
        if stored.status not in {"candidate", "captured"}:
            raise ModelOSExecutionError("only a candidate or captured state cut can abort")
        self._cuts[stored.handle] = replace(
            stored,
            status="restored",
            metadata={**stored.metadata, "restore_reason": reason},
        )
        return {
            "status": "restored",
            "restored_to": stored.parent_handle,
            "reason": reason,
            "output_visible": False,
        }

    def restore(self, target: ModelOSStateCut) -> Mapping[str, Any]:
        stored = self._require_cut(target)
        if stored.status in {"restored", "aborted"}:
            raise ModelOSExecutionError("cannot restore to an invalid state cut")
        self._current_handle = stored.handle
        return {
            "status": "restored",
            "current_handle": stored.handle,
            "state_fingerprint": stored.state_fingerprint,
        }

    def read_state(self, cut: ModelOSStateCut, address: str) -> Any:
        stored = self._require_cut(cut)
        self.state_schema.slot(address)
        return deepcopy(self._states[stored.handle].get(address))


FrameContinuation = Callable[[Frame, ConsumerContract, int], Mapping[str, Any]]


class FrameModelBackend:
    """Bridge the kernel to Saturn's immutable tensor-owning Frame model.

    The adapter keeps tensors in Frame-owned storage and exposes only frame and
    slot fingerprints in Model OS receipts. A continuation callback must
    return JSON-safe evidence; logits, key/value pages, and other payloads stay
    in the backend or its native consumer.
    """

    def __init__(
        self,
        *,
        model_abi: ModelABI,
        state_schema: StateSchema,
        initial_frame: Frame,
        continuation: FrameContinuation | None = None,
    ) -> None:
        self.model_abi = model_abi
        self.state_schema = state_schema
        self._continuation = continuation
        self._frames: dict[str, Frame] = {}
        self._cuts: dict[str, ModelOSStateCut] = {}
        self._counter = 0
        self._current_handle: str | None = None
        for address in state_schema.addresses:
            if address not in initial_frame.slots:
                raise ModelOSAdmissionError(
                    f"initial Frame is missing Model OS address {address!r}"
                )
        genesis = self._new_cut(
            frame=initial_frame,
            parent_handle=None,
            status="committed",
            metadata={"kind": "genesis", "frame_id": initial_frame.frame_id},
        )
        self._current_handle = genesis.handle

    @property
    def current_statecut(self) -> ModelOSStateCut:
        if self._current_handle is None:
            raise ModelOSExecutionError("Frame backend has no current state cut")
        return self._cuts[self._current_handle]

    def _new_cut(
        self,
        *,
        frame: Frame,
        parent_handle: str | None,
        status: str,
        metadata: Mapping[str, Any],
    ) -> ModelOSStateCut:
        self._counter += 1
        handle = f"model-os://frame-statecut/{self._counter:08d}-{frame.frame_id[:24]}"
        cut = ModelOSStateCut(
            handle=handle,
            model_abi_fingerprint=self.model_abi.fingerprint,
            state_schema_fingerprint=self.state_schema.fingerprint,
            state_fingerprint=frame.frame_id,
            parent_handle=parent_handle,
            status=status,
            metadata={**dict(metadata), "frame_id": frame.frame_id},
        )
        self._frames[handle] = frame
        self._cuts[handle] = cut
        return cut

    def _require_cut(self, cut: ModelOSStateCut) -> ModelOSStateCut:
        stored = self._cuts.get(cut.handle)
        if stored is None:
            raise ModelOSExecutionError(f"unknown Frame state cut {cut.handle!r}")
        if stored.model_abi_fingerprint != self.model_abi.fingerprint:
            raise ModelOSExecutionError("Frame state cut belongs to another model ABI")
        return stored

    @staticmethod
    def _updates(updates: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(updates, Mapping):
            raise ModelOSExecutionError("Frame updates must be a mapping")
        normalized: dict[str, Any] = {}
        for address, value in updates.items():
            if not isinstance(address, str) or not address.strip():
                raise ModelOSExecutionError("Frame update addresses must be non-empty strings")
            normalized[address] = value
        return normalized

    def capture_statecut(
        self,
        parent: ModelOSStateCut | None,
        *,
        metadata: Mapping[str, Any],
    ) -> ModelOSStateCut:
        base = self._require_cut(parent) if parent is not None else self.current_statecut
        if base.status in {"restored", "aborted"}:
            raise ModelOSExecutionError("cannot branch from a restored Frame state cut")
        return self._new_cut(
            frame=self._frames[base.handle],
            parent_handle=base.handle,
            status="captured",
            metadata=metadata,
        )

    def preview(
        self,
        parent: ModelOSStateCut,
        plan: ModelOSExecutionPlan,
        updates: Mapping[str, Any],
    ) -> tuple[ModelOSStateCut, Mapping[str, Any]]:
        stored_parent = self._require_cut(parent)
        if stored_parent.status in {"restored", "aborted"}:
            raise ModelOSExecutionError("cannot preview from a restored Frame state cut")
        normalized = self._updates(updates)
        undeclared = set(normalized) - set(plan.writes)
        if undeclared:
            raise ModelOSAdmissionError(
                f"updates exceed declared writes: {sorted(undeclared)}"
            )
        for address in normalized:
            self.state_schema.slot(address)
        invalidates = tuple(
            dict.fromkeys(
                invalidated
                for address in normalized
                for invalidated in self.state_schema.slot(address).invalidates
            )
        )
        child_frame = self._frames[stored_parent.handle].with_updates(
            normalized,
            act=f"model-os:{plan.plan_id}",
            implementation="saturn.model_os.FrameModelBackend",
            invalidates=invalidates,
        )
        candidate = self._new_cut(
            frame=child_frame,
            parent_handle=stored_parent.handle,
            status="candidate",
            metadata={"plan_fingerprint": plan.fingerprint},
        )
        return candidate, {
            "accepted": True,
            "execution_mode": "saturn-frame",
            "frame_id": child_frame.frame_id,
            "updated_addresses": sorted(normalized),
            "invalidated_addresses": list(invalidates),
            "state_fingerprint": candidate.state_fingerprint,
        }

    def continue_execution(
        self,
        candidate: ModelOSStateCut,
        *,
        consumer: ConsumerContract,
        horizon: int,
    ) -> Mapping[str, Any]:
        stored = self._require_cut(candidate)
        if stored.status != "candidate":
            raise ModelOSExecutionError("only a candidate Frame state cut can continue")
        if not isinstance(horizon, int) or horizon < 0:
            raise ModelOSExecutionError("horizon must be a non-negative integer")
        frame = self._frames[stored.handle]
        if self._continuation is None:
            return {
                "accepted": True,
                "execution_mode": "saturn-frame",
                "consumer_id": consumer.consumer_id,
                "horizon": horizon,
                "frame_id": frame.frame_id,
                "native_consumer_executed": False,
            }
        try:
            evidence = self._continuation(frame, consumer, horizon)
            return _safe_object(evidence, name="Frame continuation evidence")
        except ModelOSKernelError:
            raise
        except Exception as exc:
            raise ModelOSExecutionError("Frame continuation callback failed") from exc

    def commit(self, candidate: ModelOSStateCut) -> Mapping[str, Any]:
        stored = self._require_cut(candidate)
        if stored.status != "candidate":
            raise ModelOSExecutionError("only a candidate Frame state cut can commit")
        self._cuts[stored.handle] = replace(stored, status="committed")
        self._current_handle = stored.handle
        return {
            "status": "committed",
            "current_handle": stored.handle,
            "frame_id": stored.state_fingerprint,
            "output_visible": True,
        }

    def abort(self, candidate: ModelOSStateCut, *, reason: str) -> Mapping[str, Any]:
        stored = self._require_cut(candidate)
        if stored.status not in {"candidate", "captured"}:
            raise ModelOSExecutionError("only a candidate or captured Frame state cut can abort")
        self._cuts[stored.handle] = replace(
            stored,
            status="restored",
            metadata={**stored.metadata, "restore_reason": reason},
        )
        return {
            "status": "restored",
            "restored_to": stored.parent_handle,
            "reason": reason,
            "output_visible": False,
        }

    def restore(self, target: ModelOSStateCut) -> Mapping[str, Any]:
        stored = self._require_cut(target)
        if stored.status in {"restored", "aborted"}:
            raise ModelOSExecutionError("cannot restore to an invalid Frame state cut")
        self._current_handle = stored.handle
        return {
            "status": "restored",
            "current_handle": stored.handle,
            "frame_id": stored.state_fingerprint,
        }

    def read_state(self, cut: ModelOSStateCut, address: str) -> Any:
        stored = self._require_cut(cut)
        self.state_schema.slot(address)
        frame = self._frames[stored.handle]
        frame.require([address])
        return frame.values[address]


Verifier = Callable[[Mapping[str, Any]], Mapping[str, Any] | bool]


class ModelOSKernel:
    """Admission and lifecycle authority for one installed Model Package."""

    def __init__(
        self,
        package: ModelPackage,
        backend: ModelBackend,
        *,
        authority_subject: str = "model-os",
    ) -> None:
        try:
            package.validate()
        except ModelOSContractError as exc:
            raise ModelOSAdmissionError(str(exc)) from exc
        if backend.model_abi.fingerprint != package.model_abi.fingerprint:
            raise ModelOSAdmissionError("backend model ABI does not match package ABI")
        if backend.state_schema.fingerprint != package.state_schema.fingerprint:
            raise ModelOSAdmissionError("backend state schema does not match package schema")
        if not authority_subject.strip():
            raise ModelOSAdmissionError("authority_subject must be non-empty")
        self.package = package
        self.backend = backend
        self.authority_subject = authority_subject

    @classmethod
    def install(
        cls,
        package: ModelPackage,
        backend: ModelBackend,
        *,
        authority_subject: str = "model-os",
    ) -> "ModelOSKernel":
        """Admit a package against a backend with exact ABI/schema identity."""

        return cls(package, backend, authority_subject=authority_subject)

    @property
    def current_statecut(self) -> ModelOSStateCut:
        return self.backend.current_statecut

    def capture(
        self,
        *,
        parent: ModelOSStateCut | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelOSStateCut:
        return self.backend.capture_statecut(
            parent or self.current_statecut,
            metadata=metadata or {"kind": "kernel-capture"},
        )

    def _check_scope(self, addresses: tuple[str, ...], *, kind: str, subject: str) -> None:
        permissions = self.package.permissions
        subjects = permissions.get("subjects", {})
        if isinstance(subjects, Mapping):
            subject_policy = subjects.get(subject, {})
        else:
            subject_policy = {}
        if not isinstance(subject_policy, Mapping):
            subject_policy = {}
        configured = subject_policy.get(kind, permissions.get(kind))
        allowed = None if configured is None else set(configured)
        if allowed is not None and "*" not in allowed:
            denied = set(addresses) - allowed
            if denied:
                raise ModelOSAdmissionError(
                    f"authority {subject!r} cannot {kind} state addresses {sorted(denied)}"
                )
        for address in addresses:
            slot = self.package.state_schema.slot(address)
            scopes = slot.read_scopes if kind == "read" else slot.write_scopes
            if kind == "write" and not scopes:
                raise ModelOSAdmissionError(
                    f"authority {subject!r} is outside write scope for {address!r}"
                )
            if scopes and subject not in scopes and "*" not in scopes:
                raise ModelOSAdmissionError(
                    f"authority {subject!r} is outside {kind} scope for {address!r}"
                )

    def plan(
        self,
        *,
        program: ModelProgram | None = None,
        parent: ModelOSStateCut | None = None,
        consumer_id: str | None = None,
        authority_subject: str | None = None,
        dose: float = 1.0,
        horizon: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelOSExecutionPlan:
        selected_program = program or self.package.program
        if selected_program.fingerprint != self.package.program.fingerprint:
            raise ModelOSAdmissionError("program is not the program admitted by this package")
        selected_parent = parent or self.current_statecut
        if selected_parent.model_abi_fingerprint != self.package.model_abi.fingerprint:
            raise ModelOSAdmissionError("parent state cut has a different model ABI")
        subject = authority_subject or self.authority_subject
        if not isinstance(dose, (int, float)) or not math.isfinite(dose) or dose < 0:
            raise ModelOSAdmissionError("dose must be a finite, non-negative number")
        if not isinstance(horizon, int) or horizon < 0:
            raise ModelOSAdmissionError("horizon must be a non-negative integer")
        declared_authorities = {
            instruction.authority for instruction in selected_program.instructions
        }
        delegated = self.package.permissions.get("authorities", ())
        if not isinstance(delegated, (list, tuple, set, frozenset)):
            raise ModelOSAdmissionError("package authorities must be a sequence")
        undelegated = declared_authorities - {subject} - set(delegated)
        if undelegated:
            raise ModelOSAdmissionError(
                f"authority subject {subject!r} cannot execute {sorted(undelegated)}"
            )
        reads = selected_program.reads
        writes = selected_program.writes
        self._check_scope(reads, kind="read", subject=subject)
        self._check_scope(writes, kind="write", subject=subject)
        selected_consumer_id = consumer_id
        if selected_consumer_id is None and selected_program.native_consumers:
            selected_consumer_id = selected_program.native_consumers[0]
        consumer = self.package.consumer(selected_consumer_id)
        continuation = self.package.continuation(consumer.consumer_id)
        if continuation is not None and horizon > continuation.horizon:
            raise ModelOSAdmissionError(
                f"horizon {horizon} exceeds continuation contract {continuation.horizon}"
            )
        if (
            selected_program.native_consumers
            and consumer.consumer_id not in selected_program.native_consumers
        ):
            raise ModelOSAdmissionError(
                f"consumer {consumer.consumer_id!r} is not referenced by the program"
            )
        plan_body = {
            "package": self.package.fingerprint,
            "program": selected_program.fingerprint,
            "parent": selected_parent.handle,
            "reads": reads,
            "writes": writes,
            "consumer": consumer.consumer_id,
            "subject": subject,
            "dose": dose,
            "horizon": horizon,
            "metadata": metadata or {},
        }
        plan_id = f"model-os://plan/{content_digest(plan_body)[:24]}"
        return ModelOSExecutionPlan(
            plan_id=plan_id,
            package_fingerprint=self.package.fingerprint,
            program_fingerprint=selected_program.fingerprint,
            parent_handle=selected_parent.handle,
            reads=reads,
            writes=writes,
            consumer_id=consumer.consumer_id,
            authority_subject=subject,
            dose=float(dose),
            horizon=horizon,
            metadata=metadata or {},
        )

    def execute(
        self,
        *,
        updates: Mapping[str, Any] | None = None,
        program: ModelProgram | None = None,
        parent: ModelOSStateCut | None = None,
        consumer_id: str | None = None,
        authority_subject: str | None = None,
        dose: float = 1.0,
        horizon: int = 0,
        metadata: Mapping[str, Any] | None = None,
        verifier: Verifier | None = None,
    ) -> ModelOSExecutionResult:
        plan = self.plan(
            program=program,
            parent=parent,
            consumer_id=consumer_id,
            authority_subject=authority_subject,
            dose=dose,
            horizon=horizon,
            metadata=metadata,
        )
        selected_parent = parent or self.current_statecut
        candidate: ModelOSStateCut | None = None
        try:
            candidate, preview = self.backend.preview(
                selected_parent,
                plan,
                updates or {},
            )
            consumer = self.package.consumer(plan.consumer_id)
            continuation = self.backend.continue_execution(
                candidate,
                consumer=consumer,
                horizon=horizon,
            )
            observation = _safe_object(
                {
                    "plan": plan.to_payload(),
                    "preview": preview,
                    "continuation": continuation,
                },
                name="execution observation",
            )
            if verifier is None:
                verification: Mapping[str, Any] = {
                    "accepted": True,
                    "reason": "default mechanics acceptance",
                }
            else:
                raw_verification = verifier(observation)
                if isinstance(raw_verification, bool):
                    verification = {"accepted": raw_verification}
                else:
                    verification = _safe_object(raw_verification, name="verification")
                    if "accepted" not in verification:
                        raise ModelOSExecutionError("verifier must return an accepted decision")
            accepted = bool(verification.get("accepted", False))
            if accepted:
                backend_receipt = self.backend.commit(candidate)
                status = "committed"
                decision = "commit"
            else:
                backend_receipt = self.backend.abort(
                    candidate,
                    reason=str(verification.get("reason", "verifier rejected candidate")),
                )
                status = "restored"
                decision = "restore"
            receipt = ModelOSExecutionReceipt(
                status=status,
                decision=decision,
                package_fingerprint=self.package.fingerprint,
                model_abi_fingerprint=self.package.model_abi.fingerprint,
                state_schema_fingerprint=self.package.state_schema.fingerprint,
                program_fingerprint=plan.program_fingerprint,
                plan_fingerprint=plan.fingerprint,
                parent_handle=selected_parent.handle,
                child_handle=candidate.handle,
                consumer_id=plan.consumer_id,
                continuation=continuation,
                verification=verification,
                backend=backend_receipt,
                metadata={"terminal_claim": False},
            )
            return ModelOSExecutionResult(
                plan=plan,
                parent=selected_parent,
                candidate=candidate,
                observation=observation,
                receipt=receipt,
            )
        except ModelOSKernelError:
            if candidate is not None:
                try:
                    self.backend.abort(candidate, reason="kernel execution error")
                except Exception:
                    pass
            raise
        except Exception as exc:
            if candidate is not None:
                try:
                    self.backend.abort(candidate, reason="backend execution error")
                except Exception:
                    pass
            raise ModelOSExecutionError(str(exc)) from exc

    def execute_atomic_token_bundle(
        self,
        *,
        token_ids: Sequence[int],
        token_roles: Sequence[str],
        control_values: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
        verifier: Verifier | None = None,
        authority_subject: str | None = None,
    ) -> tuple[dict[str, Any], Any]:
        """Admit and execute one backend-native atomic token block.

        The kernel owns package/program/permission admission and binds that plan
        into the physical backend transaction. The backend still owns private
        tensors and the actual provisional K/V commit or restore.
        """

        physical = getattr(self.backend, "execute_atomic_token_bundle", None)
        if not callable(physical):
            raise ModelOSAdmissionError("installed backend has no atomic token-bundle surface")
        supplied_metadata = dict(metadata or {})
        plan = self.plan(
            parent=self.current_statecut,
            authority_subject=authority_subject,
            horizon=1,
            metadata={**supplied_metadata, "kind": "atomic-token-bundle"},
        )
        undeclared = set(control_values) - set(plan.writes)
        if undeclared:
            raise ModelOSAdmissionError(
                f"atomic token-bundle controls exceed declared writes: {sorted(undeclared)}"
            )
        bound_metadata = {
            **supplied_metadata,
            "model_os_package_fingerprint": self.package.fingerprint,
            "model_os_program_fingerprint": plan.program_fingerprint,
            "model_os_plan_fingerprint": plan.fingerprint,
            "model_os_authority_subject": plan.authority_subject,
        }
        verification_holder: dict[str, Any] = {}

        def kernel_verifier(bundle_observation: Mapping[str, Any]) -> Mapping[str, Any]:
            observation = _safe_object(
                {
                    **dict(bundle_observation),
                    "model_os_plan": plan.to_payload(),
                },
                name="atomic token-bundle observation",
            )
            if verifier is None:
                verification: Mapping[str, Any] = {
                    "accepted": True,
                    "reason": "default mechanics acceptance",
                }
            else:
                raw = verifier(observation)
                if isinstance(raw, bool):
                    verification = {"accepted": raw}
                else:
                    verification = _safe_object(raw, name="atomic token-bundle verification")
                    if "accepted" not in verification:
                        raise ModelOSExecutionError(
                            "atomic token-bundle verifier must return an accepted decision"
                        )
            verification_holder.update(dict(verification))
            return verification

        backend_receipt, private_output = physical(
            token_ids=token_ids,
            token_roles=token_roles,
            control_values=control_values,
            metadata=bound_metadata,
            verifier=kernel_verifier,
        )
        receipt = dict(backend_receipt)
        accepted = verification_holder.get("accepted") is True
        expected_decision = "commit" if accepted else "restore"
        if receipt.get("decision") != expected_decision:
            raise ModelOSExecutionError("atomic backend decision disagrees with kernel verifier")
        receipt["model_os_kernel"] = {
            "schema": MODEL_OS_ATOMIC_BUNDLE_EXECUTION_SCHEMA,
            "status": "committed" if accepted else "restored",
            "decision": expected_decision,
            "package_fingerprint": self.package.fingerprint,
            "model_abi_fingerprint": self.package.model_abi.fingerprint,
            "state_schema_fingerprint": self.package.state_schema.fingerprint,
            "program_fingerprint": plan.program_fingerprint,
            "plan_fingerprint": plan.fingerprint,
            "authority_subject": plan.authority_subject,
            "consumer_id": plan.consumer_id,
            "verification": dict(verification_holder),
            "terminal_claim": False,
        }
        return receipt, private_output

    def read(self, address: str, *, cut: ModelOSStateCut | None = None) -> Any:
        selected_cut = cut or self.current_statecut
        self._check_scope((address,), kind="read", subject=self.authority_subject)
        return self.backend.read_state(selected_cut, address)

    def restore(self, target: ModelOSStateCut) -> Mapping[str, Any]:
        if target.model_abi_fingerprint != self.package.model_abi.fingerprint:
            raise ModelOSAdmissionError("restore target has a different model ABI")
        return self.backend.restore(target)

    def status(self) -> dict[str, Any]:
        return {
            "schema": MODEL_OS_KERNEL_SCHEMA,
            "package_id": self.package.package_id,
            "package_fingerprint": self.package.fingerprint,
            "model_abi": self.package.model_abi.identity,
            "model_abi_fingerprint": self.package.model_abi.fingerprint,
            "state_schema_fingerprint": self.package.state_schema.fingerprint,
            "program_id": self.package.program.program_id,
            "program_fingerprint": self.package.program.fingerprint,
            "current_statecut": self.current_statecut.to_payload(),
            "terminal_claim": False,
        }


__all__ = [
    "MODEL_OS_EXECUTION_PLAN_SCHEMA",
    "MODEL_OS_EXECUTION_RECEIPT_SCHEMA",
    "MODEL_OS_ATOMIC_BUNDLE_EXECUTION_SCHEMA",
    "MODEL_OS_KERNEL_SCHEMA",
    "MODEL_OS_STATE_CUT_SCHEMA",
    "FrameModelBackend",
    "InMemoryModelBackend",
    "ModelBackend",
    "ModelOSAdmissionError",
    "ModelOSExecutionError",
    "ModelOSExecutionPlan",
    "ModelOSExecutionReceipt",
    "ModelOSExecutionResult",
    "ModelOSKernel",
    "ModelOSKernelError",
    "ModelOSStateCut",
]
