"""Live-only deterministic CPU ATen discovery replay over public typed cuts.

Capture executes first; this adapter pauses the subsequent replay. It owns no
durable hydration promise and never uses recorded intermediates as roots.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field

import torch
from saturn_pub import Act, Receipt, StateCut
from saturn_pub.values import digest

from .discovery_catalog import CatalogError, StructuralCatalog
from .forward_interventions import (
    ForwardInterventionError,
    _frontier_identity,
    _validate_replacement,
)
from .forward_recording import ATenStepper, ForwardRecordingError, _decode, tensor_descriptor


class DiscoveryRuntimeError(ForwardRecordingError):
    pass


@dataclass(frozen=True)
class ReplayCut:
    """A retained process-local native frontier, sealed by a public StateCut."""
    pc: int
    _owner: str = field(repr=False)
    _state_cut: StateCut = field(repr=False)
    _environment: dict = field(repr=False)
    _steps: list = field(repr=False)
    _acts: list = field(repr=False)

    @property
    def fingerprint(self):
        return self._state_cut.fingerprint

    @property
    def state_cut(self):
        return self._state_cut

    def manifest(self):
        return {"schema":"mdb-discovery-live-cut-v1", "fingerprint":self.fingerprint,
                "pc":self.pc, "restore_support":"live", "native_durable":False,
                "state_cut":self._state_cut.manifest()}


class ReplayForward:
    """One replay cursor; every successful primitive is separately retained."""

    def __init__(self, capture, catalog=None):
        self.trace = copy.deepcopy(capture.trace)
        self.catalog = catalog or StructuralCatalog(self.trace)
        if self.catalog.trace_fingerprint != self.trace["fingerprint"]:
            raise DiscoveryRuntimeError("catalog belongs to a different native trace")
        self._machine = ATenStepper(self.trace, capture.initial_values)
        # Copy the entire tensor object graph in one operation: independent
        # per-value clones would destroy aliases between admitted roots.
        self._machine.environment = copy.deepcopy(capture.initial_values)
        self._owner = uuid.uuid4().hex
        self._acts = []
        self.work = {"replay_primitives":0, "attempted_primitives":0, "restores":0, "forks":0}
        self.last_error = None

    @property
    def pc(self):
        return self._machine.pc

    @property
    def complete(self):
        return self.pc == len(self.trace["events"])

    def advance_one(self):
        if self.complete:
            raise StopIteration("forward execution complete")
        old_environment = self._machine.environment.copy()
        old_pc, old_count = self.pc, len(self._machine.steps)
        self.work["attempted_primitives"] += 1
        try:
            row = self._machine.step()
            for key in row["event"]["outputs"]:
                actual = self._machine.environment[key]
                reference = self.trace["values"][key]
                if list(actual.shape) != reference["shape"] or str(actual.dtype) != reference["dtype"]:
                    raise DiscoveryRuntimeError(f"native output geometry changed at {key}")
                if actual.device.type != "cpu" or not bool(torch.isfinite(actual).all()):
                    raise DiscoveryRuntimeError(f"native output violates CPU/finite contract at {key}")
        except Exception as exc:
            self._machine.environment = old_environment
            self._machine.pc = old_pc
            del self._machine.steps[old_count:]
            self.last_error = {"pc":old_pc, "type":type(exc).__name__, "message":str(exc),
                               "mechanics_status":"errored", "retained_pc":old_pc}
            raise DiscoveryRuntimeError(f"native transition failed at {old_pc}: {exc}") from exc
        self.work["replay_primitives"] += 1
        self.last_error = None
        receipt = Receipt.make(operation="mdb.discovery.native-transition", status="observed",
            trace_fingerprint=self.trace["fingerprint"], program_id=self.catalog.program_id,
            before_pc=old_pc, after_pc=self.pc, operator=row["event"]["operator"],
            exact_outputs=row["exact_outputs"], verification="computed output descriptors",
            mechanics_status="valid", execution_mode="record-replay")
        return {**copy.deepcopy(row), "next_pc":self.pc, "receipt":receipt.to_dict()}

    def _value_id(self, selector):
        if isinstance(selector, str) and selector in self.trace["values"]:
            return selector
        try:
            resolved = self.catalog.resolve(selector)
        except CatalogError as exc:
            raise DiscoveryRuntimeError(str(exc)) from exc
        if "value_id" not in resolved:
            raise DiscoveryRuntimeError("read/write requires an explicit tensor port")
        return resolved["value_id"]

    def read(self, value):
        key = self._value_id(value)
        if key not in self._machine.environment:
            raise DiscoveryRuntimeError(f"value is not live/computed at this cut: {key}")
        return self._machine.environment[key].detach().clone(memory_format=torch.preserve_format)

    def inspect(self):
        return {"pc":self.pc, "complete":self.complete, "source":"live",
                "authority":"live-ATen-replay-environment", "values":{
                    key:{**tensor_descriptor(value), "producer":self.trace["values"][key]["producer"],
                         "origin":self.trace["values"][key]["origin"], "writable":self._writable(key)}
                    for key,value in self._machine.environment.items()}}

    def outputs(self):
        if not self.complete:
            raise DiscoveryRuntimeError("forward outputs are not computed")
        return copy.deepcopy(_decode(self.trace["outputs"], self._machine.environment))

    def capture(self):
        environment = copy.deepcopy(self._machine.environment)
        closure = _frontier_identity(environment)
        state = StateCut(model_identity=self.trace["model_identity"], execution=self.trace["execution"],
            boundary="mdb.discovery.native-safe-point-v1", parent=self.trace["fingerprint"],
            _payload={"frontier":closure, "pc":self.pc, "owner":self._owner,
                      "program_id":self.catalog.program_id, "acts":copy.deepcopy(self._acts),
                      "prefix_steps_digest":digest(self._machine.steps)},
            execution_point={"next_event":self.pc},
            surface={"restore_support":"live", "native_durable":False,
                     "alias_scope":"complete backing storage and view geometry"})
        return ReplayCut(self.pc,self._owner,state,environment,copy.deepcopy(self._machine.steps),
                         copy.deepcopy(self._acts))

    def restore(self, cut):
        if not isinstance(cut, ReplayCut) or cut._owner != self._owner:
            raise DiscoveryRuntimeError("cut belongs to a different input/trace owner")
        cut._state_cut.verify()
        payload = cut._state_cut.payload
        if payload["pc"] != cut.pc or payload["frontier"] != _frontier_identity(cut._environment):
            raise DiscoveryRuntimeError("retained native frontier changed")
        if payload["acts"] != cut._acts or payload["prefix_steps_digest"] != digest(cut._steps):
            raise DiscoveryRuntimeError("retained native provenance changed")
        self._machine.environment = copy.deepcopy(cut._environment)
        self._machine.pc = cut.pc
        self._machine.steps = copy.deepcopy(cut._steps)
        self._acts = copy.deepcopy(cut._acts)
        self.last_error = None
        self.work["restores"] += 1
        return cut.manifest()

    def fork(self, cut=None):
        cut = self.capture() if cut is None else cut
        branch = object.__new__(ReplayForward)
        branch.trace, branch.catalog, branch._owner = self.trace, self.catalog, self._owner
        branch._machine = object.__new__(ATenStepper)
        branch._machine.trace, branch._machine.pc, branch._machine.steps = self.trace, 0, []
        branch._machine.environment = {}
        branch._acts, branch.last_error = [], None
        branch.work = {"replay_primitives":0, "attempted_primitives":0, "restores":0, "forks":0}
        branch.restore(cut)
        self.work["forks"] += 1
        return branch

    def _writable(self, key):
        row = self.trace["values"][key]
        return (row["producer"] is not None and key in self._machine.environment and
                not any(key in event["inputs"] for event in self.trace["events"][:self.pc]))

    def apply(self, value, tensor, sign="replace"):
        key = self._value_id(value)
        if not self._writable(key):
            raise DiscoveryRuntimeError("edits require a produced, live, unconsumed tensor; roots are read-only")
        parent = self._machine.environment[key]
        try:
            _validate_replacement(parent, tensor, key)
        except ForwardInterventionError as exc:
            raise DiscoveryRuntimeError(str(exc)) from exc
        if sign not in {"replace", "add"}:
            raise DiscoveryRuntimeError("supported Act operations are replace and add")
        act = Act.replace(key,tensor,name="discovery-ssa-replace") if sign == "replace" else Act.add(
            key,tensor,name="discovery-ssa-add")
        act.verify_implementation()
        writes = dict(act.implementation({key:parent}))
        if set(writes) != {key}:
            raise DiscoveryRuntimeError("Act write set differs from admitted tensor port")
        try:
            _validate_replacement(parent,writes[key],key)
        except ForwardInterventionError as exc:
            raise DiscoveryRuntimeError(str(exc)) from exc
        applied = writes[key].detach().clone(memory_format=torch.preserve_format)
        before = tensor_descriptor(parent)
        after = tensor_descriptor(applied)
        receipt = Receipt.make(operation="mdb.discovery.ssa-act", status="observed", pc=self.pc,
            trace_fingerprint=self.trace["fingerprint"], value_id=key, act=act.manifest(),
            before=before, after=after, invalidation={"future_values":"not-yet-computed",
                "prepared_frontiers":"not-shared", "alias_scope":"selected SSA handle only"},
            mechanics_status="valid", causal_effect="not-established")
        self._machine.environment[key] = applied
        self._acts.append(receipt.to_dict())
        return receipt.to_dict()


__all__ = ["ReplayForward", "ReplayCut", "DiscoveryRuntimeError"]
