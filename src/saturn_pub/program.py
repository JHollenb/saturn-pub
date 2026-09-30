"""Recipient-bound schedules compiled from measured panel rows."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import Act, Adapter, Receipt, Session
from .store import LocalStore
from .values import canonical, clone, digest


@dataclass(frozen=True)
class Instruction:
    after_steps: int
    act: Act


@dataclass(frozen=True)
class Program:
    model_identity: str
    execution: dict[str, Any]
    entry_boundary: str
    instructions: tuple[Instruction, ...]
    evidence_parent: str
    evidence_plan: str
    evidence_arm: str
    context: str
    continuation_steps: int

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "saturn-pub-program-v2",
            "model_identity": self.model_identity,
            "execution": self.execution,
            "entry_boundary": self.entry_boundary,
            "instructions": [
                {"after_steps": item.after_steps, "act": item.act.manifest()}
                for item in self.instructions
            ],
            "evidence_parent": self.evidence_parent,
            "evidence_plan": self.evidence_plan,
            "evidence_arm": self.evidence_arm,
            "context": self.context,
            "continuation_steps": self.continuation_steps,
        }

    @classmethod
    def compile(
        cls,
        session: Session,
        panel: dict[str, Any],
        arm: str,
        instructions: tuple[Instruction, ...],
        *,
        context: str,
    ) -> Program:
        cut = session.capture()
        rows = [row for row in panel["rows"] if row["name"] == arm]
        if len(rows) != 1 or rows[0]["status"] != "completed" or not instructions or not context:
            raise ValueError("compilation requires one completed measured arm and explicit context")
        if panel["parent"] != cut.fingerprint:
            raise ValueError("panel parent does not match the recipient state")
        if any(item.after_steps < 0 for item in instructions):
            raise ValueError("instruction delay must be non-negative")
        for item in instructions:
            item.act.require_reproducible()
        measured = [
            receipt["act"] for receipt in rows[0]["receipts"] if receipt["operation"] == "apply"
        ]
        if measured != [item.act.manifest() for item in instructions]:
            raise ValueError("program operations do not match the measured arm")
        # v1 certifies only the panel's pre-continuation schedule.
        if any(item.after_steps for item in instructions):
            raise ValueError("nonzero delays require a separately measured temporal panel")
        return cls(
            session.adapter.model_identity,
            dict(session.adapter.execution),
            cut.boundary,
            instructions,
            cut.fingerprint,
            panel["plan"],
            arm,
            context,
            panel["continuation_steps"],
        )

    def run(
        self, session: Session, *, context: str, continuation_steps: int | None = None
    ) -> Session:
        cut = session.capture()
        continuation_steps = (
            self.continuation_steps if continuation_steps is None else continuation_steps
        )
        if continuation_steps != self.continuation_steps:
            raise ValueError("continuation budget is outside the measured program support")
        if (
            cut.model_identity != self.model_identity
            or dict(cut.execution) != self.execution
            or cut.boundary != self.entry_boundary
            or cut.fingerprint != self.evidence_parent
            or context != self.context
        ):
            raise ValueError("program recipient/context is outside its measured support")
        branch = session.fork(cut)
        for instruction in self.instructions:
            instruction.act.require_reproducible()
            if instruction.after_steps:
                branch.continue_(instruction.after_steps)
            branch.apply(instruction.act)
        branch.continue_(continuation_steps)
        return branch


class GenericJSONAdapter(Adapter):
    """A declarative adapter factory for small JSON-only replay organisms."""

    def __init__(self, specification: Mapping[str, Any]):
        specification = clone(dict(specification))
        canonical(specification)
        if specification.get("kind") != "generic-json":
            raise ValueError("generic adapter specification requires kind='generic-json'")
        self.specification = specification
        self.model_identity = str(specification["model_identity"])
        self.execution = dict(specification["execution"])
        self._boundary = str(specification.get("boundary", "json-step"))
        self._slots = tuple(specification.get("slots", ()))
        if not self._slots or any(not isinstance(slot, str) or not slot for slot in self._slots):
            raise ValueError("generic JSON adapter requires named slots")

    def boundary(self, state: Mapping[str, Any]) -> str:
        cursor = self.specification.get("boundary_slot")
        return self._boundary if cursor is None else f"{self._boundary}:{state[cursor]}"

    def validate(self, state: Mapping[str, Any]) -> None:
        if set(state) != set(self._slots):
            raise ValueError("generic JSON state does not match its declared slots")
        canonical(dict(state))

    def addresses(self, state: Mapping[str, Any]) -> tuple[str, ...]:
        self.validate(state)
        return self._slots

    def advance(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        self.validate(state)
        result = clone(dict(state))
        transition = self.specification.get("advance", {"kind": "identity"})
        kind = transition.get("kind")
        if kind == "identity":
            return result
        if kind == "increment":
            slot = transition["slot"]
            if slot not in result or type(result[slot]) not in (int, float):
                raise ValueError("generic increment requires a numeric declared slot")
            result[slot] += transition.get("amount", 1)
            self.validate(result)
            return result
        raise ValueError("unknown generic JSON transition")


def _python_factory_digest(module: str, qualname: str) -> str:
    target: Any = importlib.import_module(module)
    for component in qualname.split("."):
        target = getattr(target, component)
    try:
        source = inspect.getsource(target).encode()
    except (OSError, TypeError):
        source = repr(target).encode()
    return hashlib.sha256(source).hexdigest()


@dataclass(frozen=True)
class ReplayBundle:
    """A sealed replay description plus local cut ancestry and optional data bytes."""

    _body: bytes

    @classmethod
    def create(
        cls,
        program: Program,
        store: LocalStore,
        *,
        branch_heads: Mapping[str, str],
        receipts: tuple[Receipt | Mapping[str, Any], ...] = (),
        environment: Mapping[str, Any] | None = None,
        adapter_factory: Mapping[str, Any],
        include_data: bool = False,
    ) -> ReplayBundle:
        if not branch_heads:
            raise ValueError("ReplayBundle requires at least one named branch head")
        factory = clone(dict(adapter_factory))
        factory.setdefault("model_identity", program.model_identity)
        factory.setdefault("execution", clone(program.execution))
        if (
            factory["model_identity"] != program.model_identity
            or factory["execution"] != program.execution
        ):
            raise ValueError("adapter factory identity differs from the measured program")
        kind = factory.get("kind")
        if kind == "python":
            observed = _python_factory_digest(factory["module"], factory["qualname"])
            supplied = factory.setdefault("implementation_digest", observed)
            if supplied != observed:
                raise ValueError("adapter factory implementation digest mismatch")
        elif kind == "generic-json":
            unsigned = {
                key: value for key, value in factory.items() if key != "implementation_digest"
            }
            observed = digest(unsigned)
            supplied = factory.setdefault("implementation_digest", observed)
            if supplied != observed:
                raise ValueError("generic adapter factory provenance digest mismatch")
        else:
            raise ValueError("adapter factory must be 'python' or 'generic-json'")
        graph = store.export_graph(branch_heads, include_data=include_data)
        for identifier, record in graph["cuts"].items():
            manifest = record["manifest"]
            if (
                manifest["model_identity"] != program.model_identity
                or manifest["execution"] != program.execution
            ):
                raise ValueError(f"cut identity differs from the program: {identifier}")
        receipt_rows = []
        for receipt in receipts:
            row = receipt.to_dict() if isinstance(receipt, Receipt) else clone(dict(receipt))
            if "fingerprint" not in row:
                row["fingerprint"] = hashlib.sha256(canonical(row)).hexdigest()
            receipt_rows.append(row)
        env = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            **({} if environment is None else clone(dict(environment))),
        }
        return cls(
            canonical(
                {
                    "schema": "saturn-pub-replay-bundle-v1",
                    "program": program.manifest(),
                    "branch_heads": dict(branch_heads),
                    "receipts": receipt_rows,
                    "environment": env,
                    "adapter_factory": factory,
                    "local_graph": graph,
                }
            )
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self._body).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**json.loads(self._body), "fingerprint": self.fingerprint}

    def verify(self) -> None:
        body = json.loads(self._body)
        if body.get("schema") != "saturn-pub-replay-bundle-v1":
            raise ValueError("unknown ReplayBundle schema")
        program = body.get("program", {})
        factory = body.get("adapter_factory", {})
        if factory.get("model_identity") != program.get("model_identity") or factory.get(
            "execution"
        ) != program.get("execution"):
            raise ValueError("ReplayBundle adapter/program identity mismatch")
        if factory.get("kind") == "generic-json":
            unsigned = {
                key: value for key, value in factory.items() if key != "implementation_digest"
            }
            if digest(unsigned) != factory.get("implementation_digest"):
                raise ValueError("ReplayBundle generic adapter provenance changed")
        graph = body.get("local_graph", {})
        if graph.get("heads") != body.get("branch_heads"):
            raise ValueError("ReplayBundle branch heads differ from its local graph")
        for name, identifier in body.get("branch_heads", {}).items():
            if not isinstance(name, str) or identifier not in graph.get("cuts", {}):
                raise ValueError("ReplayBundle branch head is not in local custody")
        for identifier, record in graph.get("cuts", {}).items():
            if digest(record.get("manifest")) != identifier:
                raise ValueError("ReplayBundle contains a cut with a broken identity")
            manifest = record["manifest"]
            if manifest.get("model_identity") != program.get("model_identity") or manifest.get(
                "execution"
            ) != program.get("execution"):
                raise ValueError("ReplayBundle cut/program identity mismatch")
        for receipt in body.get("receipts", ()):
            if not isinstance(receipt, dict) or "fingerprint" not in receipt:
                raise ValueError("ReplayBundle receipt is not sealed")
            sealed = dict(receipt)
            fingerprint = sealed.pop("fingerprint")
            if hashlib.sha256(canonical(sealed)).hexdigest() != fingerprint:
                raise ValueError("ReplayBundle receipt fingerprint mismatch")

    def save(self, path: str | Path) -> str:
        self.verify()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        body = canonical(self.to_dict())
        destination.write_bytes(body)
        return self.fingerprint

    export = save

    @classmethod
    def load(cls, path: str | Path) -> ReplayBundle:
        data = json.loads(Path(path).read_bytes())
        fingerprint = data.pop("fingerprint", None)
        bundle = cls(canonical(data))
        if fingerprint != bundle.fingerprint:
            raise ValueError("ReplayBundle fingerprint mismatch")
        bundle.verify()
        return bundle

    def _adapter(self, adapter_factory: Callable[[], Adapter] | None = None) -> Adapter:
        body = json.loads(self._body)
        provenance = body["adapter_factory"]
        if adapter_factory is not None:
            adapter = adapter_factory()
        elif provenance["kind"] == "generic-json":
            adapter = GenericJSONAdapter(provenance)
        else:
            if (
                _python_factory_digest(provenance["module"], provenance["qualname"])
                != provenance["implementation_digest"]
            ):
                raise ValueError("adapter factory implementation changed")
            target: Any = importlib.import_module(provenance["module"])
            for component in provenance["qualname"].split("."):
                target = getattr(target, component)
            adapter = target(**provenance.get("kwargs", {}))
        if not isinstance(adapter, Adapter):
            raise ValueError("adapter factory did not produce an Adapter")
        if (
            adapter.model_identity != provenance["model_identity"]
            or dict(adapter.execution) != provenance["execution"]
        ):
            raise ValueError("hydrated adapter identity differs from its provenance")
        return adapter

    def restore(
        self,
        store: LocalStore,
        *,
        allow_data: bool = False,
        adapter_factory: Callable[[], Adapter] | None = None,
    ) -> Adapter:
        self.verify()
        body = json.loads(self._body)
        # Construct and check the native identity before importing any state bytes.
        adapter = self._adapter(adapter_factory)
        graph = body["local_graph"]
        if graph.get("data_included"):
            store.import_graph(graph, allow_data=allow_data)
        elif allow_data:
            raise ValueError("ReplayBundle was exported without state data")
        return adapter
