"""Hyperion-inspired recipient-bound schedules compiled from measured panel rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core import Act, Session


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
            if instruction.after_steps:
                branch.continue_(instruction.after_steps)
            branch.apply(instruction.act)
        branch.continue_(continuation_steps)
        return branch
