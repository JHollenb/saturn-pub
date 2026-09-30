"""Same-parent causal panels. Measurements stay separate from certification."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .core import Act, Session, StateCut
from .values import describe, digest


@dataclass(frozen=True)
class Arm:
    name: str
    acts: tuple[Act, ...] = ()
    role: str = "candidate"


@dataclass(frozen=True)
class Investigation:
    question: str
    arms: tuple[Arm, ...]
    steps: int

    def __post_init__(self) -> None:
        names = [arm.name for arm in self.arms]
        if not self.question or len(set(names)) != len(names) or not self.arms or self.steps < 1:
            raise ValueError("question, distinct arms, and a positive continuation budget required")

    @property
    def fingerprint(self) -> str:
        return digest(
            {
                "question": self.question,
                "steps": self.steps,
                "arms": [
                    {
                        "name": arm.name,
                        "role": arm.role,
                        "acts": [act.manifest() for act in arm.acts],
                    }
                    for arm in self.arms
                ],
            }
        )

    @classmethod
    def doses(
        cls, question: str, address: str, payload: Any, doses: Sequence[float], *, steps: int
    ) -> Investigation:
        arms = [Arm("native", role="native")]
        for index, dose in enumerate(doses):
            arms.append(
                Arm(
                    f"dose-{index}:{dose}",
                    (Act.add(address, payload, dose=dose),),
                    "no-op" if dose == 0 else "candidate",
                )
            )
        return cls(question, tuple(arms), steps)

    def run(
        self,
        session: Session,
        evaluator: Callable[[Session], Mapping[str, Any]],
        *,
        parent: StateCut | None = None,
    ) -> dict[str, Any]:
        cut = parent or session.capture()
        rows = []
        for arm in self.arms:
            branch = session.fork(cut)
            try:
                for act in arm.acts:
                    branch.apply(act)
                branch.continue_(self.steps)
                # Evaluate a private child so evaluators cannot mutate the measured branch.
                metrics = dict(evaluator(branch.fork()))
                row = {
                    "name": arm.name,
                    "role": arm.role,
                    "status": "completed",
                    "metrics": metrics,
                    "output": describe(branch.capture().payload),
                    "receipts": [r.to_dict() for r in branch.receipts],
                }
            except Exception as exc:
                row = {
                    "name": arm.name,
                    "role": arm.role,
                    "status": "instrument-error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "receipts": [r.to_dict() for r in branch.receipts],
                }
            rows.append(row)
        return {
            "schema": "saturn-pub-investigation-v1",
            "question": self.question,
            "plan": self.fingerprint,
            "parent": cut.fingerprint,
            "rows": rows,
            "terminal_status": "not-assessed",
            "independent_specimens": 1,
            "logical_arms": len(self.arms),
            "continuation_steps": self.steps,
            "budget_unit": "adapter-transition",
            "model_loads": 1,
        }
