"""A parameter-free graph machine used by the software debugger example."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from saturn_pub import Adapter, Session
from saturn_pub.contracts import ExecutionPoint, SlotSpec, SurfaceManifest


class GraphMachine(Adapter):
    """Execute supplied graph/rule state; the adapter owns no learned parameters."""

    model_identity = "parameter-free-graph-machine-v1"
    execution = {
        "family": "graph-machine",
        "adapter": "examples.software_machine.GraphMachine",
        "program": "exact-float32-cpu-v1",
    }

    def boundary(self, state: Mapping[str, Any]) -> str:
        cursor = state["cursor"]
        return "halted" if cursor == len(state["graph"]) else f"node:{state['graph'][cursor]}"

    def point(self, state: Mapping[str, Any]) -> ExecutionPoint:
        cursor = state["cursor"]
        operator = "halt" if cursor == len(state["graph"]) else state["graph"][cursor]
        return ExecutionPoint(
            "graph-machine",
            cursor,
            "halted" if operator == "halt" else "execute",
            operator=operator,
            index=cursor,
            next_operation=operator,
            local_clock=cursor,
        )

    def surface(self, state: Mapping[str, Any]) -> SurfaceManifest:
        return SurfaceManifest(
            (
                SlotSpec("graph", role="structure", producer="caller", consumer="dispatcher"),
                SlotSpec("rules", role="program", producer="caller", consumer="rule"),
                SlotSpec(
                    "memory",
                    role="memory",
                    writable=True,
                    producer="caller-or-Act",
                    consumer="load",
                ),
                SlotSpec("cursor", role="clock", producer="transition", consumer="dispatcher"),
                SlotSpec(
                    "register",
                    role="carrier",
                    writable=True,
                    producer="load-or-rule",
                    consumer="rule-or-emit",
                ),
                SlotSpec("output", role="consumer-output", producer="emit", consumer="caller"),
            ),
            "parameter-free-graph-state-v1",
            "graph-output",
            horizon="three-native-transitions",
        )

    def validate(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"graph", "rules", "memory", "cursor", "register", "output"}:
            raise ValueError("graph machine state closure is incomplete")
        if state["graph"] != ("load", "rule", "emit"):
            raise ValueError("unsupported graph")
        if set(state["rules"]) != {"scale", "bias"}:
            raise ValueError("scale and bias rules required")
        if type(state["cursor"]) is not int or not 0 <= state["cursor"] <= 3:
            raise ValueError("cursor outside graph")
        for slot in ("memory", "register", "output"):
            value = state[slot]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
                raise ValueError(f"{slot} must be a float32 tensor")
        if state["memory"].shape != (4,) or state["register"].shape != (1,):
            raise ValueError("invalid memory/register geometry")
        if state["output"].shape != (1,):
            raise ValueError("invalid output geometry")

    def addresses(self, state: Mapping[str, Any]) -> tuple[str, ...]:
        return ("graph", "rules", "memory", "cursor", "register", "output")

    def advance(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        cursor = state["cursor"]
        if cursor == 3:
            raise ValueError("graph machine is halted")
        result = dict(state)
        if cursor == 0:
            result["register"] = state["memory"][:1].clone()
        elif cursor == 1:
            result["register"] = (
                state["register"] * state["rules"]["scale"] + state["rules"]["bias"]
            )
        else:
            result["output"] = state["register"].clone()
        result["cursor"] = cursor + 1
        return result

    def session(self) -> Session:
        return Session(
            self,
            {
                "graph": ("load", "rule", "emit"),
                "rules": {"scale": 2.0, "bias": 1.0},
                "memory": torch.tensor([3.0, 17.0, 19.0, 23.0]),
                "cursor": 0,
                "register": torch.zeros(1),
                "output": torch.zeros(1),
            },
        )


def create_graph_machine(**_: Any) -> GraphMachine:
    """Stable, importable adapter factory for ReplayBundle provenance."""
    return GraphMachine()


__all__ = ["GraphMachine", "create_graph_machine"]
