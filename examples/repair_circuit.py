"""A supplied two-writer neural circuit for controlled debugger demonstrations.

No parameters or symbolic roles are learned. Two fixed linear residual writers
reconstruct the same source; the native consumer thresholds the final carrier.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from saturn_pub import Adapter, ExecutionPoint, Session, SlotSpec, SurfaceManifest, TransitionSpec
from saturn_pub.values import describe, digest


class RepairCircuit(Adapter):
    operations = ("writer.first", "writer.repair", "consumer.threshold")

    def __init__(self):
        self.writers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False) for _ in range(2)])
        with torch.no_grad():
            for writer in self.writers:
                writer.weight.copy_(torch.eye(2))
        self.writers.eval().requires_grad_(False)
        self._weights = digest(describe(self.writers.state_dict()))
        self.model_identity = "supplied-repair-circuit:" + self._weights
        self.execution = {
            "family": "controlled-repair",
            "adapter": "RepairCircuit-v1",
            "program": "cpu-float32-residual-writers-v1",
            "torch": torch.__version__,
            "consumer_threshold": 0.5,
        }

    def validate_execution(self):
        if digest(describe(self.writers.state_dict())) != self._weights:
            raise ValueError("frozen repair circuit weights changed")

    def boundary(self, state):
        return "halted" if state["cursor"] == 3 else self.operations[state["cursor"]]

    def point(self, state):
        cursor = state["cursor"]
        return ExecutionPoint(
            "controlled-repair",
            0,
            "halted" if cursor == 3 else "execute",
            self.boundary(state),
            cursor,
            next_operation=self.boundary(state),
            local_clock=cursor,
        )

    def surface(self, state):
        return SurfaceManifest(
            (
                SlotSpec("source", role="source", producer="caller", consumer="writers"),
                SlotSpec(
                    "carrier",
                    role="carrier",
                    writable=True,
                    producer="residual-writer",
                    consumer="threshold",
                ),
                SlotSpec("output", role="consumer-output", producer="threshold", consumer="caller"),
                SlotSpec("cursor", role="clock", producer="dispatcher", consumer="dispatcher"),
            ),
            "controlled-repair-state-v1",
            "two-channel-contact-threshold",
        )

    def transition(self, state):
        reads = ("source", "carrier") if state["cursor"] < 2 else ("carrier",)
        writes = ("carrier", "cursor") if state["cursor"] < 2 else ("output", "cursor")
        return TransitionSpec(
            self.boundary(state),
            reads,
            writes,
            consumer="two-channel-contact-threshold",
            footprint="declared",
        )

    def validate(self, state: Mapping[str, Any]):
        if set(state) != {"source", "carrier", "output", "cursor"}:
            raise ValueError("incomplete repair circuit closure")
        if type(state["cursor"]) is not int or not 0 <= state["cursor"] <= 3:
            raise ValueError("invalid circuit cursor")
        for name in ("source", "carrier", "output"):
            value = state[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (2,)
                or value.dtype != torch.float32
                or value.device.type != "cpu"
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"invalid CPU float32 two-channel slot: {name}")

    def addresses(self, state):
        return ("source", "carrier", "output", "cursor")

    @torch.inference_mode()
    def advance(self, state):
        cursor = state["cursor"]
        if cursor == 3:
            raise ValueError("circuit halted")
        result = dict(state)
        if cursor < 2:
            # Real native linear computation, with supplied redundant identity weights.
            result["carrier"] = state["carrier"] + self.writers[cursor](
                state["source"] - state["carrier"]
            )
        else:
            result["output"] = (state["carrier"] >= 0.5).to(torch.float32)
        result["cursor"] = cursor + 1
        return result

    def session(self):
        return Session(
            self,
            {
                "source": torch.tensor([1.0, 0.0]),
                "carrier": torch.zeros(2),
                "output": torch.zeros(2),
                "cursor": 0,
            },
        )
