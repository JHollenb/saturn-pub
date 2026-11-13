"""Opt-in atomic forecast admission along the native five-joint tail.

Native advancement and context validation stay inherited from the qualified
forecast adapter. This distinct surface admits four-port replacements only at
joint2/3/4 and the following single0 boundary; clocks remain read-only.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from saturn_pub import Act
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.core import Adapter
from saturn_pub.values import clone, describe, digest

from . import flux_forecast_closure as _previous
from . import flux_joint_forecast_closure as _joint

PORTS = _previous.PORTS
SOURCE_PORTS = _previous.SOURCE_PORTS
CURSORS = (2, 3, 4)
ADAPTER_ID = "mdb-flux-joint-tail-closure-v1"
STATE_SCHEMA = "mdb-flux2-joint-tail-closure-state-v1"
OPERATION_ID = "mdb.flux.replace_joint_tail_closure"
OPERATION_VERSION = "1"
DEPENDENCY_SHA256 = "45dd999fe54103ca7676248652798dd295fbc727d0fdf38cd912fa48b8011c23"
JOINT_DEPENDENCY_SHA256 = "bb352ae4859335abdd751d33ccc2774fa4d37c53f5167c5012183c693b0d709c"


def _verify_dependencies():
    for module, expected in ((_previous, DEPENDENCY_SHA256), (_joint, JOINT_DEPENDENCY_SHA256)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise ValueError("qualified forecast closure dependency source differs")


def _contract():
    return {"version": 1, "atomic_ports": list(PORTS),
        "boundaries": {"joint": list(CURSORS), "single": [0]}, "clock_writable": False,
        "validation_layout_dependency_sha256": DEPENDENCY_SHA256,
        "prior_joint_dependency_sha256": JOINT_DEPENDENCY_SHA256}


def _live_boundary(state):
    return (state["step"] < len(state["timesteps"])
        and ((state["phase"] == "joint" and state["cursor"] in CURSORS)
            or (state["phase"] == "single" and state["cursor"] == 0)))


class FluxJointTailClosureAdapter(_previous.FluxForecastClosureAdapter):
    """Unchanged native operations with a separately identified tail surface."""

    def __init__(self, *args, **kwargs):
        _verify_dependencies()
        super().__init__(*args, **kwargs)
        if self.joint_layers != 5:
            raise ValueError("joint-tail closure requires exactly five native joint blocks")
        self.execution = {**self.execution, "adapter": ADAPTER_ID, "forecast_closure": _contract()}

    def addresses(self, state):
        # Native base policy avoids inheriting the older joint0/1 surface.
        result = Flux2KleinAdapter.addresses(self, state)
        if not _live_boundary(state):
            return result
        return tuple(dict.fromkeys((*result, *PORTS, *SOURCE_PORTS)))

    def surface(self, state):
        original = Flux2KleinAdapter.surface(self, state)
        slots = tuple(replace(slot, writable=slot.name in PORTS) for slot in original.slots) if _live_boundary(state) else original.slots
        return replace(original, slots=slots, state_schema=STATE_SCHEMA)

    def write(self, state, writes, invalidates):
        if not _live_boundary(state):
            return Flux2KleinAdapter.write(self, state, writes, invalidates)
        if invalidates or set(writes) != set(PORTS):
            raise ValueError("joint-tail forecast admission requires exactly four atomic closure ports")
        for key in PORTS:
            _previous._same_layout(writes[key], state[key], key)
        return Adapter.write(self, state, writes, ())


def closure_act(cut, *, latent, image, text, context, name="joint-tail-forecast-closure"):
    """Build a thirteen-source-bound atomic Act without casts or clock writes."""
    _verify_dependencies()
    if (not _live_boundary(cut.payload) or cut.execution.get("adapter") != ADAPTER_ID
            or cut.execution.get("forecast_closure") != _contract()):
        raise ValueError("joint-tail forecast closure requires live native joint2/3/4 or single0")
    supplied = {"latent": latent, "context": context, "image": image, "text": text}
    copied = {key: _previous._copy_layout(supplied[key], cut.payload[key]) for key in PORTS}
    operation = _previous._ReplaceClosure(clone(copied))
    Act.register_operation(OPERATION_ID, OPERATION_VERSION, operation, pure=True)
    return Act.registered(name, (*PORTS, *SOURCE_PORTS), PORTS, operation_id=OPERATION_ID,
        operation_version=OPERATION_VERSION, implementation=operation, parameters={"value": clone(copied)},
        numerical_contract="behavioral-intervention",
        preconditions={key: digest(describe(cut.payload[key])) for key in (*PORTS, *SOURCE_PORTS)})


__all__ = ["FluxJointTailClosureAdapter", "closure_act", "PORTS", "SOURCE_PORTS", "CURSORS", "ADAPTER_ID"]
