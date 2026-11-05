"""Opt-in atomic forecast closure at live native FLUX joint1/joint2.

The earlier joint0/joint1 adapter remains byte-for-byte independent. Native
advancement and context validation are inherited unchanged; only this distinct
typed admission surface permits the four-port replacement at joint2.
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

PORTS = _previous.PORTS
SOURCE_PORTS = _previous.SOURCE_PORTS
CURSORS = (1, 2)
ADAPTER_ID = "mdb-flux-joint-forecast-closure-v1"
STATE_SCHEMA = "mdb-flux2-joint-forecast-closure-state-v1"
OPERATION_ID = "mdb.flux.replace_joint_forecast_closure"
OPERATION_VERSION = "1"
DEPENDENCY_SHA256 = "45dd999fe54103ca7676248652798dd295fbc727d0fdf38cd912fa48b8011c23"


def _verify_dependency():
    if hashlib.sha256(Path(_previous.__file__).read_bytes()).hexdigest() != DEPENDENCY_SHA256:
        raise ValueError("qualified forecast closure validation/layout dependency source differs")


def _contract():
    return {"version": 1, "atomic_ports": list(PORTS), "phase": "joint", "cursors": list(CURSORS),
        "clock_writable": False, "validation_layout_dependency_sha256": DEPENDENCY_SHA256}


def _live_boundary(state):
    return state["phase"] == "joint" and state["cursor"] in CURSORS and state["step"] < len(state["timesteps"])


class FluxJointForecastClosureAdapter(_previous.FluxForecastClosureAdapter):
    """Native operation granularity with separately identified joint1/2 admission."""

    def __init__(self, *args, **kwargs):
        _verify_dependency()
        super().__init__(*args, **kwargs)
        if self.joint_layers < 3:
            raise ValueError("joint forecast closure requires live native joint1 and joint2 boundaries")
        self.execution = {**self.execution, "adapter": ADAPTER_ID, "forecast_closure": _contract()}

    def addresses(self, state):
        # Deliberately bypass the v1 override: it would expose the complete
        # closure at joint0, outside this adapter's two declared boundaries.
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
            # Calling super here would inherit v1's different joint0 policy.
            return Flux2KleinAdapter.write(self, state, writes, invalidates)
        if invalidates or set(writes) != set(PORTS):
            raise ValueError("joint forecast admission requires exactly four atomic closure ports")
        for key in PORTS:
            _previous._same_layout(writes[key], state[key], key)
        return Adapter.write(self, state, writes, ())


def closure_act(cut, *, latent, image, text, context, name="joint-forecast-closure"):
    """Source-bound four-port Act; layout copies never cast or advance clocks."""
    _verify_dependency()
    if (not _live_boundary(cut.payload) or cut.execution.get("adapter") != ADAPTER_ID
            or cut.execution.get("forecast_closure") != _contract()):
        raise ValueError("joint forecast closure requires the opt-in adapter at live native joint1/joint2")
    supplied = {"latent": latent, "context": context, "image": image, "text": text}
    copied = {key: _previous._copy_layout(supplied[key], cut.payload[key]) for key in PORTS}
    operation = _previous._ReplaceClosure(clone(copied))
    Act.register_operation(OPERATION_ID, OPERATION_VERSION, operation, pure=True)
    return Act.registered(name, (*PORTS, *SOURCE_PORTS), PORTS, operation_id=OPERATION_ID,
        operation_version=OPERATION_VERSION, implementation=operation, parameters={"value": clone(copied)},
        numerical_contract="behavioral-intervention",
        preconditions={key: digest(describe(cut.payload[key])) for key in (*PORTS, *SOURCE_PORTS)})


__all__ = ["FluxJointForecastClosureAdapter", "closure_act", "PORTS", "SOURCE_PORTS", "CURSORS", "ADAPTER_ID"]
