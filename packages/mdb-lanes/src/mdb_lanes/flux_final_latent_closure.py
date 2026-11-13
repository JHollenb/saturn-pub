"""Opt-in complete forecast admission before the last step or after commit.

The terminal closure is the native scheduler-reset state: only latent remains
a tensor among the four atomic ports, while context/image/text are exactly None.
Native advancement and validation are inherited unchanged.
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
from . import flux_joint_tail_closure as _tail

PORTS = _previous.PORTS
SOURCE_PORTS = _previous.SOURCE_PORTS
ADAPTER_ID = "mdb-flux-final-latent-closure-v1"
STATE_SCHEMA = "mdb-flux2-final-latent-closure-state-v1"
OPERATION_ID = "mdb.flux.replace_final_latent_closure"
OPERATION_VERSION = "1"
DEPENDENCY_SHA256 = "45dd999fe54103ca7676248652798dd295fbc727d0fdf38cd912fa48b8011c23"
JOINT_DEPENDENCY_SHA256 = "bb352ae4859335abdd751d33ccc2774fa4d37c53f5167c5012183c693b0d709c"
TAIL_DEPENDENCY_SHA256 = "2788c984045098532009e105219e7c9cfcc38732a54d892bdc296149a979b586"


def _verify_dependencies():
    for module, expected in ((_previous, DEPENDENCY_SHA256), (_joint, JOINT_DEPENDENCY_SHA256),
            (_tail, TAIL_DEPENDENCY_SHA256)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise ValueError("qualified forecast closure dependency source differs")


def _contract():
    return {"version": 1, "atomic_ports": list(PORTS), "clock_writable": False,
        "boundaries": [{"step": "len(timesteps)-1", "phase": "single", "cursor": 0, "live": True},
            {"step": "len(timesteps)", "phase": "project", "cursor": 0,
                "reset_ports": ["context", "image", "text"]}],
        "validation_layout_dependency_sha256": DEPENDENCY_SHA256,
        "prior_joint_dependency_sha256": JOINT_DEPENDENCY_SHA256,
        "prior_tail_dependency_sha256": TAIL_DEPENDENCY_SHA256}


def _admitted_boundary(state):
    steps = len(state["timesteps"])
    return ((state["step"] == steps-1 and state["phase"] == "single" and state["cursor"] == 0)
        or (state["step"] == steps and state["phase"] == "project" and state["cursor"] == 0))


def _same_layout(value, reference, path):
    if reference is None:
        if value is not None:
            raise ValueError("terminal forecast reset port must remain None: " + path)
    else:
        _previous._same_layout(value, reference, path)


def _copy_layout(value, reference):
    if reference is None:
        if value is not None:
            raise ValueError("terminal forecast reset payload must remain None")
        return None
    return _previous._copy_layout(value, reference)


class FluxFinalLatentClosureAdapter(_previous.FluxForecastClosureAdapter):
    """Unchanged native operations with explicit single0/terminal admission."""

    def __init__(self, *args, **kwargs):
        _verify_dependencies()
        super().__init__(*args, **kwargs)
        self.execution = {**self.execution, "adapter": ADAPTER_ID, "forecast_closure": _contract()}

    def addresses(self, state):
        result = Flux2KleinAdapter.addresses(self, state)
        if not _admitted_boundary(state):
            return result
        return tuple(dict.fromkeys((*result, *PORTS, *SOURCE_PORTS)))

    def surface(self, state):
        original = Flux2KleinAdapter.surface(self, state)
        slots = tuple(replace(slot, writable=slot.name in PORTS) for slot in original.slots) if _admitted_boundary(state) else original.slots
        return replace(original, slots=slots, state_schema=STATE_SCHEMA)

    def write(self, state, writes, invalidates):
        if not _admitted_boundary(state):
            return Flux2KleinAdapter.write(self, state, writes, invalidates)
        if invalidates or set(writes) != set(PORTS):
            raise ValueError("final-latent forecast admission requires exactly four atomic closure ports")
        for key in PORTS:
            _same_layout(writes[key], state[key], key)
        return Adapter.write(self, state, writes, ())


def closure_act(cut, *, latent, image, text, context, name="final-latent-forecast-closure"):
    """Preserve all source/clock bindings and native terminal reset None values."""
    _verify_dependencies()
    if (not _admitted_boundary(cut.payload) or cut.execution.get("adapter") != ADAPTER_ID
            or cut.execution.get("forecast_closure") != _contract()):
        raise ValueError("final-latent forecast closure requires final-step/single0 or terminal project0")
    supplied = {"latent": latent, "context": context, "image": image, "text": text}
    copied = {key: _copy_layout(supplied[key], cut.payload[key]) for key in PORTS}
    operation = _previous._ReplaceClosure(clone(copied))
    Act.register_operation(OPERATION_ID, OPERATION_VERSION, operation, pure=True)
    return Act.registered(name, (*PORTS, *SOURCE_PORTS), PORTS, operation_id=OPERATION_ID,
        operation_version=OPERATION_VERSION, implementation=operation, parameters={"value": clone(copied)},
        numerical_contract="behavioral-intervention",
        preconditions={key: digest(describe(cut.payload[key])) for key in (*PORTS, *SOURCE_PORTS)})


__all__ = ["FluxFinalLatentClosureAdapter", "closure_act", "PORTS", "SOURCE_PORTS", "ADAPTER_ID"]
