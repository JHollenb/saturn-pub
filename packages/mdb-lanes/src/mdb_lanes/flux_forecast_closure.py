"""Opt-in atomic forecast admission at native FLUX joint0/joint1 boundaries.

Native execution is inherited unchanged. This distinct adapter contract adds a
typed four-port replacement; it never advances or writes the execution clock.
Import only with the public diffusion dependencies installed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch
from saturn_pub import Act
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.core import Adapter
from saturn_pub.values import clone, describe, digest

PORTS = ("latent", "context", "image", "text")
SOURCE_PORTS = ("step", "phase", "cursor", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas", "noise_prediction")
OPERATION_ID = "mdb.flux.replace_forecast_closure"
OPERATION_VERSION = "1"


def _live_boundary(state):
    return (state["phase"] == "joint" and state["cursor"] in (0, 1)
        and state["step"] < len(state["timesteps"]))


def _same_layout(value, reference, path):
    if isinstance(reference, torch.Tensor):
        if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
                or value.dtype != reference.dtype or value.device != reference.device
                or value.stride() != reference.stride() or not bool(torch.isfinite(value).all())):
            raise ValueError("forecast closure tensor layout/type/finite mismatch: " + path)
    elif isinstance(reference, dict):
        if not isinstance(value, dict) or set(value) != set(reference):
            raise ValueError("forecast closure mapping mismatch: " + path)
        for key in reference:
            _same_layout(value[key], reference[key], path + "." + key)
    elif isinstance(reference, tuple):
        if not isinstance(value, tuple) or len(value) != len(reference):
            raise ValueError("forecast closure tuple mismatch: " + path)
        for index, (child, parent) in enumerate(zip(value, reference)):
            _same_layout(child, parent, path + "." + str(index))
    else:
        raise ValueError("unsupported forecast closure leaf: " + path)


def _copy_layout(value, reference):
    if isinstance(reference, torch.Tensor):
        if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
                or value.dtype != reference.dtype or value.device != reference.device
                or not bool(torch.isfinite(value).all())):
            raise ValueError("forecast closure payload shape/dtype/device/finite mismatch")
        with torch.inference_mode():
            result = torch.empty_strided(reference.shape, reference.stride(),
                dtype=reference.dtype, device=reference.device)
            result.copy_(value)
        return result
    if isinstance(reference, dict) and isinstance(value, dict) and set(value) == set(reference):
        return {key: _copy_layout(value[key], reference[key]) for key in reference}
    if isinstance(reference, tuple) and isinstance(value, tuple) and len(value) == len(reference):
        return tuple(_copy_layout(child, parent) for child, parent in zip(value, reference))
    raise ValueError("forecast closure payload structure mismatch")


class FluxForecastClosureAdapter(Flux2KleinAdapter):
    """Native Klein execution with an explicitly writable complete forecast state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.granularity != "operation":
            raise ValueError("forecast closure requires operation granularity")
        self.execution = {**self.execution, "adapter": "mdb-flux-forecast-closure-v1",
            "forecast_closure": {"version": 1, "atomic_ports": list(PORTS),
                "phase": "joint", "cursors": [0, 1], "clock_writable": False}}

    def addresses(self, state):
        result = super().addresses(state)
        if not _live_boundary(state):
            return result
        return tuple(dict.fromkeys((*result, "context", *SOURCE_PORTS)))

    def surface(self, state):
        original = super().surface(state)
        if not _live_boundary(state):
            return replace(original, state_schema="mdb-flux2-forecast-closure-state-v1")
        slots = tuple(replace(slot, writable=(slot.name in PORTS)) for slot in original.slots)
        return replace(original, slots=slots, state_schema="mdb-flux2-forecast-closure-state-v1")

    def validate(self, state):
        super().validate(state)
        if state["context"] is None:
            return
        width = self.model.config.num_attention_heads * self.model.config.attention_head_dim
        length = state["image"].shape[1] + state["text"].shape[1]
        context = state["context"]
        for key, shape in (("temb", (1, width)), ("img_mod", (1, 6 * width)),
                ("txt_mod", (1, 6 * width)), ("single_mod", (1, 3 * width))):
            value = context[key]
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or value.dtype != self.dtype or value.device != self.device):
                raise ValueError("forecast closure context geometry/dtype/device mismatch: " + key)
        rope = context["rope"]
        if not isinstance(rope, tuple) or len(rope) != 2:
            raise ValueError("forecast closure rope must contain cosine and sine")
        for value in rope:
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (length, self.model.config.attention_head_dim)
                    or value.dtype != torch.float32 or value.device != self.device):
                raise ValueError("forecast closure rope geometry/dtype/device mismatch")

    def write(self, state, writes, invalidates):
        if _live_boundary(state):
            if invalidates or set(writes) != set(PORTS):
                raise ValueError("joint forecast admission requires exactly four atomic closure ports")
            for key in PORTS:
                _same_layout(writes[key], state[key], key)
            return Adapter.write(self, state, writes, ())
        return super().write(state, writes, invalidates)


@dataclass(frozen=True)
class _ReplaceClosure:
    value: Mapping

    def __call__(self, reads):
        return dict(self.value)


def closure_act(cut, *, latent, image, text, context, name="forecast-closure"):
    """Build a source-bound pure Act, preserving every current tensor layout.

    The caller supplies all four values, including a complete predicted context.
    Layout adaptation copies values without casts or arithmetic. The adapter
    remains the authority for admission and never accepts clock changes.
    """
    if not _live_boundary(cut.payload) or cut.execution.get("adapter") != "mdb-flux-forecast-closure-v1":
        raise ValueError("forecast closure requires the opt-in adapter at native joint0/joint1")
    supplied = {"latent": latent, "context": context, "image": image, "text": text}
    copied = {key: _copy_layout(supplied[key], cut.payload[key]) for key in PORTS}
    operation = _ReplaceClosure(clone(copied))
    Act.register_operation(OPERATION_ID, OPERATION_VERSION, operation, pure=True)
    return Act.registered(name, (*PORTS, *SOURCE_PORTS), PORTS, operation_id=OPERATION_ID,
        operation_version=OPERATION_VERSION, implementation=operation,
        parameters={"value": clone(copied)}, numerical_contract="behavioral-intervention",
        preconditions={key: digest(describe(cut.payload[key])) for key in (*PORTS, *SOURCE_PORTS)})


__all__ = ["FluxForecastClosureAdapter", "closure_act", "PORTS"]
