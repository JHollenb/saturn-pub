"""Derive a terminal clock/reset scaffold from an observed live project cut.

This is declared non-native state construction, never skipped native execution.
The scaffold's latent is an observed-prefix placeholder. The returned public Act
installs the supplied predicted latent; only the caller applies it and renders.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from saturn_pub import Act, Session, StateCut
from saturn_pub.adapters import flux2 as _native
from saturn_pub.values import clone, describe, digest

from . import flux_final_latent_closure as _closure

ADAPTER_SOURCE_SHA256 = "2004b81971a258f1ca6295fe69d24767dc6bb6b92e103fa30cf6ccb0249b5a86"
CLOSURE_SOURCE_SHA256 = "b70f3d68f7fb0b1c0e2892bb7ab4efb8828a906ef80d9acb6f794608d673e579"
PORTS = _closure.PORTS
SOURCE_PORTS = _closure.SOURCE_PORTS


def _require(value, message):
    if not value:
        raise ValueError(message)


def _source_hash(module):
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class DerivedTerminal:
    """Validated scaffold, unapplied Act, and separately counted derivation."""

    scaffold: StateCut
    action: Act
    _record: Mapping[str, Any] = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_record", clone(dict(self._record)))

    @property
    def derivation(self):
        """A private copy of the self-sealed, non-native derivation metadata."""
        return clone(dict(self._record))


def derive_terminal(adapter, prefix_cut, *, predicted_latent, name="derived-terminal-prediction"):
    """Return a prefix-derived shell and four-port Act, performing no advance.

    The source must be a compatible, live native project boundary. Its complete
    thirteen-field payload supplies all carried values, terminal time is derived
    from its timestep-grid length, and the four native reset slots become None.
    The prefix latent remains only an explicitly labelled scaffold placeholder.
    Supplied predicted values are copied to its qualified layout without casts.

    Store ``scaffold`` and ``derivation`` separately, then apply ``action`` through
    the public Session/Navigator. Derivation metadata is not a native transition
    or an execution-receipt emission, and proves no intervening native operations.
    """
    _require(_source_hash(_native) == ADAPTER_SOURCE_SHA256
        and _source_hash(_closure) == CLOSURE_SOURCE_SHA256,
        "qualified native adapter or closure source differs")
    _require(type(adapter) is _closure.FluxFinalLatentClosureAdapter,
        "derived terminal requires the qualified final-latent closure adapter")
    _require(isinstance(prefix_cut, StateCut), "derived terminal source must be a StateCut")
    prefix_cut.verify()
    _require(prefix_cut.model_identity == adapter.model_identity
        and dict(prefix_cut.execution) == dict(adapter.execution),
        "derived terminal source model/execution identity differs")
    state = prefix_cut.payload
    _require(set(state) == {*PORTS, *SOURCE_PORTS}, "derived terminal requires the complete thirteen-field source")
    _require(state["phase"] == "project" and state["cursor"] == 0
        and 0 <= state["step"] < len(state["timesteps"])
        and all(state[key] is None for key in ("context", "image", "text", "noise_prediction")),
        "derived terminal requires a live project clock and native reset closure")
    _require(adapter.capture_external() is None, "external mutable closure is outside the qualified derivation")
    # Public construction revalidates the payload, source point, native surface
    # and complete model/execution metadata without hydrating/advancing a future.
    checked = Session(adapter, state, parent=prefix_cut.parent)
    _require(checked.capture(retain=False).fingerprint == prefix_cut.fingerprint,
        "observed source cut metadata does not match its native payload boundary")
    terminal_payload = {**state, "step": len(state["timesteps"]), "phase": "project", "cursor": 0,
        "context": None, "image": None, "text": None, "noise_prediction": None}
    derived = Session(adapter, terminal_payload, parent=prefix_cut.fingerprint)
    scaffold = derived.capture(retain=False)
    action = _closure.closure_act(scaffold, latent=predicted_latent, context=None, image=None, text=None, name=name)
    actual_payload = {**terminal_payload, **action.parameters["value"]}
    record = {"schema": "mdb-flux-derived-terminal-v1", "operation": "derive-terminal-clock-reset-scaffold",
        "authority": "declared non-native state construction from an observed prefix",
        "source_cut": prefix_cut.fingerprint, "source_point": dict(prefix_cut.execution_point or {}),
        "source_model_identity": prefix_cut.model_identity, "source_execution": dict(prefix_cut.execution),
        "source_preconditions": {key: digest(describe(state[key])) for key in (*PORTS, *SOURCE_PORTS)},
        "scaffold_cut": scaffold.fingerprint, "scaffold_parent": scaffold.parent,
        "scaffold_point": dict(scaffold.execution_point or {}),
        "scaffold_payload": describe(terminal_payload), "predicted_terminal_payload": describe(actual_payload),
        "placeholder_latent_source": {"cut": prefix_cut.fingerprint, "slot": "latent", "authority": "observed prefix; not native terminal latent"},
        "supplied_predicted_latent": describe(predicted_latent),
        "layout_adapted_predicted_latent": describe(actual_payload["latent"]), "action": action.manifest(),
        "rule": {"step": "len(source.timesteps)", "phase": "project", "cursor": 0,
            "execution_point_index": None, "reset": ["context", "image", "text", "noise_prediction"],
            "unchanged_sources": ["conditioning", "img_ids", "txt_ids", "timesteps", "sigmas"]},
        "source_hashes": {"native_adapter": ADAPTER_SOURCE_SHA256, "final_closure": CLOSURE_SOURCE_SHA256,
            "derivation_helper": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "native_steps": 0, "acts_applied": 0, "execution_receipt_emissions": 0,
        "scope": "This derived shell is not an observed native future or a continuation receipt. The caller may apply the returned source-bound Act and use the native terminal consumer before a native reference exists. Scientific forecast precision and later native comparisons are separate."}
    record["fingerprint"] = digest(record)
    return DerivedTerminal(scaffold, action, record)


__all__ = ["DerivedTerminal", "derive_terminal", "ADAPTER_SOURCE_SHA256", "CLOSURE_SOURCE_SHA256"]
