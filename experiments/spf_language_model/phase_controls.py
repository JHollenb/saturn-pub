"""One scoped causal control: first-layer transition phase frequency 0 or 1.

This edits a registered integer-valued frequency buffer already consumed by
both the differentiable reference and canonical PhaseSequenceMixer. It adds no
parameter, controller, readout, primitive or bypass. Frequency zero removes A's
rotation; learned attenuation remains. B/C phases and payload generation still
run. Whole-model downstream changes remain consequences of the intervention.
The caller's public Saturn snapshot owns complete parent restoration.
"""

from __future__ import annotations

import operator

import torch
from memory_microscope import _fingerprint


@torch.no_grad()
def apply_transition_control(model, layers=(0,), frequency=0):
    """Return an explicit buffer receipt; leave uncaptured architecture alone.

    This experiment admits only layer 0 and scalar frequency 0/1. The selected
    buffer retains its dtype, shape, device and storage allocation. Every other
    state-dict value is checked unchanged after the edit. Optimizer topology,
    parameters, parameter gradients and architecture metadata are untouched.
    """
    if tuple(layers) != (0,):
        raise ValueError("this bounded control supports only layers=(0,)")
    if isinstance(frequency, bool):
        raise ValueError("frequency must be the integer 0 or 1")
    try:
        frequency = operator.index(frequency)
    except TypeError as error:
        raise ValueError("frequency must be the integer 0 or 1") from error
    if frequency not in (0, 1):
        raise ValueError("this bounded control supports only frequency 0 or 1")
    selected = "layers.0.transition_frequencies"
    state = model.state_dict()
    if selected not in state or not hasattr(model.layers[0], "transition_frequencies"):
        raise ValueError("model has no canonical first-layer transition frequency buffer")
    if selected not in dict(model.named_buffers()):
        raise ValueError(
            "transition frequency must be a registered buffer, not a trainable parameter"
        )
    buffer = model.layers[0].transition_frequencies
    if buffer.ndim != 1 or buffer.numel() != model.layers[0].state_size:
        raise ValueError("transition frequency buffer does not match the state geometry")
    before_state = {name: value.detach().clone() for name, value in state.items()}
    before_hash = _fingerprint(model)
    before_storage = buffer.data_ptr()
    buffer.fill_(frequency)
    after_state = model.state_dict()
    unchanged = set(before_state) == set(after_state) and all(
        torch.equal(before_state[name], after_state[name])
        for name in before_state
        if name != selected
    )
    if not unchanged or before_storage != buffer.data_ptr():
        raise RuntimeError("transition control changed storage or an undeclared model value")
    return {
        "schema": "spf-first-layer-transition-frequency-control-v1",
        "selected_layers": [0],
        "actual_transition_frequency": frequency,
        "changed_buffers": [
            {
                "name": selected,
                "before": before_state[selected].cpu().tolist(),
                "after": buffer.detach().cpu().tolist(),
                "changed": not torch.equal(before_state[selected], buffer),
            }
        ],
        "model_sha256_before": before_hash,
        "model_sha256_after": _fingerprint(model),
        "other_state_dict_values_unchanged": True,
        "architecture_metadata_modified": False,
        "mechanism": "A phase=feature*omega; omega0 is identity rotation with learned attenuation retained",
        "native_support": "canonical PhaseSequenceMixer transition_frequencies already supports integer zero",
        "scope": "first-layer A rotation only; B/C/decay/V/readout parameters unchanged",
        "terminal_status": "not-assessed",
    }
