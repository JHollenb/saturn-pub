"""pyvene interop: the Qwen3 model-card shim and the specs of mdb addresses
(``core.foreign_names`` attaches the specs to symbol-table rows).

pyvene 0.1.8 has Qwen2 entries only, so ``IntervenableModel`` on a Qwen3 model raises
``KeyError: Qwen3ForCausalLM``. Qwen3's decoder modules carry the same names as Qwen2's
(``layers[l].self_attn.{q,k,v,o}_proj``, ``layers[l].mlp``), so ``qwen3_shim`` copies the
Qwen2 entries of ``pyvene.models.intervenable_modelcard`` to the Qwen3 classes (the workaround
in obsidian/reviews/2026-11-14-library-env-setup.md). Residual sites use an explicit module
path (``"model.layers[l].output"``, unit ``pos``), which needs no model-card entry.

Measured on tiny Qwen3 (``tests/test_interop_symtab.py``): ``collect`` of a value cell's spec
(``head_value_output``, unit ``h.pos``) equals mdb's cache cell, and of a ``h_out`` spec equals
mdb's ``t<r>.L<l>.h_out``, within float32 rounding.
"""

from __future__ import annotations

import re
from typing import Any

_CELL = re.compile(r"^cache\.L(\d+)\.v\[kv(\d+),row=(\d+)\]$")
_PORT = re.compile(r"^t(\d+)\.L(\d+)\.(h_out|h)$")
_PAIRS = (("Qwen2Model", "Qwen3Model"), ("Qwen2ForCausalLM", "Qwen3ForCausalLM"),
          ("Qwen2ForSequenceClassification", "Qwen3ForSequenceClassification"))


def qwen3_shim() -> list[str]:
    """Give pyvene's model card Qwen3 entries copied from Qwen2 (idempotent); the classes added."""
    from pyvene.models import intervenable_modelcard as card
    from transformers.models.qwen2 import modeling_qwen2 as q2
    from transformers.models.qwen3 import modeling_qwen3 as q3

    added = []
    for old, new in _PAIRS:
        src, dst = getattr(q2, old, None), getattr(q3, new, None)
        if src is None or dst is None or src not in card.type_to_module_mapping:
            continue
        if dst not in card.type_to_module_mapping:
            card.type_to_module_mapping[dst] = card.type_to_module_mapping[src]
            card.type_to_dimension_mapping[dst] = card.type_to_dimension_mapping[src]
            added.append(new)
    return added


def spec(address: str) -> dict[str, Any]:
    """The pyvene representation of an mdb address: ``cache.L<l>.v[kv<g>,row=<r>]`` (one KV
    head's value at one position) or ``t<r>.L<l>.h_out`` / ``t<r>.L<l>.h`` (the residual
    leaving / entering block ``l``). ``unit_locations`` is in pyvene's call form."""
    text = address.replace(" ", "")
    match = _CELL.match(text)
    if match:
        layer, kv, row = (int(x) for x in match.groups())
        return {"layer": layer, "component": "head_value_output", "unit": "h.pos",
                "unit_locations": (([[kv]], [[row]]),)}
    match = _PORT.match(text)
    if match:
        row, layer, port = int(match[1]), int(match[2]), match[3]
        component = f"model.layers[{layer}].output" if port == "h_out" else "block_input"
        return {"layer": layer, "component": component, "unit": "pos",
                "unit_locations": [[[row]]]}
    raise KeyError(f"no pyvene spec for {address!r} (a value cell or t<r>.L<l>.h|h_out)")


def collect(model: Any, tokens: list[int], address: str) -> Any:
    """The value pyvene's ``CollectIntervention`` gathers at ``address``'s spec in one forward
    of ``tokens`` (batch 1)."""
    import pyvene as pv
    import torch

    qwen3_shim()
    found = spec(address)
    config = pv.IntervenableConfig(representations=[pv.RepresentationConfig(
        layer=found["layer"], component=found["component"], unit=found["unit"],
        max_number_of_units=1, intervention_type=pv.CollectIntervention)])
    wrapped = pv.IntervenableModel(config, model)
    ids = torch.tensor([list(tokens)], device=next(model.parameters()).device)
    with torch.no_grad():
        out = wrapped({"input_ids": ids}, unit_locations={"base": found["unit_locations"]})
    return out[0][1][0]


__all__ = ["collect", "qwen3_shim", "spec"]
