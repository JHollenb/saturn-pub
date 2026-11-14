"""pyvene <-> mdb: the Qwen3 model-card shim and the cell-level interchange config.

pyvene 0.1.8 maps Qwen2 but not Qwen3 (``KeyError: Qwen3ForCausalLM`` in
``IntervenableModel``). Qwen3's decoder module tree is Qwen2's (``model.layers[l].self_attn
.{q,k,v,o}_proj``, ``mlp``), so ``register_qwen3()`` copies the Qwen2 entries of
``type_to_module_mapping`` and ``type_to_dimension_mapping`` to the Qwen3 classes (verified on
Qwen3-0.6B-Base, obsidian/reviews/2026-11-14-library-env-setup.md). Idempotent.

``value_cell_config(layer, kv, head_dim)``: the pyvene config of mdb's cell
``cache.L<l>.v[kv<g>,row=r]`` as a sequence-scope interchange at position ``r`` of
``model.layers[l].self_attn.v_proj.output`` restricted to the KV group's ``head_dim`` columns
(a ``subspace_partition`` of the projection output). mdb's ``set_cell`` at the same cell with
prefill <= r reproduces it (both change every reader of row r at layer l).
"""

from __future__ import annotations

from typing import Any


def register_qwen3() -> list[str]:
    """Add Qwen3 classes to pyvene's model-card maps (copies of Qwen2's); returns what it added."""
    import pyvene.models.intervenable_modelcard as card
    from transformers.models.qwen2 import modeling_qwen2 as q2
    from transformers.models.qwen3 import modeling_qwen3 as q3

    added = []
    for src, dst in ((q2.Qwen2Model, q3.Qwen3Model), (q2.Qwen2ForCausalLM, q3.Qwen3ForCausalLM)):
        for table in (card.type_to_module_mapping, card.type_to_dimension_mapping):
            if src in table and dst not in table:
                table[dst] = table[src]
                added.append(f"{dst.__name__}")
    return sorted(set(added))


def value_cell_config(layer: int, kv: int, head_dim: int, num_kv: int) -> dict[str, Any]:
    """The pyvene representation config of value cell ``cache.L<layer>.v[kv<kv>]`` (the row is
    the ``unit_locations`` position at call time)."""
    import pyvene as pv

    return {"layer": layer, "component": f"model.layers[{layer}].self_attn.v_proj.output",
            "unit": "pos", "max_number_of_units": 1,
            "intervention_type": pv.VanillaIntervention,
            "subspace_partition": [[i * head_dim, (i + 1) * head_dim] for i in range(num_kv)],
            "kv": kv}


__all__ = ["register_qwen3", "value_cell_config"]
