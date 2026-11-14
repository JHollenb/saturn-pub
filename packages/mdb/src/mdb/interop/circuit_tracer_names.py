"""circuit-tracer interop: an import shim and the correspondence of its graph nodes to mdb
addresses (``core.foreign_names.ct_*``).

circuit-tracer 0.5.0 declares ``huggingface-hub<1.0``; under hub 1.x its package import fails
on ``HF_HUB_ENABLE_HF_TRANSFER`` (obsidian/reviews/2026-11-14-library-env-setup.md).
``import_circuit_tracer`` sets that constant to False before importing. Attribution itself does
not run under transformers 5.x / nnsight 0.8 / TransformerLens 4 (``ReplacementModel`` import
errors; ``circuit_tracer_graph`` and docs/INTEROP.md give the pins that run it); the node ids
and the module locations of circuit-tracer's Qwen3 mapping are checked
(``tests/test_interop_symtab.py``).
"""

from __future__ import annotations

from typing import Any

# circuit-tracer's Qwen3 ``feature_hook_mapping`` key -> the mdb port at that location
HOOK_PORTS = {"mlp.hook_in": "n", "mlp.hook_out": "m"}


def import_circuit_tracer() -> Any:
    """``circuit_tracer`` imported under huggingface_hub 1.x (the constant shim)."""
    import huggingface_hub.constants as constants

    if not hasattr(constants, "HF_HUB_ENABLE_HF_TRANSFER"):
        constants.HF_HUB_ENABLE_HF_TRANSFER = False
    import circuit_tracer

    return circuit_tracer


def qwen3_locations(layer: int) -> dict[str, tuple[str, str]]:
    """mdb port -> (nnsight envoy path, input|output) from circuit-tracer's own Qwen3 mapping."""
    import_circuit_tracer()
    from circuit_tracer.utils.tl_nnsight_mapping import get_mapping

    mapping = get_mapping("Qwen3ForCausalLM")
    out = {HOOK_PORTS[k]: (path.format(layer=layer), side)
           for k, (path, side) in mapping.feature_hook_mapping.items() if k in HOOK_PORTS}
    out["emb"] = (mapping.embed_location, "output")
    return out


__all__ = ["HOOK_PORTS", "import_circuit_tracer", "qwen3_locations"]
