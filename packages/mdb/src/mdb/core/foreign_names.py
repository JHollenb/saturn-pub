"""Foreign names of mdb cells, rows and reads (symbol-table additions review 2026-11-14, 2.8).

Pure text, no imports above ``core``: ``analysis.symbols`` attaches these to every table row,
``interop.names`` re-exports them beside its port table (``residual_names`` lives there, since
it needs the port table). TransformerLens names are the bridge-native hooks of
``interop.names.NAMES``; nnsight paths are module paths of a Hugging Face Qwen2/Qwen3/Llama.
"""

from __future__ import annotations

from typing import Any

# # ``verified`` says how a mapping is known: ``measured`` (tests/test_interop_cells.py reads the
# same value through the tool and mdb on tiny Qwen2/Qwen3, bitwise on prefilled rows), or
# ``unverified`` (written from the library's documentation; not installed or not tested here).
# A Neuronpedia link needs a supplied id mapping (Neuronpedia's model/source ids are not
# SAELens's release/sae_id), so it is labelled ``supplied`` and only built when one is given.

MEASURED_MAP = "measured"
UNVERIFIED_MAP = "unverified"
SUPPLIED_MAP = "supplied"

_CELL_PORT = {
    # mdb cache tensor -> (TL hook, TL index, nnsight path, nnsight index)
    # v: the cache holds v_proj's output (no v norm on Qwen2/Qwen3/Llama)
    "v": ("blocks.{l}.attn.v.hook_out", "[:, {r}, {g}, :]",
          "model.model.layers[{l}].self_attn.v_proj.output", "[:, {r}, {lo}:{hi}]"),
    # k: the cache holds the rotated (and k-normed) key
    "k": ("blocks.{l}.attn.hook_rot_k", "[:, {r}, {g}, :]",
          "model.model.layers[{l}].self_attn.source.attention_interface_1.inputs[0][2]",
          "[:, {g}, {r}, :]"),
}


def cell_names(layer: int, kv: int, row: int, port: str = "v", *,
               head_dim: int) -> dict[str, Any]:
    """Foreign names of the cache cell ``cache.L<layer>.<port>[kv<kv>,row=<row>]``: the
    TransformerLens hook and index, the nnsight path and index (both ``measured``), and a pyvene
    representation spec (``unverified``: pyvene is not installed here)."""
    if port not in _CELL_PORT:
        raise KeyError(f"cache port {port!r} is k or v")
    tl_hook, tl_index, nn_path, nn_index = _CELL_PORT[port]
    fill = {"l": layer, "r": row, "g": kv, "lo": kv * head_dim, "hi": (kv + 1) * head_dim}
    out = {
        "address": f"cache.L{layer}.{port}[kv{kv},row={row}]",
        "tl": {"hook": tl_hook.format(**fill), "index": tl_index.format(**fill),
               "legacy": (f"blocks.{layer}.attn.hook_{port}" if port == "v" else None),
               "verified": MEASURED_MAP},
        "nnsight": {"path": nn_path.format(**fill), "index": nn_index.format(**fill),
                    "verified": MEASURED_MAP},
        "pyvene": {"layer": layer,
                   "component": "value_output" if port == "v" else "key_output",
                   "unit": "pos", "unit_locations": [[row]],
                   "subspace": [kv * head_dim, (kv + 1) * head_dim],
                   "verified": UNVERIFIED_MAP,
                   "note": "pyvene's Qwen2 model map component string not checked against "
                           "installed source"},
    }
    return out


def row_names(row: int) -> dict[str, Any]:
    """Foreign names of a whole token row ``t<row>`` (every layer's k and v cells)."""
    return {
        "address": f"t{row}",
        "tl": {"hook": "blocks.*.attn.v.hook_out", "index": f"[:, {row}, :, :]",
               "keys": "blocks.*.attn.hook_rot_k", "embed": f"hook_embed[:, {row}]",
               "verified": MEASURED_MAP},
        "nnsight": {"path": "model.model.layers[*].self_attn.v_proj.output",
                    "index": f"[:, {row}, :]",
                    "embed": f"model.model.embed_tokens.output[:, {row}]",
                    "verified": MEASURED_MAP},
        "pyvene": {"component": "value_output", "unit": "pos", "unit_locations": [[row]],
                   "verified": UNVERIFIED_MAP},
    }


def read_names(row: int, layer: int, head: int, source: int | None = None) -> dict[str, Any]:
    """Foreign names of the read ``t<row>.L<layer>.h<head>`` (of ``source``, or every source):
    its attention weight in TL ``hook_pattern`` and nnsight's ``self_attn.output[1]``."""
    s = ":" if source is None else str(source)
    return {"address": f"t{row}.L{layer}.h{head}" + ("" if source is None else f"<-t{source}"),
            "tl": {"hook": f"blocks.{layer}.attn.hook_pattern",
                   "index": f"[:, {head}, {row}, {s}]", "verified": MEASURED_MAP},
            "nnsight": {"path": f"model.model.layers[{layer}].self_attn.output[1]",
                        "index": f"[:, {head}, {row}, {s}]", "verified": MEASURED_MAP}}


def neuronpedia_url(mapping: dict[str, Any] | None, release: str | None, sae_id: str | None,
                    feature: int) -> dict[str, Any]:
    """A Neuronpedia feature URL from a supplied mapping ``{"<release>/<sae_id>": {"model":
    M, "source": S}}`` (``supplied``), or the reason there is none."""
    key = f"{release}/{sae_id}"
    found = (mapping or {}).get(key)
    if not found:
        return {"state": "unresolved",
                "reason": f"no supplied Neuronpedia id mapping for {key} (Neuronpedia ids are "
                          "not SAELens ids)"}
    return {"url": f"https://www.neuronpedia.org/{found['model']}/{found['source']}/"
                   f"{int(feature)}", "verified": SUPPLIED_MAP}


def ct_node(layer: int, row: int, feature: int) -> dict[str, Any]:
    """A circuit-tracer attribution-graph node for a transcoder feature (``unverified``:
    circuit-tracer is not installed; it needs transformers <= 4.57)."""
    return {"layer": layer, "position": row, "feature": int(feature),
            "id": f"{layer}_{int(feature)}_{row}", "verified": UNVERIFIED_MAP}


def binding_names(cells: list[Any], *, head_dim: int, kv_heads: int, layers: int,
                  limit: int = 8) -> dict[str, Any]:
    """Foreign names for a binding's cells (``reads.Source`` items): a whole row gets
    ``row_names``; a cell set lists up to ``limit`` cells' ``cell_names`` and counts the
    rest."""
    out: list[dict[str, Any]] = []
    total = 0
    for cell in cells:
        if cell.layers is None and cell.kv is None and cell.ports is None:
            total += 1
            if len(out) < limit:
                out.append(row_names(cell.row))
            continue
        for layer in (cell.layers if cell.layers is not None else range(layers)):
            for kv in (cell.kv if cell.kv is not None else range(kv_heads)):
                for port in (cell.ports or ("k", "v")):
                    total += 1
                    if len(out) < limit:
                        out.append(cell_names(layer, kv, cell.row, port, head_dim=head_dim))
    return {"cells": out, "omitted": max(0, total - len(out))}
