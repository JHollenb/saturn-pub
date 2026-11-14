"""One name table: mdb ports <-> TransformerLens (TransformerBridge, v4) hook names <-> nnsight
module paths, and the layout conversions between them.

Measured on tiny Qwen2/Qwen3 (``tests/test_interop.py``): every mapped value is bitwise equal
across mdb ``trace``, an unprocessed ``TransformerBridge`` wrapping the same module, and nnsight
on the same module. Weight-processed TransformerLens (``enable_compatibility_mode()``) lives in
another basis; see ``docs/INTEROP.md`` for what changes.

Layouts (one sequence position ``r`` of a full forward over ``T`` tokens):

- ``seq``: tool ``(batch, pos, X)`` <-> mdb ``(batch, 1, X)``
- ``heads_flat``: TL ``(batch, pos, head, head_dim)`` <-> mdb ``(batch, 1, head*head_dim)``
- ``heads``: nnsight ``(batch, pos, head, head_dim)`` <-> mdb ``(batch, 1, head, head_dim)``
- ``heads_t``: TL q_norm/k_norm ``(batch, head, pos, head_dim)`` <-> mdb ``(batch, 1, head, head_dim)``
- ``rot_q``: TL ``(batch, pos, head, head_dim)`` <-> mdb ``(batch, head, 1, head_dim)``
- ``rot_k``: TL ``(batch, pos, kv, head_dim)`` <-> mdb ``(batch, kv, r+1, head_dim)`` (mdb's ``kr``
  holds the cached rows too)
- ``scores``: ``(batch, head, query, key)`` <-> mdb ``(batch, head, 1, r+1)``

This module imports no torch at import time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Name:
    port: str
    tl: str | None              # bridge-native hook name (``{l}`` = layer)
    tl_aliases: tuple[str, ...]  # other TL names of the same value (legacy / component names)
    tl_layout: str | None
    nn: str | None              # nnsight path relative to ``model.model.layers[l]``
    nn_layout: str | None
    note: str = ""


_A = "blocks.{l}.attn."
NAMES: tuple[Name, ...] = (
    Name("h", "blocks.{l}.hook_resid_pre", ("blocks.{l}.hook_in", "blocks.{l}.ln1.hook_in"),
         "seq", ".input", "seq"),
    Name("a", "blocks.{l}.ln1.hook_out", ("blocks.{l}.attn.hook_in",), "seq",
         ".input_layernorm.output", "seq",
         "ln1.hook_normalized is RMSNorm without w_in; equal to a only with fold_ln"),
    Name("q", _A + "q.hook_out", (_A + "hook_q",), "heads_flat", ".self_attn.q_proj.output", "seq"),
    Name("qn", _A + "q_norm.hook_out", (), "heads_t", ".self_attn.q_norm.output", "heads"),
    Name("k", _A + "k.hook_out", (_A + "hook_k",), "heads_flat", ".self_attn.k_proj.output", "seq"),
    Name("kn", _A + "k_norm.hook_out", (), "heads_t", ".self_attn.k_norm.output", "heads"),
    Name("v", _A + "v.hook_out", (_A + "hook_v",), "heads_flat", ".self_attn.v_proj.output", "seq"),
    Name("qr", _A + "hook_rot_q", (), "rot_q", ".self_attn.source.attention_interface_1.inputs[0][1]",
         "bh"),
    Name("kr", _A + "hook_rot_k", (), "rot_k", ".self_attn.source.attention_interface_1.inputs[0][2]",
         "bh_k"),
    Name("s", _A + "hook_attn_scores", (), "scores", None, None,
         "mdb s is a replica (scaling*qr kr^T + mask); nnsight has no module output for it"),
    Name("p", _A + "hook_pattern", (), "scores", ".self_attn.output[1]", "scores"),
    Name("A", _A + "hook_z", (_A + "o.hook_in",), "heads_flat", ".self_attn.o_proj.input", "seq"),
    Name("o", "blocks.{l}.hook_attn_out", (_A + "hook_out", _A + "o.hook_out"), "seq",
         ".self_attn.o_proj.output", "seq"),
    Name("r", "blocks.{l}.hook_resid_mid", ("blocks.{l}.ln2.hook_in",), "seq",
         ".post_attention_layernorm.input", "seq"),
    Name("n", "blocks.{l}.ln2.hook_out", ("blocks.{l}.mlp.hook_in", "blocks.{l}.mlp.gate.hook_in",
                                          "blocks.{l}.mlp.in.hook_in"), "seq",
         ".mlp.input", "seq", "ln2.hook_normalized is RMSNorm without w_post"),
    Name("g", "blocks.{l}.mlp.gate.hook_out", ("blocks.{l}.mlp.hook_pre",), "seq",
         ".mlp.gate_proj.output", "seq"),
    Name("sg", None, (), None, ".mlp.act_fn.output", "seq",
         "TransformerLens has no hook on silu(g); read g and apply silu"),
    Name("u", "blocks.{l}.mlp.in.hook_out", ("blocks.{l}.mlp.hook_pre_linear",), "seq",
         ".mlp.up_proj.output", "seq"),
    Name("f", "blocks.{l}.mlp.hook_post", ("blocks.{l}.mlp.out.hook_in",), "seq",
         ".mlp.down_proj.input", "seq"),
    Name("m", "blocks.{l}.hook_mlp_out", ("blocks.{l}.mlp.hook_out", "blocks.{l}.mlp.out.hook_out"),
         "seq", ".mlp.down_proj.output", "seq"),
    Name("h_out", "blocks.{l}.hook_resid_post", ("blocks.{l}.hook_out",), "seq", ".output", "seq"),
)
GLOBAL_NAMES = {
    "emb": ("hook_embed", ("embed.hook_out",), "model.model.embed_tokens.output"),
    "fn": ("ln_final.hook_out", ("unembed.hook_in",), "model.model.norm.output"),
    "logits": ("unembed.hook_out", ("hook_unembed",), "model.lm_head.output"),
}
BY_PORT = {name.port: name for name in NAMES}
_LAYER = re.compile(r"^L(?P<l>\d+)\.(?P<port>\w+)$")


class NoName(KeyError):
    """The address or tool name has no exact counterpart (the reason is the message)."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "no counterpart"


def _split(address: str) -> tuple[int | None, str]:
    match = _LAYER.fullmatch(address.strip())
    if match:
        return int(match["l"]), match["port"]
    if address.strip() in GLOBAL_NAMES:
        return None, address.strip()
    raise NoName(f"{address!r} is not L<l>.<port> or one of {', '.join(GLOBAL_NAMES)}")


def to_tl(address: str) -> str:
    """``L22.A`` -> ``blocks.22.attn.hook_z`` (bridge-native name)."""
    layer, port = _split(address)
    if layer is None:
        return GLOBAL_NAMES[port][0]
    name = BY_PORT.get(port)
    if name is None:
        raise NoName(f"mdb has no port {port!r}")
    if name.tl is None:
        raise NoName(f"{address}: {name.note}")
    return name.tl.format(l=layer)


def to_nnsight(address: str) -> str:
    """``L22.A`` -> ``model.model.layers[22].self_attn.o_proj.input``."""
    layer, port = _split(address)
    if layer is None:
        return GLOBAL_NAMES[port][2]
    name = BY_PORT.get(port)
    if name is None or name.nn is None:
        raise NoName(f"{address}: " + (name.note if name else f"mdb has no port {port!r}"))
    return f"model.model.layers[{layer}]{name.nn}"


def _reverse_tl() -> list[tuple[re.Pattern[str], str]]:
    out = []
    for name in NAMES:
        for hook in ((name.tl,) if name.tl else ()) + name.tl_aliases:
            out.append((re.compile("^" + re.escape(hook).replace(r"\{l\}", r"(\d+)") + "$"),
                        name.port))
    return out


_TL_REVERSE = _reverse_tl()


def from_tl(hook: str) -> str:
    """``blocks.22.attn.hook_z`` (or an alias) -> ``L22.A``."""
    for glob, (native, aliases, _) in GLOBAL_NAMES.items():
        if hook == native or hook in aliases:
            return glob
    for pattern, port in _TL_REVERSE:
        match = pattern.fullmatch(hook)
        if match:
            return f"L{match.group(1)}.{port}"
    raise NoName(f"TransformerLens hook {hook!r} has no mdb port (e.g. ln*.hook_scale, "
                 "ln*.hook_normalized, hook_cos/hook_sin, *.hook_in of q/k/v projections)")


_NN = re.compile(r"^(?:model\.)?model\.layers\[(\d+)\](.*)$")


def from_nnsight(path: str) -> str:
    """``model.model.layers[22].self_attn.o_proj.input`` -> ``L22.A``."""
    for glob, (_, _, nn) in GLOBAL_NAMES.items():
        if path in (nn, nn.removeprefix("model.")):
            return glob
    match = _NN.fullmatch(path.strip())
    if match:
        for name in NAMES:
            if name.nn == match.group(2):
                return f"L{match.group(1)}.{name.port}"
    raise NoName(f"nnsight path {path!r} has no mdb port")


def table(ports: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """The name table as rows (for docs and the REPL)."""
    return [{"port": n.port, "transformer_lens": n.tl, "aliases": list(n.tl_aliases),
             "nnsight": None if n.nn is None else f"model.model.layers[l]{n.nn}",
             "note": n.note} for n in NAMES if ports is None or n.port in ports]


# --- layouts ---------------------------------------------------------------------------------


def to_mdb(layout: str, value: Any, row: int, *, heads: int | None = None) -> Any:
    """One row ``row`` of a tool's full-sequence tensor, in mdb's layout for that port."""
    if layout == "seq":
        return value[:, row:row + 1]
    if layout == "heads_flat":
        piece = value[:, row:row + 1]
        return piece.reshape(*piece.shape[:2], -1)
    if layout == "heads":
        return value[:, row:row + 1]
    if layout == "heads_t":    # TL q_norm/k_norm hooks: (batch, head, pos, head_dim)
        return value[:, :, row:row + 1].transpose(1, 2)
    if layout == "rot_q":
        return value[:, row:row + 1].transpose(1, 2)
    if layout == "rot_k":
        return value[:, :row + 1].transpose(1, 2)
    if layout == "scores":
        return value[:, :, row:row + 1, :row + 1]
    if layout == "bh":         # nnsight eager argument (batch, head, pos, head_dim)
        return value[:, :, row:row + 1]
    if layout == "bh_k":
        return value[:, :, :row + 1]
    raise ValueError(f"unknown layout {layout!r}")


def to_tool(layout: str, value: Any, *, head_dim: int) -> Any:
    """A full-sequence mdb ``trace`` value (``(batch, pos, ...)``, as ``mdb.trace`` keeps the
    native layout) in the TransformerLens layout of the mapped hook."""
    if layout in ("seq", "heads", "scores"):
        return value
    if layout == "heads_flat":
        return value.reshape(*value.shape[:2], -1, head_dim)
    if layout in ("rot_q", "rot_k", "heads_t"):   # (batch, pos, head, d) <-> (batch, head, pos, d)
        return value.transpose(1, 2)
    raise ValueError(f"unknown layout {layout!r}")


# --- foreign names of cells, rows and reads: ``mdb.core.foreign_names`` (re-exported here;
# analysis attaches them to symbol-table rows and may not import this layer) ---
from mdb.core.foreign_names import (  # noqa: E402,F401
    MEASURED_MAP, SUPPLIED_MAP, UNVERIFIED_MAP, binding_names, cell_names, ct_node,
    neuronpedia_url, read_names, row_names,
)


def residual_names(layer: int, port: str = "h_out", *, repo: str | None = None,
                   row: int | None = None) -> dict[str, Any]:
    """Foreign names of a residual site ``L<layer>.<port>`` (``h`` or ``h_out``): TL hook,
    nnsight path and, for a Qwen-Scope repo SAELens lists, its ``release``/``sae_id``."""
    from mdb.analysis.sae import saelens_ref

    index = "" if row is None else f"[:, {row}]"
    return {"address": f"L{layer}.{port}" if row is None else f"t{row}.L{layer}.{port}",
            "tl": {"hook": to_tl(f"L{layer}.{port}") + index, "verified": MEASURED_MAP},
            "nnsight": {"path": to_nnsight(f"L{layer}.{port}") + index,
                        "verified": MEASURED_MAP},
            "saelens": {**saelens_ref(repo, layer, port),
                        "verified": MEASURED_MAP if repo else UNVERIFIED_MAP}}


