"""Holds: keep ports of a run at a reference run's values while the run continues (gap audit
2026-11-14, P4).

The attribution-graph "frozen attention and norms" linearisation, and exp4's held arms, kept the
answer row's attention weights and RMSNorm scales at the base run's values at every layer after
an edit; exp4 did it by stepping each later layer and re-setting ``p``, ``a`` and ``n`` by hand
(``mdb-experiments/results/showcase-20261114/exp4/exp4_brackets.py`` in the repository). A hold is that as a debugger switch:

``run.hold("t@last.L>1.p", source=base)`` / ``run.hold("t@last.L>1.norm_scale", source=base)``

On every covered ``(row, layer)`` the run reaches, the hold stages one ordinary ``replace`` edit
(``PortEdit``) at the held port whose ``origin`` is ``{kind: "hold", hold, origin, via, address,
sha256, where, cut, rule}``. The edit is exact and labelled: checkpoints, restore, replay,
``committed_edits`` and every record carry it like any edit, and the gap between a held and a
free branch is a measured quantity. Kinds:

* a port (``p``, ``A``, ``o``, ``m``, ``a``, ``n``, ... any editable port): the port is replaced by
  the source run's value at the same address (the source must have computed it; it is read, never
  moved). The value is cast once to this run's dtype (a no-op on the same model).
* ``attn_norm_scale`` / ``mlp_norm_scale`` / ``norm_scale`` (both): the RMSNorm SCALE is held,
  not its output: ``a = w_in * (h * rho_ref).to(dtype)`` with ``h`` this run's layer input and
  ``rho_ref = rsqrt(mean(h_ref^2) + eps)`` from the source's ``h`` (``n`` likewise from ``r``):
  the native RMSNorm's own operation order (fp32 scale, one cast, weight multiply), so holding a
  run against itself is bit-identical (tested). The input is read by a probe of the pending
  layer that stops at it (with every edit staged before it), one native layer call per held
  norm per layer.
* a cache cell ``cache.L<l>.k|v[kv,row=...]``: one state-cell replace with the source's block
  (cache rows are written once, so the held cell stays held).

The final norm is not a decoder port: ``final_held_logits(run, row, source)`` computes the
logits with the final RMSNorm scale held (``lm_head(w_F * (h_out * rho_F,ref).to(dtype))``,
derived-exact from native ``h_out``).

A hold is the last write at its port: an edit staged at a held port is overridden (stage it at
an earlier port). Holds act on layers the run commits while they are active; a restored path
replays the edits it recorded. Rows: ``t<r>.``, ``t@last.`` or ``t*.``/none (every row the run
steps). Layers: ``L5``, ``L>5``, ``L>=5``, ``L3-5`` (inclusive), ``L*``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch

from .ports import PortEdit, tensor_digest

NORMS = {"attn_norm_scale": (("a", "h", "input_layernorm"),),
         "mlp_norm_scale": (("n", "r", "post_attention_layernorm"),),
         "norm_scale": (("a", "h", "input_layernorm"), ("n", "r", "post_attention_layernorm"))}
_SPEC = re.compile(r"^(?:t(?P<row>\d+|@last|\*)\.)?L(?P<layers>\*|>=?\d+|\d+-\d+|\d+)\."
                   r"(?P<port>\w+)$")


@dataclass
class Hold:
    """One hold: ``port`` (a port name or a norm-scale kind) at ``layers`` x ``rows`` (None:
    every row) held at ``source``'s values."""

    name: str
    port: str
    layers: tuple[int, ...]
    rows: tuple[int, ...] | None
    source: Any
    text: str = ""
    cells: list[dict[str, Any]] = field(default_factory=list)

    def covers(self, row: int, layer: int) -> bool:
        return layer in self.layers and (self.rows is None or row in self.rows)

    @property
    def norm(self) -> bool:
        return self.port in NORMS

    def targets(self) -> tuple[str, ...]:
        """The ports this hold writes."""
        return tuple(p for p, _, _ in NORMS[self.port]) if self.norm else (self.port,)

    def describe(self) -> dict[str, Any]:
        from .refs import run_label

        return {"name": self.name, "address": self.text, "port": self.port,
                "layers": list(self.layers), "rows": None if self.rows is None else list(self.rows),
                "source": run_label(self.source), "kind": "norm-scale" if self.norm else "value",
                "writes": list(self.targets())}


def _layers(text: str, count: int) -> tuple[int, ...]:
    if text == "*":
        return tuple(range(count))
    if text.startswith(">="):
        return tuple(range(int(text[2:]), count))
    if text.startswith(">"):
        return tuple(range(int(text[1:]) + 1, count))
    if "-" in text:
        low, high = (int(x) for x in text.split("-"))
        if high < low:
            raise ValueError(f"layer range L{text} is empty")
        return tuple(range(low, high + 1))
    return (int(text),)


def parse(run: Any, text: str) -> tuple[str, tuple[int, ...], tuple[int, ...] | None]:
    """``(port, layers, rows)`` of a hold address (module docstring)."""
    match = _SPEC.match(text.strip())
    if match is None:
        raise ValueError(f"cannot parse hold {text!r}: write [t<row>|t@last|t*.]L<layers>.<port> "
                         "(layers 5, >5, >=5, 3-5, *)")
    port = match["port"]
    if port not in NORMS:
        spec = run.program.spec(port)
        if not spec.editable:
            raise ValueError(f"port {port} cannot be held: {spec.edit}")
        port = spec.name
    layers = _layers(match["layers"], run.adapter.layers)
    bad = [x for x in layers if not 0 <= x < run.adapter.layers]
    if bad or not layers:
        raise ValueError(f"hold {text!r}: layers outside 0..{run.adapter.layers - 1}")
    row = match["row"]
    rows = (None if row in (None, "*") else (len(run.tokens) - 1,) if row == "@last"
            else (int(row),))
    return port, layers, rows


def _module(run: Any, layer: int, name: str) -> Any:
    adapter = run.adapter
    return getattr(getattr(adapter.backbone, adapter.spec.layers)[layer], name)


def rms_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    """``rsqrt(mean(x^2) + eps)`` over the last axis in float32 (the native RMSNorm scale)."""
    x = x.to(torch.float32)
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def held_norm(module: Any, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The RMSNorm output with its scale held: ``w * (x * scale).to(dtype)`` (the native
    operation order with ``rsqrt(var + eps)`` replaced by ``scale``)."""
    return module.weight * (x.to(torch.float32) * scale).to(x.dtype)


def _pointer(hold: Hold, address: str, value: torch.Tensor, rule: str) -> dict[str, Any]:
    from .refs import run_label

    source = hold.source
    return {"kind": "hold", "hold": hold.name, "origin": run_label(source), "via": "live",
            "address": address, "sha256": tensor_digest(value), "where": source.where,
            "cut": source.checkpoint(pin=False)["cut"], "rule": rule}


def _source_read(hold: Hold, address: str) -> torch.Tensor:
    try:
        return hold.source.read(address)
    except ValueError as error:
        raise ValueError(f"hold {hold.name}: the source has not computed {address} ({error}); "
                         f"run it there first") from None


def edits_for(run: Any, row: int, layer: int, base: list[PortEdit],
              probe_input: Any) -> list[PortEdit]:
    """The hold edits of ``(row, layer)`` in program order. ``base`` are the user's staged
    edits; ``probe_input(port, edits)`` returns this run's value at ``port`` of the pending layer
    with ``edits`` staged (a probe that stops there)."""
    names = run.program.names
    wanted: list[tuple[int, Hold, str, Any]] = []
    for hold in run._holds:
        if hold.cells or not hold.covers(row, layer):
            continue
        if hold.norm:
            for port, source_port, module in NORMS[hold.port]:
                wanted.append((names.index(port), hold, port, (source_port, module)))
        else:
            wanted.append((names.index(hold.port), hold, hold.port, None))
    wanted.sort(key=lambda item: item[0])
    out: list[PortEdit] = []
    for _, hold, port, norm in wanted:
        address = f"t{row}.L{layer}.{port}"
        if norm is None:
            value = _source_read(hold, address)
            if value.dtype != run.adapter.dtype or value.device != torch.device(run.adapter.device):
                value = value.to(device=run.adapter.device, dtype=run.adapter.dtype)
            rule = "replace with the source's value at the same address"
        else:
            source_port, module_name = norm
            module = _module(run, layer, module_name)
            eps = float(module.variance_epsilon)
            reference = _source_read(hold, f"t{row}.L{layer}.{source_port}")
            x = probe_input(source_port, [*base, *out])
            value = held_norm(module, x, rms_scale(reference, eps))
            rule = (f"{module_name} scale held: w * ({source_port} * rsqrt(mean("
                    f"{source_port}_source^2) + eps)).to(dtype)")
        out.append(PortEdit(port, value, "replace", selection=None,
                            origin=_pointer(hold, f"t{row}.L{layer}.{port}", value, rule)))
    return out


def source_addresses(run: Any, hold: Hold, from_row: int) -> list[str]:
    """The source addresses ``hold`` reads on the rows from ``from_row`` on that its source has
    computed (a checkpoint keeps them, ``Snapshot``): the held port, or the norm's input."""
    if hold.cells:
        return []
    last = max(len(hold.source.tokens) - 1, getattr(hold.source._trail[-1], "row", 0))
    rows = range(from_row, last + 1) if hold.rows is None else [
        r for r in hold.rows if r >= from_row]
    ports = ([source for _, source, _ in NORMS[hold.port]] if hold.norm else [hold.port])
    return [f"t{r}.L{layer}.{port}" for r in rows for layer in hold.layers for port in ports]


class Snapshot:
    """A hold's source as a checkpoint keeps it: its values at the addresses the hold reads
    from the checkpoint's row on (as the source held them when the checkpoint was saved), its
    name, cursor and cut (the hold edits' origin pointers name them, as on the source run). An
    address it does not hold is one the source had not computed then: refused as a live source
    refuses it."""

    def __init__(self, name: str, where: str, cut: str, values: dict[str, torch.Tensor],
                 tokens: list[int]):
        from types import SimpleNamespace

        self.name, self.where, self._cut, self.values = name, where, cut, values
        self.tokens = list(tokens)
        rows = [int(a[1:a.index(".")]) for a in values] or [len(self.tokens) - 1]
        self._trail = (SimpleNamespace(fingerprint=cut, row=max(rows)),)   # what Run reads

    def read(self, address: str) -> torch.Tensor:
        found = self.values.get(address)
        if found is None:
            raise ValueError(f"{address} was not computed by the source when the checkpoint "
                             "was saved")
        return found

    def checkpoint(self, *, pin: bool = False) -> dict[str, Any]:
        return {"cut": self._cut}


def is_hold(edit: Any) -> bool:
    origin = getattr(edit, "origin", None)
    return isinstance(origin, dict) and origin.get("kind") == "hold"


def final_held_logits(run: Any, row: int, source: Any) -> torch.Tensor:
    """Logits of ``row`` with the final RMSNorm scale held at ``source``'s (derived-exact from
    this run's native ``h_out`` of the last layer): ``lm_head(w_F * (h * rho_F,source))``."""
    adapter = run.adapter
    last = adapter.layers - 1
    norm = getattr(adapter.backbone, adapter.spec.final_norm)
    h = run.read(f"t{row}.L{last}.h_out")
    reference = source.read(f"t{row}.L{last}.h_out")
    held = held_norm(norm, h, rms_scale(reference, float(norm.variance_epsilon)))
    with torch.no_grad():
        return adapter.model.lm_head(held)


__all__ = ["Hold", "NORMS", "Snapshot", "edits_for", "source_addresses", "final_held_logits", "held_norm", "is_hold", "parse",
           "rms_scale"]
