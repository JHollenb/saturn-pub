"""Selectors and cross-run origins for the Python ``Run.set`` (gap audit 2026-11-14, P5).

The REPL's ``set ADDR[sel] = ORIGIN:ADDR[sel]`` builds one masked edit from a selector and takes
the donor value with its origin pointer (``ops.builtin.port_request``); a Python script had to
build the mask by hand and pass a bare tensor (origin lost). Here the same two pieces are engine
functions, so ``Run.set`` and ``Run.set_cell`` take them directly:

* ``Run.ref(ADDR[sel])`` reads a value of a run (a port or stop occurrence, or a cache cell
  ``cache.L<l>.k|v[kv,row,d]`` as the next read sees it) and returns a ``ValueRef``: the selected
  block plus its origin pointer ``{origin, via, address, sha256, shape, dtype, where, cut}`` (the
  ``Debugger.origin_value`` pointer shape, ``via`` = ``live``). Reading leaves the donor as it
  was.
* ``Run.set("t5.L1.v[kv1]", ref_or_value)`` resolves the selector against the port's axes
  (``breakpoints.plan``, the REPL's planner), builds ONE masked edit and records the selector text
  as ``selection`` and the ref's pointer as ``origin``. A ref must have the selected block's shape
  exactly (a patch copies a block, it does not broadcast); a number or a tensor broadcasts to the
  selection. ``Run.set("cache.L1.v[kv0,row=2]", ...)`` is a state-cell edit (``set_cell``) with
  rows/kv/d from the selector.

Address rows: ``t<row>.``, ``t@last.`` (the last prompt row) or none (the cursor's row).
Casts: a donor block is moved to this run's device and dtype in one ``.to`` (never twice).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .ports import tensor_digest


@dataclass(frozen=True)
class ValueRef:
    """A value read from a run at an address, with where it came from (``pointer``)."""

    value: torch.Tensor
    pointer: dict[str, Any]

    @property
    def address(self) -> str:
        return self.pointer["address"]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.value.shape)


def split(text: str) -> tuple[str, str | None]:
    """``(base, selector text or None)`` of ``ADDR[sel]``."""
    text = text.strip()
    if "[" not in text:
        return text, None
    base, _, rest = text.partition("[")
    if not rest.endswith("]"):
        raise ValueError(f"unbalanced selector in {text!r}")
    return base.strip(), rest[:-1].strip() or None


def _row_text(run: Any, base: str) -> str:
    """``t@last.`` -> ``t<last prompt row>.``; other text unchanged."""
    if base.startswith("t@last."):
        return f"t{len(run.tokens) - 1}." + base[len("t@last."):]
    if base.startswith("t@"):
        raise ValueError(f"{base}: the only row role a run knows is @last (prompt sets bind others)")
    return base


def _selector(text: str | None) -> Any:
    from mdb.core import address as addresses

    if not text:
        return addresses.Selector()
    from .breakpoints import parse_selector

    return parse_selector(text)


def resolve(run: Any, text: str) -> dict[str, Any]:
    """The parts of ``ADDR[sel]`` on ``run``: ``kind`` (``cell`` or ``port``), the canonical
    ``address`` text, the ``selector`` (parsed), ``selection`` (its text) and, for a port,
    ``key``/``label``/``row``; for a cell ``layer``/``tensor``."""
    base, sel = split(text)
    base = _row_text(run, base)
    selector = _selector(sel)
    if base.startswith("cache."):
        parts = base.split(".")
        if len(parts) != 3 or not parts[1].startswith("L") or parts[2] not in ("k", "v"):
            raise ValueError(f"a cache cell is cache.L<layer>.k|v[...], not {base!r}")
        layer = int(parts[1][1:])
        return {"kind": "cell", "layer": layer, "tensor": parts[2], "selector": selector,
                "selection": sel, "address": base + (f"[{sel}]" if sel else "")}
    key = run.parse(base)
    label = run.stops()[key[1]]
    canonical = f"t{key[0]}.{label}" + (f"[{sel}]" if sel else "")
    return {"kind": "port", "key": key, "label": label, "row": key[0], "selector": selector,
            "selection": sel, "address": canonical}


def _axes(run: Any, label: str) -> tuple[str, ...]:
    from .breakpoints import stop_axes

    return stop_axes(run.program, label)


def _picks(axes, shape, selector, run, row):
    from .breakpoints import SelectionError, plan

    try:
        return plan(axes, list(shape), selector, head_dim=run.adapter.head_dim, row=row)
    except SelectionError as error:
        raise ValueError(str(error)) from None


def _block(value: torch.Tensor, picks) -> torch.Tensor:
    out = value
    for axis, pick in enumerate(picks):
        if pick is not None:
            out = out.index_select(axis, torch.tensor(list(pick), dtype=torch.long,
                                                      device=out.device))
    return out


def run_label(run: Any) -> str:
    """The origin name of a run: its debugger name, else ``run@<root cut[:12]>``."""
    if getattr(run, "name", None):
        return run.name
    return f"run@{run.cut_trail()[0][0][:12]}"


def ref(run: Any, text: str) -> ValueRef:
    """``Run.ref``: the selected block of ``ADDR[sel]`` read from ``run`` and its pointer."""
    found = resolve(run, text)
    if found["kind"] == "cell":
        value = run.read_cache(found["layer"], found["tensor"])
        from .edits import CELL_AXES

        picks = _picks(CELL_AXES, value.shape, found["selector"], run, None)
    else:
        value = run.read(found["address"].partition("[")[0])
        axes = _axes(run, found["label"])
        row = found["row"] if "row" in axes and found["label"] not in ("start",) else None
        picks = _picks(axes, value.shape, found["selector"], run, row)
    block = _block(value, picks).detach().clone()
    pointer = {"origin": run_label(run), "via": "live", "address": found["address"],
               "sha256": tensor_digest(block), "shape": list(block.shape),
               "dtype": str(block.dtype).removeprefix("torch."), "where": run.where,
               "cut": run.checkpoint(pin=False)["cut"]}
    return ValueRef(block, pointer)


def _cast(run: Any, value: torch.Tensor) -> torch.Tensor:
    """One move to this run's device and dtype together (a no-op when both already match)."""
    device, dtype = run.adapter.device, run.adapter.dtype
    if value.dtype != dtype or value.device != torch.device(device):
        value = value.to(device=device, dtype=dtype)
    return value


def port_edit(run: Any, found: dict[str, Any], value: Any, origin: dict[str, Any] | None,
              mask: Any, *, shape: tuple[int, ...] | None = None) -> dict[str, Any]:
    """``{value, mask, selection, origin}`` of a port edit from a resolved address (``resolve``)
    and a value (number, tensor or ``ValueRef``). ``shape``: the port's shape in the address's
    own row (default: the pending transition's, ``Run.port_shape``)."""
    if isinstance(value, ValueRef):
        origin = origin if origin is not None else dict(value.pointer)
        is_ref, value = True, value.value
    else:
        is_ref = False
    selector = found["selector"]
    if not (selector.keys or selector.tokens):
        if is_ref:
            value = _cast(run, value)
        return {"value": value, "mask": mask, "selection": None, "origin": origin}
    if mask is not None:
        raise ValueError("give a selector or a mask, not both")
    port = found["label"].split(".", 1)[1]
    shape = tuple(run.port_shape(port)) if shape is None else tuple(shape)
    axes = _axes(run, found["label"])
    picks = _picks(axes, shape, selector, run, found["row"] if "row" in axes else None)
    index = tuple(
        torch.tensor(list(range(shape[a]) if pick is None else pick), dtype=torch.long).view(
            [-1 if b == a else 1 for b in range(len(shape))])
        for a, pick in enumerate(picks))
    chosen = tuple(shape[a] if pick is None else len(pick) for a, pick in enumerate(picks))
    full_mask = torch.zeros(shape, dtype=torch.bool)
    full_mask[index] = True
    if isinstance(value, torch.Tensor):
        if is_ref and tuple(value.shape) != chosen:
            raise ValueError(f"{origin['address']} selects shape {list(value.shape)}; "
                             f"{found['address']} selects {list(chosen)}: select the same block "
                             "on both sides")
        if torch.broadcast_shapes(tuple(value.shape), chosen) != chosen:
            raise ValueError(f"value of shape {list(value.shape)} does not broadcast to the "
                             f"selection {list(chosen)}")
        value = _cast(run, value)
        full = torch.zeros(shape, dtype=run.adapter.dtype, device=value.device)
        full[tuple(i.to(value.device) for i in index)] = value.expand(chosen)
        value = full
    return {"value": value, "mask": full_mask, "selection": found["selection"], "origin": origin}


def cell_edit(run: Any, found: dict[str, Any], value: Any,
              origin: dict[str, Any] | None) -> dict[str, Any]:
    """``set_cell`` keyword arguments of a cache-cell address (``resolve``) and a value."""
    keys = dict(found["selector"].keys)
    if "row" not in keys:
        raise ValueError(f"{found['address']}: a cache cell edit names its rows (row=...)")
    from .breakpoints import _indices

    def picks(key: str) -> tuple[int, ...] | None:
        return None if key not in keys else tuple(_indices(keys[key], None, key))

    if isinstance(value, ValueRef):
        origin = origin if origin is not None else dict(value.pointer)
        value = _cast(run, value.value)
    return {"layer": found["layer"], "tensor": found["tensor"], "rows": picks("row"),
            "kv": picks("kv"), "d": picks("d"), "value": value, "origin": origin,
            "selection": found["selection"]}


__all__ = ["ValueRef", "cell_edit", "port_edit", "ref", "resolve", "run_label", "split"]
