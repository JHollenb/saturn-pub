"""nnsight <-> mdb: read ports by mdb address, and batched interventions written as mdb edits.

mdb steps one sequence; nnsight batches. ``batch_whatif`` takes edits in mdb's address language
and measures them over many prompts in a few batched nnsight traces:

- a port edit ``L22.A[h6]`` (``h``/``kv``/``i`` selectors) applies at each prompt's LAST row, as
  mdb's ``whatif`` does at the pending row;
- a state-cell edit ``cache.L22.v[kv1,row=11]`` has two scopes. ``reader`` is mdb's meaning: only
  the last row reads the changed cell (the native layer runs with the patched value row, then
  every earlier row's ``o_proj`` input is put back to its clean value, so earlier rows are
  bitwise clean). ``sequence`` is ordinary activation patching: every later row of that layer
  reads the patched value too, and the change spreads through their residuals.

Values from a donor prompt (``value="donor"``) are taken from a batched donor pass. Prompts are
grouped by (recipient length, donor length) so no batch is padded. Every number returned is
``measured`` by native kernels in a batched program: it is not mdb's batch-1 stepped program,
so mdb re-measures a sample (``mdb.interop.verify``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mdb.core import address as addresses

from . import names

EDIT_OPS = ("replace", "add", "scale")
PORT_ORDER = tuple(n.port for n in names.NAMES)
NN_EDITABLE = ("h", "a", "q", "k", "v", "A", "o", "r", "n", "g", "sg", "u", "f", "m", "h_out")


def version() -> str:
    from importlib.metadata import version as _v

    return _v("nnsight")


def wrap(model: Any) -> Any:
    from nnsight import NNsight

    return NNsight(model)


_STEP = re.compile(r"\.(\w+)|\[(\d+)\]")


def envoy(nm: Any, address: str) -> Any:
    """The nnsight proxy for ``L<l>.<port>`` (or ``emb``/``fn``/``logits``) on ``nm`` (an
    ``NNsight`` wrapper of the Hugging Face causal LM). Use inside ``nm.trace(...)``."""
    path = names.to_nnsight(address).removeprefix("model")
    node = nm
    for attr, index in _STEP.findall(path):
        node = getattr(node, attr) if attr else node[int(index)]
    return node


def read(nm: Any, tokens: Sequence[Sequence[int]], addresses_: Sequence[str],
         rows: Sequence[int] | None = None) -> dict[str, Any]:
    """Save ``addresses_`` (``L<l>.<port>``) over a batch of equal-length prompts in one trace;
    returns ``{t<r>.L<l>.<port>: tensor (batch, ...) in mdb layout}`` for each row."""
    import torch

    ids = torch.tensor([list(t) for t in tokens], device=_device(nm))
    eager = [a for a in addresses_ if names._split(a)[1] in EAGER_ARGS]
    plain = sorted((a for a in addresses_ if a not in eager), key=_read_key)
    saved = {}
    with torch.no_grad(), nm.trace(ids):
        for address in plain:
            saved[address] = envoy(nm, address).save()
    # the eager-attention arguments come through ``.source``, which must be requested before
    # the attention forward starts; one more trace per layer keeps the requests in order
    for layer in sorted({names._split(a)[0] for a in eager}):
        here = sorted((a for a in eager if names._split(a)[0] == layer), key=_execution_key)
        with torch.no_grad(), nm.trace(ids):
            for address in here:
                saved[address] = envoy(nm, address).save()
    out = {}
    rows = range(ids.shape[1]) if rows is None else rows
    for address, value in saved.items():
        layer, port = names._split(address)
        spec = names.BY_PORT[port]
        for row in rows:
            out[f"t{row}.{address}"] = names.to_mdb(spec.nn_layout, value, row).detach()
    return out


# --- batched what-if ----------------------------------------------------------------------


@dataclass(frozen=True)
class Edit:
    """One edit in mdb's language. ``value``: a number; ``"donor"`` (the donor prompt's value
    at the same port, last row; for a cell, at ``donor_row`` or the same row); or ``"self"``
    (a cell only: the recipient's own clean value at ``donor_row``, the disruption control)."""

    address: str
    value: Any = 0.0
    op: str = "replace"
    scope: str = "reader"
    donor_row: int | None = None

    def __post_init__(self):
        if self.op not in EDIT_OPS:
            raise ValueError(f"op must be one of {EDIT_OPS}")
        if self.scope not in ("reader", "sequence"):
            raise ValueError("scope is 'reader' (mdb: only the last row reads the cell) or "
                             "'sequence' (activation patching)")
        if not (self.value in ("donor", "self") or isinstance(self.value, (int, float))):
            raise ValueError("value is a number, 'donor' or 'self'")
        if self.value == "self" and (self.donor_row is None or not
                                     self.address.startswith("cache.")):
            raise ValueError("'self' takes a cell address and a donor_row")


EAGER_ARGS = ("qr", "kr")
# nnsight serves requests in execution order: the attention module's own output (``p``) is
# returned after o_proj has run, so ``p`` reads after ``A`` and ``o``.
NN_READ_ORDER = ("h", "a", "q", "qn", "k", "kn", "v", "A", "o", "p", "r", "n", "g", "sg", "u",
                 "f", "m", "h_out")


def _read_key(address: str) -> tuple[int, int]:
    layer, port = names._split(address)
    if layer is None:
        return ({"emb": -1}.get(port, 10**6), 0)
    return (layer, NN_READ_ORDER.index(port))


def _execution_key(address: str) -> tuple[int, int]:
    layer, port = names._split(address)
    if layer is None:
        return ({"emb": -1}.get(port, 10**6), 0)
    return (layer, PORT_ORDER.index(port))


def _indices(value: Any, size: int) -> list[int]:
    if isinstance(value, int):
        return [value]
    if isinstance(value, addresses.Span):
        stop = size if value.stop is None else value.stop
        return list(range(value.start or 0, stop))
    if isinstance(value, addresses.AnyOf):
        return list(value.values)
    raise ValueError(f"selector value {value!r}")


@dataclass(frozen=True)
class _Plan:
    kind: str                 # "port" | "cell"
    layer: int
    port: str                 # port, or "k"/"v" for a cell
    columns: tuple[int, ...] | None  # flat column indices in the native (batch, pos, X) tensor
    rows: tuple[int, ...]     # cell rows (port edits: the last row)
    edit: Edit


def plan(edit: Edit, config: Any) -> _Plan:
    """Parse an mdb address into the native columns the edit touches."""
    heads = int(config.num_attention_heads)
    kv = int(config.num_key_value_heads)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    text = edit.address.strip()
    if text.startswith("cache."):
        cell = addresses.parse(text)
        keys = dict(cell.selector.keys)
        if "row" not in keys:
            raise ValueError(f"{text}: a cell edit names its rows (row=R)")
        rows = tuple(_indices(keys["row"], 10**9))
        heads_ = _indices(keys["kv"], kv) if "kv" in keys else list(range(kv))
        cols = [h * head_dim + d for h in heads_ for d in
                (_indices(keys["d"], head_dim) if "d" in keys else range(head_dim))]
        if cell.tensor == "k" and edit.value in ("donor", "self") and edit.donor_row is not None:
            if len(rows) != 1 or edit.donor_row != rows[0]:
                raise ValueError(f"{text}: a key cell from another row would carry that row's "
                                 "rotation; nnsight patches k before RoPE, so donor_row must "
                                 "equal the row")
        return _Plan("cell", cell.layer, cell.tensor, tuple(cols), rows, edit)
    occurrence = addresses.parse(text if text.startswith("t") else f"t0.{text}")
    port, layer = occurrence.port, occurrence.layer
    if port not in NN_EDITABLE:
        raise ValueError(f"{text}: nnsight edits ports {', '.join(NN_EDITABLE)}")
    keys = dict(occurrence.selector.keys)
    width = {"q": heads * head_dim, "A": heads * head_dim, "k": kv * head_dim,
             "v": kv * head_dim, "g": config.intermediate_size, "sg": config.intermediate_size,
             "u": config.intermediate_size, "f": config.intermediate_size}.get(
                 port, config.hidden_size)
    cols: list[int] | None = None
    if "h" in keys:
        if port not in ("q", "A"):
            raise ValueError(f"{text}: h selects query heads of q or A")
        cols = [h * head_dim + d for h in _indices(keys["h"], heads) for d in range(head_dim)]
    elif "kv" in keys:
        if port not in ("k", "v"):
            raise ValueError(f"{text}: kv selects key/value heads of k or v")
        cols = [h * head_dim + d for h in _indices(keys["kv"], kv) for d in range(head_dim)]
    elif "i" in keys:
        cols = _indices(keys["i"], width)
    elif keys:
        raise ValueError(f"{text}: nnsight edits take h, kv or i selectors")
    return _Plan("port", layer, port, None if cols is None else tuple(cols), (-1,), edit)


def _apply(x: Any, plan_: _Plan, value: Any, rows: Sequence[int]) -> None:
    """In place on the native tensor ``x`` (batch, pos, X): the edit at ``rows``/columns."""
    import torch

    cols = (slice(None) if plan_.columns is None
            else torch.tensor(plan_.columns, device=x.device))
    for i, row in enumerate(rows):
        current = x[:, row][:, cols]
        operand = value if not isinstance(value, torch.Tensor) else value[:, i].to(x.dtype)
        if plan_.edit.op == "replace":
            new = torch.empty_like(current).copy_(operand if isinstance(operand, torch.Tensor)
                                                  else torch.full_like(current, operand))
        elif plan_.edit.op == "add":
            new = current + operand
        else:
            new = current * operand
        block = x[:, row]
        block[:, cols] = new
        x[:, row] = block


def _value(p: _Plan, donor_vals: dict[Any, Any], device: Any) -> Any:
    """The edit operand: a float, or the donor's values ``(batch, rows, columns)``."""
    import torch

    if p.edit.value not in ("donor", "self"):
        return float(p.edit.value)
    src = donor_vals[(p.edit.value, p.layer, p.port)]
    if p.kind == "cell":
        src_rows = [p.edit.donor_row] * len(p.rows) if p.edit.donor_row is not None else p.rows
    else:
        src_rows = (-1,)
    cols = slice(None) if p.columns is None else torch.tensor(p.columns, device=device)
    return torch.stack([src[:, r][:, cols] for r in src_rows], 1)


def batch_whatif(model: Any, recipients: Sequence[Sequence[int]],
                 consumers: Sequence[tuple[int, int]], arms: dict[str, Sequence[Edit]], *,
                 donors: Sequence[Sequence[int]] | None = None,
                 batch_size: int = 32) -> dict[str, Any]:
    """Measure each arm (a list of ``Edit``) on every recipient prompt, batched.

    ``consumers[i] = (a, b)``: the margin ``logit[a] - logit[b]`` at recipient ``i``'s last row.
    Returns ``{"base": [M_i], "arms": {name: {"edited": [...], "delta": [...]}}, "groups": n,
    "traces": n}``; every M is float64 of the native logits.
    """
    import torch

    nm = wrap(model)
    config = model.config
    plans = {name: [plan(e, config) for e in edits] for name, edits in arms.items()}
    needs_donor = any(p.edit.value == "donor" for ps in plans.values() for p in ps)
    if needs_donor and (donors is None or len(donors) != len(recipients)):
        raise ValueError("donor values need one donor prompt per recipient")
    groups: dict[tuple[int, int], list[int]] = {}
    for i, tokens in enumerate(recipients):
        key = (len(tokens), len(donors[i]) if donors is not None else 0)
        groups.setdefault(key, []).append(i)
    n = len(recipients)
    base = [None] * n
    out = {name: [None] * n for name in arms}
    traces = 0
    device = _device(nm)
    reader_layers = sorted({p.layer for ps in plans.values() for p in ps
                            if p.kind == "cell" and p.edit.scope == "reader"})
    for _key, members in groups.items():
        for start in range(0, len(members), batch_size):
            chunk = members[start:start + batch_size]
            ids = torch.tensor([list(recipients[i]) for i in chunk], device=device)
            a_ids = torch.tensor([consumers[i][0] for i in chunk], device=device)
            b_ids = torch.tensor([consumers[i][1] for i in chunk], device=device)

            def margin(logits):
                last = logits[:, -1].to(torch.float64)
                return (last.gather(1, a_ids[:, None]) - last.gather(1, b_ids[:, None]))[:, 0]

            clean_A, donor_vals = {}, {}
            own = {(p.layer, p.port) for ps in plans.values() for p in ps if p.edit.value == "self"}
            keys = sorted({(layer, "A") for layer in reader_layers} | own,
                          key=lambda lp: (lp[0], PORT_ORDER.index(lp[1])))
            with torch.no_grad(), nm.trace(ids):
                for layer, port in keys:
                    saved = envoy(nm, f"L{layer}.{port}").save()
                    if port == "A" and layer in reader_layers:
                        clean_A[layer] = saved
                    if (layer, port) in own:
                        donor_vals[("self", layer, port)] = saved
                logits = nm.output.logits.save()
            traces += 1
            for j, i in enumerate(chunk):
                base[i] = float(margin(logits)[j])
            if needs_donor:
                d_ids = torch.tensor([list(donors[i]) for i in chunk], device=device)
                wanted = sorted({(p.layer, p.port) for ps in plans.values() for p in ps
                                 if p.edit.value == "donor"},
                                key=lambda lp: (lp[0], PORT_ORDER.index(lp[1])))
                with torch.no_grad(), nm.trace(d_ids):
                    for layer, port in wanted:
                        donor_vals[("donor", layer, port)] = envoy(nm, f"L{layer}.{port}").save()
                traces += 1
            for name, ps in plans.items():
                ordered = sorted(ps, key=lambda p: (p.layer, PORT_ORDER.index(p.port)))
                with torch.no_grad(), nm.trace(ids):
                    for layer in sorted({p.layer for p in ordered}):
                        here = [p for p in ordered if p.layer == layer]
                        restore = any(p.kind == "cell" and p.edit.scope == "reader" for p in here)
                        ports = {p.port for p in here} | ({"A"} if restore else set())
                        for port in sorted(ports, key=PORT_ORDER.index):
                            x = envoy(nm, f"L{layer}.{port}")
                            if port == "A" and restore:
                                # reader scope: earlier rows read the clean cache at this layer
                                x[:, :-1] = clean_A[layer][:, :-1]
                            for p in [p for p in here if p.port == port]:
                                _apply(x, p, _value(p, donor_vals, device), p.rows)
                    edited = nm.output.logits.save()
                traces += 1
                for j, i in enumerate(chunk):
                    out[name][i] = float(margin(edited)[j])
    return {"base": base,
            "arms": {name: {"edited": vals, "delta": [e - b for e, b in zip(vals, base)]}
                     for name, vals in out.items()},
            "groups": len(groups), "traces": traces, "nnsight": version()}


def _device(nm: Any) -> Any:
    return next(nm._module.parameters()).device
