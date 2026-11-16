"""``path-diverged`` for transformers (binary ground truth 2026-11-16, lesson 7; request
``mdb-verdict-path-diverged``).

On the binaries, 60-85% of aligned-resample patches of branch-feeding cells changed the control
path (MEASURED, SCORES.md run 2(c)); read as data effects they would have been misread. The
transformer analogue of a branch outcome is an attention selection: the source a head weights
most. After an edit at row ``r``, layer ``l``, the downstream selections are the ``p`` rows of
every head at every row ``>= r`` from the first layer the edit reaches (``l`` for a port that
computes before ``p`` in the layer, else ``l + 1``).

``paired(run, edits, consumer)`` measures from one branch point (``Branches``, stepped):

* ``total``: the edits, run free to the consumer;
* ``frozen``: the same edits with every downstream ``p`` held at the reference's (the unedited
  continuation's) pattern, ``hold t*.L>=k.p`` (``engine.holds``): only values flow, selections
  are the base's;

and reads the argmax of every downstream ``p`` row on the base and on the free arm (Branches
probes). ``path.diverged`` when any argmax moved. A row whose top two weights are equal in either
run is a tie and is not counted (``ties`` gives how many). ``first`` names the earliest moved
selection in execution order (row, then layer, then head), ``moved`` counts moved (row, layer,
head) selections and ``heads`` the distinct (layer, head). ``verdict``: ``path-diverged`` when
diverged, else the free arm's ``effect`` / ``no effect`` against the Branches floor. The total
and frozen effects are always both reported: when a path diverged, their gap is what the moved
selections carried.

``argmax_rows`` and ``diverged`` are plain functions on tensors so batched drivers decide
divergence with the same rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .finite import MEASURED, num

RULE = ("path-diverged when the argmax over sources of any downstream p row (heads x rows >= the "
        "edit row x layers the edit reaches) differs between the base and the edited run; rows "
        "whose top two weights tie in either run are not counted")


def _torch():
    import torch

    return torch


def argmax_rows(p: Any) -> Any:
    """``p`` [..., sources] -> the argmax per row, ``-1`` where the top two weights tie."""
    torch = _torch()
    flat = p.reshape(-1, p.shape[-1]).float()
    if flat.shape[-1] < 2:
        return torch.zeros(flat.shape[0], dtype=torch.long, device=flat.device)
    top = flat.topk(2, dim=-1)
    return torch.where(top.values[:, 0] > top.values[:, 1], top.indices[:, 0], -1)


def diverged(base: Any, edited: Any, names: Sequence[str] | None = None) -> dict[str, Any]:
    """Compare two argmax tensors of the same shape (``argmax_rows`` per selection, in execution
    order along the first axis; ``names[i]`` names selection ``i``)."""
    torch = _torch()
    base, edited = torch.as_tensor(base).reshape(-1), torch.as_tensor(edited).reshape(-1)
    tie = (base < 0) | (edited < 0)
    moved = (base != edited) & ~tie
    idx = torch.nonzero(moved).flatten().tolist()
    out = {"diverged": bool(idx), "moved": len(idx), "ties": int(tie.sum()),
           "selections": int(base.numel()), "rule": RULE}
    if idx and names is not None:
        out["first"] = names[idx[0]]
    return out


def first_layer(run: Any, port: str, layer: int) -> int:
    """The first layer whose ``p`` an edit at ``L<layer>.<port>`` reaches."""
    names = run.program.names
    return layer if names.index(port) < names.index("p") else layer + 1


def _site(run: Any, address: str) -> tuple[int, int, str]:
    head, _, _ = address.partition("[")
    row, layer, port = head.split(".")
    return int(row[1:]), int(layer[1:]), port


def paired(run: Any, edits: Sequence[Any], consumer: Any, *, name: str = "paired"
           ) -> dict[str, Any]:
    """Total and frozen-selection effects of ``edits`` (``branches.Edit`` or ``(address,
    value, op, mask)``; port edits of one row) at ``consumer``, with the path report (module
    docstring). The run is left where it was."""
    from .branches import Branches, edit

    edits = [e if hasattr(e, "target") else edit(e[0], e[1], e[2], mask=e[3]) for e in edits]
    sites = [_site(run, e.target) for e in edits]
    row = min(r for r, _, _ in sites)
    start = min(first_layer(run, port, layer) for r, layer, port in sites if r == row)
    entry_layer = min(layer for r, layer, _ in sites if r == row)
    last = int(getattr(consumer, "row", row))
    rows = list(range(row, last + 1))
    layers = list(range(start, run.adapter.layers))
    names = [f"t{r}.L{l}.p[h{h}]" for r in rows for l in layers
             for h in range(run.adapter.num_heads)]

    def probe(r: Any) -> list[float]:
        torch = _torch()
        if not layers:
            return []
        found = [argmax_rows(r.read(f"t{x}.L{l}.p")) for x in rows for l in layers]
        return torch.cat(found).double().tolist()

    where = run.where
    br = Branches(run, consumer, at=f"t{row}.L{entry_layer}.h", name=name,
                  probes={"argmax": probe})
    try:
        br.add("total", edits)
        if layers:
            br.add("frozen", edits, holds=[f"t*.L>={start}.p"])
        found = {r["name"]: r for r in br.run()}
    finally:
        br.restore()
        if run.where != where:
            (run.until if run.parse(where) > run.parse(run.where) else run.rewind)(where)
    total = found["total"]
    frozen = found.get("frozen", total)
    base_ids = [int(v["value"]) for v in br.base_probes.get("argmax", [])]
    free_ids = [int(v["value"]) for v in total.get("probes", {}).get("argmax", [])]
    path = diverged(base_ids, free_ids, names) if layers else {
        "diverged": False, "moved": 0, "ties": 0, "selections": 0, "rule": RULE,
        "note": "no layer downstream of the edit computes p"}
    if path["moved"]:
        torch = _torch()
        moved = (torch.tensor(base_ids) != torch.tensor(free_ids)) & (torch.tensor(base_ids) >= 0) \
            & (torch.tensor(free_ids) >= 0)
        path["heads"] = len({names[i].split(".", 1)[1] for i in torch.nonzero(moved).flatten()
                             .tolist()})
    above = bool(total.get("above_floor", total["dM"]["value"] != 0))
    verdict = "path-diverged" if path["diverged"] else ("effect" if above else "no effect")
    return {"verdict": verdict, "path": path,
            "total": {"dM": total["dM"], "above_floor": total.get("above_floor")},
            "frozen": {"dM": frozen["dM"], "above_floor": frozen.get("above_floor"),
                       "holds": f"t*.L>={start}.p at the reference" if layers else "none"},
            "frozen_minus_total": num(frozen["dM"]["value"] - total["dM"]["value"], MEASURED),
            "floor": total.get("floor"), "consumer": br.text, "layers": [start, len(layers)],
            "rows": [row, last]}


__all__ = ["RULE", "argmax_rows", "diverged", "first_layer", "paired"]
