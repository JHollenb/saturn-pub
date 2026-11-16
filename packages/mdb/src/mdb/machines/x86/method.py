"""Symbol-table recovery on a binary by intervention (the run-1 method of the 2026-11-16 PREREG).

Frozen specification (PREREG, run 1), and how each clause is implemented here:

1. **Cells from write events.** The base run's cells (``trace.Trace``): every byte (flag bit) a tick
   writes starts a cell that lasts until the site is rewritten; the injected input and sites read
   before any write are initial cells. Undefined flags start no cell.
2. **Liveness by aligned-resample interchange over the 32 donors.** For every cell and donor, the
   donor value is the donor run's cell with the same alignment key (execution index of the defining
   tick + write slot; ``init`` + site for initial cells). Each donor whose value differs from the
   base's is one lane: the base input runs, the cell is overwritten with the donor value right
   after its defining tick, and the program runs to its end. The cell is **live** if at least one
   lane changes y (the last ``write``'s bytes or the exit code; a fault or the step cap counts as a
   change). A donor that lacks the aligned cell, or holds the same value, is not run (no evidence);
   every cell reports how many donors were informative.
3. **Merge by identical transfer.** Live cells a, b merge when, on every donor lane of a in which
   b's aligned instance exists, b takes exactly a's patched value, b's base value equals a's, and
   b exists in at least one such lane. Merging is transitive (union-find).
4. **Slots keyed by (pc, operand).** A binding's slot is the set of (defining pc, written operand)
   of its cells; initial cells key by (``init``, site).

Output: ``Recovery`` (bindings over live cells, dead cells, per-cell arrays) and records
``mdb-symbol-table-v1`` (one page per 16 bindings) and ``mdb-circuit-map-v1``. Circuit edges come
from the same patch lanes: binding A -> B when patching a cell of A changes a cell of B; each lane's
change is typed ``ctrl`` (the lane's pc path had diverged before B's cell was defined), ``addr`` (an
effective address had), or ``data``. These are total effects, so the edges are transitive.

Baselines (PREREG run 1) on the same cell universe: ``b1`` site-as-symbol (one label per register,
EFLAGS, or 8-byte memory slot, over the whole run) and ``b2`` value-equality clustering (cells with
equal values on the base and every donor input).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from . import index as xi
from .binary import FLAG0, NSLOT_LANE, Binary, site_name
from .lanes import Lanes, reference
from .trace import INIT, Trace, record

SPEC = ("cells from write events; liveness by aligned-resample interchange (live if >= 1 of the donors "
        "changes y); merge when patching one cell transfers identically to the other on every donor "
        "where the other's aligned instance exists; slots keyed by (pc, operand)")


@dataclass
class Recovery:
    binary: Binary
    base: Trace
    donors: list
    live: np.ndarray                  # bool [N]
    informative: np.ndarray           # int [N]: donors whose aligned value differed (lanes run)
    effect_donors: np.ndarray         # int [N]: lanes that changed y
    binding: np.ndarray               # int [N]: binding id, -1 dead
    edges: dict = field(default_factory=dict)       # (a_binding, b_binding|-2 for y) -> {data, addr, ctrl}
    lanes_run: int = 0
    result: object = None             # the recorded lanes.Result of [base, *donors]
    b1: np.ndarray | None = None
    b2: np.ndarray | None = None

    @property
    def n_bindings(self) -> int:
        return int(self.binding.max()) + 1 if len(self.binding) else 0

    def members(self, k: int) -> list:
        return [self.base.cells[i] for i in np.nonzero(self.binding == k)[0]]

    def slots(self, k: int) -> list[tuple]:
        out = []
        for c in self.members(k):
            key = ("init", site_name(c.site)) if c.def_ == INIT else (f"{c.pc:#x}", self.binary.operand_text(c.pc, c.slot))
            if key not in out:
                out.append(key)
        return out


def resident_pages(binary: Binary, traces: list) -> list[int]:
    """Writable pages any run touched, plus one neighbour on each side (patched pointers)."""
    pages = set()
    for tr in traces:
        for _t, a, _o, _n in tr.mem_writes:
            pages.add(a & ~0xFFF)
    if binary.input_addr is not None:
        pages.add(binary.input_addr & ~0xFFF)
    out = set()
    for p in pages:
        out |= {p - 4096, p, p + 4096}
    return sorted(p for p in out if p in binary.perms and "w" in binary.perms[p])


def aligned_values(base: Trace, other: Trace) -> np.ndarray:
    """Per base cell: the value of the cell with the same alignment key in ``other`` (-1 absent)."""
    out = np.full(len(base.cells), -1, np.int64)
    for i, c in enumerate(base.cells):
        j = other.by_key.get(c.key)
        if j is not None:
            out[i] = other.cells[j].value
    return out


def lane_alignment(base: Trace, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For B lanes' execution keys [B, T]: per base (non-initial) cell, the lane tick holding the same
    execution instance (tick [B, N], found [B, N])."""
    dev = keys.device
    q = torch.tensor([c.key[0] if c.def_ != INIT else -2 for c in base.cells], dtype=torch.int64, device=dev)
    sk, order = keys.sort(dim=1)
    pos = torch.searchsorted(sk, q.expand(keys.shape[0], -1).contiguous()).clamp(max=keys.shape[1] - 1)
    found = sk.gather(1, pos) == q
    return order.gather(1, pos), found


def recover(binary: Binary, base_input: bytes | None, donor_inputs: list, *, device="cpu", chunk: int = 2048,
            cap_factor: int = 4, log=print) -> Recovery:
    dev = torch.device(device)
    traces, res = record(binary, [base_input, *donor_inputs], device=device)
    base, donors = traces[0], traces[1:]
    if base.stop != "exit":
        raise ValueError(f"the base run did not exit ({base.stop})")
    resident = resident_pages(binary, traces)
    cells = base.cells
    N = len(cells)
    V = np.stack([aligned_values(base, d) for d in donors]) if donors else np.zeros((0, N), np.int64)
    bval = np.array([c.value for c in cells], np.int64)
    plan = [(i, j) for i in range(N) for j in range(len(donors)) if V[j, i] >= 0 and V[j, i] != bval[i]]
    informative = np.zeros(N, np.int64)
    for i, _j in plan:
        informative[i] += 1
    log(f"{binary.name}: T={base.T} cells={N} patch lanes={len(plan)}")

    ref = reference(binary, _take(res, 0), dev)
    cap = cap_factor * base.T + 64
    base_y = (base.y, base.exit_code)
    is_init = torch.tensor([c.def_ == INIT for c in cells], device=dev)
    slot = torch.tensor([max(c.slot, 0) for c in cells], dtype=torch.int64, device=dev)
    lane_slot_ok = torch.tensor([0 <= c.slot < NSLOT_LANE for c in cells], device=dev)
    defs = torch.tensor([c.def_ for c in cells], dtype=torch.int64, device=dev)
    bv = torch.tensor(bval, device=dev)
    patched = sorted({i for i, _ in plan})
    row_of = {a: r for r, a in enumerate(patched)}
    A = len(patched)
    ident = torch.ones(A, N, dtype=torch.uint8, device=dev)
    seen = torch.zeros(A, N, dtype=torch.uint8, device=dev)
    changed = {e: torch.zeros(A, N, dtype=torch.uint8, device=dev) for e in ("data", "addr", "ctrl")}
    ychanged = {e: torch.zeros(A, dtype=torch.uint8, device=dev) for e in ("data", "addr", "ctrl")}
    effect = np.zeros(N, np.int64)
    for s in range(0, len(plan), chunk):
        part = plan[s:s + chunk]
        B = len(part)
        ai = torch.tensor([i for i, _ in part], dtype=torch.int64, device=dev)
        pv = torch.tensor([int(V[j, i]) for i, j in part], dtype=torch.int64, device=dev)
        lanes = Lanes(binary, [base_input] * B, device=dev, cap=cap, resident=resident)
        lanes.patch(defs[ai] + 1, torch.tensor([cells[i].site for i, _ in part], device=dev), pv)
        lanes.compare_to(ref, torch.zeros(B, dtype=torch.int64, device=dev))
        r = lanes.run(record="slots")
        yc = torch.tensor([(y, int(e)) != base_y or int(st) != 1 for y, e, st in
                           zip(r.y, r.exit_code.tolist(), r.stop.tolist())], device=dev)
        for (i, _j), c in zip(part, yc.tolist()):
            effect[i] += int(c)
        keys = xi.keys(binary, r.pc, r.pc >= 0)
        tick, found = lane_alignment(base, keys)
        val = r.slots.reshape(B, -1).gather(1, tick * NSLOT_LANE + slot[None, :]).long()
        present = found & lane_slot_ok & (val >= 0)
        # initial cells: unchanged except the patched one itself
        val = torch.where(is_init, bv, val)
        present = present | is_init
        own = torch.arange(N, device=dev) == ai[:, None]
        val = torch.where(own, pv[:, None], val)
        ok = (~present | (val == pv[:, None])).to(torch.uint8)
        rows = torch.tensor([row_of[i] for i, _ in part], device=dev)
        ident.index_reduce_(0, rows, ok, "amin")
        seen.index_reduce_(0, rows, present.to(torch.uint8), "amax")
        diff = present & (val != bv) & ~own
        ctrl_before = (r.ctrl_at[:, None] >= 0) & (r.ctrl_at[:, None] < defs)
        addr_before = (r.addr_at[:, None] >= 0) & (r.addr_at[:, None] < defs) & ~ctrl_before
        for e, m in (("ctrl", diff & ctrl_before), ("addr", diff & addr_before),
                     ("data", diff & ~ctrl_before & ~addr_before)):
            changed[e].index_reduce_(0, rows, m.to(torch.uint8), "amax")
        ctl_y, adr_y = r.ctrl_at >= 0, (r.addr_at >= 0) & (r.ctrl_at < 0)
        for e, m in (("ctrl", yc & ctl_y), ("addr", yc & adr_y), ("data", yc & ~ctl_y & ~adr_y)):
            ychanged[e].index_reduce_(0, rows, m.to(torch.uint8), "amax")
        log(f"  lanes {s + B}/{len(plan)}")
    live = effect > 0
    # merge by identical transfer
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    live_t = torch.tensor(live, device=dev)
    if A:
        pr = torch.tensor(patched, device=dev)
        merge = (ident.bool() & seen.bool() & live_t[None, :] & live_t[pr][:, None]
                 & (bv[None, :] == bv[pr][:, None]))
        merge[torch.arange(A, device=dev), pr] = False
        for ra, b_ in torch.nonzero(merge).tolist():
            x, y = find(patched[ra]), find(b_)
            if x != y:
                parent[max(x, y)] = min(x, y)
    roots = sorted({find(i) for i in range(N) if live[i]}, key=lambda i: (cells[i].def_, i))
    bid = {r_: k for k, r_ in enumerate(roots)}
    binding = np.array([bid[find(i)] if live[i] else -1 for i in range(N)], np.int64)
    # typed edges between bindings, and to the consumer
    edges: dict = {}
    if A:
        for e in ("data", "addr", "ctrl"):
            for ra, b_ in torch.nonzero(changed[e]).tolist():
                ka, kb = binding[patched[ra]], binding[b_]
                if ka >= 0 and kb >= 0 and ka != kb:
                    edges.setdefault((int(ka), int(kb)), {"data": 0, "addr": 0, "ctrl": 0})[e] += 1
            for ra in torch.nonzero(ychanged[e])[:, 0].tolist():
                ka = binding[patched[ra]]
                if ka >= 0:
                    edges.setdefault((int(ka), -2), {"data": 0, "addr": 0, "ctrl": 0})[e] += 1
    rec = Recovery(binary, base, donors, live, informative, effect, binding, edges, len(plan), res)
    rec.b1, rec.b2 = baselines(base, V)
    return rec


def _take(res, lane: int):
    """One lane of a fully recorded ``Result`` as a one-lane Result."""
    from .lanes import Result  # noqa: PLC0415

    sl = slice(lane, lane + 1)
    out = Result(res.stop[sl], res.T[sl], res.exit_code[sl], res.y[sl])
    out.pc = res.pc[sl]
    out.full = {k: v[sl] for k, v in res.full.items()}
    return out


def baselines(base: Trace, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """B1 site-as-symbol and B2 value-equality labels per base cell (PREREG run 1 baselines)."""
    b1, b2, names, groups = [], [], {}, {}
    for i, c in enumerate(base.cells):
        if c.site < FLAG0:
            key = ("reg", c.site // 8)
        elif c.site < FLAG0 + 6:
            key = ("eflags",)
        else:
            key = ("mem", c.site & ~7)
        b1.append(names.setdefault(key, len(names)))
        vec = (c.value, *V[:, i].tolist())
        b2.append(groups.setdefault(vec, len(groups)))
    return np.array(b1, np.int64), np.array(b2, np.int64)


__all__ = ["Recovery", "recover", "baselines", "SPEC", "resident_pages", "aligned_values", "lane_alignment"]
