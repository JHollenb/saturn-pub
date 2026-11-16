"""Run-2 machinery: a patch grid {zero, unaligned resample, aligned resample} x {step index, execution
index} over cells of a base run, executed as lanes, one row per (cell, rule, alignment, donor).

The cell list is the base run's (``Trace.cells``; the PREREG asks for every live cell). For a cell c
defined at tick t on site s, the patched value is:

- ``zero``: 0. The alignment does not enter (the base run is the reference), so both alignments
  share one lane.
- ``unaligned`` resample: the value of a cell on the same site in donor run d, drawn uniformly over
  that site's cells in d (seeded by (seed, cell, donor)): the right kind of value from the wrong
  time. Both alignments share one lane.
- ``aligned`` resample, ``step`` index: the value of site s in donor d after tick t (same step
  index, same site), i.e. activation patching by position.
- ``aligned`` resample, ``exec`` index: the value of the donor cell with c's alignment key
  (execution index of the defining tick + write slot), the PREREG's donor family Q.

Each patch runs on two lanes from the base input: free, and held (branch outcomes and effective
addresses frozen to the base run, ``lanes`` hold). Per row: ``verdict`` of the free lane (``fault``
> ``nonterminating`` > ``path-diverged`` > ``effect`` / ``no effect``; ``no-donor`` when the donor
has no such cell; a donor value equal to the base value is not run: ``noop``), the total effect
(``y_changed``) and the frozen-selection effect (``frozen_effect``) as the paired column, the
divergence ticks, the consumer delta (changed y bytes, exit-code change), and the structural copy
web (``Trace.web_labels``, this backend's own) of the base cell and of the donor cell, so type
coherence can be read as base_web == donor_web. No scoring happens here.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from .binary import site_name
from .lanes import Lanes, reference
from .method import _take, resident_pages
from .session import verdict_of
from .trace import INIT, record

RULES = ("zero", "unaligned", "aligned")
ALIGNS = ("step", "exec")


def _web(key) -> str:
    return f"init:{key[1]:#x}" if key[0] == "init" else f"{key[0]:x}/{key[1]}"


def run_grid(binary, base_input, donor_inputs, *, cells=None, device="cpu", chunk: int = 4096, seed: int = 0,
             hold: bool = True, cap_factor: int = 4, log=print, traces=None) -> dict:
    dev = torch.device(device)
    if traces is None:
        traces, res = record(binary, [base_input, *donor_inputs], device=device)
    else:
        traces, res = traces
    base, donors = traces[0], traces[1:]
    ref = reference(binary, _take(res, 0), dev)
    resident = resident_pages(binary, traces)
    bweb = [_web(k) for k in base.web_labels()]
    dweb = [[_web(k) for k in d.web_labels()] for d in donors]
    by_site = [{} for _ in donors]
    for j, d in enumerate(donors):
        for c in d.cells:
            by_site[j].setdefault(c.site, []).append(c.id)
    first_div = []
    for d in donors:
        T = min(d.T, base.T)
        first_div.append(next((t for t in range(T) if d.pc[t] != base.pc[t]), T))
    cells = list(range(len(base.cells))) if cells is None else list(cells)
    rows, lane_of = [], {}

    def lane(at, site, value):
        return lane_of.setdefault((at, site, value), len(lane_of))

    for i in cells:
        c = base.cells[i]
        at = 0 if c.def_ == INIT else c.def_ + 1
        common = {"cell": i, "site": c.site, "def": c.def_, "pc": c.pc, "slot": c.slot, "base_value": c.value,
                  "base_web": bweb[i]}

        def add(rule, align, donor, value, dtick, web):
            row = {**common, "rule": rule, "align": align, "donor": donor, "value": -1 if value is None else value,
                   "donor_tick": dtick, "donor_web": web,
                   "after_divergence": donor >= 0 and c.def_ >= first_div[donor]}
            if value is None:
                row["status"] = "no-donor"
            elif value == c.value:
                row["status"] = "noop"
            else:
                row["status"], row["lane"] = "run", lane(at, c.site, value)
            rows.append(row)

        for align in ALIGNS:
            add("zero", align, -1, 0, -1, "")
        for j, d in enumerate(donors):
            pool = by_site[j].get(c.site, [])
            rng = random.Random(hash((seed, i, j)) & 0xFFFFFFFF)
            u = d.cells[rng.choice(pool)] if pool else None
            for align in ALIGNS:
                add("unaligned", align, j, None if u is None else u.value, -2 if u is None else u.def_,
                    "" if u is None else dweb[j][u.id])
            k = d.by_key.get(c.key)
            add("aligned", "exec", j, None if k is None else d.cells[k].value, -2 if k is None else d.cells[k].def_,
                "" if k is None else dweb[j][k])
            if c.def_ < d.T:
                dc = d.cell_at(c.site, c.def_)
                add("aligned", "step", j, d.value(c.site, c.def_), -2 if dc is None else dc.def_,
                    "" if dc is None else dweb[j][dc.id])
            else:
                add("aligned", "step", j, None, -2, "")
    patches = sorted(lane_of, key=lane_of.get)
    log(f"{binary.name}: grid rows {len(rows)} unique patches {len(patches)} lanes {len(patches) * (2 if hold else 1)}")
    base_y = (base.y, base.exit_code)
    free = {k: [] for k in ("stop", "ctrl_at", "addr_at", "y", "exit")}
    held = {k: [] for k in ("stop", "y", "exit", "moved_at")}
    cap = cap_factor * base.T + 64
    per = max(1, chunk // (2 if hold else 1))
    for s in range(0, len(patches), per):
        part = patches[s:s + per]
        n = len(part)
        B = 2 * n if hold else n
        lanes = Lanes(binary, [base_input] * B, device=dev, cap=cap, resident=resident)
        at = torch.tensor([p[0] for p in part], dtype=torch.int64, device=dev)
        site = torch.tensor([p[1] for p in part], dtype=torch.int64, device=dev)
        val = torch.tensor([p[2] for p in part], dtype=torch.int64, device=dev)
        rep = (lambda x: torch.cat([x, x])) if hold else (lambda x: x)
        lanes.patch(rep(at), rep(site), rep(val))
        rix = torch.zeros(B, dtype=torch.int64, device=dev)
        if hold:
            lanes.hold_to(ref, rix, torch.arange(B, device=dev) >= n)
        else:
            lanes.compare_to(ref, rix)
        r = lanes.run()
        for q in range(n):
            free["stop"].append(r.stop_kind(q))
            free["ctrl_at"].append(int(r.ctrl_at[q]))
            free["addr_at"].append(int(r.addr_at[q]))
            free["y"].append(r.y[q])
            free["exit"].append(int(r.exit_code[q]))
            if hold:
                held["stop"].append(r.stop_kind(n + q))
                held["y"].append(r.y[n + q])
                held["exit"].append(int(r.exit_code[n + q]))
                held["moved_at"].append(int(r.moved_at[n + q]))
        log(f"  patches {s + n}/{len(patches)}")
    out = {k: [] for k in ("cell", "site", "def", "pc", "slot", "rule", "align", "donor", "donor_tick", "value",
                           "base_value", "status", "verdict", "y_changed", "path_diverged", "ctrl_at", "addr_at",
                           "stop", "frozen_effect", "frozen_stop", "moved_at", "y_diff_bytes", "y_diff_mask",
                           "exit_delta", "base_web", "donor_web", "after_divergence")}
    for row in rows:
        for k in ("cell", "site", "def", "pc", "slot", "rule", "align", "donor", "donor_tick", "value", "base_value",
                  "status", "base_web", "donor_web", "after_divergence"):
            out[k].append(row[k])
        if row["status"] != "run":
            out["verdict"].append(row["status"] if row["status"] == "no-donor" else "no effect")
            for k, v in (("y_changed", False), ("path_diverged", False), ("ctrl_at", -1), ("addr_at", -1),
                         ("stop", ""), ("frozen_effect", False), ("frozen_stop", ""), ("moved_at", -1),
                         ("y_diff_bytes", 0), ("y_diff_mask", ""), ("exit_delta", 0)):
                out[k].append(v)
            continue
        q = row["lane"]
        y, e, st = free["y"][q], free["exit"][q], free["stop"][q]
        yc = (y, e) != base_y or st != "exit"
        div = free["ctrl_at"][q] >= 0 or free["addr_at"][q] >= 0
        out["verdict"].append(verdict_of(st, div, yc))
        out["y_changed"].append(yc)
        out["path_diverged"].append(div)
        out["ctrl_at"].append(free["ctrl_at"][q])
        out["addr_at"].append(free["addr_at"][q])
        out["stop"].append(st)
        mask = [i for i in range(max(len(y), len(base.y))) if i >= len(y) or i >= len(base.y) or y[i] != base.y[i]]
        out["y_diff_bytes"].append(len(mask))
        out["y_diff_mask"].append(f"{sum(1 << i for i in mask):x}" if mask else "")
        out["exit_delta"].append(e - base.exit_code)
        if hold:
            fy = (held["y"][q], held["exit"][q]) != base_y or held["stop"][q] != "exit"
            out["frozen_effect"].append(fy)
            out["frozen_stop"].append(held["stop"][q])
            out["moved_at"].append(held["moved_at"][q])
        else:
            out["frozen_effect"].append(False)
            out["frozen_stop"].append("")
            out["moved_at"].append(-1)
    meta = {"program": binary.name, "T": base.T, "cells": len(cells), "rows": len(rows), "patches": len(patches),
            "donors": len(donors), "first_pc_divergence_per_donor": first_div, "hold": hold, "seed": seed,
            "site_names": {str(s): site_name(s) for s in sorted(set(out["site"]))}}
    return {"columns": {k: np.array(v) for k, v in out.items()}, "meta": meta}


__all__ = ["run_grid", "RULES", "ALIGNS"]
