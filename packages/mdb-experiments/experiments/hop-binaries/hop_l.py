"""Study L (latent two-hop on binaries): bridge truth, the ported forecast rule, multi-donor interchange.

    python hop_l.py --phase passive  --ref REF --containers CDIR --programs hop2-O0 ... --out result
    python hop_l.py --phase interchange --ref REF --containers CDIR --programs ... --forecasts F.json --out result

PREREG: saturn/experiments/2026-11-16-cot-latent-binaries/L/PREREG.md. Per program (tag) the 33 inputs of binctr
run 0 (default + in00..in31) are run as torchx86 lanes (``mdb.machines.x86``); every input is one item.

passive (no intervention): per item
  - truth: units holding a bridge variable. (i) R-src: gtkey ``rsrc.locate`` on the binctr DWARF sidecar, at every
    in-scope state, kept where the located bytes equal the source value of the bridge (reference.bridges);
    (ii) R-val: a unit (same tick, same slot group on all 33 runs; the programs are straight-line) whose word
    decodes to the bridge's source value on all 33 inputs with an identity or slice decode (``rval.decodes``).
  - candidates: value units (dst register, second destination, stored memory bytes) defined at a tick >= 0 and
    before the answer's write syscall, that are input-derived (data/addr forward closure from gt_input) and not in the answer's copy web.
  - F_direct (primary port of "argmax of summed reads to_consumer"): a unit's score is the sum over its distinct
    reader ticks of the number of output bytes whose copy web contains a cell that reader defines (the direct
    contribution: the read feeds the answer value itself, as p.W_O.v feeds the margin in one step).
  - F_total (secondary): the same sum with the number of output bytes reachable from the reader's writes through
    any chain of data/addr reads.
  - ties: hit@k is the expected value under a uniform random order inside tied groups.
  - novelty N: a unit is a re-encoding of the input when its word decodes to x on all 33 inputs under the full
    R-val decode class (identity, slice, xor, affine, slice_affine).
interchange (native, GPU lanes): per item, for the F_direct top-1 unit(s), every truth unit and the F_total top-1
  unit(s): 16 donors (the next 16 inputs, cyclic), aligned by execution key, all bytes of the unit at once; a free
  lane and a held lane (branch outcomes and effective addresses frozen to the item's own run). Plus one ``whatif``
  call (the session verb) per program on the default item's first truth unit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE / "src", HERE.parents[2] / "mdb/src"):
    if p.exists():
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

RUNS = ["default"] + [f"in{k:02d}" for k in range(32)]
N_DONORS = 16
GROUPS = {"dst": range(0, 8), "d2": range(8, 16), "sp": range(16, 24), "mem": range(24, 32)}
VALUE_GROUPS = ("dst", "d2", "mem")
TRUTH_DECODES = ("identity", "slice")


def group_of(slot: int) -> str:
    for g, r in GROUPS.items():
        if slot in r:
            return g
    return "flag" if 32 <= slot < 38 else "sys"


def inputs_of(cdir: Path, prog: str) -> tuple[list, list]:
    out, checks = [], []
    for r in RUNS:
        meta = json.loads((cdir / prog / r / "meta.json").read_text())
        gi = meta["inputs"].get("gt_input")
        out.append(bytes.fromhex(gi["hex"]) if gi else None)
        checks.append({"steps": meta["steps"], "stdout_hex": meta["stdout_hex"], "exit_code": meta["exit_code"]})
    return out, checks


# ------------------------------------------------------------------------------------------ per-trace structure
class Item:
    """One run's units, readers, copy webs and the answer bytes."""

    def __init__(self, tr, binary):
        self.tr = tr
        self.units = {}                       # (def tick, group) -> [cell ids]
        for c in tr.cells:
            if c.def_ >= 0:
                self.units.setdefault((c.def_, group_of(c.slot)), []).append(c.id)
        self.unit_of = {cid: k for k, cs in self.units.items() for cid in cs}
        self.cells_def_at = {}
        for c in tr.cells:
            if c.def_ >= 0:
                self.cells_def_at.setdefault(c.def_, []).append(c.id)
        # read events (tick, cell, edge)
        by_site = {}
        for c in tr.cells:
            by_site.setdefault(c.site, []).append(c)
        self.read_cells = {t: [] for t in range(tr.T)}
        for c in tr.cells:
            for t, e in c.readers:
                self.read_cells[t].append((c.id, e))
        # answer: the data bytes of the last write syscall
        wt = [t for t, tk in enumerate(tr.ticks) if tk.kind == "syscall" and any(s >= 0x1000 for s, e in tk.reads)]
        self.y_tick = wt[-1]
        self.y_cells = [cid for cid, e in self.read_cells[self.y_tick] if e == "data" and tr.cells[cid].site >= 0x1000]
        self.root = tr.webs()
        self.y_roots = {self.root[c] for c in self.y_cells}
        ia, isz = binary.input_addr, binary.input_size
        self.derived = [False] * len(tr.cells)
        for c in tr.cells:
            if c.def_ < 0 and ia is not None and ia <= c.site < ia + isz:
                self.derived[c.id] = True
        for t in range(tr.T):
            hit = any(self.derived[cid] for cid, e in self.read_cells[t] if e in ("data", "addr"))
            for cid in self.cells_def_at.get(t, ()):
                self.derived[cid] = hit

    def y_direct(self, t: int) -> int:
        roots = {self.root[c] for c in self.cells_def_at.get(t, ())}
        return sum(self.root[c] in roots for c in self.y_cells)

    def y_reach(self, t: int) -> int:
        tainted = set(self.cells_def_at.get(t, ()))
        for t2 in range(t + 1, self.tr.T):
            if any(cid in tainted for cid, e in self.read_cells[t2] if e in ("data", "addr")):
                tainted |= set(self.cells_def_at.get(t2, ()))
        return sum(c in tainted for c in self.y_cells)

    def candidates(self) -> list:
        out = []
        for k, cs in self.units.items():
            if k[1] not in VALUE_GROUPS or k[0] >= self.y_tick:      # value units defined before the answer is emitted
                continue
            if not any(self.derived[c] for c in cs):
                continue
            if any(self.root[c] in self.y_roots for c in cs):
                continue
            out.append(k)
        return sorted(out)

    def readers_of(self, k) -> dict:
        """Distinct reader ticks of a unit -> set of edge types."""
        out = {}
        for c in self.units[k]:
            for t, e in self.tr.cells[c].readers:
                out.setdefault(t, set()).add(e)
        return out

    def scores(self) -> dict:
        cache_d, cache_r = {}, {}
        res = {}
        for k in self.candidates():
            rd = self.readers_of(k)
            fd = sum(cache_d.setdefault(t, self.y_direct(t)) for t in rd)
            ft = sum(cache_r.setdefault(t, self.y_reach(t)) for t in rd)
            res[k] = (fd, ft)
        return res

    def word(self, k) -> int:
        """Unit value: the destination register after the tick, or the stored bytes little-endian."""
        t, g = k
        tr = self.tr
        if g in ("dst", "d2"):
            c = tr.cells[self.units[k][0]]
            return tr.reg(t, c.site // 8)
        cs = sorted((tr.cells[c] for c in self.units[k]), key=lambda c: c.site)
        return sum(c.value << (8 * i) for i, c in enumerate(cs[:8]))


def ranking(scores: dict, which: int) -> list:
    """[(score, [unit keys])] in descending score."""
    groups = {}
    for k, s in scores.items():
        groups.setdefault(s[which], []).append(k)
    return [(s, sorted(groups[s])) for s in sorted(groups, reverse=True)]


def hit_at(rank: list, truth: set, k: int) -> float:
    """Expected hit@k under a uniform random order inside tied groups."""
    left = k
    for _s, units in rank:
        if left <= 0:
            return 0.0
        g, m = len(units), sum(u in truth for u in units)
        if g <= left:
            if m:
                return 1.0
            left -= g
            continue
        return 1.0 - math.comb(g - m, left) / math.comb(g, left)
    return 0.0


# ------------------------------------------------------------------------------------------ truth
def truth_units(items, inputs, tag, dw, ref_mod) -> tuple[list, dict]:
    from isa_bench.gtkey import rsrc, rval  # noqa: PLC0415

    names = list(ref_mod.bridges(tag, inputs[0]).keys())
    bvals = [ref_mod.bridges(tag, x) for x in inputs]
    per_item = [dict() for _ in items]            # unit -> set of variable names
    report = {"variables": names, "rsrc_status": {}, "rsrc_units": {}, "rval_units": {}}
    # (i) R-src, value-checked
    for n in names:
        v = next((x for x in dw.vars if x["label"] == n), None)
        st_counts = {}
        hits = 0
        for i, it in enumerate(items):
            tr = it.tr
            for t in range(-1, tr.T - 1):
                pc = int(tr.pc[t + 1])
                if v is None or not dw.in_scope(v, pc):
                    continue
                regs = [tr.reg(t, r) for r in range(16)]
                st, keys, _ = rsrc.locate(dw, v, pc, regs)
                st_counts[st] = st_counts.get(st, 0) + 1
                if st != "located":
                    continue
                val = sum(tr.value(kk, t) << (8 * j) for j, kk in enumerate(keys))
                if val != bvals[i][n]:
                    continue
                for kk in keys:
                    c = tr.cell_at(kk, t)
                    if c is None:
                        continue
                    u = it.unit_of.get(c.id, ("init", c.site))
                    per_item[i].setdefault(u, set()).add(n)
                    hits += 1
        report["rsrc_status"][n] = st_counts
        report["rsrc_units"][n] = sorted({str(u) for d in per_item for u, ns in d.items() if n in ns})
    # (ii) R-val identity / slice on units aligned by (tick, group)
    T = items[0].tr.T
    same = all(it.tr.T == T and np.array_equal(it.tr.pc, items[0].tr.pc) for it in items)
    report["straight_line_aligned"] = bool(same)
    if same:
        keys = [k for k in items[0].units if k[1] in VALUE_GROUPS and all(k in it.units for it in items)]
        for n in names:
            target = np.array([b[n] for b in bvals], dtype=np.uint64)
            found = []
            for k in keys:
                X = np.array([it.word(k) for it in items], dtype=np.uint64)
                ds = [d for d in rval.decodes(X, target, 1) if d[0] in TRUTH_DECODES]
                if ds:
                    found.append({"unit": list(k), "decode": [list(map(str, d)) for d in ds]})
                    for i in range(len(items)):
                        per_item[i].setdefault(k, set()).add(n)
                    if not any(d[0] == "identity" for d in ds):
                        report.setdefault("slice_only_units", []).append(list(k))
            report["rval_units"][n] = found
    return per_item, report


def novelty(items, inputs) -> dict:
    """unit -> True when the unit's word is NOT an R-val re-encoding of the input byte (full decode class)."""
    from isa_bench.gtkey import rval  # noqa: PLC0415

    x = np.array([b[0] for b in inputs], dtype=np.uint64)
    keys = [k for k in items[0].units if k[1] in VALUE_GROUPS and all(k in it.units for it in items)]
    out = {}
    for k in keys:
        X = np.array([it.word(k) for it in items], dtype=np.uint64)
        out[k] = not rval.decodes(X, x, 1)
    return out


# ------------------------------------------------------------------------------------------ passive phase
def passive(tag, binary, items, inputs, dw, ref_mod) -> dict:
    truth, trep = truth_units(items, inputs, tag, dw, ref_mod)
    nov = novelty(items, inputs)
    rows = []
    for i, it in enumerate(items):
        sc = it.scores()
        rd, rt = ranking(sc, 0), ranking(sc, 1)
        tset = set(truth[i]) if tag.split("-")[0] != "shortcut" else set()
        top_d = rd[0][1] if rd and rd[0][0] > 0 else []
        top_t = rt[0][1] if rt and rt[0][0] > 0 else []
        edges = {}
        for u in truth[i]:
            if u[0] == "init" or u not in it.units:
                continue
            for t, es in it.readers_of(u).items():
                consumer = all(it.unit_of.get(c) not in truth[i] for c in it.cells_def_at.get(t, ()))
                for e in es:
                    edges.setdefault("all", {}).setdefault(e, 0)
                    edges["all"][e] += 1
                    if consumer:
                        edges.setdefault("consuming", {}).setdefault(e, 0)
                        edges["consuming"][e] += 1
        slice_only = {tuple(u) for u in trep.get("slice_only_units", [])}
        rsrc_units = {u for n in trep["variables"] for u in trep["rsrc_units"].get(n, [])}
        tset_id = {u for u in tset if u not in slice_only or str(u) in rsrc_units}
        rows.append({
            "item": i, "input": inputs[i].hex(), "y": it.tr.y.hex(),
            "identity_truth": {"F_direct_hit1": hit_at(rd, tset_id, 1), "F_direct_hit3": hit_at(rd, tset_id, 3),
                               "F_total_hit1": hit_at(rt, tset_id, 1), "F_total_hit3": hit_at(rt, tset_id, 3)},
            "n_candidates": len(sc),
            "truth": sorted([list(map(str, u)), sorted(v)] for u, v in truth[i].items()),
            "truth_candidates": sorted(list(u) for u in tset if u in sc),
            "F_direct": {"rank": [[s, [list(u) for u in us]] for s, us in rd[:4]], "top1": [list(u) for u in top_d],
                         "hit1": hit_at(rd, tset, 1), "hit3": hit_at(rd, tset, 3)},
            "F_total": {"rank": [[s, [list(u) for u in us]] for s, us in rt[:4]], "top1": [list(u) for u in top_t],
                        "hit1": hit_at(rt, tset, 1), "hit3": hit_at(rt, tset, 3)},
            "novel_top1_direct": [bool(nov.get(u, True)) for u in top_d],
            "units_text": {str(u): describe(it, u) for u in set(top_d) | set(top_t) | set(truth[i]) if u in it.units},
            "bridge_edges": edges})
    return {"tag": tag, "T": items[0].tr.T, "truth_report": trep, "items": rows}


def describe(it, u) -> str:
    tr = it.tr
    tk = tr.ticks[u[0]]
    return f"i{u[0]} {tk.pc:#x} {tk.text} [{u[1]}] = {it.word(u):#x}"


# ------------------------------------------------------------------------------------------ interchange phase
def interchange(tag, binary, traces, res, items, inputs, plan, dev, log) -> dict:
    from mdb.machines.x86.lanes import Lanes, reference  # noqa: PLC0415
    from mdb.machines.x86.method import resident_pages  # noqa: PLC0415

    n = len(items)
    jobs = []                                  # (item, unit, role, donor, patches[(at, site, val)], informative)
    for i, it in enumerate(items):
        for role, units in plan[i].items():
            for u in units:
                u = tuple(u)
                if u not in it.units:
                    continue
                cells = [it.tr.cells[c] for c in it.units[u]]
                for j in range(1, N_DONORS + 1):
                    d = (i + j) % n
                    dtr = traces[d]
                    pat, inf, miss = [], False, False
                    for c in cells:
                        k = dtr.by_key.get(c.key)
                        if k is None:
                            miss = True
                            break
                        v = dtr.cells[k].value
                        inf |= v != c.value
                        pat.append((c.def_ + 1, c.site, v))
                    if not miss:
                        jobs.append((i, u, role, d, pat, inf))
    log(f"{tag}: interchange patches {len(jobs)} lanes {2 * len(jobs)}")
    ref = reference(binary, res, torch.device(dev))
    resident = resident_pages(binary, traces)
    out = []
    per = 2048
    for s in range(0, len(jobs), per):
        part = jobs[s:s + per]
        m = len(part)
        B = 2 * m
        lanes = Lanes(binary, [inputs[p[0]] for p in part] * 2, device=dev, cap=4 * max(t.T for t in traces) + 64,
                      resident=resident)
        width = max(len(p[4]) for p in part)
        for b in range(width):
            at = [p[4][b][0] if b < len(p[4]) else -1 for p in part] * 2
            site = [p[4][b][1] if b < len(p[4]) else -1 for p in part] * 2
            val = [p[4][b][2] if b < len(p[4]) else 0 for p in part] * 2
            lanes.patch(at, site, val)
        rix = [p[0] for p in part] * 2
        lanes.hold_to(ref, rix, [False] * m + [True] * m)
        r = lanes.run()
        for q, (i, u, role, d, _pat, inf) in enumerate(part):
            base = (traces[i].y, traces[i].exit_code)
            tot = (r.y[q], int(r.exit_code[q])) != base or r.stop_kind(q) != "exit"
            frz = (r.y[m + q], int(r.exit_code[m + q])) != base or r.stop_kind(m + q) != "exit"
            out.append({"item": i, "unit": list(u), "role": role, "donor": d, "informative": bool(inf),
                        "total": bool(tot), "frozen": bool(frz), "stop": r.stop_kind(q),
                        "to_donor_y": r.y[q] == traces[d].y and not (traces[d].y == traces[i].y),
                        "ctrl_at": int(r.ctrl_at[q]), "addr_at": int(r.addr_at[q]),
                        "moved_under_hold": int(r.moved_at[m + q])})
    return {"tag": tag, "rows": out}


def whatif_once(tag, ref_dir, inputs, items, plan0, dev) -> dict:
    """The session verb on the default item: seek to the first truth unit's def, write donor 1's value."""
    from mdb.machines.x86 import attach  # noqa: PLC0415
    from mdb.machines.x86.binary import GPRS  # noqa: PLC0415

    units = [tuple(u) for u in plan0.get("truth", [])]
    if not units:
        return {"tag": tag, "skipped": "no truth unit"}
    u = units[0]
    it, dn = items[0], items[1]
    c = it.tr.cells[it.units[u][0]]
    k = dn.tr.by_key.get(c.key)
    run = attach(ref_dir / tag, inputs[0], device=dev)
    run.seek(u[0] + 1)
    if u[1] in ("dst", "d2"):
        text = GPRS[c.site // 8]
        value = dn.word(u)
    else:
        cs = sorted((it.tr.cells[x] for x in it.units[u]), key=lambda x: x.site)
        text = f"mem[{cs[0].site:#x}:{len(cs)}]"
        value = dn.word(u)
    r = run.whatif(text, value)
    return {"tag": tag, "unit": list(u), "address": text, "value": hex(value), "donor_item": 1, "result": r}


def shortcut_check(items, inputs, ref_mod) -> dict:
    """G11: units of the shortcut whose word decodes to T1[x] on all 33 inputs (identity/slice, and full class)."""
    from isa_bench.gtkey import rval  # noqa: PLC0415

    target = np.array([ref_mod.T1[x[0]] for x in inputs], dtype=np.uint64)
    keys = [k for k in items[0].units if k[1] in VALUE_GROUPS and all(k in it.units for it in items)]
    idsl, full = [], []
    for k in keys:
        X = np.array([it.word(k) for it in items], dtype=np.uint64)
        ds = rval.decodes(X, target, 1)
        if ds:
            full.append(list(k))
        if any(d[0] in TRUTH_DECODES for d in ds):
            idsl.append(list(k))
    return {"value_units": len(keys), "identity_or_slice": idsl, "full_class": full}


# ------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("passive", "interchange"), required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--containers", type=Path, required=True)
    ap.add_argument("--programs-dir", type=Path, default=HERE / "programs/hop")
    ap.add_argument("--programs", nargs="+", required=True)
    ap.add_argument("--forecasts", type=Path, help="directory of <tag>.passive.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("result"))
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    sys.path.insert(0, str(a.programs_dir))
    import reference as ref_mod  # noqa: PLC0415
    from isa_bench.gtkey import rsrc  # noqa: PLC0415
    from mdb.machines.x86 import Binary  # noqa: PLC0415
    from mdb.machines.x86.trace import record  # noqa: PLC0415

    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"torch {torch.__version__} device {dev} {torch.cuda.get_device_name() if dev == 'cuda' else ''}",
          flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for tag in a.programs:
        t0 = time.perf_counter()
        binary = Binary.load(a.ref / tag)
        inputs, checks = inputs_of(a.containers, tag)
        inputs = [x if x is not None else ref_mod.DEFAULTS[tag.split("-")[0]] for x in inputs]
        traces, res = record(binary, inputs, device=dev)
        lane_ok = sum(tr.T == ch["steps"] and tr.y.hex() == ch["stdout_hex"] and tr.exit_code == ch["exit_code"]
                      and tr.stop == "exit" for tr, ch in zip(traces, checks))
        items = [Item(tr, binary) for tr in traces]
        dw = rsrc.Dwarf.load(a.containers / tag / "dwarf.json")
        if a.phase == "passive":
            out = passive(tag, binary, items, inputs, dw, ref_mod)
            out["lanes_equal_binctr"] = f"{lane_ok}/{len(traces)}"
            (a.out / f"{tag}.passive.json").write_text(json.dumps(out, indent=1, default=str))
            summary[tag] = {"lanes_equal_binctr": out["lanes_equal_binctr"],
                            "F_direct_hit1": sum(r["F_direct"]["hit1"] for r in out["items"]),
                            "F_direct_hit3": sum(r["F_direct"]["hit3"] for r in out["items"]),
                            "F_total_hit1": sum(r["F_total"]["hit1"] for r in out["items"]),
                            "F_total_hit3": sum(r["F_total"]["hit3"] for r in out["items"]),
                            "items": len(out["items"]), "s": round(time.perf_counter() - t0, 1)}
        else:
            plan = []
            passive_rows = json.loads((a.forecasts / f"{tag}.passive.json").read_text())["items"]
            for i, it in enumerate(items):
                pr = passive_rows[i]
                truth = [(int(t), g) for (t, g), _names in pr["truth"] if t != "init"]
                plan.append({"forecast_direct": pr["F_direct"]["top1"], "truth": truth,
                             "forecast_total": pr["F_total"]["top1"]})
            out = interchange(tag, binary, traces, res, items, inputs, plan, dev, lambda m: print(m, flush=True))
            out["lanes_equal_binctr"] = f"{lane_ok}/{len(traces)}"
            out["whatif"] = whatif_once(tag, a.ref, inputs, items, plan[0], dev)
            if tag.startswith("shortcut"):
                out["shortcut_holds_T1x"] = shortcut_check(items, inputs, ref_mod)
            (a.out / f"{tag}.interchange.json").write_text(json.dumps(out, indent=1, default=str))
            summary[tag] = {"lanes_equal_binctr": out["lanes_equal_binctr"], "rows": len(out["rows"]),
                            "whatif": {k: out["whatif"].get("result", {}).get(k) for k in
                                       ("verdict", "total_effect", "frozen_effect", "label")},
                            "s": round(time.perf_counter() - t0, 1)}
        print(json.dumps({tag: summary[tag]}), flush=True)
    (a.out / f"summary-{a.phase}.json").write_text(json.dumps(summary, indent=1))
    digest = hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()
    print(json.dumps({"phase": a.phase, "summary_sha256": digest}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
