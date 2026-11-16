"""Beast worker: run-1 method (symbol table by intervention) and run-2 machinery (patch grid) on gt binaries.

    python worker.py --ref REF --containers CDIR --programs bsort-O0 ... [--grid] [--device cuda] --out result

Reads per program only: ``REF/programs/<p>/{mem.bin,trace.npz,prog.elf}`` (image, initial state, ELF
symbol table) and, from ``CDIR/<p>/<run>/meta.json``, the injected ``inputs.gt_input`` bytes and the
container's steps / stdout / exit code (the S4-style lane check). Nothing else in the containers
(no webs, no DWARF, no copy descriptors) is opened.

Writes ``out/<p>/method/{cells.npz, symbol_table.jsonl, circuit_map.json, table.txt, summary.json}`` and
with ``--grid`` ``out/<p>/grid/{columns.npz, meta.json}``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE / "src", HERE.parents[2] / "mdb/src"):
    if p.exists():
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def inputs_of(cdir: Path, prog: str) -> tuple[list, list]:
    runs = ["default"] + [f"in{k:02d}" for k in range(32)]
    out, checks = [], []
    for r in runs:
        meta = json.loads((cdir / prog / r / "meta.json").read_text())
        gi = meta["inputs"].get("gt_input")
        out.append(bytes.fromhex(gi["hex"]) if gi else None)
        checks.append({"run": r, "steps": meta["steps"], "stdout_hex": meta["stdout_hex"], "exit_code": meta["exit_code"]})
    return out, checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--containers", type=Path, required=True)
    ap.add_argument("--programs", nargs="+", required=True)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--grid-chunk", type=int, default=16384)
    ap.add_argument("--out", type=Path, default=Path("result"))
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    from mdb.machines.x86 import Binary
    from mdb.machines.x86.grid import run_grid
    from mdb.machines.x86.method import recover
    from mdb.machines.x86.records import circuit_map, circuit_map_record, symbol_table_records, table_text
    from mdb.machines.x86.trace import default_input

    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"torch {torch.__version__} device {dev} {torch.cuda.get_device_name() if dev == 'cuda' else ''}", flush=True)
    for prog in a.programs:
        t0 = time.time()
        b = Binary.load(a.ref / "programs" / prog)
        ins, checks = inputs_of(a.containers, prog)
        base_in = ins[0] if ins[0] is not None else default_input(b)
        rec = recover(b, base_in, ins[1:], device=dev, chunk=a.chunk, log=lambda m: print(m, flush=True))
        t_method = time.time() - t0
        lane_check = []
        for tr, c in zip([rec.base, *rec.donors], checks):
            lane_check.append(tr.T == c["steps"] and tr.y.hex() == c["stdout_hex"] and tr.exit_code == c["exit_code"])
        out = a.out / prog / "method"
        out.mkdir(parents=True, exist_ok=True)
        tr = rec.base
        cells = tr.cells
        np.savez_compressed(
            out / "cells.npz",
            site=np.array([c.site for c in cells], np.int64), def_=np.array([c.def_ for c in cells], np.int64),
            kill=np.array([c.kill for c in cells], np.int64), bits=np.array([c.bits for c in cells], np.int64),
            slot=np.array([c.slot for c in cells], np.int64), pc=np.array([c.pc for c in cells], np.int64),
            value=np.array([c.value for c in cells], np.int64), live=rec.live, binding=rec.binding,
            informative=rec.informative, effect_donors=rec.effect_donors, b1=rec.b1, b2=rec.b2,
            name=np.array([tr.describe(c) for c in cells]),
            structural_read=np.array([bool(c.readers) for c in cells]))
        with open(out / "symbol_table.jsonl", "w") as fh:
            for r in symbol_table_records(rec):
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        (out / "circuit_map.json").write_text(json.dumps({"record": circuit_map_record(rec), "map": circuit_map(rec)},
                                                         sort_keys=True))
        (out / "table.txt").write_text(table_text(rec, limit=10_000) + "\n")
        summary = {"program": prog, "T": tr.T, "cells": len(cells), "live": int(rec.live.sum()),
                   "bindings": rec.n_bindings, "lanes": rec.lanes_run, "edges": len(rec.edges),
                   "lanes_equal_binctr": f"{sum(lane_check)}/{len(lane_check)}", "duplicate_exec_keys":
                   any(t.duplicate_keys for t in [rec.base, *rec.donors]), "method_seconds": round(t_method, 1),
                   "device": dev, "torch": torch.__version__}
        print(json.dumps(summary), flush=True)
        if a.grid:
            t1 = time.time()
            live = [i for i, lv in enumerate(rec.live) if lv]
            g = run_grid(b, base_in, ins[1:], cells=live, device=dev, chunk=a.grid_chunk,
                         traces=([rec.base, *rec.donors], rec.result), log=lambda m: print(m, flush=True))
            gout = a.out / prog / "grid"
            gout.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(gout / "columns.npz", **g["columns"])
            g["meta"]["seconds"] = round(time.time() - t1, 1)
            (gout / "meta.json").write_text(json.dumps(g["meta"], indent=1))
            col = g["columns"]
            counts = {f"{r}/{al}": dict(zip(*np.unique(col["verdict"][(col["rule"] == r) & (col["align"] == al)],
                                                       return_counts=True)))
                      for r in ("zero", "unaligned", "aligned") for al in ("step", "exec")}
            summary["grid"] = {"rows": int(len(col["verdict"])), "seconds": g["meta"]["seconds"],
                               "verdicts": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in counts.items()}}
            print(json.dumps(summary["grid"]), flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
