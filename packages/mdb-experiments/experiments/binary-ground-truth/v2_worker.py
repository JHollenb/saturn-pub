"""Beast worker: method_v2 (POST HOC) next to the frozen run-1 method, on every gt binary.

    python v2_worker.py --ref REF --containers CDIR --programs sum8-O0 ... --out result

Per program: the frozen ``method.recover`` (unchanged, 3d0baff code), then ``method_v2.lessons`` in
three forms (both lessons, words only, dead copies only). Reads only what worker.py reads (image,
initial state, injected inputs, container steps/stdout/exit for the lane check). Writes
``out/<p>/v2/cells.npz`` (per cell: site, def_, kill, value, live, the frozen binding and the three
v2 bindings, v2 status per cell, copy-web root, B2 label) and ``summary.json``.
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

from worker import inputs_of  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--containers", type=Path, required=True)
    ap.add_argument("--programs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--out", type=Path, default=Path("result"))
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    from mdb.machines.x86 import Binary
    from mdb.machines.x86.method import recover
    from mdb.machines.x86.method_v2 import SPEC, lessons
    from mdb.machines.x86.trace import default_input

    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"torch {torch.__version__} device {dev}", flush=True)
    for prog in a.programs:
        t0 = time.time()
        if not (a.ref / "programs" / prog).exists() or not (a.containers / prog).exists():
            print(json.dumps({"program": prog, "skipped": "no reference or containers"}), flush=True)
            continue
        b = Binary.load(a.ref / "programs" / prog)
        ins, checks = inputs_of(a.containers, prog)
        base_in = ins[0] if ins[0] is not None else default_input(b)
        rec = recover(b, base_in, ins[1:], device=dev, chunk=a.chunk, log=lambda m: print(m, flush=True))
        lane_ok = [tr.T == c["steps"] and tr.y.hex() == c["stdout_hex"] and tr.exit_code == c["exit_code"]
                   for tr, c in zip([rec.base, *rec.donors], checks)]
        both, words, dead = lessons(rec), lessons(rec, dead_copies=False), lessons(rec, words=False)
        cells = rec.base.cells
        out = a.out / prog / "v2"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "cells.npz",
            site=np.array([c.site for c in cells], np.int64), def_=np.array([c.def_ for c in cells], np.int64),
            kill=np.array([c.kill for c in cells], np.int64), value=np.array([c.value for c in cells], np.int64),
            live=rec.live, binding=rec.binding, binding_v2=both.binding, binding_words=words.binding,
            binding_dead=dead.binding, status_v2=np.array([both.status[k] if k >= 0 else "unbound"
                                                           for k in both.binding]),
            web=both.web, b2=rec.b2, structural_read=np.array([bool(c.readers) for c in cells]))
        summary = {"program": prog, "cells": len(cells), "live": int(rec.live.sum()),
                   "bindings_frozen": rec.n_bindings, "bindings_v2": both.n_bindings,
                   "v2_live": sum(s == "live" for s in both.status),
                   "v2_dead": sum(s == "dead" for s in both.status),
                   "bindings_words": words.n_bindings, "bindings_dead_only": dead.n_bindings,
                   "bound_cells_frozen": int((rec.binding >= 0).sum()),
                   "bound_cells_v2": int((both.binding >= 0).sum()),
                   "lanes_equal_binctr": f"{sum(lane_ok)}/{len(lane_ok)}", "seconds": round(time.time() - t0, 1),
                   "device": dev, "spec": SPEC,
                   "cuda_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else None}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
