"""POST-HOC rescore of method_v2 against the gtkey answer key (isa-bench main 4b72307 scorer).

    python rescore_v2.py V2_JOB_DIR OUT.json

Records per program (from v2_worker.py's cells.npz, job-b68a43b90e6f): the frozen method (rerun,
same code as run 1), v2 (both lessons), words only, dead copies only, and the key's B1/B2 baselines.
Metrics: M1 V-measure vs R-web and M2 recall@0.5 vs R-src (DWARF), universe = the key's web-live
occurrences (``S.Key``), exactly as run 1. Programs scored in run 1: sum8-O0 bigadd-O0 bsort-O0
(the key the lessons were read from); the rest are a semi-held-out check (the lessons were chosen
before seeing them, but they are not a sealed prediction).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ISA = Path("/Users/jakeholl/domains/isa-bench")
spec = importlib.util.spec_from_file_location("score_runs", ISA / "scripts/score_runs.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
S = R.S
R.KEY = ISA / ".claude/worktrees/gtkey/outputs/gtkey"
R.KJOB.update({"isort-O0": "job-524503e74f16", "isort-O2": "job-524503e74f16",
               "sum8-O2": "job-524503e74f16"})
RUN1 = ("sum8-O0", "bigadd-O0", "bsort-O0")
PROGS = ("sum8-O0", "bigadd-O0", "bsort-O0", "sum8-O2", "bigadd-O2", "bsort-O2", "fsm-O0", "fsm-O2",
         "isort-O0", "isort-O2", "fnv-O0", "fnv-O2")


def record(m, field, tag, name):
    b = m[field]
    nb = int(b.max()) + 1 if len(b) and b.max() >= 0 else 0
    bind = [{"symbol": f"b{k}", "kind": "web", "cells": []} for k in range(nb)]
    for s, d, e, k in zip(m["site"].tolist(), m["def_"].tolist(), m["kill"].tolist(), b.tolist()):
        if k >= 0:
            bind[k]["cells"].append(R._piece(s, d, e))
    return {"schema": "mdb-symbol-table-v1", "kind": "symbol-table", "ext": S.EXT, "program": tag,
            "run": "default", "method": name, "bindings": [x for x in bind if x["cells"]], "edges": []}


def metrics(key, rec):
    w, s = S.score(key, rec, reference="web"), S.score(key, rec, reference="src")
    return {"bindings": w["bindings"], "V_rweb": w["M1"]["v"], "h_rweb": w["M1"]["h"], "c_rweb": w["M1"]["c"],
            "dwarf_recall@0.5": s["M2"]["recall@0.5"], "dwarf_labels": s["M2"]["labels"],
            "V_rsrc": s["M1"]["v"]}


def main(v2_dir, out):
    v2_dir = Path(v2_dir)
    res = {"post_hoc": True, "scorer": "isa-bench 4b72307 scripts/score_runs.py + gtkey/score.py",
           "key": str(R.KEY), "v2_job": v2_dir.name, "run1_programs": RUN1, "programs": {}}
    for tag in PROGS:
        f = v2_dir / tag / "v2/cells.npz"
        if not f.exists():
            res["programs"][tag] = {"missing": str(f)}
            continue
        m = dict(np.load(f))
        key = S.Key(R.kdir(tag))
        row = {name: metrics(key, record(m, fld, tag, name)) for name, fld in
               (("frozen", "binding"), ("v2", "binding_v2"), ("v2_words_only", "binding_words"),
                ("v2_dead_only", "binding_dead"))}
        for name, fn in (("B1", S.b1), ("B2", S.b2)):
            row[name] = metrics(key, fn(key, tag))
        if tag in R.MJOB:                 # the frozen rerun reproduces run 1's bindings?
            old = R.mcells(tag)
            row["frozen_equals_run1"] = bool(np.array_equal(old["binding"], m["binding"])
                                             and np.array_equal(old["site"], m["site"]))
        res["programs"][tag] = row
        print(tag, json.dumps({k: (v.get("V_rweb"), v.get("dwarf_recall@0.5")) for k, v in row.items()
                               if isinstance(v, dict)}), flush=True)
    Path(out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:3])
