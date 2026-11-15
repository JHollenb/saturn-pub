#!/usr/bin/env python3
"""B2 watchpoint subset: all 67 primary direct reads, 67 margin-nearest primary anyway items (greedy
nearest-neighbour on the 11-13 native f=1 margin, without replacement, direct items in ascending
margin order), 30 primary rationalized (seeded sample of 50). Adds the no-hint answer letter N from
the 10-15 reports. Uses only margins already measured on 11-13 (no monitor score).

Usage: python3 build_subset.py OLD_RESULTS_DIR OUT_JSON
"""
import glob
import gzip
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OLD, OUT = Path(sys.argv[1]), Path(sys.argv[2])
items = {it["uid"]: it for it in json.loads((HERE / "items.json").read_text())["items"]}
margin = {}
for p in sorted(glob.glob(str(HERE / "data/*.jsonl.gz"))):
    for line in gzip.decompress(Path(p).read_bytes()).decode().splitlines():
        r = json.loads(line)
        if r["f"] == 1.0:
            margin[r["uid"]] = r["margin"]
none = {}
for name, pat in (("greedy", "full-job-25469e40d45e/report.json"), ("E1", "e1-s*-full-*/report.json"), ("E4", "e4-s*-full-*/report.json")):
    for p in sorted(glob.glob(str(OLD / pat))):
        for r in json.loads(Path(p).read_text())["rows"]:
            none[f"{name}|{r['id']}|{r.get('sample', 0)}"] = r["none"]["answer"]
prim = [it for it in items.values() if it["hint_type"] != "sycophancy"]
direct = sorted([it for it in prim if it["label"] == "direct"], key=lambda it: margin[it["uid"]])
pool = [it for it in prim if it["label"] == "anyway"]
pairs, anyway = [], []
for d in direct:
    best = min(pool, key=lambda a: (abs(margin[a["uid"]] - margin[d["uid"]]), a["uid"]))
    pool.remove(best)
    anyway.append(best)
    pairs.append([d["uid"], best["uid"], round(margin[d["uid"]], 4), round(margin[best["uid"]], 4)])
rat = sorted([it for it in prim if it["label"] == "rationalized"], key=lambda it: it["uid"])
rat = random.Random(20261114).sample(rat, 30)
out = []
for it in direct + anyway + rat:
    s, h, q, k = it["uid"].split("|")
    rec = dict(it)
    rec["no_hint_letter"] = none[f"{s}|{q}|{k}"]
    rec["margin_1113"] = margin[it["uid"]]
    rec["fracs"] = [1.0]
    out.append(rec)
md = sorted(margin[d["uid"]] for d in direct)
ma = sorted(margin[a["uid"]] for a in anyway)
body = {"schema": "mdb-cot-watch-subset-v1", "seed": 20261114, "n": {"direct": len(direct), "anyway": len(anyway), "rationalized": len(rat)},
        "margin_median": {"direct": md[len(md) // 2], "anyway_matched": ma[len(ma) // 2]},
        "pairs": pairs, "items": out}
OUT.write_text(json.dumps(body, ensure_ascii=False))
print(json.dumps({k: body[k] for k in ("n", "margin_median")}))
print("anyway N == X:", sum(r["no_hint_letter"] == r["x"] for r in out if r["label"] == "anyway"), "direct N == X:", sum(r["no_hint_letter"] == r["x"] for r in out if r["label"] == "direct"))
