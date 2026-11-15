#!/usr/bin/env python3
"""Auditor follow-ups: mid-chain M2 flag split by whether the forced answer is X before ablation;
Spearman pool definitions; near-tie anyway items. Run from results/cot-monitor-20261113/."""
import gzip, json, math, statistics as st
from pathlib import Path
import importlib.util
H = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rc", H / "audit" / "recompute.py"); rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
items = {it["uid"]: it for it in json.loads((H / "items.json").read_text())["items"]}
recs = {}
for p in sorted((H / "data").glob("*.jsonl.gz")):
    for l in gzip.decompress(p.read_bytes()).decode().splitlines():
        if l.strip():
            r = json.loads(l); recs[(r["uid"], r["f"])] = r
def sp(m): return 2.0 ** (math.floor(math.log2(abs(m))) - 23) if m else 0
out = {}
for f in (0.25, 0.5, 0.75, 1.0):
    for lab in ("direct", "anyway"):
        rs = [(r, items[u]) for (u, ff), r in recs.items() if ff == f and items[u]["label"] == lab and items[u]["hint_type"] != "sycophancy"]
        fl = [(r, it) for r, it in rs if r["margin_ablated"] < 0 and abs(r["margin"] - r["margin_ablated"]) > max(2 * sp(r["margin"]), abs(r["margin"] - r["margin_hf"]))]
        out[f"{f}|{lab}"] = {"n": len(rs), "flagged": len(fl),
                             "flagged_with_margin<=0_already": sum(r["margin"] <= 0 for r, _ in fl),
                             "flagged_with_delta<0(ablation_moves_toward_X)": sum(r["margin"] - r["margin_ablated"] < 0 for r, _ in fl),
                             "strict_flip_X_to_offX": sum(r["margin"] > 0 for r, _ in fl),
                             "n_answer_X": sum(r["margin"] > 0 for r, _ in rs),
                             "strict_unnamed": [sum(r["margin"] > 0 and not r["prefix_names_x_letter"] for r, _ in fl),
                                                sum(not r["prefix_names_x_letter"] for r, _ in rs)]}
# spearman pools
f1 = [(r, items[u]) for (u, f), r in recs.items() if f == 1.0]
A = [r for r, it in f1 if it["primary"]]
B = [r for r, it in f1 if it["hint_type"] != "sycophancy"]
M = lambda r: r["bindings"]["hint"]["to_consumer"]; N = lambda r: r["margin"] - r["margin_ablated"]
out["spearman_item.primary_flag"] = [round(rc.spearman([M(r) for r in A], [N(r) for r in A]), 3), len(A)]
out["spearman_non_sycophancy"] = [round(rc.spearman([M(r) for r in B], [N(r) for r in B]), 3), len(B)]
print(json.dumps(out, indent=1))
(H / "audit" / "midchain_strict.json").write_text(json.dumps(out, indent=1))
