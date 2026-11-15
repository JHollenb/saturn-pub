#!/usr/bin/env python3
"""Auditor: TPR on direct at fixed false-alarm counts on anyway (primary, f=1) for B2, M1/margin, M2/margin, M3/margin."""
import gzip, json
from pathlib import Path
H = Path(__file__).resolve().parent.parent
items = {it["uid"]: it for it in json.loads((H / "items.json").read_text())["items"]}
recs = {}
for p in sorted((H / "data").glob("*.jsonl.gz")):
    for l in gzip.decompress(p.read_bytes()).decode().splitlines():
        if l.strip():
            r = json.loads(l); recs[(r["uid"], r["f"])] = r
L = "ABCD"
def m3(r, xi):
    hf, vz = r["hf_letters"], r["hf_vzero_suffix_letters"]; o = [j for j in range(4) if j != xi]
    return (hf[xi] - max(hf[j] for j in o)) - (vz[xi] - max(vz[j] for j in o))
D, A = [], []
for (u, f), r in recs.items():
    it = items[u]
    if f != 1.0 or it["hint_type"] == "sycophancy" or it["label"] not in ("direct", "anyway"):
        continue
    mg = r["margin"]; xi = L.index(it["x"])
    s = {"B2": -mg, "M1_rel": r["bindings"]["hint"]["to_consumer"] / mg, "M2_rel": (mg - r["margin_ablated"]) / mg,
         "M3_rel": m3(r, xi) / mg}
    (D if it["label"] == "direct" else A).append(s)
out = {}
for m in ("B2", "M1_rel", "M2_rel", "M3_rel"):
    neg = sorted((a[m] for a in A), reverse=True)
    out[m] = {str(fa): sum(d[m] > neg[fa] for d in D) for fa in (0, 3, 6, 10, 16, 20)}
print(json.dumps(out, indent=1)); (H / "audit" / "tpr_at_fa.json").write_text(json.dumps(out, indent=1))
