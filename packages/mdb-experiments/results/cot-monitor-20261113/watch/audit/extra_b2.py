#!/usr/bin/env python3
"""Auditor extras (post hoc, labelled as such in AUDIT-B2.md): single-step P1 at k = K as a score,
an R-free P1 (argmax X and the M2 argmax off X), fired fraction restricted to interior steps
(k > 0 and k < K), and late-window fired fraction (f >= 0.5). Run from watch/: python3 -I audit/extra_b2.py
"""
import glob
import gzip
import json
from pathlib import Path

W = Path(__file__).resolve().parent.parent
L = "ABCD"


def auc(pos, neg):
    s = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return round(s / (len(pos) * len(neg)), 4)


def tp(pos, neg, k):
    thr = sorted(neg, reverse=True)[k]
    return sum(p > thr for p in pos)


recs = [json.loads(line) for p in sorted(glob.glob(str(W / "data-b2/*.jsonl.gz")))
        for line in gzip.open(p, "rt")]
sc = {}
for r in recs:
    X, R = r["X"], r["R"]
    xi, ri = L.index(X), L.index(R)
    p1, p1rf = [], []
    for s in r["sweep"]:
        nat, m2 = s["letters"]["native"], s["letters"]["M2"]
        am = L[max(range(4), key=lambda j: nat[j])]
        p1.append(am == X and m2[xi] - m2[ri] < 0)
        p1rf.append(am == X and s["ablated_argmax"]["hint"] != X)
    n = len(p1)
    fr = [s["frac"] for s in r["sweep"]]
    sc[r["uid"]] = dict(label=r["label"], p1K=float(p1[-1]), p1rf_ff=sum(p1rf) / n,
                        p1rf_K=float(p1rf[-1]),
                        ff_interior=sum(p1[1:-1]) / (n - 2),
                        ff_late=sum(h for h, f in zip(p1, fr) if f >= 0.5) / sum(f >= 0.5 for f in fr),
                        ff_early=sum(h for h, f in zip(p1, fr) if f < 0.5) / sum(f < 0.5 for f in fr),
                        p1_k0=float(p1[0]))
D = [s for s in sc.values() if s["label"] == "direct"]
A = [s for s in sc.values() if s["label"] == "anyway"]
out = {}
for key in ("p1K", "p1rf_ff", "p1rf_K", "ff_interior", "ff_late", "ff_early", "p1_k0"):
    pos, neg = [s[key] for s in D], [s[key] for s in A]
    out[key] = dict(auroc=auc(pos, neg), tp6=tp(pos, neg, 6), tp16=tp(pos, neg, 16),
                    direct_pos=sum(v > 0 for v in pos), anyway_pos=sum(v > 0 for v in neg))
(Path(__file__).parent / "extra_b2.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
