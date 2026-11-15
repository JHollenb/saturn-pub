"""Freeze the exp1 country feature list (exp1/prereg.json census rule) from job A's L19 active
sets at the carrier row: 64 base items + 44 d1/d2 items.  python census_build.py A_DIR OUT"""
import collections
import json
import statistics
import sys

src, out = sys.argv[1], sys.argv[2]
recs = []
for name in ("exp2-items-s0.jsonl", "exp2-items-s1.jsonl", "census-items-s1.jsonl"):
    recs += [json.loads(line) for line in open(f"{src}/{name}")]
by_c = collections.defaultdict(list)
for r in recs:
    by_c[r["country"]].append(r)
n = len(recs)
present = collections.defaultdict(lambda: collections.Counter())
acts = collections.defaultdict(list)
for r in recs:
    for f, a in zip(r["l19"]["ids"], r["l19"]["acts"]):
        present[f][r["country"]] += 1
        acts[(f, r["country"])].append(a)
features, why = {}, {}
for f, cnt in present.items():
    best = None
    for c, k in cnt.items():
        nc = len(by_c[c])
        other = sum(v for cc, v in cnt.items() if cc != c)
        share, null = k / nc, other / (n - nc)
        if k >= 2 and share >= 0.5 and null <= 2 / 57:
            if best is None or share > best[1]:
                best = (c, share, null, k, nc, other)
    if best:
        features[f] = best[0]
        why[f] = {"country": best[0], "on": f"{best[3]}/{best[4]}", "other": f"{best[5]}/{n - best[4]}"}
a_truth = {}
for c in by_c:
    mine = [(statistics.median(acts[(f, c)]), f) for f, cc in features.items() if cc == c]
    if mine:
        med, f = max(mine)
        a_truth[c] = {"feature": f, "a_median": med, "n": len(acts[(f, c)])}
doc = {"rule": "exp1/prereg.json census_rule_frozen_before_depth_arms", "items": n,
       "countries_with_items": {str(c): len(v) for c, v in by_c.items()},
       "features": {str(f): c for f, c in sorted(features.items())},
       "evidence": {str(f): v for f, v in sorted(why.items())},
       "a_truth": {str(c): v for c, v in a_truth.items()},
       "entry1_features": {"China": 17491, "Japan": 32521, "Greece": 22638, "Russia": 23913,
                           "Spain": 11201, "Netherlands": 27085, "Italy": 26232}}
doc["entry1_admitted"] = {k: (str(v) in doc["features"]) for k, v in doc["entry1_features"].items()}
json.dump(doc, open(out, "w"), indent=1)
print(f"items {n} features {len(features)} countries covered {len(a_truth)}/{len(by_c)}")
print("entry1 admitted", doc["entry1_admitted"])
