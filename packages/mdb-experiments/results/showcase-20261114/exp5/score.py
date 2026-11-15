"""Score exp5 (prereg exp5/prereg.json).  python score.py ITEMS_JSONL OUT_JSON"""
import json
import statistics as st
import sys


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


R = [json.loads(line) for line in open(sys.argv[1])]
n = len(R)
n1a = sum(1 for r in R if abs(r["native_batched"]["set"]) - abs(r["native_batched"]["random"]) >= 0.5)
n1b = sum(1 for r in R if abs(r["native_batched"]["set"]) > abs(r["native_batched"]["answer_null"]))
fc, nat, by_layer = [], [], {}
for r in R:
    fl = r["floor"] + abs(r["base_batched"] - r["base_stepped"])
    for k, v in r["native_batched"].items():
        if not k.startswith("L") or abs(v) <= fl:
            continue
        f = r["forecast"][k]
        fc.append(f)
        nat.append(v)
        by_layer.setdefault(int(k[1:].split(".")[0]), []).append(f / v)
ratio = [a / b for a, b in zip(fc, nat)]
lay = {x: st.median(v) for x, v in sorted(by_layer.items())}
dif = []
for r in R:
    for k, v in r["stepped"].items():
        dif.append((abs(r["native_batched"][k] - v), r["spacing"]))
n4 = sum(1 for d, s in dif if d <= 2 * s)
set_ratio = [r["forecast"]["set"] / r["native_batched"]["set"] for r in R if r["native_batched"]["set"]]
summary = {
    "n_B": n,
    "N1": {"named_beats_random_by_0.5": f"{n1a}/{n}", "named_beats_answer_null": f"{n1b}/{n}",
           "holds": n > 0 and n1a / n >= 0.75 and n1b / n >= 0.69,
           "median_dM_named": st.median([r["native_batched"]["set"] for r in R]),
           "median_dM_random": st.median([r["native_batched"]["random"] for r in R]),
           "median_dM_answer_null": st.median([r["native_batched"]["answer_null"] for r in R])},
    "N2": {"neurons_above_floor": len(fc), "spearman": spearman(fc, nat) if len(fc) > 2 else None,
           "median_ratio": st.median(ratio) if ratio else None,
           "holds": len(fc) > 2 and spearman(fc, nat) >= 0.8 and 0.4 <= st.median(ratio) <= 0.7,
           "set_forecast_over_native_median": st.median(set_ratio) if set_ratio else None},
    "N3": {"median_ratio_by_layer": lay,
           "holds": lay.get(20, 1) <= 0.5 and all(lay.get(x, 0) >= 0.8 for x in (26, 27))},
    "N4": {"batched_minus_stepped_within_2_spacings": f"{n4}/{len(dif)}", "holds": bool(dif) and n4 / len(dif) >= 0.9,
           "max_abs": max(d for d, _ in dif) if dif else None},
    "falsifier_fires": not (n > 0 and n1a / n >= 0.75 and n1b / n >= 0.69),
}
json.dump(summary, open(sys.argv[2], "w"), indent=1)
print(json.dumps(summary, indent=1))
