"""B1 donors per prereg_b1.json arms b and c (item metadata only: family, language, bridge,
prompt length, span length; X1 donor lists). python b1_donors.py > b1_donors.json"""
import json

T = json.load(open("b1_sets.json"))["truths"]
x1 = {r["id"]: r for r in json.load(open("../ext/x1-17-depth-panel.json"))["panel"]}
ids = sorted(T)


def nearest(i, pool):
    pool = sorted(pool, key=lambda k: (abs(T[k]["n"] - T[i]["n"]), k))
    return pool[0] if pool else None


out = {}
for i in ids:
    t = T[i]
    fam = [k for k in ids if T[k]["family"] == t["family"] and k != i]
    b = None
    if i in x1 and x1[i]["donors"]:
        d = x1[i]["donors"][0]
        if d in T and T[d]["language"] != t["language"]:
            b = d
    if b is None:
        b = nearest(i, [k for k in fam if T[k]["language"] != t["language"]])
    rec = {"cell": b}
    if t["family"] in ("d3-lm", "d3-per", "d4-work"):
        same_len = [k for k in fam if len(T[k]["span"]) == len(t["span"])]
        rec["e1"] = nearest(i, [k for k in same_len if T[k]["language"] != t["language"]])
        rec["control"] = nearest(i, [k for k in same_len if T[k]["bridge"] == t["bridge"]])
    out[i] = rec
json.dump(out, open("/dev/stdout", "w"), indent=1)
