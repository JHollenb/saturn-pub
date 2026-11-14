"""Chain-readability tables from y2-*.jsonl (python3 y2_summary.py [GLOB]); writes y2-summary.json.

Definitions are those of prereg_hard.json. Wrong-capital error: the answer token is the capital
token of some country in the candidate sets (capital_tokens of candidates.json plus the
candidates_ext capitals); its country is that country.
"""
import glob
import json
import statistics as st
import sys
from collections import defaultdict

pattern = sys.argv[1] if len(sys.argv) > 1 else "y2-*.jsonl"
rows = [json.loads(line) for p in sorted(glob.glob(pattern)) for line in open(p)]
base = json.load(open("../candidates.json"))
cap_of = {v: k for k, v in base["capital_tokens"].items()}
ext = {i["id"]: i for i in json.load(open("candidates_ext.json"))["items"]}
hub_of_country = {i["bridge"]: i["country_token"] for i in ext.values()}


def rate(xs):
    xs = list(xs)
    return [sum(xs), len(xs), round(sum(xs) / len(xs), 3) if xs else None]


fams = defaultdict(list)
for r in rows:
    fams[r["family"]].append(r)
fams["ALL"] = rows
out = {}
for fam, rs in sorted(fams.items()):
    ok = [r for r in rs if r["correct"]]
    bad = [r for r in rs if not r["correct"]]
    d = {"n": len(rs), "accuracy": rate(r["correct"] for r in rs)}
    for k in ("bridge", "const", "mass", "lens"):
        d[f"readable_{k}"] = rate(r[f"{k}_named_ok"] for r in ok)
    d["bridge_site_L20_KV6"] = rate(r["bridge_site"][1:] == [20, 6] for r in ok)
    d["bridge_row_is_r0"] = rate(r["bridge_site"][0] == ext[r["id"]]["row"] for r in ok)
    fa = [r["first_appearance"] for r in ok if r["bridge_named_ok"] and r["first_appearance"] is not None]
    d["first_appearance_median"] = st.median(fa) if fa else None
    d["first_appearance_le_20"] = rate(x <= 20 for x in fa)
    wc, nc = [], []
    for r in bad:
        country = cap_of.get(r["answer_id"])
        if country is not None and country != r["bridge"]:
            hub = hub_of_country.get(country)
            wc.append(hub is not None and r["bridge_named"].strip() == country)
        else:
            nc.append(r["bridge_named_ok"])
    d["errors_wrong_capital_view_names_answer_country"] = rate(wc)
    d["errors_other_view_names_true_bridge"] = rate(nc)
    causal = [c for r in ok for c in r.get("causal", [])]
    if causal:
        d["causal_pairs"] = len(causal)
        d["causal_content_donor"] = rate(c["content_donor"] for c in causal)
        d["causal_answer_donor"] = rate(c["answer_donor"] for c in causal)
        d["causal_together"] = rate(c["content_donor"] == c["answer_donor"] for c in causal)
        d["causal_2x2 (content_donor, answer_donor)"] = {
            f"{a},{b}": sum(1 for c in causal if c["content_donor"] == a and c["answer_donor"] == b)
            for a in (True, False) for b in (True, False)}
    hits = [r["hits"] for r in ok if "hits" in r]
    if hits:
        for k in ("view", "F1", "F3", "const", "mass_view"):
            d[f"gt_hit_{k}"] = rate(h.get(k, False) for h in hits)
        d["gt_sites"] = sorted({tuple(r["gt"]["site"][1:]) for r in ok if "gt" in r})[:6]
        d["gt_effect_median"] = st.median(r["gt"]["effect"] for r in ok if "gt" in r)
    out[fam] = d
json.dump(out, open("y2-summary.json", "w"), indent=1)
for fam, d in out.items():
    print(fam, json.dumps(d))
