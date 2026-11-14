"""Cut this example's data and showcase reference from the saved showcase outputs (light Python,
no model).  python make_subset.py   (run from this directory; writes items.json, reference.json)

Pairs: the first 34 of exp7's 64 Run 1a base pairs in id order (16 city, 16 currency, 2 landmark).
They hold the one pair where pyvene and mdb disagree on the flip verdict (currency:rupiah) and
all 4 pairs the SAE subspace flips.
"""
import hashlib
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent / "results"
SC = RES / "showcase-20261114"
N = 34

spec = json.load(open(RES / "twohop-20261113/next/r1_items.json"))
census = json.load(open(SC / "exp1/census.json"))
vocab = json.load(open(SC / "das/followup/vocab.json"))
exp7 = [json.loads(line) for line in open(SC / "exp7/out/exp7-items.jsonl")]
fu = {}
for k in range(4):
    for line in open(SC / f"das/followup/out/score-full{k}.jsonl"):
        r = json.loads(line)
        fu[r["id"]] = r

base = sorted([i for i in spec["items"] if i["group"] == "base"], key=lambda i: i["id"])
by_id = {i["id"]: i for i in base}
assert [r["id"] for r in exp7] == [i["id"] for i in base]          # same order as the showcase
chosen = base[:N]


def slim(item):
    return {"id": item["id"], "ids": item["ids"], "span": item["span"], "answer": item["answer"],
            "country_token": item.get("country_token")}


pairs = []
for item in chosen:
    d = item["donor"]
    donor = by_id.get(d["id"]) or {"id": d["id"], "ids": d["ids"], "span": [d["row"]],
                                   "answer": d["answer"], "country_token": d.get("country_token")}
    pairs.append({"recipient": slim(item), "donor": slim(donor)})
features = sorted(int(f) for f in census["features"])
items = {"source": "results/twohop-20261113/next/r1_items.json (group base, sorted by id), "
                   f"first {N} of 64",
         "census_features": features,
         "census_sha256": hashlib.sha256(open(SC / "exp1/census.json", "rb").read()).hexdigest(),
         "readout_tokens": {"full69": vocab["full69"], "a69": vocab["a69"]},
         "pairs": pairs}
json.dump(items, open(HERE / "items.json", "w"))

rows = []
for r in exp7[:N]:
    f = fu[r["id"]]
    assert f["M0"] == r["M0"] and f["floor"] == r["floor"]
    rows.append({"id": r["id"], "donor": r["donor"], "crow": r["crow"], "donor_crow": r["donor_crow"],
                 "M0": r["M0"], "floor": r["floor"], "rank": r["rank"], "features": r["features"],
                 "whole": r["arms"]["whole"]["M"], "sae": r["arms"]["sae"]["M"],
                 "random": r["arms"]["random"]["M"], "pyvene": r["pyvene"]["M"],
                 "pyvene_clean": r["pyvene"]["M_clean"], "branches_M0": f["branches_M0"],
                 "full69_k32": f["arms"]["full69_k32"]["M"], "a69_k32": f["arms"]["a69_k32"]["M"]})


def flips(arm, m=1.0):
    return sum(1 for r in rows if r["M0"] > m * r["floor"] and r[arm] < -m * r["floor"])


frac = {arm: st.median([(r["M0"] - r[arm]) / (r["M0"] - r["whole"]) for r in rows
                        if r["M0"] != r["whole"]]) for arm in ("sae", "random")}
agree = sum((r["whole"] < -r["floor"]) == (r["pyvene"] < -r["floor"]) for r in rows)
gap = [abs(r["pyvene"] - r["whole"]) for r in rows]
kern = [abs(r["pyvene_clean"] - r["M0"]) for r in rows]
summary = {"pairs": len(rows), "eligible": sum(r["M0"] > r["floor"] for r in rows),
           "flips": {a: flips(a) for a in ("whole", "pyvene", "sae", "random", "full69_k32", "a69_k32")},
           "flips_margin4": {a: flips(a, 4.0) for a in ("whole", "pyvene", "sae", "random",
                                                       "full69_k32", "a69_k32")},
           "pyvene_mdb_verdict_agree": agree, "median_rank": st.median(r["rank"] for r in rows),
           "max_rank": max(r["rank"] for r in rows),
           "median_frac_moved": frac,
           "pyvene_mdb_gap": {"median": st.median(gap), "max": max(gap)},
           "kernel_path_clean_gap": {"median": st.median(kern), "max": max(kern)}}
json.dump({"source": "results/showcase-20261114/exp7/out/exp7-items.jsonl (job-c8144767543d); "
                     "readout arms from das/followup/out/score-full*.jsonl (job-8b01f1a12d84, "
                     "job-33a4b67a28eb, job-60bce10afa0c, job-70ebd29d8f10)",
           "summary": summary, "pairs": rows}, open(HERE / "reference.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
