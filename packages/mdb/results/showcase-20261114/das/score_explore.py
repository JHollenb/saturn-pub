"""Score the exploratory addendum (prereg_explore.json).  python score_explore.py OUTDIR -> summary_explore.json
Reads OUTDIR/score-explore*.jsonl, bases.pt + bases-indist.pt; same flip rule as score.py."""
import glob
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from mdb.analysis.subspace import principal_angles  # noqa: E402

OUT = Path(sys.argv[1])
M = {}
for f in sorted(glob.glob(str(OUT / "score-explore*.jsonl"))):      # several jobs: merge arms per pair
    for line in open(f):
        r = json.loads(line)
        if r["id"] in M:
            assert M[r["id"]]["M0"] == r["M0"] and M[r["id"]]["arms"]["whole"]["M"] == r["arms"]["whole"]["M"]
            M[r["id"]]["arms"].update(r["arms"])
        else:
            M[r["id"]] = r
R = [M[k] for k in sorted(M)]
B = torch.load(OUT / "bases.pt", weights_only=False)
X = torch.load(OUT / "bases-indist.pt", weights_only=False)
B.update({k: v for k, v in X.items() if k != "meta"})
arms = list(R[0]["arms"])
flip = lambda r, a, m: r["M0"] > m * r["floor"] and r["arms"][a]["M"] < -m * r["floor"]  # noqa: E731
iia = {}
for m in (1, 4):
    el = [r for r in R if r["M0"] > m * r["floor"]]
    w = sum(flip(r, "whole", m) for r in el)
    iia[f"margin{m}"] = {"eligible": len(el), "whole": w, "arms": {a: {"flips": sum(flip(r, a, m) for r in el),
                         "frac_of_whole": sum(flip(r, a, m) for r in el) / w} for a in arms}}
moved = {a: st.median([(r["M0"] - r["arms"][a]["M"]) / (r["M0"] - r["arms"]["whole"]["M"])
                       for r in R if r["M0"] != r["arms"]["whole"]["M"]]) for a in arms}


def pk(a, b):
    ang = torch.tensor(principal_angles(a, b), dtype=torch.float64)
    return float(torch.cos(torch.deg2rad(ang)).pow(2).sum() / (a.shape[1] * b.shape[1]) ** 0.5)


ang = {}
for k in X["meta"]["KS"]:
    d = B[f"dasin{k}"].double()
    row = {}
    for other in (f"symtab{k}", f"das{k}", f"meanin{k}", f"random{k}"):
        if other in B:
            a = principal_angles(d, B[other].double())
            row[other] = {"min_deg": min(a), "median_deg": st.median(a), "pk_norm": pk(d, B[other].double())}
    info = X[f"dasin{k}_info"]
    row["train_probe_flips"] = f"{sum(m < 0 for m in info['train_probe_M'])}/{len(info['train_probe_M'])}"
    row["final_loss_mean_last20"] = st.mean(info["loss_curve"][-20:])
    row["pyvene_test_M_lt_0"] = sum(m < 0 for m in info["pyvene_test_M"])
    ang[f"k{k}"] = row
s = {"pairs": len(R), "status": "EXPLORATORY (prereg_explore.json, unsealed, in-sample for dasin/meanin)",
     "IIA": iia, "median_frac_moved": moved, "dasin": ang, "indist_pairs": X["meta"]["pairs"]}
json.dump(s, open(Path(__file__).resolve().parent / "summary_explore.json", "w"), indent=1)
print(json.dumps({m: {a: v["flips"] for a, v in iia[m]["arms"].items()} for m in iia}, indent=0))
