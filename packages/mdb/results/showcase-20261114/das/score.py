"""Score DAS (prereg.json).  python score.py OUTDIR  -> OUTDIR/../summary.json (printed)

Reads OUTDIR/score-*.jsonl (mdb stepped arms per test pair), OUTDIR/bases.pt (bases + DAS
training info) and OUTDIR/admit.json. IIA at margin 1 (exp7 floor) and margin 4 (audit), the
fraction of whole-cell IIA, the median fraction of the whole-cell margin change moved, principal
angles and the projection kernel (normalised: sum cos^2 / sqrt(k_a k_b)), and the pyvene-vs-mdb
flip agreement for DAS.
"""
import glob
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from mdb.analysis.subspace import principal_angles  # noqa: E402

OUT = Path(sys.argv[1])
R = {}
for f in sorted(glob.glob(str(OUT / "score-full*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        R[r["id"]] = r
R = [R[k] for k in sorted(R)]
B = torch.load(OUT / "bases.pt", weights_only=False)
KS = B["meta"]["KS"]
admit = json.load(open(OUT / "admit.json"))
arms = list(R[0]["arms"])


def flip(r, arm, m):
    return r["M0"] > m * r["floor"] and r["arms"][arm]["M"] < -m * r["floor"]


iia = {}
for m in (1, 4):
    elig = [r for r in R if r["M0"] > m * r["floor"]]
    whole = sum(flip(r, "whole", m) for r in elig)
    iia[f"margin{m}"] = {"eligible": len(elig), "whole": whole, "arms": {
        a: {"flips": sum(flip(r, a, m) for r in elig),
            "frac_of_whole": (sum(flip(r, a, m) for r in elig) / whole) if whole else None} for a in arms}}
moved = {a: st.median([(r["M0"] - r["arms"][a]["M"]) / (r["M0"] - r["arms"]["whole"]["M"])
                       for r in R if r["M0"] != r["arms"]["whole"]["M"]]) for a in arms}


def pk(a, b):
    ang = torch.tensor(principal_angles(a, b), dtype=torch.float64)
    return float(torch.cos(torch.deg2rad(ang)).pow(2).sum() / (a.shape[1] * b.shape[1]) ** 0.5)


angles = {}
for k in KS:
    d = B[f"das{k}"].double()
    row = {}
    for fam in ("symtab", "symcap", "meandiff", "random"):
        q = B[f"{fam}{k}"].double()
        a = principal_angles(d, q)
        row[fam] = {"min_deg": min(a), "median_deg": st.median(a), "max_deg": max(a), "pk_norm": pk(d, q)}
    sae = [torch.tensor(r["sae_basis"], dtype=torch.float64) for r in R if r.get("sae_rank")]
    mins = [min(principal_angles(d, q)) for q in sae]
    pks = [pk(d, q) for q in sae]
    row["sae_per_pair"] = {"median_min_deg": st.median(mins), "median_pk_norm": st.median(pks),
                           "n": len(sae)}
    if k > 1:
        prev = B[f"das{KS[KS.index(k) - 1]}"].double()
        row["das_prev_nested_min_deg"] = max(principal_angles(prev, d))       # prev inside k?
    angles[f"k{k}"] = row
dasinfo = {}
mdb_by = {r["id"]: r for r in R}
tests = sorted(mdb_by)
for k in KS:
    info = B[f"das{k}_info"]
    fit = info["train_probe_M"]
    pv = info["pyvene_test_M"]          # in test_pairs() order = sorted base ids
    agree = sum((pv[i] < -mdb_by[t]["floor"]) == flip(mdb_by[t], f"das{k}", 1) for i, t in enumerate(tests))
    dasinfo[f"k{k}"] = {"train_probe_flips_M_lt_0": f"{sum(m < 0 for m in fit)}/{len(fit)}",
                        "final_loss_mean_last20": st.mean(info["loss_curve"][-20:]),
                        "pyvene_test_flips": sum(pv[i] < -mdb_by[t]["floor"] and mdb_by[t]["M0"] > mdb_by[t]["floor"]
                                                 for i, t in enumerate(tests)),
                        "pyvene_mdb_same_verdict": f"{agree}/{len(tests)}",
                        "pyvene_mdb_max_abs_M_diff": max(abs(pv[i] - mdb_by[t]["arms"][f"das{k}"]["M"]) for i, t in enumerate(tests)),
                        "orth_err": info["orth_err"], "seconds": info["seconds"]}
s = {"pairs": len(R), "train_items_admitted": sum(a["correct"] for a in admit if a["split"] == "train"),
     "train_items": sum(a["split"] == "train" for a in admit),
     "train_countries": len({a["bridge"] for a in admit if a["split"] == "train" and a["correct"]}),
     "das_pairs": B["meta"]["pairs"], "KS": KS, "IIA": iia, "median_frac_moved": moved,
     "sae_rank_median": st.median([r.get("sae_rank", 0) for r in R]),
     "angles": angles, "das": dasinfo,
     "symtab_explained": B["meta"]["symtab_sv"][:8], "n_country_tokens": len(B["meta"]["country_tokens"])}
json.dump(s, open(OUT.parent / "summary.json" if OUT.name == "out" else OUT / "summary.json", "w"), indent=1)
print(json.dumps({"IIA": iia, "moved": moved}, indent=1))
