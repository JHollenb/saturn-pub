"""Score the follow-up (prereg.json).  python score.py OUTDIR -> OUTDIR/../summary.json (+ printed tables)
Reads OUTDIR/score-full*.jsonl, and das/out/score-full*.jsonl for the 11b reproduction check."""
import glob
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from mdb.analysis.subspace import iia  # noqa: E402

OUT = Path(sys.argv[1])
R = {}
for f in sorted(glob.glob(str(OUT / "score-full*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        R[r["id"]] = r
R = [R[k] for k in sorted(R)]
old = {}
for f in glob.glob(str(Path(__file__).resolve().parent.parent / "out" / "score-full*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        old[r["id"]] = r
arms = max((list(r["arms"]) for r in R), key=len)
for r in R:      # a pair with no active census feature has an empty exp7 SAE set: empty subspace = no change (M = M0), as in 11b
    for a in arms:
        if a not in r["arms"]:
            assert a == "sae" or a.startswith("saeC"), a
            r["arms"][a] = {"M": r["M0"], "argmax": None, "effect_floors": 0.0, "k": 0, "mass": 0.0}
tab = {}
for a in arms:
    recs = [{"M0": r["M0"], "floor": r["floor"], "M": r["arms"][a]["M"]} for r in R]
    t = iia(recs)
    mv = [(r["M0"] - r["arms"][a]["M"]) / (r["M0"] - r["arms"]["whole"]["M"]) for r in R if r["M0"] != r["arms"]["whole"]["M"]]
    tab[a] = {"flips1": t["margin1"]["flips"], "flips4": t["margin4"]["flips"], "elig1": t["margin1"]["eligible"],
              "elig4": t["margin4"]["eligible"], "moved_median": st.median(mv),
              "mass_median": st.median(r["arms"][a]["mass"] for r in R),
              "k_median": st.median(r["arms"][a]["k"] for r in R)}
repro = {}
for k in (1, 2, 4, 8, 16, 32, 64):
    key = f"symtab{k}"
    if key in next(iter(old.values()))["arms"] or True:
        diffs = [abs(r["arms"][f"full69_k{k}"]["M"] - old[r["id"]]["arms"][key]["M"]) for r in R
                 if key in old.get(r["id"], {}).get("arms", {})]
        repro[f"k{k}"] = {"n": len(diffs), "max_abs_dM": max(diffs) if diffs else None}
repro["whole_max"] = max(abs(r["arms"]["whole"]["M"] - old[r["id"]]["arms"]["whole"]["M"]) for r in R)
repro["sae_max"] = max(abs(r["arms"]["sae"]["M"] - old[r["id"]]["arms"]["sae"]["M"]) for r in R)
repro["branches_M0_max"] = max(abs(r["branches_M0"] - r["M0"]) for r in R)
repro["img_dev_max"] = max(r["img_dev"] for r in R)
rk = {json.dumps(r["rankA"]) for r in R}, {json.dumps(r["rankB"]) for r in R}
repro["rank_A_unique"], repro["rank_B_unique"] = len(rk[0]), len(rk[1])
s = {"pairs": len(R), "arms": tab, "reproduction_vs_11b": repro, "rankA": R[0]["rankA"], "rankB": R[0]["rankB"],
     "script_sha256": sorted({r["script_sha256"] for r in R}), "prereg_sha256": sorted({r["prereg_sha256"] for r in R}),
     "vocab_sha256": sorted({r["vocab_sha256"] for r in R}), "labels": "MEASURED, not audited"}
json.dump(s, open(OUT.parent / "summary.json" if OUT.name == "out" else OUT / "summary.json", "w"), indent=1)


def row(prefix, ks):
    return " | ".join(f"{tab[f'{prefix}_k{k}']['flips1']} / {tab[f'{prefix}_k{k}']['flips4']} ({tab[f'{prefix}_k{k}']['moved_median']:.2f})"
                      for k in ks)


print("pairs", len(R), "whole", tab["whole"]["flips1"], tab["whole"]["flips4"], "sae", tab["sae"]["flips1"], tab["sae"]["flips4"], tab["sae"]["moved_median"])
print("Q1 k=8,16,32,64")
for p in ("full69", "a69", "aall", "lto"):
    print(p, "|", row(p, (8, 16, 32, 64)))
print("Q2 k=1,2,4,8,16,32")
for p in ("full69", "saeA", "saeB", "saeC"):
    print(p, "|", row(p, (1, 2, 4, 8, 16, 32)))
print("mass k=32:", {p: round(tab[f"{p}_k32"]["mass_median"], 2) for p in ("full69", "a69", "aall", "lto", "saeA", "saeB", "saeC")})
print("repro", json.dumps(repro))
