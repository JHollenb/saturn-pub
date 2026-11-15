"""Feed the DAS result back into mdb as objects: one ``SubspaceBinding`` per arm and k (the cell
L20 KV6 country subspace, a row-free template bound per item by the carrier rule), with its basis
authority, provenance (jobs, data digests) and the measured IIA attached as labels.

python bind_results.py OUTDIR  -> das/subspaces.json (list of SubspaceBinding.to_json())
CPU only (reads bases.pt and the scored jsonl).
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from mdb.analysis.subspace import subspace_binding  # noqa: E402

OUT = Path(sys.argv[1])
B = torch.load(OUT / "bases.pt", weights_only=False)
EX = OUT / "bases-indist.pt"
if EX.exists():
    B.update({k: v for k, v in torch.load(EX, weights_only=False).items() if k != "meta"})
summary = json.load(open(HERE / "summary.json"))
ex = json.load(open(HERE / "summary_explore.json")) if (HERE / "summary_explore.json").exists() else None
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
COMMON = {"model": "Qwen3-1.7B-Base bf16", "cell_rule": "L20 KV6 value at the frozen R_argmax carrier row",
          "prereg_sha256": sha(HERE / "prereg.json"), "driver_sha256": sha(HERE / "das.py"),
          "bases_sha256": sha(OUT / "bases.pt")}
AUTH = {"das": "fitted", "dasin": "fitted", "meandiff": "fitted", "meanin": "fitted",
        "symtab": "derived-exact", "symcap": "derived-exact", "random": "supplied"}
WHAT = {"das": "pyvene LowRankRotatedSpaceIntervention trained on 87 items / 38 countries disjoint from the test countries (job-14bfa70f5ddb)",
        "dasin": "pyvene DAS trained in-sample on the 74 test-country items, exp7 pairs excluded (exploratory, job-af6bb2400d89)",
        "meandiff": "PCA of centred per-country class means of training-item cells (job-0c288755878c cells)",
        "meanin": "PCA of centred per-answer class means of test-country cells (exploratory, in-sample)",
        "symtab": "PCA of W_g^T(gamma_F * W_U[country token]) over 69 country tokens (training-free)",
        "symcap": "PCA of W_g^T(gamma_F * W_U[capital token]) over 70 capital tokens (training-free)",
        "random": "QR of a seed-20261114 Gaussian"}
out = []
for key, val in B.items():
    fam = key.rstrip("0123456789")
    if fam not in AUTH or not key[len(fam):].isdigit():
        continue
    sb = subspace_binding(f"country-{key}", 20, 6, val.double(), authority=AUTH[fam],
                          provenance={**COMMON, "method": WHAT[fam], "arm": key},
                          orthonormalize=fam not in ("das", "dasin"))
    for src in (summary, ex):
        if src and key in src["IIA"]["margin1"]["arms"]:
            sb.attach("iia", {m: {"flips": src["IIA"][m]["arms"][key]["flips"],
                                  "eligible": src["IIA"][m]["eligible"], "whole_flips": src["IIA"][m]["whole"]}
                              for m in ("margin1", "margin4")})
            sb.attach("median_frac_of_whole_margin_moved", src["median_frac_moved"][key])
    out.append(sb.to_json())
json.dump(out, open(HERE / "subspaces.json", "w"))
print(len(out), "subspace bindings ->", HERE / "subspaces.json")
