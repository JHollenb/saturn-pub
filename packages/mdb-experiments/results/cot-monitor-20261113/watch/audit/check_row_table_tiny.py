"""Auditor check: rows.row_table chunked masked ablation vs mdb's stepped per-row p-zero, for
EVERY row of the tiny fp32 model (not only the 3 rows the builder's test samples), several chunk
sizes, including rows inside the ablated columns. Also: suffix_sweep vs stepped at every k.
Run: PYTHONPATH=<wt>/packages/mdb/src:<wt>/src common exec python audit/check_row_table_tiny.py
"""
import json

import torch

from mdb.analysis import probe, rows
from mdb.ops.debugger import open_tiny

IDS = [(i * 7) % 31 + 1 for i in range(28)]
LET = [20, 21, 22, 23]
NP = 14
SUF = [25, 26, 27]
B = {"sink": [0], "question": list(range(1, 6)), "hint": list(range(6, 10)),
     "template": list(range(10, NP))}

torch.manual_seed(0)
dbg = open_tiny("qwen2", "fp32")


def stepped(seq, zero=()):
    run = dbg.run(seq)
    if zero:
        for layer in range(run.adapter.layers):
            run.until(f"L{layer}.h")
            m = torch.zeros(run.address_shape(f"L{layer}.p"), dtype=torch.bool)
            m[..., list(zero)] = True
            run.set(f"L{layer}.p", 0.0, "replace", mask=m)
    run.until("logits")
    return run.read("logits").reshape(-1)[LET].double()


out = {}
run = dbg.run(IDS)
binds = dict(B, chain=list(range(NP, len(IDS))))
ref_nat = {t: stepped(IDS[: t + 1]) for t in range(1, len(IDS))}
ref_abl = {t: stepped(IDS[: t + 1], [c for c in B["hint"] if c <= t]) for t in range(1, len(IDS))}
for chunk in (1, 3, 5, 512):
    tr = rows.row_table(run, binds, letters=LET, X="B", R="C", arms={"hint": B["hint"]},
                        chunk=chunk)
    worst_out, worst_in, worst_nat = 0.0, 0.0, 0.0
    for t in range(1, len(IDS)):
        rec = tr.records[t]
        a = float(ref_abl[t][1] - ref_abl[t][2])
        n = float(ref_nat[t][1] - ref_nat[t][2])
        worst_nat = max(worst_nat, abs(rec["margin"] - n))
        d = abs(rec["ablated"]["hint"] - a)
        if t in B["hint"]:
            worst_in = max(worst_in, d)
        elif t > max(B["hint"]):
            worst_out = max(worst_out, d)
    out[f"chunk{chunk}"] = {"native_max": worst_nat, "abl_rows_after_hint_max": worst_out,
                            "abl_rows_inside_hint_max": worst_in}
# suffix_sweep at every k
chain_len = len(IDS) - NP
tr = probe.suffix_sweep(run, list(range(chain_len + 1)), SUF, n_prompt=NP, bindings=B,
                        letters=LET, X="B", R="A", null_steps=tuple(range(chain_len + 1)))
wn = wm = wc = 0.0
for rec in tr.records:
    k = rec["step"]
    seq = IDS[: NP + k] + SUF
    nat = stepped(seq)
    wn = max(wn, abs(rec["margin"] - float(nat[1] - nat[0])))
    m2 = stepped(seq, B["hint"])
    wm = max(wm, abs(rec["ablated"]["hint"] - float(m2[1] - m2[0])))
    if k:
        cm = stepped(seq, list(range(NP, NP + k)))
        wc = max(wc, abs(rec["ablated"]["chain_so_far"] - float(cm[1] - cm[0])))
out["suffix_sweep_all_k"] = {"native_max": wn, "M2_max": wm, "CM1_max": wc,
                             "null_vs_fresh": tr.meta["null_vs_fresh_max_abs"]}
print(json.dumps(out, indent=1))
