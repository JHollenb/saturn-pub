"""FT1: first-token / sink test (next/prereg_firsttoken.json).

python ft1.py MODEL OUT [PICK]

Input: ft_items.json. One model load. Per item: (a) contrast vs share per source row at the answer
row; (b) first-word swaps; (c) zero and mask cuts of t0 at the answer row (all layers, and one
layer L* with the P09 closed forms), plus matched-mass random-row cuts.
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

from mdb.analysis.finite import f64
from mdb.analysis.forecast import Margin
from mdb.analysis.reads import head_mix, readout
from mdb.repl.__main__ import _load

MODEL, OUT = sys.argv[1:3]
PICK = sys.argv[3] if len(sys.argv) > 3 else ""
T0 = time.monotonic()
torch.set_grad_enabled(False)
dbg = _load(Path(MODEL), "bf16", "cuda", True, None)
tok = dbg.tokenizer
L, H = dbg.adapter.layers, dbg.adapter.num_heads
items = json.load(open("ft_items.json"))["items"]
if PICK:
    items = [items[int(k)] for k in PICK.split(",")]
OUTF = open(f"{OUT}.jsonl", "w")
WORDS = ("A", "In", "Which")
NEG = -1e4
print(f"load {time.monotonic() - T0:.1f}s; {len(items)} items", flush=True)


def ids_of(text):
    return tok(text, add_special_tokens=False)["input_ids"]


def first_tok(word):
    return ids_of(" " + word)[0]


def native(ids):
    run = dbg.run(ids, name="nat")
    j = len(ids) - 1
    run.until(f"t{j}.logits")
    return run, run.read().reshape(-1).float()


def cut(ids, kind, rows_by_layer, layers):
    """rows_by_layer[l] = list over heads of source-row lists to cut at the answer row."""
    j = len(ids) - 1
    run = dbg.run(ids, name="cut")
    for x in layers:
        run.until(f"t{j}.L{x}.h")
        port = "p" if kind == "zero" else "s"
        shape = tuple(run.address_shape(f"t{j}.L{x}.{port}"))
        mask = torch.zeros(shape, dtype=torch.bool)
        for h, rows in enumerate(rows_by_layer[x]):
            if rows:
                mask[0, h, 0, rows] = True
        if not bool(mask.any()):
            continue
        run.set(f"t{j}.L{x}.{port}", 0.0 if kind == "zero" else NEG, "replace", mask=mask)
    run.until(f"t{j}.logits")
    return run, run.read().reshape(-1).float()


def spacing(x):
    x = abs(x)
    return 2.0 ** (math.floor(math.log2(x)) - 7) if x > 0 else 2.0 ** -133


for k, item in enumerate(items):
    t_item = time.monotonic()
    ids = item["ids"]
    j = len(ids) - 1
    rec = {"id": item["id"], "group": item["group"], "family": item["family"],
           "tokenizer_matches": ids_of(item["prompt"]) == ids, "t0_text": tok.decode([ids[0]])}
    run, logits = native(ids)
    t1, t2 = [int(x) for x in logits.topk(2).indices]
    rec["argmax"], rec["argmax_text"] = t1, tok.decode([t1])
    rec["runner"], rec["runner_text"] = t2, tok.decode([t2])
    rec["M0"] = float(logits[t1] - logits[t2])
    reader = readout(run, Margin(j, t1, t2))
    c = f64(reader.vector).reshape(-1)
    n = len(ids)
    share_row = torch.zeros(n, dtype=torch.float64)
    contrast_row = torch.zeros(n, dtype=torch.float64)
    share_t0 = torch.zeros(L, H, dtype=torch.float64)
    contrast_t0 = torch.zeros(L, H, dtype=torch.float64)
    p_all = torch.zeros(L, H, n, dtype=torch.float64)
    for x in range(L):
        mix = head_mix(run, j, x, {})
        cp = (mix.o_parts() @ c.to(mix.o_parts().device)).double().cpu()     # (H, n)
        p = f64(mix.p).reshape(H, -1)[:, :n]
        tot = cp.sum(1, keepdim=True)
        con = cp - p * tot
        share_row += cp.sum(0)
        contrast_row += con.sum(0)
        share_t0[x], contrast_t0[x] = cp[:, 0], con[:, 0]
        p_all[x] = p

    def rank_of_t0(v):
        order = v.abs().argsort(descending=True).tolist()
        return order.index(0) + 1
    rec["a"] = {"t0_share": float(share_row[0]), "t0_contrast": float(contrast_row[0]),
                "t0_rank_share": rank_of_t0(share_row), "t0_rank_contrast": rank_of_t0(contrast_row),
                "t0_mass": float(p_all[:, :, 0].sum()), "mass_total": float(p_all.sum()),
                "top_share_rows": [[int(r), tok.decode([ids[r]]), float(share_row[r])]
                                   for r in share_row.abs().argsort(descending=True)[:5]],
                "top_contrast_rows": [[int(r), tok.decode([ids[r]]), float(contrast_row[r])]
                                      for r in contrast_row.abs().argsort(descending=True)[:5]]}
    # (b) first-word swaps
    words = item["prompt"].split(" ", 1)
    swaps = {}
    for w in WORDS:
        v_ids = ids_of(w + " " + words[1])
        _, lg = native(v_ids)
        am = int(lg.argmax())
        swaps[w] = {"argmax": am, "text": tok.decode([am]), "follows": am == first_tok(w),
                    "unchanged": am == t1}
    lines = item["prompt"].split("\n")
    both = [("A " + ln.split(" ", 1)[1]) if i < 2 and ln.startswith("The ") else ln
            for i, ln in enumerate(lines)]
    _, lg = native(ids_of("\n".join(both)))
    am = int(lg.argmax())
    swaps["A_both_lines"] = {"argmax": am, "text": tok.decode([am]), "follows": am == first_tok("A"),
                             "unchanged": am == t1, "exploratory": True}
    rec["b"] = swaps
    # (c) sink cuts, all layers
    sink = {x: [[0] for _ in range(H)] for x in range(L)}
    rng = random.Random(k)
    exclude = {0, j, *item["span"]}
    pool = [r for r in range(1, j) if r not in exclude]
    rand, achieved = {}, []
    for x in range(L):
        rand[x] = []
        for h in range(H):
            order = pool[:]
            rng.shuffle(order)
            got, rows = 0.0, []
            for r in order:
                if got >= float(p_all[x, h, 0]):
                    break
                rows.append(r)
                got += float(p_all[x, h, r])
            rand[x].append(rows)
            achieved.append(got / float(p_all[x, h, 0]) if float(p_all[x, h, 0]) else 1.0)
    cuts = {}
    for name, rows in (("t0", sink), ("random", rand)):
        for kind in ("zero", "mask"):
            _, lg = cut(ids, kind, rows, range(L))
            am = int(lg.argmax())
            cuts[f"{name}_{kind}"] = {"argmax": am, "text": tok.decode([am]),
                                      "moved": am != t1, "dM": float(lg[t1] - lg[t2]) - rec["M0"]}
    cuts["random_mass_ratio_median"] = sorted(achieved)[len(achieved) // 2]
    # single layer L*: closed forms
    ls = int(share_t0.sum(1).abs().argmax())
    o_nat = f64(run.read(f"t{j}.L{ls}.o")).reshape(-1)
    pred = {"zero": float(-share_t0[ls].sum()),
            "mask": float(sum(-contrast_t0[ls, h] / (1 - p_all[ls, h, 0]) for h in range(H)))}
    single = {"layer": ls}
    for kind in ("zero", "mask"):
        er, lg = cut(ids, kind, sink, [ls])
        meas = float(c @ (f64(er.read(f"t{j}.L{ls}.o")).reshape(-1) - o_nat))
        single[kind] = {"predicted": pred[kind], "measured": meas, "error": meas - pred[kind],
                        "within_2_spacings": abs(meas - pred[kind]) <= 2 * spacing(meas),
                        "dM": float(lg[t1] - lg[t2]) - rec["M0"], "argmax_text": tok.decode([int(lg.argmax())])}
    cuts["single_layer"] = single
    rec["c"] = cuts
    rec["seconds"] = round(time.monotonic() - t_item, 2)
    OUTF.write(json.dumps(rec) + "\n")
    OUTF.flush()
    print("FT " + json.dumps({"id": item["id"], "g": item["group"], "am": rec["argmax_text"],
                              "rank": [rec["a"]["t0_rank_share"], rec["a"]["t0_rank_contrast"]],
                              "swap": {w: s["text"] for w, s in swaps.items()},
                              "cut": {k2: v["text"] for k2, v in cuts.items() if isinstance(v, dict) and "text" in v},
                              "L*": [ls, round(single["zero"]["error"], 4), round(single["mask"]["error"], 4)],
                              "s": rec["seconds"]}), flush=True)
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
