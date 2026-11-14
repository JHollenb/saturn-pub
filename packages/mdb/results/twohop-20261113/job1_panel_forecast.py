"""Job 1 (passive): two-hop accuracy, hop 1 / hop 2 alone, admission, donors, and the sealed
symbol-table forecast of the bridge carrier. No edit runs in this job.

python job1_panel_forecast.py MODEL candidates.json OUT_PREFIX [LIMIT]

Forecast (prereg F1): at the recipient's native run (cursor at t<n-1>.logits), for every
descriptor row r, layer l and KV group g, c_d(r,l,g) = sum over the group's query heads of the
exact direct contribution of the answer row's read of row r to M[a, b_d] (mdb reads
``to_consumer``, P19: p[h,r] W_o,h v[g,r] through the final-norm readout). The bulk is computed
from ``reads.head_mix`` parts and ``reads.readout`` vectors (the same two calls ``reads`` makes);
one official ``reads`` call per item checks that the bulk equals ``to_consumer`` exactly.
"""
import json
import sys
import time

import torch

from mdb.analysis.forecast import Margin
from mdb.analysis.reads import head_mix, readout, reads
from mdb.ops import dispatch
from mdb.repl.__main__ import _load

MODEL, CAND, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
T0 = time.monotonic()
torch.set_grad_enabled(False)
dbg = _load(__import__("pathlib").Path(MODEL), "bf16", "cuda", True, None)
tok = dbg.tokenizer
spec = json.load(open(CAND))
items = spec["items"][:LIMIT] if LIMIT else spec["items"]
caps = sorted({i["answer"] for i in spec["items"]})
countries = sorted({c for i in spec["items"] for c in i["country_accept"]})
watch = sorted(set(caps) | set(countries))
H = dbg.adapter.num_heads
KV = dbg.adapter.num_kv_heads
L = dbg.adapter.layers
print(f"load {time.monotonic() - T0:.1f}s; {len(items)} candidates; heads {H} kv {KV} layers {L}",
      flush=True)


def native(ids, name="main"):
    run = dbg.run(ids, name=name)
    run.until(f"t{len(ids) - 1}.logits")
    return run, run.read().reshape(-1).float()


def top(logits, k=5):
    v, i = logits.topk(k)
    return [[int(a), tok.decode([int(a)]), round(float(b), 3)] for a, b in zip(i, v)]


# ---------------------------------------------------------------- accuracy, hop 1, hop 2
rows = []
for item in items:
    t = time.monotonic()
    _, lg = native(item["ids"])
    _, l1 = native(item["hop1_ids"], "hop1")
    _, l2 = native(item["hop2_ids"], "hop2")
    rec = {"id": item["id"], "argmax": int(lg.argmax()), "argmax_text": tok.decode([int(lg.argmax())]),
           "correct": int(lg.argmax()) == item["answer"], "top5": top(lg),
           "watch": {str(w): round(float(lg[w]), 4) for w in watch},
           "hop1_argmax": tok.decode([int(l1.argmax())]),
           "hop1_correct": int(l1.argmax()) in item["country_accept"],
           "hop1_top5": top(l1),
           "hop2_argmax": tok.decode([int(l2.argmax())]),
           "hop2_correct": int(l2.argmax()) == item["answer"], "seconds": time.monotonic() - t}
    rows.append(rec)
    print("ACC " + json.dumps({k: rec[k] for k in ("id", "argmax_text", "correct", "hop1_argmax",
                                                     "hop1_correct", "hop2_argmax", "hop2_correct")}),
          flush=True)
byid = {r["id"]: r for r in rows}
acc = sum(r["correct"] for r in rows) / len(rows)
print(f"ACCURACY {acc:.3f} ({sum(r['correct'] for r in rows)}/{len(rows)}) "
      f"T {time.monotonic() - T0:.1f}s", flush=True)

# ---------------------------------------------------------------- panel, donors, controls
admitted = [i for i in items if byid[i["id"]]["correct"]]
order = {i["id"]: k for k, i in enumerate(items)}
panel = []
for k, item in enumerate(admitted):
    ring = admitted[k + 1:] + admitted[:k]
    donors = [d for d in ring if d["family"] == item["family"] and d["n"] == item["n"]
              and d["bridge"] != item["bridge"] and d["answer"] != item["answer"]][:2]
    if not donors:
        continue
    same = [d for d in ring if d["bridge"] == item["bridge"] and d["id"] != item["id"]]
    aligned = [d for d in same if d["family"] == item["family"] and d["n"] == item["n"]]
    control = (aligned or same or [None])[0]
    panel.append({"id": item["id"], "donors": [d["id"] for d in donors],
                  "control": None if control is None else control["id"],
                  "control_aligned": bool(aligned)})
if len(panel) > 64:     # family round-robin, candidate order inside a family
    fams = {}
    for p in panel:
        fams.setdefault(p["id"].split(":")[0], []).append(p)
    keep, k = [], 0
    while len(keep) < 64:
        for f in sorted(fams):
            if k < len(fams[f]) and len(keep) < 64:
                keep.append(fams[f][k])
        k += 1
    panel = sorted(keep, key=lambda p: order[p["id"]])
print(f"PANEL {len(panel)} recipients of {len(admitted)} admitted", flush=True)
json.dump({"accuracy": acc, "rows": rows, "panel": panel}, open(f"{OUT}-panel.json", "w"),
          indent=1)

# ---------------------------------------------------------------- forecast (passive)
cand = {i["id"]: i for i in items}
forecasts = []
weights = {}
for entry in panel:
    t = time.monotonic()
    item = cand[entry["id"]]
    n, span, r0 = item["n"], item["span"], item["row"]
    j = n - 1
    run, lg = native(item["ids"])
    donors = [cand[d] for d in entry["donors"]]
    margins = [Margin(j, item["answer"], d["answer"]) for d in donors]
    ct = item["country_token"]
    cmargins = [Margin(j, ct, d["country_token"]) for d in donors
                if ct is not None and d["country_token"] is not None]
    vec = torch.stack([readout(run, m).vector for m in margins])           # (D, d) float64
    cvec = torch.stack([readout(run, m).vector for m in cmargins]) if cmargins else None
    c = torch.zeros(len(span), L, KV, dtype=torch.float64)
    c2 = torch.zeros(len(span), L, KV, dtype=torch.float64)
    mass = torch.zeros(len(span), L, KV, dtype=torch.float64)
    write = torch.zeros(L, KV, vec.shape[1], dtype=torch.float64)          # r0 group writes
    head = torch.zeros(len(span), L, H, dtype=torch.float64)
    for layer in range(L):
        mix = head_mix(run, j, layer, weights)
        parts = mix.o_parts()[:, span, :]                                    # (H, S, d)
        per_head = parts @ vec.T                                             # (H, S, D)
        head[:, layer, :] = per_head.mean(-1).T
        for h in range(H):
            g = mix.kappa[h]
            c[:, layer, g] += per_head[h].mean(-1)
            if cvec is not None:
                c2[:, layer, g] += (parts[h] @ cvec.T).mean(-1)
            mass[:, layer, g] += mix.p[h, span]
            write[layer, g] += parts[h, span.index(r0)]
    weights.clear()
    # one official reads call: the bulk equals reads' to_consumer for donor 0 at r0
    official = reads(run, f"t{r0}", consumer=margins[0], causal=False, top=1)
    got = {}
    for e in official.tensors["entries"]:
        g = int(e["name"].split(".h")[-1]) // (H // KV)
        got[(e["layer"], g)] = got.get((e["layer"], g), 0.0) + float(e["_value"])
    mine0 = torch.zeros(L, KV, dtype=torch.float64)
    for layer in range(L):
        mix = head_mix(run, j, layer, weights)
        p0 = mix.o_parts()[:, r0, :] @ vec[0]
        for h in range(H):
            mine0[layer, mix.kappa[h]] += p0[h]
    weights.clear()
    check = max(abs(got[(layer, g)] - float(mine0[layer, g])) for layer in range(L)
                for g in range(KV))

    def best(grid, k=10):
        flat = grid.reshape(-1)
        vals, idx = flat.topk(k)
        out = []
        for v, i in zip(vals.tolist(), idx.tolist()):
            ri, rest = divmod(i, L * KV)
            layer, g = divmod(rest, KV)
            out.append({"row": span[ri], "layer": layer, "kv": g, "value": v})
        return out

    # E9-style naming of the r0 group write: rank of the bridge country among the vocabulary
    norm = getattr(dbg.adapter.backbone, dbg.adapter.spec.final_norm)
    W = dbg.adapter.model.lm_head.weight                                     # bf16 on GPU
    # derived-local ranking: gamma * write rounded once to the head's dtype, bf16 matmul with
    # float32 accumulation (no float copy of the whole unembedding)
    x = (write.reshape(L * KV, -1) * norm.weight.double().cpu()).to(device=W.device, dtype=W.dtype)
    proj = (x @ W.T).float()
    rank = None
    if ct is not None:
        rank = (proj > proj[:, ct:ct + 1]).sum(-1).reshape(L, KV).cpu().tolist()
    f = {"id": item["id"], "span": span, "r0": r0, "donors": entry["donors"],
         "native_margins": [float(lg[item["answer"]] - lg[d["answer"]]) for d in donors],
         "F1": best(c)[0], "F1_top10": best(c), "F2": best(c2)[0] if cvec is not None else None,
         "F2_top10": best(c2) if cvec is not None else None,
         "F3": best(mass)[0], "F3_top10": best(mass),
         "reads_check_max_abs": check,
         "c": c.tolist(), "c_country": c2.tolist() if cvec is not None else None,
         "mass": mass.tolist(), "head_contrib": head.tolist(),
         "country_rank_r0": rank, "seconds": time.monotonic() - t}
    forecasts.append(f)
    print("FORECAST " + json.dumps({k: f[k] for k in ("id", "F1", "F2", "F3", "reads_check_max_abs",
                                                      "seconds")}), flush=True)
# the symbol table with the declared descriptor symbol, for the record (first 4 recipients)
tables = []
ctx = dbg.ops(None)
for entry in panel[:4]:
    item = cand[entry["id"]]
    native(item["ids"])
    ctx = dbg.ops(None)
    addresses = ",".join(f"t{r}" for r in item["span"])
    try:
        dispatch(ctx, "symbol", {"name": "desc", "addresses": addresses})
        d0 = cand[entry["donors"][0]]
        tab = dispatch(ctx, "symbols", {"consumer": f"M[{item['answer']},{d0['answer']}]",
                                        "limit": 4})
        tables.append({"id": item["id"], "table": tab})
        print("SYMBOLS " + json.dumps(tab)[:3000], flush=True)
    except Exception as error:  # noqa: BLE001
        tables.append({"id": item["id"], "error": f"{type(error).__name__}: {error}"})
        print("SYMBOLS error", type(error).__name__, error, flush=True)
json.dump({"sealed_at_job": "job1 (passive)", "rule": "prereg F1/F2/F3", "forecasts": forecasts,
           "symbol_tables": tables}, open(f"{OUT}-forecast.json", "w"))
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
