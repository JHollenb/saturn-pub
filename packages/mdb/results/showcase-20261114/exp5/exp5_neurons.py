"""Job E (exp5/prereg.json): an nnsight attribution-patching forecast of neuron knock-outs,
sealed in mdb on held-out items and verified natively.  python exp5_neurons.py OUTDIR [PICK]

1. nnsight ``mdb.interop.nn.grads``: one forward + one backward per equal-length group of the
   64 base items: dM/df and f at t<last>.L20..27.f, M = M[answer, donor answer].
2. Panels: within each country (sorted by id) items alternate A, B, A, ...; one-item countries
   are A only. Per country with A items: per layer the 8 neurons with the most negative summed
   attribution forecast over its A items, kept only if negative on every A item (recurrence).
   Forecast per neuron = dM/df . (mean_f - f) (knock-out to the 64-item panel mean, as Run 1).
3. Per B item: an mdb run at t<last>.L20.h; supplied.seal records the nnsight forecast of every
   named neuron and of the named set (sum), with the address map; written before any arm.
4. Native, batched: nnsight ``batch_whatif`` per country over its B items: 64 single-neuron
   knock-outs, the named set, a random set (same layer counts, seeded) and an answer-token-fixed
   null set (from the top-5x neurons of each layer ranked by |gamma_F (W_U[a]-W_U[b]) .
   W_down[:, i]| summed over the country's B items, named excluded).
5. mdb stepped authority on 2 arms per B item (the named set and its top single neuron):
   checkpoint at t<last>.L20.h, Run.set masked replace of f at each layer, continue, read;
   batched - stepped recorded as part of the floor.
The log prints item ids and seconds only.
"""
import collections
import json
import random
import sys
import time

import torch

sys.path.insert(0, ".")
from common import floor_of, items, load, log, spacing  # noqa: E402
from mdb.analysis import supplied  # noqa: E402
from mdb.analysis.finite import f64  # noqa: E402
from mdb.interop import nn  # noqa: E402

OUT = sys.argv[1]
PICK = sys.argv[2] if len(sys.argv) > 2 else ""
LAYERS = list(range(20, 28))
dbg = load()
ad = dbg.adapter
model = ad.model
rng = random.Random(20261114)
base = sorted(items("base"), key=lambda i: i["id"])
if PICK:
    base = base[: int(PICK)]
log("load")
VER = nn.version()
AMAP = {f"t<last>.L{x}.f": f"model.model.layers[{x}].mlp.down_proj.input" for x in LAYERS}

# ---- 1. gradients
by_len = collections.defaultdict(list)
for i in base:
    by_len[len(i["ids"])].append(i)
G, V, M0 = {}, {}, {}
for n, group in by_len.items():
    res = nn.grads(model, [i["ids"] for i in group], [f"L{x}.f" for x in LAYERS],
                   [(i["answer"], i["donor"]["answer"]) for i in group])
    for j, it in enumerate(group):
        G[it["id"]] = torch.stack([res["grads"][f"L{x}.f"][j, -1].double().cpu() for x in LAYERS])
        V[it["id"]] = torch.stack([res["values"][f"L{x}.f"][j, -1].double().cpu() for x in LAYERS])
        M0[it["id"]] = res["margin"][j]
log(f"grads {len(by_len)} groups")
MEAN = torch.stack(list(V.values())).mean(0)                         # (8, 6144)
FC = {k: G[k] * (MEAN - V[k]) for k in G}                             # forecast per neuron

# ---- 2. panels and nomination
by_c = collections.defaultdict(list)
for i in base:
    by_c[i["bridge"]].append(i)
A, Bp = [], []
for c, its in by_c.items():
    for j, it in enumerate(sorted(its, key=lambda i: i["id"])):
        (A if j % 2 == 0 or len(its) == 1 else Bp).append(it)
named = {}
for c, its in by_c.items():
    a_its = [i for i in its if i in A]
    if not a_its or not any(i in Bp for i in its):
        continue
    S = torch.stack([FC[i["id"]] for i in a_its])                       # (nA, 8, 6144)
    neg_all = (S < 0).all(0)
    total = torch.where(neg_all, S.sum(0), torch.zeros_like(S[0]))
    sel = []
    for li in range(len(LAYERS)):
        order = total[li].argsort()
        sel += [(li, int(k)) for k in order[:8].tolist() if total[li, k] < 0]
    named[c] = sel
log(f"named {len(named)} countries")
W_down = {x: model.model.layers[x].mlp.down_proj.weight for x in LAYERS}
final = getattr(ad.backbone, ad.spec.final_norm)
gamma = final.weight.float()
W_U = model.lm_head.weight

# ---- 3. seal supplied forecasts (B items), before any native arm
sealed = {}
fh = open(f"{OUT}/exp5-sealed{'-smoke' if PICK else ''}.jsonl", "w")
for it in Bp:
    c = it["bridge"]
    if c not in named:
        continue
    last = len(it["ids"]) - 1
    run = dbg.run(it["ids"], name="seal")
    run.until(f"t{last}.L20.h")
    tgt = {"kind": "margin", "row": last, "indices": [it["answer"], it["donor"]["answer"]]}
    recs = {}
    for li, k in named[c]:
        x = LAYERS[li]
        r = supplied.seal(run, target=tgt, edit={"address": f"t{last}.L{x}.f[i={k}]", "row": last,
                                                 "layer": x, "op": "replace with the 64-item mean"},
                          value=float(FC[it["id"]][li, k]), library="nnsight", version=VER,
                          address_map=AMAP, method="attribution patching: dM/df . (mean_f - f)")
        recs[f"L{x}.n{k}"] = {"id": r["id"], "value": float(FC[it["id"]][li, k])}
    setv = float(sum(FC[it["id"]][li, k] for li, k in named[c]))
    r = supplied.seal(run, target=tgt, edit={"address": "named set", "row": last, "layer": 20,
                                             "op": "replace with the 64-item mean"},
                      value=setv, library="nnsight", version=VER, address_map=AMAP,
                      method="sum of the named neurons' attribution forecasts")
    recs["set"] = {"id": r["id"], "value": setv}
    sealed[it["id"]] = recs
    fh.write(json.dumps({"id": it["id"], "country": c, "records": recs}) + "\n")
    del run
fh.close()
log(f"sealed {len(sealed)} items")

# ---- 4. batched native arms per country
native = {}
for c, sel in named.items():
    b_its = [i for i in Bp if i["bridge"] == c]
    if not b_its:
        continue
    t0 = time.monotonic()
    edits = {f"L{LAYERS[li]}.n{k}": [nn.Edit(f"L{LAYERS[li]}.f[i={k}]", float(MEAN[li, k]))]
             for li, k in sel}
    edits["set"] = [nn.Edit(f"L{LAYERS[li]}.f[i={k}]", float(MEAN[li, k])) for li, k in sel]
    per_layer = collections.Counter(li for li, _ in sel)
    rnd, ans = [], []
    for li, cnt in per_layer.items():
        x = LAYERS[li]
        taken = {k for l2, k in sel if l2 == li}
        pool = [k for k in range(MEAN.shape[1]) if k not in taken]
        rnd += [(li, k) for k in rng.sample(pool, cnt)]
        score = torch.zeros(MEAN.shape[1], device=W_U.device)
        for it in b_its:
            d = gamma * (W_U[it["answer"]].float() - W_U[it["donor"]["answer"]].float())
            score += (d @ W_down[x].float()).abs()
        top = [k for k in score.argsort(descending=True).tolist() if k not in taken][: 5 * cnt]
        ans += [(li, k) for k in rng.sample(top, cnt)]
    edits["random"] = [nn.Edit(f"L{LAYERS[li]}.f[i={k}]", float(MEAN[li, k])) for li, k in rnd]
    edits["answer_null"] = [nn.Edit(f"L{LAYERS[li]}.f[i={k}]", float(MEAN[li, k])) for li, k in ans]
    res = nn.batch_whatif(model, [i["ids"] for i in b_its],
                          [(i["answer"], i["donor"]["answer"]) for i in b_its], edits)
    for j, it in enumerate(b_its):
        native[it["id"]] = {"base": res["base"][j],
                            "delta": {k: v["delta"][j] for k, v in res["arms"].items()},
                            "traces": res["traces"], "seconds": round(time.monotonic() - t0, 2)}
    log(f"batch {c} {len(b_its)} items {len(edits)} arms {time.monotonic() - t0:.1f}s")

# ---- 5. mdb stepped authority (2 arms per item)
out = open(f"{OUT}/exp5-items{'-smoke' if PICK else ''}.jsonl", "w")
for it in Bp:
    if it["id"] not in native:
        continue
    t0 = time.monotonic()
    c = it["bridge"]
    sel = named[c]
    last = len(it["ids"]) - 1
    run = dbg.run(it["ids"], name="step")
    run.until(f"t{last}.logits")
    lg0 = run.read(f"t{last}.logits").reshape(-1).float()
    a, b = it["answer"], it["donor"]["answer"]
    m0 = float(lg0[a] - lg0[b])
    nat = native[it["id"]]
    singles = {k: v for k, v in nat["delta"].items() if k.startswith("L")}
    topn = max(singles, key=lambda k: abs(singles[k]))
    run2 = dbg.run(it["ids"], name="step2")
    run2.until(f"t{last}.L20.h")
    ck = run2.checkpoint("pre")
    stepped = {}
    for arm, cells in (("set", sel), (topn, [(LAYERS.index(int(topn[1:].split(".")[0])),
                                               int(topn.split(".n")[1]))])):
        for li in sorted({li for li, _ in cells}):
            x = LAYERS[li]
            run2.until(f"t{last}.L{x}.h")
            val = torch.zeros(1, 1, MEAN.shape[1], dtype=ad.dtype, device=W_U.device)
            mask = torch.zeros(1, 1, MEAN.shape[1], dtype=torch.bool, device=W_U.device)
            for l2, k in cells:
                if l2 == li:
                    val[0, 0, k] = float(MEAN[li, k])
                    mask[0, 0, k] = True
            run2.set(f"t{last}.L{x}.f", val, mask=mask)
        run2.until(f"t{last}.logits")
        lg = run2.read(f"t{last}.logits").reshape(-1).float()
        stepped[arm] = float(lg[a] - lg[b]) - m0
        run2.restore(ck)
    rec = {"id": it["id"], "country": c, "base_batched": nat["base"], "base_stepped": m0,
           "floor": floor_of(lg0, a, b), "spacing": spacing(max(abs(float(lg0[a])), abs(float(lg0[b])))),
           "native_batched": nat["delta"], "stepped": stepped, "top_single": topn,
           "forecast": {k: v["value"] for k, v in sealed[it["id"]].items()},
           "records": {k: v["id"] for k, v in sealed[it["id"]].items()},
           "judged": {k: supplied.judge({"id": sealed[it["id"]][k]["id"], "body": {"supplied": {"value": sealed[it["id"]][k]["value"]}}},
                                        nat["delta"][k], magnitude=max(abs(float(lg0[a])), abs(float(lg0[b]))),
                                        dtype=torch.bfloat16, extra_floor=abs(nat["base"] - m0))
                      for k in sealed[it["id"]]}}
    out.write(json.dumps(rec, default=float) + "\n")
    out.flush()
    del run, run2
    log(f"S {it['id']} {time.monotonic() - t0:.2f}s")
json.dump({"A": [i["id"] for i in A], "B": [i["id"] for i in Bp],
           "named": {c: [[LAYERS[li], k] for li, k in s] for c, s in named.items()},
           "nnsight": VER}, open(f"{OUT}/exp5-panels{'-smoke' if PICK else ''}.json", "w"), indent=1)
log("TOTAL")
