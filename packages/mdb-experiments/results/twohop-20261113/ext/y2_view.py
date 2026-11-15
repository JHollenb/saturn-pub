"""Y2: the chain view (prereg_hard.json), its baselines, the causal check and the ground truth.

python y2_view.py MODEL candidates_ext.json X1_PANEL X1_FORECAST FAMS OUT

For every candidate of the families FAMS (correct and wrong): the symbol-table view of the hidden
chain at the answer row, with consumer = margin between the two largest logits (the symbols
default). For the X1 panel recipients: the causal check (donor V cell at the view's bridge site:
does the read content and the answer move together?) and the cell_sweep ground truth.
"""
import json
import sys
import time
from pathlib import Path

import torch

from mdb.analysis.forecast import Margin
from mdb.analysis.reads import head_mix, readout
from mdb.analysis.sweep import Cell, cell_sweep
from mdb.repl.__main__ import _load

MODEL, CAND, PANEL, FORE, FAMS, OUT = sys.argv[1:7]
FAMS = tuple(FAMS.split(","))
T0 = time.monotonic()
torch.set_grad_enabled(False)
dbg = _load(Path(MODEL), "bf16", "cuda", True, None)
tok, model = dbg.tokenizer, dbg.model
L, H, KV = dbg.adapter.layers, dbg.adapter.num_heads, dbg.adapter.num_kv_heads
spec = json.load(open(CAND))
items = [i for i in spec["items"] if i["family"].startswith(FAMS)]
cand = {i["id"]: i for i in spec["items"]}
panel = {p["id"]: p for p in json.load(open(PANEL))["panel"]}
fore = {f["id"]: f for f in json.load(open(FORE))["forecasts"]}
hubs = sorted({i["country_token"] for i in spec["items"] if i["country_token"]})
hub_t = torch.tensor(hubs, device=model.lm_head.weight.device)
norm = getattr(dbg.adapter.backbone, dbg.adapter.spec.final_norm)
gamma = norm.weight.double().cpu()
W = model.lm_head.weight
CONST = (20, 6)
LENS_LAYER = 24
OUTF = open(f"{OUT}.jsonl", "w")
print(f"load {time.monotonic() - T0:.1f}s; {len(items)} candidates; {len(hubs)} hub tokens",
      flush=True)


def name(writes):
    """Best hub token of each write (rows of a (n, d) float64 tensor) and the full-vocab top 5
    of the first one."""
    x = (writes * gamma).to(device=W.device, dtype=W.dtype)
    proj = (x @ W[hub_t].T).float()
    return [hubs[int(i)] for i in proj.argmax(-1)]


def top5(write):
    x = (write * gamma).to(device=W.device, dtype=W.dtype)
    return [tok.decode([int(i)]) for i in (W @ x).float().topk(5).indices]


def view(run, item, logits):
    j, span, r0 = item["n"] - 1, item["span"], item["row"]
    t1, t2 = [int(x) for x in logits.topk(2).indices]
    vec = readout(run, Margin(j, t1, t2)).vector
    S = len(span)
    contrib = torch.zeros(S, L, KV, dtype=torch.float64)
    mass = torch.zeros(S, L, KV, dtype=torch.float64)
    head_c = torch.zeros(L, H, dtype=torch.float64)
    writes = torch.zeros(S, L, KV, vec.numel(), dtype=torch.float64)
    weights = {}
    for layer in range(L):
        mix = head_mix(run, j, layer, weights)
        parts = mix.o_parts()[:, span, :]                         # (H, S, d)
        c = parts @ vec                                            # (H, S)
        head_c[layer] = c.sum(1)
        for h in range(H):
            g = mix.kappa[h]
            contrib[:, layer, g] += c[h]
            mass[:, layer, g] += mix.p[h, span]
            writes[:, layer, g] += parts[h]
    weights.clear()

    def site(grid):
        i = int(grid.reshape(-1).argmax())
        ri, rest = divmod(i, L * KV)
        return ri, *divmod(rest, KV)
    bi, bl, bg = site(contrib)
    mi, ml, mg = site(mass)
    named = name(writes[bi, bl, bg][None])[0]
    # first appearance at the bridge row: most-attended group per layer
    per_layer = writes[bi, torch.arange(L), mass[bi].argmax(-1)]   # (L, d)
    layer_names = name(per_layer)
    first = next((l_ for l_ in range(L) if layer_names[l_] == named), None)
    span_mass = mass.sum(0).sum(-1)                                # (L,) over rows and groups
    life = [l_ for l_ in range(L) if span_mass[l_] >= 0.1]
    readers = torch.topk(head_c.reshape(-1), 3).indices.tolist()
    ri0 = span.index(r0)
    return {
        "top1": tok.decode([t1]), "top2": tok.decode([t2]),
        "bridge_site": [span[bi], bl, bg], "bridge_named": named,
        "bridge_top5": top5(writes[bi, bl, bg]),
        "bridge_contrib": float(contrib[bi, bl, bg]),
        "first_appearance": first, "layer_names": [tok.decode([x]) for x in layer_names],
        "descriptor_life": [life[0], life[-1]] if life else None,
        "readers": [[divmod(x, H)[0], divmod(x, H)[1], float(head_c.reshape(-1)[x])] for x in readers],
        "const_named": name(writes[ri0, CONST[0], CONST[1]][None])[0],
        "mass_site": [span[mi], ml, mg], "mass_named": name(writes[mi, ml, mg][None])[0],
    }


def lens(item):
    r0 = item["row"]
    run = dbg.run(item["ids"][:r0 + 1], name="trunc")
    run.until(f"t{r0}.L{LENS_LAYER}.h")
    x = run.read().reshape(-1).double().cpu()
    x = x * (x.pow(2).mean() + float(norm.variance_epsilon)) ** -0.5
    return name(x[None])[0]


for item in items:
    t = time.monotonic()
    run = dbg.run(item["ids"], name="main")
    run.until(f"t{item['n'] - 1}.logits")
    logits = run.read().reshape(-1).float()
    am = int(logits.argmax())
    rec = {"id": item["id"], "family": item["family"], "bridge": item["bridge"],
           "true_hub": item["country_token"], "correct": am in item["answer_accept"],
           "answer": tok.decode([am]), "answer_id": am}
    v = view(run, item, logits)
    rec.update(v)
    rec["lens_named"] = lens(item)
    for key in ("bridge_named", "const_named", "mass_named", "lens_named"):
        rec[key + "_ok"] = rec[key] == item["country_token"]
        rec[key] = tok.decode([rec[key]])
    entry = panel.get(item["id"])
    if entry is not None and rec["correct"]:
        r, layer, g = v["bridge_site"]
        donors = [cand[d] for d in entry["donors"]]
        causal, arms, dvs = [], [], []
        for d in donors:
            drun = dbg.run(d["ids"], name="donor")
            dv = [drun.read_cache(x, "v", staged=False).clone() for x in range(L)]
            dvs.append(dv)
            er = dbg.run(item["ids"], name="edit")
            er.set_cell(layer, "v", dv[layer][:, g:g + 1, r:r + 1], rows=[r], kv=[g])
            er.until(f"t{item['n'] - 1}.logits")
            el = er.read().reshape(-1).float()
            mix = head_mix(er, item["n"] - 1, layer, {})
            w = sum(mix.o_parts()[h, r] for h in range(H) if mix.kappa[h] == g)
            content = name(w[None])[0]
            causal.append({"donor": d["id"], "content_donor": content == d["country_token"],
                           "content_recipient": content == item["country_token"],
                           "content": tok.decode([content]),
                           "answer_donor": int(el.argmax()) == d["answer"],
                           "answer_kept": int(el.argmax()) in item["answer_accept"],
                           "answer": tok.decode([int(el.argmax())])})
        rec["causal"] = causal
        # ground truth: V-only single cells over span x layers x KV heads
        base = dbg.run(item["ids"], name="sweep")
        keys = []
        for k, d in enumerate(donors):
            for rr in item["span"]:
                for x in range(L):
                    for gg in range(KV):
                        arms.append([Cell(x, "v", rr, gg, dvs[k][x][0, gg, rr])])
                        keys.append((k, rr, x, gg))
        toks = [item["answer"]] + [d["answer"] for d in donors]
        out = cell_sweep(base, arms, tokens=toks, batch=96)
        eff = {}
        for i, (key, lg) in enumerate(zip(keys, out["logits"])):
            n = out["null"][i // 96]
            k = key[0]
            eff.setdefault(key[1:], []).append(float((lg[0] - lg[1 + k]) - (n[0] - n[1 + k])))
        mean = {s: sum(x) / len(x) for s, x in eff.items()}
        order = sorted(mean, key=mean.get)
        gt = order[0]
        f = fore.get(item["id"])
        rec["gt"] = {"site": list(gt), "effect": mean[gt], "second": list(order[1]),
                     "second_effect": mean[order[1]], "null_vs_stepped": out["null_vs_stepped"]}
        rec["hits"] = {"view": tuple(v["bridge_site"]) == gt,
                       "const": gt == (item["row"], *CONST),
                       "mass_view": tuple(v["mass_site"]) == gt}
        if f is not None:
            for k_ in ("F1", "F3"):
                rec["hits"][k_] = (f[k_]["row"], f[k_]["layer"], f[k_]["kv"]) == gt
    rec["seconds"] = round(time.monotonic() - t, 2)
    OUTF.write(json.dumps(rec) + "\n")
    OUTF.flush()
    print("VIEW " + json.dumps({k: rec.get(k) for k in ("id", "correct", "answer", "bridge_site",
                                                        "bridge_named", "bridge_named_ok",
                                                        "const_named_ok", "lens_named_ok",
                                                        "first_appearance", "hits", "seconds")}),
          flush=True)
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
