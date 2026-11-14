"""Hop-math arms (prereg_hopmath.json), fp32 CPU replica of the answer row from Job C captures.

python hopmath.py MODEL_DIR CAPDIR SUPP_CAPDIR SCORE_JSON SUPSCORE_JSON OUT [PART]
PART: oprep (§3 operand repair) | h3 (§5 arms 1-4) | all (default). Log: ids and seconds only.
"""
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, DynamicCache

MODEL, CAPDIR, SCAP, SCORE, SUPSCORE, OUT = sys.argv[1:7]
PART = sys.argv[7] if len(sys.argv) > 7 else "all"
T0 = time.monotonic()
torch.set_grad_enabled(False)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
cfg = model.config
L, H, KV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
DH = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
layers = model.model.layers
gamma = model.model.norm.weight.double()
eps_f = float(getattr(model.model.norm, "variance_epsilon", getattr(model.model.norm, "eps", 1e-6)))
W_U = model.lm_head.weight
P = json.load(open("panel.json"))
PS = json.load(open("panel_supp.json"))
SETS = P["sets"]
S = json.load(open(SCORE))
SS = json.load(open(SUPSCORE))
LC, KC = S["model"]["L_c"], S["model"]["kv_c"]
per, sper = S["per_item"], SS["per_item"]
items = {i["id"]: i for i in P["items"]}
sitems = {i["id"]: i for i in PS["items"]}
print(f"load {time.monotonic() - T0:.1f}s", flush=True)


def safe(i):
    return i.replace(":", "_").replace(" ", "_").replace("/", "_")


_cache = {}


def cap(iid, supp=False):
    key = (iid, supp)
    if key not in _cache:
        if len(_cache) > 6:
            _cache.pop(next(iter(_cache)))
        _cache[key] = torch.load(Path(SCAP if supp else CAPDIR) / "cap" / f"{safe(iid)}.pt",
                                 map_location="cpu", weights_only=False)
    return _cache[key]


def Wg(x, g):
    W = layers[x].self_attn.o_proj.weight.double()
    return sum(W[:, h * DH:(h + 1) * DH] for h in range(H) if h // (H // KV) == g)


REC = {}
HOOK = {"replace": {}}


def _mk(kind, x):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        key = (kind, x)
        if key in HOOK["replace"]:
            o = HOOK["replace"][key].to(o.dtype).reshape(o.shape)
        REC[key] = o.reshape(-1).double()
        if isinstance(out, tuple):
            return (o,) + tuple(out[1:])
        return o
    return hook


for x in range(L):
    layers[x].self_attn.register_forward_hook(_mk("o", x))
    layers[x].mlp.register_forward_hook(_mk("m", x))
model.model.norm.register_forward_hook(lambda mod, inp, out: REC.__setitem__("h1", inp[0].reshape(-1).double()))


def forward(c, edits=None, replace=None):
    """edits: {(layer, kv, row): value(dh)}; replace: {("o"|"m", layer): vector(d)}"""
    last = c["last"]
    cache = DynamicCache(config=cfg)
    for x in range(L):
        v = c["v"][x][None, :, :last].float()
        if edits and any(k[0] == x for k in edits):
            v = v.clone()
            for (lx, g, r), val in edits.items():
                if lx == x:
                    v[0, g, r] = val.float()
        cache.update(c["k"][x][None, :, :last].float(), v, x)
    HOOK["replace"] = replace or {}
    REC.clear()
    lg = model(torch.tensor([[c["ids_last"]]]), past_key_values=cache, use_cache=True).logits[0, -1].double()
    HOOK["replace"] = {}
    return lg, {k: v.clone() for k, v in REC.items()}


def ready(it, supp=False):
    c = cap(it["id"], supp)
    c["ids_last"] = it["ids"][c["last"]]
    return c


def margin_set(lg, a, setk):
    rest = [t for t in SETS[setk] if t != a]
    return float(lg[a] - lg[rest].max())


def row_of(iid, supp=False):
    return (sper if supp else per)[iid]["row_at_item"]


# same-country mean cells over CORRECT main-panel items, per layer in [LC-3, LC+1]
BAND = [x for x in range(LC - 3, LC + 2) if 0 <= x < L]
cells = {}
for iid, it in items.items():
    if not per[iid]["correct"]:
        continue
    c = cap(iid)
    r = row_of(iid)
    cells.setdefault(it["country_token"], []).append({x: c["v"][x][KC, r].float().clone() for x in BAND})


def mean_cell(country, x, exclude=None):
    lst = cells.get(country, [])
    return torch.stack([d[x] for d in lst]).mean(0) if lst else None


out = {"L_c": LC, "kv_c": KC, "band": BAND}
if PART in ("oprep", "all"):
    rows = []
    pool = [(it, False) for it in items.values() if it["family"] in ("d3-lm", "d3-per", "d4-work") and not per[it["id"]]["correct"]]
    pool += [(it, True) for it in sitems.values() if it["family"] == "d5" and not sper[it["id"]]["correct"]]
    for it, supp in pool:
        t = time.monotonic()
        iid = it["id"]
        pr = sper[iid] if supp else per[iid]
        rec = {"id": iid, "family": it["family"], "rank": pr["country_rank"],
               "stratum": "readable" if (pr["country_rank"] is None or pr["country_rank"] > 3) else "right_operand"}
        if mean_cell(it["country_token"], LC) is None:
            rec["excluded"] = "no correct main-panel item of this country"
            rows.append(rec)
            continue
        c = ready(it, supp)
        r = pr["row_at_item"]
        lg0, _ = forward(c)
        lg1, _ = forward(c, {(LC, KC, r): mean_cell(it["country_token"], LC)})
        lg2, _ = forward(c, {(x, KC, r): mean_cell(it["country_token"], x) for x in BAND})
        rec.update({"null_argmax_ok": int(lg0.argmax()) == int(c["argmax"]), "cell_repaired": int(lg1.argmax()) == it["answer"],
                    "band_repaired": int(lg2.argmax()) == it["answer"],
                    "dM_cell": float(lg1[it["answer"]] - lg1.max()) - float(lg0[it["answer"]] - lg0.max())})
        rows.append(rec)
        print("OP " + json.dumps({"id": iid, "s": round(time.monotonic() - t, 2)}), flush=True)
    # replica donor-moves on correct base items (the reference rate, same footing)
    moves = []
    for it in items.values():
        if it["block"] != "base" or not per[it["id"]]["correct"]:
            continue
        don = items[it["donor"]]
        mc = mean_cell(don["country_token"], LC)
        if mc is None:
            continue
        c = ready(it)
        lg, _ = forward(c, {(LC, KC, row_of(it["id"])): mc})
        moves.append(int(lg.argmax()) == don["answer"])
    out["oprep"] = {"items": rows, "replica_donor_moves_base": [sum(moves), len(moves)]}

if PART in ("h3", "all"):
    U = torch.stack([gamma * W_U[t].double() for t in SETS["countries"]])          # (57, d)
    Uo, _ = torch.linalg.qr(U.T)                                                     # (d, 57)
    Wc = Wg(LC, KC)                                                                  # (d, dh)
    Mx = Uo.T @ Wc
    Pnull = torch.eye(DH, dtype=torch.float64) - torch.linalg.pinv(Mx) @ Mx
    corr = {blk: [i for i in items.values() if i["block"] == blk and per[i["id"]]["correct"]] for blk in ("base", "depth")}
    setk = {"base": "cities", "depth": "languages"}
    # arm 2 means
    means = {}
    for blk, lst in corr.items():
        for kind in ("o", "m"):
            for x in range(LC, L):
                means[(blk, kind, x)] = torch.stack([cap(i["id"])["ans"][kind][x].double() for i in lst])
    res = {}
    for blk in ("base", "depth"):
        recs = []
        SH = __import__("os").environ.get("SHARD", "0/1")
        si, sn = (int(x) for x in SH.split("/"))
        for n_, it in enumerate([i for i in items.values() if i["block"] == blk]):
            if n_ % sn != si:
                continue
            t = time.monotonic()
            iid = it["id"]
            c = ready(it)
            r = row_of(iid)
            a = it["answer"]
            v = c["v"][LC][KC, r].double()
            lg0, rec0 = forward(c)
            M0 = margin_set(lg0, a, setk[blk])
            rec = {"id": iid, "family": it["family"], "correct": per[iid]["correct"]}
            # arm 4 (all items, language count on depth)
            for nm, vv in (("comp", Pnull @ v), ("ctry", v - Pnull @ v)):
                lg, _ = forward(c, {(LC, KC, r): vv})
                rec[f"a4_{nm}_names"] = SETS[setk[blk]][int(lg[SETS[setk[blk]]].argmax())] == a
                rec[f"a4_{nm}_move"] = abs(margin_set(lg, a, setk[blk]) - M0) / abs(M0) if M0 else None
            if per[iid]["correct"]:
                # arm 3
                u = gamma * W_U[it["country_token"]].double()
                av = Wc.T @ u
                lg, _ = forward(c, {(LC, KC, r): av * (av @ v) / (av @ av)})
                rec["a3_ratio"] = margin_set(lg, a, setk[blk]) / M0 if M0 else None
                # arm 1: donor-country mean cell, move + ladder at doses 0.5, 1.0
                don = items[it["donor"]]
                md = mean_cell(don["country_token"], LC)
                if md is not None:
                    b = don["answer"]
                    h0 = rec0["h1"]
                    rho0 = float((h0.pow(2).mean() + eps_f) ** -0.5)
                    cvec = rho0 * gamma * (W_U[a].double() - W_U[b].double())
                    Mn = float(lg0[a] - lg0[b])
                    lad = {}
                    for alpha in (0.5, 1.0):
                        lg, rec1 = forward(c, {(LC, KC, r): v + alpha * (md.double() - v)})
                        if alpha == 1.0:
                            rec["a1_moves"] = int(lg.argmax()) == b
                        nd = float(lg[a] - lg[b]) - Mn
                        direct = float(cvec @ (rec1[("o", LC)] - rec0[("o", LC)]))
                        mlp = [float(cvec @ (rec1[("m", x)] - rec0[("m", x)])) for x in range(LC, L)]
                        rer = [float(cvec @ (rec1[("o", x)] - rec0[("o", x)])) for x in range(LC + 1, L)]
                        h1 = rec1["h1"]
                        rho1 = float((h1.pow(2).mean() + eps_f) ** -0.5)
                        norm = (rho1 - rho0) / rho0 * float(cvec @ h1)
                        lad[str(alpha)] = {"native_delta": nd, "direct": direct, "mlp": sum(mlp), "mlp_by_layer": mlp,
                                           "reread": sum(rer), "norm": norm,
                                           "closure_gap": nd - (direct + sum(mlp) + sum(rer) + norm)}
                    rec["a1_ladder"] = lad
                # arm 2: per-layer necessity
                prof = {}
                for kind in ("m", "o"):
                    for x in range(LC, L):
                        mm = means[(blk, kind, x)]
                        others = [j for j, i in enumerate(corr[blk]) if i["id"] != iid]
                        mean = mm[others].mean(0)
                        lg, _ = forward(c, replace={(kind, x): mean})
                        prof[f"{kind}{x}"] = (M0 - margin_set(lg, a, setk[blk])) / M0 if M0 else None
                rec["a2"] = prof
            recs.append(rec)
            res[blk] = recs
            out["h3"] = res
            json.dump(out, open(OUT, "w"), indent=1, default=str)      # incremental: survives a timeout
            print("H3 " + json.dumps({"id": iid, "s": round(time.monotonic() - t, 2)}), flush=True)
        res[blk] = recs
    out["h3"] = res
json.dump(out, open(OUT, "w"), indent=1, default=str)
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
