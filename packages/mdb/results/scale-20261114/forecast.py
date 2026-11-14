"""Job F (prereg_scale.json, sealed between Job C and Job A): passive forecasts of the single-cell
donor transplant on M = logit(a) - logit(b), from the capture only, fp32 CPU replica.

python forecast.py MODEL_DIR CAPDIR OUT_PREFIX [PICK]
Per item, every span cell (layer, kv, span row with an aligned donor row, aligned from the span end):
  prop  passive P05 score  sum_{h in g} p_h(r) d . W_O,h (v_d - v)  (d = gamma_F (W_U[a] - W_U[b]))
  fo    first-order  grad_V M . (v_d - v), one backward through the replica's answer row
  re    replica re-forward of the answer row with that one cell edited (all other K/V fixed), for
        the 3 most negative fo cells and the most negative prop cell
Writes OUT_PREFIX.pt (id -> {"cells": (N, 3) long [layer, kv, row], "prop", "fo"}) and
OUT_PREFIX.json (replica arms, replica null error, seconds). Log: ids and seconds only.
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import DynamicCache

from mdb.repl.__main__ import _load

MODEL, CAPDIR, OUTP = sys.argv[1:4]
PICK = sys.argv[4] if len(sys.argv) > 4 else ""
T0 = time.monotonic()
dbg = _load(Path(MODEL), "fp32", "cpu", True, None)
ad = dbg.adapter
model = ad.model
for p_ in model.parameters():
    p_.requires_grad_(False)
L, H, KV, DH = ad.layers, ad.num_heads, ad.num_kv_heads, ad.head_dim
gq = torch.tensor([h // (H // KV) for h in range(H)])
gamma = getattr(ad.backbone, ad.spec.final_norm).weight.detach().double()
W_U = model.lm_head.weight.detach()
WO = [getattr(ad.backbone, ad.spec.layers)[x].self_attn.o_proj.weight.detach().double() for x in range(L)]
P = json.load(open("panel.json"))
items = P["items"]
if PICK:
    items = items[int(PICK.split("-")[0]):int(PICK.split("-")[1])]
by_id = {i["id"]: i for i in P["items"]}
TOKS = sorted({t for v in P["sets"].values() for t in v} | {i["answer"] for i in P["items"]})
print(f"load {time.monotonic() - T0:.1f}s", flush=True)


def safe(i):
    return i.replace(":", "_").replace(" ", "_").replace("/", "_")


def load(iid):
    return torch.load(Path(CAPDIR) / "cap" / f"{safe(iid)}.pt", map_location="cpu", weights_only=False)


def forward(cap, last, V, token):
    cache = DynamicCache(config=model.config)
    for x in range(L):
        cache.update(cap["k"][x][None, :, :last].float(), V[x], x)
    return model(torch.tensor([[token]]), past_key_values=cache, use_cache=True).logits[0, -1]


out_t, out_j = {}, {}
for item in items:
    t = time.monotonic()
    cap, dcap = load(item["id"]), load(item["donor"])
    don = by_id[item["donor"]]
    last, a, b = cap["last"], item["answer"], don["answer"]
    pairs = list(zip(reversed(item["span"]), reversed(don["span"])))
    V = [cap["v"][x][None, :, :last].float().clone().requires_grad_(True) for x in range(L)]
    with torch.enable_grad():
        lg = forward(cap, last, V, item["ids"][last])
        M = lg[a].double() - lg[b].double()
        M.backward()
    M0 = float(M)
    null_err = float((lg.detach()[TOKS] - cap["logits"]).abs().max())
    d = gamma * (W_U[a].double() - W_U[b].double())
    cells, fo, prop = [], [], []
    for x in range(L):
        dWo = (d @ WO[x]).reshape(H, DH)                                  # (H, dh)
        pa = cap["ans"]["p"][x].double()                                  # (H, rows)
        for r, dr in pairs:
            dv = (dcap["v"][x][:, dr].double() - cap["v"][x][:, r].double())    # (KV, dh)
            g = V[x].grad[0, :, r].double()                               # (KV, dh)
            f = (g * dv).sum(-1)                                          # (KV,)
            s = torch.zeros(KV, dtype=torch.float64)
            s.index_add_(0, gq, pa[:, r] * (dWo * dv[gq]).sum(-1))
            for k in range(KV):
                cells.append((x, k, r))
            fo.append(f)
            prop.append(s)
    fo, prop = torch.cat(fo), torch.cat(prop)
    cells_t = torch.tensor(cells)
    pick = [int(i) for i in fo.argsort()[:3]] + [int(prop.argmin())]
    arms = []
    with torch.no_grad():
        for idx in dict.fromkeys(pick):
            x, k, r = cells[idx]
            dr = dict(pairs)[r]
            V2 = [v.detach() for v in V]
            V2[x] = V2[x].clone()
            V2[x][0, k, r] = dcap["v"][x][k, dr].float()
            lg2 = forward(cap, last, V2, item["ids"][last])
            arms.append({"cell": [x, k, r], "fo": float(fo[idx]), "prop": float(prop[idx]),
                         "replica_dM": float(lg2[a].double() - lg2[b].double()) - M0,
                         "rank_fo": pick.index(idx) if idx in pick[:3] else None,
                         "is_prop_top": idx == pick[3]})
    out_t[item["id"]] = {"cells": cells_t, "prop": prop.float(), "fo": fo.float()}
    out_j[item["id"]] = {"M_replica": M0, "replica_null_err": null_err, "arms": arms,
                         "seconds": round(time.monotonic() - t, 2)}
    del cap, dcap, V
    print("F " + json.dumps({"id": item["id"], "s": out_j[item["id"]]["seconds"]}), flush=True)
torch.save(out_t, OUTP + ".pt")
json.dump(out_j, open(OUTP + ".json", "w"), indent=1)
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
