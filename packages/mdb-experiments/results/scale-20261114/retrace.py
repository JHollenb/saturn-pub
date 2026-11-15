"""EXPLORATORY (investigation node-selection-and-retrace-with-scale; not preregistered): count candidate
switches across layers, from the Job C captures.

python retrace.py MODEL_DIR CAPDIR SCORE_JSON OUT
Per main-panel item:
  carrier write: at row_at_item, KV kv_c, layers 0..L_c, the best country of gamma_F * W_U over the cell
    write W_g v (unit mass); switches = number of consecutive-layer changes of that argmax; also the
    switches among layers where the cell names a country with margin > 0 over the runner-up only.
  answer row: lens (final norm, W_U) of h at every layer over the item's answer set (cities or
    languages); switches over all layers and over layers >= L_c; the layer after which it never changes.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

MODEL, CAPDIR, SCORE, OUT = sys.argv[1:5]
torch.set_grad_enabled(False)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
cfg = model.config
L, H, KV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
DH = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
gamma = model.model.norm.weight.double()
eps = float(getattr(model.model.norm, "variance_epsilon", 1e-6))
W_U = model.lm_head.weight
P = json.load(open("panel.json"))
SETS = P["sets"]
S = json.load(open(SCORE))
LC, KC = S["model"]["L_c"], S["model"]["kv_c"]
RO = {k: gamma[None, :] * W_U[v].double() for k, v in SETS.items()}
WG = []
for x in range(LC + 1):
    W = model.model.layers[x].self_attn.o_proj.weight.double()
    WG.append(sum(W[:, h * DH:(h + 1) * DH] for h in range(H) if h // (H // KV) == KC))


def safe(i):
    return i.replace(":", "_").replace(" ", "_").replace("/", "_")


def switches(seq):
    return sum(a != b for a, b in zip(seq, seq[1:]))


out = {"model": Path(MODEL).name, "L_c": LC, "kv_c": KC, "layers": L, "items": {}}
for it in P["items"]:
    iid = it["id"]
    c = torch.load(Path(CAPDIR) / "cap" / f"{safe(iid)}.pt", map_location="cpu", weights_only=False)
    r = S["per_item"][iid]["row_at_item"]
    cw = []
    for x in range(LC + 1):
        s = RO["countries"] @ (WG[x] @ c["v"][x][KC, r].double())
        cw.append(int(s.argmax()))
    setk = "languages" if it["block"] == "depth" else "cities"
    lens = []
    for x in range(L):
        h = c["ans"]["h"][x].double()
        hn = h * torch.rsqrt(h.pow(2).mean() + eps)
        lens.append(int((RO[setk] @ hn).argmax()))
    final = lens[-1]
    settle = max((x for x in range(L) if lens[x] != final), default=-1) + 1
    ti = SETS["countries"].index(it["country_token"])
    ai = SETS[setk].index(it["answer"]) if it["answer"] in SETS[setk] else None
    out["items"][iid] = {"family": it["family"], "correct": S["per_item"][iid]["correct"],
                         "carrier_switches": switches(cw), "carrier_switches_after_first_true":
                         switches(cw[cw.index(ti):]) if ti in cw else None,
                         "carrier_final_true": cw[-1] == ti,
                         "answer_switches": switches(lens), "answer_switches_after_Lc": switches(lens[LC:]),
                         "answer_settle_layer": settle, "answer_settle_frac": settle / L,
                         "answer_lens_final_true": final == ai}
    del c
json.dump(out, open(OUT, "w"), indent=1)
print("TOTAL", len(out["items"]))
