"""Audit, one CPU job on Beast (no GPU, no model forward): independent re-implementation from the
Job C captures and the raw safetensors (no mdb, no score.py code) of
  (1) the carrier readout (names country / rank) at the builder's carrier cell and row,
  (2) co-readouts: raw, rank-1 country projection (as prereg), FULL country-subspace projection
      (all 57 country directions removed from write and candidates), lexical null, and an
      answer-permutation null (the projected readout scored against other items' answers),
  (3) L_first at the carrier row with a 20-draw chance null (prereg used one draw per item),
  (4) a KV-index lineage test: token-space OV maps of every (layer, kv) of the three Qwen3 models on
      the panel's candidate tokens, correlation between models at matched relative depth, same kv
      index vs different kv index.

python a3_beast_cpu.py OUT.json      (cwd = payload with panel.json, carriers.json)
"""
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

T0 = time.monotonic()
torch.set_grad_enabled(False)
OUT = sys.argv[1]
P = json.load(open("panel.json"))
CAR = json.load(open("carriers.json"))          # model -> {L_c, kv_c, row_at_item}
SETS = P["sets"]
MROOT, CROOT = Path("/mnt/big/llm-models"), Path("/mnt/ssd1tb/cache/beast/scale")
MODELS = ["Qwen2.5-0.5B", "Qwen3-0.6B-Base", "Qwen2.5-1.5B", "Qwen3-1.7B-Base", "Qwen2.5-3B", "Qwen3-4B-Base"]
import os
SMOKE = bool(os.environ.get("SMOKE"))
if SMOKE:
    MODELS = ["Qwen2.5-0.5B", "Qwen3-0.6B-Base"]
    P["items"] = [i for i in P["items"] if i["family"] in ("currency", "d2")][:4] + [i for i in P["items"] if i["family"] == "d2"][:3]


def safe(i):
    return i.replace(":", "_").replace(" ", "_").replace("/", "_")


class W:
    def __init__(self, d):
        idx = d / "model.safetensors.index.json"
        self.map = json.load(open(idx))["weight_map"] if idx.exists() else None
        self.d = d
        self.cfg = json.load(open(d / "config.json"))

    def get(self, k):
        f = self.d / (self.map[k] if self.map else "model.safetensors")
        with safe_open(str(f), "pt") as g:
            return g.get_tensor(k).double()


def proj_out_basis(Q):
    return lambda X: X - (X @ Q) @ Q.T


res = {"models": {}}
OVS = {}
TOK = sorted({t for v in SETS.values() for t in v})
for m in MODELS:
    t0 = time.monotonic()
    w = W(MROOT / m)
    cfg = w.cfg
    L, H, KV = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    dh = cfg.get("head_dim") or cfg["hidden_size"] // H
    G = H // KV
    E = w.get("model.embed_tokens.weight")
    WU = E if cfg.get("tie_word_embeddings") else w.get("lm_head.weight")
    gF = w.get("model.norm.weight")
    RO = {k: gF[None, :] * WU[v] for k, v in SETS.items()}                # (n, d)
    LC, KC = CAR[m]["L_c"], CAR[m]["kv_c"]
    WO = {}
    for x in range(L):
        o = w.get(f"model.layers.{x}.self_attn.o_proj.weight")              # (d, H*dh)
        WO[x] = torch.stack([o[:, g * G * dh:(g + 1) * G * dh].reshape(o.shape[0], G, dh).sum(1) for g in range(KV)])  # (KV, d, dh)
    # country subspace (orthonormal basis of the 57 country readout directions)
    Qc, _ = torch.linalg.qr(RO["countries"].T)                              # (d, 57)
    full = proj_out_basis(Qc)
    rng = random.Random(1)
    per = {}
    for it in P["items"]:
        iid = it["id"]
        cap = torch.load(CROOT / m / "cap" / f"{safe(iid)}.pt", map_location="cpu", weights_only=False)
        v = cap["v"]
        r = CAR[m]["row_at_item"][iid]
        wv = WO[LC][KC] @ v[LC][KC, r].double()
        sc = RO["countries"] @ wv
        ti = SETS["countries"].index(it["country_token"])
        rank = int((sc > sc[ti]).sum()) + 1
        rec = {"family": it["family"], "block": it["block"], "rank": rank}
        # L_first with 20 fake draws
        names = [{int((RO["countries"] @ (WO[x][k] @ v[x][k, r].double())).argmax()) for k in range(KV)} for x in range(LC + 1)]
        first = lambda c: next((x for x in range(LC + 1) if c in names[x]), None)
        rec["L_first"] = first(ti)
        others = [i for i in range(len(SETS["countries"])) if i != ti]
        rec["L_first_fake20"] = [first(c) for c in rng.sample(others, 20)]
        # co-readouts
        u = gF * WU[it["country_token"]]
        uh = u / u.norm()
        p1 = lambda X: X - (X @ uh)[..., None] * uh if X.dim() == 2 else X - (X @ uh) * uh
        targets = {"cities": it["capital_token"]}
        if it["block"] == "depth":
            targets["languages"] = it["answer"]
        co = {}
        for sk, tt in targets.items():
            if tt is None or tt not in SETS[sk]:
                continue
            R = RO[sk]
            ans = SETS[sk].index(tt)
            co[sk] = {"raw": int((R @ wv).argmax()), "rank1": int((p1(R) @ p1(wv)).argmax()),
                      "full": int((full(R) @ full(wv[None, :])[0]).argmax()), "lex": int((R @ u).argmax()), "ans": ans}
        rec["co"] = co
        per[iid] = rec
        del cap, v
    # aggregate: answer-permutation null for each readout (hit rate against other items' answers
    # with a different answer), per block
    agg = {}
    for sk, blk in (("cities", "base"), ("languages", "depth")):
        ids = [i for i in per if per[i]["block"] == blk and sk in per[i]["co"]]
        a = {}
        for kind in ("raw", "rank1", "full", "lex"):
            hits = sum(per[i]["co"][sk][kind] == per[i]["co"][sk]["ans"] for i in ids)
            perm_rates = []
            for i in ids:
                oth = [per[j]["co"][sk]["ans"] for j in ids if per[j]["co"][sk]["ans"] != per[i]["co"][sk]["ans"]]
                perm_rates.append(sum(per[i]["co"][sk][kind] == o for o in oth) / len(oth))
            lexfail = [i for i in ids if per[i]["co"][sk]["lex"] != per[i]["co"][sk]["ans"]]
            a[kind] = {"hits": [hits, len(ids)], "perm_null_expect": round(sum(perm_rates), 2),
                       "hits_where_lex_fails": [sum(per[i]["co"][sk][kind] == per[i]["co"][sk]["ans"] for i in lexfail), len(lexfail)],
                       "perm_null_where_lex_fails": round(sum(perm_rates[ids.index(i)] for i in lexfail), 2)}
        agg[sk] = a
    # names country on correct base, and L_first medians with the 20-draw chance
    lf = {}
    for fam in ("d2", "d3-lm", "d3-per", "d4-work"):
        ids = [i for i in per if per[i]["family"] == fam]
        code = lambda x: LC + 1 if x is None else x
        lf[fam] = {"med_all": st.median([code(per[i]["L_first"]) for i in ids]),
                   "fake20_med_all": st.median([code(x) for i in ids for x in per[i]["L_first_fake20"]]),
                   "share_real_le_fake_median": None}
    res["models"][m] = {"L_c": LC, "kv_c": KC, "co": agg, "L_first": lf,
                        "rank_by_item": {i: per[i]["rank"] for i in per}, "L_first_by_item": {i: per[i]["L_first"] for i in per},
                        "seconds": round(time.monotonic() - t0, 1)}
    # token-space OV maps for the lineage test (Qwen3 only)
    if m.startswith("Qwen3"):
        T = torch.tensor(TOK)
        ET = E[T]
        ov = torch.zeros(L, KV, len(TOK), len(TOK), dtype=torch.float32)
        for x in range(L):
            gl = w.get(f"model.layers.{x}.input_layernorm.weight")
            n = ET * torch.rsqrt(ET.pow(2).mean(-1, keepdim=True) + 1e-6) * gl
            vw = w.get(f"model.layers.{x}.self_attn.v_proj.weight")              # (KV*dh, d)
            for g in range(KV):
                vv = n @ vw[g * dh:(g + 1) * dh].T                                 # (T, dh)
                out = vv @ WO[x][g].T                                              # (T, d)
                M = (out * gF) @ WU[T].T                                           # (T, T)
                M = M - M.mean(1, keepdim=True)
                ov[x, g] = M.float()
        OVS[m] = ov
    print(m, res["models"][m]["seconds"], flush=True)
    del E, WU, WO

if SMOKE:
    json.dump(res, open(OUT, "w"), default=str)
    print("SMOKE OK", {m: res["models"][m]["co"] for m in MODELS})
    sys.exit(0)
# lineage test: for model pairs, matched relative depth layers, corr(OV_A[x,g], OV_B[x',g'])
pairs = [("Qwen3-0.6B-Base", "Qwen3-1.7B-Base"), ("Qwen3-1.7B-Base", "Qwen3-4B-Base"), ("Qwen3-0.6B-Base", "Qwen3-4B-Base")]
lin = {}
for A, B in pairs:
    oa, ob = OVS[A], OVS[B]
    La, Lb = oa.shape[0], ob.shape[0]
    same, diff, kv6 = [], [], []
    for x in range(La):
        xb = round(x * (Lb - 1) / (La - 1))
        a = oa[x].reshape(8, -1)
        b = ob[xb].reshape(8, -1)
        a = (a - a.mean(1, keepdim=True)) / a.std(1, keepdim=True)
        b = (b - b.mean(1, keepdim=True)) / b.std(1, keepdim=True)
        C = (a @ b.T) / a.shape[1]                                  # (8, 8)
        same += C.diagonal().tolist()
        diff += [float(C[i, j]) for i in range(8) for j in range(8) if i != j]
        # does the best match of each A head sit at the same index?
        kv6.append(int(C.argmax(1).eq(torch.arange(8)).sum()))
    lin[f"{A}|{B}"] = {"mean_corr_same_kv": round(st.mean(same), 4), "mean_corr_diff_kv": round(st.mean(diff), 4),
                       "best_match_same_index_per_layer_mean_of_8": round(st.mean(kv6), 3)}
# carrier-layer KV6 specifically: corr of the carrier cells across models vs other heads at those layers
cl = {"Qwen3-0.6B-Base": 20, "Qwen3-1.7B-Base": 20, "Qwen3-4B-Base": 24}
cc = {}
for A, B in pairs:
    a = OVS[A][cl[A]].reshape(8, -1)
    b = OVS[B][cl[B]].reshape(8, -1)
    a = (a - a.mean(1, keepdim=True)) / a.std(1, keepdim=True)
    b = (b - b.mean(1, keepdim=True)) / b.std(1, keepdim=True)
    C = (a @ b.T) / a.shape[1]
    cc[f"{A}|{B}"] = [[round(float(z), 3) for z in row] for row in C]
# country-copy score of every kv at the carrier layer (diagonal rank on the 57 countries)
cidx = torch.tensor([TOK.index(t) for t in SETS["countries"]])
copy = {}
for m, ov in OVS.items():
    x = cl[m]
    copy[m] = []
    for g in range(8):
        M = ov[x, g][cidx][:, cidx]
        copy[m].append(int((M.argmax(1) == torch.arange(len(cidx))).sum()))
res["lineage"] = {"matched_depth": lin, "carrier_layer_corr_8x8": cc, "country_copy_top1_of_57_at_carrier_layer": copy}
res["seconds"] = round(time.monotonic() - T0, 1)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res["lineage"], indent=1))
for m in MODELS:
    print(m, json.dumps(res["models"][m]["co"]), json.dumps(res["models"][m]["L_first"]))
print("TOTAL", res["seconds"])
