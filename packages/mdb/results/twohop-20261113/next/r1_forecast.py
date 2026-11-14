"""Offline, sealed first-order forecasts of every Run 1 arm from the captures only (prereg_run1.json).

python r1_forecast.py MODEL CAPDIR GROUPS OUT

Reads only CAPDIR/cap/<group>/*.pt (native state + pre-edit operands) and r1_items.json; never
the arms files. A float32 replica of the answer-row computation (other rows' K/V fixed from the
capture; the row's own K/V recomputed) is validated against the captured ports, then one
reverse pass of M = (W_U[a]-W_U[b]) . final_norm(h_out_27) gives dM/do_l, dM/dm_l, dM/dh_20;
each arm's forecast is the gradient . its pre-edit change (P07 cells, P09 cuts, W_down Delta f).
"""
import json
import math
import sys
import time
from pathlib import Path

import torch

from mdb.repl.__main__ import _load

MODEL, CAPDIR, GROUPS, OUT = sys.argv[1:5]
T0 = time.monotonic()
dbg = _load(Path(MODEL), "fp32", "cpu", True, None)
ad = dbg.adapter
L, H, KV, DH = ad.layers, ad.num_heads, ad.num_kv_heads, ad.head_dim
blocks = getattr(ad.backbone, ad.spec.layers)
final = getattr(ad.backbone, ad.spec.final_norm)
W_U = ad.model.lm_head.weight
theta = float(getattr(ad.model.config, "rope_theta", 1e6))
inv_freq = 1.0 / (theta ** (torch.arange(0, DH, 2, dtype=torch.float64) / DH))
items = {i["id"]: i for i in json.load(open("r1_items.json"))["items"]}
print(f"load {time.monotonic() - T0:.1f}s", flush=True)
for p_ in ad.model.parameters():
    p_.requires_grad_(False)


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, pos):
    f = pos * inv_freq
    emb = torch.cat([f, f]).float()
    cos, sin = emb.cos(), emb.sin()
    x1, x2 = x[..., : DH // 2], x[..., DH // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def layer(x_, h, K, V, j, tap_o=None, tap_m=None, keep=None):
    blk = blocks[x_]
    at = blk.self_attn
    xn = rms(h, blk.input_layernorm.weight, blk.input_layernorm.variance_epsilon)
    q = (at.q_proj.weight @ xn).view(H, DH)
    k = (at.k_proj.weight @ xn).view(KV, DH)
    v = (at.v_proj.weight @ xn).view(KV, DH)
    q = rms(q, at.q_norm.weight, at.q_norm.variance_epsilon)
    k = rms(k, at.k_norm.weight, at.k_norm.variance_epsilon)
    q, k = rope(q, j), rope(k, j)
    Ka = torch.cat([K[:, :j], k[:, None]], 1)       # (KV, j+1, DH)
    Va = torch.cat([V[:, :j], v[:, None]], 1)
    g = torch.arange(H) // (H // KV)
    s = torch.einsum("hd,hnd->hn", q, Ka[g]) * DH ** -0.5
    p = torch.softmax(s.float(), -1)
    A = torch.einsum("hn,hnd->hd", p, Va[g])
    o = at.o_proj.weight @ A.reshape(-1)
    if tap_o is not None:
        o = o + tap_o
    r = h + o
    mlp = blk.mlp
    n = rms(r, blk.post_attention_layernorm.weight, blk.post_attention_layernorm.variance_epsilon)
    gate, up = mlp.gate_proj.weight @ n, mlp.up_proj.weight @ n
    f = torch.nn.functional.silu(gate) * up
    m = mlp.down_proj.weight @ f
    if tap_m is not None:
        m = m + tap_m
    if keep is not None:
        keep[x_] = {"p": p.detach(), "A": A.detach(), "o": o.detach(), "h_out": (r + m).detach(),
                    "f": f.detach()}
    return r + m


def wo_block(x_, h, delta):
    """W_O,h . delta (delta: (DH,))."""
    W = blocks[x_].self_attn.o_proj.weight
    return W[:, h * DH:(h + 1) * DH] @ delta


def spacing(x):
    x = abs(x)
    return 2.0 ** (math.floor(math.log2(x)) - 7) if x > 0 else 2.0 ** -133


out = {}
L0PRED = {}
for group in GROUPS.split(","):
    for path in sorted(Path(CAPDIR, "cap", group).glob("*.pt")):
        t = time.monotonic()
        cap = torch.load(path, map_location="cpu", weights_only=False)
        ids = cap["ids"].tolist()
        j = int(cap["j"])
        item = next(i for i in items.values() if i["ids"] == ids and i["group"] == group)
        K = [cap[f"k{x}"][0].float() for x in range(L)]
        V = [cap[f"v{x}"][0].float() for x in range(L)]
        toks = cap["toks"].tolist()
        a, b = toks
        # validation: teacher-forced per layer
        keep = {}
        err = []
        for x in range(L):
            h = cap[f"L{x}.h"].reshape(-1)
            ho = layer(x, h, K[x], V[x], j, keep=keep)
            ref = cap[f"L{x}.h_out"].reshape(-1)
            err.append(float((ho - ref).norm() / ref.norm()))
        # free run with taps
        taps_o = [torch.zeros(ad.model.config.hidden_size, requires_grad=True) for _ in range(L)]
        taps_m = [torch.zeros(ad.model.config.hidden_size, requires_grad=True) for _ in range(L)]
        h = cap["L0.h"].reshape(-1).clone()
        h20 = None
        with torch.enable_grad():
            for x in range(L):
                if x == 20:
                    h = h.detach().requires_grad_(True)
                    h20 = h
                h = layer(x, h, K[x], V[x], j, taps_o[x], taps_m[x])
            hn = rms(h, final.weight, final.variance_epsilon)
            M = (W_U[a] - W_U[b]) @ hn
            M.backward()
        M_native = float(cap["logits"][a] - cap["logits"][b])
        go = [tp.grad for tp in taps_o]
        gm = [tp.grad for tp in taps_m]
        gh20 = h20.grad
        p_cap = {x: cap[f"L{x}.p"].reshape(H, -1)[:, :j + 1] for x in range(L)}
        A_cap = {x: cap[f"L{x}.A"].reshape(H, DH) for x in range(L)}
        rec = {"id": item["id"], "group": group, "M_native": M_native, "M_replica": float(M),
               "replica_rel_err_by_layer": err}
        g_of = torch.arange(H) // (H // KV)

        def cell_forecast(x_, r, gg, dv):
            dvv = dv.float() - V[x_][gg, r]
            do = sum(wo_block(x_, h_, p_cap[x_][h_, r] * dvv) for h_ in range(H) if int(g_of[h_]) == gg)
            return float(go[x_] @ do)
        if "donor_v" in cap:
            dvs = cap["donor_v"]
            relay = int(cap["relay"])
            span = item["span"]
            fc = {}
            for r in list(span) + [relay]:
                for x in range(4, L):
                    rd = min(r, dvs[x].shape[2] - 1)
                    for gg in range(KV):
                        fc[f"cell|{r}|{x}|{gg}"] = cell_forecast(x, r, gg, dvs[x][0, gg, rd])
            rd = min(relay, dvs[0].shape[2] - 1)
            fc[f"relay_band|{relay}"] = sum(cell_forecast(x, relay, gg, dvs[x][0, gg, rd])
                                            for x in range(L) for gg in range(KV))
            for name, row in (("t0", 0), ("relay", relay)):
                for kind in ("zero", "mask"):
                    tot, tot32 = 0.0, 0.0
                    for x in range(L):
                        for h_ in range(H):
                            gg = int(g_of[h_])
                            vr = V[x][gg, row]
                            for src, store in ((p_cap[x], "bf"), (keep[x]["p"], "f32")):
                                pj = src[h_, row]
                                A = A_cap[x][h_] if store == "bf" else keep[x]["A"][h_]
                                dA = -pj * vr if kind == "zero" else pj * (A - vr) / (1 - pj)
                                val = float(go[x] @ wo_block(x, h_, dA))
                                if store == "bf":
                                    tot += val
                                else:
                                    tot32 += val
                    fc[f"cut|{name}_{kind}"] = tot
                    fc[f"cut32|{name}_{kind}"] = tot32
            if "swap_values" in cap:
                r0 = item["row"]
                for n_, val in cap["swap_values"].items():
                    fc[f"swap|{n_}"] = cell_forecast(20, r0, 6, val.reshape(-1))
                fc["2x2|B"] = float(gh20 @ (cap["donor_h20"].reshape(-1) - cap["L20.h"].reshape(-1)))
                fc["2x2|AB"] = fc["swap|donor0"] + fc["2x2|B"]
            if "knockout_sets" in cap:
                for n_, sets in cap["knockout_sets"].items():
                    tot = 0.0
                    for x, idx in sets.items():
                        x = int(x)
                        f_nat = cap[f"L{x}.f"].reshape(-1)
                        df = torch.zeros_like(f_nat)
                        df[idx] = cap["fmean"][x].reshape(-1)[idx] - f_nat[idx]
                        tot += float(gm[x] @ (blocks[x].mlp.down_proj.weight @ df))
                    fc[f"knockout|{n_}"] = tot
            # L0 closed forms (the cut's first layer sees the native input): predicted Delta o_0
            # with the captured bf16 p and with the replica's float32 p, for the f32-renorm check
            l0 = {}
            for name, row in (("t0", 0), ("relay", relay)):
                for kind in ("zero", "mask"):
                    for store in ("bf", "f32"):
                        do = 0
                        for h_ in range(H):
                            vr = V[0][int(g_of[h_]), row]
                            pj = (p_cap[0] if store == "bf" else keep[0]["p"])[h_, row]
                            A = A_cap[0][h_] if store == "bf" else keep[0]["A"][h_]
                            dA = -pj * vr if kind == "zero" else pj * (A - vr) / (1 - pj)
                            do = do + wo_block(0, h_, dA)
                        l0[f"{name}_{kind}|{store}"] = do.detach().clone()
            L0PRED[item["id"]] = l0
            rec["forecast"] = fc
        rec["seconds"] = round(time.monotonic() - t, 2)
        out[item["id"]] = rec
        print("FC " + json.dumps({"id": item["id"], "err_max": max(err), "M": [M_native, float(M)],
                                  "n": len(rec.get("forecast", {})), "s": rec["seconds"]}), flush=True)
json.dump(out, open(OUT, "w"))
torch.save(L0PRED, OUT.replace(".json", "-l0pred.pt"))
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
