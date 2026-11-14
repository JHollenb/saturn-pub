"""B1 sealed S5 forecast (prereg_b1.json): first-order forecast of the carrier-cell transplant
(arm b) on M = logit(answer) - logit(donor answer), from the 1c captures only (never the arms).
Replica and reverse pass as r1_forecast.py; carrier rows by the frozen rule (R_argmax, L20
h12+h13 over the span, t0 excluded) on the captured answer-row p.

python b1_forecast.py MODEL CAPDIR OUT
"""
import json
import math  # noqa: F401 - spacing() below, copied from r1_forecast
import sys
import time
from pathlib import Path

import torch

from mdb.repl.__main__ import _load

MODEL, CAPDIR, OUT = sys.argv[1:4]
T0 = time.monotonic()
dbg = _load(Path(MODEL), "fp32", "cpu", True, None)
ad = dbg.adapter
L, H, KV, DH = ad.layers, ad.num_heads, ad.num_kv_heads, ad.head_dim
blocks = getattr(ad.backbone, ad.spec.layers)
final = getattr(ad.backbone, ad.spec.final_norm)
W_U = ad.model.lm_head.weight
theta = float(getattr(ad.model.config, "rope_theta", 1e6))
inv_freq = 1.0 / (theta ** (torch.arange(0, DH, 2, dtype=torch.float64) / DH))
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



def safe(i):
    return i.replace(":", "_").replace(" ", "_").replace("/", "_")


def carrier_row(cap, span):
    p = cap["L20.p"].reshape(H, -1)
    rows = [r for r in span if r != 0]
    return max(rows, key=lambda r: (float(p[12, r] + p[13, r]), r))


T = json.load(open("b1_sets.json"))["truths"]
DON = json.load(open("b1_donors.json"))
out = {}
for iid in sorted(T):
    t = time.monotonic()
    cap = torch.load(Path(CAPDIR) / "cap" / "depth" / f"{safe(iid)}.pt", map_location="cpu", weights_only=False)
    did = DON[iid]["cell"]
    dcap = torch.load(Path(CAPDIR) / "cap" / "depth" / f"{safe(did)}.pt", map_location="cpu", weights_only=False)
    span, dspan = T[iid]["span"], T[did]["span"]
    j = int(cap["j"])
    crow, drow = carrier_row(cap, span), carrier_row(dcap, dspan)
    K = [cap[f"k{x}"][0].float() for x in range(L)]
    V = [cap[f"v{x}"][0].float() for x in range(L)]
    a, b = T[iid]["language"], T[did]["language"]
    err = []
    for x in range(L):
        ho = layer(x, cap[f"L{x}.h"].reshape(-1), K[x], V[x], j)
        ref = cap[f"L{x}.h_out"].reshape(-1)
        err.append(float((ho - ref).norm() / ref.norm()))
    taps = [torch.zeros(ad.model.config.hidden_size, requires_grad=True) for _ in range(L)]
    h = cap["L0.h"].reshape(-1).clone()
    with torch.enable_grad():
        for x in range(L):
            h = layer(x, h, K[x], V[x], j, taps[x])
        M = (W_U[a] - W_U[b]) @ rms(h, final.weight, final.variance_epsilon)
        M.backward()
    g20 = taps[20].grad
    p = cap["L20.p"].reshape(H, -1)
    dv = dcap["v20"][0, 6, drow].float() - V[20][6, crow]
    fc = sum(float(g20 @ wo_block(20, h_, p[h_, crow] * dv)) for h_ in range(H) if h_ // (H // KV) == 6)
    out[iid] = {"carrier_row": crow, "donor": did, "donor_row": drow, "a": a, "b": b,
                "forecast_dM": fc, "M_replica": float(M), "replica_err_max": max(err),
                "seconds": round(time.monotonic() - t, 2)}
    print("FC " + json.dumps({"id": iid, "err_max": round(max(err), 4), "s": out[iid]["seconds"]}), flush=True)
json.dump(out, open(OUT, "w"), indent=1)
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
