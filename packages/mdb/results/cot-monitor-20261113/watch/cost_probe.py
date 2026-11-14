"""B2 prep: per-step cost of (A) mdb stepped rows (symbols table + native hint ablation per row) vs
(B) a batched exact per-row share of declared spans (hooks, one forward) and (C) a batched native
per-row hint ablation (one masked forward against a cache whose hint V cells are zeroed).

TIMING AND EXACTNESS ONLY: the log prints seconds and max |difference| between methods, never a
monitor score or a margin. Consumer at row t: M[X, R] on row t's letter-token logits (R fixed to the
item's no-hint letter if != X else the correct letter; a placeholder until the prereg defines it).

Usage: python cost_probe.py MODEL_DIR SUBSET_JSON UID[,UID] [--rows N]   (or --tiny for a CPU check)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time

import torch

SUFFIX = "\n</think>\n\nThe answer is ("
LET = "ABCD"


def spans_for(tok, text, ids, body, hint):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    assert enc["input_ids"] == ids
    b0 = text.index(body)
    b1 = b0 + len(body)
    h0, h1 = b1, b1 + len(hint)
    sp = {"question": [], "hint": []}
    for t, (a, b) in enumerate(enc["offset_mapping"]):
        if t and a < b1 and b > b0:
            sp["question"].append(t)
        elif t and a < h1 and b > h0:
            sp["hint"].append(t)
    return sp


def batched_shares(model, ids, spans, u, dev):
    """Exact per-row direct contribution of each span (dict name -> rows) to consumer direction u
    (d,), read through the final norm: c_t = rho_t (gamma * u). Hooks on every layer's v_proj and
    self_attn (eager returns the weights), reduced on the fly; float64. Returns {name: (T,)}."""
    cfg = model.config
    H, G = cfg.num_attention_heads, cfg.num_key_value_heads
    dh = cfg.hidden_size // H
    layers = model.model.layers
    gamma = model.model.norm.weight.double()
    gu = gamma * u.double()
    T = len(ids)
    acc = {k: torch.zeros(T, dtype=torch.float64, device=dev) for k in spans}
    masks = {k: torch.zeros(T, dtype=torch.float64, device=dev) for k in spans}
    for k, rows in spans.items():
        masks[k][rows] = 1.0
    store = {}
    hooks = []
    for li, layer in enumerate(layers):
        att = layer.self_attn
        wo = att.o_proj.weight.double().reshape(cfg.hidden_size, H, dh)       # (d, H, dh)
        back = torch.einsum("dhe,d->he", wo, gu)                              # (H, dh)

        def vh(mod, inp, out, li=li):
            store[li] = out[0].double().reshape(T, G, dh)

        def ah(mod, inp, out, li=li, back=back):
            p = out[1][0].double()                                            # (H, T, T)
            v = store.pop(li)                                                 # (T, G, dh)
            for h in range(H):
                vb = v[:, h // (H // G), :] @ back[h]                         # (T,) value . W_o,h^T gu
                for k in spans:
                    acc[k] += p[h] @ (vb * masks[k])                         # (T,) per reader row

        hooks.append(att.v_proj.register_forward_hook(vh))
        hooks.append(att.register_forward_hook(ah))
    def nh(mod, inp, out):
        store["pre_norm"] = inp[0][0].double()

    hooks.append(model.model.norm.register_forward_hook(nh))
    try:
        model.model(input_ids=torch.tensor([ids], device=dev))
    finally:
        for hk in hooks:
            hk.remove()
    hlast = store.pop("pre_norm")                                             # final-norm input
    rho = (hlast.pow(2).mean(-1) + model.model.norm.variance_epsilon) ** -0.5
    return {k: acc[k] * rho for k in spans}


def batched_ablation(model, ids, hint_rows, lt, dev, xi, ri):
    """Native per-row hint ablation for every row in one masked forward: row t attends to the native
    K/V of rows < t with the hint rows' V zeroed, and to its own (recomputed) k/v. Returns the
    native and ablated M[X,R] per row (T,)."""
    T = len(ids)
    x = torch.tensor([ids], device=dev)
    nat = model(input_ids=x, use_cache=True)
    cache = nat.past_key_values
    for lay in cache.layers:
        lay.values[:, :, hint_rows] = 0
    neg = torch.finfo(torch.float32).min
    m = torch.full((T, 2 * T), neg, device=dev)
    ar = torch.arange(T, device=dev)
    m[:, :T] = torch.where(ar[None, :] < ar[:, None], 0.0, neg)               # cache rows s < t
    m[ar, T + ar] = 0.0                                                       # its own new row
    pos = ar[None]
    o = model(input_ids=x, past_key_values=cache, attention_mask=m[None, None], position_ids=pos)
    ln = nat.logits[0][:, lt].double()
    la = o.logits[0][:, lt].double()
    return ln[:, xi] - ln[:, ri], la[:, xi] - la[:, ri]


def loop_ablation(model, ids, hint_rows, lt, dev, xi, ri, rows):
    """Reference for batched_ablation at a few rows: prefill ids[:t], zero hint V, run token t."""
    out = []
    for t in rows:
        c = model(input_ids=torch.tensor([ids[:t]], device=dev), use_cache=True).past_key_values
        for lay in c.layers:
            lay.values[:, :, hint_rows] = 0
        o = model(input_ids=torch.tensor([[ids[t]]], device=dev), past_key_values=c,
                  position_ids=torch.tensor([[t]], device=dev))
        lg = o.logits[0, -1][lt].double()
        out.append(float(lg[xi] - lg[ri]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("subset", nargs="?")
    ap.add_argument("uids", nargs="?")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    import mdb
    from mdb.analysis import symbols
    from mdb.analysis.forecast import Margin
    if a.tiny:
        dbg = mdb.open_tiny("qwen2", "fp32")
        model, dev = dbg.model, "cpu"
        ids = [(i * 7) % 31 + 1 for i in range(60)]
        cases = [("tiny", ids, {"question": list(range(1, 5)), "hint": list(range(5, 9))}, 10, 4, 9, 2)]
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager").to("cuda").eval()
        dev = "cuda"
        dbg = mdb.attach(model, tok)
        letter_ids = [tok.encode(SUFFIX + c, add_special_tokens=False)[-1] for c in LET]
        items = {it["uid"]: it for it in json.loads(open(a.subset).read())["items"]}
        cases = []
        for uid in a.uids.split(","):
            it = items[uid]
            p = tok.encode(it["prompt"], add_special_tokens=False)
            ch = tok.encode(it["chain"], add_special_tokens=False)
            sp = spans_for(tok, it["prompt"], p, it["body"], it["hint"])
            xi = LET.index(it["x"])
            r = it["no_hint_letter"] if it["no_hint_letter"] != it["x"] else it["correct"]
            cases.append((uid[:12], p + ch, sp, len(p), letter_ids[xi], letter_ids[LET.index(r)], len(p)))
    for name, ids, sp, n_p, idx, idr, first in cases:
        T = len(ids)
        lt = torch.tensor([idx, idr], device=dev)
        # (A) mdb stepped rows: a few rows spread over the chain
        rows = [first + (T - first) * (j + 1) // (a.rows + 1) for j in range(a.rows)]
        ta = []
        for t in rows:
            t0 = time.time()
            run = dbg.run(ids[:t + 1])
            ck = run.checkpoint()
            run.until("logits")
            symbols.declare(run, "hint", [f"t{s}" for s in sp["hint"]])
            symbols.declare(run, "question", [f"t{s}" for s in sp["question"]])
            symbols.declare(run, "chain", [f"t{s}" for s in range(n_p, t + 1)])
            symbols.table(run, consumer=Margin(t, idx, idr), limit=16)
            t1 = time.time()
            run.restore(ck)
            for layer in range(model.config.num_hidden_layers):
                run.until(f"L{layer}.h")
                mk = torch.zeros(run.address_shape(f"L{layer}.p"), dtype=torch.bool)
                mk[..., sp["hint"]] = True
                run.set(f"L{layer}.p", 0.0, "replace", mask=mk)
            run.until("logits")
            ta.append((t, round(t1 - t0, 3), round(time.time() - t1, 3)))
            del run
        # (B) batched exact shares, all rows, one forward
        u = (model.lm_head.weight[idx] - model.lm_head.weight[idr]).to(dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        shares = batched_shares(model, ids, {"hint": sp["hint"], "question": sp["question"], "chain": list(range(n_p, T))}, u, dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        tb = time.time() - t0
        # exactness of (B) vs mdb symbols at the last sampled row (difference only)
        t = rows[-1]
        run = dbg.run(ids[:t + 1])
        run.until("logits")
        symbols.declare(run, "hint", [f"t{s}" for s in sp["hint"]])
        tab = symbols.table(run, consumer=Margin(t, idx, idr), limit=16)
        mdb_hint = next(b["to_consumer"]["value"] for b in tab.report["bindings"] if b["symbol"] == "hint")
        diff_b = abs(float(shares["hint"][t]) - mdb_hint)
        del run
        # (C) batched native per-row hint ablation, one masked forward
        t0 = time.time()
        nat, abl = batched_ablation(model, ids, sp["hint"], lt, dev, 0, 1)
        if dev == "cuda":
            torch.cuda.synchronize()
        tc = time.time() - t0
        ref = loop_ablation(model, ids, sp["hint"], lt, dev, 0, 1, rows)
        diff_c = max(abs(float(abl[t]) - r) for t, r in zip(rows, ref))
        peak = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0
        print(json.dumps({"case": name, "T": T, "chain_rows": T - n_p,
                          "A_mdb_per_row_s(table, ablation)": ta,
                          "B_batched_share_all_rows_s": round(tb, 3), "B_vs_mdb_symbols_abs_diff": diff_b,
                          "C_batched_ablation_all_rows_s": round(tc, 3), "C_vs_loop_abs_diff_max": diff_c,
                          "peak_cuda_mb": round(peak)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
