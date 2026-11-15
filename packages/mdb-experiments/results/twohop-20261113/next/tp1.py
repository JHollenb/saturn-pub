"""TP1: token x position x sensitivity map (next/prereg_tp1.json + prereg_tp1_addendum.json).

python tp1.py MODEL SAEDIR OUTDIR

Which library does what:
- nnsight (0.8.2, via mdb_interop.nn.wrap on a weight-sharing clone): the one forward + ONE
  backward per prompt; saves h (each layer's input and the final-norm input), v_proj output,
  attention p, and dM/dh for every row and layer (M = top1 - top2 logit at the last row).
- mdb (exact parts): the gradient check against mdb's own forward-mode Jacobian
  (propagate.whatif linear term, P14) on 2 items; the per-position / per-token / global mean
  ablations as native stepped runs (Run.set at t<r>.L14.h, prefill=r); authority labels.
- SAELens (6.55.5, TopKSAE built from the Qwen-Scope weights): reconstruction error per row at
  layers 0, 4, 8, 14, 20, 24, 27 (hook resid_post = h_out); checked against mdb's
  analysis.sae FeatureBasis.select (active-set equality) on a few rows.
- TransformerLens (4.4.0 TransformerBridge, separate float32 CPU copy, compatibility mode:
  fold_ln, center_writing_weights, center_unembed): the centred residual norms of 8 prompts
  against the native ones and against the identity h - mean_d(h).
Per-row arrays go to OUTDIR (torch.save per prompt) for offline re-slicing.
"""
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, "interop_pkg")
from mdb_interop import nn as inn  # noqa: E402
from mdb_interop.clone import shared_clone  # noqa: E402

from mdb.analysis import sae as msae  # noqa: E402
from mdb.analysis.forecast import Margin  # noqa: E402
from mdb.analysis.propagate import Edit, whatif  # noqa: E402
from mdb.repl.__main__ import _load  # noqa: E402

MODEL, SAEDIR, OUTDIR = sys.argv[1:4]
MID = 14
SAE_LAYERS = (0, 4, 8, 14, 20, 24, 27)
T0 = time.monotonic()
OUT = Path(OUTDIR)
OUT.mkdir(parents=True, exist_ok=True)
timing = {}


def mark(name):
    timing[name] = round(time.monotonic() - T0, 1)
    print(f"[{timing[name]:7.1f}s] {name}", flush=True)


import os  # noqa: E402
# main: nnsight + mdb check + SAELens; abl: one mdb ablation shard; tl: TransformerLens arm on CPU
# (its float32 copy + in-place compatibility processing peaked > 21 GB RAM, job-97d34f7eb3b5)
PART = os.environ.get("TP1_PART", "main")
SHARD, NSHARD = int(os.environ.get("TP1_SHARD", 0)), int(os.environ.get("TP1_NSHARD", 1))
if PART == "tl":
    from transformers import AutoConfig, AutoTokenizer
    tok, cfg = AutoTokenizer.from_pretrained(MODEL), AutoConfig.from_pretrained(MODEL)
else:
    dbg = _load(Path(MODEL), "bf16", "cuda", True, None)
    tok, model = dbg.tokenizer, dbg.adapter.model
    cfg = model.config
    for prm in model.parameters():  # only input grads are needed; weight grads cost +3.4 GB (job-b3bc03382981)
        prm.requires_grad_(False)
    nm = inn.wrap(shared_clone(model))
L, H, KV, DH = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
prompts = json.load(open("tp1_prompts.json"))["prompts"]
if os.environ.get("TP1_N"):
    prompts = prompts[:int(os.environ["TP1_N"])]
mark("load")

# ------------------------------------------------------------------ nnsight forward + backward
summary, h_mid, gnorm_mid = [], {}, {}
for k, pr in enumerate(prompts if PART == "main" else []):
    ids = tok(pr["text"], add_special_tokens=False)["input_ids"]
    n = len(ids)
    x = torch.tensor([ids], device="cuda")
    # trace-body locals do not reach this frame (job-9044d2a117a4, job-ba98642a9496): every value
    # that leaves the trace goes into a container created out here
    grads, hs_s, vs, ps, tt = [], [], [], [], {}
    with nm.trace(x):
        emb = nm.model.embed_tokens.output
        emb.requires_grad_(True)
        hs = []
        for i in range(L):
            hs.append(nm.model.layers[i].input)
            vs.append(nm.model.layers[i].self_attn.v_proj.output.save())
            ps.append(nm.model.layers[i].self_attn.output[1].save())
        hs.append(nm.model.norm.input)
        for h in hs:
            hs_s.append(h.save())
        lg = nm.lm_head.output[0, -1].float()
        top = lg.topk(2)
        tt["v"], tt["i"] = top.values.save(), top.indices.save()
        M = top.values[0] - top.values[1]
        with M.backward():
            for h in reversed(hs):
                grads.append(h.grad.save())
    grads = list(reversed(grads))
    tv, ti = tt["v"], tt["i"]
    hstack = torch.stack([h[0].detach().float() for h in hs_s])            # (L+1, n, d)
    gstack = torch.stack([g[0].detach().float() for g in grads])
    att = torch.stack([p[0].detach().float() for p in ps])                 # (L, H, n, n)
    g_of = torch.arange(H, device=att.device) // (H // KV)
    vnorm, vsa = [], []
    for i in range(L):
        v = vs[i][0].detach().float().view(n, KV, DH)
        vnorm.append(v.norm(dim=-1).sum(-1).cpu())
        pl = att[i, :, -1, :]                                              # (H, n)
        A = torch.einsum("hs,shd->hd", pl, v[:, g_of, :])
        vsa.append((pl.T * (v[:, g_of, :] - A[None]).norm(dim=-1)).sum(-1).cpu())
    t1, t2 = [int(t) for t in ti]
    rec = {"k": k, "name": pr["name"], "kind": pr["kind"], "ids": ids, "t1": t1, "t2": t2,
           "M": float(tv[0] - tv[1])}
    torch.save({**rec, "h": hstack.to(torch.bfloat16).cpu(), "grad": gstack.to(torch.bfloat16).cpu(),
                "p_last": att[:, :, -1, :].cpu(), "received": att.sum(1).sum(1).cpu(),
                "vnorm": torch.stack(vnorm), "v_minus_a": torch.stack(vsa)}, OUT / f"p{k:03d}.pt")
    h_mid[k] = hstack[MID].cpu()
    gnorm_mid[k] = gstack[MID].norm(dim=-1).cpu()
    summary.append(rec)
    del hstack, gstack, att, grads, hs_s, vs, ps
if PART == "main":
    torch.save({"h_mid": h_mid, "gnorm_mid": gnorm_mid, "summary": summary}, OUT / "mid.pt")
else:
    st_ = torch.load(OUT / "mid.pt", weights_only=False)
    h_mid, gnorm_mid, summary = st_["h_mid"], st_["gnorm_mid"], st_["summary"]
mark("nnsight forward/backward")

# ------------------------------------------------------------------ mdb Jacobian check (2 items)
checks = []
for k in ((0, 1) if PART == "main" else ()):
    s = summary[k]
    ids, j = s["ids"], len(s["ids"]) - 1
    g20 = torch.load(OUT / f"p{k:03d}.pt", weights_only=False)["grad"][20, j].float()
    rng = torch.Generator().manual_seed(k)
    for name, u in (("random", torch.randn(g20.numel(), generator=rng)), ("grad_dir", g20.clone())):
        u = u / u.norm()
        run = dbg.run(ids, name="chk")
        run.until(f"t{j}.L20.h")
        res = whatif(run, Edit(f"t{j}.L20.h", u.reshape(1, 1, -1).to(torch.bfloat16).cuda(), "add"),
                     Margin(j, s["t1"], s["t2"]))
        lin = res.report["prediction"].get("linear", {}).get("value", {})
        lin = lin.get("value") if isinstance(lin, dict) else lin
        written = u.to(torch.bfloat16).float()
        checks.append({"prompt": k, "direction": name, "nnsight_grad_dot_u": float(g20 @ written),
                       "mdb_linear": lin, "mdb_native_change": res.report.get("effect", {}).get("native")})
mark("mdb Jacobian check")

# ------------------------------------------------------------------ mdb mean ablations at L14 input
allrows = [(k, r) for k in h_mid for r in range(h_mid[k].shape[0])]
gmean = torch.stack([h_mid[k][r] for k, r in allrows]).mean(0)
by_pos, by_tok = defaultdict(list), defaultdict(list)
for k, r in allrows:
    by_pos[r].append(h_mid[k][r])
    by_tok[summary[k]["ids"][r]].append(h_mid[k][r])
pos_mean = {r: torch.stack(v).mean(0) for r, v in by_pos.items() if len(v) >= 3}
tok_mean = {t: torch.stack(v).mean(0) for t, v in by_tok.items() if len(v) >= 3}
norms = sorted(((float(h_mid[k][r].norm()), k, r) for k, r in allrows
                if r < h_mid[k].shape[0] - 1), reverse=True)
top = norms[:max(1, len(norms) // 100)]
half = len(norms) // 2
med = norms[half - len(top) // 2: half - len(top) // 2 + len(top)]


def ablate(ids, row, vec, t1, t2):
    j = len(ids) - 1
    run = dbg.run(ids, name="abl", prefill=row) if row > 0 else dbg.run(ids, name="abl", prefill=0)
    run.until(f"t{row}.L{MID}.h")
    run.set(f"t{row}.L{MID}.h", vec.reshape(1, 1, -1).to(torch.bfloat16).cuda(), "replace")
    run.until(f"t{j}.logits")
    lg = run.read().reshape(-1).float()
    return float(lg[t1] - lg[t2]), int(lg.argmax()) == t1


abl = []
work = [(tag, x) for tag, rows in (("top1pct", top), ("median", med)) for x in rows]
work = [w for n_, w in enumerate(work) if PART == "abl" and n_ % NSHARD == SHARD]
for tag, (nrm, k, r) in work:
    if True:
        s = summary[k]
        rec = {"set": tag, "prompt": k, "row": r, "token": s["ids"][r], "token_text": tok.decode([s["ids"][r]]),
               "norm": nrm, "M0": s["M"], "grad_norm": float(gnorm_mid[k][r])}
        for name, vec in (("global", gmean), ("position", pos_mean.get(r)),
                          ("token", tok_mean.get(s["ids"][r]))):
            if vec is None:
                rec[name] = None
                continue
            m1, kept = ablate(s["ids"], r, vec, s["t1"], s["t2"])
            rec[name] = {"dM": m1 - s["M"], "argmax_kept": kept}
        abl.append(rec)
mark("mdb ablations")

# ------------------------------------------------------------------ SAELens reconstruction error
sae_out = {"layers": list(SAE_LAYERS), "per_prompt": {}, "check": []}
try:
    if PART != "main":
        raise RuntimeError("not run in this part")
    from sae_lens.saes.topk_sae import TopKSAE, TopKSAEConfig
    mset = msae.open_set("qwen-scope", SAEDIR)
    for layer in SAE_LAYERS:
        state = torch.load(Path(SAEDIR) / f"layer{layer}.sae.pt", map_location="cpu", weights_only=True)
        scfg = TopKSAEConfig(d_in=state["W_dec"].shape[0], d_sae=state["W_enc"].shape[0], k=50,
                             dtype="float32", device="cuda", apply_b_dec_to_input=False)
        sl = TopKSAE(scfg)
        sl.load_state_dict({"W_enc": state["W_enc"].T.contiguous(), "W_dec": state["W_dec"].T.contiguous(),
                            "b_enc": state["b_enc"], "b_dec": state["b_dec"]}, strict=False)
        sl = sl.to("cuda").eval()
        basis = mset.get(layer)
        for k, s in enumerate(summary):
            hh = torch.load(OUT / f"p{k:03d}.pt", weights_only=False)["h"][layer + 1].float().cuda()
            with torch.no_grad():
                acts = sl.encode(hh)
                rec_ = sl.decode(acts)
            err = (hh - rec_).norm(dim=-1) / hh.norm(dim=-1).clamp_min(1e-9)
            sae_out["per_prompt"].setdefault(k, {})[layer] = err.cpu().tolist()
            if k < 2:
                for r in (0, len(s["ids"]) - 1):
                    ids_m, _ = basis.select(hh[r].cpu())
                    ids_s = sorted(int(i) for i in acts[r].nonzero().reshape(-1).tolist())
                    sae_out["check"].append({"prompt": k, "layer": layer, "row": r,
                                             "same_active_set": sorted(ids_m) == ids_s,
                                             "n_mdb": len(ids_m), "n_saelens": len(ids_s)})
        del sl
        torch.cuda.empty_cache()
    sae_out["state"] = "done"
except Exception as error:  # noqa: BLE001 - reported
    sae_out["state"] = f"failed: {type(error).__name__}: {error}"[:400]
mark("SAELens")

# ------------------------------------------------------------------ TransformerLens centring arm
tl_out = {}
try:
    if PART != "tl":
        raise RuntimeError("not run in this part")
    from transformers import AutoModelForCausalLM

    from mdb_interop import tl as itl
    cpu_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    br = itl.bridge(cpu_model, tok, in_place=True)
    br.enable_compatibility_mode(disable_warnings=True)
    tl_out["processed"] = itl.processed(br)
    rows = []
    for k in random.Random(0).sample(range(len(summary)), 8):
        s = summary[k]
        with torch.no_grad():
            _, cache = br.run_with_cache(torch.tensor([s["ids"]]))
        saved = torch.load(OUT / f"p{k:03d}.pt", weights_only=False)["h"].float()
        for layer in (1, 7, 14, 21, 27):
            key = f"blocks.{layer}.hook_resid_pre"
            c = cache[key][0].float() if key in cache else None
            if c is None:
                continue
            nat = saved[layer]
            ident = nat - nat.mean(-1, keepdim=True)
            rows.append({"prompt": k, "layer": layer,
                         "tl_norm": c.norm(dim=-1).tolist(), "native_norm": nat.norm(dim=-1).tolist(),
                         "identity_rel_err": float((c - ident).norm() / ident.norm())})
    tl_out["rows"] = rows
    tl_out["state"] = "done"
except Exception as error:  # noqa: BLE001 - reported
    tl_out["state"] = f"failed: {type(error).__name__}: {error}"[:400]
mark("TransformerLens")

json.dump({"summary": summary, "checks": checks, "ablations": abl, "sae": sae_out, "tl": tl_out,
           "timing": timing, "versions": {"nnsight": inn.version()}},
          open({"main": "tp1-results.json", "tl": "tp1-tl.json"}.get(PART, f"tp1-abl-{SHARD}.json"), "w"))
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)
