"""TP1: token x position x sensitivity map (next/prereg_tp1.json).

python tp1.py MODEL OUTDIR

96 prompts (32 two-hop items + 64 varied texts, tp1_prompts.json). Per prompt one forward and ONE
backward of M = top1 - top2 logit at the last row (HF eager model, hidden states as graph
leaves' descendants with retain_grad). Saved per prompt to OUTDIR (torch.save): h (layers+1, n,
d) bf16, dM/dh (layers+1, n, d) bf16, attention p from the last row (layers, H, n) and the
column-summed mass received over all query rows (layers, n), ||v|| per row (layers, n), the
last-row reader-weighted ||v - a|| (layers, n). Then: top-energy rows at L14 get three native
ablations of h at the L14 input (global mean, per-position mean, per-token mean).
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, OUTDIR = sys.argv[1:3]
MID = 14
T0 = time.monotonic()
OUT = Path(OUTDIR)
OUT.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager").cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)
cfg = model.config
L, H, KV, DH = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
layers = model.model.layers
prompts = json.load(open("tp1_prompts.json"))["prompts"]
print(f"load {time.monotonic() - T0:.1f}s; {len(prompts)} prompts", flush=True)

vcap = {}


def vhook(i):
    def f(mod, inp, out):
        vcap[i] = out.detach()
    return f


hooks = [layers[i].self_attn.v_proj.register_forward_hook(vhook(i)) for i in range(L)]
summary = []
h_mid = {}
for k, pr in enumerate(prompts):
    ids = tok(pr["text"], add_special_tokens=False)["input_ids"]
    n = len(ids)
    x = torch.tensor([ids]).cuda()
    emb = model.model.embed_tokens(x).detach().requires_grad_(True)
    with torch.enable_grad():
        out = model(inputs_embeds=emb, output_hidden_states=True, output_attentions=True)
        hs = out.hidden_states                       # L+1 tensors (1, n, d)
        for h in hs:
            h.retain_grad()
        lg = out.logits[0, -1].float()
        t1, t2 = lg.topk(2).indices.tolist()
        M = lg[t1] - lg[t2]
        M.backward()
    hstack = torch.stack([h[0].detach() for h in hs]).to(torch.bfloat16).cpu()
    gstack = torch.stack([h.grad[0] for h in hs]).to(torch.bfloat16).cpu()
    att = torch.stack([a[0].float() for a in out.attentions])          # (L, H, n, n)
    p_last = att[:, :, -1, :].cpu()                                    # (L, H, n)
    received = att.sum(1).sum(1).cpu()                                 # (L, n): over heads and queries
    vnorm, vsa = [], []
    for i in range(L):
        v = vcap[i][0].float().view(n, KV, DH)                         # (n, KV, DH)
        vnorm.append(v.norm(dim=-1).sum(-1).cpu())
        g = torch.arange(H, device=v.device) // (H // KV)
        A = torch.einsum("hs,shd->hd", att[i, :, -1, :].cuda(), v[:, g, :])  # (H, DH)
        diff = (v[:, g, :] - A[None]).norm(dim=-1)                     # (n, H)
        vsa.append((att[i, :, -1, :].cuda().T * diff).sum(-1).cpu())   # reader-weighted ||v - a||
    rec = {"ids": ids, "t1": t1, "t2": t2, "M": float(M), "kind": pr["kind"], "name": pr["name"]}
    torch.save({**rec, "h": hstack, "grad": gstack, "p_last": p_last, "received": received,
                "vnorm": torch.stack(vnorm), "v_minus_a": torch.stack(vsa)}, OUT / f"p{k:03d}.pt")
    h_mid[k] = hstack[MID].float()
    summary.append(rec)
    del out, hs, att
print(f"forward/backward {time.monotonic() - T0:.1f}s", flush=True)
for hk in hooks:
    hk.remove()
# means at the L14 input (hidden_states[MID] is the input of layer MID)
allrows = [(k, r) for k in h_mid for r in range(h_mid[k].shape[0])]
gmean = torch.stack([h_mid[k][r] for k, r in allrows]).mean(0)
by_pos, by_tok = defaultdict(list), defaultdict(list)
for k, r in allrows:
    by_pos[r].append(h_mid[k][r])
    by_tok[summary[k]["ids"][r]].append(h_mid[k][r])
pos_mean = {r: torch.stack(v).mean(0) for r, v in by_pos.items() if len(v) >= 3}
tok_mean = {t: torch.stack(v).mean(0) for t, v in by_tok.items() if len(v) >= 3}
# top-energy rows at MID (excluding the last row): top 1% of all rows by ||h||
norms = sorted(((float(h_mid[k][r].norm()), k, r) for k, r in allrows
                if r < h_mid[k].shape[0] - 1), reverse=True)
top = norms[:max(1, len(norms) // 100)]
# plus a matched sample of median-energy rows as the comparison
mid = norms[len(norms) // 2 - len(top) // 2: len(norms) // 2 + len(top) - len(top) // 2]


def run_replaced(ids, row, vec):
    def pre(mod, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        hidden = hidden.clone()
        hidden[0, row] = vec.to(hidden.dtype).to(hidden.device)
        if args:
            return (hidden,) + args[1:], kwargs
        kwargs["hidden_states"] = hidden
        return args, kwargs
    hk = layers[MID].register_forward_pre_hook(pre, with_kwargs=True)
    with torch.no_grad():
        lg = model(torch.tensor([ids]).cuda()).logits[0, -1].float()
    hk.remove()
    return lg


abl = []
for tag, rows in (("top1pct", top), ("median", mid)):
    for nrm, k, r in rows:
        s = summary[k]
        rec = {"set": tag, "prompt": k, "row": r, "token": s["ids"][r], "norm": nrm, "M0": s["M"]}
        for name, vec in (("global", gmean), ("position", pos_mean.get(r)),
                          ("token", tok_mean.get(s["ids"][r]))):
            if vec is None:
                rec[name] = None
                continue
            lg = run_replaced(s["ids"], r, vec)
            rec[name] = {"dM": float(lg[s["t1"]] - lg[s["t2"]]) - s["M"],
                         "argmax_kept": int(lg.argmax()) == s["t1"]}
        abl.append(rec)
json.dump({"summary": summary, "ablations": abl}, open("tp1-results.json", "w"))
print(f"ablations {len(abl)}; TOTAL {time.monotonic() - T0:.1f}s", flush=True)
