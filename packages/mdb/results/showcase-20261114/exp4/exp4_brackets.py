"""Job D (exp4/prereg.json): the attribution-graph gap split into brackets with floors.

python exp4_brackets.py OUTDIR [PICK]

Per base item (32: the first 8 by id of each 1a family), consumer M[answer, donor answer]:
1. Nodes: the mwhanna lowl0 transcoders (Base-fit layers L20..L24) encode t<last>.L<l>.n; the 5
   active features with the largest direct effect rho_F c . a_s d_s (float64) are the nodes;
   plus one null node: a random active feature at node 1's layer rescaled to its write norm.
2. Edge (nnsight, mdb.interop.nn naming): the gradient of M w.r.t. t<last>.L<l>.m with the
   answer row's attention weights and every RMSNorm scale frozen at the base run and the MLPs
   live: in one nnsight trace per (item, layer), for every layer after l the q_proj and k_proj
   outputs are detached (the edit touches only the answer row, whose p depends on its own q and
   k only), and input_layernorm / post_attention_layernorm / the final norm outputs are replaced
   by input * (gamma * rho_base) with rho_base the base run's scale (detached). A_s = g . a_s d_s;
   the edge forecast of the removal is -A_s, sealed as a supplied-forecast record.
3. mdb native arms from one checkpoint at t<last>.L<l>.h (Run.set, supplied vector edits):
   arm0 free (m += -a d); arm1 = arm0 + the answer row's p at every later layer set to the base
   run's p; arm2 = arm1 + the norm scales held: a = gamma_in * h * rho_base at each later layer
   (h read at the stop), n = gamma_post * r * rho_base likewise, and the final scale frozen
   (M2 = rho_F,base c . h_out,last). Null arm: arm1 + arm2 machinery with no feature edit.
Brackets: re-routing = arm0 - arm1; norm = arm1 - arm2; nonlinearity = arm2 - (-A).
The log prints item ids and seconds only.
"""
import json
import random
import sys
import time

import torch
from safetensors.torch import load_file

sys.path.insert(0, ".")
from common import TC_DIR, floor_of, items, load, log  # noqa: E402
from mdb.analysis import supplied  # noqa: E402
from mdb.analysis.finite import f64  # noqa: E402
from mdb.interop import nn  # noqa: E402

OUT = sys.argv[1]
PICK = sys.argv[2] if len(sys.argv) > 2 else ""
LAYERS = list(range(20, 25))
dbg = load()
ad = dbg.adapter
model = ad.model
dev = model.lm_head.weight.device
NL = ad.layers
TC = {}
for x in LAYERS:
    st = load_file(f"{TC_DIR}/layer_{x}.safetensors")
    TC[x] = {"W_enc": st["W_enc"].to(dev), "b_enc": st["b_enc"].float(), "W_dec": st["W_dec"],
             "W_enc_cpu": st["W_enc"]}
log("load")
final = model.model.norm
GF = final.weight
rng = random.Random(20261114)
fams = {}
for i in sorted(items("base"), key=lambda i: i["id"]):
    fams.setdefault(i["family"], []).append(i)
panel = [i for f in sorted(fams) for i in fams[f][:8]]
if PICK:
    panel = [panel[int(k)] for k in PICK.split(",")]
VER = nn.version()
AMAP = {"t<last>.L<l>.m": "model.model.layers[l].mlp.output",
        "freeze q/k": "model.model.layers[l'].self_attn.{q,k}_proj.output (detached)",
        "freeze norms": "model.model.layers[l'].{input,post_attention}_layernorm.output, model.model.norm.output"}


def rho(x, eps):
    return (x.float().pow(2).mean(-1, keepdim=True) + eps).rsqrt()


def frozen_grad(ids, layer, a, b, scales):
    """dM/dm_l at the last row with p and all norm scales frozen (nnsight trace)."""
    import nnsight
    nm = nn.wrap(model)
    t = torch.tensor([ids], device=dev)
    with nn._Frozen(model), torch.enable_grad(), nm.trace(t):
        emb = nm.model.embed_tokens.output
        emb.requires_grad_(True)
        mout = nm.model.layers[layer].mlp.output
        for x in range(layer + 1, NL):
            L_ = nm.model.layers[x]
            L_.input_layernorm.output = (L_.input_layernorm.input * scales["a"][x]).to(torch.bfloat16)
            L_.self_attn.q_proj.output = L_.self_attn.q_proj.output.detach()
            L_.self_attn.k_proj.output = L_.self_attn.k_proj.output.detach()
            L_.post_attention_layernorm.output = (L_.post_attention_layernorm.input * scales["n"][x]).to(torch.bfloat16)
        nm.model.norm.output = (nm.model.norm.input * scales["F"]).to(torch.bfloat16)
        lg = nm.lm_head.output[0, -1].float()
        M = lg[a] - lg[b]
        g = nnsight.save([])
        with M.backward():
            g.append(mout.grad[0, -1].detach().double().cpu())
    return g[0]


fh = open(f"{OUT}/exp4-items{'-smoke' if PICK else ''}.jsonl", "w")
for item in panel:
    t0 = time.monotonic()
    ids, last = item["ids"], len(item["ids"]) - 1
    a, b = item["answer"], item["donor"]["answer"]
    nat = dbg.run(ids, name="nat")
    nat.until(f"t{last}.logits")
    lg = nat.read(f"t{last}.logits").reshape(-1).float()
    M0 = float(lg[a] - lg[b])
    hF = f64(nat.read(f"t{last}.L{NL - 1}.h_out")).reshape(-1).cpu()
    rF = float((hF.pow(2).mean() + float(final.variance_epsilon)) ** -0.5)
    c = (f64(GF) * (f64(model.lm_head.weight[a]) - f64(model.lm_head.weight[b]))).cpu()
    base = {}
    for x in range(LAYERS[0], NL):
        L_ = model.model.layers[x]
        h = nat.read(f"t{last}.L{x}.h").reshape(-1)
        r = nat.read(f"t{last}.L{x}.r").reshape(-1)
        base[x] = {"p": nat.read(f"t{last}.L{x}.p").clone(),
                   "ra": rho(h, L_.input_layernorm.variance_epsilon), "rn": rho(r, L_.post_attention_layernorm.variance_epsilon)}
    # nodes
    nodes = []
    for x in LAYERS:
        n_ = nat.read(f"t{last}.L{x}.n").reshape(-1)
        pre = (TC[x]["W_enc"] @ n_.to(torch.bfloat16)).float().cpu() + TC[x]["b_enc"]
        fid = (pre > 0).nonzero().reshape(-1)
        acts = torch.relu(TC[x]["W_enc_cpu"].index_select(0, fid).double() @ n_.double().cpu()
                          + TC[x]["b_enc"].index_select(0, fid).double())
        D = TC[x]["W_dec"].index_select(0, fid).double()
        de = rF * (D * acts[:, None]) @ c
        for k in range(len(fid)):
            if acts[k] > 0:
                nodes.append((float(de[k]), x, int(fid[k]), float(acts[k]), D[k] * acts[k]))
    nodes.sort(key=lambda z: -z[0])
    top = nodes[:5]
    same = [z for z in nodes[5:] if z[1] == top[0][1]]
    if same:
        de_r, x_r, f_r, a_r, w_r = rng.choice(same)
        w_r = w_r * (top[0][4].norm() / (w_r.norm() + 1e-30))
        top = top + [(de_r, x_r, f_r, a_r, w_r, "random")]
    rec = {"id": item["id"], "M0": M0, "floor": floor_of(lg, a, b), "nodes": []}
    run = dbg.run(ids, name="arms")
    run.until(f"t{last}.L{LAYERS[0]}.h")
    ck = run.checkpoint("pre")
    grads = {}
    for node in top:
        de, x, f, act, w = node[:5]
        kind = node[5] if len(node) > 5 else "node"
        if x not in grads:
            # base rho at every row for the frozen-scale trace (read from an nnsight clean pass)
            nm = nn.wrap(model)
            with torch.no_grad(), nm.trace(torch.tensor([ids], device=dev)):
                saved = {}
                for x2 in range(x + 1, NL):
                    L2 = nm.model.layers[x2]
                    saved[("a", x2)] = L2.input_layernorm.input.save()
                    saved[("n", x2)] = L2.post_attention_layernorm.input.save()
                saved[("F", 0)] = nm.model.norm.input.save()
            sc = {"a": {}, "n": {}}
            for x2 in range(x + 1, NL):
                L2 = model.model.layers[x2]
                sc["a"][x2] = (rho(saved[("a", x2)], L2.input_layernorm.variance_epsilon) * L2.input_layernorm.weight.float()).detach()
                sc["n"][x2] = (rho(saved[("n", x2)], L2.post_attention_layernorm.variance_epsilon) * L2.post_attention_layernorm.weight.float()).detach()
            sc["F"] = (rho(saved[("F", 0)], final.variance_epsilon) * GF.float()).detach()
            grads[x] = frozen_grad(ids, x, a, b, sc)
        A = float(grads[x] @ w)
        sealed = supplied.seal(nat, target={"kind": "margin", "row": last, "indices": [a, b]},
                               edit={"address": f"t{last}.L{x}.m", "row": last, "layer": x,
                                     "op": f"add -a d of transcoder L{x}.f{f}"},
                               value=-A, library="nnsight", version=VER, address_map=AMAP,
                               method="frozen-pattern, frozen-norm-scale gradient . write (attribution-graph edge)")
        arms = {}
        for arm in ("arm0", "arm1", "arm2", "null"):
            run.until(f"t{last}.L{x}.h")
            if arm != "null":
                supplied.vector_edit(run, f"t{last}.L{x}.m", (-w).to(torch.bfloat16).reshape(1, 1, -1).to(dev),
                                     op="add", library="mwhanna transcoders", version="lowl0", address_map=AMAP)
            if arm in ("arm1", "arm2", "null"):
                for x2 in range(x + 1, NL):
                    run.until(f"t{last}.L{x2}.h")
                    run.set(f"t{last}.L{x2}.p", base[x2]["p"])
                    if arm in ("arm2", "null"):
                        L2 = model.model.layers[x2]
                        h = run.read(f"t{last}.L{x2}.h")
                        run.set(f"t{last}.L{x2}.a", (L2.input_layernorm.weight * (h.float() * base[x2]["ra"]).to(h.dtype)))
                        run.until(f"t{last}.L{x2}.r")
                        r = run.read(f"t{last}.L{x2}.r")
                        run.set(f"t{last}.L{x2}.n", (L2.post_attention_layernorm.weight * (r.float() * base[x2]["rn"]).to(r.dtype)))
            run.until(f"t{last}.logits")
            lgx = run.read(f"t{last}.logits").reshape(-1).float()
            if arm in ("arm2", "null"):
                hl = f64(run.read(f"t{last}.L{NL - 1}.h_out")).reshape(-1).cpu()
                arms[arm] = rF * float(c @ hl) - rF * float(c @ hF)
            else:
                arms[arm] = float(lgx[a] - lgx[b]) - M0
            run.restore(ck)
        rec["nodes"].append({"kind": kind, "layer": x, "feature": f, "act": act, "direct": de,
                             "edge": -A, "record": sealed["id"], **arms,
                             "rerouting": arms["arm0"] - arms["arm1"], "norm": arms["arm1"] - arms["arm2"],
                             "nonlin": arms["arm2"] - (-A)})
    del run, nat
    rec["seconds"] = round(time.monotonic() - t0, 2)
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    log(f"D {item['id']} {rec['seconds']}s")
log("TOTAL")
