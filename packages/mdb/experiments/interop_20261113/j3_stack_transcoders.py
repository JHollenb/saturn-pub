"""J3 (stack S1, first run): transcoder nomination -> nnsight batched native measurement -> mdb
sealed prediction, native batch-1 measurement and error split, with the transcoder's error node
measured as a node of its own.

Qwen3-1.7B-Base bf16 CUDA eager. circuit-tracer's per-layer transcoders for Qwen3-1.7B
(mwhanna/qwen3-1.7b-transcoders-lowl0: input mlp.hook_in = mdb L.n, output mlp.hook_out = L.m)
are read with safetensors directly: circuit-tracer 0.5 pins transformers <= 4.57 and cannot be
imported next to transformers 5.13, and its full attribution graph needs all 28 layers (37.6 GB);
nomination here uses the graph's direct edge to the margin only (A_s = a_s rho^F c.d_s, the P19
direct share), which is the first term of circuit-tracer's edge weight.

argv: MODEL_BASE MODEL_POST TC_DIR OUT [N_PROMPTS] [N_VERIFY]
"""
import json
import sys
import time
import warnings

import torch
from safetensors import safe_open

warnings.filterwarnings("ignore")
T0 = time.time()
BASE, POST, TC, OUT = sys.argv[1:5]
N_PROMPTS = int(sys.argv[5]) if len(sys.argv) > 5 else 12
N_VERIFY = int(sys.argv[6]) if len(sys.argv) > 6 else 6
R: dict = {"model": BASE, "post_model": POST, "transcoders": TC, "timing": {}}


def lap(name):
    R["timing"][name] = round(time.time() - T0, 2)
    print(f"[{R['timing'][name]:7.1f}s] {name}", flush=True)


from transformers import AutoTokenizer  # noqa: E402

import mdb  # noqa: E402
from mdb.engine.models import load_local  # noqa: E402
from mdb.engine.ports import program_for  # noqa: E402
from mdb.interop import nn, tl, verify  # noqa: E402

CITIES = ("Paris France|Rome Italy|Berlin Germany|Madrid Spain|Tokyo Japan|Beijing China|"
          "Moscow Russia|Cairo Egypt|Athens Greece|Vienna Austria|Lisbon Portugal|Dublin Ireland|"
          "Oslo Norway|Stockholm Sweden|Helsinki Finland|Warsaw Poland|Budapest Hungary|"
          "Bangkok Thailand|Ottawa Canada|Canberra Australia|Nairobi Kenya|Lima Peru|"
          "Havana Cuba|Tehran Iran|Baghdad Iraq|Kabul Afghanistan|Manila Philippines|"
          "Hanoi Vietnam|Jakarta Indonesia|Ankara Turkey|Copenhagen Denmark|Brussels Belgium|"
          "Amsterdam Netherlands|Damascus Syria|Beirut Lebanon|Caracas Venezuela")
tok = AutoTokenizer.from_pretrained(BASE)
e = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
STEM = "The tower is located in the city of"
ROW = len(e(STEM))
model = load_local(BASE, dtype="bf16", device="cuda")
cfg = model.config
LAYERS, D = cfg.num_hidden_layers, cfg.hidden_size
program = program_for(cfg.model_type)
lap("load base")

# ------------------------------------------------------------- stage 0: Qwen3 real size parity
probe = e(f"{STEM} Paris, It is the capital of")
ids = torch.tensor([probe], device="cuda")
ports = [f"L{l}.{p}" for l in range(LAYERS) for p in program.names]
with torch.no_grad():
    ref = model(ids).logits
with mdb.trace(model, ports=ports) as t, torch.no_grad():
    traced = model(ids).logits
stage0 = {"trace_passive_logits_equal": bool(torch.equal(ref, traced))}
head_dim = cfg.head_dim
exported = tl.trace_cache(t, head_dim=head_dim)
_, cache = tl.bridge(model, tok).run_with_cache(ids)
bad = {}
for hook, value in exported.items():
    other = cache[hook]
    if hook.endswith("hook_attn_scores"):
        causal = torch.ones(len(probe), len(probe), dtype=torch.bool, device="cuda").tril()
        other, value = other[..., causal], value[..., causal]
    if not torch.equal(other, value):
        bad[hook] = float((other.double() - value.double()).abs().max())
stage0["tl_vs_trace_hooks"] = len(exported)
stage0["tl_vs_trace_unequal"] = bad
cache = None
dbg = mdb.attach(model, tok)
run = dbg.run(probe)
run.until("logits")
stage0["stepped_vs_full_max_abs_logit"] = float(
    (run.read("logits").reshape(-1).double() - ref[0, -1].double()).abs().max())
t = None
R["stage0"] = stage0
lap("stage 0")

# ------------------------------------------------------------- prompts
items = []
for entry in CITIES.split("|"):
    city, country = entry.split(" ", 1)
    c, k = e(" " + city), e(" " + country)
    if len(c) == 1 and len(k) == 1:
        items.append((city, country, k[0]))
prompts = [e(f"{STEM} {city}, It is the capital of") for city, *_ in items]
with torch.no_grad():
    top = model(torch.tensor(prompts, device="cuda")).logits[:, -1].argmax(-1).tolist()
kept = [(it, p) for it, p, tp in zip(items, prompts, top) if tp == it[2]]
R["cities_correct"] = [it[0] for it, _ in kept]
pairs = [(kept[i], kept[(i + 1) % len(kept)]) for i in range(len(kept))][:N_PROMPTS]
recipients = [a[1] for a, _ in pairs]
consumers = [(a[0][2], b[0][2]) for a, b in pairs]
R["pairs"] = [[a[0][0], b[0][0]] for a, b in pairs]
LAST = len(recipients[0]) - 1
lap(f"prompts {len(pairs)}")

# native n, m at the last row of every layer, and the margin direction through the frozen norm
with mdb.trace(model, ports=[f"L{l}.{p}" for l in range(LAYERS) for p in ("n", "m")]
               + [f"L{LAYERS - 1}.h_out"]) as t, torch.no_grad():
    model(torch.tensor(recipients, device="cuda"))
n_last = {l: t[f"L{l}.n"][:, -1].float().cpu() for l in range(LAYERS)}
m_last = {l: t[f"L{l}.m"][:, -1].float().cpu() for l in range(LAYERS)}
h_final = t[f"L{LAYERS - 1}.h_out"][:, -1].double().cpu()
t = None
rho = 1.0 / torch.sqrt(h_final.pow(2).mean(-1) + cfg.rms_norm_eps)          # (P,)
W_U = model.lm_head.weight.detach().double().cpu()
w_norm = model.model.norm.weight.detach().double().cpu()
c_dir = torch.stack([(W_U[a] - W_U[b]) * w_norm for a, b in consumers])     # (P, D)
mlp_dla = {l: (rho[:, None] * m_last[l].double() * c_dir).sum(-1) for l in range(LAYERS)}
R["mlp_dla_mean"] = {l: float(v.mean()) for l, v in mlp_dla.items()}
available = []
for l in range(LAYERS):
    try:
        with safe_open(f"{TC}/layer_{l}.safetensors", framework="pt") as f:
            available.append(l)
    except Exception:
        pass
chosen = sorted(available, key=lambda l: -float(mlp_dla[l].abs().mean()))[:3]
R["transcoder_layers_available"], R["transcoder_layers_chosen"] = available, chosen
lap(f"layers {chosen}")


def load_tc(layer):
    out = {}
    with safe_open(f"{TC}/layer_{layer}.safetensors", framework="pt") as f:
        for key in f.keys():
            out[key] = f.get_tensor(key).float()
    return out


def run_tc(tc, x):
    pre = x @ tc["W_enc"].T + tc["b_enc"]
    thr = tc.get("activation_function.threshold", tc.get("threshold"))
    acts = pre * (pre > thr) if thr is not None else torch.relu(pre)
    recon = acts @ tc["W_dec"] + tc["b_dec"]
    skip = x @ tc["W_skip"].T if "W_skip" in tc else torch.zeros_like(recon)
    return acts, recon + skip, skip


def fvu(target, recon):
    return float((target - recon).pow(2).sum() / (target - target.mean(0)).pow(2).sum())


nodes = {}
fvus = {}
w_post = {l: model.model.layers[l].post_attention_layernorm.weight.detach().float().cpu()
          for l in chosen}
for layer in chosen:
    tc = load_tc(layer)
    keys = {k: list(v.shape) for k, v in tc.items()}
    by_input = {}
    for name, x in (("n", n_last[layer]), ("n_over_w_post", n_last[layer] / w_post[layer])):
        acts, recon, skip = run_tc(tc, x)
        by_input[name] = {"fvu": fvu(m_last[layer], recon), "l0": float((acts > 0).sum(-1)
                          .float().mean())}
    use = min(by_input, key=lambda k: by_input[k]["fvu"])
    x = n_last[layer] if use == "n" else n_last[layer] / w_post[layer]
    acts, recon, skip = run_tc(tc, x)
    error = m_last[layer] - recon                                          # (P, D)
    direct = acts.double() * (rho[:, None] * (c_dir @ tc["W_dec"].double().T))  # (P, F)
    top = direct.abs().topk(3, dim=-1).indices
    feats = []
    for k in range(3):
        idx = top[:, k]
        vec = -(acts[torch.arange(len(idx)), idx][:, None] * tc["W_dec"][idx])  # minus a_s d_s
        feats.append({"feature": idx.tolist(),
                      "activation": acts[torch.arange(len(idx)), idx].tolist(),
                      "direct": direct[torch.arange(len(idx)), idx].tolist(),
                      "vector": vec})
    nodes[layer] = {"input": use, "features": feats, "error": error,
                    "error_direct": (rho * (error.double() * c_dir).sum(-1)).tolist(),
                    "features_all": -(recon - skip - tc["b_dec"])}
    fvus[layer] = {"keys": keys, **by_input, "used": use}
    tc = None
R["fvu_base"] = fvus
lap("transcoders encoded")

# ------------------------------------------------------------- nnsight batch, every node
arms = {}
for layer, node in nodes.items():
    for k, f in enumerate(node["features"]):
        arms[f"L{layer}_f{k}"] = [nn.Edit(f"L{layer}.m", f["vector"], "add")]
    arms[f"L{layer}_error"] = [nn.Edit(f"L{layer}.m", -node["error"], "add")]
    arms[f"L{layer}_all_features"] = [nn.Edit(f"L{layer}.m", node["features_all"], "add")]
batch = nn.batch_whatif(model, recipients, consumers, arms, batch_size=16)
R["batch"] = {"base": batch["base"], "arms": {k: v["delta"] for k, v in batch["arms"].items()}}
lap("nnsight batch")

# ------------------------------------------------------------- mdb: sealed, native, split
rows = []
for i in range(min(N_VERIFY, len(recipients))):
    values, claims = {}, []
    for layer, node in nodes.items():
        for k, f in enumerate(node["features"][:1]):
            name = f"f{layer}x{k}"
            dbg.import_values(name, {f"t{LAST}.L{layer}.m": f["vector"][i].reshape(1, 1, D)},
                              tokens=recipients[i], provenance={"tool": "transcoder",
                                                                "feature": str(f["feature"][i])})
            claims.append(verify.Claim(f"L{layer}.m", f"{name}:t{LAST}.L{layer}.m", "add",
                                       "M[%d,%d]" % consumers[i],
                                       batch["arms"][f"L{layer}_f{k}"]["delta"][i],
                                       "nnsight batch", extra={
                                           "node": f"L{layer} feature {f['feature'][i]}",
                                           "graph_direct": -f["direct"][i], "prompt": i}))
        name = f"err{layer}"
        dbg.import_values(name, {f"t{LAST}.L{layer}.m": -node["error"][i].reshape(1, 1, D)},
                          tokens=recipients[i], provenance={"tool": "transcoder error node"})
        claims.append(verify.Claim(f"L{layer}.m", f"{name}:t{LAST}.L{layer}.m", "add",
                                   "M[%d,%d]" % consumers[i],
                                   batch["arms"][f"L{layer}_error"]["delta"][i], "nnsight batch",
                                   extra={"node": f"L{layer} error",
                                          "graph_direct": -node["error_direct"][i], "prompt": i}))
    run = dbg.run(recipients[i])
    for row in verify.verify(dbg, run, claims):
        row.pop("report", None)
        row["batched_base"] = batch["base"][i]
        rows.append(row)
    lap(f"verify prompt {i}")
R["mdb_verify"] = rows


def _dump():
    nodes_out = {l: {"input": v["input"], "error_direct": v["error_direct"],
                     "features": [{k: x for k, x in f.items() if k != "vector"}
                                  for f in v["features"]]} for l, v in nodes.items()}
    json.dump({**R, "nodes": nodes_out}, open(OUT, "w"), indent=1, default=str)


_dump()  # checkpoint before the post-trained model
model = dbg = run = batch = None
import gc  # noqa: E402

gc.collect()
torch.cuda.empty_cache()

# ------------------------------------------------------------- FVU on the post-trained model
post = load_local(POST, dtype="bf16", device="cuda")
with mdb.trace(post, ports=[f"L{l}.{p}" for l in chosen for p in ("n", "m")]) as t, \
        torch.no_grad():
    post(torch.tensor(recipients, device="cuda"))
fvu_post = {}
for layer in chosen:
    tc = load_tc(layer)
    n = t[f"L{layer}.n"][:, -1].float().cpu()
    m = t[f"L{layer}.m"][:, -1].float().cpu()
    wp = post.model.layers[layer].post_attention_layernorm.weight.detach().float().cpu()
    x = n if fvus[layer]["used"] == "n" else n / wp
    _, recon, _ = run_tc(tc, x)
    fvu_post[layer] = fvu(m, recon)
    tc = None
R["fvu_post"] = fvu_post
R["fvu_ratio_base_over_post"] = {l: fvus[l][fvus[l]["used"]]["fvu"] / fvu_post[l] for l in chosen}
R["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
lap("fvu post")
_dump()
lap("done")
