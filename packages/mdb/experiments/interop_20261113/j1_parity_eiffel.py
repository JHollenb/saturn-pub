"""J1: numerical parity (HF-eager / mdb trace / mdb stepped / TL bridge / TL processed / nnsight)
and the Eiffel -> Paris -> France finding redone with TransformerLens' toolkit and nnsight, then
re-measured natively by mdb (sealed prediction, native measurement, P20 verdict).

Qwen2.5-1.5B, bf16, CUDA, eager. Writes one JSON to argv[2].
"""
import json
import os
import sys
import time
import warnings

import torch

warnings.filterwarnings("ignore")
T0 = time.time()
MODEL = sys.argv[1]
OUT = sys.argv[2]
DEV = "cuda"
R: dict = {"model": MODEL, "timing": {}}


def lap(name):
    R["timing"][name] = round(time.time() - T0, 2)
    print(f"[{R['timing'][name]:7.1f}s] {name}", flush=True)


from transformers import AutoTokenizer  # noqa: E402

import mdb  # noqa: E402
from mdb.engine.models import load_local  # noqa: E402
from mdb.engine.ports import program_for  # noqa: E402
from mdb.interop import names, nn, tl, verify  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
e = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
P1 = "The Eiffel Tower is located in the city of"
paris = e(P1 + " Paris, It is the capital of")
rome = e("The Colosseum is located in the city of Rome, It is the capital of")
FR, IT, PA, RO = e(" France")[0], e(" Italy")[0], e(" Paris")[0], e(" Rome")[0]
assert len(paris) == len(rome) == 18 and paris[11] == PA and rome[11] == RO, (paris, rome)
LAST = len(paris) - 1
R.update(tokens={"paris": paris, "rome": rome}, ids={"France": FR, "Italy": IT})
model = load_local(MODEL, dtype="bf16", device=DEV)
cfg = model.config
H, KV = cfg.num_attention_heads, cfg.num_key_value_heads
HD = cfg.hidden_size // H
program = program_for("qwen2")
LAYERS = cfg.num_hidden_layers
ports = [f"L{l}.{p}" for l in range(LAYERS) for p in program.names]
lap("load")
R["versions"] = {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                 "transformer_lens": tl.version(), "nnsight": nn.version(),
                 "flags": {"bf16_reduced_reduction":
                           torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}}
ids = torch.tensor([paris], device=DEV)


def margin(logits, row=-1):
    last = logits[0, row].double()
    return float(last[FR] - last[IT])


def diff(a, b):
    a, b = a.detach().double(), b.detach().double()
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)]}
    d = (a - b).abs()
    return {"max": float(d.max()), "equal": bool(torch.equal(a, b)),
            "frac_diff": float((d > 0).double().mean())}


# ---------------------------------------------------------------- A. parity
with torch.no_grad():
    ref = model(ids).logits
R["base_M_hf_full"] = margin(ref)
with mdb.trace(model, ports=ports) as t, torch.no_grad():
    traced = model(ids).logits
R["trace_passive_logits_equal"] = bool(torch.equal(traced, ref))
lap("trace")
exported = tl.trace_cache(t, head_dim=HD)
bridge = tl.bridge(model, tok)
tl_logits, cache = bridge.run_with_cache(ids)
R["tl_unprocessed_logits"] = diff(tl_logits, ref)
per = {}
for hook, value in exported.items():
    port = names.from_tl(hook).split(".", 1)[1]
    d = diff(cache[hook], value)
    if hook.endswith("hook_attn_scores"):
        causal = torch.ones(len(paris), len(paris), dtype=torch.bool, device=DEV).tril()
        d = diff(cache[hook][..., causal], value[..., causal])
    s = per.setdefault(port, {"equal": 0, "n": 0, "max": 0.0})
    s["n"] += 1
    s["equal"] += int(d.get("equal", False))
    s["max"] = max(s["max"], d.get("max", float("inf")))
R["parity_tl_unprocessed_vs_trace"] = per
lap("tl unprocessed parity")
nm = nn.wrap(model)
readable = [a for a in ports if names.BY_PORT[a.split(".")[1]].nn is not None]
got = nn.read(nm, [paris], readable)
per = {}
for address in readable:
    port = address.split(".")[1]
    spec = names.BY_PORT[port]
    layout = {"rot_q": "bh", "rot_k": "bh_k"}.get(spec.tl_layout, spec.nn_layout)
    for row in (0, 11, LAST):
        d = diff(got[f"t{row}.{address}"], names.to_mdb(layout, t[address], row))
        s = per.setdefault(port, {"equal": 0, "n": 0, "max": 0.0})
        s["n"] += 1
        s["equal"] += int(d.get("equal", False))
        s["max"] = max(s["max"], d.get("max", float("inf")))
R["parity_nnsight_vs_trace"] = per
del got
lap("nnsight parity")

dbg = mdb.attach(model, tok)
run = dbg.run(paris)
run.until("logits")
stepped_logits = run.read("logits").reshape(-1)
R["stepped_vs_full_logits"] = diff(stepped_logits, ref[0, LAST])
R["base_M_stepped"] = float(stepped_logits.double()[FR] - stepped_logits.double()[IT])
per = {}
for address in ports:
    port = address.split(".")[1]
    spec = names.BY_PORT[port]
    layout = {"rot_q": "bh", "rot_k": "bh_k"}.get(spec.tl_layout, spec.nn_layout) or "scores"
    s = per.setdefault(port, {"equal": 0, "n": 0, "max": 0.0})
    try:
        d = diff(run.read(f"t{LAST}.{address}"), names.to_mdb(layout, t[address], LAST))
    except Exception as error:  # record the refusal / defect per port
        s.setdefault("errors", []).append(f"{address}: {type(error).__name__}: {error}"[:200])
        continue
    s["n"] += 1
    s["equal"] += int(d.get("equal", False))
    s["max"] = max(s["max"], d.get("max", float("inf")))
    if port == "h_out":
        per.setdefault("_first_layer_differs", None)
        if not d.get("equal") and per["_first_layer_differs"] is None:
            per["_first_layer_differs"] = address
R["parity_stepped_vs_trace_row17"] = per
lap("stepped parity")

# ---------------------------------------------------------------- B. TL toolkit on the finding
_, rome_cache = bridge.run_with_cache(torch.tensor([rome], device=DEV))
z = "blocks.22.attn.hook_z"


def run_hooks(hooks):
    with torch.no_grad():
        return bridge.run_with_hooks(ids, fwd_hooks=hooks)


def zero_head(value, hook):
    value[:, -1, 6] = 0
    return value


def v_patch(rows_scope):
    def fn(value, hook):
        value[:, 11, 1] = rome_cache["blocks.22.attn.hook_v"][:, 11, 1]
        return value
    return fn


base_tl = margin(tl_logits)
R["tl"] = {"base": base_tl}
R["tl"]["zero_L22h6_last"] = margin(run_hooks([(z, zero_head)])) - base_tl
R["tl"]["patch_v_L22kv1_row11_sequence"] = margin(
    run_hooks([("blocks.22.attn.hook_v", v_patch("seq"))])) - base_tl
R["tl"]["patch_k_L22kv1_row11_sequence_rot"] = margin(run_hooks([(
    "blocks.22.attn.hook_rot_k",
    lambda v, hook: v.__setitem__((slice(None), 11, 1),
                                  rome_cache["blocks.22.attn.hook_rot_k"][:, 11, 1]) or v)])) - base_tl
lap("tl hooks")
try:
    from transformer_lens import patching

    def metric(logits):
        return torch.tensor(margin(logits) - base_tl)

    t1 = time.time()
    by_layer_pos = patching.get_act_patch_resid_pre(bridge, ids, rome_cache, metric)
    R["tl"]["act_patch_resid_pre"] = by_layer_pos.tolist()
    heads = patching.get_act_patch_attn_head_out_all_pos(bridge, ids, rome_cache, metric)
    R["tl"]["act_patch_head_out_all_pos"] = heads.tolist()
    R["tl"]["act_patch_seconds"] = time.time() - t1
except Exception as error:  # record the refusal
    R["tl"]["act_patch_error"] = f"{type(error).__name__}: {error}"[:2000]
lap("tl activation patching")
# frozen-final-norm DLA per head of L22 and per layer, by hand from the unprocessed cache
W_U = model.lm_head.weight.double()
w_norm = model.model.norm.weight.double()
dirn = (W_U[FR] - W_U[IT]) * w_norm
fn_in = cache["ln_final.hook_in"][0, -1].double()
rho = 1.0 / torch.sqrt(fn_in.pow(2).mean() + cfg.rms_norm_eps)
dla = {}
for layer in range(LAYERS):
    dla[f"L{layer}.attn"] = float(rho * cache[f"blocks.{layer}.hook_attn_out"][0, -1].double() @ dirn)
    dla[f"L{layer}.mlp"] = float(rho * cache[f"blocks.{layer}.hook_mlp_out"][0, -1].double() @ dirn)
W_O = model.model.layers[22].self_attn.o_proj.weight.double()
zz = cache[z][0, -1].double()
dla["L22.heads"] = [float(rho * (W_O[:, h * HD:(h + 1) * HD] @ zz[h]) @ dirn) for h in range(H)]
dla["embed"] = float(rho * cache["hook_embed"][0, -1].double() @ dirn)
R["dla_frozen_norm_unprocessed"] = dla
lap("dla")
bridge = cache = None
torch.cuda.empty_cache()

# ---------------------------------------------------------------- C. nnsight on the finding
arms = {"zero_L22h6": [nn.Edit("L22.A[h6]", 0.0, "scale")],
        "v_L22kv1_row11_reader": [nn.Edit("cache.L22.v[kv1,row=11]", "donor")],
        "v_L22kv1_row11_sequence": [nn.Edit("cache.L22.v[kv1,row=11]", "donor",
                                            scope="sequence")],
        "k_L22kv1_row11_reader": [nn.Edit("cache.L22.k[kv1,row=11]", "donor")],
        "v_L23kv0_row11_reader": [nn.Edit("cache.L23.v[kv0,row=11]", "donor")],
        "f_L23_4197_zero": [nn.Edit("L23.f[i=4197]", 0.0)]}
batch = nn.batch_whatif(model, [paris], [(FR, IT)], arms, donors=[rome])
R["nnsight"] = {"base": batch["base"][0],
                **{k: v["delta"][0] for k, v in batch["arms"].items()}}
lap("nnsight arms")

# ---------------------------------------------------------------- D. mdb verifies natively
_, rome_cache2 = tl.bridge(model, tok).run_with_cache(torch.tensor([rome], device=DEV))
tl.import_cache(dbg, "tlrome", rome_cache2, rome, layers=[22, 23], ports=["v", "kr", "A"])
del rome_cache2
dbg.run(rome, name="rome")
run = dbg.run(paris)
C = "M[%d,%d]" % (FR, IT)
claims = [
    verify.Claim("L22.A[h6]", 0.0, "scale", C, R["tl"]["zero_L22h6_last"], "transformer_lens"),
    verify.Claim("L22.A[h6]", 0.0, "scale", C, R["nnsight"]["zero_L22h6"], "nnsight"),
    verify.Claim("cache.L22.v[kv1,row=11]", "rome:cache.L22.v[kv1,row=11]", consumer=C,
                 claimed=R["nnsight"]["v_L22kv1_row11_reader"], tool="nnsight reader"),
    verify.Claim("cache.L22.v[kv1,row=11]", "tlrome:cache.L22.v[kv1,row=11]", consumer=C,
                 claimed=R["tl"]["patch_v_L22kv1_row11_sequence"],
                 tool="transformer_lens sequence (value from TL cache)"),
    verify.Claim("cache.L22.k[kv1,row=11]", "rome:cache.L22.k[kv1,row=11]", consumer=C,
                 claimed=R["nnsight"]["k_L22kv1_row11_reader"], tool="nnsight reader"),
    verify.Claim("cache.L23.v[kv0,row=11]", "rome:cache.L23.v[kv0,row=11]", consumer=C,
                 claimed=R["nnsight"]["v_L23kv0_row11_reader"], tool="nnsight reader"),
    verify.Claim("L23.f[i=4197]", 0.0, "replace", C, R["nnsight"]["f_L23_4197_zero"], "nnsight"),
    verify.Claim("L22.A[h6]", "tlrome:t17.L22.A", consumer=C, tool="TL cache port import"),
]
rows = verify.verify(dbg, run, claims)
for row in rows:
    row.pop("report", None)
R["mdb_verify"] = rows
lap("mdb verify")
json.dump(R, open(OUT, "w"), indent=1, default=str)  # checkpoint before the processed arm
keep = {f"L{l}.{p}": t[f"L{l}.{p}"].detach().clone() for l in (0, 22, 27) for p in ("p", "h_out")}
model = dbg = run = nm = t = None
torch.cuda.empty_cache()


class _T(dict):
    pass


t = _T(keep)
# ---------------------------------------------------------------- E. processed TL
# (HookedTransformer numerics) on a separate copy, after the main model is freed (host RAM:
# processing on the CPU passed 13.8 GB, job-7beddf882ce2)
PROCESSED = os.environ.get("J1_PROCESSED", "bf16").split(",")
for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
    if dtype_name not in PROCESSED:
        continue
    try:
        copy = load_local(MODEL, dtype="bf16" if dtype is torch.bfloat16 else "fp32", device=DEV)
        with torch.no_grad():
            m_cpu = margin(copy(ids).logits)  # same dtype and device, unprocessed
        b2 = tl.bridge(copy, tok, in_place=True)
        before = copy.model.layers[0]._original_component.self_attn.v_proj.weight.detach().clone() \
            if hasattr(copy.model.layers[0], "_original_component") else None
        b2.enable_compatibility_mode(disable_warnings=True)
        lg, c2 = b2.run_with_cache(ids)
        _, rc2 = b2.run_with_cache(torch.tensor([rome], device=DEV))
        out = {"M": margin(lg), "M_minus_hf": margin(lg) - R["base_M_hf_full"],
               "M_unprocessed_same_device": m_cpu, "M_minus_unprocessed_same_device":
               margin(lg) - m_cpu, "device": DEV,
               "processed": tl.processed(b2), "top1": int(lg[0, -1].argmax())}
        if before is not None:
            after = copy.model.layers[0]._original_component.self_attn.v_proj.weight
            out["hf_params_rewritten_in_place"] = not torch.equal(before, after)
        pattern = {}
        for layer in (0, 22, 27):
            hk = f"blocks.{layer}.attn.hook_pattern"
            pattern[layer] = float((c2[hk].double() - t[f"L{layer}.p"].double()).abs().max())
        out["pattern_max_diff_vs_native"] = pattern
        res = {}
        for layer in (0, 22, 27):
            a = c2[f"blocks.{layer}.hook_resid_post"][0].double()
            b = t[f"L{layer}.h_out"][0].double()
            res[layer] = {"raw": float((a - b).abs().max()),
                          "centred": float(((a - a.mean(-1, keepdim=True))
                                            - (b - b.mean(-1, keepdim=True))).abs().max())}
        out["resid_post_diff_vs_native"] = res
        stack, labels = c2.stack_head_results(layer=-1, pos_slice=-1, apply_ln=True,
                                              return_labels=True)
        attrs = c2.logit_attrs(stack, tokens=torch.tensor(FR), incorrect_tokens=torch.tensor(IT),
                               pos_slice=-1)
        out["dla_heads_L22"] = [float(attrs[labels.index(f"L22H{h}")]) for h in range(H)]
        acc, alab = c2.decompose_resid(layer=-1, pos_slice=-1, apply_ln=True, return_labels=True)
        la = c2.logit_attrs(acc, tokens=torch.tensor(FR), incorrect_tokens=torch.tensor(IT),
                            pos_slice=-1)
        out["dla_layers"] = {lab: float(v) for lab, v in zip(alab, la.reshape(-1).tolist())}

        def zh(value, hook):
            value[:, -1, 6] = 0
            return value
        with torch.no_grad():
            out["zero_L22h6_last"] = margin(b2.run_with_hooks(ids, fwd_hooks=[(z, zh)])) - out["M"]

            def vp(value, hook):
                value[:, 11, 1] = rc2["blocks.22.attn.hook_v"][:, 11, 1]
                return value
            out["patch_v_L22kv1_row11_sequence"] = margin(b2.run_with_hooks(
                ids, fwd_hooks=[("blocks.22.attn.hook_v", vp)])) - out["M"]
        R[f"tl_processed_{dtype_name}"] = out
        b2 = c2 = rc2 = copy = None
    except Exception as error:
        R[f"tl_processed_{dtype_name}"] = {"error": f"{type(error).__name__}: {error}"[:2000]}
    torch.cuda.empty_cache()
    lap(f"tl processed {dtype_name}")

R["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
json.dump(R, open(OUT, "w"), indent=1, default=str)
lap("done")
