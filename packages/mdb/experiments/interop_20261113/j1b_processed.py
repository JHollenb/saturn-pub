"""J1b: TransformerLens weight processing (enable_compatibility_mode, HookedTransformer numerics)
against native values on the Eiffel prompt. One bf16 model: native trace first, then the bridge
rewrites the same module in place and processes its weights. argv: MODEL OUT [fp32]."""
import json
import sys
import time
import warnings

import torch

warnings.filterwarnings("ignore")
T0 = time.time()
MODEL, OUT = sys.argv[1], sys.argv[2]
DTYPE = sys.argv[3] if len(sys.argv) > 3 else "bf16"
# weight processing of Qwen2.5-1.5B in bf16 on the 16 GB GPU ran out of memory (15.9 GB peak,
# job-d133d104371f), so the whole comparison (native and processed) can run on the CPU
DEV = sys.argv[4] if len(sys.argv) > 4 else "cuda"
from transformers import AutoTokenizer  # noqa: E402

import mdb  # noqa: E402
from mdb.engine.models import load_local  # noqa: E402
from mdb.interop import tl  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
e = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
paris = e("The Eiffel Tower is located in the city of Paris, It is the capital of")
rome = e("The Colosseum is located in the city of Rome, It is the capital of")
FR, IT = e(" France")[0], e(" Italy")[0]
ids, rids = torch.tensor([paris], device=DEV), torch.tensor([rome], device=DEV)
model = load_local(MODEL, dtype=DTYPE, device=DEV)
L = model.config.num_hidden_layers


def margin(logits):
    last = logits[0, -1].double()
    return float(last[FR] - last[IT])


ports = [f"L{l}.{p}" for l in range(L) for p in ("p", "h_out", "a", "n", "v", "A", "o", "m")]
with mdb.trace(model, ports=ports) as t, torch.no_grad():
    native_logits = model(ids).logits
native = margin(native_logits)
R = {"dtype": DTYPE, "device": DEV, "M_native": native, "top1_native": int(native_logits[0, -1].argmax())}
before = model.model.layers[0].self_attn.v_proj.weight.detach().clone()
bridge = tl.bridge(model, tok, in_place=True)
bridge.enable_compatibility_mode(disable_warnings=True)
after = model.model.layers[0]._original_component.self_attn.v_proj.weight
R["hf_params_rewritten_in_place"] = not torch.equal(before, after)
before = None
logits, cache = bridge.run_with_cache(ids)
R["M_processed"] = margin(logits)
R["M_processed_minus_native"] = R["M_processed"] - native
R["top1_processed"] = int(logits[0, -1].argmax())
R["logits_minus_native_centred_max"] = float(
    ((logits[0, -1].double() - logits[0, -1].double().mean())
     - (native_logits[0, -1].double() - native_logits[0, -1].double().mean())).abs().max())
diffs = {}
for port, hook in (("p", "attn.hook_pattern"), ("h_out", "hook_resid_post"),
                   ("a", "ln1.hook_out"), ("n", "ln2.hook_out"), ("v", "attn.v.hook_out"),
                   ("A", "attn.hook_z"), ("o", "hook_attn_out"), ("m", "hook_mlp_out")):
    worst, worst_c = 0.0, 0.0
    for layer in range(L):
        mine = t[f"L{layer}.{port}"].double()
        theirs = cache[f"blocks.{layer}.{hook}"].double().reshape(mine.shape)
        worst = max(worst, float((theirs - mine).abs().max()))
        if port in ("h_out", "o", "m"):
            cm = mine - mine.mean(-1, keepdim=True)
            worst_c = max(worst_c, float((theirs - cm).abs().max()))
    diffs[port] = {"max_abs": worst, **({"max_abs_vs_centred_native": worst_c}
                                       if port in ("h_out", "o", "m") else {})}
R["port_diff_processed_vs_native"] = diffs
try:
    stack, labels = cache.stack_head_results(layer=-1, pos_slice=-1, apply_ln=True,
                                             return_labels=True)
    attrs = cache.logit_attrs(stack, tokens=torch.tensor(FR), incorrect_tokens=torch.tensor(IT),
                              pos_slice=-1)
    R["dla_heads_L22"] = [float(attrs[labels.index(f"L22H{h}")]) for h in range(12)]
    acc, alab = cache.decompose_resid(layer=-1, pos_slice=-1, apply_ln=True, return_labels=True)
    la = cache.logit_attrs(acc, tokens=torch.tensor(FR), incorrect_tokens=torch.tensor(IT),
                           pos_slice=-1)
    R["dla_layers"] = {lab: float(v) for lab, v in zip(alab, la.reshape(-1).tolist())}
except Exception as error:
    R["dla_error"] = f"{type(error).__name__}: {error}"[:1500]
cache = None
_, rcache = bridge.run_with_cache(rids)


def zh(value, hook):
    value[:, -1, 6] = 0
    return value


def vp(value, hook):
    value[:, 11, 1] = rcache["blocks.22.attn.hook_v"][:, 11, 1]
    return value


with torch.no_grad():
    R["zero_L22h6_last"] = margin(bridge.run_with_hooks(
        ids, fwd_hooks=[("blocks.22.attn.hook_z", zh)])) - R["M_processed"]
    R["patch_v_L22kv1_row11_sequence"] = margin(bridge.run_with_hooks(
        ids, fwd_hooks=[("blocks.22.attn.hook_v", vp)])) - R["M_processed"]
R["seconds"] = time.time() - T0
R["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0.0
json.dump(R, open(OUT, "w"), indent=1, default=str)
print(json.dumps(R, default=str)[:3000])
