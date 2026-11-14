"""Job F (exp6/prereg.json): TransformerLens weight processing as a reparametrisation null.

python exp6_tlnull.py MODEL_DIR LC KVC OUTDIR [PICK]

Per model (Qwen2.5-0.5B: L21 KV0; Qwen3-0.6B-Base: L20 KV6, scale-20261114 carriers), the 48
base items of the scale panel, consumer M[answer, donor answer] at the last row.
mdb (native, stepped, bf16 CUDA): base margin; patch effects = zero each query head's A at L_c
and L_c+1 at the last row (checkpoint/restore), and the carrier V interchange from the donor in
sequence scope (prefill = carrier row, the carrier row's v of the carrier KV group set to the
donor's at t<crow>.L<Lc>.h). Carrier row: argmax of the answer row's L_c attention summed over
the carrier KV group's query heads, over the span (t0 excluded, ties later).
TransformerLens 4.4 TransformerBridge: (U16) unprocessed bf16 over the same weights
(tl.bridge, shared clone); (P16) processed bf16 = a second bf16 copy, enable_compatibility_mode();
(U32, P32) the same in fp32 (CUDA, TF32 off). Patches by run_with_hooks with names_filter-style
hook names from mdb.interop.names.to_tl (A -> hook_z, v -> v.hook_out, p -> hook_pattern), all
arms of an item in one batch; U16 also at batch 1 per arm (kernel-path null). Each P16 patch
effect is sealed as a supplied-forecast record (library transformer-lens, processing recorded)
and judged by analysis.supplied.judge against mdb's native effect with mdb's floor.
The log prints item ids and seconds only.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
from common import log, spacing  # noqa: E402
from mdb.analysis import supplied  # noqa: E402
from mdb.interop import names, tl  # noqa: E402

MODEL, LC, KVC, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
PICK = sys.argv[5] if len(sys.argv) > 5 else ""
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_grad_enabled(False)
from mdb.repl.__main__ import _load  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

dbg = _load(Path(MODEL), "bf16", "cuda", True, None)
ad = dbg.adapter
H, KV, DH = ad.num_heads, ad.num_kv_heads, ad.head_dim
GH = [h for h in range(H) if h // (H // KV) == KVC]
tok = AutoTokenizer.from_pretrained(MODEL)
U16 = tl.bridge(ad.model, tok)


def copy_bridge(dtype):
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, attn_implementation="eager").cuda().eval()
    return tl.bridge(m, tok, in_place=True)


P16 = copy_bridge(torch.bfloat16)
P16.enable_compatibility_mode()
U32 = copy_bridge(torch.float32)
P32 = copy_bridge(torch.float32)
P32.enable_compatibility_mode()
PROC = {k: tl.processing(b) for k, b in (("U16", U16), ("P16", P16), ("U32", U32), ("P32", P32))}
log("load")
panel = json.load(open("panel.json"))
items = {i["id"]: i for i in panel["items"]}
base = sorted([i for i in panel["items"] if i["block"] == "base"], key=lambda i: i["id"])
if PICK:
    base = base[: int(PICK)]
HZ = {x: names.to_tl(f"L{x}.A") for x in (LC, LC + 1)}
HV = names.to_tl(f"L{LC}.v")
HP = names.to_tl(f"L{LC}.p")
VER = tl.version()


def ids_of(i):
    return i["ids"]


def crow_of(run, item):
    last = len(item["ids"]) - 1
    p = run.read(f"t{last}.L{LC}.p").float().reshape(H, -1)[GH].sum(0)
    span = [r for r in item["span"] if r > 0]
    best = max(span, key=lambda r: (float(p[r]), r))
    return best


def mdb_item(item, donor_v):
    ids, last = item["ids"], len(item["ids"]) - 1
    run = dbg.run(ids, name="nat")
    run.until(f"t{last}.logits")
    lg = run.read(f"t{last}.logits").reshape(-1).float()
    crow = crow_of(run, item)
    del run
    run = dbg.run(ids, name="arms")
    run.until(f"t{last}.L{LC}.h")
    ck = run.checkpoint("pre")
    eff = []
    for x in (LC, LC + 1):
        for h in range(H):
            run.until(f"t{last}.L{x}.h")
            shape = run.port_shape("A")
            mask = torch.zeros(shape, dtype=torch.bool, device="cuda")
            mask.reshape(-1)[h * DH:(h + 1) * DH] = True
            run.set(f"t{last}.L{x}.A", torch.zeros(shape, dtype=ad.dtype, device="cuda"), mask=mask)
            run.until(f"t{last}.logits")
            eff.append(run.read(f"t{last}.logits").reshape(-1).float())
            run.restore(ck)
    del run
    return lg, crow, eff


def mdb_vcell(item, crow, donor_v):
    ids, last = item["ids"], len(item["ids"]) - 1
    run = dbg.run(ids, name="vx", prefill=crow)
    run.until(f"t{crow}.L{LC}.h")
    shape = run.port_shape("v")
    val = torch.zeros(shape, dtype=ad.dtype, device="cuda")
    mask = torch.zeros(shape, dtype=torch.bool, device="cuda")
    val.reshape(-1)[KVC * DH:(KVC + 1) * DH] = donor_v.to("cuda", ad.dtype)
    mask.reshape(-1)[KVC * DH:(KVC + 1) * DH] = True
    run.set(f"t{crow}.L{LC}.v", val, mask=mask)
    run.until(f"t{last}.logits")
    out = run.read(f"t{last}.logits").reshape(-1).float()
    del run
    return out


def mdb_donor_v(item):
    ids, last = item["ids"], len(item["ids"]) - 1
    run = dbg.run(ids, name="donor")
    run.until(f"t{last}.logits")
    crow = crow_of(run, item)
    v = run.read_cache(LC, "v", staged=False)[0, KVC, crow].clone()
    del run
    return crow, v


def tl_donor_v(bridge, item, crow):
    got = {}

    def grab(x, hook):
        got["v"] = x[0, crow, KVC * DH:(KVC + 1) * DH].clone()
        return x
    bridge.run_with_hooks(torch.tensor([item["ids"]], device="cuda"), fwd_hooks=[(HV, grab)])
    return got["v"]


def tl_arms(bridge, item, crow, donor_v, batch=True):
    """[base, zero head h at LC (H), at LC+1 (H), V interchange] last-row logits; pattern at LC."""
    ids = torch.tensor([item["ids"]], device="cuda")
    n = 2 * H + 2
    pat = {}

    def zero(x_layer):
        def fn(x, hook):
            for j in range(x.shape[0]):
                arm = j if batch else fn.arm
                k = arm - 1 - (0 if x_layer == LC else H)
                if 0 <= k < H and (arm - 1) // H == (0 if x_layer == LC else 1):
                    x[j, -1, k * DH:(k + 1) * DH] = 0
            return x
        return fn

    def vx(x, hook):
        for j in range(x.shape[0]):
            arm = j if batch else vx.arm
            if arm == n - 1:
                x[j, crow, KVC * DH:(KVC + 1) * DH] = donor_v.to(x.dtype)
        return x

    def keep_p(x, hook):
        if (batch and "p" not in pat) or (not batch and vx.arm == 0):
            pat["p"] = x[0, :, -1].float().cpu().clone()
        return x
    hooks = [(HZ[LC], zero(LC)), (HZ[LC + 1], zero(LC + 1)), (HV, vx), (HP, keep_p)]
    if batch:
        lg = bridge.run_with_hooks(ids.expand(n, -1), fwd_hooks=hooks)[:, -1].float().cpu()
        return lg, pat["p"]
    rows = []
    for arm in range(n):
        for fn in (hooks[0][1], hooks[1][1], vx):
            fn.arm = arm
        rows.append(bridge.run_with_hooks(ids, fwd_hooks=hooks)[0, -1].float().cpu())
    return torch.stack(rows), pat["p"]


recs = []
fh = open(f"{OUT}/exp6-{Path(MODEL).name}{'-smoke' if PICK else ''}.jsonl", "w")
dcache = {}
for item in base:
    t0 = time.monotonic()
    donor = items[item["donor"]]
    a, b = item["answer"], donor["answer"]
    if donor["id"] not in dcache:
        dcache[donor["id"]] = mdb_donor_v(donor)
    dcrow, dv = dcache[donor["id"]]
    lg, crow, heads = mdb_item(item, dv)
    vx = mdb_vcell(item, crow, dv)
    native = torch.stack([lg] + heads + [vx]).cpu()
    M = lambda L: (L[:, a] - L[:, b]).double()  # noqa: E731
    rec = {"id": item["id"], "crow": crow, "donor_crow": dcrow, "a": a, "b": b,
           "mdb": M(native).tolist(),
           "max_abs_logit": float(native[:, [a, b]].abs().max())}
    pats = {}
    for name, br in (("U16", U16), ("P16", P16), ("U32", U32), ("P32", P32)):
        dvb = tl_donor_v(br, donor, dcrow)
        L_, p_ = tl_arms(br, item, crow, dvb, batch=True)
        rec[name] = M(L_).tolist()
        pats[name] = p_
        if name == "U16":
            L1, _ = tl_arms(br, item, crow, dvb, batch=False)
            rec["U16_b1"] = M(L1).tolist()
    rec["pattern_maxdiff"] = {"bf16": float((pats["P16"] - pats["U16"]).abs().max()),
                              "fp32": float((pats["P32"] - pats["U32"]).abs().max())}
    rec["pattern_heads_ge_0.1"] = {k: [h for h in range(H) if float((pats[p] - pats[u])[h].abs().max()) >= 0.1]
                                   for k, p, u in (("bf16", "P16", "U16"), ("fp32", "P32", "U32"))}
    # P16 claims enter mdb as sealed supplied forecasts of each patch effect, judged on mdb's floor
    run = dbg.run(item["ids"], name="seal")
    run.until(f"t{len(item['ids']) - 1}.L0.h")
    judged = []
    base_m = rec["mdb"][0]
    for k in range(1, len(rec["mdb"])):
        claim = rec["P16"][k] - rec["P16"][0]
        r = supplied.seal(run, target={"kind": "margin", "row": len(item["ids"]) - 1, "indices": [a, b]},
                          edit={"address": f"arm{k}", "row": len(item["ids"]) - 1, "layer": LC,
                                "op": "zero head A" if k < 2 * H + 1 else "V interchange"},
                          value=claim, library="transformer-lens", version=VER,
                          address_map={"A": HZ[LC], "v": HV, "processing": json.dumps(PROC["P16"])},
                          method="processed bridge (enable_compatibility_mode) patch effect, batched")
        j = supplied.judge(r, rec["mdb"][k] - base_m, magnitude=rec["max_abs_logit"], dtype=torch.bfloat16)
        judged.append({"record": r["id"], "native": j["native"]["value"], "claim": claim,
                       "floor": j["floor"]["value"], "native_effect": j["native_effect"]})
    rec["judged"] = judged
    del run
    rec["seconds"] = round(time.monotonic() - t0, 2)
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    log(f"F {item['id']} {rec['seconds']}s")
json.dump({"processing": PROC, "tl": VER, "hooks": {"A": HZ, "v": HV, "p": HP}, "heads": H, "kv": KV,
           "carrier": [LC, KVC]}, open(f"{OUT}/exp6-{Path(MODEL).name}-prov{'-smoke' if PICK else ''}.json", "w"),
          indent=1, default=str)
log("TOTAL")
