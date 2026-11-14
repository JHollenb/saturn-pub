# %% [markdown]
# # The attribution-graph gap, split into brackets: transcoder nodes on Qwen3-1.7B-Base
#
# **Question.** An attribution graph (circuit-tracer style) draws an edge from a transcoder
# feature to the output and labels it with a number: the effect of that feature on a logit
# margin. Two numbers are commonly used. The **direct edge** is the feature's write read
# straight through the unembedding (`rho_F c . a_s d_s`). The **frozen-gradient edge** is the
# gradient of the margin with the attention patterns and the RMSNorm scales held at the base
# run, so later layers respond linearly through their MLPs (`A_s = g . a_s d_s`). How far is
# each from what the model does when the feature is removed, and where does the gap come from?
#
# **How mdb answers it.** mdb runs the real model and edits it while it runs. Removing a node is
# a native edit (`m += -a_s d_s` at the node's MLP write). mdb can also *hold* parts of the
# forward at the base run's values while the edited run continues: the answer row's attention
# pattern `p` (a hold on `t@last.L>l.p`) and the RMSNorm scales (`t@last.L>l.norm_scale`, plus
# the final norm's scale). This gives three native arms per node:
#
# * `arm0`: the free removal (what the model does);
# * `arm1`: the removal with the attention pattern held;
# * `arm2`: the removal with the pattern and every norm scale held.
#
# The gap between `arm0` and the frozen-gradient edge `-A` splits into three **brackets**:
# re-routing `arm0 - arm1`, norm `arm1 - arm2` and nonlinearity `arm2 - (-A)` (what is left: the
# MLPs' nonlinearity over a finite edit).
#
# **The experiment** (showcase exp4, `results/showcase-20261114/exp4/`, one seed, bf16, eager):
# all 32 of exp4's two-hop items (`items.json`; `MDB_ITEMS=0,1` picks a subset), consumer `M = logit(answer) - logit(donor
# answer)` at the last row. Nodes are the 5 active features of the mwhanna
# `qwen3-1.7b-transcoders-lowl0` transcoders (layers L20..L24, answer row) with the largest
# direct effect, plus one random active feature at node 1's layer rescaled to node 1's write
# norm.
#
# **Framing (from the independent audit, `results/showcase-20261114/AUDIT-SHOWCASE.md`).** Most
# brackets are tiny. The median bracket is about one bf16 step against a floor of 2 steps, and
# only 44 of exp4's 160 node-items have any bracket above the floor. So the prereg's pass/fail
# lines (G1's 0.105 "fail", G2's largest-bracket vote) sit at quantisation resolution. The
# example therefore reads the result **on the node-items whose native removal is above the
# floor** (`|arm0| > floor`). There, on exp4's 32 items, the re-routing bracket is 0.095 of
# `|arm0|` (G1), the frozen-gradient edge accounts for 0.92 of the native effect (0.97 above 4
# floors) and the direct edge's native share is 0.61-0.64. The transcoder direct edge
# overstates the native effect by about 1.6x on above-floor nodes, not 2x. The frozen-gradient
# edge explains that gap: it is later layers' linear response. G4 (rank agreement of the edge
# and `arm0`) is Spearman 0.93.
#
# Everything runs as a Python library on Beast through mrun (`run_on_beast.sh`).

# %%
import json
import math
import os
import re
import statistics as st
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL = Path(os.environ.get("MDB_MODEL", "/mnt/big/llm-models/Qwen3-1.7B-Base"))
TC_DIR = os.environ.get("MDB_TRANSCODERS", "/mnt/big/llm-models/qwen3-1.7b-transcoders-lowl0")
LAYERS = list(range(20, 25))                 # the transcoder layers exp4 used (Base-fit)
PICK = os.environ.get("MDB_ITEMS", "")      # e.g. "0,1" for a smoke run (indices into items.json)
OUT = Path(os.environ.get("MRUN_ARTIFACTS_DIR") or os.environ.get("MDB_OUT") or "out")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = os.environ.get("MDB_DEVICE", "cuda")
T0 = time.monotonic()
ITEMS = json.load(open(HERE / "items.json"))["items"]
REF = json.load(open(HERE / "reference.json"))
if PICK:
    ITEMS = [ITEMS[int(k)] for k in PICK.split(",")]
RESULT: dict = {"model": str(MODEL), "transcoders": TC_DIR, "items": [i["id"] for i in ITEMS]}


def log(msg):
    print(f"[{time.monotonic() - T0:7.1f}s] {msg}", flush=True)

# %% [markdown]
# ## 1. Load the model, declare custody, open a record sink
#
# `load_local` + `attach` give the debugger (bf16, eager attention, as exp4). `dbg.custody`
# declares the script and the preregistration (`prereg.json`, exp4's, copied here): every record
# sealed from now on carries their sha256, the library versions and the kernel path.
# `dbg.records_to` appends every sealed record (the supplied-forecast edges, the branches
# records) to a JSONL file, verified on read-back.

# %%
import torch

from mdb.engine.models import load_local
from mdb.ops.debugger import attach

torch.set_grad_enabled(False)
dbg = attach(load_local(MODEL, dtype="bf16", device=DEVICE))
model = dbg.adapter.model
NL = dbg.adapter.layers
dbg.custody(script=HERE / "frozen_gradient_brackets.py", prereg=HERE / "prereg.json",
            kernel_path="mdb-stepped")
sink = dbg.records_to(OUT / "records.jsonl")
log(f"model loaded: {NL} layers, d_model {dbg.adapter.hidden_size}")

# %% [markdown]
# ## 2. Transcoders as an mdb decomposition (P3)
#
# `sae.open_transcoders` opens the mwhanna directory (`layer_<L>.safetensors`, nothing loaded
# until a layer is used). A `TranscoderBasis` reads the MLP input `L<l>.n` and writes the MLP
# output `L<l>.m`: `acts = relu(W_enc n + b_enc)`, write `m = b_dec + sum_f a_f d_f + e`, with
# the error `e` defined against the native `m`, so the identity closes in float64.
# `components.transcoder_binding` makes that a `ComponentBinding` of the run's MLP write: one
# component per active feature, plus `bdec` and `error`. `component_table` gives each
# component's exact share of a scalar consumer read straight through the unembedding at the
# native final-norm scale: that is the **direct edge** (`rho_F c . a_s d_s`), the same vector
# as `reads.readout(run, Margin).vector`.
#
# exp4 hand-wrote this loader, the encoder, the float64 re-encode and the direct effect; here
# they are mdb objects. The node rule (top 5 by direct effect, P10 "suspect lists") is not an
# mdb object yet, so it stays as three lines of plain Python.

# %%
from mdb.analysis import components, sae
from mdb.analysis.finite import f64
from mdb.analysis.forecast import Margin
from mdb.analysis.reads import readout

TSET = sae.open_transcoders("lowl0", TC_DIR, layers=LAYERS, cache=len(LAYERS))


def nominate(nat, last, consumer, item):
    """Nodes: the 5 active features (L20..L24, answer row) with the largest direct effect, plus
    exp4's random node (its recorded pick) rescaled to node 1's write norm. Plain Python (P10)."""
    nodes, writes = [], {}
    for x in LAYERS:
        cb = components.transcoder_binding(nat, f"tc{x}", f"t{last}.L{x}.m", TSET.get(x))
        table = components.component_table(nat, cb, consumer)
        assert table["partition"]["holds"], table["partition"]
        parts = {c.feature: c for c in cb.components if c.feature is not None}
        for row in table["components"]:
            f = row.get("feature")
            if f is not None and cb.acts.get(f, 0.0) > 0:
                nodes.append((float(row["to_consumer"]["value"]), x, f, cb.acts[f]))
                writes[(x, f)] = parts[f].value.reshape(-1)       # a_s d_s, float64
    nodes.sort(key=lambda z: -z[0])
    top = [{"kind": "node", "direct": de, "layer": x, "feature": f, "act": a,
            "w": writes[(x, f)]} for de, x, f, a in nodes[:5]]
    pick = item["random_node"]
    if pick is not None:
        de = {(x, f): d for d, x, f, _ in nodes}.get((pick["layer"], pick["feature"]))
        w = writes.get((pick["layer"], pick["feature"]))
        if w is None:     # not active under the float32 screen: say so, skip the node
            log(f"  random node L{pick['layer']}.f{pick['feature']} inactive here; skipped")
        else:
            w = w * (top[0]["w"].norm() / (w.norm() + 1e-30))
            top.append({"kind": "random", "direct": de, "layer": pick["layer"],
                        "feature": pick["feature"], "act": None, "w": w})
    return top

# %% [markdown]
# ## 3. The frozen-gradient edge (nnsight), addressed through mdb's name table
#
# The edge is a library number: nnsight's gradient of `M` with respect to the node layer's MLP
# output at the answer row, in one forward and backward pass. For every later layer the `q_proj`
# and `k_proj` outputs are detached (the edit touches only the answer row, whose pattern
# depends on the edit only through its own q and k, so this freezes `p` exactly). The norm
# outputs are replaced by `input * gamma * rho_base`, with `rho_base` read from a clean pass at
# every row, and the MLPs are left live. `A_s = g . a_s d_s`, and the edge forecast for the
# removal is `-A_s`.
#
# The nnsight module paths come from `interop.names.to_nnsight` (mdb's port name table, checked
# bitwise against nnsight in `tests/test_interop.py`) instead of exp4's hand-typed map.
# nnsight writes module outputs, so the norm writes go to the module whose *input* is the mdb
# port read (`L.h` -> `input_layernorm`, `L.r` -> `post_attention_layernorm`, `fn` -> the final
# norm).
#
# Each edge is sealed as a `supplied-forecast` record *before* any native arm runs. The record
# holds the library, its version, the address map and the custody, and goes to the sink.

# %%
import nnsight

from mdb.analysis import supplied
from mdb.engine.holds import rms_scale
from mdb.interop import names, nn

NN_VERSION = nn.version()
ADDR = {"m": "L{l}.m", "a": "L{l}.a", "q": "L{l}.q", "k": "L{l}.k", "r": "L{l}.r"}
AMAP = {f"t<last>.{v.format(l='<l>')}": names.to_nnsight(v.format(l=0)).replace("[0]", "[<l>]")
        for v in ADDR.values()}
AMAP["final norm"] = names.to_nnsight("fn")


def envoy(nm, address):
    """(nnsight envoy, attribute) of an mdb address via names.to_nnsight, e.g. L21.a ->
    (nm.model.layers[21].input_layernorm, 'output')."""
    *path, attr = names.to_nnsight(address).split(".")
    obj = nm                                    # the leading 'model' is the nnsight wrapper
    for part in path[1:]:
        hit = re.fullmatch(r"(\w+)\[(\d+)\]", part)
        obj = getattr(obj, hit[1])[int(hit[2])] if hit else getattr(obj, part)
    return obj, attr


def base_scales(ids, layer):
    """gamma * rho_base of every later norm at every row, from a clean nnsight pass."""
    nm = nn.wrap(model)
    with torch.no_grad(), nm.trace(torch.tensor([ids], device=DEVICE)):
        saved = nnsight.save({})
        for x2 in range(layer + 1, NL):
            saved[("a", x2)] = envoy(nm, f"L{x2}.a")[0].input.save()
            saved[("n", x2)] = envoy(nm, f"L{x2}.r")[0].input.save()
        saved[("F", 0)] = envoy(nm, "fn")[0].input.save()
    sc = {"a": {}, "n": {}}
    for x2 in range(layer + 1, NL):
        L2 = model.model.layers[x2]
        sc["a"][x2] = (rms_scale(saved[("a", x2)], L2.input_layernorm.variance_epsilon)
                       * L2.input_layernorm.weight.float()).detach()
        sc["n"][x2] = (rms_scale(saved[("n", x2)], L2.post_attention_layernorm.variance_epsilon)
                       * L2.post_attention_layernorm.weight.float()).detach()
    sc["F"] = (rms_scale(saved[("F", 0)], model.model.norm.variance_epsilon)
               * model.model.norm.weight.float()).detach()
    return sc


def frozen_grad(ids, layer, a, b):
    """dM/dm_l at the answer row with p and every norm scale frozen (nnsight)."""
    sc = base_scales(ids, layer)
    nm = nn.wrap(model)
    with nn._Frozen(model), torch.enable_grad(), nm.trace(torch.tensor([ids], device=DEVICE)):
        nm.model.embed_tokens.output.requires_grad_(True)
        mod, attr = envoy(nm, f"L{layer}.m")
        mout = getattr(mod, attr)
        for x in range(layer + 1, NL):
            ln1 = envoy(nm, f"L{x}.a")[0]
            ln1.output = (ln1.input * sc["a"][x]).to(torch.bfloat16)
            for port in ("q", "k"):
                proj = envoy(nm, f"L{x}.{port}")[0]
                proj.output = proj.output.detach()
            ln2 = envoy(nm, f"L{x}.r")[0]
            ln2.output = (ln2.input * sc["n"][x]).to(torch.bfloat16)
        fn = envoy(nm, "fn")[0]
        fn.output = (fn.input * sc["F"]).to(torch.bfloat16)
        lg = nm.lm_head.output[0, -1].float()
        M = lg[a] - lg[b]
        g = nnsight.save([])
        with M.backward():
            g.append(mout.grad[0, -1].detach().double().cpu())
    return g[0]

# %% [markdown]
# ## 4. The native arms: `Branches` with holds (P1 + P4)
#
# `dbg.branches(run, Margin(last, a, b), at="t<last>.L20.h")` takes a checkpoint at the branch
# point. Each arm restores it, switches its holds on, stages its edits at their layers, runs to
# the consumer's logits, reads `M`, releases the holds and restores again. All arms share one
# floor: 2 bf16 spacings at the base margin's larger `|logit|` (`verdict.margin_floor`, exp4's
# `floor_of`). Each arm row records `dM`, `|dM|` in floors and the stability band.
#
# Per node (layer `l`, write `w = a_s d_s`):
#
# * `arm0`: `m_l += -w` (the free removal);
# * `arm1`: the same edit with `hold t@last.L>l.p`: the answer row's attention weights at every
#   later layer are replaced by the base run's;
# * `arm2`: also `hold t@last.L>l.norm_scale`, with `final_scale="held"`. The RMSNorm *scale*
#   (not its output) of both norms at every later layer is held at the base run's,
#   `a = w_in * (h * rho_base).to(dtype)`, and the logits are read with the final norm's scale
#   held (`holds.final_held_logits`).
#
# Nulls: `null.L<l>` (holds on, no edit: the hold machinery must reproduce the base margin) and
# `random-gauss` (`add_null("random", like="n0.arm0")`: a seeded Gaussian at node 1's address
# with its write norm). The random-feature node gets its own three arms.
#
# exp4 built `arm1`/`arm2` by stepping every later layer and re-setting `p`, `a` and `n` by
# hand. The holds stage the same edits as labelled hold edits. exp4 read `arm2` as a float64
# linear readout of the final state at the base final scale (`rho_F,base c . (h_out -
# h_out,base)`), not through bf16 logits. To compare with it on the same footing, we also run
# each `arm2` once more with the holds switched on directly (`run.hold`) and read that float64
# readout (`arm2_f64`). The brackets use `arm2_f64`, as exp4's did.

# %%
from mdb.analysis.branches import edit
from mdb.analysis.verdict import floors, margin_floor


def hold_names(x, norms):
    out = [f"t@last.L>{x}.p"]
    return out + [f"t@last.L>{x}.norm_scale"] if norms else out


def arm2_f64(br, run, x, value, ro_vec, h_ref):
    """arm2 read as exp4 did: float64 rho_F,base c . (h_out - h_out,base) (holds on directly)."""
    last = br.row
    br.restore()
    for held in hold_names(x, True):
        run.hold(held, source=br.reference, name=f"f64:{held}")
    try:
        run.until(f"t{last}.L{x}.h")
        run.set(f"t{last}.L{x}.m", value, "add")
        run.until(f"t{last}.logits")
        h = f64(run.read(f"t{last}.L{NL - 1}.h_out")).reshape(-1).cpu()
    finally:
        for held in hold_names(x, True):
            run.release(f"f64:{held}")
        br.restore()
    return float(ro_vec @ h) - float(ro_vec @ h_ref)

# %% [markdown]
# ## 5. Run it: per item, nominate, seal the edges, measure the arms
#
# The library edge is judged against the native removal with `supplied.judge` (same floor,
# stability band: stable above 4 floors).

# %%
ROWS = []
for item in ITEMS:
    t_item = time.monotonic()
    ids, last = item["ids"], len(item["ids"]) - 1
    a, b = item["answer"], item["donor_answer"]
    cons = Margin(last, a, b)
    nat = dbg.run(ids, name="nat")
    nat.until(f"t{last}.logits")
    logits = nat.read(f"t{last}.logits").reshape(-1)
    M0 = float(logits[a].float() - logits[b].float())
    floor = margin_floor(logits, a, b)
    ro = readout(nat, cons)                                    # rho_F gamma (W_U[a] - W_U[b])
    ro_vec = f64(ro.vector).reshape(-1).cpu()
    h_ref = f64(nat.read(f"t{last}.L{NL - 1}.h_out")).reshape(-1).cpu()
    nodes = nominate(nat, last, cons, item)
    # the edges: one frozen gradient per node layer, sealed before any arm runs
    grads = {}
    for nd in nodes:
        x = nd["layer"]
        if x not in grads:
            grads[x] = frozen_grad(ids, x, a, b)
        nd["edge"] = -float(grads[x] @ nd["w"])
        nd["record"] = supplied.seal(
            nat, target={"kind": "margin", "row": last, "indices": [a, b]},
            edit={"address": f"t{last}.L{x}.m", "row": last, "layer": x,
                  "op": f"add -a d of transcoder L{x}.f{nd['feature']}"},
            value=nd["edge"], library="nnsight", version=NN_VERSION, address_map=AMAP,
            method="frozen-pattern, frozen-norm-scale gradient . write (attribution-graph edge)")
    # the arms
    run = dbg.run(ids, name="arms")
    br = dbg.branches(run, cons, at=f"t{last}.L{LAYERS[0]}.h", name=f"exp4:{item['id']}")
    branch_base_same = br.M0 == M0 and br.floor == floor     # Branches reads the same base
    for k, nd in enumerate(nodes):
        x = nd["layer"]
        nd["value"] = (-nd["w"]).to(torch.bfloat16).reshape(1, 1, -1).to(DEVICE)
        origin = {"kind": "supplied", "library": "mwhanna transcoders", "version": "lowl0",
                  "address_map": {f"t{last}.L{x}.m": names.to_nnsight(f"L{x}.m")},
                  "node": f"L{x}.f{nd['feature']}"}
        e = [edit(f"t{last}.L{x}.m", nd["value"], "add", origin=origin)]
        br.add(f"n{k}.arm0", e)
        br.add(f"n{k}.arm1", e, holds=hold_names(x, False))
        br.add(f"n{k}.arm2", e, holds=hold_names(x, True), final_scale="held")
    for x in sorted({nd["layer"] for nd in nodes}):
        br.add(f"null.L{x}", [], holds=hold_names(x, True), final_scale="held")
    br.add_null("random-gauss", "random", like="n0.arm0", seed=0)
    br.run()
    sealed_br = br.seal()
    res = {name: arm.result for name, arm in br.arms.items()}
    rec = {"id": item["id"], "M0": M0, "floor": floor, "branches_record": sealed_br["id"],
           "branch_base_same": branch_base_same,
           "random_gauss": res["random-gauss"]["dM"]["value"], "nodes": []}
    for k, nd in enumerate(nodes):
        x = nd["layer"]
        arms = {f"arm{j}": res[f"n{k}.arm{j}"]["dM"]["value"] for j in range(3)}
        arms["arm2_f64"] = arm2_f64(br, run, x, nd["value"], ro_vec, h_ref)
        verdict = supplied.judge(nd["record"], arms["arm0"], magnitude=max(
            abs(float(logits[a])), abs(float(logits[b]))), dtype=logits.dtype)
        rec["nodes"].append({
            "kind": nd["kind"], "layer": x, "feature": nd["feature"], "act": nd["act"],
            "direct": nd["direct"], "edge": nd["edge"], "record": nd["record"]["id"], **arms,
            "null": res[f"null.L{x}"]["dM"]["value"],
            "arm0_floors": floors(arms["arm0"], floor),
            "arm0_stability": res[f"n{k}.arm0"].get("stability"),
            "edge_sign_agrees": verdict["sign_agrees"],
            "rerouting": arms["arm0"] - arms["arm1"], "norm": arms["arm1"] - arms["arm2_f64"],
            "nonlin": arms["arm2_f64"] - nd["edge"]})
    rec["seconds"] = round(time.monotonic() - t_item, 2)
    ROWS.append(rec)
    log(f"{item['id']}: M0 {M0:.3f} floor {floor} nodes {len(nodes)} arms {len(br.arms)} "
        f"[{rec['seconds']}s]")
    del br, run, nat
with open(OUT / "exp4-items.jsonl", "w") as fh:
    for rec in ROWS:
        fh.write(json.dumps(rec) + "\n")

# %% [markdown]
# ## 6. One item, read as a debugger session
#
# Node by node: the direct edge (`-direct` is its forecast of the removal), the frozen-gradient
# edge, the three native arms, and the brackets in floors.

# %%
r = ROWS[0]
print(f"{r['id']}  M0 {r['M0']}  floor {r['floor']}")
print(f"{'node':>12} {'-direct':>8} {'edge':>8} {'arm0':>7} {'arm1':>7} {'arm2':>8} "
      f"{'reroute/fl':>10} {'norm/fl':>8} {'nonlin/fl':>9}")
for n in r["nodes"]:
    tag = f"L{n['layer']}.f{n['feature']}" + ("*" if n["kind"] == "random" else "")
    print(f"{tag:>12} {-(n['direct'] or 0):8.3f} {n['edge']:8.3f} {n['arm0']:7.3f} "
          f"{n['arm1']:7.3f} {n['arm2_f64']:8.3f} {n['rerouting'] / r['floor']:10.2f} "
          f"{n['norm'] / r['floor']:8.2f} {n['nonlin'] / r['floor']:9.2f}")
print("(* = the random node)")

# %% [markdown]
# ## 7. The key numbers, on the above-floor node-items, next to the showcase
#
# One rule (`summarise`) over the nodes measured now and over exp4's saved records for the
# **same items** (`reference.json`, job-09feeeb61cbe). Ratios are medians over node-items with
# `|arm0| > floor` (and `> 4` floors). `arm0/(-direct)` is the native share of the direct edge;
# its inverse is the direct edge's overstatement.

# %%


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for q in range(i, j + 1):
                rk[o[q]] = (i + j) / 2
            i = j + 1
        return rk
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    den = (sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry)) ** 0.5
    return num / den if den else None


def summarise(records):
    med = lambda xs: st.median(xs) if xs else None  # noqa: E731
    N = [dict(n, floor=r["floor"]) for r in records for n in r["nodes"] if n["kind"] == "node"]
    Z = [dict(n, floor=r["floor"]) for r in records for n in r["nodes"] if n["kind"] == "random"]
    out = {"items": len(records), "node_items": len(N)}
    for k in (1, 4):
        E = [n for n in N if abs(n["arm0"]) > k * n["floor"]]
        out[f">{k}fl"] = {
            "n": len(E),
            "G1_rerouting_over_arm0": med([abs(n["rerouting"]) / abs(n["arm0"]) for n in E]),
            "arm0_over_edge": med([n["arm0"] / n["edge"] for n in E if n["edge"]]),
            "arm0_over_direct": med([n["arm0"] / -n["direct"] for n in E if n["direct"]]),
            "edge_over_direct": med([n["edge"] / -n["direct"] for n in E if n["direct"]])}
    out["direct_overstatement_>1fl"] = (1 / out[">1fl"]["arm0_over_direct"]
                                        if out[">1fl"]["arm0_over_direct"] else None)
    out["any_bracket_above_floor"] = sum(
        any(abs(n[k]) > n["floor"] for k in ("rerouting", "norm", "nonlin")) for n in N)
    out["G1_all"] = med([abs(n["rerouting"]) / abs(n["arm0"]) for n in N if n["arm0"]])
    out["G3_norm_below_floor"] = sum(abs(n["norm"]) <= n["floor"] for n in N)
    out["G4_spearman_edge_arm0"] = spearman([n["edge"] for n in N], [n["arm0"] for n in N])
    out["median_abs_brackets"] = {k: med([abs(n[k]) for n in N])
                                  for k in ("rerouting", "norm", "nonlin")}
    out["null_machinery_within_floor"] = sum(
        abs(n["null"]) <= r["floor"] for r in records for n in r["nodes"])
    out["random_node_all_brackets_within_floor"] = sum(
        all(abs(z[k]) <= z["floor"] for k in ("rerouting", "norm", "nonlin")) for z in Z)
    out["random_nodes"] = len(Z)
    return out


ref_records = [REF["items"][i["id"]] for i in ITEMS]
now = summarise(ROWS)
show = summarise(ref_records)
RESULT["summary_now"], RESULT["summary_showcase_same_items"] = now, show
RESULT["full_panel_audited"] = REF["full_panel_audited"]


def flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


fmt = lambda v: "None" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))  # noqa: E731
print(f"{'key number (same items)':48} {'showcase':>10} {'now':>10}")
for (key, v_show), (_, v_now) in zip(flat(show), flat(now)):
    print(f"{key:48} {fmt(v_show):>10} {fmt(v_now):>10}")
gauss = [r["random_gauss"] for r in ROWS]
RESULT["random_gauss_within_floor"] = f"{sum(abs(g) <= r['floor'] for g, r in zip(gauss, ROWS))}/{len(ROWS)}"
print("Gaussian matched-norm null (new) within floor:", RESULT["random_gauss_within_floor"])
above = [n for r in ROWS for n in r["nodes"] if n["kind"] == "node" and abs(n["arm0"]) > r["floor"]]
RESULT["edge_sign_agrees_above_floor"] = f"{sum(bool(n['edge_sign_agrees']) for n in above)}/{len(above)}"
print("edge sign agrees with the native removal (above floor):", RESULT["edge_sign_agrees_above_floor"])

# %% [markdown]
# ## 8. Do the mdb objects reproduce the showcase, node for node?
#
# Same items, same model, same kernel path, so the numbers should be equal:
#
# * the nodes (layer, feature) and the random node;
# * `direct` and `edge`: mdb's float64 re-encode (`TranscoderBasis.encode`) and exp4's
#   hand-written one run the same float64 matmul over a different set of rows, so the
#   activations, and hence `direct` and `A = g . a_s d_s`, can differ in the last bit
#   (smoke job-2092d2852921: 2e-16 and 9e-16). We report the max difference;
# * `arm0`, `arm1` (bf16 logits through `Branches` and holds vs exp4's hand-stepped arms:
#   bitwise expected);
# * `arm2_f64` (holds vs exp4's hand-set `p`, `a`, `n`, same float64 readout: equal up to
#   float64 association order in the readout);
# * `arm2` from `Branches` reads bf16 logits through `final_held_logits`, so it differs from
#   exp4's float64 readout by bf16 rounding; we report that difference in floors.

# %%
checks = {"branch_base_same": sum(r["branch_base_same"] for r in ROWS), "nodes_same": 0, "items": 0, "arm0_equal": 0, "arm1_equal": 0, "edge_equal": 0,
          "null_equal": 0, "node_items": 0}
dmax = {"direct": 0.0, "edge": 0.0, "arm2_f64": 0.0, "arm2_bf16_vs_f64_floors": 0.0}
for r, s in zip(ROWS, ref_records):
    checks["items"] += 1
    same = [(n["kind"], n["layer"], n["feature"]) for n in r["nodes"]] == \
           [(n["kind"], n["layer"], n["feature"]) for n in s["nodes"]]
    checks["nodes_same"] += same
    for n in r["nodes"]:
        dmax["arm2_bf16_vs_f64_floors"] = max(dmax["arm2_bf16_vs_f64_floors"],
                                              abs(n["arm2"] - n["arm2_f64"]) / r["floor"])
    if not same:
        continue
    for n, m in zip(r["nodes"], s["nodes"]):
        checks["node_items"] += 1
        checks["arm0_equal"] += n["arm0"] == m["arm0"]
        checks["arm1_equal"] += n["arm1"] == m["arm1"]
        checks["edge_equal"] += n["edge"] == m["edge"]
        checks["null_equal"] += n["null"] == m["null"]
        if n["direct"] is not None and m["direct"] is not None:
            dmax["direct"] = max(dmax["direct"], abs(n["direct"] - m["direct"]))
        dmax["edge"] = max(dmax["edge"], abs(n["edge"] - m["edge"]))
        dmax["arm2_f64"] = max(dmax["arm2_f64"], abs(n["arm2_f64"] - m["arm2"]))
RESULT["reproduction"] = {"counts": checks, "max_abs_diff": dmax}
print(json.dumps(RESULT["reproduction"], indent=1))
RESULT["records"] = {"path": str(OUT / "records.jsonl"),
                     "kinds": {}}
for rec_ in sink.records():
    RESULT["records"]["kinds"][rec_["kind"]] = RESULT["records"]["kinds"].get(rec_["kind"], 0) + 1
custody_block = next((x["body"].get("custody") for x in sink.records() if x["kind"] == "branches"), None)
RESULT["custody"] = {k: custody_block.get(k) for k in ("script_sha256", "prereg_sha256", "kernel_path")} \
    if custody_block else None
print("records:", RESULT["records"]["kinds"], "custody:", json.dumps(RESULT["custody"])[:300])

# %%
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("matplotlib not installed: skipping the plot")
if plt is not None:
    N = [dict(n, floor=r["floor"]) for r in ROWS for n in r["nodes"] if n["kind"] == "node"]
    up = [n for n in N if abs(n["arm0"]) > n["floor"]]
    lo = [n for n in N if abs(n["arm0"]) <= n["floor"]]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for axis, key, title in ((ax[0], "edge", "frozen-gradient edge"),
                             (ax[1], "direct", "direct edge (sign flipped)")):
        sign = -1 if key == "direct" else 1
        axis.scatter([sign * n[key] for n in lo], [n["arm0"] for n in lo], s=12, c="#bbbbbb",
                     label="|arm0| <= floor")
        axis.scatter([sign * n[key] for n in up], [n["arm0"] for n in up], s=14, c="#1f6fb4",
                     label="|arm0| > floor")
        lim = min(sign * n[key] for n in N) * 1.1
        axis.plot([lim, 0], [lim, 0], "k--", lw=0.8)
        axis.set_xlabel(f"{title} forecast of the removal")
        axis.set_ylabel("native removal arm0")
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "brackets.png", dpi=130)
    try:
        from IPython.display import display
        display(fig)
    except ImportError:
        pass
RESULT["seconds"] = round(time.monotonic() - T0, 1)
json.dump(RESULT, open(OUT / "brackets_results.json", "w"), indent=1, default=str)
print(f"done in {RESULT['seconds']}s; wrote {OUT}/brackets_results.json")
print("RESULT_JSON " + json.dumps(RESULT, default=str))
