# %% [markdown]
# # SAE features as component bindings of a carrier cell (Qwen3-1.7B-Base, Qwen-Scope)
#
# **Question.** In a two-hop prompt ("the capital of the country whose currency is the yen is"),
# Qwen3-1.7B-Base carries the bridge country in one value cell: layer 20, KV group 6, at the
# span row the answer row attends to most (the *carrier*, found by a frozen rule). A sparse
# autoencoder (SAE) splits the residual that feeds that cell into features. Is the country
# carried by one named feature, and does removing just that feature's part of the cell matter
# to the answer?
#
# **How mdb answers it.** mdb is a debugger for language models: it runs the real model, lets
# you read and edit any cell, and measures the native change of a consumer (here the margin
# `M = logit(answer capital) - logit(donor's capital)` at the last row).
#
# * `Binding.component` splits the carrier cell `cache.L20.v[kv6,row=r]` over the Qwen-Scope L19
#   basis into **components**: one per active feature, the decoder bias and the SAE error. The
#   parts add up to the cell's float64 image exactly.
# * `symbols.table(..., components=[...], causal=True)` puts that binding in the symbol table:
#   each component's exact direct share of `M`, its content readout over the 57 panel
#   countries (does it name the right country?), the SAELens ids, and the measured liveness of
#   the declared `carrier` symbol.
# * `Branches` measures, per layer L15..L23, four native arms at the carrier row with one shared
#   floor: remove the top country component, remove a random non-country component rescaled to
#   the same norm, remove the error component, zero the cell. A built-in matched-norm Gaussian
#   null (`add_null("random")`) runs next to them at L20.
#
# **Result to look for (audited, `results/showcase-20261114/AUDIT-SHOWCASE.md`).** L2 is the
# headline: at L20, removing the named top country component moves `M` by a median **0.59** of
# zeroing the whole cell; a matched-norm random component **0.0**; the error component
# **0.11** (64 items). The partition identity (components sum to the cell on 64/64, max gap
# 1.8e-15) is a float64 check of the code, not a finding about the model. The SAELens check
# (108/108 active sets) compares mdb's `to_saelens()` conversion with SAELens's TopK encode, not
# SAELens's own loader (that registry load fails offline, SC1).
#
# **This example** runs the first 24 of the 64 base items (16 currency and 8 person items,
# from the showcase's shard `s0`; `items.json` holds the first 32, `MDB_NITEMS` picks how
# many) and prints its numbers next to the showcase's numbers for the same items
# (`reference.json`). One seed, bf16, eager attention, Beast GPU.

# %%
import json
import os
import random
import statistics as st
import time
from pathlib import Path

HERE = Path(os.environ.get("MDB_EXAMPLE_DIR", "."))
MODEL = Path(os.environ.get("MDB_MODEL", "/mnt/big/llm-models/Qwen3-1.7B-Base"))
SAE = os.environ.get("MDB_SAE", "/mnt/big/llm-models/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50")
NITEMS = int(os.environ.get("MDB_NITEMS", "24"))      # first N of items.json (max 32)
DEVICE = os.environ.get("MDB_DEVICE", "cuda")
OUT = Path(os.environ.get("MRUN_ARTIFACTS_DIR") or os.environ.get("MDB_OUT") or "out")
OUT.mkdir(parents=True, exist_ok=True)
CL, CG = 20, 6                                          # carrier: L20, KV group 6
SWEEP_LAYERS = list(range(15, 24))                      # remove-component arms at these layers
LINEAGE = list(range(9, 20))                            # decoder-cosine lineage L9..L19
RECHECK = 2                                             # batched arms re-measured stepped
T0 = time.monotonic()
DATA = json.load(open(HERE / "items.json"))
REF = json.load(open(HERE / "reference.json"))["items"]
ITEMS = DATA["items"][:NITEMS]
COUNTRIES = DATA["countries"]
RESULT: dict = {"model": str(MODEL), "sae": SAE, "n_items": len(ITEMS)}


def log(msg):
    print(f"[{time.monotonic() - T0:7.1f}s] {msg}", flush=True)


# %% [markdown]
# ## 1. Load the model, attach mdb, declare custody
#
# `load_local` is mdb's loader (bf16, eager attention, no download); `mdb.attach` wraps the
# model. `dbg.custody(script=, prereg=)` stamps every record sealed from now on with the sha256
# of this script and of the preregistration (`prereg.json`, the showcase's exp2 prereg, copied
# unchanged), the library versions and the kernel path. `dbg.records_to` appends each sealed
# record (symbol tables, branches) to `records.jsonl`.

# %%
import torch
from transformers import AutoTokenizer

import mdb
from mdb.engine.models import load_local

torch.set_grad_enabled(False)
model = load_local(MODEL, dtype="bf16", device=DEVICE)
tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
dbg = mdb.attach(model, tok)
custody = dbg.custody(script=HERE / "sae_component_binding.py", prereg=HERE / "prereg.json",
                      kernel_path="batched cell arms + mdb-stepped rechecks/nulls/liveness")
sink = dbg.records_to(OUT / "records.jsonl")
RESULT["custody"] = {k: custody.get(k) for k in ("script_sha256", "prereg_sha256", "kernel_path")}
log(f"model loaded; custody script {str(custody['script_sha256'])[:12]} "
    f"prereg {str(custody['prereg_sha256'])[:12]}")

# %% [markdown]
# ## 2. The Qwen-Scope SAEs, and the SAELens check
#
# `open_set` gives mdb's `FeatureBasis` per layer (float64 identity
# `x = b_dec + sum_f a_f d_f + e`). The basis of layer `l - 1` (`resid_post`) holds the residual
# `L<l>.h` that feeds the value cell of layer `l`, so the carrier cell L20 uses the L19 basis.
#
# `to_saelens()` builds a SAELens `TopKSAE` from the same weights with a strict state-dict
# load. Per item we check that SAELens's `encode` picks the same active set as mdb. What this
# does and does not show: it checks mdb's conversion against SAELens's TopK encode. SAELens's own
# loader (`SAE.from_pretrained` from the registry) is attempted too, but it needs the Hugging
# Face hub; offline on Beast it fails (the showcase recorded `LocalEntryNotFoundError`, SC1).

# %%
from mdb.analysis import sae

need = sorted(set(LINEAGE) | {x - 1 for x in SWEEP_LAYERS})
SAES = sae.open_set("qwen-scope", SAE, layers=need, cache=len(need) + 1)
B = {x: SAES.get(x) for x in need}
B19 = B[19]
prov = {"basis": B19.describe()["saelens"]}
try:
    import sae_lens
    from sae_lens import SAE as SAELensSAE

    prov["sae_lens_version"] = sae_lens.__version__
    try:
        SAELensSAE.from_pretrained("qwen-scope-3-1.7b-base-w32k-l50", "layer19", device="cpu")
        prov["registry_load"] = {"ok": True}
    except Exception as error:  # noqa: BLE001  (offline: the hub is not reachable)
        prov["registry_load"] = {"ok": False, "error": f"{type(error).__name__}"}
    SL = B19.to_saelens()                                # strict load
    prov["to_saelens"] = "ok"
except ImportError as error:
    SL = None
    prov["to_saelens"] = f"sae_lens not installed: {error}"
RESULT["saelens"] = prov
print(json.dumps(prov, indent=1))


def saelens_agrees(h, ids):
    if SL is None:
        return None
    acts = SL.encode(h.float().reshape(1, -1).cpu())
    return sorted(int(i) for i in acts[0].nonzero().reshape(-1).tolist()) == sorted(ids)


log("SAEs ready")

# %% [markdown]
# ## 3. One item, step by step
#
# For each item:
#
# 1. Run the prompt natively to the last row's logits (`prefill` = the first span row, so the
#    span rows are *stepped* and their residuals can be read). The carrier row is the frozen
#    B1 rule: the argmax over the span of the answer row's L20 heads 12+13 attention
#    (`circuit_map.slot_rows`).
# 2. Declare the `carrier` symbol (`cache.L20.v[kv6,row=r]`) and split it with
#    `Binding.component` over the L19 basis.
# 3. `symbols.table(run, consumer=M, sets=[countries], components=[cb], limit=1, causal=True)`:
#    the page holds the declared carrier (its mass, content and measured liveness: the answer
#    row's reads of the cell removed) and the `components` section (per component: activation,
#    exact direct share of `M`, content readout, SAELens ids). A **country component** is a
#    feature component whose content readout names the item's true country; the **top** one is
#    the country component with the largest direct share. A **random component** is drawn
#    uniformly (seed 20261114) from the active non-country features, as preregistered.
# 4. The same split at every layer L15..L23 (basis `l - 1`) gives each layer's top and random
#    component (`component_table`).
# 5. One `Branches` object per item, branch point = the last row's first port, consumer `M`.
#    Per layer: `top` (the cell minus the top country component), `random` (minus the random
#    component rescaled to the top's norm), `error` (minus the error component), `zero` (the
#    cell zeroed). They are whole-cell replacements, so they run in batched forwards
#    (`br.run(mode="batched", recheck=2)`: the two largest are re-measured on the stepped path).
#    `add_null("gauss", "random", like="L20:top")` adds mdb's built-in null: a seeded Gaussian
#    change of the same norm at the same cell (stepped). Every arm row carries `dM`, the shared
#    floor, `effect_floors` and `above_floor`; `br.seal()` writes a `branches` record.
#
# The feature lineage (step 6) is plain Python: feature lifetime strips (P9) are not an mdb
# object yet. Starting from the L19 top country feature, each lower layer's feature with the
# largest decoder cosine to the previous one is followed; the onset is the lowest layer down to
# which every link has cosine >= 0.7 and is active at the carrier row (L9 is censored).
# Prompt-set iteration (P7) is also not built: the loop over items is a plain `for`.

# %%
from mdb.analysis import symbols
from mdb.analysis import circuit_map as cm
from mdb.analysis.branches import edit
from mdb.analysis.components import component_binding, component_table
from mdb.analysis.forecast import Margin
from mdb.analysis.symbols import Binding, Candidates
from mdb.analysis.verdict import margin_floor

CARRIER = cm.Slot("carrier", "carrier", "R_argmax", layers=(CL,), kv=(CG,), rule_layer=CL,
                  rule_heads=(12, 13))
rng = random.Random(20261114)        # the showcase's draw order, so the random picks match it


def comp_rows(tab):
    """The component rows of a component table, flattened."""
    out = []
    for r in tab["components"]:
        c = r.get("content", {}).get("countries", {})
        out.append({"c": r["component"], "f": r.get("feature"),
                    "e": r["to_consumer"]["value"], "best": c.get("best"),
                    "names_truth": c.get("names_truth"), "norm": r["value_norm"]["value"]})
    return out


def top_country(rows):
    cands = [r for r in rows if r["f"] is not None and r["names_truth"]]
    return max(cands, key=lambda r: r["e"]) if cands else None


def lineage(f19, acts_by_layer):
    """Backward decoder-cosine chain of f19 (plain Python, P9 not built)."""
    chain = {19: (f19, 1.0, f19 in acts_by_layer[19])}
    d = B19.direction(f19).float()
    for x in range(18, LINEAGE[0] - 1, -1):
        W = B[x].W_dec
        cos = (d @ W) / (W.norm(dim=0) * d.norm() + 1e-12)
        g = int(cos.argmax())
        chain[x] = (g, float(cos[g]), g in acts_by_layer[x])
        d = B[x].direction(g).float()
    onset = 19
    for x in range(19, LINEAGE[0] - 1, -1):
        g, c, on = chain[x]
        if c >= 0.7 and on:
            onset = x
        else:
            break
    return onset


def study(item):
    t = time.monotonic()
    ids, span = item["ids"], item["span"]
    last = len(ids) - 1
    ans, foil, country = item["answer"], item["donor"]["answer"], item["country_token"]
    M = Margin(last, ans, foil)
    cands = [Candidates("countries", tuple(COUNTRIES), truth=country)]
    # 1. native run and carrier row
    run = dbg.run(ids, name="nat", prefill=span[0])
    first_port = run.program.names[0]
    run.until(f"t{last}.logits")
    nat = run.read(f"t{last}.logits")
    crow = cm.slot_rows(CARRIER, run, {"name": item["id"], "span": span, "last": last})[0]
    rec = {"id": item["id"], "carrier_row": crow,
           "correct": int(nat.reshape(-1).argmax()) == ans}
    h19 = run.read(f"t{crow}.L{CL}.h")
    enc = B19.encode(h19)
    rec["saelens_agrees"] = saelens_agrees(h19, enc["ids"])
    # 2-3. the declared carrier, its component binding, the symbol table
    cell = f"cache.L{CL}.v[kv{CG},row={crow}]"
    symbols.declare(run, "carrier", [cell])
    cb = Binding.component(run, "carrier-features", cell, B19)
    table = symbols.table(run, consumer=M, sets=cands, components=[cb], limit=1, causal=True,
                          tokenizer=tok)
    carrier = table.report["bindings"][0]
    tab = table.report["components"][0]
    rows = comp_rows(tab)
    ce = tab["cell_to_consumer"]["value"]
    top = top_country(rows)
    nonc = [r for r in rows if r["f"] is not None and not r["names_truth"]]
    rnd = rng.choice(nonc) if nonc else None
    err = [r for r in rows if r["c"] == "error"][0]
    rec.update(cell_e=ce, top_feature=top["f"] if top else None,
               top_share=top["e"] / ce if top and ce else None,
               error_share=err["e"] / ce if ce else None,
               random_feature=rnd["f"] if rnd else None,
               random_share=rnd["e"] / ce if rnd and ce else None,
               partition_gap=tab["partition"]["gap"]["value"],
               partition_holds=tab["partition"]["holds"],
               carrier_symbol={"mass": carrier.get("mass"), "causal": carrier.get("causal"),
                               "content_best": (carrier.get("content") or {})
                               .get("countries", {}).get("best")})
    # 6. lineage of the top feature (plain Python)
    if top is not None:
        acts_by = {x: set(B[x].encode(run.read(f"t{crow}.L{x}.h_out"))["ids"]) for x in LINEAGE}
        rec["lineage_onset"] = lineage(top["f"], acts_by)
    # 4. per-layer component bindings: each layer's top and random component
    per_layer = {}
    for x in SWEEP_LAYERS:
        cbx = component_binding(run, f"c{x}", f"cache.L{x}.v[kv{CG},row={crow}]", B[x - 1])
        rx = comp_rows(component_table(run, cbx, M, sets=cands))
        tp = top_country(rx)
        nc = [r for r in rx if r["f"] is not None and not r["names_truth"]]
        per_layer[x] = (cbx, tp, rng.choice(nc) if nc else None)
    rec["layer_top"] = {str(x): (v[1]["f"] if v[1] else None) for x, v in per_layer.items()}
    rec["layer_random"] = {str(x): (v[2]["f"] if v[2] else None) for x, v in per_layer.items()}
    dt = run.read_cache(CL, "v", staged=False).dtype
    block = lambda v: v.to(dt).reshape(1, 1, 1, -1)   # the cell block shape (1, kv, rows, d)  # noqa: E731
    del run
    # 5. Branches: one checkpoint, all arms, one floor
    rb = dbg.run(ids, name="branches", prefill=span[0])
    br = dbg.branches(rb, M, at=f"t{last}.L0.{first_port}", name=f"exp2:{item['id']}")
    batched = []
    for x, (cbx, tp, rn) in per_layer.items():
        target = f"cache.L{x}.v[kv{CG},row={crow}]"
        if tp is not None:
            br.add(f"L{x}:top", [edit(target, block(cbx.edited(remove=(tp["c"],))))])
            batched.append(f"L{x}:top")
            if rn is not None:
                vr = cbx.get(rn["c"]).value
                scale = cbx.get(tp["c"]).value.norm() / (vr.norm() + 1e-30)
                br.add(f"L{x}:random", [edit(target, block(cbx.edited(add=-vr * scale)))])
                batched.append(f"L{x}:random")
        br.add(f"L{x}:error", [edit(target, block(cbx.edited(remove=("error",))))])
        br.add(f"L{x}:zero", [edit(target, block(torch.zeros_like(cbx.native)))])
        batched += [f"L{x}:error", f"L{x}:zero"]
    br.run(batched, mode="batched", recheck=RECHECK)
    if f"L{CL}:top" in br.arms:
        br.add_null(f"L{CL}:gauss", "random", like=f"L{CL}:top", seed=20261114)
        br.run([f"L{CL}:gauss"])                         # an `add` edit: stepped
    br.seal()
    res = {n: a.result for n, a in br.arms.items()}
    nvs = res[batched[0]]["kernel_delta_null"]["value"]                  # whole vocabulary
    nvs_own = res[batched[0]]["kernel_delta_null_consumer_tokens"]["value"]   # answer, foil
    rec.update(base_m=br.M0, floor=res[batched[0]]["floor"]["value"], branch_floor=br.floor,
               null_vs_stepped=nvs, null_vs_stepped_consumer_tokens=nvs_own,
               floor_showcase_rule=margin_floor(br.base_logits, ans, foil, extra=nvs_own),
               arms={n.replace("L", "", 1): r["dM"]["value"] for n, r in res.items()},
               effect_floors={n: (r.get("effect_floors") or {}).get("value")
                              for n, r in res.items()},
               rechecks={n: {"batched": r["dM"]["value"], "stepped": r["stepped"]["dM"]["value"]}
                         for n, r in res.items() if "stepped" in r})
    del rb, br
    rec["seconds"] = round(time.monotonic() - t, 2)
    return rec


# %% [markdown]
# ## 4. Run the items
#
# The log prints, per item, the top country feature, its share of the carrier's direct effect
# on `M`, and the L20 arms as fractions of zeroing the cell, next to the showcase's values for
# the same item.

# %%
RECS = []
for item in ITEMS:
    r = study(item)
    RECS.append(r)
    ref = REF[r["id"]]
    z, rz = r["arms"].get("20:zero"), ref["arms"].get("20:zero")
    frac = lambda arms, k, zz: (arms.get(f"20:{k}") / zz) if zz and f"20:{k}" in arms else None  # noqa: E731
    fmt = lambda v: "  -  " if v is None else f"{v:5.2f}"  # noqa: E731
    print(f"{r['id']:28} f{r['top_feature']} (showcase f{ref['top_feature']}) "
          f"share {fmt(r['top_share'])}/{fmt(ref['top_share'])}  L20 top/zero "
          f"{fmt(frac(r['arms'], 'top', z))}/{fmt(frac(ref['arms'], 'top', rz))}  "
          f"rnd {fmt(frac(r['arms'], 'random', z))}  gauss {fmt(frac(r['arms'], 'gauss', z))}  "
          f"err {fmt(frac(r['arms'], 'error', z))}  [{r['seconds']}s]", flush=True)
RESULT["items"] = RECS
log("items done")

# %% [markdown]
# ## 5. Key numbers: now versus the showcase on the same items
#
# One scorer runs over this run's records and over the showcase's records for the same items
# (`reference.json`, from `results/showcase-20261114/exp2/out/exp2-items-s0.jsonl`):
#
# * **L2** (headline): median over items of `dM(arm) / dM(zero cell)` at L20.
# * **L1**: the top country component's share of the carrier's direct effect (bar: >= 0.5 on
#   40/64), the error component's share (bar: <= 0.2 on 48/64). The preregistered bars are for
#   64 items; here the counts are over the subset.
# * **L4**: the layers at which removing the top country component is above the floor. The
#   showcase floor is 2 bf16 spacings at the larger |logit| plus the batched-vs-stepped delta of
#   the null arm measured on the two margin tokens, once. The `Branches` floor of a batched arm
#   adds the null delta over the whole vocabulary, twice (a margin is a difference of two
#   logits), so it is larger and L4 is shown with both rules (the row field
#   `kernel_delta_null_consumer_tokens` gives the showcase's delta).
# * **L3**: lineage onset (median; the preregistered falsifier is onset >= L15).
# * **partition**: the largest float64 gap between the sum of the component shares and the
#   cell image's share. This is a code check (an algebraic identity), not a model finding.

# %%
med = lambda xs: st.median(xs) if xs else None  # noqa: E731


def score(recs, floor_key):
    n = len(recs)
    out = {"n": n}
    tops = [r["top_share"] for r in recs if r["top_share"] is not None]
    errs = [r["error_share"] for r in recs if r["error_share"] is not None]
    rnds = [r["random_share"] for r in recs if r["random_share"] is not None]
    out["L1"] = {"top_share_ge_0.5": sum(s >= 0.5 for s in tops),
                 "error_share_le_0.2": sum(s <= 0.2 for s in errs),
                 "median_top_share": med(tops), "median_error_share": med(errs),
                 "median_random_share": med(rnds)}
    rat = {"top": [], "random": [], "error": [], "gauss": []}
    for r in recs:
        z = r["arms"].get("20:zero")
        if not z:
            continue
        for k in rat:
            if f"20:{k}" in r["arms"]:
                rat[k].append(r["arms"][f"20:{k}"] / z)
    out["L2"] = {k: med(v) for k, v in rat.items() if v}
    out["L2"]["n"] = len(rat["top"])
    hist = {x: 0 for x in SWEEP_LAYERS}
    inside = never = 0
    for r in recs:
        above = [x for x in SWEEP_LAYERS if abs(r["arms"].get(f"{x}:top", 0.0)) > r[floor_key]]
        for x in above:
            hist[x] += 1
        inside += bool(above) and set(above) <= set(range(17, 22))
        never += not above
    out["L4"] = {"only_inside_L17_21": inside, "never_above_floor": never,
                 "layer_histogram": {str(k): v for k, v in hist.items()}}
    on = [r["lineage_onset"] for r in recs if r.get("lineage_onset") is not None]
    out["L3"] = {"median_onset": med(on), "onset_ge_L15": sum(o >= 15 for o in on), "n": len(on)}
    out["partition_max_gap"] = max(r["partition_gap"] for r in recs)
    out["saelens_agrees"] = sum(1 for r in recs if r["saelens_agrees"])
    out["correct"] = sum(1 for r in recs if r["correct"])
    return out


ref_recs = [REF[r["id"]] for r in RECS]
now = score(RECS, "floor")
now_rule = score(RECS, "floor_showcase_rule")
then = score(ref_recs, "floor")
RESULT["summary"] = {"now_branches_floor": now, "now_showcase_floor_rule": now_rule,
                     "showcase_same_items": then}
pairs = [("L2 median top/zero", now["L2"].get("top"), then["L2"].get("top")),
         ("L2 median random-component/zero", now["L2"].get("random"), then["L2"].get("random")),
         ("L2 median error/zero", now["L2"].get("error"), then["L2"].get("error")),
         ("L2 median gauss-null/zero (new)", now["L2"].get("gauss"), None),
         ("L1 median top share", now["L1"]["median_top_share"], then["L1"]["median_top_share"]),
         ("L1 median error share", now["L1"]["median_error_share"],
          then["L1"]["median_error_share"]),
         ("L1 top share >= 0.5 (count)", now["L1"]["top_share_ge_0.5"],
          then["L1"]["top_share_ge_0.5"]),
         ("L1 error share <= 0.2 (count)", now["L1"]["error_share_le_0.2"],
          then["L1"]["error_share_le_0.2"]),
         ("L3 median onset", now["L3"]["median_onset"], then["L3"]["median_onset"]),
         ("L3 onset >= L15 (count)", now["L3"]["onset_ge_L15"], then["L3"]["onset_ge_L15"]),
         ("L4 only inside L17..21, showcase floor rule", now_rule["L4"]["only_inside_L17_21"],
          then["L4"]["only_inside_L17_21"]),
         ("L4 only inside L17..21, Branches floor", now["L4"]["only_inside_L17_21"], None),
         ("L4 above floor at L20, showcase floor rule", now_rule["L4"]["layer_histogram"]["20"],
          then["L4"]["layer_histogram"]["20"]),
         ("partition max gap (code check)", now["partition_max_gap"], then["partition_max_gap"]),
         ("SAELens active set agrees (count)", now["saelens_agrees"], then["saelens_agrees"]),
         ("correct (count)", now["correct"], then["correct"])]
print(f"{'key number':48} {'now':>12} {'showcase':>12}")
for name, a, b in pairs:
    f = lambda v: "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
    print(f"{name:48} {f(a):>12} {f(b):>12}")
print("L4 layer histogram now (Branches floor):", now["L4"]["layer_histogram"])
print("L4 layer histogram now (showcase rule): ", now_rule["L4"]["layer_histogram"])
print("L4 layer histogram showcase:            ", then["L4"]["layer_histogram"])

# %% [markdown]
# Per-item agreement with the showcase: the same carrier row, the same top country feature at
# every layer, the same random picks, and the same `dM` for every batched arm (same kernel path,
# same arm order). The two re-measured stepped arms per item give the batched-vs-stepped
# difference; the liveness of the declared `carrier` (stepped, fresh prefix cache) must agree
# with the batched `zero` arm at L20 within that difference.

# %%
same = {"carrier_row": 0, "top_feature": 0, "layer_tops": 0, "random_picks": 0, "arms_dM": 0}
max_arm_diff, live_diff, kernel = 0.0, [], []
for r, ref in zip(RECS, ref_recs):
    same["carrier_row"] += r["carrier_row"] == ref["carrier_row"]
    same["top_feature"] += r["top_feature"] == ref["top_feature"]
    same["layer_tops"] += all(r["layer_top"][k] == ref["layer_top"][k] for k in ref["layer_top"])
    same["random_picks"] += (r["random_feature"] == ref["random_feature"] and all(
        r["layer_random"][k] == ref["layer_random"][k] for k in ref["layer_random"]))
    common = [k for k in ref["arms"] if k in r["arms"]]
    diffs = [abs(r["arms"][k] - ref["arms"][k]) for k in common]
    same["arms_dM"] += len(common) == len(ref["arms"]) and max(diffs, default=0.0) == 0.0
    max_arm_diff = max([max_arm_diff, *diffs])
    causal = (r["carrier_symbol"]["causal"] or {}).get("delta")
    if isinstance(causal, dict) and "20:zero" in r["arms"]:
        live_diff.append(abs(causal["value"] - r["arms"]["20:zero"]))
    kernel += [abs(v["batched"] - v["stepped"]) for v in r["rechecks"].values()]
agree = {**{k: f"{v}/{len(RECS)}" for k, v in same.items()},
         "max_abs_arm_dM_difference": max_arm_diff,
         "max_liveness_vs_zero_arm": max(live_diff, default=None),
         "max_batched_vs_stepped_recheck": max(kernel, default=None),
         "median_floor_branches": med([r["floor"] for r in RECS]),
         "median_floor_showcase_rule": med([r["floor_showcase_rule"] for r in RECS]),
         "records_sealed": sink.count}
RESULT["agreement"] = agree
print(json.dumps(agree, indent=1))
# The preregistered L2 bars (top >= 0.5, random <= 0.15, error <= 0.2) are judged on the
# 64-item median; on a subset they are shown for information, not checked.
print("L2 prereg bars on this subset (information): top >= 0.5", (now["L2"].get("top") or 0) >= 0.5,
      "| random <= 0.15", abs(now["L2"].get("random") or 0) <= 0.15,
      "| error <= 0.2", abs(now["L2"].get("error") or 0) <= 0.2)
checks = [("L2 medians equal the showcase's on these items",
           all(now["L2"].get(k) == then["L2"].get(k) for k in ("top", "random", "error"))),
          ("partition holds on every item (code check)", all(r["partition_holds"] for r in RECS)),
          ("SAELens active set agrees on every item", now["saelens_agrees"] == len(RECS)),
          ("same top country feature as the showcase", same["top_feature"] == len(RECS)),
          ("every batched arm dM equals the showcase's", same["arms_dM"] == len(RECS)),
          ("L4 (showcase floor rule) equals the showcase's",
           now_rule["L4"] == then["L4"])]
RESULT["checks"] = {name: bool(ok) for name, ok in checks}
for name, ok in checks:
    print("PASS" if ok else "DIFF", name)
print(f"{sum(ok for _, ok in checks)}/{len(checks)} checks pass")

# %%
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("matplotlib not installed: skipping the plot")
if plt is not None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    arms = ["top", "random", "gauss", "error"]
    ax[0].bar([a for a in arms], [now["L2"].get(a) or 0.0 for a in arms], color="#3b6ea5")
    ax[0].axhline(0.0, color="k", lw=0.5)
    ax[0].set_title("L20: median dM(arm) / dM(zero cell)")
    hist = now_rule["L4"]["layer_histogram"]
    ax[1].bar(list(hist), list(hist.values()), color="#b5651d")
    ax[1].set_title("items with the top-component removal above floor")
    ax[1].set_xlabel("layer")
    fig.tight_layout()
    fig.savefig(OUT / "sae_component_binding.png", dpi=130)
RESULT["seconds"] = round(time.monotonic() - T0, 1)
json.dump(RESULT, open(OUT / "sae_component_binding_results.json", "w"), indent=1, default=str)
print(f"done in {RESULT['seconds']}s; wrote {OUT}/sae_component_binding_results.json")
print("RESULT_JSON " + json.dumps({k: RESULT[k] for k in ("n_items", "summary", "agreement",
                                                          "checks", "seconds")}, default=str))
