# %% [markdown]
# # Watch, break, edit, continue: a wrong country feature in Qwen3-1.7B-Base
#
# **The prompt family.** Two- and three-hop recall prompts such as *"... Franz Kafka ... the
# language spoken in his country is"*: the model must recall the entity, then its country, then
# that country's language. The answer is the language token. On 27 of 110 such prompts
# (d3/d4 items of the two-hop study) Qwen3-1.7B-Base answers with the wrong language.
#
# **Where the country lives.** An earlier study found a *carrier cell*: the layer-20 value cell
# of KV group 6 at one span row (`cache.L20.v[kv6,row=R]`, the row the answer row's L20 heads 12
# and 13 attend to most, the frozen `R_argmax` rule). Transplanting that one cell from another
# prompt moves the answer. A Qwen-Scope SAE at layer 19 (its output is the residual entering
# layer 20) splits the residual that writes that cell into features; a frozen census
# (`census.json`) labels 58 of its features as *country features* (24 countries).
#
# **The debugger loop this example shows** (mdb as a debugger, the libraries feeding it):
#
# 1. **watch**: a `FeatureCondition` watch on the carrier cell: *"the most active country
#    feature names a country other than the item's"* (`best_feature(COUNTRIES) != truth`).
# 2. **break**: run forward to the answer row's layer 20; the watch fires (or not) there.
# 3. **checkpoint** at the stop, then **edit** in named **branches** from it:
#    `R1` swaps one named SAE component (wrong country feature out, true country feature in),
#    `R2` replaces the whole cell with a donor's (a same-country correct item, chosen by a
#    `Pool` rule, with its origin recorded), and two **nulls** at R1's norm.
# 4. **continue** each branch to the logits and read the answer margin against one shared floor.
#
# **What the audited result says** (`results/showcase-20261114/AUDIT-SHOWCASE.md`, full panel of
# 110 items, MEASURED, jobs job-17ce950a0f0b and job-a38bd2b83820):
#
# * The watch is noisy. It fires on 18/27 wrong items **and on 30/83 correct items** (false
#   alarms). mdb's own `content(C)` readout of the same cell names the true country on 78/83
#   correct items and only 5/27 wrong ones.
# * The one-feature edit R1 repairs 12/27 wrong items; a random-feature swap at the same norm
#   repairs 2/26; replacing the whole cell (R2) repairs 22/27. One named feature carries part of
#   the country, not all of it.
#
# This example runs **a subset of 39 items**: all 27 wrong items, 6 correct items on which the
# watch fired (false alarms) and 6 correct items on which it stayed quiet (first by id in each
# group; `items.json`). It prints every key number next to the showcase's number on the same items.
#
# Everything runs as a Python library on one GPU. `MDB_MODEL` and `MDB_SAE` default to Beast paths.

# %%
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

MODEL = os.environ.get("MDB_MODEL", "/mnt/big/llm-models/Qwen3-1.7B-Base")
SAE_DIR = os.environ.get("MDB_SAE", "/mnt/big/llm-models/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50")
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT = Path(os.environ.get("MRUN_ARTIFACTS_DIR") or os.environ.get("MDB_OUT") or "out")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = os.environ.get("MDB_DEVICE", "cuda")
SMOKE = os.environ.get("MDB_SMOKE", "") == "1"   # two items (one wrong, one false alarm)
CL, CG = 20, 6                                   # the carrier cell: layer 20, KV group 6
T0 = time.monotonic()
RESULT: dict = {"model": MODEL, "sae": SAE_DIR, "smoke": SMOKE}


def log(msg):
    print(f"[{time.monotonic() - T0:7.1f}s] {msg}", flush=True)


# %% [markdown]
# ## 1. Load the model through mdb, and declare custody
#
# `mdb.attach(model, tokenizer)` wraps a loaded Hugging Face Qwen3 (bf16, eager attention, eval
# mode); `load_local` is the same offline loader the showcase job used. `dbg.custody(...)`
# declares the script and the preregistration (`prereg.json`, the showcase's, copied here): every
# record sealed from now on carries their sha256, the library versions and the kernel path.
# `dbg.records_to(...)` appends every sealed record (pools, branches) to a JSONL file.

# %%
import torch
from transformers import AutoTokenizer

import mdb
from mdb.engine.models import load_local

torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
dbg = mdb.attach(load_local(MODEL, dtype="bf16", device=DEVICE), tok)
script = HERE / "watch_edit_continue.py"
custody = dbg.custody(script=script if script.is_file() else None, prereg=HERE / "prereg.json",
                      kernel_path="mdb-stepped")
sink = dbg.records_to(OUT / "records.jsonl")
RESULT["custody"] = {k: custody.get(k) for k in ("script_sha256", "prereg_sha256", "kernel_path")}
log(f"model loaded; custody script {str(custody['script_sha256'])[:12]} "
    f"prereg {custody['prereg_sha256'][:12]}")

# %% [markdown]
# ## 2. The items, the census and the SAE
#
# `items.json` holds the 39 items (token ids, the entity span, the true country and language
# tokens), the candidate donors, the country token set and the showcase's random-null draws.
# `census.json` is the frozen country-feature census (feature -> country token, and per country
# the truth feature with its median activation `a_truth`, a supplied constant).
# The SAE is mdb's `FeatureBasis` over the Qwen-Scope `layer19.sae.pt` (`open_set`); its site is
# `L19.h_out`, which is mdb's `L20.h`, the residual the carrier cell is a linear image of.

# %%
from mdb.analysis import sae

DATA = json.load(open(HERE / "items.json"))
CEN = json.load(open(HERE / "census.json"))
LABEL = {int(f): int(c) for f, c in CEN["features"].items()}          # feature -> country token
TRUTHF = {int(c): (int(v["feature"]), float(v["a_median"])) for c, v in CEN["a_truth"].items()}
COUNTRY_FEATS = sorted(LABEL)
ITEMS = DATA["items"]
if SMOKE:
    ITEMS = [i for i in ITEMS if i["id"] in ("d3-per:Franz Kafka", "d3-lm:Anitkabir")]
B19 = sae.open_set("qwen-scope", SAE_DIR, layers=[19], cache=2).get(19)
print(len(ITEMS), "items:", sum(not i["correct"] for i in ITEMS), "wrong,",
      sum(i["correct"] for i in ITEMS), "correct;", len(COUNTRY_FEATS), "country features;",
      "SAE site", B19.site, "k", B19.k)

# %% [markdown]
# ## 3. The donor pool (P6 `Pool`)
#
# R2 needs a donor cell: the carrier cell of a *correct* item of the same country. A `Pool`
# collects that cell once per candidate (`dbg.pool(items, "cache.L20.v[kv6,row=@carrier]")`),
# and `pool.donor(item, rule="same:country", order=...)` hands out the first eligible donor as a
# `ValueRef` whose pointer names the pool, the rule, the donor id and the donor run's origin, so
# the edit that uses it says where its value came from. The item itself is excluded by
# construction.
#
# Two details keep the numbers comparable with the showcase:
#
# * `prefill=lambda item: item["span"][0]` collects each cell on the showcase's kernel path (the
#   span rows stepped, not read from the batched prefill). This option was added to
#   `pools.collect` for this example; the pool's kernel path is then `mdb-stepped`.
# * The order is the showcase's frozen code rule, reproduced exactly:
#   `order = {own family: 0, "d3-lm": 1, "d3-per": 1, "d4-work": 1, "d2": 2, "d1": 3}`, sorted
#   by `(order[family], id)`. In a Python dict literal the later `"d3-lm": 1` overrides the own
#   family's 0, so for these d3/d4 items "same family first" (the prereg's wording) never
#   applied: the rule that ran is "any d3/d4 first, then d2, then d1, by id".
#
# The `@carrier` row of each candidate is the frozen `R_argmax` row (the showcase's live value
# for the 79 candidates it scored, the B1 job's for the 38 others; `carrier_from` in `items.json`).

# %%
countries = {i["country"] for i in ITEMS}
POOL_ITEMS = [dict(p, labels={"country": p["country"]}) for p in DATA["pool_items"]
              if p["country"] in countries]
pool = dbg.pool(POOL_ITEMS, f"cache.L{CL}.v[kv{CG},row=@carrier]", labels=("country",),
                name="carrier-v", prefill=lambda item: item["span"][0])
pool_record = pool.seal()
FAMILY = {p["id"]: p["family"] for p in POOL_ITEMS}
RESULT["pool"] = {"n": len(pool), "kernel_path": pool.kernel_path, "record": pool_record["id"]}
log(f"pool: {len(pool)} cells, kernel path {pool.kernel_path}, record {pool_record['id'][:16]}")


def donor_order(item):
    """The showcase's frozen donor order (dict-literal semantics kept, see above)."""
    order = {item["family"]: 0, "d3-lm": 1, "d3-per": 1, "d4-work": 1, "d2": 2, "d1": 3}
    return sorted(pool.ids, key=lambda k: (order.get(FAMILY[k], 4), k))


# %% [markdown]
# ## 4. One item, step by step
#
# For each item:
#
# 1. **Native run** (`dbg.run(ids, prefill=span[0])`) to the answer row's logits: the native
#    answer, the foil (the top non-truth token) and the carrier row. The carrier rule
#    (`circuit_map.slot_rows`, `R_argmax`) reads the answer row's L20 attention, which comes after
#    the cell, so the run is restored to the answer row's start before the watch is armed.
# 2. **Watch**: `breakpoints.Table.add_watch(f"t{last}.L20.h", ..., mode="level",
#    compiled=FeatureCondition(...))` with a `FeatureContext` over the carrier cell, plus a plain
#    breakpoint at the same stop; `run_forward` runs to it. The predicate reads
#    `best_feature(COUNTRIES)` (label of the most active country feature) and the constant
#    `truth`. At the stop we also read `content(C)`: the country token the cell's current value
#    writes most through `W_O` and the unembedding (rho-free ranking over the country tokens).
# 3. **Component binding** (`components.component_binding`): the cell split over the SAE into
#    one component per active feature, the decoder bias and the SAE error, in float64.
#    R1 = cell + image(a_truth * d_truth) - component(most active wrong country feature).
# 4. **Branches** from the stop (`dbg.branches(run, Margin(last, language, foil))`): the branch
#    point is a checkpoint; each arm restores it, stages its edit, continues to the logits,
#    reads the margin and restores. All arms share one floor, 2 bf16 spacings at the larger
#    |logit| of the base margin (`verdict.margin_floor`), and each row reports `|dM|` in floors.
#    Arms: `R1`; `R2` (the pool donor's whole cell); `RND` (the showcase's random-feature null:
#    one random active non-country feature's component out, an inactive non-country feature's
#    image in at the same activation, rescaled to R1's change norm; the showcase's draws are
#    replayed from `items.json`); `rnd` (`add_null("rnd", "random", like="R1")`, the Branches
#    built-in matched-norm Gaussian change at the same cell).
# 5. `br.seal()` writes a `branches` record (every arm's edit manifest with its origin, the floor,
#    custody) to the records file.
#
# **Watch over layers.** The gap audit asked for an edge-mode watch over `t@carrier.L*.h`, so
# that "fired" would mean "first became true at layer l". The pattern machinery accepts `L*` and
# `mode="edge"`, but a `FeatureContext` is one basis at one (row, layer) site: evaluated at every
# layer stop it would read the same L20 cell each time, a constant predicate. A layer-indexed
# context over a per-layer SAE set (P9, feature strips) is not built, and the census labels
# country features at L19 only. So the watch stays the showcase's level watch at `t<last>.L20.h`.

# %%
from mdb.analysis import circuit_map as cm
from mdb.analysis.branches import edit
from mdb.analysis.components import component_binding
from mdb.analysis.features import FeatureCondition, FeatureContext
from mdb.analysis.forecast import Margin, Refusal
from mdb.engine import breakpoints as bp

CARRIER = cm.Slot("carrier", "carrier", "R_argmax", layers=(CL,), kv=(CG,), rule_layer=CL,
                  rule_heads=(12, 13))
WATCH = "best_feature(COUNTRIES) >= 0 and best_feature(COUNTRIES) != truth"
COUNTRY_TOKENS = DATA["country_tokens"]


def debug_item(item):
    t0 = time.monotonic()
    ids, span, last = item["ids"], item["span"], len(item["ids"]) - 1
    lang, country = item["language"], item["country"]
    # 1. native run: carrier row, native answer, foil
    run = dbg.run(ids, name="nat", prefill=span[0])
    run.until(f"t{last}.start")
    start = run.checkpoint("answer-row-start")
    run.until(f"t{last}.logits")
    crow = cm.slot_rows(CARRIER, run, {"name": item["id"], "span": span, "last": last})[0]
    nat = run.read(f"t{last}.logits").reshape(-1).float()
    top2 = nat.topk(2).indices.tolist()
    foil = top2[0] if top2[0] != lang else top2[1]
    rec = {"id": item["id"], "role": item["role"], "correct": item["correct"], "carrier_row": crow,
           "language": lang, "foil": foil, "native_logits": [float(nat[lang]), float(nat[foil])],
           "native_argmax": int(nat.argmax())}
    run.restore(start)
    # 2. watch + break at the answer row's layer 20, then run forward
    ctx = FeatureContext(B19, crow, CL, CG, sets={"COUNTRIES": COUNTRY_FEATS}, labels=LABEL,
                         candidates={"C": COUNTRY_TOKENS}, constants={"truth": country})
    table = bp.Table()
    watch = table.add_watch(f"t{last}.L{CL}.h", run.program, mode="level",
                            compiled=FeatureCondition(WATCH, ctx))
    table.add_break(f"t{last}.L{CL}.h", run.program)
    stop = bp.run_forward(run, table, budget=4)
    rec["stop"] = {"reason": stop["reason"], "where": stop["where"]}
    rec["watch_fired"] = any(h["id"] == watch.id for h in stop["hits"])
    rec["best_feature_label"] = ctx.best_feature(run, "COUNTRIES")
    rec["content_C"] = ctx.content(run, "C")
    # 3. the cell split into named components; the R1 change
    cb = component_binding(run, "carrier", f"cache.L{CL}.v[kv{CG},row={crow}]", B19)
    acts = cb.acts
    wrong = [(a, f) for f, a in acts.items() if f in LABEL and LABEL[f] != country and a > 0]
    fw = max(wrong)[1] if wrong else None
    tf = TRUTHF.get(country)
    delta = torch.zeros_like(cb.native)
    if tf is not None:
        delta = delta + cb.feature_value(tf[0], tf[1], B19)
    if fw is not None:
        # the measured component of the wrong feature: its image a_fw * d_fw at the native rho
        # (the showcase's expression; cb.get(f"feature[{fw}]").value is the same to f64 roundoff)
        delta = delta - cb.feature_value(fw, acts[fw], B19)
    rec["r1"] = {"wrong_feature": fw, "truth_feature": list(tf) if tf else None,
                 "delta_norm": float(delta.norm()), "cell_norm": float(cb.native.norm()),
                 "component_vs_image_max_abs": None if fw is None else float(
                     (cb.get(f"feature[{fw}]").value - cb.feature_value(fw, acts[fw], B19))
                     .abs().max())}
    # 4. branches from the stop
    cell = f"cache.L{CL}.v[kv{CG},row={crow}]"
    dtype = dbg.adapter.dtype
    as_cell = lambda v: v.to(dtype).reshape(1, 1, 1, -1)    # noqa: E731 (one cast, f64 -> bf16)
    br = dbg.branches(run, Margin(last, lang, foil), name=f"exp1:{item['id']}")
    if float(delta.norm()) > 0:
        br.add("R1", [edit(cell, as_cell(cb.native + delta))])
    try:
        donor = pool.donor(item, rule="same:country", order=donor_order(item))
        rec["donor"] = donor.pointer["donor"]
        br.add("R2", [edit(cell, donor)])
    except Refusal as refusal:          # no correct item of this country in the pool
        rec["donor"] = None
        print("no donor:", refusal)
    draw = DATA["rnd_draws"].get(item["id"])
    if "R1" in br.arms and draw is not None:
        fr, fa = draw
        dn = -cb.feature_value(fr, acts[fr], B19) + cb.feature_value(fa, acts[fr], B19)
        dn = dn * (delta.norm() / (dn.norm() + 1e-30))
        br.add("RND", [edit(cell, as_cell(cb.native + dn))])
        br.add_null("rnd", "random", like="R1", seed=20261114)
    rows = {r["name"]: r for r in br.run()}
    rec["floor"] = br.floor
    rec["M0"] = br.M0
    rec["arms"] = {name: {"argmax": r["argmax"], "dM": r["dM"]["value"],
                          "floors": (r.get("effect_floors") or {}).get("value"),
                          "stability": r.get("stability"),
                          "repaired": r["argmax"] == lang} for name, r in rows.items()}
    if "R2" in rows:
        r2_origin = rows["R2"]["cells"][0]["origin"]
        rec["r2_origin"] = {k: r2_origin.get(k) for k in ("kind", "pool", "rule", "donor", "record")}
    rec["record"] = br.seal()["id"]
    rec["seconds"] = round(time.monotonic() - t0, 2)
    del br, run
    return rec


# %% [markdown]
# ## 5. Run the loop over the items
#
# P7 (a prompt-set `each` in Python) is not built, so the item loop is plain Python. Each line
# shows, per arm, whether the argmax after the edit is the true language (`lang`) or not
# (`other`), and `|dM|` in floors.

# %%
RECS = []
for item in ITEMS:
    rec = debug_item(item)
    RECS.append(rec)
    arms = " ".join(f"{k}:{'lang' if v['repaired'] else 'other'}({v['floors']:.1f}fl)"
                    for k, v in rec["arms"].items() if v["floors"] is not None)
    log(f"{rec['id'][:42]:42} {'wrong' if not rec['correct'] else 'right'} "
        f"watch={'FIRE' if rec['watch_fired'] else 'quiet'} content={'true' if rec['content_C'] == item['country'] else 'other'} "
        f"donor={str(rec['donor'])[:24]} {arms} [{rec['seconds']}s]")
RESULT["items"] = RECS
print("one R2 edit's origin (from the pool):", json.dumps(RECS[0].get("r2_origin")))

# %% [markdown]
# ## 6. The key numbers, and the showcase on the same items
#
# * **Watch**: fired on wrong items (hits) and on correct items (false alarms).
# * **content(C)**: names the true country, on correct and on wrong items.
# * **Repairs** (argmax becomes the true language) on wrong items: R1 (one named component),
#   R2 (whole cell), RND (random feature swap at R1's norm), rnd (Gaussian at R1's norm, new here).
# * R1's margin move as a fraction of R2's (median over wrong items).
#
# `reference.json` holds the showcase's per-item outputs for these 39 items, cut from
# `results/showcase-20261114/exp1/out/exp1-items.jsonl`. The edits, the floor rule and the kernel
# path are the showcase's, so the per-item margins should agree exactly; any difference is listed.

# %%
REF = json.load(open(HERE / "reference.json"))
W = [r for r in RECS if not r["correct"]]
C = [r for r in RECS if r["correct"]]
truth = {i["id"]: i["country"] for i in ITEMS}


def frac(rs, test):
    return f"{sum(1 for r in rs if test(r))}/{len(rs)}"


def repaired(r, arm):
    return arm in r["arms"] and r["arms"][arm]["repaired"]


ratios = [r["arms"]["R1"]["dM"] / r["arms"]["R2"]["dM"] for r in W
          if "R1" in r["arms"] and r["arms"].get("R2", {}).get("dM")]
summary = {
    "n": len(RECS), "wrong": len(W), "correct": len(C),
    "fired_wrong": frac(W, lambda r: r["watch_fired"]),
    "fired_correct": frac(C, lambda r: r["watch_fired"]),
    "content_true_wrong": frac(W, lambda r: r["content_C"] == truth[r["id"]]),
    "content_true_correct": frac(C, lambda r: r["content_C"] == truth[r["id"]]),
    "R1_repairs_wrong": frac(W, lambda r: repaired(r, "R1")),
    "R1_repairs_where_fired": frac([r for r in W if r["watch_fired"]], lambda r: repaired(r, "R1")),
    "R2_repairs_wrong": frac(W, lambda r: repaired(r, "R2")),
    "RND_repairs_wrong": frac([r for r in W if "RND" in r["arms"]], lambda r: repaired(r, "RND")),
    "rnd_gauss_repairs_wrong": frac([r for r in W if "rnd" in r["arms"]], lambda r: repaired(r, "rnd")),
    "R1_on_correct_argmax_changes": frac([r for r in C if "R1" in r["arms"]],
                                         lambda r: r["arms"]["R1"]["argmax"] != r["native_argmax"]),
    "median_R1_over_R2": st.median(ratios) if ratios else None, "n_ratio": len(ratios)}
RESULT["summary"] = summary
ref_sum = REF["summary"]
print(f"{'key number':34} {'showcase (same items)':>22} {'now':>10}")
for key in ("fired_wrong", "fired_correct", "content_true_wrong", "content_true_correct",
            "R1_repairs_wrong", "R1_repairs_where_fired", "R2_repairs_wrong", "RND_repairs_wrong",
            "R1_on_correct_argmax_changes", "median_R1_over_R2", "rnd_gauss_repairs_wrong"):
    old = ref_sum.get(key, "(new)")
    now = summary[key]
    fmt = (lambda x: f"{x:.3f}") if isinstance(now, float) else str
    print(f"{key:34} {fmt(old) if old != '(new)' else old:>22} {fmt(now):>10}")
print("full 110-item panel (audited):", json.dumps(REF["full_panel"]))

# per-item agreement with the showcase
diffs = []
for r in RECS:
    ref = REF["per_item"][r["id"]]
    pairs = [("carrier_row", r["carrier_row"], ref["carrier_row"]),
             ("foil", r["foil"], ref["foil"]),
             ("native_logits", r["native_logits"], ref["native_logits"]),
             ("floor", r["floor"], ref["floor"]),
             ("watch_fired", r["watch_fired"], ref["watch_fired"]),
             ("best_feature_label", r["best_feature_label"], ref["best_feature_label"]),
             ("content_C", r["content_C"], ref["content_C"]),
             ("wrong_feature", r["r1"]["wrong_feature"], ref["wrong_feature"]),
             ("donor", r["donor"], ref["donor"])]
    pairs.append(("R1 delta norm", round(r["r1"]["delta_norm"], 6), round(ref["delta_norm"], 6)))
    for arm in ("R1", "R2", "RND"):
        if arm in ref["arms"] or arm in r["arms"]:
            got = r["arms"].get(arm, {})
            want = ref["arms"].get(arm, {})
            pairs.append((f"{arm} dM", got.get("dM"), want.get("dM")))
            pairs.append((f"{arm} argmax", got.get("argmax"), want.get("argmax")))
    for name, now, old in pairs:
        same = (abs(now - old) <= 1e-9 if isinstance(now, float) and isinstance(old, float)
                else now == old)
        if not same:
            diffs.append({"id": r["id"], "what": name, "now": now, "showcase": old})
RESULT["per_item_differences"] = diffs
n_checked = len(RECS)
print(f"per-item agreement with the showcase: {n_checked - len({d['id'] for d in diffs})}/{n_checked} "
      f"items identical on every compared field")
for d in diffs[:40]:
    print("DIFF", json.dumps(d))

# %% [markdown]
# **Reading.** The watch is a debugger trigger, not a diagnosis: on these items it fires on
# wrong and on correct items alike, while `content(C)` of the same cell separates them much
# better. Swapping the one named country component repairs some wrong items and does far better
# than a random swap at the same norm, but replacing the whole cell does much better still: the
# country is carried by more of the cell than the one dictionary feature. The showcase also
# sealed a margin forecast after the R1 edit and verified it at the logits (26/26); that is a
# check of mdb's float64 replay of layers 21-27 on an edited state, not evidence that R1
# repairs anything, so this example does not repeat it.

# %%
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("matplotlib not installed: skipping the plot")
if plt is not None and W:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].bar(["wrong items", "correct items"],
              [sum(r["watch_fired"] for r in W) / len(W), sum(r["watch_fired"] for r in C) / max(1, len(C))],
              color=["#b5651d", "#7a7a7a"])
    ax[0].set_ylim(0, 1)
    ax[0].set_title("Watch fired (fraction)")
    arms = ["R1", "R2", "RND", "rnd"]
    ax[1].bar(arms, [sum(repaired(r, a) for r in W) for a in arms], color="#3b6ea5")
    ax[1].set_title(f"Wrong items repaired (of {len(W)})")
    fig.tight_layout()
    fig.savefig(OUT / "watch_edit_continue.png", dpi=130)
    try:
        from IPython.display import display
        display(fig)
    except ImportError:
        pass

RESULT["records"] = {"path": str(OUT / "records.jsonl"), "n": len(sink.records())}
RESULT["seconds"] = round(time.monotonic() - T0, 1)
RESULT["peak_vram_mb"] = (round(torch.cuda.max_memory_allocated() / 2**20)
                          if torch.cuda.is_available() else None)
json.dump(RESULT, open(OUT / "watch_edit_continue_results.json", "w"), indent=1, default=str)
log(f"done; {RESULT['records']['n']} sealed records in {RESULT['records']['path']}")
print("RESULT_JSON " + json.dumps({k: v for k, v in RESULT.items() if k != "items"}, default=str))
