# %% [markdown]
# # Composer neurons: nnsight nominates, mdb seals the forecasts, measures and judges
#
# **Question.** In a two-hop prompt (`... the country whose currency is the yen is ...`, answer
# the capital), a few MLP neurons at the answer row of layers 20-27 *compose* the answer: knocking
# them out lowers the margin `M = logit(answer) - logit(donor's answer)`. Can a library's
# gradients name them on some items and forecast, before any intervention, what the knock-out
# does on *other* items?
#
# **The loop shown here** (mdb as a debugger, the libraries feeding mdb objects):
#
# 1. **nnsight** (`mdb.interop.nn.grads`): one forward and one backward give `dM/df` and `f` at
#    `t<last>.L20..27.f` (the MLP's `down_proj` input, 6144 neurons per layer) on the 64 items.
# 2. **mdb `Pool`**: the knock-out value of each neuron is the *leave-one-out* mean of `f` over the
#    other 63 items (`pool.mean_for(item)`), a value with a pointer saying which items went in.
# 3. **Nomination** (plain Python): panel A items name, per country and layer, the 8 neurons with
#    the most negative forecast `dM/df . (mean - f)`, kept only if negative on every A item.
# 4. **mdb `supplied.seal`**: on every held-out (panel B) item, the nnsight forecast of each named
#    neuron and of the named set is sealed as a `supplied-forecast` record *before any arm runs*.
# 5. **nnsight `batch_whatif`**: the 64 single-neuron knock-outs, the named set and a random neuron
#    set, batched per country (the library measures).
# 6. **mdb `Branches`** (stepped, the authority): per B item, from a checkpoint at `t<last>.L20.h`,
#    the named set and its strongest neuron re-measured on mdb's own stepped kernel path, against
#    mdb's nulls (`add_null`: a matched-norm random change on the same neurons, a self-patch, and
#    an answer-direction write) and one shared floor.
# 7. **mdb `supplied.judge`**: every sealed forecast is scored against the native change with
#    mdb's margin noise floor (`verdict.margin_floor`: 2 bf16 spacings at the larger |logit| + the
#    measured batched-vs-stepped base gap). This is deliberately wider than the 1-spacing verdict
#    floor that `verify` and `whatif` use, because here we compare effects measured on different
#    realisations (batched vs stepped), not a prediction against its own native run.
#
# **Original result** (an earlier run of this experiment; single seed, bf16): the named set beats the random neuron set by >= 0.5 on 26/26 held-out items
# (median dM -4.06 named vs 0.0 random vs 0.06 for an answer-token neuron null); the forecasts rank
# the single-neuron effects well (Spearman 0.94 over 71 neurons above the floor) but overstate
# their size (median forecast/native 0.83, outside the preregistered 0.4-0.7 band); nnsight
# batched and mdb stepped agree within 2 spacings on 51/52 arms.
#
# **What "out of sample" means here:** panel B items are *held-out items of the same
# country*, so they share the answer token with the A items that named the neurons. That closes
# item-level selection, not answer-level selection. The answer-direction null below asks how much
# of a write of the same size could come from pointing it straight at the answer's unembedding.
#
# This example runs the whole original panel (64 items, 38 A, 26 B in 19 countries) in about
# 4 minutes on one GPU; `MDB_COUNTRIES=Spain,Japan` limits the measured countries (the smoke).

# %%
import collections
import json
import os
import random
import statistics as st
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL = Path(os.environ.get("MDB_MODEL", "/mnt/big/llm-models/Qwen3-1.7B-Base"))
OUT = Path(os.environ.get("MRUN_ARTIFACTS_DIR") or os.environ.get("MDB_OUT") or "out")
OUT.mkdir(parents=True, exist_ok=True)
COUNTRIES = [c for c in os.environ.get("MDB_COUNTRIES", "").split(",") if c]   # "" = all 19
DEVICE = os.environ.get("MDB_DEVICE", "cuda")
LAYERS = list(range(20, 28))
SEED = 20261114
T0 = time.monotonic()
ITEMS = json.load(open(HERE / "items.json"))["items"]        # the 64 base items, sorted by id
REF = json.load(open(HERE / "reference.json"))                # the original run's saved numbers
RESULT: dict = {"model": str(MODEL), "countries": COUNTRIES or "all"}


def log(msg):
    print(f"[{time.monotonic() - T0:6.1f}s] {msg}", flush=True)


# %% [markdown]
# ## 1. Load the model through mdb, declare custody, open a record sink
#
# `dbg.custody(script=, prereg=)` stamps every record sealed from now on with the sha256 of this
# script and of the preregistration (`prereg.json`, copied unchanged from the original run), the
# library versions and the kernel path. `dbg.records_to(path)` appends every sealed record
# (supplied forecasts, pools, branches) to a JSONL file, whole, so nothing is reduced to an id.

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mdb
from mdb.analysis import supplied, verdict
from mdb.analysis.branches import edit
from mdb.analysis.forecast import Margin
from mdb.interop import names, nn

tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL), torch_dtype=torch.bfloat16, attn_implementation="eager",
    local_files_only=True).to(DEVICE).eval()
torch.set_grad_enabled(False)
dbg = mdb.attach(model, tok)
CUSTODY = dbg.custody(script=HERE / "composer_neurons.py", prereg=HERE / "prereg.json",
                      kernel_path="mdb-stepped")
(OUT / "records.jsonl").unlink(missing_ok=True)
SINK = dbg.records_to(OUT / "records.jsonl")
VER = nn.version()
# mdb address -> the nnsight module path the library read (recorded in every sealed forecast)
AMAP = {f"t<last>.L{x}.f": names.to_nnsight(f"L{x}.f") for x in LAYERS}
log(f"loaded; nnsight {VER}; script sha256 {CUSTODY['script_sha256'][:12]}; "
    f"prereg sha256 {CUSTODY['prereg_sha256'][:12]}")

# %% [markdown]
# ## 2. nnsight gradients: `dM/df` and `f` at the answer row of L20..27
#
# `nn.grads` takes equal-length prompts (no padding), so items are grouped by length: one forward
# and one backward per group.

# %%
by_len = collections.defaultdict(list)
for it in ITEMS:
    by_len[len(it["ids"])].append(it)
G, V = {}, {}
for group in by_len.values():
    res = nn.grads(model, [i["ids"] for i in group], [f"L{x}.f" for x in LAYERS],
                   [(i["answer"], i["donor"]["answer"]) for i in group])
    for j, it in enumerate(group):
        G[it["id"]] = torch.stack([res["grads"][f"L{x}.f"][j, -1].double().cpu() for x in LAYERS])
        V[it["id"]] = torch.stack([res["values"][f"L{x}.f"][j, -1].double().cpu() for x in LAYERS])
log(f"grads: {len(by_len)} length groups, {len(G)} items")

# %% [markdown]
# ## 3. A `Pool` per layer: the knock-out value is a leave-one-out mean
#
# `dbg.pool(items, "t@last.L20.f")` runs each item on mdb's stepped path to the answer row's
# `f` and keeps the value with its origin. `pool.mean_for(item)` is the mean over every *other*
# item (the item cannot be in it), returned as a `ValueRef` whose pointer names the pool's sealed
# record, the rule and the items digest. The original run used the 64-item mean, which includes the
# item itself; leave-one-out removes it, so the edits here are 1/63 of
# `mean - f` further from `f` than the original's, and the forecasts move with them.
#
# The pool's values are mdb's *stepped* `f`; nnsight's come from a batched full-sequence forward.
# They differ by bf16 rounding on a kernel path (printed below in spacings). The forecast of an
# edit is `dM/df . (mean_loo - f_nnsight)`: exactly the change the nnsight batched arm makes.

# %%
POOLS = {x: dbg.pool(ITEMS, f"t@last.L{x}.f", labels=("bridge",), name=f"f-L{x}") for x in LAYERS}
for p in POOLS.values():
    p.seal()                                           # a `pool` record, with custody, in the sink
F = {it["id"]: torch.stack([mdb_f.value(it).reshape(-1).double() for mdb_f in POOLS.values()])
     for it in ITEMS}                                  # mdb's own f (stepped), (8, 6144)
LOO_REF = {(it["id"], x): POOLS[x].mean_for(it) for it in ITEMS for x in LAYERS}
LOO = {it["id"]: torch.stack([LOO_REF[(it["id"], x)].value.reshape(-1) for x in LAYERS])
       for it in ITEMS}
diff = torch.stack([V[k] - F[k] for k in V])
size = torch.stack([torch.maximum(V[k].abs(), F[k].abs()) for k in V])
in_sp = diff.abs() / torch.exp2(torch.floor(torch.log2(size.clamp_min(2.0 ** -126)))) * 128
ptr = LOO_REF[(ITEMS[0]["id"], 20)].pointer
RESULT["f_nnsight_vs_mdb"] = {"elements": diff.numel(), "differ": int((diff != 0).sum()),
                              "max_abs": float(diff.abs().max()),
                              "max_bf16_spacings": float(in_sp.max())}
log(f"pools: 8 x {len(POOLS[20])} items; f nnsight vs mdb stepped: {RESULT['f_nnsight_vs_mdb']}; "
    f"mean_for pointer: rule {ptr['rule']!r}, n {ptr['n']}, excluded {ptr['excluded']!r}")
FC = {k: G[k] * (LOO[k] - V[k]) for k in G}           # nnsight's forecast of each knock-out

# %% [markdown]
# ## 4. Panels and nomination (plain Python)
#
# The A/B split and the nomination rule are short plain Python, exactly as preregistered: within each country
# (sorted by id) items alternate A, B, A, ...; one-item countries are A only. Per country and
# layer, the 8 neurons with the most negative summed forecast over the country's A items, kept
# only if the forecast is negative on every A item.
#
# The nomination statistic is the preregistered one, on the library's own values: the forecast
# to the 64-item mean of nnsight's `f`. Only A items enter it, and it is the original run's
# statistic bit for bit, so the named sets are the original run's. (On held-out items the edits and forecasts
# use the pool's leave-one-out mean.) We also report how many named neurons would change if the
# nomination used the pool forecast `FC` instead: a ranking of near-tied sums moves with
# bf16-level differences in `f`.

# %%
by_c = collections.defaultdict(list)
for it in ITEMS:
    by_c[it["bridge"]].append(it)
A, B = [], []
for c, its in by_c.items():
    for j, it in enumerate(its):
        (A if j % 2 == 0 or len(its) == 1 else B).append(it)


def nominate(forecast):
    out = {}
    for c, its in by_c.items():
        a_its = [i for i in its if i in A]
        if not a_its or not any(i in B for i in its):
            continue
        S = torch.stack([forecast[i["id"]] for i in a_its])
        total = torch.where((S < 0).all(0), S.sum(0), torch.zeros_like(S[0]))
        out[c] = [(li, int(k)) for li in range(len(LAYERS))
                  for k in total[li].argsort()[:8].tolist() if total[li, k] < 0]
    return out


MEAN64 = torch.stack(list(V.values())).mean(0)        # nomination statistic only (prereg)
named = nominate({k: G[k] * (MEAN64 - V[k]) for k in G})
via_pool = nominate(FC)
same_panels = [i["id"] for i in A] == REF["A"] and [i["id"] for i in B] == REF["B"]
same_named = {c: [[LAYERS[li], k] for li, k in s] == REF["named"].get(c) for c, s in named.items()}
moved = {c: len(set(named[c]) - set(via_pool.get(c, []))) for c in named}
log(f"panels A {len(A)} B {len(B)} (same as showcase: {same_panels}); named sets in "
    f"{len(named)} countries, identical to the showcase's in {sum(same_named.values())}; "
    f"a pool-forecast nomination would swap {sum(moved.values())} of "
    f"{sum(len(s) for s in named.values())} neurons ({sum(v > 0 for v in moved.values())} countries)")
RESULT["nomination"] = {"same_panels": same_panels, "countries": len(named),
                        "identical_to_showcase": sum(same_named.values()),
                        "differ": [c for c, ok in same_named.items() if not ok],
                        "pool_forecast_would_swap": moved}
MEASURED_C = [c for c in named if not COUNTRIES or c in COUNTRIES]
B_RUN = [it for it in B if it["bridge"] in MEASURED_C]

# %% [markdown]
# ## 5. Seal the forecasts on the held-out items, before any arm runs
#
# `supplied.seal` writes a `supplied-forecast` record: the library's number (authority
# `supplied`, never `measured`), the library and version, the method, the address map
# (`names.to_nnsight`), the edit (with the pool record it takes its value from) and where the run
# stood. The sink keeps every record whole.

# %%
SEALED = {}
for it in B_RUN:
    last = len(it["ids"]) - 1
    run = dbg.run(it["ids"], name="seal")
    run.until(f"t{last}.L20.h")
    tgt = {"kind": "margin", "row": last, "indices": [it["answer"], it["donor"]["answer"]]}
    recs = {}
    for li, k in named[it["bridge"]]:
        x = LAYERS[li]
        recs[f"L{x}.n{k}"] = supplied.seal(
            run, target=tgt, value=float(FC[it["id"]][li, k]), library="nnsight", version=VER,
            edit={"address": f"t{last}.L{x}.f[i={k}]", "row": last, "layer": x,
                  "op": "replace with the leave-one-out mean", "pool": POOLS[x].record["id"]},
            address_map=AMAP, method="attribution patching: dM/df . (mean_loo - f)")
    recs["set"] = supplied.seal(
        run, target=tgt, value=float(sum(FC[it["id"]][li, k] for li, k in named[it["bridge"]])),
        library="nnsight", version=VER, address_map=AMAP,
        edit={"address": "named set", "row": last, "layer": 20,
              "op": "replace with the leave-one-out mean",
              "pools": [POOLS[x].record["id"] for x in LAYERS]},
        method="sum of the named neurons' attribution forecasts")
    SEALED[it["id"]] = recs
log(f"sealed {sum(len(r) for r in SEALED.values())} forecasts on {len(SEALED)} B items "
    f"(sink holds {SINK.count} records)")

# %% [markdown]
# ## 6. The library measures: nnsight batched knock-outs
#
# One `batch_whatif` per country over its B items: 64 single-neuron arms, the named set, and a
# random neuron set (same count per layer, named excluded, seed 20261114). Each item gets its own
# leave-one-out value (a per-prompt tensor). The random draw replays the original run's seeded stream
# exactly (its answer-token neuron null drew from the same stream; that null is replaced by
# mdb's answer-direction null in step 7, and its draw is consumed here only to keep the stream
# aligned). A random *set of other neurons* is a selection null; mdb's built-in nulls are change
# nulls at the same address, so the selection null stays plain Python.

# %%
def cols(cells, li):
    return sorted(k for l2, k in cells if l2 == li)


def nn_edits(cells, its):
    out = []
    for li in sorted({li for li, _ in cells}):
        ks = cols(cells, li)
        val = torch.stack([LOO[i["id"]][li, ks] for i in its])          # (items, neurons)
        out.append(nn.Edit(f"L{LAYERS[li]}.f[i={'|'.join(map(str, ks))}]", val))
    return out


rng = random.Random(SEED)
RANDOM_SET = {}
for c, sel in named.items():                       # every country, in order: the stream
    rnd = []
    for li, cnt in collections.Counter(li for li, _ in sel).items():
        taken = {k for l2, k in sel if l2 == li}
        rnd += [(li, k) for k in rng.sample([k for k in range(6144) if k not in taken], cnt)]
        rng.sample(range(5 * cnt), cnt)            # the original run's answer-token null draw
    RANDOM_SET[c] = rnd

NATIVE = {}
for c in MEASURED_C:
    its = [i for i in B_RUN if i["bridge"] == c]
    t0 = time.monotonic()
    arms = {f"L{LAYERS[li]}.n{k}": nn_edits([(li, k)], its) for li, k in named[c]}
    arms["set"] = nn_edits(named[c], its)
    arms["random_set"] = nn_edits(RANDOM_SET[c], its)
    res = nn.batch_whatif(model, [i["ids"] for i in its],
                          [(i["answer"], i["donor"]["answer"]) for i in its], arms)
    for j, it in enumerate(its):
        NATIVE[it["id"]] = {"base": res["base"][j],
                            "delta": {k: v["delta"][j] for k, v in res["arms"].items()}}
    log(f"nnsight batch {c}: {len(its)} items x {len(arms)} arms, {res['traces']} traces, "
        f"{time.monotonic() - t0:.1f}s")

# %% [markdown]
# ## 7. The debugger checks: `Branches` on mdb's stepped path
#
# Per B item, `dbg.branches(run, Margin(...), at="t<last>.L20.h")` takes a checkpoint at the
# branch point and measures every arm from it on one floor. Arms are written in mdb's selector
# grammar on the `f` port, `t<last>.L22.f[i=465|606|...]`, with the pool's leave-one-out value
# and its origin pointer; `Run.set` builds the mask.
#
# * `set`: the named set; `top`: its neuron with the largest nnsight batched effect;
#   `random_set`: the random neuron set of step 6.
# * `add_null("random", like="set")`: a seeded Gaussian change over the *same* neurons, rescaled
#   to the norm of the knock-out's change (is the direction of the knock-out special?).
# * `add_null("self", like="set")`: the neurons replaced by their own value; `dM` must be 0.
# * `add_null("answer", like="set-write@m")`: the answer-direction null writes along the
#   readout direction (`reads.readout`, the answer-minus-foil unembedding through the final
#   norm), so it needs a residual-stream port. `set-write@m` is the knock-out's write at each
#   layer's MLP output on the unedited run, `W_down[:, set] (mean_loo - f)`; it only sets the
#   norm (it is not measured). The null puts that norm along the answer direction: the direct
#   effect of a write of the same size aimed at the answer, a *ceiling* for direct unembedding
#   alignment rather than a matched alternative.
#
# `Branches.run(mode="batched", recheck=k)` batches *state-cell* arms (`cache.L<l>.k|v`) and, since
# mdb 0.2.0, port edits such as these `f` knock-outs too (release-gate probe job-36ee29ad9cb2: 24
# `f` arms batched, max |batched - stepped| 0.375 at about 49 floors). This validated run predates
# that and took its batched numbers from nnsight (step 6), with `Branches` stepped as the authority
# for the arms below.

# %%
W_DOWN = {x: model.model.layers[x].mlp.down_proj.weight for x in LAYERS}


def f_edits(cells, it):
    last, out = len(it["ids"]) - 1, []
    for li in sorted({li for li, _ in cells}):
        x, ks = LAYERS[li], cols(cells, li)
        sel = "i=" + "|".join(map(str, ks))
        ref = LOO_REF[(it["id"], x)]
        out.append(edit(f"t{last}.L{x}.f[{sel}]", ref.value.reshape(-1)[ks],
                        origin={**ref.pointer, "selection": sel}))
    return out


def m_writes(cells, it):
    last, out = len(it["ids"]) - 1, []
    for li in sorted({li for li, _ in cells}):
        x, ks = LAYERS[li], cols(cells, li)
        d = (LOO[it["id"]][li, ks] - F[it["id"]][li, ks]).to(W_DOWN[x].device)
        dm = (W_DOWN[x][:, ks].double() @ d).cpu()
        out.append(edit(f"t{last}.L{x}.m", dm, op="add",
                        origin={"kind": "derived", "rule": "W_down[:, set] (mean_loo - f), "
                                "the knock-out's write at m on the unedited run"}))
    return out


ROWS, STEPPED_ARMS = [], ["set", "top", "random_set", "random", "self", "answer"]
for it in B_RUN:
    t0 = time.monotonic()
    c, last = it["bridge"], len(it["ids"]) - 1
    a, b = it["answer"], it["donor"]["answer"]
    nat = NATIVE[it["id"]]
    singles = {k: v for k, v in nat["delta"].items() if k.startswith("L")}
    top = max(singles, key=lambda k: abs(singles[k]))
    top_cell = (LAYERS.index(int(top[1:].split(".")[0])), int(top.split(".n")[1]))
    run = dbg.run(it["ids"], name="item")
    br = dbg.branches(run, Margin(last, a, b), at=f"t{last}.L20.h", name=f"exp5:{it['id']}")
    br.add("set", f_edits(named[c], it))
    br.add("top", f_edits([top_cell], it))
    br.add("random_set", f_edits(RANDOM_SET[c], it))
    br.add_null("random", "random", like="set", seed=SEED)
    br.add_null("self", "self", like="set")
    br.add("set-write@m", m_writes(named[c], it))
    br.add_null("answer", "answer", like="set-write@m")
    arms = {r["name"]: r for r in br.run(STEPPED_ARMS)}
    br.seal()                                          # a `branches` record, in the sink
    lg0 = br.base_logits.reshape(-1)
    size = max(abs(float(lg0[a])), abs(float(lg0[b])))
    sp = verdict.spacing(size, torch.bfloat16)
    extra = abs(nat["base"] - br.M0)                  # batched vs stepped base gap
    floor = verdict.margin_floor(lg0, a, b, extra=extra)
    judged = {k: supplied.judge(r, nat["delta"][k], magnitude=size, dtype=torch.bfloat16,
                                extra_floor=extra, kernel_path="batched")
              for k, r in SEALED[it["id"]].items()}
    judged["set@stepped"] = supplied.judge(SEALED[it["id"]]["set"], arms["set"]["dM"]["value"],
                                           magnitude=size, dtype=torch.bfloat16,
                                           kernel_path="mdb-stepped")
    stepped = {k: arms[k]["dM"]["value"] for k in STEPPED_ARMS}
    ROWS.append({"id": it["id"], "country": c, "base_batched": nat["base"], "base_stepped": br.M0,
                 "spacing": sp, "floor": floor, "branch_floor": br.floor, "top_single": top,
                 "batched": nat["delta"], "stepped": stepped,
                 "stability": {k: arms[k].get("stability") for k in STEPPED_ARMS},
                 "effect_floors": {k: (arms[k].get("effect_floors") or {}).get("value")
                                   for k in STEPPED_ARMS},
                 "forecast": {k: r["body"]["supplied"]["value"] for k, r in SEALED[it["id"]].items()},
                 "judged": judged})
    del run, br
    log(f"{it['id']:28s} set: batched {nat['delta']['set']:+.3f} stepped {stepped['set']:+.3f} "
        f"| random_set {stepped['random_set']:+.3f} random {stepped['random']:+.3f} "
        f"self {stepped['self']:+.3f} answer {stepped['answer']:+.3f} "
        f"| {time.monotonic() - t0:.1f}s")
with open(OUT / "items.jsonl", "w") as fh:
    for r in ROWS:
        fh.write(json.dumps(r, default=float) + "\n")

# %% [markdown]
# ## 8. Score (the original run's scoring rules) and compare with the original run on the same items
#
# * **N1** named set vs random neuron set: `|dM_set| - |dM_random_set| >= 0.5` (batched, as
#   preregistered), plus the same on the stepped path and against mdb's two change nulls.
# * **N2** Spearman between forecast and native single-neuron effect over neurons above the floor;
#   median forecast/native ratio (preregistered band 0.4-0.7).
# * **N3** per-layer median ratio; **N4** `|batched - stepped| <= 2 spacings` on the set and top
#   arms.

# %%
def spearman(x, y):
    def rank(v):
        o, r, i = sorted(range(len(v)), key=lambda i: v[i]), [0.0] * len(v), 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    den = (sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry)) ** 0.5
    return num / den if den else None


def score(rows, batched, stepped):
    """The original run's scorer on rows of {batched: {...}, stepped: {...}} (functions pick the arms)."""
    n = len(rows)
    fc, nat, lay = [], [], collections.defaultdict(list)
    for r in rows:
        for k, v in r["batched"].items():
            if k.startswith("L") and abs(v) > r["floor"]:
                fc.append(r["forecast"][k])
                nat.append(v)
                lay[int(k[1:].split(".")[0])].append(r["forecast"][k] / v)
    dif = [(abs(r["batched"][k] - stepped(r, k)), r["spacing"]) for r in rows
           for k in ("set", r["top_single"])]
    ratio = [p / q for p, q in zip(fc, nat)]
    return {"n_B": n,
            "N1_set_beats_random_set_by_0.5": sum(
                abs(batched(r, "set")) - abs(batched(r, "random_set")) >= 0.5 for r in rows),
            "median_dM_set": st.median(batched(r, "set") for r in rows),
            "median_dM_random_set": st.median(batched(r, "random_set") for r in rows),
            "N2_neurons_above_floor": len(fc),
            "N2_spearman": spearman(fc, nat) if len(fc) > 2 else None,
            "N2_median_ratio": st.median(ratio) if ratio else None,
            "N3_median_ratio_by_layer": {x: st.median(v) for x, v in sorted(lay.items())},
            "N4_within_2_spacings": sum(d <= 2 * s for d, s in dif), "N4_arms": len(dif),
            "N4_max_abs": max(d for d, _ in dif) if dif else None}


NOW = score(ROWS, lambda r, k: r["batched"][k], lambda r, k: r["stepped"]["set" if k == "set"
                                                                         else "top"])
SC_ROWS = [{**REF["items"][r["id"]], "batched": {**REF["items"][r["id"]]["singles_batched"],
                                                 **REF["items"][r["id"]]["batched"],
                                                 "random_set": REF["items"][r["id"]]["batched"]["random"]},
            "forecast": REF["items"][r["id"]]["forecast_singles"],
            "floor": REF["items"][r["id"]]["floor"] + abs(REF["items"][r["id"]]["base_batched"]
                                                          - REF["items"][r["id"]]["base_stepped"])}
           for r in ROWS]
SHOWCASE = score(SC_ROWS, lambda r, k: r["batched"][k],
                 lambda r, k: r["stepped"]["set" if k == "set" else r["top_single"]])
SHOWCASE["median_dM_answer_token_null"] = st.median(r["batched"]["answer_null"] for r in SC_ROWS)
stepped_n1 = {
    "set_beats_random_set_by_0.5": sum(abs(r["stepped"]["set"]) - abs(r["stepped"]["random_set"])
                                       >= 0.5 for r in ROWS),
    "set_beats_random_change_by_0.5": sum(abs(r["stepped"]["set"]) - abs(r["stepped"]["random"])
                                          >= 0.5 for r in ROWS),
    "set_beats_answer_direction": sum(abs(r["stepped"]["set"]) > abs(r["stepped"]["answer"])
                                      for r in ROWS),
    "self_patch_zero": sum(r["stepped"]["self"] == 0.0 for r in ROWS),
    "median": {k: st.median(r["stepped"][k] for r in ROWS) for k in STEPPED_ARMS},
    "set_stable_over_4_floors": sum(r["stability"]["set"] == "stable" for r in ROWS),
    "set_over_answer_direction_median": st.median(r["stepped"]["set"] / r["stepped"]["answer"]
                                                  for r in ROWS if r["stepped"]["answer"])}
judge_set = [r["judged"]["set@stepped"] for r in ROWS]
NOW["stepped"] = stepped_n1
NOW["set_forecast_sign_agrees_stepped"] = sum(bool(j["sign_agrees"]) for j in judge_set)
NOW["set_forecast_over_stepped_median"] = st.median(j["ratio"]["value"] for j in judge_set
                                                    if j["ratio"])
RESULT.update(now=NOW, showcase_same_items=SHOWCASE, items=len(ROWS),
              records=SINK.count)

print(f"\n== showcase (job-09feeeb61cbe) vs now, same {len(ROWS)} B items")
for key, label in [("N1_set_beats_random_set_by_0.5", "N1 set beats random set by 0.5 (batched)"),
                   ("median_dM_set", "median dM named set (batched)"),
                   ("median_dM_random_set", "median dM random set (batched)"),
                   ("N2_neurons_above_floor", "N2 neurons above floor"),
                   ("N2_spearman", "N2 Spearman forecast vs native"),
                   ("N2_median_ratio", "N2 median forecast/native"),
                   ("N4_within_2_spacings", "N4 batched-stepped within 2 spacings"),
                   ("N4_max_abs", "N4 max |batched - stepped|")]:
    print(f"  {label:42s} showcase {SHOWCASE[key]!s:>22.22}   now {NOW[key]!s:>22.22}")
print("  N3 median ratio by layer   showcase", {x: round(v, 3) for x, v in
                                                 SHOWCASE["N3_median_ratio_by_layer"].items()})
print("                             now     ", {x: round(v, 3) for x, v in
                                                 NOW["N3_median_ratio_by_layer"].items()})
print(f"  showcase answer-token neuron null median {SHOWCASE['median_dM_answer_token_null']:+.3f} "
      "(not rerun; replaced by mdb's answer-direction null)")
print("\n== mdb stepped arms (Branches), now")
for k, v in stepped_n1.items():
    print(f"  {k:34s} {v if not isinstance(v, dict) else {a: round(x, 3) for a, x in v.items()}}")
print(f"  set forecast sign agrees (stepped)  {NOW['set_forecast_sign_agrees_stepped']}/{len(ROWS)}; "
      f"median set forecast / stepped dM {NOW['set_forecast_over_stepped_median']:.3f}")

print("\n== per item, showcase then now: batched dM of the named set and the random set, "
      "stepped dM of the set, the sealed set forecast, the top single neuron")
for r in ROWS:
    s = REF["items"][r["id"]]
    print(f"  {r['id']:28s} set {s['batched']['set']:+7.3f} {r['batched']['set']:+7.3f}  "
          f"random_set {s['batched']['random']:+6.3f} {r['batched']['random_set']:+6.3f}  "
          f"stepped set {s['stepped']['set']:+7.3f} {r['stepped']['set']:+7.3f}  "
          f"forecast {s['forecast_set']:+7.3f} {r['forecast']['set']:+7.3f}  "
          f"top {s['top_single']}{'=' if s['top_single'] == r['top_single'] else '!=' + r['top_single']}")

# %% [markdown]
# ## 9. Reading the result
#
# Look for: the named neuron sets identical to the original run's (same statistic, step 4);
# the named set moving the margin by several logits on every held-out item while the random
# neuron set and the matched-norm random change on the same neurons stay near 0; the self-patch
# at exactly 0 (the stepped kernel path reproduces itself); nnsight batched and mdb stepped
# within 2 spacings; the forecasts ranking the single-neuron effects well but overstating their
# size. Every number above is in the record sink (`records.jsonl`): 8 pool records, one
# supplied-forecast record per neuron and item, one branches record per item, each carrying the
# script and prereg sha256.

# %%
SINK.close()
back = SINK.records()                                  # every line re-verified on read
RESULT["record_kinds"] = dict(collections.Counter(r["kind"] for r in back))
RESULT["seconds"] = round(time.monotonic() - T0, 1)
json.dump(RESULT, open(OUT / "composer_neurons_results.json", "w"), indent=1, default=str)
print(f"records verified: {RESULT['record_kinds']}")
print(f"done in {RESULT['seconds']}s; wrote {OUT}/composer_neurons_results.json")
print("RESULT_JSON " + json.dumps(RESULT, default=str))
