# %% [markdown]
# # SAE variable census on Qwen3-1.7B-Base with Qwen-Scope SAEs
#
# **Question.** A sparse autoencoder (SAE) rewrites a residual vector `x` as
# `x = b_dec + sum_f a_f d_f + e`: a bias, a few active feature directions `d_f` with
# activations `a_f`, and an **error term** `e` (what the dictionary does not capture). The
# identity is exact. If the model *uses* a variable, is that variable carried by the features, or
# does it live in the error term?
#
# **How mdb answers it.** mdb is a debugger for language models. It runs the real model, lets
# you edit any intermediate value (a *port*, such as the residual stream `L20.h_out`, or a cache
# cell such as `cache.L20.v[row=3]`), and measures the native change in an output number (the
# *consumer*, here a logit margin). So we can define a variable by an intervention, and then
# split the effect of that intervention into a features part and an error part.
#
# **The experiment** (one seed, bf16, eager attention, k=50 Qwen-Scope SAEs):
#
# 1. Prompts `"The city of {X} is the capital of"` for two cities X and Y; the entity token
#    (row 3) is the *variable*. Consumer: `logit(country_X) - logit(country_Y)`.
# 2. Transplant Y's value cell into X's run and measure the margin change (the variable's effect).
# 3. Replace the residual at the same site by Y's reconstruction + Y's error, by Y's
#    reconstruction alone (the features part Phi), and by Y's error alone (the error part E).
# 4. Then, on one prompt: splice out the SAE error at a layer, ablate the strongest feature, and
#    replace every layer by its reconstruction.
#
# Published result (see `expected_outputs.md`): the variable is in the features (median feature
# share 0.970, error share 0.046 over 18 pairs), but the SAE error still carries a lot of
# *computation* (removing it at layer 14 moves the answer by a median 1.25 logits).
#
# Everything runs as a Python library. `MODEL` and `SAE` below default to Beast paths; change
# them for a local GPU.

# %%
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

MODEL = Path(os.environ.get("MDB_MODEL", "/mnt/big/llm-models/Qwen3-1.7B-Base"))
SAE = os.environ.get("MDB_SAE", "/mnt/big/llm-models/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50")
NPAIRS = int(os.environ.get("MDB_NPAIRS", "6"))        # consecutive city pairs (the study used 18)
READER_LAYERS = [16, 18, 19, 20, 21, 22]               # value-cell transplant layers tried
SPLICE_LAYERS = [8, 14, 18, 22]                        # error-splice layers
OUT = Path(os.environ.get("MRUN_ARTIFACTS_DIR") or os.environ.get("MDB_OUT") or "out")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = os.environ.get("MDB_DEVICE", "cuda")
T0 = time.monotonic()
RESULT: dict = {"model": str(MODEL), "sae": SAE, "npairs": NPAIRS}

# %% [markdown]
# ## 1. Load the model through mdb (as a library, not the TUI)
#
# `mdb.attach(model, tokenizer)` wraps a loaded Hugging Face Qwen2/Qwen3 model (eager attention,
# eval mode, bf16). `dbg.ops(None)` gives a *context*; `dispatch(ctx, verb, args)` runs the same
# verbs the REPL has (`start`, `until`, `set`, `whatif`, `features`, ...), returning JSON-like
# dicts whose numbers carry an authority label (`measured`, `derived-exact`, ...).

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mdb
from mdb.ops import OpError, dispatch

tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL), torch_dtype=torch.bfloat16, attn_implementation="eager",
    local_files_only=True).to(DEVICE).eval()
dbg = mdb.attach(model, tok)
ctx = dbg.ops(None)
print(f"model loaded in {time.monotonic() - T0:.0f}s on {DEVICE}; "
      f"{model.config.num_hidden_layers} layers, d_model {model.config.hidden_size}")


def op(name, args):
    """One mdb verb. A refusal is returned as {'error': ...} so the study carries on."""
    try:
        return dispatch(ctx, name, args)
    except OpError as error:
        return {"op": name, "error": str(error)}


def one(word):
    ids = tok.encode(word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def margin(a, b):
    """Native logit(a) - logit(b) at the cursor's row (read from the real forward)."""
    row = int(ctx.run.where.split(".")[0][1:])
    logits = ctx.run.read(f"t{row}.logits").reshape(-1).double()
    return float(logits[a] - logits[b]), int(logits.argmax())


# %% [markdown]
# ## 2. Load the Qwen-Scope SAE through SAELens
#
# The Qwen-Scope suite is a directory of `layer{L}.sae.pt` files (`W_enc, b_enc, W_dec, b_dec`)
# with a `config.json` (`k = 50`, hook point `resid_post`, i.e. mdb's `L<l>.h_out`).
#
# * `open_set` gives mdb's own `FeatureBasis` (float64 identity `x = b_dec + sum a d + e`);
#   mdb's `sae load` verb registers the same directory for `features`, `whatif` and `set`.
# * `FeatureBasis.to_saelens()` builds the identical SAELens `TopKSAE` the way SAELens's own
#   `qwen_scope` loader does, with a **strict** state-dict load (a missing or misnamed key is an
#   error, never silent random weights), and the `hook_name` metadata (`blocks.L.hook_resid_post`).
#   `describe()["saelens"]` also gives the registry release and id
#   (`qwen-scope-3-1.7b-base-w32k-l50`, `layer20`) for `SAE.from_pretrained`.
#
# **Caveat for Qwen-Scope.** Its training filtered out extreme-norm activations, mostly the
# first token (the attention sink). Row 0 is therefore off the SAE's training distribution and
# its error is large *by construction*; the census reads rows 3 and 7, which are unaffected. We
# check that below.

# %%
from mdb.analysis.sae import open_set

res = op("sae", {"action": "load", "name": "qs", "path": SAE, "cache": 6})
print("mdb sae load:", res.get("error") or "ok")
sset = open_set("qs", SAE)
basis = sset.get(20)
print("site", basis.site, "k", basis.k, "d_sae", basis.d_sae, "revision", sset.revision)
print("SAELens ref:", basis.describe()["saelens"])

try:
    sl = basis.to_saelens(device=DEVICE)                  # strict load
    probe = tok("The city of Paris is the capital of", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        hs = model(**probe, output_hidden_states=True).hidden_states[21]   # = L20.h_out
    check = {}
    for row in (3, 0):
        h = hs[0, row].float()
        with torch.no_grad():
            acts = sl.encode(h)
            rec = sl.decode(acts)
        sl_ids = sorted(int(i) for i in acts.nonzero().reshape(-1).tolist())
        mdb_ids = sorted(basis.select(h.cpu())[0])
        check[f"row{row}"] = {
            "same_active_set": sl_ids == mdb_ids, "n_active": len(sl_ids),
            "error_frac_of_norm": float((h - rec).norm() / h.norm())}
    RESULT["saelens_check"] = check
    print(json.dumps(check, indent=1))
    print("row 0 error fraction exceeds row 3 (the t0 caveat):",
          check["row0"]["error_frac_of_norm"] > check["row3"]["error_frac_of_norm"])
    del sl
except ImportError as error:
    RESULT["saelens_check"] = f"sae_lens not installed: {error}"
    print(RESULT["saelens_check"])

# %% [markdown]
# ## 3. Declare the variable binding
#
# The variable is "the entity token of the prompt", defined by what it does: the value cell
# (`cache.L{l}.v[row=3]`) of the entity row, transplanted from another city's run. We make it a
# named mdb **symbol**: run the transplant as a `whatif` (sealed prediction, then native
# measurement), then `symbol entity = whatif` records the writer, its readers and the measured
# effect. A transplant with no measurable effect would be refused, so a named symbol always has
# an effect behind it.

# %%
TABLE = [("Paris", "France"), ("Rome", "Italy"), ("Berlin", "Germany"), ("Moscow", "Russia"),
         ("Athens", "Greece"), ("Beijing", "China"), ("Madrid", "Spain"), ("London", "England"),
         ("Prague", "Czech"), ("Vienna", "Austria"), ("Amsterdam", "Netherlands"),
         ("Lisbon", "Portugal"), ("Budapest", "Hungary"), ("Tokyo", "Japan"), ("Cairo", "Egypt"),
         ("Seoul", "Korea"), ("Copenhagen", "Denmark"), ("Stockholm", "Sweden"),
         ("Brussels", "Belgium")]
TEMPLATE = "The city of {} is the capital of"
entries = []
for city, country in TABLE:
    ids = tok.encode(TEMPLATE.format(city), add_special_tokens=False)
    if one(" " + city) is None or one(" " + country) is None or len(ids) != 8:
        continue                                          # keep prompts of 8 tokens, entity at row 3
    entries.append((city, country, ids, one(" " + country)))
pairs = [(entries[i], entries[(i + 1) % len(entries)]) for i in range(len(entries))][:NPAIRS]
print(len(entries), "entries;", len(pairs), "pairs:", [f"{a[0]}->{b[0]}" for a, b in pairs])

BINDING = {"variable": "entity", "row": 3, "writer": "cache.L{L}.v[row=3]",
           "origin": "cf (the other city's run)", "consumer": "M[answer_X, answer_Y]"}
(xa, ya, ida, ka), (xb, yb, idb, kb) = pairs[0]
op("start", {"prompt": idb, "name": "cf"})
op("until", {"target": "logits"})
op("start", {"prompt": ida, "name": "main"})
op("until", {"target": "logits"})
cell = "cache.L20.v[row=3]"
w = op("whatif", {"target": cell, "value": f"cf:{cell}", "op": "replace",
                  "consumer": f"M[{ka},{kb}]"})
if "error" not in w:
    print(f"whatif {cell} <- cf:  native {w['measured']['delta']['value']}  "
          f"state {w.get('state')}")
sym = op("symbol", {"name": "entity", "result": "whatif"})
RESULT["symbol"] = sym.get("error") or "declared: entity"
print("symbol:", RESULT["symbol"])

# %% [markdown]
# ## 4. The census: features versus error
#
# For each pair (X, Y), with consumer `M = logit(country_X) - logit(country_Y)`:
#
# 1. `cell`: transplant Y's value cell at the entity row into X's run, at each layer in
#    `READER_LAYERS`; the layer with the largest drop is `L*`. This is the variable's effect.
# 2. Run X with that transplant and read the SAE parts of the resulting residual at
#    `L*.h_out` (= `L{L*+1}.h`): `recon@T` (features) and `error@T`.
# 3. Patch X's own residual at that site natively, three ways, each as a fresh run:
#    `Delta` = replace by `recon@T + error@T` (the whole residual), `Phi` = add
#    `recon@T - recon` (features only), `E` = add `error@T - error` (error only).
#
# Feature share `sigma_Phi = Phi / Delta` and error share `sigma_E = E / Delta`. Consistency
# check: the whole-residual patch must reproduce the cell transplant (`|Delta - cell| <= 0.5`).

# %%


def native(ids_x, ka, kb, base, steps):
    """A fresh main run with edits staged at their stops, run to logits; margin change vs base."""
    op("start", {"prompt": ids_x, "name": "main"})
    for stop, target, value, kind in steps:
        if stop is not None:
            op("until", {"target": stop})
        args = {"target": target, "value": value}
        if kind:
            args["op"] = kind
        if "error" in op("set", args):
            return None
    op("until", {"target": "logits"})
    return margin(ka, kb)[0] - base


ROWS = []
for (xa, ya, ida, ka), (xb, yb, idb, kb) in pairs:
    t_pair = time.monotonic()
    C = f"M[{ka},{kb}]"
    row = {"pair": f"{xa}->{xb}"}
    op("start", {"prompt": idb, "name": "cf"})
    op("until", {"target": "logits"})
    row["cf_margin"], _ = margin(ka, kb)
    op("start", {"prompt": ida, "name": "main"})
    op("until", {"target": "logits"})
    base, argmax = margin(ka, kb)
    row["base"], row["argmax_text"] = base, tok.decode([argmax])
    # the variable's effect: value-cell transplant per layer, the best layer L*
    row["cell"] = {}
    for layer in READER_LAYERS:
        cell = f"cache.L{layer}.v[row=3]"
        row["cell"][layer] = native(ida, ka, kb, base, [(None, cell, f"cf:{cell}", None)])
    good = {k: v for k, v in row["cell"].items() if v is not None}
    best = min(good, key=good.get)
    row["best_layer"] = best
    # the transplanted run T, then three native patches at L{best+1}.h
    op("start", {"prompt": ida, "name": "T"})
    op("set", {"target": f"cache.L{best}.v[row=3]", "value": f"cf:cache.L{best}.v[row=3]"})
    op("until", {"target": "logits"})
    site = f"L{best + 1}.h"
    row["reader"] = {}
    for key, value, kind in (("delta", "recon@T+error@T", "replace"),
                             ("phi", "recon@T-recon", "add"),
                             ("E", "error@T-error", "add")):
        row["reader"][key] = native(ida, ka, kb, base, [(site, site, value, kind)])
    # the direct (skip-path) effect of the SAE error per splice layer, then its native splice
    op("start", {"prompt": ida, "name": "main"})
    op("until", {"target": "logits"})
    row["splice"] = {}
    for layer in SPLICE_LAYERS:
        feat = op("features", {"address": f"L{layer}.h_out", "consumer": C, "top": 1})
        direct = feat["accounting"]["error"]["value"]
        d = native(ida, ka, kb, base, [(f"L{layer + 1}.h", f"L{layer + 1}.h", "error*-1", "add")])
        row["splice"][layer] = {"native": d, "error_direct": direct}
    row["seconds"] = round(time.monotonic() - t_pair, 1)
    ROWS.append(row)
    rd = row["reader"]
    print(f"{row['pair']:22} base {base:6.2f} L*={best} cell {row['cell'][best]:8.3f} "
          f"Delta {rd['delta']:8.3f} Phi {rd['phi']:8.3f} E {rd['E']:7.3f}  [{row['seconds']}s]",
          flush=True)
RESULT["rows"] = ROWS

# %% [markdown]
# ## 5. Splice and steering on one prompt: the error term is causal
#
# On the recall prompt `"The Eiffel Tower is located in the city of"` (consumer
# `M[" Paris", " Rome"]`, base margin 9.5), `features ... consumer=M[a,b]` splits the margin into
# the *direct* effect of each feature, of the bias and of the error (the skip path to the
# readout), with exact closure. Then, with sealed predictions and a native measurement each:
#
# * **error splice** at a layer: `whatif L14.h_out = error*-1` replaces the residual by its
#   reconstruction. Amplification = native change / (minus the error's direct share).
# * **steering**: remove the feature with the largest direct effect (`feature[ID,coef]*-1`).
#   Compare the native change with the sealed local-linear estimate: they disagree
#   (nonlinear, compensated).
# * **replacement model**: replace *every* layer's residual by its reconstruction, one pass.

# %%
Pa, Ro = one(" Paris"), one(" Rome")
A = f"M[{Pa},{Ro}]"
P1 = "The Eiffel Tower is located in the city of"
op("start", {"prompt": P1, "name": "main"})
op("until", {"target": "logits"})
base_recall, _ = margin(Pa, Ro)
print("recall margin", base_recall)
detail = {"base": base_recall, "splice": {}}

f22 = op("features", {"address": "L22.h_out", "consumer": A, "top": 10})
top = max(f22["features"], key=lambda f: f["effect"]["value"])
detail["top_feature"] = {"id": top["feature"], "direct": top["effect"]["value"]}
print("largest direct feature at L22:", detail["top_feature"])

for layer in (14, 22):
    w = op("whatif", {"target": f"L{layer}.h_out", "value": "error*-1", "op": "add",
                      "consumer": A})
    detail["splice"][layer] = None if "error" in w else w["measured"]["delta"]["value"]
    print(f"error splice L{layer}: native dM = {detail['splice'][layer]}")

w = op("whatif", {"target": "L22.h_out", "value": f"feature[{top['feature']},coef]*-1",
                  "op": "add", "consumer": A})
if "error" not in w:
    detail["ablate"] = {"native": w["measured"]["delta"]["value"],
                        "linear": w["prediction"]["linear"]["value"]["value"],
                        "exact": w["prediction"]["exact"]["value"]["value"]}
else:
    detail["ablate"] = w["error"]
print("ablate top feature at L22:", detail["ablate"])

# the replacement model: every layer's residual replaced by its reconstruction in one pass
op("start", {"prompt": P1, "name": "chain"})
for layer in range(1, 28):
    op("until", {"target": f"L{layer}.h"})
    op("set", {"target": f"L{layer}.h", "value": "error*-1", "op": "add"})
op("until", {"target": "logits"})
m, am = margin(Pa, Ro)
detail["chain"] = {"margin": m, "top_token": tok.decode([am])}
print("replacement model margin", m, "top token", repr(tok.decode([am])))
RESULT["detail"] = detail

# %% [markdown]
# ## 6. The key numbers and a plot
#
# `sigma_Phi` (features) and `sigma_E` (error) are medians over pairs with `|Delta| >= 1`.
# Splice amplification `A = native / -direct` is taken over pairs whose direct error share has
# magnitude above 0.05. `expected_outputs.md` lists what the original 18-pair run measured; with
# fewer pairs the medians move a little, so the per-pair rows are the sharper check.

# %%
med = lambda xs: st.median(xs) if xs else float("nan")  # noqa: E731
good = [r for r in ROWS if r["reader"]["delta"] and abs(r["reader"]["delta"]) >= 1.0]
sphi = [r["reader"]["phi"] / r["reader"]["delta"] for r in good]
se = [r["reader"]["E"] / r["reader"]["delta"] for r in good]
consist = sum(abs(r["reader"]["delta"] - r["cell"][r["best_layer"]]) <= 0.5 for r in ROWS)
summary = {"n_pairs": len(ROWS), "median_sigma_phi": med(sphi), "median_sigma_E": med(se),
           "delta_matches_cell": f"{consist}/{len(ROWS)}", "splice": {}}
for layer in SPLICE_LAYERS:
    nat = [r["splice"][layer]["native"] for r in ROWS if r["splice"][layer]["native"] is not None]
    amp = [r["splice"][layer]["native"] / -r["splice"][layer]["error_direct"]
           for r in ROWS if r["splice"][layer]["native"] is not None
           and abs(r["splice"][layer]["error_direct"]) > 0.05]
    summary["splice"][layer] = {"median_abs_native": med([abs(x) for x in nat]),
                                "median_abs_A": med([abs(x) for x in amp]), "n_A": len(amp)}
RESULT["summary"] = summary
print(json.dumps(summary, indent=1))

# %% [markdown]
# Reproduction check against the original run (`expected_outputs.md`, same pairs in the same
# order, one seed, bf16): per-pair margins must agree within 0.5 logits (the bf16 noise floor
# the study used) and the detail numbers within the stated tolerances.

# %%
EXPECTED_PAIRS = {   # pair: (base, Delta, Phi, E) from job-6aa7bc89a6ed (census rows)
    "Paris->Rome": (8.25, -8.25, -8.0, 0.0), "Rome->Berlin": (9.0, -14.375, -14.125, -1.25),
    "Berlin->Moscow": (10.625, -11.5, -12.5, 0.75),
    "Moscow->Athens": (12.75, -16.5, -14.75, -1.75),
    "Athens->Beijing": (10.25, -12.875, -11.125, -0.625),
    "Beijing->Madrid": (15.5, -13.25, -10.625, -2.0)}
checks = []
for r in ROWS:
    if r["pair"] in EXPECTED_PAIRS:
        eb, ed, ep, ee = EXPECTED_PAIRS[r["pair"]]
        ok = (abs(r["base"] - eb) <= 0.5 and abs(r["reader"]["delta"] - ed) <= 0.5
              and abs(r["reader"]["phi"] - ep) <= 0.5 and abs(r["reader"]["E"] - ee) <= 0.5)
        checks.append((f"census {r['pair']}", ok))
checks += [("recall base margin 9.5", abs(detail["base"] - 9.5) <= 0.25),
           ("top direct feature is f1077", detail["top_feature"]["id"] == 1077),
           ("replacement model margin ~ -3.23 (within 0.5)",
            abs(detail["chain"]["margin"] - (-3.2344)) <= 0.5),
           ("replacement model top token '-wide'", detail["chain"]["top_token"] == "-wide"),
           ("ablate f1077 native ~ -1.75 (within 0.5)",
            isinstance(detail["ablate"], dict) and abs(detail["ablate"]["native"] + 1.75) <= 0.5),
           ("ablate f1077 linear estimate ~ +0.30 (within 0.1)",
            isinstance(detail["ablate"], dict) and abs(detail["ablate"]["linear"] - 0.2955) <= 0.1),
           ("error splice L14 on recall ~ -0.5 (within 0.5)",
            detail["splice"][14] is not None and abs(detail["splice"][14] + 0.5) <= 0.5)]
sc = RESULT.get("saelens_check")
if isinstance(sc, dict):
    checks.append(("SAELens active set == mdb active set (rows 3 and 0)",
                   sc["row3"]["same_active_set"] and sc["row0"]["same_active_set"]))
    checks.append(("row 0 error fraction > row 3 (t0 caveat)",
                   sc["row0"]["error_frac_of_norm"] > sc["row3"]["error_frac_of_norm"]))
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
    names = [r["pair"] for r in ROWS]
    xs = range(len(ROWS))
    ax[0].bar([x - 0.2 for x in xs], [r["reader"]["phi"] / r["reader"]["delta"] for r in ROWS],
              0.4, label="features (Phi / Delta)")
    ax[0].bar([x + 0.2 for x in xs], [r["reader"]["E"] / r["reader"]["delta"] for r in ROWS],
              0.4, label="SAE error (E / Delta)")
    ax[0].set_xticks(list(xs), names, rotation=40, ha="right", fontsize=7)
    ax[0].set_title("Where the entity variable lives")
    ax[0].legend(fontsize=7)
    ax[1].bar([str(l) for l in SPLICE_LAYERS],
              [summary["splice"][l]["median_abs_native"] for l in SPLICE_LAYERS], color="#b5651d")
    ax[1].set_title("Median |margin change| from splicing out the SAE error")
    ax[1].set_xlabel("layer")
    fig.tight_layout()
    fig.savefig(OUT / "sae_census.png", dpi=130)
    try:
        from IPython.display import display
        display(fig)
    except ImportError:
        pass
RESULT["seconds"] = round(time.monotonic() - T0, 1)
json.dump(RESULT, open(OUT / "sae_census_results.json", "w"), indent=1, default=str)
print(f"done in {RESULT['seconds']}s; wrote {OUT}/sae_census_results.json")
print("RESULT_JSON " + json.dumps(RESULT, default=str))
