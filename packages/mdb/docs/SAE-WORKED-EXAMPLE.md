# SAE worked example: a variable census and causal error accounting

This page shows how to use mdb with a sparse autoencoder (SAE) suite, using the Qwen3-1.7B-Base and Qwen-Scope result as the example. Every command and function below exists in `mdb.ops.features` and `mdb.analysis.sae`; every number was measured on the developers' GPU (one RTX 4080, bf16, eager) and is not
reproducible from this package alone; [example 01](../examples/01-sae-census-qwen3-1.7b/) reruns
the census part on six pairs.

## Why bother: the headline result

An SAE writes a residual vector as `x = b_dec + sum_f a_f d_f + e`. The error `e` is the remainder, so the identity is exact. mdb runs the real model, edits any port or cache cell, and measures the native change in logits with the gap split into causes. Together they answer a question neither answers alone: how much of a counterfactually defined variable does the dictionary see, and how much lives in what it drops?

Measured on Qwen3-1.7B-Base, bf16, eager, Qwen-Scope `SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` (k = 50, revision `ce1a79d9`), single seed:

- The variable is in the features. Transplanting the entity row's value cell moves the answer; the whole-residual change splits into a feature part and an error part. Over 18 city pairs the median feature share is 0.970 and the median error share 0.046.
- The computation is not. Splicing out the error at layer 14 moves the answer by a median 1.25 logits, about 9 times its direct share (median absolute amplification 9.04 on 14 pairs). Replacing every layer's residual by its reconstruction takes the recall margin from 9.5 to -3.23 and the top token off " Paris" (, one prompt pair).
- Steering is nonlinear. Removing the feature with the largest direct effect (f1077 at L22, direct +6.80) moved the margin by -1.75, while the sealed local-linear estimate was +0.30.

## 1. Load a model and an SAE

mdb needs a loaded Hugging Face Qwen2 or Qwen3 model in eager attention and eval mode.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import mdb
from mdb.ops import dispatch

path = "/path/to/Qwen3-1.7B-Base"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
dbg = mdb.attach(model, tok)
ctx = dbg.ops(None)            # a context for dispatch; the REPL verbs are ops
```

In the REPL (`mdb engine MODEL`) the same ops are verbs. Load the SAE set:

```
(mdb) sae load qs /path/to/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50
```

```python
dispatch(ctx, "sae", {"action": "load", "name": "qs", "path": "/path/to/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"})
```

Usage: `sae load NAME PATH [layers=A:B] [k=K] [cache=N] [verify=true] | sae list | sae drop NAME`. The directory holds `config.json` (with `k` and `hook_point`) and `layer{L}.sae.pt` files with `W_enc b_enc W_dec b_dec`. `hook_point: resid_post` of layer `L` is the port `L<L>.h_out`, the same value as `L<L+1>.h`. A single `.pt` file needs `layer=` and `k=`; `port=h_out|h` overrides the hook. `verify=true` checks the file hashes against a manifest. `cache=N` is how many layers stay in memory (default 4); the example jobs used `cache=28` and peaked near 17 GB of host RAM, so keep the default unless you need all layers hot.

Check the site alignment once: the per-layer profile (next section) with `shift=-1` and `shift=1` applies each SAE a layer off its site. On Qwen3-1.7B the declared site had the lowest error fraction of the norm at L4, L10, L14, L20 and L22 (for example L14: 0.424 declared against 0.504 and 0.485), but the separation is modest because the identity closes at any site.

## 2. `features` and the error row

Start a run and stop at the answer row, then look at a port:

```
(mdb) start The Eiffel Tower is located in the city of
(mdb) until logits
(mdb) features L22.h_out top=10 lens=5
```

`features ADDR [top=K] [sae=NAME] [consumer=M[a,b]] [lens=N] [circuit=true] [match=UNIT]` lists the active features (activations recomputed in float64 from the native bits, the top-k selection in float32 as published), then the `b_dec` row, then the `ERROR` row with `||e||/||x||` and its energy, then the identity closure (about 1e-15). The error is never dropped. `lens=N` adds the tokens each feature direction promotes (a logit lens, labelled derived-local). `ADDR` must be a whole residual port: `L<i>.h_out` or `L<i>.h`.

## 3. Margin split with closure

Add `consumer=M[a,b]` (token ids, logit a minus logit b):

```
(mdb) features L22.h_out consumer=M[12095,21718] top=10 lens=5
```

Each feature, the bias and the error get an exact direct effect on the margin (P19, along the skip path; see [PROOF-IDS.md](PROOF-IDS.md)), and the listing closes: `margin = direct (features + bias + error) + later layers + head bias + remainder`, the remainder measured. In  at `L20.h_out` the recall margin 9.5 splits as direct 3.8818 (features 3.1969, bias 0.022257, error 0.66265) plus later layers 5.5445 plus remainder 0.0736. The direct effect is only the skip path. A feature with a large direct effect can still move the answer very little when removed (section 5).

The per-layer profile is the quickest look at what the SAE drops:

```
(mdb) features L*.h_out consumer=M[12095,21718]
(mdb) features L*.h_out consumer=M[12095,21718] shift=-1
```

It prints, per layer, the direct share of the skip path, the features, bias and error terms, `error/direct`, the error's share, `||e||/||x||` and the closure. `circuit=true` (with a consumer) stores a circuit record whose error node is never folded away.

## 4. Error splice per layer

Replace the port by its reconstruction (add minus the error) and measure natively:

```
(mdb) whatif L14.h_out = error*-1 add consumer=M[12095,21718]
```

```python
r = dispatch(ctx, "whatif", {"target": "L14.h_out", "value": "error*-1", "op": "add",
                             "consumer": "M[12095,21718]"})
r["measured"]["delta"]["value"]      # native change, as in the census driver
```

`whatif` seals an exact float64 chain and a local-linear estimate before it runs the edit, then reports the native change, the split into linear, nonlinear, realisation and rounding, and a verdict against the floor (twice the bf16 spacing, 0.25 or more at these magnitudes). The result also carries `result["split"]["terms"]` and `result["prediction"]["exact"]`. To get the causal share against the direct one, compare the splice's native change to the error's direct effect from `features L*.h_out`: the amplification `A = native / -direct`. Census table, 18 pairs:

| Layer | median abs native change | pairs above 0.5 | median abs A (pairs used) |
|---|---|---|---|
| L8 | 0.312 | 2/18 | 6.82 (5) |
| L14 | 1.250 | 15/18 | 9.04 (14) |
| L18 | 0.562 | 9/18 | 2.93 (17) |
| L20 | 0.625 | 9/18 | 1.49 (16) |
| L22 | 0.812 | 12/18 | 0.51 (18) |
| L24 | 0.500 | 9/18 | 0.54 (18) |

Pairs with a direct share under 0.05 in magnitude are dropped from `A`. Single-layer splices do not sum: each one changes every later layer's error.

The native replacement model, where every layer is replaced by its reconstruction in one pass, stages the splice at each `L<l>.h` while stepping:

```python
dispatch(ctx, "start", {"prompt": prompt, "name": "chain"})
for layer in range(1, 28):
    dispatch(ctx, "until", {"target": f"L{layer}.h"})
    dispatch(ctx, "set", {"target": f"L{layer}.h", "value": "error*-1", "op": "add"})
dispatch(ctx, "until", {"target": "logits"})
```

The error is read at the current, already-edited value, so each step reconstructs the previous steps' output. Result: recall margin 9.5 to -3.2344, top token "-wide"; follow-up margin 8.25 to 2.5195, top token "ways".

## 5. Feature edits in `set`, `whatif` and `compare`

A feature expression is a value anywhere an edit value is accepted. Terms (no spaces, never evaluated, joined with `+` or `-`, each optionally scaled with `X*` or `*X`):

| Term | Adds |
|---|---|
| `feature[ID]*X` | `X` times the unit decoder direction `d_ID` |
| `feature[ID,coef]*X` | `X` times `a_ID d_ID`, the feature's own contribution at the site (`*-1` ablates it) |
| `error` | the error `e` (`error*-1` splices in the reconstruction) |
| `recon` | `b_dec + sum a_f d_f` |
| `bias` | `b_dec` |
| `TERM@RUN` | the term evaluated on the named run's value at the same site |

```
(mdb) whatif L22.h_out = feature[1077,coef]*-1 add consumer=M[12095,21718]
(mdb) whatif L20.h_out = feature[25654,coef]*4 add consumer=M[12095,21718]
(mdb) set L20.h_out = recon replace
```

Ablating f1077 at L22: native -1.75, exact chain -1.857, linear +0.2955, nonlinear -2.153, against a direct effect of +6.80. Dose curve for f25554 at L21 on the follow-up (France minus Italy): 0.5 gives 0.5, 1 gives 0.75, 2 gives 1, 4 gives 1.688, where the linear estimate at 4 is 5.124.

`compare A B [consumer=C]` runs two edits (`ADDR=VALUE [op]`) side by side with the same consumer:

```python
dispatch(ctx, "compare", {"a": "L22.h_out=-feature[1077,coef] add",
                          "b": "L22.h_out=-feature[19276,coef] add",
                          "consumer": "M[12095,21718]"})
```

### Cross-run values with `@RUN`

Name a second run, then take its value at the same site. `RUN` is an identifier (the origin is read up to the next sign):

```python
dispatch(ctx, "start", {"prompt": counterfactual, "name": "rome"})
dispatch(ctx, "until", {"target": "logits"})
dispatch(ctx, "start", {"prompt": base_prompt, "name": "main"})
dispatch(ctx, "until", {"target": "logits"})
# swap the counterfactual's feature part in, keep this run's error
dispatch(ctx, "whatif", {"target": "L22.h_out", "value": "recon@rome-recon", "op": "add",
                         "consumer": "M[9625,15344]"})
# only the error part
dispatch(ctx, "whatif", {"target": "L22.h_out", "value": "error@rome-error", "op": "add",
                         "consumer": "M[9625,15344]"})
# the two named features, each swapped to the other run's activation
dispatch(ctx, "whatif", {"target": "L22.h_out",
         "value": "feature[1077,coef]@rome-feature[1077,coef]+feature[15296,coef]@rome-feature[15296,coef]",
         "op": "add", "consumer": "M[9625,15344]"})
```

`recon@rome-recon` is the feature part of the other run minus this run's (`Φ`). `error@rome-error` is the error part (`E`). `recon@rome+error@rome` with `replace` is the whole residual from the other run. Whole-residual cells can also come from another run through `set ADDR = ORIGIN:ADDR`, as in `set cache.L20.v[row=3] = cf:cache.L20.v[row=3]`.

Measured (France minus Italy, base 8.25): at L22 the full residual patch moves the margin by -14.75, the feature part by -14.125, the error part by -0.75, one feature each way (f1077, f15296) by -11.75 (80 percent of the full patch), three each way by -12.625. At L23 three each way gives -14, 95 percent of -14.75.

## 6. Unit to feature matching and feature traces

Name an mdb unit by the decoder directions closest to its write:

```
(mdb) features L20.h_out match=L20.A[h13] top=8 lens=5
(mdb) features L22.h_out match=L22.f[i=5773]
```

Accepted units: `L<l>.f[i=N]` (MLP neuron N), `L<l>.A[hH]` (head H's output), `L<l>.A[hH,src=S]` (head H reading source row S). The listing gives each feature's cosine, coefficient, whether it is active, and the fraction of the write outside the listed span. In the recall prompt's `L20.A[h13]` matched f25654 at cos 0.68197. Known gap: the neuron match cosine ignores the sign of the neuron's activation, so a neuron can show a negative cosine and still write positively into a feature.

Trace one feature through every layer of the cursor's row:

```
(mdb) feature 25654 layer=20 top=4
```

It prints the encoder probe at `h`, `r` and `h_out` of every layer, the layer where the probe first crosses the top-k cutoff, and the exact attention and MLP write shares with the top heads and neurons. For f25654 at L20: attention writes +141.86, head 13 +108.08, head 12 +32.343; the L22 MLP writes +212.29, neuron 5773 +189.19. To stop on a feature activation, watch it with a `FeatureCondition` (see [example 04](../examples/04-watch-edit-continue-country-feature/)).

## 7. The census recipe in Python

The census asks, for pairs of prompts that differ at one entity row, how much of the effect of transplanting the entity's value cell is carried by features and how much by the error. Template `"The city of {X} is the capital of"` (8 tokens, entity row 3, pending row 7); consumer `M[country_X, country_Y]`; the counterfactual is Y's prompt. This is the logic of the original census driver, trimmed.

```python
import torch
from mdb.ops import dispatch

def margin(ctx, a, b):
    row = int(ctx.run.where.split(".")[0][1:])
    logits = ctx.run.read(f"t{row}.logits").reshape(-1).double()
    return float(logits[a] - logits[b])

def run_to_logits(ctx, ids, name):
    dispatch(ctx, "start", {"prompt": ids, "name": name})
    dispatch(ctx, "until", {"target": "logits"})

def native(ctx, ids_x, ka, kb, base, steps):
    """Fresh main run, edits staged at their stops, run to logits; margin change vs base."""
    dispatch(ctx, "start", {"prompt": ids_x, "name": "main"})
    for stop, target, value, kind in steps:
        if stop:
            dispatch(ctx, "until", {"target": stop})
        args = {"target": target, "value": value}
        if kind:
            args["op"] = kind
        dispatch(ctx, "set", args)
    dispatch(ctx, "until", {"target": "logits"})
    return margin(ctx, ka, kb) - base

def census_pair(ctx, ids_x, ids_y, ka, kb, layers=(16, 18, 19, 20, 21, 22)):
    run_to_logits(ctx, ids_y, "cf")
    run_to_logits(ctx, ids_x, "main")
    base = margin(ctx, ka, kb)
    # 1. the variable: V-cell transplant of the entity row, best layer
    cells = {L: native(ctx, ids_x, ka, kb, base,
                       [(None, f"cache.L{L}.v[row=3]", f"cf:cache.L{L}.v[row=3]", None)])
             for L in layers}
    best = min(cells, key=cells.get)
    # 2. the cell's run, read at the best layer's port
    dispatch(ctx, "start", {"prompt": ids_x, "name": "T"})
    dispatch(ctx, "set", {"target": f"cache.L{best}.v[row=3]",
                          "value": f"cf:cache.L{best}.v[row=3]"})
    dispatch(ctx, "until", {"target": "logits"})
    site = f"L{best + 1}.h"                      # same value as L{best}.h_out
    out = {}
    for key, value, kind in (("delta", "recon@T+error@T", "replace"),   # whole residual
                             ("phi", "recon@T-recon", "add"),           # feature part
                             ("E", "error@T-error", "add")):            # error part
        out[key] = native(ctx, ids_x, ka, kb, base, [(site, site, value, kind)])
    out["sigma_phi"] = out["phi"] / out["delta"]
    out["sigma_E"] = out["E"] / out["delta"]
    out["cell"] = cells[best]
    return out
```

Edits are staged at their stops (`until` the port, then `set`) and each arm is a fresh run, so the arms do not contaminate each other. The consistency check is `abs(out["delta"] - out["cell"]) <= floor` (the census used 0.5): on the real run it held on 18/18 pairs. Per-layer splice arms are one more `native` call with `("L{l+1}.h", "L{l+1}.h", "error*-1", "add")`. After the loop, take medians of `sigma_phi` and `sigma_E` over pairs with `abs(delta) >= 1`. Measured, 18 pairs: 0.970 and 0.046.

If you only want the numbers on one prompt pair with sealed predictions, call `whatif` instead of `native`; it costs 1.3 to 13.5 s per call on this model, which is why the census arms ran natively and only the detail pair was sealed.

## 8. The result tables, for reference

Source side (entity-row residual at `L<l>.h_out` carried into later K and V cells, minus a null arm), median over 18 pairs: L4 feature share 0.978, error share 0.023; L10 0.957 and 0.038; L16 0.936 and 0.045. Worst reader-side pair: Copenhagen to Stockholm with feature share 0.61 and error share 0.22.

## 9. Limits

- 18 city pairs, not the 32 preregistered: the city list yields 19 entries and 18 consecutive pairs.
- Four of the 18 prompts do not answer with the country (" the" three times, " South" once).
- The census is one template. The splice-by-layer detail, steering curves, full-chain replacement, feature swaps and unit matching are one prompt pair.
- k = 50 only; the k = 100 suite was not downloaded.
- Logit-lens token lists are float32 rankings (derived-local), not decompositions.
- mdb has a Qwen3 port program verified by the selfcheck on Qwen3-1.7B-Base, but not on other sizes.
- Single seed.

## 10. Known gaps

- A `whatif` costs about 0.45 s per layer between the edit and the consumer on this model, so
  sealed sweeps are slow; the census arms ran natively for that reason.
- `match` ignores the sign of a neuron's activation.
- This page predates `start prefill=K` and feature watches (`FeatureCondition`, see
  [example 04](../examples/04-watch-edit-continue-country-feature/)): it patches the entity row
  through a truncated run and a cache-cell carry, which `prefill=K` now makes unnecessary.

The unit tests in `tests/test_sae.py` run on a tiny Qwen2 and Qwen3 with random SAEs: the identity closes, the margin split closes, a feature edit equals adding the decoder direction, every op result is labelled, error splice and coefficient ablation, cross-run terms and refusals, term parsing, and feature-trace writers summing to the probe.
