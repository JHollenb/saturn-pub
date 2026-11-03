# Compiled-state residual view

`mdb.compiled_state` asks how far a linear *state compiler* reproduces a model
on a templated prompt family, and at which layer it stops. A compiled state
replaces the residual entering layer `L` at every suffix position with a
per-position linear function of declared symbols that the position can causally
see; the model's own layers `L..` then run on top. The view reports, per
(position, layer), the mean native norm, the mean residual norm
`||native - compiled||` and pooled R²; per layer, native top-1 agreement at the
score positions, with error recall and precision when an evaluator reference is
supplied; and per feature set, the agreement *horizon*.

The method and its first results come from the 2026-11-02 forecast-cost Parts
D and E (Qwen2.5-1.5B, three-digit addition): digit identities alone reproduce
the model through L22 and break at L23, the layer where the dynamic symbol
table places the compose step; column sums, carries and pairs move the horizon
to L24; one map per carry state reaches L28 with zero transformer layers.

## Prompt families

A `PromptFamily` is a shared prefix, named suffix positions, finite `Symbol`s
and named `FeatureSet`s. Each symbol declares its cardinality and
`visible_from`, the first suffix position whose causal context determines it.
mdb never derives symbol values from tokens: the family supplies them per row,
so derived symbols such as column sums or carries are declared like any other.
A feature set lists the symbols a compiler may read; `switch` fits one map per
value of a symbol at the positions that can see it, while earlier positions
keep the pooled map.

```python
from mdb.compiled_state import FeatureSet, PromptFamily, Symbol

family = PromptFamily(
    name="toy", prefix="shots\n", positions=("x0", "x1", "op", "y"),
    symbols=(Symbol("x0", 3, 0), Symbol("x1", 3, 1), Symbol("g", 2, 1)),
    feature_sets={"digits": FeatureSet(("x0", "x1")),
                  "digits|g": FeatureSet(("x0", "x1"), switch="g")},
    score_positions=(2, 3))
```

The feature layout is one bias column followed by one one-hot block per symbol,
in declared order. `visibility(family)` is the causal mask; `design` builds the
one-hot rows and rejects values outside a symbol's cardinality. The family is
content-addressed (`family.fingerprint`) and round-trips through JSON.

## Fit, compile and score

`CompilerFit` streams ridge sufficient statistics (per-position Gram and
`X^T Y`) for every feature set at the requested layers; `solve(feature_set,
layer, lam)` solves `(G + lam I) W = X^T Y` over the visible feature-set
columns, regularizing the bias like every other column. A `CompiledStateMap`
compiles states from symbol values and saves as `family.json` plus one JSON
manifest and one `.npy` per map; the manifest binds the family fingerprint and
the sha256 of the float32 weight bytes, and `verify_maps` rechecks both.

`ResidualAccumulator` pools R² per position as `1 - SSE / SST` with
`SST = sum ||x - mean||^2` over rows and hidden dimensions. A position whose
native state does not vary (for example a fixed operator token at layer 0) has
R² `None`. `agreement` scores rows jointly over the score positions and reports
Wilson intervals. `horizon` returns the deepest tested layer that, with every
shallower tested layer, reaches the threshold; it also reports the first break
and whether agreement recovers after it.

These functions are framework-free; numpy is imported only when map math runs.

## Native runner

`mdb.compiled_state_runner` runs the measurement on Hugging Face causal
decoders. `DecoderSplice` computes the prefix K/V once and runs each suffix
batch through the unmodified model with a copy of that cache. A forward
pre-hook on decoder layer `L`, or on the final norm when `L` equals the layer
count, observes the residual entering `L` or replaces it at every suffix
position. Layers below `L` still execute; nothing at or above `L` reads their
suffix outputs, so the splice is exactly "compiled state at `L`, native layers
`L..` on top". Because they execute, the runner makes no speed claim.

```python
from mdb.compiled_state_runner import Rows, measure

report, maps = measure(model, family, prefix_ids, Rows(fit_ids, fit_values),
                       Rows(eval_ids, eval_values, reference_ids),
                       layers=[4, 12, 20, 22, 23, 24, 26, 28], lam=1.0, threshold=0.95)
```

`measure` accumulates the fit moments on the device, solves every
(feature set, layer) map, splices each on the evaluation rows and returns a
sealed `mdb-compiled-state-residual-v1` report plus the maps. Each report also
carries two references:

- `native_splice_control`: the captured native state spliced back at each layer
  on the first evaluation batch, as max |Δlogit| (0 on the tiny-Qwen2 test);
- `noise_ceiling`: native top-1 against native top-1 with every batch boundary
  shifted by half a batch. Its error recall bounds what any compiler can
  reproduce under batch-shape noise.

## Command line

```bash
mdb compiled-state report.json                      # ladder per feature set
mdb compiled-state report.json --position eq        # norms and R² at one position
mdb compiled-state report.json --feature-set F1-digits --layer 22
mdb compiled-state report.json --maps maps/         # verify map digests named by the report
mdb compiled-state report.json --json
```

The command verifies the report fingerprint first and needs no model framework.

## Qualification

Job `job-5ac69ae2b1d4` (Beast RTX 4080, CUDA BF16, torch 2.13.0, transformers
5.14.1) ran the view through the sealed mdb payload path on Qwen2.5-1.5B
three-digit addition; the record is
[compiled-state-addition-20261102.json](../qualification/compiled-state-addition-20261102.json).
[`compiled_state_addition.py`](../qualification/compiled_state_addition.py)
declares the forecast-cost Part D/E family as mdb symbols (digits, column sums,
carries, pairs, totals and the carry-state switch, with Part D/E visibility),
rebuilds the same domain, seed and split, fits on the first 32,768 rows of Part
H's nested train order (20% of Part E's 160,200) and evaluates on the first
2,048 rows of the Part D/E random test sample (49 native errors; native
accuracy 97.6%). A 64-row probe found 0 tokenization mismatches. The worker
took 103 s with peak RSS 4.8 GB and peak VRAM 8.3 GB; `measure` itself took 57 s.

| Feature set | Horizon at 0.95 (first break) | Part E | Agreement at selected L (Part E, 32,768 rows) |
|---|---|---|---|
| F1 digits | L22 (L23) | L22 (L23) | L4 0.987 (0.990), L22 0.967 (0.971), L23 0.791 (0.790), L24 0.037 (0.029) |
| F4 + sums, carries, pairs | L24 (L26) | L24 (L26) | L24 0.978 (0.981), L26 0.668 (0.692), L28 0.395 (0.403) |
| F4 carry-switched | L28 (none) | L28 | L28 0.976 (0.979) |
| F5 + column totals | L28 (none) | L28 | L28 0.976 (0.979) |

F1's R² at L4 by position (a2 … r1) is 1.000, 0.817, 0.821, 0.905, 0.982,
0.823, 0.832, 0.911, 0.996, 0.866, against Part D/E's 1.00, 0.82, 0.82, 0.91,
0.98, 0.82, 0.83, 0.91, 1.00, 0.87. Error recall falls with depth as in Part E
(F1: 0.63 at L4, 0.43 at L22, 0 at L24; F4 and F5 reach 0 by L26), with wide
intervals on 49 errors. The native-splice control is exactly 0 at all eight
layers. The batch-shift noise ceiling is 1.000 (error recall 1.000, Wilson
[0.927, 1.000]): on this sample, shifting batch boundaries changed no native
answer. Bit-level batch noise therefore does not limit error recall here; Part
D's 0.9985 pass consistency compared different sequence lengths, which this
ceiling does not vary. Single seed; no timing claim.

## Limits

The view measures a linear compiler of declared finite symbols on fixed-length
templates. Open-vocabulary slots need embedding features, and templates with
variable lengths need position alignment; neither is implemented. Agreement is
greedy top-1 at the score positions; KL to the native distribution is not
reported. The runner does not skip the layers below `L`, inject compiled K/V or
splice a subset of positions, so it cannot time a compiled predictor.
