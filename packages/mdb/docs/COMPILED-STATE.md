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
decoders with full attention. `DecoderSplice` computes the prefix K/V once.
`run(hidden, L)` executes only decoder layers `L..` on a supplied residual,
calling the model's own decoder layers in a manual loop: each layer reads
`[prefix | injected | own]` keys and values under a causal mask over absolute
positions, with RoPE at the absolute positions. No layer below `L` executes,
and `L` equal to the layer count runs no layer at all, so the compiled state
goes straight to the final norm and head.

- **True resume.** `run.run(state_at_L, L)` with the residual entering `L` at
  every suffix position.
- **Partial positions and K/V injection.** `run.run(x, L, positions=[7, 8, 9],
  kv={l: (k, v)}, kv_positions=[0, ..., 6])` runs only the listed positions;
  the others contribute the supplied post-RoPE K/V at every layer `l >= L`
  (compiled, or native for an oracle). This is the forecast-cost Part F T3c
  predictor: answer positions run `L..27`, operand K/V are compiled.
- **Native passes.** `native(ids, capture=..., capture_kv=...)` is the
  prefix-cached forward through the same loop; `decode(ids, steps, vocab)` is
  KV-cached greedy decoding and also returns the final residual at every
  position whose input is known; `readout(hidden, vocab)` is the final norm
  and head, optionally restricted to a token subset.
- `forward(ids, capture=..., splice=(L, state))` is the independent hook path
  through the unmodified model, kept for the parity gates.

`kv_rows`/`kv_unrows` flatten one position's K/V at one layer to a
`2 * kv_heads * head_dim` row and back; `CompilerFit` fits K/V maps on those
rows (`kv_layers=...`, `solve(..., target="kv")`), pooled over any switch.

```python
from mdb.compiled_state_runner import Rows, measure

report, maps = measure(model, family, prefix_ids, Rows(fit_ids, fit_values),
                       Rows(eval_ids, eval_values, reference_ids),
                       layers=[20, 28], partial_layers=[20], lam=1.0, threshold=0.95)
```

`measure` accumulates the fit moments on the device, solves every
(feature set, layer) residual map, and with `partial_layers` also K/V maps for
every layer from the shallowest partial layer up. On the evaluation rows it
resumes each residual map at `L` (full arm) and runs each partial arm twice,
with compiled K/V and with native K/V (`native_kv_oracle`). It returns a sealed
`mdb-compiled-state-residual-v1` report plus the maps, keyed `(feature_set, L)`
and `(feature_set, L, "kv")`. The report also carries `noise_ceiling`: native
top-1 against native top-1 with every batch boundary shifted by half a batch.
Its error recall bounds what any compiler can reproduce under batch-shape noise.

## Parity gates

`parity(run, family, ids, layers, partial_layers=...)` runs on one batch of
native rows and compares top-1 and max |Δlogit| at the score positions:

- `manual_native_vs_hf`: the manual layer loop from embeddings vs the model's
  own forward;
- `resume[L]`: resuming from the captured native state at `L` vs manual native;
- `kv_injection[L]`: only the score positions run from their native state, the
  others inject their native K/V, vs manual native;
- `partial_vs_hook[L]`: that partial run vs the hook-path splice of native
  states at `L`.

The gate passes when every check has top-1 agreement 1.0 (and max |Δlogit| at
most `tol`, if given); `exact` says whether every max |Δlogit| is 0. `measure`
puts the gate for its own layers in every report.

## Timing protocol

`mdb.timing` is the registered protocol for speed claims. `time_arms(arms,
baseline=..., gate=...)` times named zero-argument callables: 5 warmup calls,
then 30 repetitions, each bracketed by `torch.cuda.synchronize` on CUDA, and
reports median, p10 and p90 milliseconds plus an `nvidia-smi` process snapshot
before and after. Every arm gets `speedup_vs_baseline` only when `gate["passed"]`
is true. Without a passing parity gate the record keeps the timings, sets
`claim` to `refused: parity gate missing or failed` and reports no speedup.
The protocol's batches are 1 and 256. The baseline for a predictor is the
prefix-cached native forward of the same rows through the same layer loop; for
a decoding chain it is KV-cached native decode.

## Forwardless serving

`mdb.compiled_state_serving` serves at `L` equal to the layer count, where no
transformer layer runs.

- `ZeroLayerChain(run, map_at_num_layers, feedback={"r0": (0, table)},
  vocab=None)` predicts the score positions in order. After step `k` it feeds
  the predicted token back into every symbol that `feedback` assigns to `k`,
  through a lookup table over the vocabulary. It returns tokens and per-step
  margins (top-1 minus top-2 logit). `teacher=` feeds given tokens back
  instead. `chain_agreement` reports free-running `exact_through` per step and
  teacher-forced step agreement against native greedy tokens.
- `serve(run, family, feature_set, template_ids, values, steps=..., tau=...,
  warmup=..., refit_every=...)` answers a prompt stream in order. The first
  `warmup` prompts get a native KV-cached decode and seed a streaming ridge
  fit. After that, each batch is predicted by the chain, and every row whose
  smallest step margin is below `tau` is served by a native decode instead;
  that row is added to the fit, and the maps are re-solved once `refit_every`
  new native rows have arrived. The record counts native prompts, native token
  steps and `forwards_per_1000_prompts`. Native tokens passed as
  `audit_tokens` score the served answers per window of prompts; they are
  never counted as forwards and never reach the fit.

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
intervals on 49 errors. The native-splice control of that runner version (the
hook path, before true resume) is exactly 0 at all eight layers. The batch-shift noise ceiling is 1.000 (error recall 1.000, Wilson
[0.927, 1.000]): on this sample, shifting batch boundaries changed no native
answer. Bit-level batch noise therefore does not limit error recall here; Part
D's 0.9985 pass consistency compared different sequence lengths, which this
ceiling does not vary. Single seed; no timing claim.

### True resume, K/V injection, timing and forwardless serving

Job `job-764839b8c085` (Beast RTX 4080, CUDA BF16, torch 2.13.0+cu130,
transformers 5.14.1; 107 s wall, worker 42 s, peak VRAM 5,544 MB by the
scheduler and 4,739 MB allocated, peak RSS 3.0 GB) ran
[`compiled_state_forwardless.py`](../qualification/compiled_state_forwardless.py)
in its smoke configuration; the record is
[compiled-state-forwardless-20261102.json](../qualification/compiled-state-forwardless-20261102.json).
Addition fits on 4,096 rows of the Part E/F train half (Part E/F used all
160,200) and scores the first 1,024 rows of the Part D/E sample (28 native
errors). G1 replays Part G's stage5 RNG order: holdout
[0, 1, 9, 18, 27, 29, 31, 36, 39], 738 train, 744 SEEN and 384 HO rows, and
0 tokenization mismatches over all 2,256 rows.

**Parity gates (64 rows each, every layer 0..28, partial runs at every layer
0..27): passed for both families.** The manual layer loop equals the HF forward
exactly (max |Δlogit| 0), and resuming from the native state equals native
exactly at every L. Native K/V injection and the partial runs agree on every
top-1, with max |Δlogit| 0.281 (addition) and 0.375 (G1); the likely cause,
not isolated, is BF16 rounding in sdpa over fewer query rows.

The rules match agreement within max(0.005, two binomial standard errors) and
speed within a factor of 0.8–1.25. Error recall on 28 errors has a standard
error near 0.1, so those rows below cannot fail and are not shown.

| Part | Claim | Earlier | mdb | Match |
|---|---|---|---|---|
| E | F4(20) answer agreement | 0.9840 | 0.9824 | yes |
| E | F4 carry-switched(28) / F5(28) answer agreement | 0.9787 / 0.9787 | 0.9727 / 0.9727 | yes |
| F | T3c(20) answer agreement (compiled operand K/V) | 0.9831 | 0.9805 | yes |
| F | T3c(20) with native operand K/V | 0.9949 | 0.9961 | yes |
| E | native prefix-cached forward, batch 1 / 256 (ms) | 10.98 / 105.18 | 11.37 / 105.19 | yes |
| E | F4(20) + digit head speedup, batch 1 / 256 | 2.96 / 3.46 | 3.23 / 3.49 | yes |
| F | T3c(20) + digit head speedup, batch 1 / 256 | 2.78 / 8.59 | 2.67 / 8.29 | yes |
| E | F5(28) + digit head speedup, batch 1 / 256 | 14.93 / 139.69 | 48.74 / 419.49 | **no** |
| G | G1 4-token zero-layer chain exact_through (744 SEEN) | 1, 1, 1, 1 | 1, 1, 1, 1 | yes |
| G | G1 chain vs KV-cached native decode, batch 1 | 11.68 | 14.15 | yes |
| G | G1 chain vs KV-cached native decode, batch 256 | 29.91 | 15.65 | **no** |

Two speed results do not match:

- **F5(28).** mdb is faster than Part E reported. The native baseline matches
  to 0.02 ms, but the predictor arms differ: mdb compiles only the three score
  positions from device symbol values (0.251 ms at batch 256), while Part E
  compiled all ten positions from A and B on the device (0.753 ms).
- **G1 chain at batch 256.** mdb is slower than Part G reported. Native decode
  matches (188.3 ms here, 190.2 ms in Part G), but the chain took 12.03 ms
  against Part G's 6.36 ms. The chain timed in this job computed a
  full-vocabulary top-2 with a float upcast at every step, for its margins;
  Part G took a bf16 argmax. `ZeroLayerChain(..., margins=False)` now takes
  the argmax path, but it has not been re-timed. The subset-readout chain (all
  train tokens) took 0.699 ms (269×) at batch 256.

The addition zero-layer F5 chain, which feeds back its own r0 and r1, matches
native greedy decode on 97.6%, 97.5% and 97.5% of rows through steps 1–3. The
native answer accuracy is 97.5%: like Part E's F5(28), the chain answers like a
calculator and keeps none of the model's errors.

Serving (`serve`), counting native forwards per 1,000 served prompts:

| Stream | tau | Native prompts per 1,000 | Served = native | Chain-served = native |
|---|---|---|---|---|
| addition, 1,024 rows, 256 warmup | none | 250 | 0.961 | 0.948 |
| addition | 2 | 291 | 0.979 | 0.970 |
| G1, 2,256 rows incl. HO fillers, 64 warmup | none | 28 | 0.342 | 0.323 |
| G1 | 2 | 132 | 0.997 | 0.999 |

With 64 warmup rows the G1 chain has not seen most fillers. A margin
threshold of 2 logits sends the unfamiliar rows to native and refits on them,
so 132 native prompts per 1,000 serve 99.7% of the stream exactly. Single seed;
smoke-scale fit for addition; 10 timed repetitions (Parts E–G used 30).

## Limits

The view measures a linear compiler of declared finite symbols on fixed-length
templates. Open-vocabulary slots need embedding features, and templates with
variable lengths need position alignment; neither is implemented. Agreement is
greedy top-1 at the score positions; KL to the native distribution is not
reported. True resume and K/V injection need a full-attention decoder whose
layers accept a cache object with `update(k, v, layer_idx)` (Qwen2-style
attention in transformers 5.x); sliding-window and hybrid layer types are
refused. The serving loop refits on the CPU from device moments and is not
timed; its margin threshold is in raw logits, not calibrated.
