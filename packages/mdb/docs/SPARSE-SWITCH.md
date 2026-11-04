# Sparse switch view

`mdb.sparse_switch` measures the per-position MLP gate *list* a model's output depends on, and scores a predictor's proposed list against it. It packages the measurement the 2026-11-03 forecast-cost Parts Z and AA use on Qwen2.5-1.5B open text.

## What it measures

A gated MLP writes `y = W_down (g * u)` with `g = act(W_gate h)` and `u = W_up h`. Against a reference gate `g_ref` (for example each token's mean gate over fit contexts), the switch at one position is `s = W_down((g - g_ref) * u)`. With a Fisher metric `F` at the MLP write, each neuron's diagonal Fisher score is

```
score_j = |(g_j - g_ref_j) * u_j| * sqrt(w_j^T F w_j),   w_j = W_down[:, j]
```

The oracle list is the top `m` neurons by this score. Keeping the true gate for those neurons and `g_ref` for the rest is the oracle sparse code. A predictor supplies its own scores, for example from a reconstructed residual pushed through the model's own gate. The view then reports, per position:

| metric | meaning |
|---|---|
| `recall` | share of the oracle top-m that the predicted top-m contains |
| `mass_recall` | oracle score mass inside the predicted list over the mass inside the oracle list |
| `flip_in` | share of oracle top-m neurons whose score error exceeds their margin above the (m+1)-th score |
| `flip_out` | share of oracle ranks m+1..2m whose score error exceeds their margin below the m-th score |
| `margin_ratio` | median score error over the oracle top-2m, divided by the score spread sc_(m/2) − sc_(2m) |

`fisher_energy` and `SparseSwitchView.energy_kept` give the Fisher energy a code keeps, `1 - Σ rᵀFr / Σ sᵀFs` with `r = s - ŝ`. That is Part Y's offline scoring rule. Part Z found it ranks codes correctly within one family, but not reliably across families.

## Use

```python
from mdb.sparse_switch import SparseSwitchView, fisher_energy, fisher_neuron_weights, switch_scores

w = fisher_neuron_weights(F, W_down)              # [I]
view = SparseSwitchView()
oracle = switch_scores(g, g_ref, u, w)            # [N, I]
proposed = switch_scores(g_hat, g_ref, u_hat, w)  # same shape
view.add_lists(layer, "predicted", 256, oracle, proposed)
view.add_switch_energy(layer, fisher_energy(s, F))
view.add_energy(layer, "predicted", fisher_energy(s - s_hat, F))
report = view.report()
```

The module needs only torch tensors, and torch is imported by the caller. Nothing in it fits parameters: every quantity is a closed-form function of the captured gates, the weights and the Fisher.

## Results (Part AA, job-e6996b016820)

Setup: Qwen2.5-1.5B, CUDA BF16 weights with an fp32 residual, WikiText, single seed. The view ran inside the job from a byte-identical copy of this module, sha256 5e00940b…. It computed every score, mask, list metric and Fisher energy below. Its neuron weights equal Part Z's formula exactly (max abs difference 0.0).

The predictor works in three steps:
1. reconstruct the gate input by closed-form ridge from the top 16 live head reads plus the four lower MLP writes;
2. push the reconstruction through the model's own RMSNorm, gate and up projections;
3. take the top 256 by the same score.

What it got:
- **List quality, held-out, mean over L10–27:** recall 0.48–0.50 against the oracle list, mass recall 0.72–0.73, margin ratio 0.66–0.76. The token-mean residual alone gives recall 0.14–0.17.
- **Block test with an exact memory**, as the share of the token-mean freeze's per-position KL gap closed:

  | Arm | All 28 layers | L10–18 |
  |---|---|---|
  | oracle list (Part Z) | 89% | — |
  | predicted list, true gate values | 57% | — |
  | predicted list, predicted values | 52% | — |
  | predicted gate at every neuron | — | 73% |
  | Part Z's linear coordinate ridge from the same kind of head reads | — | 21–24% |

- **The offline Fisher energy tracks the causal gain** across the predicted codes at L10–18.

Full tables: `FINDINGS-AA.md` in `saturn/experiments/2026-11-02-forecast-cost-and-compressed-addition`.

## Codes with pools, bases and FLOP fractions (Part AB, job-f2add067bef1)

`code_products` writes a code as `W_down(products)`: exact gates on a kept set (a per-position list, a static `pool_from_counts` pool or a proxy-selected set) and a base elsewhere (token-mean gate, global-mean gate, i.e. `B_0 n` with `global_base_matrix`, or the token-table hybrid). `flop_fraction` gives the derived multiply-accumulates per replaced column as a share of a full column from the layer shapes. It is derived, not timed. `pool_mass` is the share of the oracle list's score mass inside a pool.

In Part AB, all codes ran as propagated columns at all 28 layers of Qwen2.5-1.5B, with an exact memory. Results are the share of the token-mean freeze's per-position KL gap closed, with agreement at offsets 32–39 after the block in parentheses:

| Code | f | KL gap closed (post-block agreement) |
|---|---|---|
| full-pool list with the token-mean base, SP256 | 1.0 | 89% (0.90) |
| full-pool list with the global base | 1.0 | 83% |
| rank-256 proxy selecting 512 neurons per position, global base | 0.33 | 62% (0.86) |
| static pool of 1,024, token-table hybrid base | 0.275 | 31% |
| static pool of 1,024, global base | 0.275 | −10% |
| static pool of 2,048, global base | 0.375 | 22% |
| global base alone | 0.175 | −134% |

Static pools hold only 33–45% (p = 1,024) of the list's score mass. Per-position selection is what lets the cheap global base work.

## Proxies and mixed-memory serving (Part AC, job-dd0421427744)

`reading_metric` builds `M = W_gᵀ diag(w² c²) W_g + W_uᵀ diag(w² e²) W_u` from the gate and up weights, a neuron weight `w` and two neuron moments (`c² = E[(silu′(a) u)²]`, `e² = E[g²]`). `metric_proxy` returns the rank-k map of the gate input that minimises `E‖M^{1/2}(I − B) n‖²` (reduced-rank regression, solved through `S = L Lᵀ` and the eigenvectors of `Lᵀ M L`, with no inverse of `M`); with `M = I` it is the PCA of `S`. `proxy_gate_inputs` applies a proxy with an optional token-table or global centre, and `hybrid_error_scores` is the token-table hybrid base's own per-neuron error. `ServingLedger`, `column_equivalents` and `readout_fraction` count a mixed-memory stream in which each position appends rows from the proxy column (cost f) or a native column (cost 1).

In Part AC, every proxy ran as a propagated column at all 28 layers of Qwen2.5-1.5B with an exact memory. KL gap closed at the first block position (and over the 32-position block), f derived from the shapes (not timed):

| Code | f | KL gap closed, first position (block) | post-block agreement |
|---|---|---|---|
| AB's PCA proxy, k = 256, 512 exact, global base | 0.331 | 62% (69%) | 0.86 |
| token-centred metric proxy, hybrid base, base-aware score | 0.331 | 63% (72%) | 0.88 |
| the same with the global score | 0.331 | 61% (70%) | 0.88 |
| the same with the weights-only centred ψ for the Fisher weight | 0.331 | 61% (71%) | 0.88 |
| the same at k = 512 | 0.436 | 74% (80%) | 0.88 |
| the same with 1,024 exact neurons | 0.381 | 65% (77%) | 0.89 |
| uncentred metric proxy, global base | 0.331 | 53% (67%) | 0.86 |

The metric proxy alone does not beat the PCA proxy (PCA already keeps 0.78–0.85 of `tr(M S)`); token centring and the rank k do. The hybrid base needs the base-aware score: with the global score it costs 5 points. Free-running serving (every position ≥ 4 replaced, uncovered tokens on a globally centred proxy) agrees with native at 0.43; margin gating at τ = 2 gives 0.83 at 825 column-equivalents per 1,000 tokens. Full tables: `FINDINGS-AC.md`.

## Coverage-complete tables and predicted-error gates (Part AD, job-9e76627f17e6)

`ridge_from_moments` fits a multi-output ridge with λ by GCV from centred moments; `shrinkage_lambda` and `shrinkage_means` turn it into coverage-complete token tables, `(c n̄_emp + λ n̄_prior) / (c + λ)` with the prior a ridge from the input embedding, so every token type has a table. `error_mass` is the score mass a code chose not to evaluate (`ê`), and `z_gate`, `kl_gate` and `quantile_threshold` gate on it. `ServingLedger` splits served and native positions and reports decide-after (with the readout) and early-abort counts.

On Qwen2.5-1.5B (all 28 layers, propagated, f derived not timed):
- The token-centred ψ-metric proxy with k = 512, 1,024 exact neurons, the base-aware score and coverage-complete tables closes **84.5%** of the per-position KL gap (88.5% over the block, post-block 0.912) at **f = 0.486**, against 89.6% for the full-pool list at f = 1.0.
- With every block position replaced (uncovered tokens through their coverage-complete rows) the block mean is 0.678: covered 0.667, uncovered 0.700 (Part AC's global-centred fallback: 0.478).
- A free stream that is predictor-built at every position agrees with native at 0.664; sending uncovered tokens to native columns (decided before the column, ρ = 0.33, 657 column-equivalents per 1,000 tokens) gives 0.751.
- Native columns reading a predictor-built memory at positions 64–127 agree at 0.891, so the stream's error is mostly the column's own.
- The model's own top-1 margin is below 1 at 47% of WikiText positions, which bounds what a margin gate can save. At ρ ≈ 0.50 the `ê` gates trade agreement for KL (KL-gate 0.527 vs margin 0.646).

Full tables: `FINDINGS-AD.md`.

## Bytes, timed records and pre-gates (Part AE, job-68b861b0e19b)

The cost helpers:

- `layer_bytes` and `bytes_per_token` derive the bytes a decode step reads from the shapes and dtype: weights, gathered exact rows, token tables and K/V. A batch's exact rows are counted one of two ways: `gather` reads each row's own rows, `union` reads the union of the batch's exact sets once.
- `timed_record` joins an `mdb.timing.time_arms` record with those bytes. It reports achieved GB/s, and the byte and time ratios against the baseline. The time ratio appears only when the parity gate passed.

The gate and table helpers:

- `pregate_table` builds empirical-Bayes key tables, with the global mean or a per-key prior (for example, a bigram key shrunk to the unigram table).
- `quantize_rows` performs symmetric per-row int8/int4 quantization.
- `spearman(ties=True)` uses average ranks, for table lookups with many tied values.
- `or_gate_thresholds` calibrates margin-OR-ê gates.
- `ServingLedger.add(groups=...)` reports agreement by a label, such as a context-count bucket.

Measured on Qwen2.5-1.5B, RTX 4080, bf16:

- **Decode timing.** One new token per DecoderSplice step, 128-token context, `layer.mlp` swapped for a sparse proxy MLP. The sparse MLP computes only:
  - the proxy,
  - top-1,024,
  - Triton-gathered exact rows,
  - the closed-form base y0 + B_0 (n − n̄).

  Results with the step CUDA-graphed:
  - Batch 1, readout included: native 6.26 ms (494 GB/s); A1 6.86 ms. That is 1.10× native's time at 0.56× its bytes (253 GB/s).
  - Batch 1, column only: 1.11× native's time at 0.48× its bytes.
  - Batch 256: the batch's exact sets cover 93% of the neurons, and A1 takes 1.67× native's time.
- **Why batch 1 is slower.** A per-op breakdown of one layer (L2-hot weights) shows the sparse MLP is latency-bound: 121 µs against native's 131 µs. `topk` alone is 42 µs. The byte saving needs a fused selection path before it can show in wall clock.
- **Decoder precision.** int8 and int4 proxy decoders keep 1.000 and 0.992 of the bf16 exact set's score mass. A two-stage proxy (64 → 2,048 → 1,024) keeps 0.855.
- **Pre-gates.** A token table of the code's fit-set error (leave-one-out tables) ranks test errors at Spearman 0.45 (bigram 0.47). As a pre-decided gate it serves no better than sending uncovered tokens to native (0.746 vs 0.752 at ρ 0.33).
- **k = 1,024.** It closes 93.5% at f = 0.697. Its free stream agrees with native at 0.763.

Full tables: `FINDINGS-AE.md`.
