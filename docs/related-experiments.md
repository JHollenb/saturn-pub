# Related experiments and papers

The [experiment recipes](../experiments/README.md) are what this package ships and reproduces.
This page collects the measurements around them: follow-up runs on the circuit-tracer
flagship and the research papers that use the same execution model. Every number here is
measured; where a result is a separate run, its job ids and result file are given.

## circuit-tracer follow-ups (Gemma-2-2B, Gemma Scope transcoders)

All runs use the flagship's frozen panel (`experiments/circuit_tracer/panel.json`, 50 admitted
prompts), its decision rule (≥ 0.5 nats necessary, ≤ 0.1 nats absent), FP32 native weights, and
the corrected activation indexing. Result files are in
[`experiments/circuit_tracer/followups/`](../experiments/circuit_tracer/followups/).

### Dose, group size, and a matched-random control

Is the graph's top-k set special, and at what size and strength does the real model's answer
change? Groups are the graph's top-k features by influence, k ∈ {5, 10, 20, 50, 100}, set to
`factor × activation` with factor ∈ {0, −0.5, −1, −2, −4}. The control is k other active
features from the same prompt, matched on layer, position, and activation band. Preregistered
in [`PREREGISTRATION-dose-random-control.md`](../experiments/circuit_tracer/followups/PREREGISTRATION-dose-random-control.md);
results in [`dose-random-control-results.json`](../experiments/circuit_tracer/followups/dose-random-control-results.json).

Share of the 50 prompts whose native top-1 answer flips (graph-selected set):

| k | factor 0 (zero) | factor −1 | factor −2 |
| --- | :-: | :-: | :-: |
| 5 | 0.18 [0.08, 0.30] | 0.50 [0.36, 0.64] | 0.64 [0.50, 0.78] |
| 20 | 0.30 [0.18, 0.44] | 0.82 [0.70, 0.92] | 0.90 [0.82, 0.98] |
| 100 | 0.64 [0.50, 0.78] | 0.96 [0.90, 1.00] | 0.96 [0.90, 1.00] |

- **The graph's selection is specifically causal.** At k = 20 the mean native drop is 1.93 vs
  0.09 nats for the matched-random set at zero-ablation (flips 15 vs 5 of 50) and 18.51 vs 1.42
  at −2× (flips 45 vs 20). The graph's drop exceeds random by ≥ 0.5 nats in 24 of 25 cells.
- **Strong steering is partly generic.** Random sets also flip 20 / 50 at −2× and saturate
  near 1.0 at −4×, so a raw steering flip count overstates specificity without a control.
- **Bigger groups flip at weaker doses.** Zero-ablation reaches 64 % at k = 100.
- **Calibration over all 1,250 cells.** Graph prediction vs native drop: Spearman ρ 0.947
  (unconstrained), 0.933 (frozen attention), 0.876 (direct effects); mean absolute error 2.77,
  3.60, and 5.85 nats.
- **Pruned circuits and supernodes do not help.** circuit-tracer's default pruning (node 0.8,
  edge 0.98) keeps a median of 971 of 4,096 features, which flips every prompt and is
  uninformative. Automatic supernodes (position × layer band; 1,160 groups) are individually
  necessary in 8 % of cases at zero-ablation (94) and 29 % at −2× (340), but their agreement with the graph (0.78) does not beat top-20
  (0.76) by the preregistered 0.10 margin.
- **Determinism.** Rebuilt graphs reproduce the flagship's top-20 on all 50 prompts.

Jobs: `job-7c5949fd0d09` (smoke), `job-a3cacc860c09`, `job-693a029d160e`, `job-fc1b97f31e10`,
`job-20c58a384add`.

### Independent check of the correction, and precision

A second, independent implementation re-read every activation from the transcoder on the clean
run and repeated the top-20 zero-ablation. Results are in
[`corrected-dose-precision-results.json`](../experiments/circuit_tracer/followups/corrected-dose-precision-results.json).

- With the first run's doses it reproduces the withdrawn native numbers exactly (median
  difference 0.000). With correct doses it matches the package's sealed
  `native_group_intervention` exactly (median difference 0.000) and flips 15 / 50, the same as
  the corrected flagship.
- Of the 13 prompts the withdrawn run called graph over-predictions, 11 recover. The one that
  remains (`tr_es_dog`) is a frozen-attention artifact (unconstrained prediction 0.17 nats).
- **Precision is not the gap.** BF16 vs FP32 native group ablation differs by a median of
  0.026 nats (max 0.235) over 50 prompts.
- **Where the graph's features sit.** Of the top-20 features, 70.0 % are at the final token,
  25.1 % at the subject span, and 4.9 % elsewhere
  ([`feature-position-table.json`](../experiments/circuit_tracer/followups/feature-position-table.json)).

Jobs: `job-e1896cd600b7` (smoke), `job-02778f862be4`.

### In progress

The same comparison with a cross-layer transcoder on Gemma-2-2B and with per-layer transcoders
on Llama-3.2-1B has not run yet; nothing is reported for it here.

## Research papers

These papers use the same execution model (capture the unmodified run once, fork every
intervention from it, require a no-op fork to reproduce the parent exactly) and are published in
the author's research repository.

- **[Single-Site Tests Miss Distributed Stores](https://github.com/JHollenb/research/blob/additional-experiments/papers/pointwise-instruments-miss-distributed-circuits/paper.md).**
  One-site activation patching gives the wrong answer when the information an output depends on
  is spread across steps or layers. Evidence from FLUX.2 Klein 4B, five language models, a
  preregistered sparse-autoencoder comparison, and a blind-graded comparison of four standard
  readouts against the model's own output. Its execution appendix describes the capture-and-fork
  execution model this package implements. A revision in preparation adds the circuit-tracer
  comparison above as §4.9 (single graph features are almost never necessary; the graph's top-20 group is).
- **[The Circuit That Survived Its Coordinates](https://github.com/JHollenb/research/blob/additional-experiments/bfl/docs/certified-semantic-circuits/paper.md).**
  Causal evidence for distributed semantic routes in a production diffusion transformer
  (FLUX.2): routes certified by native rendering, with prompt-held-out replication. The
  [FLUX.2 writer recipe](../experiments/flux2_writer/README.md) uses the same block-level
  execution and native-render consumer.

See also: [circuit-tracer interop](circuit-tracer.md), [validation log](validation.md),
[Instrument Trial](trial.md).
