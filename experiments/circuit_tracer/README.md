# circuit-tracer native edge test on Gemma-2-2B

This directory holds the committed, scrubbed verdict tables from running
`saturn_pub.interop.circuit_tracer.native_edge_test` on real `google/gemma-2-2b` weights with
the Gemma Scope transcoders (circuit-tracer's `"gemma"` preset). The measurement plan — prompt
panel, admission rule, thresholds, `top_k`, group interventions, prediction modes, and
statistics — was **preregistered** in [`PREREGISTRATION.md`](PREREGISTRATION.md) and
[`panel.json`](panel.json) **before** the run (panel `sha256`
`f787b9cf0c1d873a88fc3e5d657e1f06fb85936ec632227dd454f4e2dd98d0b6`).

Each `bundles/<prompt>/` holds a hash-pinned `edge_verdicts.json` + `manifest.json` that
re-derives **offline** with `saturn_pub.interop.circuit_tracer.verify_edge_bundle` — no GPU,
model, or circuit-tracer needed. [`summary.json`](summary.json) collects every headline number,
CI, and job id. The notebook
[`notebooks/05_circuit_tracer_native_edges.ipynb`](../../notebooks/05_circuit_tracer_native_edges.ipynb)
loads these committed bundles + `summary.json`, re-derives the verdicts in-process, and prints
the aggregate tables — it runs offline.

## What ran

Two models, one resident at a time, on one GPU host (RTX 4080, 16 GB), split into
≤12-minute jobs:

1. **Attribution graph + graph-side predictions (circuit-tracer, bf16, GPU).**
   `ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="nnsight")` builds an
   attribution graph per prompt; feature nodes are ranked by `compute_node_influence` and the
   top-20 selected. For each we run circuit-tracer's **own** `feature_intervention` (ablate the
   feature to zero) and record its predicted target-logprob drop, plus two **group** predictions —
   jointly ablate the whole top-20 set (`multiplier 0`) and steer it to −2× activation
   (`multiplier -2`) — under **three** intervention modes: frozen-attention (the circuit-tracer
   default), unconstrained (`freeze_attention=False`), and direct-effects
   (`constrained_layers=range(n_layers)`). A bf16 clean top-1 pre-filter skips attribution on
   candidates that will not be admitted.
2. **Native test (saturn-pub, fp32, CPU).** The real weights are wrapped with the native Gemma-2
   `DecoderAdapter`, validated (stepped == native logits, 16-token greedy == native, mid-layer cut
   fresh-process replay), and `native_edge_test` runs the *same* interventions on the real model.
   **Admission of record**: a prompt is tested only if the native fp32 greedy top-1 equals the
   target; failures are recorded and dropped, never replaced.

## Admission

56 preregistered candidates across 7 task families → **50 admitted** (6 dropped, recorded, not
replaced):

| family | candidates | admitted |
| --- | --- | --- |
| multi_hop_factual | 8 | 8 |
| capitals | 8 | 8 |
| antonyms | 8 | 6 |
| translation | 8 | 7 |
| acronym_abbreviation | 8 | 8 |
| arithmetic_sequence | 8 | 8 |
| category_rhyme | 8 | 5 |

Dropped: `ant_happy`, `tr_es_water`, `cat_trumpet`, `rh_cat`, `rh_light` (native top-1 ≠ target
at the bf16 pre-filter) and `ant_light` (bf16 top-1 matched but fp32 native top-1 did not).

## Correction (2026-10-05)

The first published run of this experiment dosed every native intervention with the wrong
activation. circuit-tracer's `Graph.activation_values` holds one value per *active* feature,
aligned with `active_features`; the edge selector read it by the *selected*-node position. Whenever
attribution kept only a subset of the active features (44 of the 50 admitted prompts hit the
4,096-node cap), each edge carried another feature's activation, typically far smaller (the
`mh_dallas` top feature: true 49.75, recorded 1.93). Graph-side zero-ablation predictions do not
use the activation and were unaffected; native ablations, both −2× steers, and the −2× graph
predictions were. `select_edges` now indexes through `selected_features` and refuses a graph
whose activations are not aligned with its active features (regression tests added). The whole
panel was rerun with the fix under the same frozen panel, rule, and `top_k`; every number below
is from the corrected run. The earlier headline (0 / 50 zero-ablation flips, 13 "Dallas-type"
graph over-predictions, a genuine multi-hop over-prediction) was an artifact of the bug and is
withdrawn. Two independent reimplementations of the corrected native arm agree with these
numbers (15 / 50 zero-ablation flips in each).

## Adapter validation (real fp32 weights)

stepped-vs-native max-abs logit Δ **1.6e-5**, 16/16 greedy exact, mid-layer cut **fresh-process
replay exact**. Residency (fp32, CUDA): `streamed` **== `resident` bitwise** (max logit Δ
**0.0**, logits bit-identical), 16-token greedy decode equals `model.generate` for both,
mid-layer cut fresh-process replay exact; **peak VRAM 2.4 GB streamed vs 10.6 GB resident**.

## Single-feature edges (1000 = 50 prompts × top-20)

Zero-ablating one transcoder feature's decoder contribution at its clean activation. Each edge
gets a neutral native-necessity label *and* a comparison to circuit-tracer's own predicted drop
(frozen-attention mode).

- **13 / 1000** edges are individually `native_necessary` (873 `native_effect_absent`, 114
  `inconclusive`); all 13 are on multi-hop (6) and translation (7) prompts.
- Graph agreement: **846 / 1000 agree** (rate **0.846**, 95% CI **[0.824, 0.868]**), **1 invert**
  (0.001 [0.000, 0.003]), 153 mixed (0.153 [0.131, 0.175]). Agreement is mostly `agree_small`:
  the replacement model also predicts a small single-feature drop.

Per family (bootstrap 95% CI over edges):

| family | prompts | edges | agree (95% CI) | invert | mixed |
| --- | --- | --- | --- | --- | --- |
| acronym_abbreviation | 8 | 160 | 0.994 [0.981, 1.000] | 0 | 1 |
| capitals | 8 | 160 | 0.963 [0.931, 0.988] | 0 | 6 |
| arithmetic_sequence | 8 | 160 | 0.925 [0.881, 0.963] | 0 | 12 |
| category_rhyme | 5 | 100 | 0.910 [0.850, 0.960] | 0 | 9 |
| antonyms | 6 | 120 | 0.875 [0.817, 0.933] | 0 | 15 |
| translation | 7 | 140 | 0.686 [0.607, 0.757] | 1 | 43 |
| multi_hop_factual | 8 | 160 | 0.581 [0.506, 0.656] | 0 | 67 |

Agreement is high on direct-recall families and lowest on the compositional ones (multi-hop,
translation), where the disagreements are almost all `mixed` rather than inversions.

## Group interventions (paper-style)

Over 50 prompts (bootstrap 95% CI over prompts), graph prediction in the frozen-attention mode:

| group | agree (95% CI) | agree_large | agree_small | invert | mixed | flips native top-1 |
| --- | --- | :-: | :-: | :-: | :-: | :-: |
| −2× steer (`m = -2`) | 0.96 [0.90, 1.00] | 47 | 1 | 0 | 2 | **45 / 50** |
| zero-ablate (`m = 0`) | 0.56 [0.42, 0.70] | 16 | 12 | 1 | 21 | **15 / 50** |

Steering the whole top-20 set to −2× (the multiplier circuit-tracer's tutorial applies to
supernodes) flips the native top-1 on **45 of 50** prompts; graph and native both show a large
drop on 47.
Zero-ablating the same 20 features flips **15 of 50**: every multi-hop prompt (8 / 8), 5 / 7
translation and 2 / 8 acronym prompts, and none of the capitals, antonyms, arithmetic, or
category-rhyme prompts.

**Is the graph's set special, or would any 20 features do?** A separate run on the same panel
(same rebuilt graphs, top-20 identical to these bundles on all 50 prompts) compared the graph's
top-20 with a matched-random set of 20 active features from the same prompt, matched on layer,
position, and activation band and excluding the graph's picks. Mean native drop, graph vs random:
zero-ablation 1.93 [0.93, 3.27] vs 0.09 [0.04, 0.15] nats (flips 15 vs 5 of 50); −2× steer 18.51
[14.72, 22.46] vs 1.42 [0.99, 1.90] nats (flips 45 vs 20 of 50). The graph's selection is
specifically causal; part of the raw −2× flip count (20 / 50 for random) is generic steering
damage. Jobs `job-7c5949fd0d09` (smoke), `job-a3cacc860c09`, `job-693a029d160e`,
`job-fc1b97f31e10`, `job-20c58a384add`; this control is not part of the sealed bundles.

**Where the graph is wrong, it under-predicts.** The only inversion is `tr_es_dog` (native
−0.004 nats against 1.53 frozen-attention), and it is a prediction-mode artifact: unconstrained
propagation predicts 0.17. Everywhere else the graph's errors run the other way. The native drop
exceeds the frozen-attention prediction by more than 0.4 nats on 11 prompts for zero-ablation
and 12 for the −2× steer. On every multi-hop prompt the real model loses more than the graph
predicts (`mh_houston` native 12.20 vs 5.21 frozen / 4.62 unconstrained; `mh_seattle` 6.95 vs
4.39 / 3.75); `ac_uk` loses 26.22 nats against a 0.17 prediction.

Mean |graph_pred − native| (nats) under each of circuit-tracer's intervention modes:

| group | frozen-attention | unconstrained | direct-effects |
| --- | :-: | :-: | :-: |
| zero-ablate | 1.292 (median 0.182; 6/50 over-predict > 0.4) | 1.274 (0.133; 0/50) | 1.820 (0.503; 17/50) |
| −2× steer | 2.955 (1.786; 26/50) | 2.524 (1.951; 15/50) | 7.330 (5.346; 8/50) |

Unconstrained propagation is the most accurate mode and never over-predicts zero-ablation by
more than 0.4 nats; the fully linearized direct-effects regime, in which the graph's edges are
computed, is the least accurate.

## Error nodes

Transcoder **error** nodes, the MLP output the transcoders do not reconstruct as a feature
direction, carry **11.4 % – 18.5 %** of node influence (median 14.0 %, mean 14.2 % over the 50
admitted prompts). They have no direction to write as a native Act, so they are reported
`uncovered`, never silently dropped. (Influence is computed from the graph alone and was not
affected by the dosing bug.)

## Jobs

Corrected run on an RTX 4080 host: `job-5cf546c13cad` (smoke on two prompts + residency
validation of record, 213 s), `job-a356191720e2` (prompts 0–17, 719 s), `job-e3031f8a2944`
(18–36, 582 s), `job-d3c6b4221dc5` (37–55, 563 s). Phase-A peak 9.0 GB VRAM; phase-B fp32 on
CPU. The withdrawn first run was `job-01a88f477ae1`, `job-d449b03eb717`, `job-b7cbdb109dc5`,
`job-781c923e71d9`, `job-aa81c5787522`; its adapter and residency validation are unaffected.

## Reproduce

The committed bundles re-derive with plain Python, no model:

```python
from saturn_pub.interop.circuit_tracer import verify_edge_bundle

assert verify_edge_bundle("experiments/circuit_tracer/bundles/mh_dallas")["ok"]
```

To recompute from scratch you need a circuit-tracer environment (transformers ≤ 4.57.3 + nnsight)
and the Gemma-2-2B + Gemma Scope transcoder weights (see
[`docs/circuit-tracer.md`](../../docs/circuit-tracer.md)). The private runner (graph build +
native test + sealing + residency check) is not part of this package; it stages `saturn_pub`
from this checkout into the circuit-tracer environment and runs the phases as separate processes
so the two models never co-reside.
