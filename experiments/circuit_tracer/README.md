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
≤11-minute jobs:

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

## Adapter validation (real fp32 weights)

stepped-vs-native max-abs logit Δ **4.5e-5**, 16/16 greedy exact, mid-layer cut **fresh-process
replay exact**. Residency (fp32, CUDA): `streamed` **== `resident` bitwise** (max logit Δ
**0.0**, logits bit-identical), 16-token greedy decode equals `model.generate` for both,
mid-layer cut fresh-process replay exact; **peak VRAM 2.4 GB streamed vs 10.6 GB resident**.

## Single-feature edges (1000 = 50 prompts × top-20)

Zero-ablating one transcoder feature's decoder contribution. Each edge gets a neutral
native-necessity label *and* a comparison to circuit-tracer's own predicted drop
(frozen-attention mode).

- **0 / 1000** edges were individually `native_necessary` (985 `native_effect_absent`, 15
  `inconclusive`) — *expected* under a redundant circuit, and **not** by itself evidence against
  the graph.
- The fair test, graph agreement: **857 / 1000 agree** (rate **0.857**, 95% CI **[0.835,
  0.879]**), **16 invert** (0.016 [0.009, 0.024]), 127 mixed (0.127 [0.106, 0.148]). Agreement is
  overwhelmingly `agree_small` (the replacement model *also* predicts a small single-feature drop).

Per family (bootstrap 95% CI over edges):

| family | prompts | edges | agree (95% CI) | invert | mixed |
| --- | --- | --- | --- | --- | --- |
| acronym_abbreviation | 8 | 160 | 0.994 [0.981, 1.000] | 0 | 1 |
| capitals | 8 | 160 | 0.969 [0.938, 0.994] | 0 | 5 |
| arithmetic_sequence | 8 | 160 | 0.938 [0.900, 0.975] | 0 | 10 |
| category_rhyme | 5 | 100 | 0.920 [0.860, 0.970] | 0 | 8 |
| antonyms | 6 | 120 | 0.883 [0.825, 0.933] | 0 | 14 |
| translation | 7 | 140 | 0.664 [0.586, 0.736] | 10 | 37 |
| multi_hop_factual | 8 | 160 | 0.637 [0.562, 0.713] | 6 | 52 |

Single-feature agreement is high on direct-recall families and lowest on the compositional
(multi-hop, translation) ones — where the graph more often predicts a single-feature effect the
real model does not show.

## Group interventions (paper-style) and the Dallas-type gap

Over 50 prompts (bootstrap 95% CI over prompts), graph prediction in the frozen-attention mode:

| group | agree (95% CI) | agree_large | agree_small | invert | mixed | flips native top-1 |
| --- | --- | :-: | :-: | :-: | :-: | :-: |
| −2× steer (`m = -2`) | 0.50 [0.36, 0.64] | 15 | 10 | 4 | 21 | **12 / 50** |
| zero-ablate (`m = 0`) | 0.34 [0.22, 0.48] | 1 | 16 | 13 | 20 | 0 |

Steering the whole top-20 set to −2× — the intervention the attribution-graphs paper uses —
moves the real model hard and **flips the native top-1 on 12 of 50 prompts**, with the graph
agreeing (`agree_large`) on 15. Joint zero-ablation rarely moves the real model (0 flips) yet the
graph predicts a large drop on **13** prompts (the "Dallas-type" overprediction at scale).

**Which prediction mode, and does the inversion survive it?** circuit-tracer's own intervention
demos call `feature_intervention` with its defaults (`freeze_attention=True,
constrained_layers=None`) — the frozen-attention mode, which is what the `graph_vs_native` tags
above use. Recomputing every group prediction under all three modes (mean |graph_pred − native|,
nats):

| group | frozen-attention | unconstrained | direct-effects |
| --- | :-: | :-: | :-: |
| zero-ablate | 1.086 (median 0.183; 19/50 over-predict > 0.4) | 0.734 (0.135; 15/50) | 1.673 (0.681; 33/50) |
| −2× steer | 2.309 (0.848; 28/50) | 1.539 (0.486; 25/50) | 2.487 (1.197; 36/50) |

The verdict is **mixed, and it depends on the family**:

- The **multi-hop** zero-ablate inversions are a genuine *graph* overprediction — they persist
  under unconstrained propagation (e.g. `mh_houston` native 0.03 vs 5.21 frozen / 4.62
  unconstrained; `mh_seattle` 0.09 vs 4.39 / 3.75).
- The **acronym** and **translation** inversions are largely a *prediction-mode* artifact — under
  unconstrained propagation they fall toward native (`ac_dna` 0.57 → 0.17; `ac_phd` 0.59 → 0.18,
  both below the 0.5 "present" line; translation `tr_es_dog` 1.53 → 0.17).
- **direct-effects** (the fully linearized regime the graph *edges* are computed in) over-predicts
  the most (33/50 and 36/50 over 0.4 nats).

So "the graph over-predicts the joint effect" is true under the paper-matching frozen-attention
setting, and for multi-hop prompts it is a property of the graph, not the prediction mode.

## Error nodes

Transcoder **error** nodes — the MLP output the transcoders do not reconstruct as a feature
direction — carry **11.4 % – 18.5 %** of node influence (median 14.0 %, mean 14.2 % over the 50
admitted prompts). They have no direction to write as a native Act, so they are reported
`uncovered`, never silently dropped.

## Jobs

Panel split into four ≤11-min chunks plus one residency job on an RTX 4080 host:
`job-01a88f477ae1` (prompts 0–13 + residency, 653 s), `job-d449b03eb717` (14–27, 433 s),
`job-b7cbdb109dc5` (28–41, 451 s), `job-781c923e71d9` (42–55, 408 s), and `job-aa81c5787522`
(residency validation of record). Phase-A peak 8.1 GB VRAM; phase-B fp32 on CPU.

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
