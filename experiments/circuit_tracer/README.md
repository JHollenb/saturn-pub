# circuit-tracer native edge test on Gemma-2-2B

This directory holds the committed, scrubbed verdict tables from running
`saturn_pub.interop.circuit_tracer.native_edge_test` on real `google/gemma-2-2b` weights with
the Gemma Scope transcoders (circuit-tracer's `"gemma"` preset). Each `bundles/<prompt>/`
holds a hash-pinned `edge_verdicts.json` + `manifest.json` that re-derives **offline** with
`saturn_pub.interop.circuit_tracer.verify_edge_bundle` — no GPU, model, or circuit-tracer
needed. `summary.json` collects the headline numbers and the job ids.

The notebook [`notebooks/05_circuit_tracer_native_edges.ipynb`](../../notebooks/05_circuit_tracer_native_edges.ipynb)
loads these bundles, shows each table, and re-derives the verdicts in-process.

## What ran

Two phases on one GPU host, one model resident at a time:

1. **Attribution graph (circuit-tracer, bf16).** `ReplacementModel.from_pretrained(
   "google/gemma-2-2b", "gemma", backend="nnsight")` builds an attribution graph targeting the
   answer token; feature nodes are ranked by `compute_node_influence` and the top-k selected.
   Each selected feature's transcoder decoder vector + activation are extracted. circuit-tracer
   runs in its own environment (transformers 4.57.3 + nnsight), which the `saturn-pub[ar]` pin
   (transformers 5.14.1) conflicts with — hence the two-environment split.
2. **Native edge test (saturn-pub, fp32).** The real Gemma-2-2B weights are wrapped with the
   native Gemma-2 `DecoderAdapter` (fp32, tf32 off, eager, batch one, greedy) and validated
   (stepped == native logits, 16-token greedy == native, mid-layer cut fresh-process replay).
   Then `native_edge_test` turns each selected edge into a native intervention and arbitrates
   it against the native consumer under the frozen `default_edge_rule`.

## Prompts

| name | prompt | answer | hops |
| --- | --- | --- | --- |
| `two_hop_dallas` | `Fact: the capital of the state containing Dallas is` | ` Austin` | two (Dallas→Texas→Austin) |
| `one_hop_france` | `The capital of France is` | ` Paris` | one |
| `identity_fox` | `A photorealistic fox sitting in a forest. The animal in this picture is a` | ` fox` | identity |

## Results

Two jobs on an RTX 4080 (16 GB) host: `job-d8761ece96f5` (smoke, 1 prompt) and
`job-2a43a5ecc5a6` (full, 3 prompts, `top_k = 20`, 162 s, phase-A peak 8.3 GB VRAM). The native
Gemma-2 adapter was validated on the real weights before any edge test:

- stepped logits vs native HF eager: max-abs Δ **3.8e-5**
- 16-token greedy continuation: **16/16 exact**, agreement 1.0
- mid-layer `StateCut`: **fresh-process replay exact**
- carrier (`layer:L+1`, last position) vs HF `resid_post[L]`: max-abs **3.1e-5** (two_hop, L20),
  **1.1e-4** (one_hop, L25), **6.1e-5** (identity, L22) — the transcoder decoder direction does
  live at Saturn's carrier.

Each prompt ran 6 edges through `carrier`+`arbitrate` (last position, single write layer, sealed
receipt + replay) and 14 through `residual_hook`. Verdicts under the frozen `default_edge_rule`
(survive ≥ 0.50 nats, collapse ≤ 0.10 nats):

| prompt | native top-1 | edges | survive | collapse | inconcl. | logit Δ range (nats) | error-node influence |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `two_hop_dallas` | ` Austin` (= answer) | 20 | 0 | 20 | 0 | −0.015 … 0.082 | 0.158 (286 nodes) |
| `one_hop_france` | ` a` (≠ ` Paris`) | 20 | 1 | 13 | 6 | −0.501 … 0.616 | 0.128 (156 nodes) |
| `identity_fox` | ` fox` (= answer) | 20 | 0 | 20 | 0 | −0.058 … 0.022 | 0.175 (442 nodes) |

**1 of 60** graph-selected feature edges survived native continuation. On the two prompts the
model answers top-1 (`two_hop_dallas`, `identity_fox`), *every* selected feature collapsed:
removing a single transcoder feature's decoder contribution moved the target log-prob by
`|Δ| ≤ 0.08` nats, far under the survival bar — high attribution weight did not translate into
single-feature native necessity. The lone survivor was `one_hop_france` L24 feature 476 (drop
0.616 nats, no top-token flip); note that prompt's own native top-1 is ` a`, not ` Paris`, so the
graph was attributed to a non-modal token and its edges mostly fall in the inconclusive band.
Across all three graphs, transcoder **error** nodes carried **13–18%** of node influence — the
part of the MLP output the transcoders do not reconstruct as a feature direction, so it has no
direction to test natively and is reported `uncovered`, never silently dropped.

Full headline numbers and the per-edge fingerprints are in
[`summary.json`](summary.json); the sealed tables are under `bundles/<prompt>/`.

## Reproduce

The committed bundles re-derive with plain Python, no model:

```python
from saturn_pub.interop.circuit_tracer import verify_edge_bundle
result = verify_edge_bundle("experiments/circuit_tracer/bundles/two_hop_dallas")
assert result["ok"]
```

To recompute from scratch you need a circuit-tracer environment and the Gemma-2-2B + Gemma
Scope transcoder weights (see [`docs/circuit-tracer.md`](../../docs/circuit-tracer.md)). The
private runner (graph build + native test + sealing) is not part of this package; it stages
`saturn_pub` from this checkout into the circuit-tracer environment and runs the two phases as
separate processes so the two models never co-reside.
