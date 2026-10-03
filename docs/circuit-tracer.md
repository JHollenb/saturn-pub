# circuit-tracer: arbitrate attribution-graph edges against the native consumer

[circuit-tracer](https://github.com/safety-research/circuit-tracer) (Anthropic Fellows /
Decode Research) builds **attribution graphs**: it replaces each MLP with a sparse
transcoder, attributes a target logit to transcoder *feature* nodes (and transcoder *error*
nodes), and reports which nodes influence the answer. A graph edge is a candidate, not a
verdict. `saturn_pub.interop.circuit_tracer` takes such a graph and lets the **unchanged
native consumer** — the real model, run to completion — decide whether each edge *survives*
or *collapses*, using the `saturn_pub.trial` seam.

This is the division of labor from [the SAELens/TransformerLens interop](interop.md), applied
to a harder tool: circuit-tracer is good at *finding* a sparse circuit; Saturn is good at
turning one of its edges into a branchable, replayable intervention on the real weights and
letting the downstream consumer arbitrate it.

```python
from saturn_pub.adapters import load
from saturn_pub.interop.circuit_tracer import native_edge_test

adapter = load("google/gemma-2-2b")          # native Gemma-2 decoder adapter, real weights
table = native_edge_test(
    graph, adapter,                          # a circuit-tracer Graph + the real model
    prompt_tokens=graph.input_tokens.tolist(),
    target_token=answer_id,
    decoder_writes=decoder_writes,           # (layer, feature) -> [(write_layer, W_dec[feature])]
    top_k=20,
)
print(table.summary())                       # surviving / collapsing / error-node share
table.seal("bundle/")                        # hash-pinned, offline-re-derivable verdict table
```

`circuit_tracer` is an **optional** dependency, imported lazily. Importing this module needs
only torch (for the native path); the offline tests and the shipped notebook drive it with a
synthetic graph and re-derive the sealed table with the stdlib-only `verify_edge_bundle`.

## What the native edge test does

1. **Rank and select.** Feature nodes are ranked by influence on the target logit with
   circuit-tracer's own `compute_node_influence` (reimplemented so the module needs no
   `circuit_tracer` import), and the top-k are selected. Transcoder **error** nodes are
   summarised, never dropped.
2. **Map each feature to its write point.** A per-layer transcoder feature at layer `L`
   writes `activation · W_dec[feature]` into the residual stream at `hook_mlp_out` of layer
   `L`, which flows into `resid_post[L]`. That residual point **is** Saturn's `layer:L+1`
   carrier (the input to layer `L+1`). This alignment is measured, not assumed (below).
3. **Re-test natively.** Each feature becomes a
   `Reading.from_external("circuit-tracer", …, asserts_effect=True)` — the graph asserts the
   edge is load-bearing. The edge is graded under a frozen `DecisionRule` by removing the
   feature's decoder contribution on the real model and measuring the native target-logit
   drop:
   - **`carrier`** operator — a last-position per-layer feature goes through a
     `Session` + `Act.add("hidden", −activation·W_dec)` + `trial.arbitrate()`, producing a
     sealed `Receipt` and a cut that replays in a fresh process.
   - **`residual_hook`** operator — a feature at any position is re-tested with a real-model
     forward hook that adds the same contribution at each write layer, graded by the same
     frozen rule. This covers the subject-position features an attribution graph cares about.
4. **Verdict.** `survives` (native consumer agrees the edge is load-bearing), `collapses`
   (native consumer shows no effect where the graph read one), or `inconclusive` (ambiguity
   band, a declined reading, or a failed control). The default rule:

   ```text
   metric  = consumer_logprob_drop  = logprob(target|clean) − logprob(target|feature removed)
   survives  (agree)  if drop ≥ 0.50 nats
   collapses (invert) if drop ≤ 0.10 nats
   inconclusive       otherwise
   ```

Each row reports the graph weight, the native logit Δ, the top-token flip, the verdict, and
the receipt id. `top_k ≈ 20` matches the SAE-20 arm of the owner's prior head-to-head.

## The address alignment is measured, not assumed

Like the SAE example's `resid_pre[L]` check, the transcoder write point is verified
numerically. Saturn's `layer:L+1` carrier (last position) equals the HF residual at the
output of layer `L` (= `resid_post[L]`, the transcoder's `feature_output_hook = hook_mlp_out`
destination). The offline test checks this against HF's own kernels; the real-weight run
reports the max-abs difference in fp32. The transcoder decoder direction `W_dec[feature]`
lives in that residual space, so subtracting `activation · W_dec[feature]` at the carrier
removes exactly the contribution the transcoder attributes to the feature.

## Supported feature types

| transcoder | write layers | native test |
| --- | --- | --- |
| per-layer (PLT; the Gemma Scope `"gemma"` preset) | one (the feature's layer) | full — `carrier`+`arbitrate` at the last position, `residual_hook` at any position |
| cross-layer (CLT) | several (the feature writes to layers `L…`) | mapped to its write-layer set and tested with `residual_hook` summing the per-layer decoder slices; the single-boundary `carrier`+`arbitrate` path is used only for one-layer writes |

`decoder_writes(layer, feature)` returns `[(write_layer, direction), …]` — one pair for a
per-layer transcoder, one per target layer for a cross-layer transcoder. A feature whose
write layer is outside the model's layer range is reported uncovered.

## Error nodes are reported, never dropped

A transcoder error node is the part of the MLP output the transcoder does **not**
reconstruct as a feature direction, so it has no direction to write as a native Act. The
table reports the error nodes as a single `uncovered` row carrying their count and share of
node influence. In the owner's prior measurement, roughly a third of end-to-end influence ran
through error nodes — influence the feature circuit simply does not expose. Reporting it is
part of being honest about what the graph can and cannot carry.

## The sealed verdict table re-derives offline

`EdgeVerdictTable.seal(dir)` writes `edge_verdicts.json` + a `manifest.json` of SHA-256
pins. `verify_edge_bundle(dir)` re-checks the hash and re-derives every covered verdict from
the sealed `native_logit_delta` scalar and the frozen `DecisionRule` with the same
`trial.grade` the arbiter uses — **stdlib only, no torch, no model, no circuit_tracer**. The
shipped notebook `notebooks/05_circuit_tracer_native_edges.ipynb` loads a committed, scrubbed
bundle, shows the table, and re-derives it in-process.

## Real-weight results (Gemma-2-2B, Gemma Scope transcoders)

Measured, `top_k = 20`, three prompts, 60 feature edges total (jobs `job-d8761ece96f5` smoke,
`job-2a43a5ecc5a6` full, 162 s, phase-A peak 8.3 GB VRAM). The native Gemma-2 adapter validated
on the real weights first: stepped-vs-native max-abs logit Δ **3.8e-5**, 16/16 greedy tokens
exact, mid-layer cut **fresh-process replay exact**. The carrier↔`resid_post` address alignment
measured **3.1e-5 / 1.1e-4 / 6.1e-5** across the three prompts. Each prompt ran 6 edges through
`carrier`+`arbitrate` and 14 through `residual_hook`.

| prompt | native top-1 | edges | survive | collapse | inconcl. | error-node influence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `two_hop_dallas` → ` Austin` | ` Austin` (= answer) | 20 | 0 | 20 | 0 | 0.158 (286 nodes) |
| `one_hop_france` → ` Paris` | ` a` (≠ answer) | 20 | 1 | 13 | 6 | 0.128 (156 nodes) |
| `identity_fox` → ` fox` | ` fox` (= answer) | 20 | 0 | 20 | 0 | 0.175 (442 nodes) |

Honest headline: **1 of 60** graph-selected edges survived the 0.50-nat bar. Removing one
transcoder feature's decoder contribution moved the native target log-prob by `|Δ| ≤ 0.08` nats
on the two prompts the model answers top-1, so no single selected feature is individually
necessary there — high attribution weight did not imply native causal necessity. The lone
survivor was `one_hop_france` L24 feat 476 (drop 0.616 nats); that prompt's own native top-1 is
` a`, not ` Paris`, so the graph targeted a non-modal token and most of its edges land in the
inconclusive band. Error nodes carried **13–18%** of node influence the feature circuit never
exposes. The verdict tables, job ids, and per-edge numbers are in
[`experiments/circuit_tracer/README.md`](../experiments/circuit_tracer/README.md) and
[`experiments/circuit_tracer/summary.json`](../experiments/circuit_tracer/summary.json).

## Reproduce

circuit-tracer's dependencies (transformers ≤ 4.57.3, nnsight) conflict with the
`saturn-pub[ar]` pin (transformers 5.14.1), so it runs in its own environment. The Gemma-2
adapter is version-robust (it reproduces the native forward on either transformers version).
Build the graph in a circuit-tracer environment, then run `native_edge_test` with the real
model wrapped by `saturn_pub.adapters.load`. The committed bundle re-derives with
`verify_edge_bundle` in any environment with `saturn_pub` installed — no GPU, model, or
circuit-tracer required.
