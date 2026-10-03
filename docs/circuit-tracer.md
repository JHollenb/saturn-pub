# circuit-tracer: arbitrate attribution-graph edges against the native consumer

[circuit-tracer](https://github.com/safety-research/circuit-tracer) (Anthropic Fellows /
Decode Research) builds **attribution graphs**: it replaces each MLP with a sparse
transcoder, attributes a target logit to transcoder *feature* nodes (and transcoder *error*
nodes), and reports which nodes influence the answer. A graph edge is a candidate, not a
verdict. `saturn_pub.interop.circuit_tracer` takes such a graph and asks the **unchanged
native consumer** — the real model, run to completion — two honest questions about each edge,
using the `saturn_pub.trial` seam: (1) is removing this feature *individually* load-bearing on
the real model (its **native-necessity** label)? and (2) does the native drop *agree* with the
drop circuit-tracer's own intervention predicts on the replacement model, or does the graph
predict a large effect the real model does not show (an **inversion**)? The second question is
the fair test: a single feature that is not individually necessary is expected under a
redundant circuit and is not, by itself, evidence against the graph.

This is the division of labor from [the SAELens/TransformerLens interop](interop.md), applied
to a harder tool: circuit-tracer is good at *finding* a sparse circuit; Saturn is good at
turning one of its edges into a branchable, replayable intervention on the real weights and
letting the downstream consumer arbitrate it.

```python
from saturn_pub.adapters import load
from saturn_pub.interop.circuit_tracer import native_edge_test

adapter = load("google/gemma-2-2b")  # native Gemma-2 decoder adapter, real weights
table = native_edge_test(
    graph,
    adapter,  # a circuit-tracer Graph + the real model
    prompt_tokens=graph.input_tokens.tolist(),
    target_token=answer_id,
    decoder_writes=decoder_writes,  # (layer, feature) -> [(write_layer, W_dec[feature])]
    top_k=20,
)
print(table.summary())  # native-necessity + graph-agreement + error share
table.seal("bundle/")  # hash-pinned, offline-re-derivable verdict table
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
4. **Native-necessity label.** A neutral label about the *real model only*: `native_necessary`
   (removing this one feature is individually load-bearing), `native_effect_absent` (no
   individual effect — expected under a redundant circuit, and *not* by itself evidence against
   the graph), or `inconclusive`. The default rule:

   ```text
   metric  = consumer_logprob_drop  = logprob(target|clean) − logprob(target|feature removed)
   native_necessary      if drop ≥ 0.50 nats
   native_effect_absent  if drop ≤ 0.10 nats
   inconclusive          otherwise
   ```

5. **Graph-vs-native agreement (the fair test).** When each edge carries
   `graph_predicted_delta` — the drop circuit-tracer's *own* `feature_intervention` predicts on
   the replacement model for the same ablation — each row gets a `graph_vs_native` tag from
   [`classify_graph_vs_native`](../src/saturn_pub/interop/circuit_tracer.py): `agree_small`
   (both drops ≤ 0.10 nats — redundancy, not a graph failure), `agree_large` (both ≥ 0.50),
   `invert` (graph predicts ≥ 0.50 but the native model shows ≤ 0.10 — a genuine disagreement),
   or `mixed`. The worker computes `graph_predicted_delta` while the replacement model is still
   loaded (phase A), so the native test (phase B) needs only the scalars.
6. **Group interventions (paper-style).** `native_group_intervention` jointly ablates the whole
   top-k feature set (`multiplier = 0`) and steers it to −2× its natural activation
   (`multiplier = -2`, matching the attribution-graphs paper's steering), adding
   `(m − 1)·activation·W_dec` for every member at its write layer, and labels the group under
   the same rule with the same `graph_vs_native` comparison to circuit-tracer's joint prediction.
   This is the intervention the paper expects to be strong; a single-feature zero ablation is
   not.

Each row reports the graph weight, the native logit Δ, the graph-predicted Δ, the
`graph_vs_native` tag, the top-token flip, the native-necessity label, and the receipt id.
`top_k ≈ 20` matches the SAE-20 arm of the owner's prior head-to-head.

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
pins. `verify_edge_bundle(dir)` re-checks the hash and re-derives, from the sealed scalars and
the frozen `DecisionRule`, every covered edge's native-necessity label (with the same
`trial.grade` the arbiter uses), its `graph_vs_native` tag, and every group verdict — **stdlib
only, no torch, no model, no circuit_tracer**. The shipped notebook
`notebooks/05_circuit_tracer_native_edges.ipynb` loads the committed, scrubbed bundles, shows the
native-vs-graph columns and the group rows, and re-derives them in-process.

## Real-weight results (Gemma-2-2B, Gemma Scope transcoders)

Measured, `top_k = 20`, **six prompts whose native top-1 is the target** (jobs
`job-52f930db443a` smoke, `job-9d26c9585494` full, 277 s, phase-A peak 8.25 GB VRAM); 120
single-feature edges + 12 group interventions. The native Gemma-2 adapter validated on the real
weights first: stepped-vs-native max-abs logit Δ **3.8e-5**, 16/16 greedy exact, mid-layer cut
**fresh-process replay exact**; carrier↔`resid_post` alignment **3.0e-5 – 7.6e-5**.

**Single features.** 0 of 120 were individually `native_necessary` — *expected* under a
redundant circuit, so not by itself evidence against the graph. The fair test is agreement with
circuit-tracer's own predicted drop: **97 of 120 agree** (overwhelmingly `agree_small`: the
replacement model also predicts a small drop), **1 is a genuine `invert`**, 22 are `mixed`. So
the graph and the real model mostly concur that single features carry little alone — a "collapse"
reading would have overstated it.

**Group interventions (paper-style).** Steering the whole top-k set to −2× activation moves the
real model hard on 2 of 6 prompts and the graph *agrees* there:

| prompt | −2× group steer: native Δ / graph-pred (nats) | tag | flips native top-1 |
| --- | --- | --- | :-: |
| `opposite_hot` → ` cold` | 6.111 / 4.644 | agree_large (native_necessary) | yes |
| `japan_tokyo` → ` Tokyo` | 7.147 / 2.456 | agree_large (native_necessary) | yes |
| `two_hop_dallas` → ` Austin` | 0.483 / 4.424 | mixed | no |
| `jupiter_largest` → ` Jupiter` | −0.216 / 0.831 | **invert** | no |
| `eiffel_paris` / `identity_fox` | ≈0 / small | agree_small / mixed | no |

`jupiter_largest` is the clearest group **inversion** (graph predicts ~0.7–0.8 nats the real
model does not show). Transcoder **error** nodes carried **11–18%** of node influence the feature
circuit never exposes (`uncovered`, never dropped). Per-prompt tables, every job id, and per-edge
numbers are in [`experiments/circuit_tracer/README.md`](../experiments/circuit_tracer/README.md)
and [`experiments/circuit_tracer/summary.json`](../experiments/circuit_tracer/summary.json).

## Reproduce

circuit-tracer's dependencies (transformers ≤ 4.57.3, nnsight) conflict with the
`saturn-pub[ar]` pin (transformers 5.14.1), so it runs in its own environment. The Gemma-2
adapter is version-robust (it reproduces the native forward on either transformers version).
Build the graph in a circuit-tracer environment, then run `native_edge_test` with the real
model wrapped by `saturn_pub.adapters.load`. The committed bundle re-derives with
`verify_edge_bundle` in any environment with `saturn_pub` installed — no GPU, model, or
circuit-tracer required.
