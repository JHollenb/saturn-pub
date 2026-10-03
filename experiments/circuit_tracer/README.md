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

1. **Attribution graph + graph-side predictions (circuit-tracer, bf16).**
   `ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="nnsight")` builds an
   attribution graph targeting the answer token; feature nodes are ranked by
   `compute_node_influence` and the top-k selected. For each selected feature we then run
   circuit-tracer's **own** `feature_intervention` on the replacement model (ablate the feature
   to zero) and record its predicted target-logprob drop — plus two group predictions (ablate
   the whole top-k set jointly, and steer it to −2× activation). The transcoder decoder vector +
   activation are extracted. circuit-tracer runs in its own environment (transformers 4.57.3 +
   nnsight), which the `saturn-pub[ar]` pin (transformers 5.14.1) conflicts with.
2. **Native test (saturn-pub, fp32).** The real Gemma-2-2B weights are wrapped with the native
   Gemma-2 `DecoderAdapter` (fp32, tf32 off, eager, batch one, greedy) and validated (stepped ==
   native logits, 16-token greedy == native, mid-layer cut fresh-process replay). `native_edge_test`
   then performs the *same* interventions on the real model and, for each, records the native
   drop, its neutral **native-necessity** label, and the **graph_vs_native** comparison to the
   replacement model's predicted drop. `native_group_intervention` does the joint ablate / −2×
   steer. Only prompts whose native top-1 is the target are tested; others are reported dropped.

## Prompts

Only prompts whose native top-1 is the target are included in the edge test (a non-top-1 target
makes every reading ambiguous). The candidate set (phase B drops any whose top-1 is not the
target, and the drop is reported):

| name | prompt | answer |
| --- | --- | --- |
| `two_hop_dallas` | `Fact: the capital of the state containing Dallas is` | ` Austin` |
| `identity_fox` | `A photorealistic fox sitting in a forest. The animal in this picture is a` | ` fox` |
| `eiffel_paris` | `The Eiffel Tower is located in the city of` | ` Paris` |
| `opposite_hot` | `The opposite of hot is` | ` cold` |
| `japan_tokyo` | `The capital of Japan is the city of` | ` Tokyo` |
| `jupiter_largest` | `The largest planet in the solar system is` | ` Jupiter` |

(The earlier `one_hop_france` = "The capital of France is" was dropped: Gemma-2-2B's native
top-1 there is ` a`, not ` Paris`.)

## Results

Two jobs on an RTX 4080 (16 GB) host: `job-52f930db443a` (smoke, 1 prompt) and
`job-9d26c9585494` (full, 6 prompts, `top_k = 20`, 277 s, phase-A peak 8.25 GB VRAM). All six
candidate prompts had their native top-1 equal to the target, so none were dropped. Adapter
validation on the real weights: stepped-vs-native max-abs Δ **3.8e-5**, 16/16 greedy exact,
mid-layer cut **fresh-process replay exact**; carrier↔`resid_post` alignment **3.0e-5 – 7.6e-5**
across the six prompts.

### Single-feature edges (120 total: 20 per prompt)

Removing one transcoder feature's decoder contribution to zero. Each edge gets a neutral
native-necessity label *and* a comparison to circuit-tracer's own predicted drop.

| prompt | native top-1 | necessary / absent / inconcl. | agree / invert / mixed vs graph | native Δ range | graph-pred Δ range | error share |
| --- | --- | --- | --- | --- | --- | --- |
| `two_hop_dallas` | ` Austin` | 0 / 20 / 0 | 10 / 1 / 9 | −0.015 … 0.082 | −0.058 … 0.635 | 0.158 (286) |
| `identity_fox` | ` fox` | 0 / 20 / 0 | 18 / 0 / 2 | −0.058 … 0.022 | −0.181 … 0.168 | 0.175 (442) |
| `eiffel_paris` | ` Paris` | 0 / 20 / 0 | 20 / 0 / 0 | −0.002 … 0.001 | −0.021 … 0.036 | 0.167 (260) |
| `opposite_hot` | ` cold` | 0 / 19 / 1 | 17 / 0 / 3 | −0.124 … 0.149 | −0.316 … 0.106 | 0.114 (156) |
| `japan_tokyo` | ` Tokyo` | 0 / 20 / 0 | 20 / 0 / 0 | −0.006 … 0.012 | −0.071 … 0.055 | 0.123 (234) |
| `jupiter_largest` | ` Jupiter` | 0 / 20 / 0 | 12 / 0 / 8 | −0.021 … 0.011 | −0.162 … 0.314 | 0.144 (234) |
| **total** | | **0 / 119 / 1** | **97 / 1 / 22** | | | 11–18% |

No single feature was individually load-bearing (0/120 `native_necessary`) — **expected under a
redundant circuit, and not evidence against the graph**. The fair comparison is the last
columns: **97 of 120** edges *agree* with the graph (overwhelmingly `agree_small` — the
replacement model *also* predicts a small drop), only **1** is a genuine `invert` (the graph
predicts a large effect the real model does not show), and 22 are `mixed`. This is the
correction to a naive "collapse" reading: the graph and the real model mostly concur that
single features carry little alone.

### Group interventions (paper-style: jointly ablate / steer the top-k set)

`multiplier = 0` ablates the whole top-20 set; `multiplier = -2` steers it to −2× activation.

| prompt | ablate: native Δ / graph-pred → tag | steer −2×: native Δ / graph-pred → tag | steer flips top-1? |
| --- | --- | --- | --- |
| `two_hop_dallas` | 0.126 / 3.371 → mixed | 0.483 / 4.424 → mixed | no |
| `identity_fox` | −0.039 / 0.080 → agree_small | −0.110 / 0.007 → agree_small | no |
| `eiffel_paris` | −0.001 / 0.080 → agree_small | 0.000 / 0.129 → mixed | no |
| `opposite_hot` | −0.114 / 0.196 → mixed | **6.111 / 4.644 → agree_large (native_necessary)** | **yes** |
| `japan_tokyo` | 0.016 / 0.063 → agree_small | **7.147 / 2.456 → agree_large (native_necessary)** | **yes** |
| `jupiter_largest` | −0.077 / 0.658 → **invert** | −0.216 / 0.831 → **invert** | no |

Steering the whole top-k set to −2× — the intervention the attribution-graphs paper uses —
moves the real model hard on 2 of 6 prompts (**6.1 and 7.1 nats, flipping the native top-1**),
and there the graph *agrees* (`agree_large`). `jupiter_largest` is the clearest **group
inversion**: circuit-tracer predicts a 0.66–0.83-nat drop that the real model does not show.
Across all six graphs, transcoder **error** nodes carry **11–18%** of node influence — influence
the feature circuit never exposes, reported `uncovered`, never silently dropped.

Full per-edge numbers and fingerprints are in [`summary.json`](summary.json); the sealed tables
are under `bundles/<prompt>/`.

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
