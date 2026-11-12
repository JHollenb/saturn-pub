# Attention port: observe and cut reads per head

`mdb.attention_port` is the single place where experiments observe attention reads and cut them per head, for Qwen-family decoders and FLUX.2 Klein. It replaces three separate implementations and the ad hoc `dispatch_attention_fn` patches in experiment scripts.

## API

```python
from mdb.attention_port import (AttentionPort, FluxBackend, QwenBackend, cut, sham, paired,
                                observe, full_probs, class_flow, kv_group, kv_groups)

port = AttentionPort(FluxBackend(pipe.transformer, text_rows=512, identity=adapter.model_identity),
                     rows={"image": range(512, S), "text": range(512), "pad": pad_rows, ...},
                     zero_method="delta")          # or "recompute"; precision="legacy" for FLUX reproduction
z = cut(["image", other_text_rows], [own_rows], heads=[(block, head), ...], renormalize=False)
with port.apply(z, observers=[observe(class_flow(onehot, weighted=True))]):
    run_the_model()
with port.apply(sham(z)):                          # same code path, nothing removed
    run_the_model()
```

- `cut(receivers, sources, heads=None, blocks=None, steps=None, *, renormalize)` returns a frozen `Cut`. Receivers are query rows and sources are key rows, both given in key coordinates. For a cached decoder call, query row `i` is key position `Sk - Sq + i`. Rows can be names from the port's `rows` table or explicit integer collections, and the two can be mixed. A `heads` entry is either `head` (that head in every allowed block) or `(block, head)`. `renormalize` has no default, so every caller picks the cut type explicitly.
- `sham(like=None, *, renormalize=None)` builds a cut with nothing removed that runs through the same code path. `sham(c)` recomputes exactly the (step, block, head, receiver) cells that `c` touches, so it gives that cut's floor. `sham(renormalize=...)` routes every head, block, step and row through the cut path, which gives the worst-case floor. `paired(c)` returns `(c, sham(c))`. Run both for every (prompt, seed).
- `port.apply(*cuts, observers=(), rows=None, log=False)` is a context manager that installs the backend patch and removes it on exit. Inside it, `port.calls` counts native attention calls, and `port.log` (when `log=True`) lists the recomputed heads, receiver rows and removed reads for each call. `port.metadata()` returns a JSON-able recipe: backend, model identity, cut specs and observer names.
- Composite cuts: several `Cut`s in one `apply` each keep their own `heads`, `blocks` and `steps`. When cuts overlap in a head, the union of their reads is removed. Mixing the two cut types in one head is refused. The 2026-11-11 `both()` helper dropped `heads=`; `test_composite_cut_honours_each_specs_heads` guards against that bug.
- `observe(reduce, steps=None, blocks=None)` stores `reduce(read)` per (step, block). `read` exposes head-major `q`, `k`, `v` (K/V already expanded to query heads), `out`, `scale`, an additive `bias`, `probs()` (an FP32 softmax cached per call) and `write_norms()`. Two reductions are built in. `full_probs(dtype)` returns [H, Sq, Sk]. `class_flow(source_classes, receiver_classes=None, weighted=False)` returns receiver-class × source-class read mass, or write mass when `weighted=True` (the `flow_p` / `flow_w` of the read-graph `make_capture`). `Observer.stack(key)` returns [steps, blocks, ...]. Observers see this call's native read before this call's own cut is applied.
- `kv_group(head, n_heads, n_kv)` / `kv_groups(n_heads, n_kv)` is the single grouped-query mapping, `h // (n_heads // n_kv)`, tested against HF `repeat_kv`.

Backends:

- `FluxBackend(transformer, text_rows=512, identity=None)` gives a scoped override of `diffusers...transformer_flux2.dispatch_attention_fn`. Calls are numbered in execution order, and `step, block = divmod(index, 5 + 20)` with the joint blocks first. Write norms use `to_add_out` for text receivers and `to_out[0]` for image receivers in joint blocks, and `to_out[:, :inner_dim]` in single blocks.
- `QwenBackend(model, identity=None, module=None)` patches `modeling_<family>.eager_attention_forward` through an `AttentionObserver` subclass, so it shares the observer's patch point and its nesting guard in both directions. The block is the decoder layer. The step increments each time layer 0 runs. When cuts are active, the returned `attn_weights` are the effective weights after the cut. The model must run with eager attention.

Capture records never contain weights. Anything derived from weights is identified by parameter name (`backend.parameter_names(block)`) and by the caller's model identity. This is compatible with the pointer-based replay plan in `REPLAYABLE-CIRCUIT-DEBUGGER-PLAN.md`.

## Two cut types, and why both

| | `renormalize=True` | `renormalize=False` |
|---|---|---|
| operation | boolean mask: the cut reads get score −inf | contribution removed: `out − Σ_cut p·v` |
| other weights | rescaled to sum to 1 | unchanged |
| question it answers | what does the receiver compute without those sources? | what did those reads carry? |
| known artefact | redistributes mass onto the remaining sources | leaves the output with less than unit mass |

The two cuts answer different questions, and each one has its own artefact. In the FLUX read-graph experiment (`saturn/experiments/2026-11-11-flux-read-graph`), the renormalising mask produced object duplication that never appeared under the no-renormalisation cut, in 4 of 4 cases. **A result counts only if both cut types agree.** Run each cut with both types, each paired with its own sham.

## Precision

The native attention call always runs unchanged on every head. Only the (head, receiver row) cells that a cut touches are recomputed in FP32 and written back. Every other head and every other row of a cut head returns the native bytes; the unit tests check this.

- `renormalize=False`, `zero_method="delta"` (the default, chosen on the measured floors below) computes `native_out − (P ⊙ removed) @ V`, with P in FP32 from the call's own q/k. When nothing is removed this is exactly `native_out`, so the sham floor is exactly 0. The only deviation from an exact FP32 cut is the native kernel's own rounding, which every uncut cell carries anyway.
- `renormalize=False`, `zero_method="recompute"` computes `(P ⊙ keep) @ V` in FP32 for the touched rows. Its floor is the gap between FP32 and the native kernel on those rows.
- `renormalize=True` computes the softmax of the masked scores in FP32 for the touched rows. Its floor is the same gap on the same rows. A delta form, `(out − c)/(1 − m)`, would be exact algebraically but amplifies native rounding by `1/(1 − m)` for large cuts, so it is not used.
- `precision="legacy"` (FLUX only) reproduces the 2026-11-11 worker arithmetic. For zero cuts, every head of each touched block is recomputed in FP32 with `/ sqrt(D)`. For renormalising cuts, a [1, H, S, S] boolean `attn_mask` goes through the native kernel. Composite cuts are combined correctly per head.

### Measured floors (FLUX.2 Klein 4B, 256 px, 4 steps, seed 4242)

MEASURED on 2026-11-12 in mrun job `job-345766dba0e8` on Beast (CUDA, BF16, TF32 off, diffusers 0.39.0, torch 2.13.0). The run used `attention_port.py` sha `9849ca89…`, the `saturn_pub` 0.4.1 wheel byte-identical to the reference run's, model identity `a09191c5…`, and peak allocated memory of 7,991 MB. The prompt was "a photorealistic black cat sitting beside a red ball in fresh snow at dawn, soft blue light". The worker and stage script are `qualification/attention_port_smoke_worker.py` and `qualification/stage_attention_port_smoke.py`. Results, including per-arm PNGs, latents, `metrics.json` and `observe.npz`, are at `beast:/mnt/hdd8tb/mdb/research/job-345766dba0e8/attention-port-smoke/`. The mrun record still showed `running` after the worker exited and its lease expired, because the server's job listing returned HTTP 500, so the job has no recorded final state.

In the table, `px MAD` is the mean absolute RGB difference over [0, 1], and `latent rel` is ‖Δlatent‖/‖latent‖ of the final latent.

| arm | compared with | px MAD | latent rel |
|---|---|---|---|
| port installed, no cuts (passthrough) | native | 0 (bit-identical) | 0 |
| observe: `class_flow` p/w plus `tok_p`/`tok_w` | native | 0 (bit-identical) | 0 |
| **new** `sham(renormalize=False)`, all heads/blocks/steps, delta | native | **0 (bit-identical)** | **0** |
| new `sham(renormalize=False)`, all cells, recompute | native | 0.00311 | 0.0395 |
| new `sham(renormalize=True)`, all cells | native | 0.00311 | 0.0395 |
| **old-style** zero sham (legacy, every block in FP32) | native | 0.00311 | 0.0395 |
| **old-style** renormalising sham (legacy, all-true `attn_mask`) | native | 0.00325 | 0.0492 |
| **new** `sham(z:own60)`, delta | iso | **0 (bit-identical)** | **0** |
| new `sham(z:own60)`, recompute | iso | 0.00383 | 0.0491 |
| **old-style** `sham(z:own60)` (legacy: all 24 heads of the 18 touched blocks in FP32, 72 of 100 calls) | iso | 0.00423 | 0.0462 |
| new `sham(m:own60)` | iso | 0.00383 | 0.0491 |
| old-style `sham(m:own60)` (legacy) | iso | 0.00430 | 0.0628 |

The old-style renormalising sham reproduces the read-graph parity value `alltrue_mask` exactly (0.0032543 px, 0.049234 latent).

The read-graph arm `z:own60:ball` used the hybrid "own" conditioning for `ball`, which is the read-graph `hybrid()` logic copied verbatim. Its receivers were the image rows plus the other text rows, its sources were the word's own text row, and it covered the 60 `(block, head)` pairs of `head_sets3.json` `sets["cat:ball"]["own60"]`. Each row below compares an arm with the saved reference image from `job-a0bca63168b4`:

| arm | reference image | px MAD (max) | vs native: px MAD / latent rel (reference value) |
|---|---|---|---|
| native | `native.cat.4242.png` | 0 (identical) | |
| iso (hybrid own, no cut) | `arm.cat.4242.iso_ball.png` | 0 (identical) | 0.037226 / 0.308601 (0.037226 / 0.308601) |
| legacy `z:own60` | `arm.cat.4242.z_own60_ball.png` | **0 (identical)** | 0.057791 / 0.522531 (0.057791 / 0.522531) |
| new delta `z:own60` | same | 0.00251 (0.178) | 0.057338 / 0.519562 |
| new recompute `z:own60` | same | 0.00413 (0.588) | 0.058311 / 0.522301 |
| legacy `m:own60` | `arm.cat.4242.m_own60_ball.png` | **0 (identical)** | 0.049238 / 0.419263 (0.049238 / 0.419264) |
| new `m:own60` | same | 0.00284 (0.072) | 0.049466 / 0.419007 |

The observe reductions `flow_p`/`flow_w` equal the saved `capture.cat.4242.npz` exactly. `tok_p`/`tok_w` are also equal once rounded to the capture's float16 storage.

What this shows (MEASURED, one prompt and one seed):

- Any nonzero perturbation grows over four steps to about 0.04–0.06 latent rel. Restricting FP32 recomputation to the 60 cut heads and their receiver rows does not shrink the floor much compared with recomputing all 24 heads of every touched block: 0.0491 against 0.0462 latent rel for the matched zero sham.
- The delta form is the change that removes the floor. Its matched sham is bit-identical, so every byte that differs under the delta `z:own60` comes from the removed reads.
- The new and legacy `z:own60` images differ from each other by 0.0025 px, less than the old floor of 0.0042, and both differ from iso by about 0.054 px.
- Renormalising cuts still carry an FP32 floor of 0.0038 px / 0.049 latent, against 0.0043 / 0.063 for legacy. Report each renormalising cut together with its `sham(c)`.

Runtime per 4-step branch: 0.34 s native, 0.38 s for the delta `z:own60`, 0.40 s for the legacy version, and 1.0 s for an all-cells delta sham. The `port.log` recompute counters are filled only on the non-legacy path.

## What it supersedes

- `saturn/experiments/2026-11-11-flux-read-graph/read_graph_worker.py`: `Port`, `make_capture` (`flow_p`, `flow_w`, `tok_*`), `cut_mask`, `both` and the `headcut` arm block. These map to `FluxBackend`, `class_flow` / custom reductions, `cut` and multiple cuts in `apply`. That worker is unchanged.
- The `dispatch_attention_fn` patches in about ten `qualification/flux_*.py` scripts. Those scripts are bound by qualification hashes and stay byte-identical. New FLUX work should use the port.
- `mdb.attention_observer.AttentionObserver` still exists. Its behaviour is unchanged, and it gained an `_intercept` hook plus an optional `module=` argument for Qwen families other than qwen2. `QwenBackend` is built on it. Use `full_probs` or a custom reduction rather than retaining full Q/K/V clones.
- `mdb.native_discovery_capture` (`output_attentions` plus a `v_proj` hook) is unchanged. New captures should use `observe(full_probs())` under `QwenBackend`.
- Grouped-query mapping: `symbol_attention.attach_attention` now calls `kv_group`. The copies in `qualification/value_coalition_worker.py`, `attention_allocation_audit.py`, `frozen_value_pair_audit.py` and `resolver_execution_audit.py` are hash-bound and keep their inline formula, which is the same mapping.

## Tests

`tests/test_attention_port.py` covers tiny CPU tensors and a randomly initialised two-layer Qwen2:

- hand-computed cuts of both types, checked against `attention_partition.dense_attention`;
- per-head isolation, bit for bit;
- a delta sham that is exactly identity, and an FP32 sham whose error stays within rounding;
- composite cuts that respect each spec's heads;
- block, step and (block, head) selection;
- refusal of cuts that leave a receiver with no source;
- exact reproduction of the legacy arithmetic;
- observer reductions;
- Qwen observe, sham and cut through returned attentions and causal logits;
- the shared nesting guard;
- `kv_group` against `repeat_kv`.

## Known gaps

- FLUX coordinates come from call order (`divmod`). An unexpected extra call raises an error only when the caller checks `port.calls`, which the smoke worker does.
- Qwen steps assume layer 0 runs once per forward. The Qwen backend has been tested only on a tiny random model, not on a pretrained lane, and its FP32 floor there has not been measured.
- A sham through the FP32 path has a nonzero floor. Report each cut's own sham with it.
- A panel generator, which would enumerate cut × type × sham per (prompt, seed) and record the both-types-agree rule, is not built yet.
