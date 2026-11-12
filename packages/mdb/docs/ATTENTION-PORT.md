# Attention port: observe and cut reads per head

`mdb.attention_port` is the single place where experiments observe attention reads and cut them per head, for Qwen-family decoders and FLUX.2 Klein. It replaces three separate implementations and the ad hoc `dispatch_attention_fn` patches in experiment scripts.

## API

```python
from mdb.attention_port import (AttentionPort, FluxBackend, QwenBackend, cut, sham, paired, floor_arms,
                                observe, full_probs, class_flow, kv_group, kv_groups)

port = AttentionPort(FluxBackend(pipe.transformer, text_rows=512, identity=adapter.model_identity),
                     rows={"image": range(512, S), "text": range(512), "pad": pad_rows, ...},
                     zero_method="delta")          # or "recompute"; precision="legacy" for FLUX reproduction
z = cut(["image", other_text_rows], [own_rows], heads=[(block, head), ...], renormalize=False)
with port.apply(z, observers=[observe(class_flow(onehot, weighted=True))], forwards=STEPS):
    run_the_model()                                # exactly STEPS transformer forwards, batch 1
for name, arm in floor_arms(z).items():            # the floor of z (see Precision)
    with port.apply(arm, forwards=STEPS):
        run_the_model()
```

- `cut(receivers, sources, heads=None, blocks=None, steps=None, *, renormalize)` returns a frozen `Cut`. Receivers are query rows and sources are key rows, both given in key coordinates. For a cached decoder call (Qwen only, because FLUX refuses Sq ≠ Sk), query row `i` is key position `Sk - Sq + i`. Rows can be names from the port's `rows` table or explicit integer collections, and the two can be mixed. A `heads` entry is either `head` (that head in every allowed block) or `(block, head)`. `renormalize` has no default, so every caller picks the cut type explicitly.
  `cut(..., zero_method=None)` overrides the port's `zero_method` for one no-renormalisation cut (`None` uses the port's).
- `sham(like=None, *, renormalize=None, zero_method=None)` builds a cut with nothing removed that runs through the same code path. `sham(c)` routes exactly the (step, block, head, receiver) cells that `c` touches. `sham(renormalize=...)` routes every head, block, step and row through the cut path. `paired(c)` returns `(c, sham(c))`. **A delta sham is exactly native by construction and is not a floor.** Only an FP32-recompute sham or a renormalising sham measures arithmetic noise. Use `floor_arms(c)`.
- `floor_arms(c)` returns the named arms that measure the arithmetic floor of a cut `c`; run each one next to `c` for the same (prompt, seed). For `renormalize=False` it returns `sham_recompute` (the recompute sham of exactly `c`'s cells; compare with the base arm) and `cut_recompute` (`c` with `zero_method="recompute"`; compare with `c`). The floor of `c` is the larger of the two distances. For `renormalize=True` it returns `sham` (the matched FP32 sham). A panel generator should record both numbers per cut.
- `AttentionPort(backend, *, rows=None, zero_method="delta", precision="native", empty="raise")`. `empty` sets what a renormalising cut does to a (head, receiver) that loses every source it can see. `"raise"` (the default) refuses with the cells listed. `"zero"` writes a zero read for those cells and records each one in `port.emptied` as `(step, block, head, receiver_query_row)`, also in `metadata()["empty"]`.
- `port.apply(*cuts, observers=(), rows=None, log=False, forwards=None)` is a context manager that installs the backend patch and removes it on exit. `forwards` is the number of model forwards inside the block: transformer forwards (= denoising steps) for FLUX, where it is **required** whenever cuts or observers are given; the number of times layer 0 runs for Qwen, where it is optional. Inside the block, `port.calls` counts native attention calls and `port.forward_count` counts forwards; both, plus `port.log` and `port.emptied`, reset on every `with`. `port.log` (when `log=True`) lists, for each call, the recomputed heads, receiver rows, removed reads, cut kind, emptied cells and natively empty receivers. `port.metadata()` returns a JSON-able recipe: backend, model identity, `zero_method`, `precision`, `empty`, `forwards`, cut specs (including each cut's `zero_method`) and observer names.
- On a clean exit, `apply` raises if cuts or observers were supplied but no attention call was intercepted, if the forward count differs from `forwards`, if any cut never matched a call (wrong `steps`/`blocks`/`heads`), or if any observer never fired.
- Composite cuts: several `Cut`s in one `apply` each keep their own `heads`, `blocks` and `steps`. When cuts of the same kind overlap in a head, the union of their reads is removed. Cuts of different kinds (renormalising, delta zero cut, recompute zero cut) may share a head only on disjoint receivers. Two kinds on the same (head, receiver) cell are refused, so run those as separate arms. The 2026-11-11 `both()` helper dropped `heads=`; `test_composite_cut_honours_each_specs_heads` guards against that bug.
- `observe(reduce, steps=None, blocks=None)` stores `reduce(read)` per (step, block). `read` exposes head-major `q`, `k`, `v` (K/V already expanded to query heads), `out`, `scale`, an additive `bias`, `probs()` (an FP32 softmax cached per call) and `write_norms()`. Two reductions are built in. `full_probs(dtype)` returns [H, Sq, Sk]. `class_flow(source_classes, receiver_classes=None, weighted=False)` returns receiver-class × source-class read mass, or write mass when `weighted=True` (the `flow_p` / `flow_w` of the read-graph `make_capture`). `Observer.stack(key)` returns [steps, blocks, ...]. Observers see this call's native read before this call's own cut is applied.
- `kv_group(head, n_heads, n_kv)` / `kv_groups(n_heads, n_kv)` is the single grouped-query mapping, `h // (n_heads // n_kv)`, tested against HF `repeat_kv`. It accepts any `numbers.Integral` (including numpy integers), refuses `bool`, and returns Python ints.
- **Batch size 1 only.** Reductions read batch item 0 and one row table applies to the whole call, so both backends refuse any intercepted call with B > 1. This includes batched classifier-free guidance and padded multi-prompt batches. Run one item per `apply`.

Backends:

- `FluxBackend(transformer, *, text_rows, identity=None)` gives a scoped override of `diffusers...transformer_flux2.dispatch_attention_fn`. Placement comes from forward hooks on every joint and single block, with joint blocks first, not from call arithmetic. The block is the hooked block's index. The step is the number of times block 0 has started in this `apply`, minus one. The hooks fire whether a block is called by the diffusers transformer forward or directly by `saturn_pub`'s FLUX.2 adapter, which never calls `transformer.forward`. `functional_call` (streamed residency) also runs module hooks. The backend refuses:
  - an attention call outside a block of this transformer;
  - a second attention call inside one block (reference-image extract mode makes two);
  - a block run out of order, or a forward stopped before its last block;
  - any call with Sq ≠ Sk (reference-image KV-cached mode, queries [txt, img] against keys [txt, ref, img]), batch > 1, an unexpected head count or fewer than `text_rows` rows. These checks apply to every call, including calls with no cut or observer;
  - a call that has a cut or observer and carries `attn_mask` or positional extras;
  - more forwards than `forwards=`, which catches classifier-free guidance run as separate forwards and a second image in the same `apply`.

  **Several images in one `apply` are not supported**: one `apply` covers one trajectory of exactly `forwards` forwards. Under classifier-free guidance run as two forwards per step, pass `forwards=2 × steps`. Steps are then forward indices, with conditional and unconditional forwards interleaved in the pipeline's order. `text_rows` is required. It is checked against the `encoder_hidden_states` that each joint block receives by keyword (diffusers and `saturn_pub` both pass it by keyword), and it selects the output projection used for write norms. Write norms use `to_add_out` for text receivers and `to_out[0]` for image receivers in joint blocks, and `to_out[:, :inner_dim]` in single blocks. The FLUX nesting guard sees only another `AttentionPort`. It cannot see the older ad hoc `dispatch_attention_fn` patches in qualification scripts, so never combine the two in one process context.
- `QwenBackend(model, identity=None, module=None)` patches `eager_attention_forward` in the module that defines the model's attention class, through an `AttentionObserver` subclass. It therefore shares the observer's patch point and its nesting guard in both directions. Qwen2 and Qwen3 each define their own `eager_attention_forward` in transformers 5.14.1 (`modeling_qwen2`, `modeling_qwen3`), with the same signature, and a tiny Qwen3 test checks that the right one is patched. The block is the decoder layer. The step increments each time layer 0 runs. When cuts are active, the returned `attn_weights` are the effective weights after the cut. **The model must run eager attention**: `install` refuses any other `config._attn_implementation`. Under `sdpa` the patched function is never called, so cuts would silently do nothing (`port.calls == 0`, identical logits), and the exit check catches any other route around the patch.
- Native mask visibility (Qwen). Hugging Face eager masks are finite, at `finfo(dtype).min`: −3.4e38 in fp32, −3.39e38 in bf16 and −65504 in fp16. A source counts as visible only when its bias is above `invisible_below(dtype) = max(−1e30, finfo.min / 2)`. Invisible sources get −inf before any recomputed softmax, so a renormalising cut never moves mass onto a future or padding position. A receiver that loses every visible source is refused, unless `empty="zero"`. A receiver the native mask already leaves with no visible source, such as a left-padding query row, has nothing a cut could remove, so it keeps its native output and weights.

Capture records never contain weights. Anything derived from weights is identified by parameter name (`backend.parameter_names(block)`) and by the caller's model identity. This is compatible with the pointer-based replay plan in `REPLAYABLE-CIRCUIT-DEBUGGER-PLAN.md`.

## Two cut types, and why both

| | `renormalize=True` | `renormalize=False` |
|---|---|---|
| operation | boolean mask: the cut reads get score −inf | contribution removed: `out − Σ_cut p·v` |
| other weights | rescaled to sum to 1 | unchanged |
| question it answers | what does the receiver compute without those sources? | what did those reads carry? |
| known artefact | redistributes mass onto the remaining sources | leaves the output with less than unit mass |

The two cuts answer different questions, and each one has its own artefact. In the FLUX read-graph experiment (`saturn/experiments/2026-11-11-flux-read-graph`), the renormalising mask produced object duplication that never appeared under the no-renormalisation cut, in 4 of 4 cases. **A result counts only if both cut types agree.** Run each cut with both types, each with its own `floor_arms`.

## Precision

The native attention call always runs unchanged on every head. Only the (head, receiver row) cells that a cut touches are recomputed in FP32 and written back. Every other head and every other row of a cut head returns the native bytes; the unit tests check this.

- `renormalize=False`, `zero_method="delta"` (the default) computes `native_out − (P ⊙ removed) @ V`, with P in FP32 from the call's own q/k. When nothing is removed this is exactly `native_out`, so **its sham is identical to native by construction and says nothing about the cut's arithmetic noise**. A delta cut keeps the native kernel's rounding of the whole row, subtracts an FP32 term from an already rounded BF16 output and rounds again. Its error against the exact cut is the native kernel's error on that row plus one BF16 rounding. When a cut removes nearly all of a receiver's mass, that error is large relative to the small output that remains: on a BF16 CPU test with removed mass 1.000, the error was 0.006 absolute against an exact output of norm 1.3e-5 (MEASURED, audit script). A (head, receiver) that loses every visible source gets exact zero, not the native residue (0.006 before this fix).
- `renormalize=False`, `zero_method="recompute"` computes `(P ⊙ keep) @ V` in FP32 for the touched rows. Its error against the exact cut is one BF16 rounding. Its sham floor is the gap between FP32 and the native kernel on those rows.
- Unit tolerances (`test_bf16_delta_and_recompute_against_float64_cut`; 64 rows, 2 heads, width 32, BF16, CPU SDPA as the native kernel). These hold per element against a float64 exact cut:
  - recompute: `|out − exact| ≤ 2⁻⁸·|exact| + 1e-5`;
  - delta: `|out − exact| ≤ |native − exact_native| + 2⁻⁸·|exact|/(1 − 2⁻⁸) + 1e-5`.

  In that test the delta and recompute versions of the same cut differ in their BF16 output.
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
- The delta form does **not** remove the floor; it only hides it from its sham. A delta sham removes nothing, so it is exactly native by construction. The delta cut itself still carries arithmetic noise, because it keeps the native kernel's rounding, subtracts an FP32 term from an already rounded BF16 output and rounds again. On this run, the arithmetic choice alone moved the same `z:own60` cut by **0.066 latent rel / 0.0036 px** (delta against recompute) and **0.031 / 0.0025** (delta against legacy). That is as large as the old sham floors (0.046–0.063 latent, 0.0042–0.0043 px). These distances were re-read from the saved latents and PNGs of `job-345766dba0e8` on 2026-11-12: 0.0664 / 0.00359 and 0.0310 / 0.00251 (MEASURED). So the effect of a cut is not "every byte that differs". Report each cut with its `floor_arms(c)` distances: its recompute sham against base, and its delta form against its recompute form.
- The new and legacy `z:own60` images differ from each other by 0.0025 px, less than the old floor of 0.0042, and both differ from iso by about 0.054 px.
- Renormalising cuts still carry an FP32 floor of 0.0038 px / 0.049 latent, against 0.0043 / 0.063 for legacy. Report each renormalising cut together with its `sham(c)`.

Runtime per 4-step branch: 0.34 s native, 0.38 s for the delta `z:own60`, 0.40 s for the legacy version, and 1.0 s for an all-cells delta sham. The `port.log` recompute counters are filled only on the non-legacy path. Observers run on legacy cut calls too: they see the native read, at the cost of one extra native kernel call per observed call.

The smoke worker (`qualification/attention_port_smoke_worker.py`) predates this revision and is hash-bound, so it was left unchanged. It calls `port.apply` without `forwards=` and counted calls itself, so rerunning it against this port fails at `apply` with the `forwards=N` refusal. A rerun needs a new worker that passes `forwards=STEPS`.

## What it supersedes

- `saturn/experiments/2026-11-11-flux-read-graph/read_graph_worker.py`: `Port`, `make_capture` (`flow_p`, `flow_w`, `tok_*`), `cut_mask`, `both` and the `headcut` arm block. These map to `FluxBackend`, `class_flow` / custom reductions, `cut` and multiple cuts in `apply`. That worker is unchanged.
- The `dispatch_attention_fn` patches in about ten `qualification/flux_*.py` scripts. Those scripts are bound by qualification hashes and stay byte-identical. New FLUX work should use the port.
- `mdb.attention_observer.AttentionObserver` still exists. Its behaviour is unchanged, and it gained an `_intercept` hook plus an optional `module=` argument for Qwen families other than qwen2. `QwenBackend` is built on it. Use `full_probs` or a custom reduction rather than retaining full Q/K/V clones.
- `mdb.native_discovery_capture` (`output_attentions` plus a `v_proj` hook) is unchanged. New captures should use `observe(full_probs())` under `QwenBackend`.
- Grouped-query mapping: `symbol_attention.attach_attention` now calls `kv_group`. The copies in `qualification/value_coalition_worker.py`, `attention_allocation_audit.py`, `frozen_value_pair_audit.py` and `resolver_execution_audit.py` are hash-bound and keep their inline formula, which is the same mapping.

## Tests

`tests/test_attention_port.py` covers tiny CPU tensors and a randomly initialised two-layer Qwen2:

- hand-computed cuts of both types, checked against `attention_partition.dense_attention`, and a several-receiver, several-head FLUX cut checked against a float64 oracle;
- per-head isolation, bit for bit;
- a delta sham that is identity by construction, and an FP32 sham whose error stays within rounding;
- `floor_arms` and per-cut `zero_method`;
- BF16 delta and recompute cuts against an exact float64 cut, within the documented tolerances;
- exact zero when a zero cut removes every source;
- composite cuts that respect each spec's heads, and mixed cut kinds allowed only on disjoint receivers;
- block, step and (block, head) selection;
- refusal of renormalising cuts that leave a receiver with no visible source, and the `empty="zero"` policy;
- exact reproduction of the legacy arithmetic, and observers running under legacy;
- FLUX placement: `forwards` required and checked, CFG-style or second-image extra forwards refused, counters reset per apply, partial, out-of-order or multi-call blocks refused, Sq ≠ Sk refused even on calls with no plan, and `text_rows` required and checked;
- batch > 1 refused (both backends);
- cuts, observers or a whole `apply` that never intercept a call are refused;
- observer reductions;
- Qwen observe, sham and cut through returned attentions and causal logits; no renormalised weight on masked (future or padding) sources (fp32 and bf16), including the BOS-sink cut; cut values against a float64 oracle built from independently captured q/k/v; a cut combined with a left-padding attention mask; KV-cached decode against a full forward; refusal of `sdpa`; a tiny Qwen3 that patches `modeling_qwen3`;
- the shared nesting guard;
- `kv_group` against `repeat_kv`, including numpy integers.

## Known gaps

- FLUX steps are forward indices within one `apply`, counted from 0. A trajectory resumed mid-way, for example from a Saturn checkpoint, is numbered from the resume point. A forward must start at block 0, because the port refuses anything else.
- Qwen steps assume layer 0 runs once per forward. The Qwen backend has been tested only on tiny random Qwen2 and Qwen3 models, not on a pretrained lane, and its FP32 floor there has not been measured.
- Every cut has an arithmetic floor, including a delta cut whose sham reads 0. Report `floor_arms(c)` with each cut.
- The port does not support batch > 1, Sq ≠ Sk on FLUX (reference-image KV modes), several images per `apply`, or non-eager Qwen attention. Each of these is refused rather than approximated.
- A panel generator, which would enumerate cut × type × sham per (prompt, seed) and record the both-types-agree rule, is not built yet.
