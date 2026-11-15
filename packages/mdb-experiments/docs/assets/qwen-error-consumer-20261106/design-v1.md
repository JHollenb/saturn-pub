# Block19 consumer of the retained H19 discrepancy — proposal only

This is a design and authenticated metadata inventory, not a frozen scientific
protocol or authorization to execute. It follows
`artifacts/mdb-flux-understanding-20261106/direction-11.md`: after consolidating
applicability, test what an unchanged native consumer does with the existing
counterexample. No tensor/checkpoint payload, model, scheduler or remote host was
read or executed for this design. Existing qualified sources remain unchanged.

## Exact two-world source selection

Use two complete saved **forecast** H19 states from turn14, both for its fixed
`prefix-negative` input. They are not native H19 observations, donor states or a
new row-sign intervention at block19.

| World ID | Source member in turn14 raw root | NPY SHA256 |
|---|---|---|
| `full-input` | `forecast/prefix-negative/Hnext.npy` | `c4a29fd4a58165e98d5deff70668b3b37747e9ca6c72e025cb5e30f4b186774b` |
| `cheap-input` | `shortcut/prefix-negative/Hnext.npy` | `3e7b095541b36b5ec3a4831c06f565cbc2314d8f76860baf31f63ae9fd6df20e` |

Each is BF16 `[1,512,2560]`, stride `[1310720,2560,1]`, stored as lossless uint16
NPY, 2,621,568 archive bytes. Their logical payload SHA256 values are respectively
`f34d3eb90ff3910fdb15fc99a5a157a7b6177807ab87a327516081b5cc4c80c5`
and `5087ead6a6e6385adb7630345c10455ece8e60790fbd4ffe3cd4f91f35a3e62a`.
“Full” means the complete functional block18 computation, whose prior native
comparison was exact; it does not mean semantic correctness.

Turn14 custody:

- Root `/mnt/big/mdb/research/job-74efe5506ecf/qwen-row-sign-result`.
- Job `job-74efe5506ecf`, request
  `fb202763231fad27712f2b525aa1ea0d041496c13079d6b4a4a7425d36e18a77`.
- Raw wrapper
  `d36f790376a85ad995cf20f01cff89fb6e04d282fe1290f677ba348ec98f4c8f`.
- Global forecast seal
  `12077f1ee8ca0f2b23c1c440e11a9dea7b5f07058270041594320289055f0121`.

The two descriptors join that raw inventory and the accepted saved audit's
pretarget transaction account. Future execution must authenticate the actual
turn14 seal/journal and verify both members are in its pretarget external
inventory before hydration/forecasting. Do not substitute the equal-valued
prior native full H19 result: lineage remains the sealed forecast lineage.

The shared context is the existing five members from turn14's authenticated
inherited source13: input IDs, attention mask, position IDs, and the original
FP32 `inv_freq` / `original_inv_freq` buffers. They still reside in
`/mnt/big/mdb/research/job-8c949b1c0278/qwen-sign-shortcut-result`, under raw manifest
`d573bffe5c0c3dfc3d1c353721f5696909f73b75fb6ff293aed2e905f4ec0c33`
and seal `a6d61b11fa67bfa0eb3a50fee66f26584b452874a590c255163102362460e652`.
Their exact paths/hashes/strides are in `planning-v1/selection-note.json`.
There are **seven selected arrays across two roots**, totaling **5,256,576 NPY
archive bytes**. No previous QKV, head, R/G/U, H20 or native outcome is an input.

## Original block19 recipe closure

`select_block19_recipes.py` reads only authenticated local JSON metadata and
preserves the exact eleven original `model.layers.19.*` entries from original398
manifest `d5060dd8137bf34568d3d9b0fda805c5979133b0c4a4538e23579196cb5d3374`.
It produced the **planning-only** subset
`planning-v1/qwen-block19-parameter-artifacts.json`, SHA256
`69a9189b00413ccb01943718a68eba2865181110aa30b04f17668a8e2e1ffb8c`.
This metadata selection does not freeze the experiment.

The set is input/post-attention RMS weights; per-head Q/K RMS weights; Q/K/V/O
linear weights; and gate/up/down MLP weights. Learned payload is **201,861,632 B**;
the original virtual NPY archives total **201,863,040 B**. All eleven come from
`model-00001-of-00002.safetensors`, whole-file size **4,967,215,360 B**, SHA256
`8c0506e7f4936fa7e26183a4fd8da4e2bdbc5990ba64ae441f965d51228f36ea`.
The 26,360-byte header is bound to
`8d1a2170cadc1eb2e68cb60f5610c304d263f29de260584fd4fadc73d40b56ec`.
Use the unchanged public standalone retriever/RecipeBank and stream one whole-file
checksum plus selected payload/NPY checks; no exported or persistent weight bank.

Use the existing original config and native/SDPA source pins, Torch2.13.0 /
Transformers5.14.1, BF16 CUDA B1, original RMS epsilon, CPU FP32 RoPE formation,
all512 rows attended, mask=None at SDPA, is_causal=True, enable_gqa=True,
dropout_p=0, scale=128**-0.5, no cache. Construct only
`Qwen3DecoderLayer(config, layer_idx=19)`. This is a complete component interface,
not a whole-Qwen Session or genuine Qwen Act.

## Minimal shared evaluator v2

Frozen `qwen_component_experiment.py` SHA81131b17 is not suitable unchanged:
it derives every input from one `sources['h']` and row signs, requires archived
positive output ports, and always creates an extra `baseline` native target.
It also carries a fixed block18 scope. Do not fake a sign vector, positive port
bank or additional baseline to fit that API.

Propose one new versioned shared evaluator, e.g.
`qwen_component_experiment_v2.py`, with an ordinary function:

```text
execute_worlds(output, layer, *,
    worlds=[{id, input_tensor, input_reference}],
    context_sources, source_bindings,
    callbacks={functional: full_block_callback},
    response={left_world: cheap-input, right_world: full-input,
              ports: [head, projection, R, M, Hnext]},
    frontier_scope, expected_counts, ...)
```

`full-input` is the response baseline **world**, not an extra execution. The core
iterates the two explicit worlds and then exactly those same two native worlds.
The one full-functional callback invokes unchanged frozen history `_block` with
index19 / profile horizon20. It reforms all norm/QKV/RoPE/attention/MLP fields
from that world's own H19. It never refreshes an inner state from a target.

Reuse the existing dense-array restoration, identities/live descriptors,
parameter-free context helper (its inputs need only the five context sources),
primitive census, module guard, tensor writer, public ForecastTransaction and
bounded finalizer. Factor the existing native hook installation so the same
input/context/QKV/head/R/M/return captures serve named worlds. Add one passive
`self_attn.o_proj` forward hook for the **actual** projection output; do not
reconstruct it as R−h. All hooks return None and are removed in finally blocks;
restore F.linear/SDPA wrappers and flags. Freeze neither old source nor old
scientific records by rewriting them.

Frontier uses supported `kind: architectural`, `origin: inherited-functional-state`,
`scope: conditional block19 hidden-state interface`, with false raw-input,
genuine-record-prefix and genuine-Qwen-Act claims. Row-sign structural support is
explicitly `not-applicable: explicit saved input worlds`; no synthetic closed-row
counts. Causal prefix equality is a separate measured input premise.

A later row-sign adapter can construct explicit worlds and attach optional
support metadata outside this core. This cut needs no predictor registry,
generic intervention language or image campaign wrapper.

## Proposed persisted ABI and chronology

Keep established plan/source/runtime/parameter-identity/timeline/transaction/
summary/request/files roles. Use a small two-root source binding:

```text
source-frontier.json:
  roots: {turn14: authenticated root+manifest+seal, context: root+manifest+seal}
  worlds: {full-input: {input_reference}, cheap-input: {input_reference}}
  shared_context: {input_ids, attention_mask, position_ids, rope_buffers...}
  context: {position_ids, cos, sin, mask/cache/profile flags}
  support: {status: not-applicable, reason: explicit-input intervention}

forecast.json:
  worlds[world].{input_reference, consumed_input_descriptor, ports}
  ports include full head/projection/R/M/Hnext and all existing internal witnesses

forecast-responses.json:
  orientation: cheap-input-minus-full-input
  baseline_world: full-input
  fields[port]: {shape, elements, dtype: float64, sha256}

native.json:
  worlds[world].{actual_input_descriptor, ports}
  context; actual consumed kwargs/input receipts; raw native head BHTD

comparisons.json / audit.json:
  per-world complete forecast−native endpoint errors
  complete forecast-world-difference−native-world-difference errors
  input premise + prefix/suffix + finite-addition diagnostics
```

Use inherited references for both input arrays. Their full source bytes plus
actual argument hashes/strides/object receipts establish what was consumed;
duplicating those arrays in every world/call is unnecessary. Newly computed
outputs are retained losslessly. Native R is the actual post-attention RMS input,
M is actual down-projection output, Hnext is actual layer return; `Hnext` is
reported at clock H20. Capture actual QKV/SDPA operands as in the qualified panel.

Order: authenticate both roots and recipes → restore seven sources → form common
context → forecast full-input and cheap-input independently → save every complete
forecast output → form the **complete FP64** cheap-minus-full response and its
per-port hashes → seal the full forecast inventory and responses globally →
exactly two native targets (full-input then cheap-input) from their original
declared inputs → saved-only endpoint/response/consumer reductions.

The two full forecast arrays per port make every signed difference reconstructible;
seal its complete FP64 digest, shape and orientation without a duplicate delta bank.
No native learned/module entry or extra positive block may occur before the seal.
Primary five-port comparisons require explicit MDB `max_elements=7,340,032`
(head2,097,152 + four residual-width fields1,310,720 each), not the default1M.

## Exact physical census and retention estimate

| Work | Functional forecasts | Native targets |
|---|---:|---:|
| Whole block19 evaluations | 2 primitive graphs | 2 original DecoderLayer calls |
| SDPA | 2 | 2 |
| Linear | 14 | 14 |
| RMS operations | 8 | 8 |
| Q/K rotary transforms | 4 tensor transforms | 2 native Q/K-pair applications |
| Native module entries | 0 | 30 |

One original eleven-weight hydration, one device placement and one final offload;
no extra baseline, full model, LM, embedding, tokenizer, FLUX, VAE, image or noise.
Excluded-component zeros are program scope, not independent profiling. Report
actual counts and observation hooks separately. Four SDPA computations total
does not imply four independent kernels or experiments.

Retaining all15 functional ports and12 native ports for both worlds is about
244,318,208 BF16 payload bytes before small NPY headers/context/metadata, using
the known original geometry. Reusing inherited h inputs avoids about10.5MB of
duplicate input captures. The five primary ports alone account for58,720,256 B
over four evaluations. These are static byte estimates, not measured reservations
or performance claims. Root should choose the finite lease/resources after
implementation qualification; no execution/staging is part of this proposal.

## Observations, not gates

Retain complete input δH19 and output deltas, whole and `[0,256)` / `[256,512)`.
Prior accepted turn14 reports say the first256 H19 rows are byte equal. Verify
that premise from the fixed source bytes when eventually loaded. Under the same
causal/no-cache profile, equal first256 inputs and positions predict equal
first256 head/projection/R/M/H20. Preserve numeric, bit and signed-zero misses;
an inapplicable prefix premise must not erase the rest of the two-world result.

Later-row persistence, spread, magnitude, relative error, amplification or
reduction are observations. Compare δH20 with δH19 in the common residual space;
do not call a head-to-residual RMS change a gain. Retain signed means, norms,
directions, changed counts, first differences, absolute and reference-relative
values, and both world-native endpoints. A small relative whole-state endpoint
error can still carry a large fraction of the signed response.

For residual/MLP decomposition use only saved arrays and FP64 reductions:

```text
δR    = δH19 + δprojection + εR
δH20  = δR   + δM          + εH
εR    = δR   − δH19 − δprojection
εH    = δH20 − δR   − δM
```

These are differences of finite-addition defects, not extra model projections.
Dot/cosine and the cross term of δR/δM can describe cancellation in their shared
coordinates, while εH must remain visible. Negative alignment alone is not proof
of a separate causal repair writer. No Wd(GU) RHS, extra baseline or learned
diagnostic call is needed.

Smallest qualification would use two synthetic unequal-width native-Qwen input
worlds with a genuinely shared prefix and different suffix, the actual common
writer/transactions and exactly two native calls. Check world-ID/input binding,
independent response orientation, no hidden baseline, prefix/miss retention,
actual projection hook, global pretarget seal, immutability and cleanup. A saved
reader should use that actual serialized fixture and the same production catalog
context builder against the installed Atlas schema. These are proposed mechanics,
not original-data outcomes or permission to implement/execute yet.
