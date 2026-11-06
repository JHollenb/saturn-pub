# The reader closes from Q/K/V to its next text state

The [previous arithmetic microscope](FLUX-READER-ARITHMETIC.md) reconstructed
the text tail exactly when supplied an observed attention head. This experiment
moves that observation boundary backward: it calculates the head from the
saved positioned Q/K/V, then carries that calculated head through the tail.

On all eight original step3/joint0 cases, the head, after-attention residual,
MLP return and final text state match the original archives byte for byte.
That is 32 primary archive comparisons over 50,331,648 BF16 values, across
eight cases in two known receivers at seed 127. The count of tensor values is
not a count of independent experiments.

This is an extracted executable reader using native numerical primitives.
It is known-input reconstruction, not fresh prospective validation or an
independent implementation of every arithmetic operation inside FlashAttention.

## What actually executed

Finite CUDA job `job-fb2d2d4395fb` runs seven attention lanes on each of the
eight K/V corners. Every lane receives the same actual text queries, complete
keys and complete values after their native positioning/normalization stages.
Only image K and/or V differ between corners. Incoming text, modulation and
the three original text matrices supply the tail.

| Lane | Query geometry | Computation |
|---|---|---|
| Automatic text | 512 actual queries | Automatic native SDPA |
| Automatic full, zeros | 512 actual + 1024 zero queries | Automatic native SDPA |
| Automatic full, repeats | 512 actual + 1024 repeated text queries | Automatic native SDPA |
| Forced math | 512 actual queries | Native SDPA math backend |
| Forced Flash | 512 actual queries | Native SDPA Flash backend |
| Staged FP32 | 512 actual queries | Explicit FP32 QK, softmax and PV, then BF16 output |
| Staged BF16 probabilities | 512 actual queries | FP32 QK/softmax, BF16 probabilities and BF16 PV |

There are 24 heads, head width 128, and 1536 key/value rows. Full-shape
controls use synthetic unused queries because the original capture retained
only text queries. They do not reconstruct the original image queries.

Both full-shape controls and the 512-query automatic call reproduce the source
text heads exactly, in all eight cases. This rules out these query-geometry
differences as the source of the observed error on these inputs. The independent
query rows have no mathematical attention dependency on other query rows; the
controls additionally check the tested finite-precision execution shapes.

The worker transcribes the pinned Diffusers native dispatch operation:
permute B/sequence/head/feature to B/head/sequence/feature, call Torch SDPA,
then permute and flatten the returned text heads. It verifies the installed
original dispatch and transformer source hashes without importing Diffusers'
runtime. Each attention call is profiled. Automatic calls actually select
`aten::_scaled_dot_product_flash_attention`; forced-math calls select
`aten::_scaled_dot_product_attention_math`. Enabled-backend flags alone are
not used as evidence of which kernel ran.

The historical source did not record its selected kernel. Exact output parity
qualifies the new executable against those captures; it does not uniquely
identify the historical kernel implementation from its outputs.

The complete physical census is 40 SDPA calls, 32 explicit staged matrix
multiplications, 16 explicit softmax operations, 48 text linear calls,
16 LayerNorm calls and 16 SwiGLU calls. Sixteen text tails execute: one per
automatic-full-zero and forced-math head in each case. There are zero full
model loads, transformer-block forwards, attention-module forwards or image
renders. Native primitive computation is explicitly counted; this is not a
zero-computation or zero-native-kernel result.

## Why the mathematical equation was insufficient for exact prediction

For a text query row j and head h, the real-valued attention equation is:

```text
S[j,i,h] = dot(Q[j,h], K[i,h]) / sqrt(128)
P[j,:,h] = softmax(S[j,:,h])
H[j,h]   = sum_i P[j,i,h] * V[i,h]
```

The native execution is a finite-precision function `F_kernel(Q,K,V)`, not
necessarily `round_BF16(H_real)`. Fused reduction, softmax and accumulation
order can distinguish implementations of the same real-valued equation.
This experiment changes the selected execution while holding its operands
fixed. It does not independently resolve every internal source of rounding.

| Head calculation | Complete head error across eight cases | Joint K/V head-effect error, left / right |
|---|---:|---:|
| Automatic, any of three query controls | Exact | Exact / exact |
| Forced Flash | Exact | Exact / exact |
| Forced math | 0.09480–0.09667% | 36.15% / 34.53% |
| Staged FP32 | 0.09482–0.09664% | 36.12% / 34.52% |
| Staged BF16 probabilities | 0.1634–0.1666% | 50.64% / 46.62% |

The earlier NumPy reference's joint head-effect errors were 36.13% / 34.55%.
The math and staged-FP32 lanes closely resemble that discrepancy. Merely
rounding the complete probability matrix to BF16 does not reproduce the
source; it worsens the tested head-effect errors. These are counterexamples
to that simple numerical account of the fused kernel.

Let `epsilon(x) = F_approx(x) - F_native(x)`. For two matched states:

```text
Delta_native = F_native(x1) - F_native(x0)
Delta_approx - Delta_native = epsilon(x1) - epsilon(x0)
```

A small complete-state error does not bound relative error on a much smaller
state change. The error's change across contexts matters. No fitted offsets
or learned correction coefficients are introduced here.

## The calculated head continues through the MLP

The automatic-full-zero head feeds the previously qualified native tail:

```text
projection = linear(H, W_attention)
U = B(T + B(g_attention * projection))
N = LayerNorm(U)
X = B(B(N * B(1 + c_mlp)) + s_mlp)
[a,b] = linear(X, W_in)
G = B(SiLU(a) * b)
M = linear(G, W_out)
T_next = B(U + B(g_mlp * M))
```

`B` denotes the declared BF16 boundaries; native matrix/normalization primitives
retain their own numerical contracts. The worker loads no original target head
or target text tensor to calculate this chain. It reads captured Q/K/V,
the incoming parent text/modulation and original matrices. Target records in
the worker are metadata references; target array values are opened by a separate
CPU audit after execution. Nevertheless all source outcomes were already known,
so this remains diagnosis rather than a new held-out prediction test.

The automatic chain reproduces every primary archive and all key-only,
value-only, interaction and total text contrasts exactly. Supplying a
forced-math head to the same native tail leaves 87.65% / 89.16% error on the
joint text effect. Its head-effect cosine of 0.935 / 0.940 becomes text-effect
cosine 0.642 / 0.609. The nonlinear and rounded receiving computation therefore
changes how the attention discrepancy appears downstream.

The new recorded runtime has BF16 reduced-precision matrix reduction enabled,
math-SDPA reduced precision disabled, TF32 disabled and strict deterministic
execution. These are fresh getter observations from this job. They do not fill
in the unknown getter values in the earlier failed metadata publication.

## What belongs in the symbol table

The generic computation ID is `law.flux.reader.qkv-to-text.native.v1`, on route
`route.flux.image-KV.text-projection-residual-mlp.v1`. Its record binds the
source model and parent states, operand frontier, clock, horizon, numerical
contract and actual effects. Original weight recipes provide the constants;
there are no new learned parameters.

This extends the [latent/reader causal account](FLUX-LATENT-READER.md):

```text
persistent latent
    -> next image projection and image K/V
    -> text attention head
    -> projected, gated residual write
    -> normalized/modulated MLP computation
    -> next working text state
    -> previously observed downstream continuation
```

The current experiment closes the numerical reader section. The earlier
campaign supplies the measured connection to continuing image state and native
futures. No new image future executes here, and no universal object dictionary
or prompt-to-image capability follows from this local closure.

CPU-only job `job-92b4b36fdaa7` independently verifies all referenced arrays and
their inventories, measures full fields and signed factorial effects, then
publishes 24 MDB EquationCatalog records. Eight records compare the complete
reader, fourteen compare numerical head effects and two compare forced-math
continuations. Misses remain searchable alongside exact comparisons.
All 24 supplied-context queries resolve and all 24 changed-policy queries
abstain. Supplied applicability is not a learned generalization rule.

The following installed-wheel commands were exercised from
`artifacts/mdb-flux-reader-attention-20261105/`:

```bash
ROOT=compact-collection-01/extracted/binding-minimal-result
python -I -m mdb equations find --catalog "$ROOT/catalog/equations" \
  --applicable-to "$ROOT/catalog/context-right.json"
python -I -m mdb equations find --catalog "$ROOT/catalog/equations" \
  --applicable-to "$ROOT/catalog/context-wrong-policy.json"
```

The first returns `cmp.flux.reader-attention.right.k1v1.auto-full-zero`, with
zero residual over all four fields. The second returns `[]`.

## Reproduction, storage and limits

The worker used the existing public MDB/Saturn-pub wheel cohort outside the
checkout. Five focused mechanics tests pass, including independent query-row
controls, input immutability, profiler observables, operation counts and the
existing tiny native-block/tail qualification. Owned Python files pass Ruff.

```bash
python -I stage_flux_reader_attention.py RUN \
  --prior LATENT_READER_CAMPAIGN --arithmetic READER_ARITHMETIC_CAMPAIGN
python -I stage_flux_reader_attention.py RUN --submit
# After native success:
python -I flux_reader_attention_audit.py AUDIT_RUN --study RUN
python -I flux_reader_attention_audit.py AUDIT_RUN --submit
# After audit success:
python -I recover_flux_directory.py AUDIT_RUN COMPACT_COLLECTION
```

The native worker's operator phase reached its eighth row at 6.061 seconds;
runner time was 23.75 seconds, peak RSS 2252.7 MB and peak VRAM 824 MB. This
is not a comparative speedup benchmark. Run-01 was staged but never submitted;
run-02 contains the executed frozen payload after an import-format correction.

The raw native corpus is 98 files totaling 252,456,571 bytes, retained on Beast.
The saved-only audit took 8.069 worker seconds, 12.56 runner seconds, 362.6 MB
peak RSS and zero VRAM. The compact audit archive is 146,373 bytes, expanding
to 649,348 bytes across 39 files. It includes
source manifests, numerical profiles, case references, comparisons, original
weight recipes, a standalone weight retriever and `retrieve_evidence.py` for
checksum-verified individual arrays. It avoids collecting the full trace by
default. Generic Saturn evidence-plane coverage is reported separately from
the dedicated array audit: run and verify both return
`evidence-incomplete-empty-target`, with zero recognized streams or receipts.
Array-only catalogs still need a recognized adapter; this is not a scientific
null or evidence-plane certification. Independent retrieval of the calculated
and original `right.k1v1` head and text archives also verifies byte equality,
with four 3,145,856-byte retrieved files. Source and calculated files are kept
under separate retrieval aliases even when their content hashes match.

The next scientific cut can now separate **which head/row changes carry the
K/V effect** from inaccuracies in its approximate numerical account. Use
same-parent head restoration or value-source subsets through this exact reader,
retain countereffects and collateral, then validate selected changes through
the unchanged native future. Fresh-state sealed predictions remain the test
of transfer. Disassembling FlashAttention's internal arithmetic is a separate
open task if an independent kernel-exact mathematical implementation is needed.
