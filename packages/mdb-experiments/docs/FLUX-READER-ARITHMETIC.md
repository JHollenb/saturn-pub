# Where the small reader-effect prediction loses accuracy

The [latent/reader experiment](FLUX-LATENT-READER.md) localized an exact carrier
route, but its numerical reference had 115–128% error on small split key/value
text effects despite only 0.17–0.19% error on full text states. This follow-up
separates local operator error from error carried through successive operations.

It is an observed-operand diagnostic. Every source target is already known.
It does not replace the original forecasts or count as fresh prospective success.

## Smaller than another image campaign

The source is the eight step3/joint0 cases of FLUX.2 Klein 4B distilled, CUDA
BF16, B1, 512 text rows, width 3072, seed 127, from `job-7afc16a70861`.
Only three original matrices are needed: `attn.to_add_out.weight`,
`ff_context.linear_in.weight`, and `ff_context.linear_out.weight`.
They contain 188,743,680 BF16 bytes. Checkpoint identity and named payload hashes
are verified through the existing recipes. Temporary expanded views are not
exported as a new parameter bank.

The instrument transcribes the pinned eager source into a small Torch primitive
graph: `functional.linear`, native `LayerNorm`, and the original SwiGLU operation
sequence, preserving complete logical matrix geometry. Three parameter-backed
linear wrappers perform no random initialization. The original Diffusers source
file is SHA-256 verified without importing its package/compilation machinery.
The graph reproduces the original captured after-attention, MLP and text
returns byte for byte in all eight cases: 24 matching archives containing
37,748,736 BF16 values. This qualifies this extracted tail on these inputs.
It does not qualify the preceding attention calculation or unseen contexts.

For each of eight cases:

| Lane | Supplied boundary | What it isolates |
|---|---|---|
| Native input | Actual attention head | Whether the native tail reproduces the source |
| Reference head | Previously frozen reference head | Attention error carried through native tail arithmetic |
| Actual after-attention | Actual residual after attention | Native MLP replay from its actual incoming state |
| Opposite reduction | Actual attention head | A changed CUDA BF16 reduction policy through the tail |
| Local opposite | Each matrix's own actual native input | Reduction-setting changes without changing operands |
| Mathematical actual head | Actual attention head | Independent equations with attention error removed |
| Mathematical actual residual | Actual after-attention residual | Independent normalization/MLP equations with earlier error removed |
| Mathematical local | Each operation's own actual input | Local error, without propagation from another reference operation |

The final residual equation is checked separately from the original source
after-attention and MLP returns. Every comparison keeps signed effects as well
as complete-state errors. Turning off reduced precision defines another numerical
lane; it is not automatically a correction to the original model.

## Equations and observation boundaries

Write `B` for the declared BF16 rounding operation, `H` for the attention head,
`T` for incoming text, and the six modulation vectors as `(sA,cA,gA,sM,cM,gM)`.

```text
P = linear(H, W_attention)
U = B(T + B(gA * P))
N = layer_norm(U)
X = B(B(N * B(1 + cM)) + sM)
[a,b] = linear(X, W_in)
G = B(SiLU(a) * b)
M = linear(G, W_out)
T_next = B(U + B(gM * M))
```

The independent reference uses FP64 affine/normalization calculations and
declared BF16 boundaries. The local version supplies native `H`, `P`, `U`, `N`,
`X`, `[a,b]`, `G` and `M` to their respective equations. The carried versions
consume their own calculated intermediates. This distinguishes a wrong local
numerical account from amplification of an earlier approximation.

Original matrix bits alone do not specify every result of finite-precision
execution. The record also binds dtype, complete geometry, runtime versions,
reduction setting, TF32 policy, deterministic mode, inference mode and module
source hashes. This assay makes no attention-kernel calls, so it cannot settle
the remaining dense-reference versus native SDPA discrepancy.

## Measured result: the small effect exposes the numerical program

For an observed field `Y` and its approximation `Y_hat`, complete-state error is
`||Y_hat - Y|| / ||Y||`. For the matched joint key/value change, the relevant
error is instead:

```text
Delta Y     = Y(k1,v1) - Y(k0,v0)
Delta Y_hat = Y_hat(k1,v1) - Y_hat(k0,v0)
effect_error = ||Delta Y_hat - Delta Y|| / ||Delta Y||
```

All reported norms use the complete field, without sampled coordinates. There
are two receiving contexts, each with four K/V corners. The key-only,
value-only and interaction contrasts remain in `audit.json` alongside the
joint change summarized here.

| Independent local equation, supplied its own actual input | Full-state error across eight cases | Joint effect error, left / right |
|---|---:|---:|
| Attention output projection | 0.2452–0.2488% | 86.21% / 88.23% |
| After-attention residual addition | Exact | Exact / exact |
| LayerNorm | 0.000072–0.01994% | 0.272% / 0.124% |
| MLP modulation | Exact | Exact / exact |
| MLP input matrix | 0.01297–0.01405% | 5.83% / 5.89% |
| SwiGLU | Exact | Exact / exact |
| MLP output matrix | 0.2465–0.2544% | 74.01% / 76.03% |
| Final text residual addition | Exact | Exact / exact |

“Local” is essential: exact SwiGLU here receives the actual paired input; it
does not imply that a chain supplying an approximate paired input is exact.
The independently calculated final residual also matches the original source
returns in all eight cases. The decoded equations and BF16 rounding boundaries
therefore account exactly for those local operations on the tested operands.

The changed CUDA reduction policy provides a discriminating intervention.
With matrix operands held fixed, the FP64 reference's disagreement with the
toggled-policy projection effect falls to 6.37% / 6.69%, and with the MLP
output effect to 9.10% / 10.25%. The MLP input matrix is unchanged by that
toggle. Toggled-policy full-state disagreements with FP64 are respectively
0.01296–0.01494% and 0.02182–0.03116% at projection and MLP output. Remaining
disagreements are retained; no policy is declared mathematically exact or a
correction to the original runtime.

This supports a specific working inference: the native numerical reduction
path is a major part of these small state changes. A reference using the same
weights and nominal real-valued equation can still implement a measurably
different finite-precision program. This is not evidence that we discovered a
new learned tensor or need fitted endpoint coefficients.

## Local errors and carried errors are different

| Whole text-tail comparison | Joint text-effect error, left / right |
|---|---:|
| Previous frozen mathematical chain | 117.99% / 111.43% |
| Exact observed attention head, independent mathematical tail | 116.86% / 112.19% |
| Previous predicted attention head, native arithmetic tail | 88.60% / 88.69% |
| Exact observed after-attention residual, independent mathematical MLP | 49.77% / 51.90% |
| Exact observed head, extracted native arithmetic tail | Exact / exact |

Removing attention-head error alone does not repair the independent tail.
Conversely, using native tail arithmetic with the previous predicted head
improves this joint effect but still leaves substantial error. These are
counterfactual boundary substitutions, not additive percentages of blame.
They identify two remaining numerical dependencies: attention-head computation
and downstream affine execution. LayerNorm has small local discrepancies;
it is not licensed as universally exact.

The native tail is an executable computation record: incoming text, attention
head, modulation, original matrices, numerical execution contract and the
ordered operations determine its next text state. In the symbol table, its
route names the image-K/V-to-text consumer path, while this record spells out
the consumer's computation. The earlier latent/reader campaign supplies the
connection to the continuing image state and future. This assay by itself
has no new image continuation and assigns no universal object meaning.

## Actual MDB workflow

CPU-only `job-c8a4bc3cc1e4` audited the saved arrays and published 16 searchable
records: eight exact source-return comparisons, six local matrix-effect
comparisons and two carried-head text-effect comparisons. Sixteen exact
applicability queries resolve; sixteen changed-reduction queries abstain.
Conditions are explicitly supplied, not inferred rules. Numerical misses are
stored in the same catalog as exact replays.

From the campaign directory, these commands were actually exercised:

```bash
ROOT=compact-collection-01/extracted/binding-minimal-result
python -I -m mdb equations find --catalog "$ROOT/catalog/equations" \
  --applicable-to "$ROOT/catalog/context-right-projection.json"
python -I -m mdb equations find --catalog "$ROOT/catalog/equations" \
  --applicable-to "$ROOT/catalog/context-wrong-reduction.json"
python -I "$ROOT/retrieve_evidence.py" native native-contract.json retriever-smoke-01
```

The first returns `cmp.flux.reader-arithmetic.right.projection-effect`,
including the measured discrepancy; the second returns `[]`. The third
retrieves and SHA-256 verifies one original metadata artifact. Every tensor
path in the source manifests can be retrieved by the same mechanism. Weight
retrieval remains a separate original-checkpoint recipe, not a new weight bank.

The saved-only audit took 83.67 seconds inside the worker, with 87.9 seconds
runner time, peak RSS 1328.5 MB and zero VRAM. Its compact archive is 251,662
bytes, expanding to 1,132,688 bytes across 32 files. Those sizes exclude the
externally retained 2.1 GB raw trace and source campaign. Custody independently
verifies all compact files and their source inventories. Focused qualification
passes 11 tests; the owned Python files pass Ruff.

Saturn's generic evidence sweep sees no recognized runtime streams or receipts
in this array-only compact package and reports
`evidence-incomplete-empty-target`. That is an adapter limitation, separate
from the dedicated source-bound array audit and catalog checks. It is not a
scientific null or generic evidence-plane certification.

## The next cut

The [attention follow-up](FLUX-READER-ATTENTION.md) now calculates the head
from captured Q/K/V and reproduces all eight original reader states exactly
through the native tail. Automatic/forced-Flash kernels match; forced-math
and explicit FP32 programs retain head-effect discrepancies that grow through
the tail. Query-shape controls also match exactly. This closes the tested
reader at native-primitive granularity, not the internal arithmetic of the
Flash kernel or fresh-state generalization. Head/row-specific route tests can
now use this exact consumer, followed by native future validation and sealed
predictions on new states.

## Execution and reproduction

One finite mrun operator lease runs 112 native linear calls, 32 LayerNorm calls,
32 SwiGLU calls and 64 independent NumPy affines. Native module counts come from
actual forward hooks. A global guard refuses undeclared modules and all module
execution during mathematical reference work. There are zero full model loads,
transformer-block forwards, attention calls or image renders.

Use the installed public wheel and authenticated source campaign to stage a
new operator experiment:

```bash
python -I stage_flux_reader_arithmetic.py RUN --prior LATENT_READER_CAMPAIGN
python -I stage_flux_reader_arithmetic.py RUN --submit
```

The completed experiment's tensors remain on Beast. Its compact audit carries
the source manifests, original weight recipes, a weight retriever and a
standalone `retrieve_evidence.py` for checksum-verified individual arrays.
This avoids downloading the entire 2.1 GB operator trace merely to search its
computation records. The audit's lazy loader preserves the frozen measurement
function, checked by its AST hash; it changes storage access rather than the
numerical comparison. No tensor is subsampled.

The tiny native-block qualification reproduces actual captured text returns with
reconstituted components, exercises direct actual-residual replay, checks the
independent local residual/modulation equations and verifies hook cleanup.
It establishes mechanics; the CUDA source comparisons establish the real-model
instrument's applicability.

## Startup and storage evidence

The first submission, `job-034908f12573`, exhausted its 420-second outer budget
during startup. It had no result directory, zero measured VRAM, and no operator
outputs. Read-only process inspection found disk sleep at `__wait_on_buffer`.
The failed job and exact log remain in `run-03/`; this is an instrument failure,
not a computation result. Two earlier stage directories were never submitted.

The next launch adds Python `-B`, early phase/stack logging, and a 600-second
outer startup allowance. Its 240-second scientific worker budget and numerical
equations are unchanged. Disabling bytecode writes removes one source of setup
IO; it is not a measured speedup. The trace localizes setup to Diffusers'
installed-package metadata scan and Torch Dynamo imports, before any result
directory or CUDA allocation. `job-f273f8182130` is canceled with that explicit
reason and preserved in `run-04/` before replacement. The final instrument uses
the Torch-only primitive graph, which passes parity against the actual tiny
native block and the original eight CUDA cases.

`job-486f616ef636` completed the declared operator census and saved every case
by worker second 111.344, then failed during module-source metadata inspection
of nested classes created with `runpy`. Its scheduler failure remains intact.
CPU-only recovery `job-e63620b1039d` verified and linked the unchanged output
files into a complete publication with zero new native or reference calls.
The recovery retained unknown runtime getter values as `null`, including the
default BF16 reduction flag. Consequently the measured lanes are named
**source runtime default** and **toggled policy**; this campaign cannot honestly
label the original default as enabled or disabled. Source archive parity binds
the default lane to the actual captured returns despite that metadata gap.

Future workers publish the numerical contract before tensor work and bind the
whole shipped primitive-graph file instead of inspecting a nested class.
Strict eager determinism now uses the public
`set_deterministic_debug_mode('error')` interface. A fresh-process test verifies
that it enables strict determinism without loading Torch's compiler modules.
This is a qualified startup change, not a measured real-model speedup; the
historical operator run used the previous interface.

During the run, local free disk fell to 1.6 GiB. Six owned duplicate transport
archives/logs from the source/receiver and latent/reader campaigns were copied
to Beast custody storage and SHA-256 verified before local removal, freeing
2,414,143,114 bytes. Extracted states, forecasts, measurements, inventories and
audits remain local and unchanged. Each collection has a transport-location
sidecar; generic complete transport verification requires retrieval first.

The additive receipt and standalone `retrieve-transport-v1.py` live under
`artifacts/mdb-flux-reader-arithmetic-20261105/`. Staging can reuse the original
verified transport receipts while rechecking every consumed extracted file;
it makes no new archive-verification claim. Original parameter recipes and
`retrieve_weights.py` remain included in new scientific artifacts as before.

The raw output corpus contains 369 files totaling 2,096,233,773 bytes after
metadata recovery. Full-archive collection exceeded its declared 1.5 GiB
transport bound, so publication switched to a CPU-only near-data audit and a
compact report. The failed collection and stopped redundant partial SCP are
retained as operational receipts. No scientific calculation was repeated to
repair publication.
