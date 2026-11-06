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

The instrument reconstitutes the original PyTorch `Linear`/`LayerNorm` and
Diffusers `Flux2FeedForward` components, preserving complete logical matrix
geometry. Their returns must be compared against the original captured
after-attention, MLP and text returns before interpreting component replay as
the source numerical path. A mismatch remains an instrument observation.

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

## Execution and reproduction

One finite mrun operator lease runs 112 native linear calls, 32 LayerNorm calls,
32 SwiGLU calls and 64 independent NumPy affines. Native module counts come from
actual forward hooks. A global guard refuses undeclared modules and all module
execution during mathematical reference work. There are zero full model loads,
transformer-block forwards, attention calls or image renders.

Use the installed public wheel and authenticated source campaign:

```bash
python -I stage_flux_reader_arithmetic.py RUN --prior LATENT_READER_CAMPAIGN
python -I stage_flux_reader_arithmetic.py RUN --submit
python -I recover_flux_directory.py RUN COLLECTION
python -I RUN/payload/analyze_flux_reader_arithmetic.py \
  COLLECTION/extracted/binding-minimal-result \
  LATENT_READER_CAMPAIGN/collection-01/extracted/binding-minimal-result audit.json
```

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

The corrected launch adds Python `-B`, early phase/stack logging, and a 600-second
outer startup allowance. Its 240-second scientific worker budget and numerical
equations are unchanged. Disabling bytecode writes removes one source of setup
IO; it is a hypothesis about the bottleneck, not a measured speedup.

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

Measured operator outcomes are added after output custody and the frozen audit.
