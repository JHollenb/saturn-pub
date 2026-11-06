# From persistent latent to the next text write

This experiment follows the source/receiver carrier route one boundary further.
At the next pass, working text is rebuilt from unchanged conditioning, but the
changed image stream can write a difference back into text through attention.
The assay separates the image keys from image values in that read, predicts the
complete receiving text write, and observes native image continuations.

The specimen is FLUX.2 Klein 4B distilled, CUDA BF16, batch one, seed 127,
512 × 512, four denoising passes. The two receiving contexts and their changes
are saved results from `job-3272cb7fc922`. They are known contexts, not held-out
prompts. See [the source/receiver experiment](FLUX-SOURCE-RECEIVER.md).

## Two frontiers, two different claims

At step 3's project entry, complete live-state comparison shows that each
baseline/changed pair differs only in `latent`. Obsolete image, working text and
context have been cleared. Replacing that latent with the opposite parent's
value therefore recreates that parent's entire live input state. Restoration
and reciprocal transplantation are exact replay controls. They do not discover
a new semantic law or independently predict a numerical future.

At step 3's joint block 0 entry, projected text is identical within each pair,
while image state differs. Four genuine native prefixes run only as far as the
positioned Q/K/V operands. Each stops before original attention executes, and
MDB rolls back to the unchanged source cut. This custom CUDA dispatcher boundary
is not the public CPU-only `record_prefix` API. Refused prefix attempts remain
in the receipt stream.

All eight local mathematical forecasts are sealed with `ForecastTransaction`
before any new native attention returns. Each predicts full text attention head,
after-attention residual, receiving MLP output and final text state. The later
images are observed native continuations; this assay does not numerically
predict those entire image futures.

## The factorial reader

Text queries, text keys/values, text residual and modulation remain baseline.
Image queries and residual also remain baseline in every reader branch.
Only the image portions of positioned keys and values can change:

| Corner | Image keys | Image values |
|---|---|---|
| `k0v0` | baseline | baseline |
| `k1v0` | changed | baseline |
| `k0v1` | baseline | changed |
| `k1v1` | changed | changed |

For text queries `Q`, with concatenated text/image operands, the equations are

```text
A(K,V) = round_BF16(softmax(Q K^T / sqrt(d)) V)
U = round_BF16(T + round_BF16(g_attention * linear_out(A)))
N = modulate(layer_norm(U), shift_mlp, scale_mlp)
[u,w] = linear_in(N)
M = linear_out_mlp(round_BF16(SiLU(u) * w))
T_next = round_BF16(U + round_BF16(g_mlp * M))
```

The independent reference uses NumPy FP64 arithmetic at declared BF16
boundaries and original checkpoint parameter bits. It is not a bit-exact CUDA
GEMM/SDPA emulator. It consumes observed native Q/K/V, so its coverage begins
there. Original weights are retrieved on demand using source-verified recipes;
no response parameters are fitted and no full parameter bank is exported.

For any measured field `F`, retain these finite contrasts:

```text
key effect   = F10 - F00
value effect = F01 - F00
interaction  = F11 - F10 - F01 + F00
total effect = F11 - F00
```

Keys change the attention distribution and values change the weighted content.
Their finite interaction need not vanish. Effects at the head, receiving MLP,
text exit and terminal image are different observables; RMS across those spaces
does not assign causal percentages or object concepts.

## Execution and evidence

One finite resident mrun lease stages four stopped prefixes, eight local
mathematical forecasts, eight latent closure futures and eight reader futures.
The complete planned census is one model load, 440 successful native operation
transitions, 400 completed transformer blocks/attention calls, four stopped
block entries, eight typed latent Acts and sixteen actual VAE images. There are
zero new prompt encodes or noise draws. Port edits inside attention are explicit
instrument overrides, not advertised as typed public Acts.

Every successful exit keeps full tensor descriptors. Complete arrays remain at
project, joint0, joint4 and terminal commit where applicable; full cuts remain
at project, joint0 and terminal commit. All four predicted text fields remain
complete. Source identity, original weight recipes, request, prediction seals,
native operands, intervention hashes and all 452 receipt rows are retained.

The tiny native integration test executes the same complete campaign with a
random small transformer and an explicitly synthetic image consumer. It tests
instrument mechanics; scientific results require the original real model.

## Reproduce

Use the qualified installed public MDB wheel, with the prior authenticated
campaign present. Real-model execution goes through mrun; the final two commands
only read saved data:

```bash
python -I stage_flux_latent_reader.py RUN --prior SOURCE_RECEIVER_CAMPAIGN
python -I stage_flux_latent_reader.py RUN --submit
python -I recover_flux_directory.py RUN COLLECTION
python -I RUN/payload/analyze_flux_latent_reader.py \
  COLLECTION/extracted/binding-minimal-result audit.json
python -I publish_flux_latent_reader.py \
  COLLECTION/extracted/binding-minimal-result audit.json RECORDS
```

The publisher creates fourteen generic computation occurrences: eight local
write predictions and six key/value/interaction contrasts. Applicability names
the exact observed frontier, backend, clock and occurrence. Changed horizons
abstain. Indexing records does not establish cross-context family generality.

Run results and remaining limits are recorded below after custody verification.
