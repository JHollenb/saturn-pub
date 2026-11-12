# Frozen Explorer predictor on MDB binding futures

The frozen Explorer reader/writer retains part of the binding response but is
worse than holding the actual previous velocity on all eight tested image and
velocity endpoints. The hold branch preserves most of the large combined
residual/write contrast through the final image. This gives a useful boundary
for the fitted predictor and evidence that much of this late contrast already
survives in the supplied current latent and previous velocity.

These are eight previously observed development conditions: four arms in each
of two contexts at seed 109. They are not eight independent contexts or new
held-out generalization examples. This assay does not separate the information
in the current latent from that in the previous velocity.

## What was actually run

The unchanged predictor comes from Time Circuit Explorer's source-native-reader
job `job-1892a00329e9`. Its ten selected original tensors implement seven affine
maps, three head RMS normalizations and the first joint block's image attention
read. An existing fitted writer consumes that read with the current latent and
the branch's own previous velocity. Its coefficients, feature normalization,
bias, unit gain and rounding were frozen before this assay; no fitting occurred.

The inputs are the final-clock states from the [residual/write binding
campaign](FLUX-BINDING-RESIDUAL-WRITE.md), job `job-7fd0c6eae7f6`. Each branch uses
its actual v2, which produced its Z3, and the conditioning reset used by its
original continuation. The earlier Explorer main panel used held v1; this panel
removes that mismatch while retaining the other context/distribution changes.

One successful mrun CUDA B1 job, `job-3e2d8f901427`, ran eight selective reader
calls, eight unchanged CPU writer products and sixteen candidate/hold VAE
decodes. All candidate and hold futures were globally sealed before historical
future arrays were opened. Two subsequent native VAE witnesses matched the
original velocity, final latent, decoder input, decoded output and RGB exactly.
All eight native scheduler and decoder-argument joins also matched.

Two earlier attempts stopped before any reader forward: one omitted the clock
constant files; the other supplied an ActSpec instead of its identity to the
registry. Their failed requests and logs are retained separately as mechanics
evidence. They contribute no scientific outcomes.

## Mathematical account

Let Z3 be the supplied latent, E the supplied contextual conditioning, and v2
the velocity actually consumed by the preceding scheduler step. The frozen
reader forms H0 from the first joint block's image queries attending to image
and text keys/values, using the original checkpoint maps, head normalization,
rotary positions and qualified BF16 attention dispatch.

At each image position, concatenate the 128 latent coordinates, 128 previous
velocity coordinates and 3072 attention coordinates:

```text
phi = [Z3, v2, H0]                       # 3328 coordinates
Delta = ((phi - mean) / scale) B + bias # frozen FP64 affine writer
v3_candidate = BF16_RNE(FP32(v2 + Delta))
v3_hold = v2
```

The actual finite native scheduler then commits each branch. In schematic
notation, preserving the source's casts and operand layout:

```text
Z4 = BF16(FP32(Z3) + (sigma4 - sigma3) * FP32(v3))
sigma3 = 0.7174965739250183; sigma4 = 0
```

The decoder argument uses MDB's qualified CPU ID scatter, inverse BN
normalization and unpatch operation before the unchanged CUDA VAE. This CPU
normalization differs from the earlier Explorer GPU normalization. Two exact
native image witnesses establish the relevant consumer mechanics here.

For each field Y, the signed effects are Y_arm - Y_Q. The complete finite
factorial interaction is:

```text
J(Y) = (Y_RW - Y_R) - (Y_W - Y_Q)
```

All differences use FP64, including subtraction of unsigned RGB arrays. The
interaction may contain inherited state interaction, attention nonlinearity,
finite arithmetic and VAE nonlinearity. It does not by itself identify a new
AND gate or locate where the interaction originated.

## Measurements

RGB RMS is measured on original 0–255 channel values against the same-state
native reference. Velocity error is RMS error divided by native velocity RMS.

| Context | Method | Velocity relative RMS, four arms | RGB RMS, four arms |
| --- | --- | --- | --- |
| Anchor | Frozen predictor | 103.23–104.46% | 25.78–28.14 |
| Anchor | Hold own v2 | 24.26–24.65% | 6.40–7.29 |
| Wording | Frozen predictor | 103.56–104.43% | 29.18–30.39 |
| Wording | Hold own v2 | 18.52–18.73% | 6.94–7.55 |

The causal contrasts remain informative despite those endpoint misses:

| RGB quantity | Anchor predictor / hold | Wording predictor / hold |
| --- | --- | --- |
| Combined RW-minus-Q response cosine | 0.8766 / 0.9952 | 0.8535 / 0.9916 |
| Full interaction cosine | 0.8543 / 0.9923 | 0.8578 / 0.9912 |
| Full interaction relative RMS error | 53.23% / 13.11% | 51.52% / 23.98% |

The frozen predictor has nonzero, substantially aligned combined responses,
while adding visible pink/purple bias and patterned surface artifacts. The
hold images preserve the native anchor's blue-sphere/red-cube assignment and
the wording context's partial purple-sphere/red-cube result. These are bounded
visual observations, not scores from a calibrated object-binding evaluator.

The panel is heterogeneous: on wording's two small individual RGB responses,
the predictor has slightly lower difference RMS error than hold, despite worse
response cosine. It loses all eight endpoints and the large combined effects;
that does not erase those individual contrast observations.

Hold exaggerates the combined RGB response magnitude by about 20% in both
contexts. Its interaction magnitude is 1.03 times native for anchor and 1.19 times
for wording. Near-unity cosine therefore does not imply exact amplitude.
Consumer stages also differ: the candidate's interaction cosine is 0.941/0.948
in raw decoded output and 0.854/0.858 in PNG RGB. Postprocessing changes the
measured effect, so both stages remain in the evidence.

The working inference is that much of the tested assignment contrast already
survives in Z3 and/or v2 at this late boundary. It is not a universal permission
to skip the final model computation: hold has nonzero velocity and pixel error,
and other prompts, seeds or earlier clocks remain untested. The failed fitted
writer does not show that H0 lacks useful information, and this experiment does
not isolate H0's contribution from its other two inputs.

## MDB execution and evidence

The [worker](../qualification/flux_frozen_reader_bridge.py) uses public v2
StateCuts/LocalStore, typed Frames and a declared-read Act, plus one global
ForecastTransaction. Input content and recorded strides are checked. The
registered Act records the declared ABI; the worker separately checks actual
H0 shape/dtype and retained finite values.

The successful seal is
`80f4a4c1fb62d7dd1388e0b5007cbf19194ec890d2b6988a0977642968fa5871`.
The transaction's seven-event chain verifies as closed. Two real future-file
canaries were denied; there were zero future-file opens before release.

Transaction verification and external-file verification are separate: MDB seals
the metadata containing external file hashes. The worker independently rehashes
145 prediction files after evaluation, and the collector rechecks those files
plus executed input sources and before/after inventories through authenticated
SSH. Python audit hooks are trusted-code instrumentation, not an adversarial OS
sandbox. Existing case metadata and historical development outcomes were
already known; no fresh held-out claim is made.

The generic Saturn evidence sweep and verify commands still report
`evidence-incomplete-empty-target` for this MDB-specific schema. Their missing
adapter is distinct from the completed dedicated transaction, external-file,
source-custody and native-consumer checks. The existing adapter request is
upvoted; no scientific null is inferred from that instrument limitation.

The successful job took 38.04 scheduler seconds, peaking at 2118.6 MB RSS and
1726 MB VRAM. It charged 56 original affines, eight attention reads, eight fitted
writer products and eighteen VAE decodes. There were zero new full denoisers,
encoders or coefficient fits. Historical prefixes and source assistance remain
inherited costs; this is not arbitrary-prompt or zero-neural generation, and
the runtime is not a matched speedup benchmark.

The compact collection contains 67 files / 7,968,286 bytes; the remote raw tree
contains 219 files / 133,146,362 bytes, excluding the separately retained job log.
It retains 152 original parameter retrieval recipes instead of pretrained
weight copies. The newly fitted writer coefficients remain under separate
custody because they cannot be retrieved from the original checkpoint. Its
original receipts contain original paths; relocation requires an explicit
byte-verified remapping receipt.

All 219 raw files were subsequently copied to the durable directory
`/mnt/big/mdb/research/job-3e2d8f901427/flux-frozen-reader-bridge-result`,
with source-before/source-after and copied-file hashes checked. The separate
durable custody receipt has SHA256
`a30c00630997f2613ae550ea0f8ccf1506144d20971a4ab87bbb4d4264b6d2e5`.
Its explicit worker-root mapping preserves original serialized receipts.

Workspace artifacts are under
`artifacts/mdb-explorer-binding-bridge-20261106/`: the frozen plan, launcher,
failed attempts, successful `run-03/compact`, original-image contact figures,
scalar analysis, independent review and retrieval metadata. The external
source task was not edited or messaged by this assay.

Two curated computation records retain the approximate predictor's counterexamples
and the hold continuation's positive trend in the existing contextual binding
family. They do not become exact equation-catalog entries or transferable rules.

## Next discriminating experiment

First distinguish persistence from where the distinction is carried. At a
common late parent, cross the original Q/RW Z3 and v2 operands while holding
conditioning, schedule and decoder fixed. Predict and seal all four hold
continuations before execution. The two diagonal cells per context already
exist; only four crossed scheduler/VAE continuations are new across both
contexts. This needs no new denoiser or attention read.
It can distinguish latent-carried contrast, velocity-carried contrast and their
receiver-dependent combination. That is a bounded state-carrier question;
the synthetic crossings do not establish natural reachability or explain how
the earlier binding block created those operands.

The earlier post-joint1 image/text crossing remains the separate cut for
locating formation and transport. A reusable computation record should bind
the actual receiving state, clock, writer, signed consumer effect, continuation
and counterexamples. Refit quality on this development panel would not itself
establish a portable law.
