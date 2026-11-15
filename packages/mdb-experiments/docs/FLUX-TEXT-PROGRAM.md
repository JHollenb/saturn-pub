# Tracing and predicting the continuing text program

Two completed, source-qualified Klein4B assays extend the eye-program workflow backward: from a current image attention write to the incoming text residual, its attention/MLP updates, and the conditioning projection that initializes it. These are maintainer qualification workflows over the installed public runtime, not new first-class internal-port commands.

## Findings

At step3/single8, two inputs differ only in working text T. One carries the earlier additive color/shape program; the other carries the measured joint program. After observing both native blocks, all edited futures use the same joint image exit and identical nontext state. A typed text-only Act installs each selected text exit:

```text
T_exit = BF16(R_selected + BF16(g * P[A_selected; M_selected]))
```

R, A and M each select low/additive or high/joint operands. Projection P and gate g are common native consumers. The eight corners distinguish carried state from current writes. The joint residual has the largest tested effect on the supplied eye contrast, but M is contextual: the joint M produces amber/gold-looking eyes on additive R and stronger violet-looking eyes on joint R. It cannot be named a context-free violet write. R000 is not the unmodified native-low image: the cube uses the common high image exit.

Before the new block executes, complete text attention and MLP fields are predicted from observed input, actual weights, modulation and position. All 512 text rows are retained: 1,572,864 attention and 4,718,592 MLP values per role. A separately implemented saved-only reconstruction matches each sealed reference exactly. Relative RMS discrepancies against native CUDA BF16 are 0.132077–0.133199% for attention and 0.024277–0.025996% for MLP. Joint predicted A+M futures differ from their matched measured corners by 0.120405 and 0.120790 RGB levels on the 0–255 scale.

The second assay fixes base last-step latent history L3 and swaps conditioning E at the legal project entry. A derived project frontier reproduces the historical pre-single6 base payload exactly in 12 native operations, and native base reproduces the historical PNG. This is a reconstruction from saved L/E/schedule, not a historical rewind capture.

Six encodings are tested: observed base/color/shape/joint, additive `B(Ec+Es−Eb)`, and interaction-only `B(Eb+Ej−Eadd)`. Joint and additive both give violet-looking eyes with different hue, geometry and shading; their whole-image MAE is 1.222418 RGB levels. The interaction-only encoding retains amber-looking eyes. It is not a portable violet command.

Six complete initial text states are sealed before any new projector entry:

```text
T_initial = BF16(E @ W_context.T)  # actual qualified layer has no bias
```

Independent FP64/BF16 reconstruction is exact against the sealed references. Native-relative RMS error is 0.255719–0.256175%. Predicted additive/joint T are installed through typed Acts and consumed by the remaining native denoiser, scheduler and VAE; final RGB MAE versus measured-T futures is 0.284126/0.296165. Maximum pixel errors are 35/50, so the average does not establish uniform pixel accuracy.

Sources are observed prompt encodings. Neither assay predicts their formation through the text encoder or the whole image from raw text. Coarse hair/clothing/framing persist, but fine preservation, independent seeds, general object relations and setup-inclusive speedup remain open. Supplied RGB projections mix hue and geometry; they are not calibrated semantic scores.

## Reproduce or audit

Use an installed MDB/Saturn-pub environment with the qualified model dependencies. Runtime imports must resolve installed public packages. The explicitly staged qualification files use `runpy` to share experimental helpers without private runtime imports.

```sh
# Stage only, then inspect the immutable request and preflight.
python -I qualification/submit_flux_text_program.py NEW_TEXT_RUN \
  --source WORKSPACE/artifacts/mdb-interaction-corrections-20261104
python -I qualification/submit_flux_conditioning_program.py NEW_CONDITIONING_RUN \
  --source WORKSPACE/artifacts/mdb-predicted-future-20261104

# One finite mrun lease per staged assay; sequential CUDA, strict preflight.
python -I qualification/submit_flux_text_program.py NEW_TEXT_RUN --submit
python -I qualification/submit_flux_text_program.py NEW_TEXT_RUN --collect JOB_ID

# Audit existing results without loading a model.
python -I qualification/analyze_flux_text_program.py TEXT_RUN NEW_AUDIT.json
python -I qualification/analyze_flux_conditioning_program.py CONDITIONING_RUN NEW_AUDIT.json \
  --source WORKSPACE/artifacts/mdb-predicted-future-20261104
python -I qualification/publish_flux_contextual_atlas.py RUN AUDIT.json NEW_ATLAS_DIR
```

Use the conditioning launcher analogously for submit/collect. Input paths bind prior remote job-local CAS, byte/size hashes and a pinned checkpoint; historical source files must remain available on the source host. The launcher declares the actual CUDA BF16 backend, one load, zero encodes, no retries, and a measured 10GB RAM/13GB VRAM exception. New workloads require fresh sizing rather than copying that exception indiscriminately.

## Evidence and costs

Text job `job-da6d4cedfd1d`: 13 images, 171 native transitions, 11 Acts, 182 receipts, two source blocks, two full-block checks and six cached projections. Worker time 51.683s; runner 100.886s. Conditioning job `job-e23befc395c8`: eight images, 234 transitions, eight Acts and 242 receipts. Worker time 55.579s; runner 91.105s. These dependent futures inherit saved-prefix costs. Reference prediction and validation remain work; these times do not prove speedup.

Public EquationCatalog/ComputationAtlas materialization adds four text A/M and six conditioning-projector occurrences. Ten exact-context queries match one record each; twenty changed-clock/receiving-state queries match zero. Laws remain generic, with explicit source, state, clock, consumer, effects and counterexamples. Shared A+M futures are not represented as two independent RGB predictions.

The [text audit](../../../../artifacts/mdb-text-program-20261104/analysis-v1.json), [conditioning audit](../../../../artifacts/mdb-conditioning-program-20261104/analysis-v1.json), [text visual review](../../../../artifacts/mdb-text-program-20261104/visual-review.json), [conditioning visual review](../../../../artifacts/mdb-conditioning-program-20261104/visual-review.json) and [combined offline evidence index](../../../../artifacts/mdb-text-program-20261104/evidence-v1/provenance.json) retain the results. Frozen offline evidence tooling indexes ten source-qualified addresses and 31 numeric rows; run, verify and query return 0. Fifty-six focused installed-environment predictor, replay, atlas and equation checks pass.

The [fresh-context frozen-composition test](FLUX-FROZEN-COMPOSITION.md) now executes that next cut. Violet color transfers under addition, while a fixed old interaction worsens the frozen eye-window match; geometry remains unresolved. Tracing the interaction through the prompt encoder is a separate upstream mechanism question. First-class typed R/A/M/gate ports and reusable paired continuation capture would reduce the qualification-specific reconstruction needed for both.
