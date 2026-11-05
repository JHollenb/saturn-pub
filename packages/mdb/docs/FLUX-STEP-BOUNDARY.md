# Sealed FLUX forecasts across a denoising commit

This qualification extends the [two and three block forecasts](FLUX-TRANSITION-CHAIN.md) through the rest of a denoising step. From an observed step2/pre-single6 state, it predicts fourteen complete blocks, conditional readout noise and the Euler latent update. A typed latent replacement then lets the next native denoising step consume the sealed result.

The experiment binds FLUX.2 Klein 4B distilled, CUDA BF16, batch1, 512×512, four denoising steps, prompt p1 and fresh seed89. It supplies the native prefix and upstream time/RoPE context. The next step's learned projection, transformer and VAE execute natively.

## Measured result

Job `job-2223e69ff840` succeeded in271.94 runner seconds with peak RSS7216.5MB and peak VRAM9698MB. The forecast seal is `ab052dac67e5397c8abd1fece8d8cc70f51f7d55c5861010c45c7851d1edea85`, recorded at200.566780 worker seconds with zero selected current-step target entries. All native futures completed by251.874289s; the summary through output preparation precedes final manifest hashing. Prediction and seal took80.043420s from the logged prediction start.

The saved-only independent audit completed in56.606s. All66,322,432 forecast scalar values—fourteen complete both-stream outputs plus readout noise and committed latent—match the independent recurrence bit for bit. This is agreement between two implementations of the declared numerical recipe. Native CUDA errors remain measured separately:

| Quantity | RMS error / corresponding native RMS | Maximum absolute error |
|---|---:|---:|
| Post19 image register | 0.688049% | 16 |
| Post19 text register | 0.134960% | 32 |
| Readout noise | 0.464679% | 0.046875 |
| Committed Z3 | 0.199362% | 0.015625 |
| Z4 after actual native final step | 0.850047% | 0.07421875 |

The predicted branch's final RGB mean absolute channel difference is 0.1076889038 out of 255, maximum 8, with 72,761 changed values out of 786,432. The sham final latent and PNG are exact. Post19 image/text bit-exact fractions are only 21.95%/22.15%, respectively; small relative RMS is separate from native bit parity. Absolute latent-error RMS increases from 0.00153738 at Z3 to 0.00661676 at Z4, a 4.30391× increase through the native final step. This branch does not show downstream repair.

The image-register relative-error curve falls after block16 while absolute error continues to rise: RMS error goes0.190683→0.363559 from blocks16→19, while native image RMS goes18.4482→52.8391. The smaller percentage does not establish correction. Readout and latent percentages also use distinct output-space denominators.

Applying the independent Euler formula retrospectively to native noise reproduces every native committed Z3 value exactly. The analogous readout calculation on native post19 image has0.196248% relative RMS error, exposing a local numerical discrepancy as well as the carried block-chain difference. These retrospective diagnostics are separately labeled and never enter the sealed forecast.

A [saved-only readout supplement](../../../../artifacts/mdb-flux-step-boundary-20261104/readout-decomposition.json) retains the complete vectors for `P-N = (P-R) + (R-N)`, where P is sealed predicted noise, R is reference readout on native post19 image, and N is native noise. Carried-state and local reference/native differences have 0.454234% and 0.196248% relative RMS, respectively, with total 0.464679%; cosine −0.162186 shows partial cancellation. Vector and squared-norm identities close exactly. This posthoc ordered decomposition is not unique causal attribution, and raw/audited forecast evidence remains unchanged.

Evidence: [independent audit](../../../../artifacts/mdb-flux-step-boundary-20261104/audit.json), [authenticated collection](../../../../artifacts/mdb-flux-step-boundary-20261104/run-02/collection.json), [native image futures](../../../../artifacts/mdb-flux-step-boundary-20261104/native-futures.png), [error observations](../../../../artifacts/mdb-flux-step-boundary-20261104/state-errors.png). The corpus has506 files /4,313,370,016 bytes; its largest member is the283,620,110-byte forecast bundle. No inference was rerun for collection or audit. The isolated installed-wheel cohort passes52 tests; the first combined test copy omitted a tiny-test fixture and is preserved separately from the corrected successful cohort.

## Prediction and state contract

At the observed boundary, the complete residual is `H6 = concat(text, image)`, shape `[1,1536,3072]`. The fixed prompt contributes512 text positions; the image contributes1024 positions. Each block6..19 independently computes complete all-token normalization, time modulation, QKV, learned Q/K normalization, RoPE, full attention, paired SwiGLU, output projection, gated write and residual addition. Each later block consumes the preceding predicted complete output. The unchanged mathematical reference uses FP64 reductions with explicit BF16 operation boundaries; it does not emulate exact CUDA SDPA reductions.

Readout uses predicted post19 image state and the observed time embedding:

```text
u = BF16(SiLU(temb))
(scale, shift) = split(BF16(linear(u, actual norm_out.linear parameters)))
n = BF16(population LayerNorm(predicted image))
a = BF16(BF16(n * BF16(1 + scale)) + shift)
predicted_noise = BF16(linear(a, actual proj_out parameters))
```

The deterministic Euler commit follows the actual weak-scalar promotion of the qualified Torch profile:

```text
dt = float32(sigma3 - sigma2)
update = BF16(BF16(dt) * predicted_noise)
predicted_Z3 = BF16(float32(native_Z2) + float32(update))
```

Here `native_Z2` is the latent stored during logical step2; the numerical forecast starts partway through that step. A zero-dimensional FP32 `dt` multiplied by a dimensional BF16 tensor has BF16 multiplication dtype. Rounding `dt` before multiplication is observable: a tiny hostile fixture distinguishes this rule from rounding an FP32 product. The worker qualifies that fixture on actual CUDA before loading the model.

The commit resets `text`, `image`, `context` and `noise_prediction` to `None`, increments `step` to3, sets `phase="project"` and `cursor=0`, and carries conditioning, position IDs and the full timestep/sigma schedule unchanged. The next native projection regenerates working text from conditioning. Working text is not directly persisted through the commit.

## Common parent replay

One native prefix takes68 transitions to step2/pre6. All target model modules are guarded during binding and prediction. The public `ForecastTransaction` seals all fourteen complete outputs, readout noise, committed latent, applicability hypotheses and reset/carry contract before the selected current-step native targets run.

After the seal:

1. Sixteen native reference transitions execute blocks6..19, readout and scheduler, reaching actual step3/project.
2. The native arm continues28 final-step operations and decodes through the actual VAE.
3. The sham arm installs the exact native Z3 through one typed Act and executes the same continuation.
4. The predicted arm installs sealed predicted Z3 through one typed Act and executes the same continuation.

Every arm shares the native clock, complete carried state and explicit reset fields. Only latent changes in the two Acts. Readout and scheduler remain read-only adapter boundaries; no forged StateCut or cursor mutation bypasses them.

The finite census is168 native transitions,2 typed Acts,170 receipt occurrences and3 VAE/PNG futures, with one model load, one prompt encoding and one noise draw. The current step's selected blocks run once after the seal; their final-step counterparts run three times. Receipt occurrences are counted separately from content-addressed receipt files, since exact replay can deduplicate files.

## Evidence and tooling

The worker uses public MDB `NativeCutNavigator`, `ForecastTransaction`, lossless tensor archives and Saturn-pub typed state, receipts, stores and replacement Acts. Complete actual BF16 parameters are archived, then loaded as one ephemeral CPU parameter bank per forecast block. Fourteen complete output arrays are retained; split aliases are omitted. The transaction budget is768MiB, the result member limit1GiB and total output limit8GiB.

All native current-segment cuts and arrays are retained. The final-step native reference retains complete arrays at29 boundaries. Candidate intermediate observations retain full worker distance measurements plus64 fixed coordinates; those witnesses do not independently establish every full candidate norm. Candidate input/final cuts and final latent/image outputs are retained completely.

Source integrity includes exact native normalization module/method, scheduler module/method and public adapter module pins, checked before model load and against actual instantiated types before the source prefix. The independent block recurrence and separate readout/Euler auditor mathematics are frozen in the request before execution. The auditor rebuilds every stage from saved operands, checks chain input/output joins, typed closure, physical census, source and transport custody, and reports signed native errors without a precision gate.

The public worker wheel lacks the private mrun binary outbox API. This campaign therefore declares authenticated SSH manifest collection before admission. The collector requires an actual succeeded, request-bound Beast job; obtains the result directory from authenticated job diagnostics; brackets an allowlisted rsync with independent remote inventories; and hashes every local regular file again. Producer, before/after and local manifests must agree. It does not fabricate an archive or a worker report. Collection and saved-only audit invoke no model.

Four [curated computation records](../../../../artifacts/mdb-flux-step-boundary-20261104/computation-records.json) name `flux.single.transition-chain`, `flux.readout.state-to-noise`, `flux.scheduler.noise-to-latent` and `flux.denoising-step.own-state-chain`. Their evidence includes 18 numerical rows and five distinct actual StateCut citations, appearing in 11 citation occurrences. [Index provenance](../../../../artifacts/mdb-flux-step-boundary-20261104/evidence-v1/provenance.json) binds the accepted audit, source snapshots and historical two/three-block records. Index run, verification and query return zero. This maintainer step uses seven explicitly supplied private Saturn stdlib modules offline; the public MDB worker has no private imports. Family curation is manual, and automatic atlas ingestion/rule induction remain unfinished.

## Maintainer workflow

Use an isolated environment with the qualified public MDB, Saturn-pub and mrun-pub distributions and declared execution dependencies installed. Staging packages those installed bytes, never sibling checkouts. This finite profile intentionally binds the measured Beast checkpoint and Linux cache paths.

```sh
export MRUN_URL=http://192.168.68.68:9025
python qualification/submit_flux_step_boundary.py /absolute/path/to/new-run
# Inspect request, source pins, full wheel inventory, command, storage and resources.
python qualification/submit_flux_step_boundary.py /absolute/path/to/new-run --submit-staged
# After the actual scheduler job succeeds:
python qualification/collect_flux_step_boundary.py /absolute/path/to/new-run job-EXACT
python qualification/analyze_flux_step_boundary.py /absolute/path/to/new-run --out /absolute/path/to/audit.json
```

The command builder rebinds actual worker arguments to the staged execution intent, including Linux cache paths and Python version, even under hostile local defaults. Admission checks the executing launcher's source identity. A scheduler resolution failure precedes HTTP submission and must be recorded separately from an admitted failed job.

The specific conditional prediction does not establish raw-prompt-to-image prediction, semantic object naming, automatic family discovery, universal transfer or an inference speedup. Its purpose is to test composition through a real persistent-state update and authentic downstream consumers.
