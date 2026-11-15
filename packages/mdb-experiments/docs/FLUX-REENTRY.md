# Sealed FLUX prediction through a commit and model reentry

Follow-up: [the joint1 chain and matched native bridge](FLUX-JOINT-CHAIN.md) extends the sealed forecast to19 transitions and separates transported upstream discrepancy from local joint1 discrepancy on identical predicted incoming state.

This experiment extends the [remaining-step forecast](FLUX-STEP-BOUNDARY.md) through the next learned projection and first joint transformer block. Starting from an observed logical-step2/pre-single6 state, it predicts fourteen complete single blocks, readout noise, the Euler latent commit, the complete step3 projection and joint0. Each stage consumes its own predicted predecessor. Actual native continuations then consume the admitted predicted latent, context and working streams together.

**Status: actual native run and saved-only audit completed; terminal status unassessed.** The sealed prediction crossed the latent commit, regenerated complete projection/context and joint0, and was admitted through the four-port closure before five actual native suffixes and VAE decodes. Native numerical differences remain nonzero. The [accepted audit](../../../../artifacts/mdb-flux-reentry-20261104/audit.json) and [public equation/Atlas publication](../../../../artifacts/mdb-flux-reentry-20261104/equation-publication-v1/result.json) retain the measured errors and their source bindings.

The finite scientific profile is FLUX.2 Klein 4B distilled, CUDA BF16, batch1, 512×512, four denoising steps, one newly encoded cone/cylinder prompt and one initial noise draw with seed97. It is one source history with five dependent futures. It supplies a native prefix and complete actual parameters; it does not predict from a raw prompt alone.

## Measured result

Job `job-5946cdd60347` succeeded in315.59 runner seconds, with peak RSS8654.4MB and VRAM9588MB. The forecast sealed at217.839779 worker seconds with all24 selected component counters zero. Prediction start to seal took50.590589s. The worker summary records285.606464s before final manifest hashing. [Authenticated collection](../../../../artifacts/mdb-flux-reentry-20261104/run-02/collection.json) retained654 files /5,310,311,478 bytes, with exact producer, before/after remote and local inventories.

The independent saved-only audit took65.1547s. All76,201,984 output scalar values match the independently implemented recurrence bit for bit: fourteen complete I/T outputs, noise, committed latent, projected I/T, complete context/RoPE and joint0 I/T. This establishes agreement between two implementations of the declared numerical recipe; it does not establish native CUDA bit parity.

| Sealed output versus corresponding native reference | Error RMS / native RMS |
|---|---:|
| Committed Z3 | 0.202351% |
| Projected image | 0.283825% |
| Projected text | 0.256406% |
| Joint0 image | 0.430989% |
| Joint0 text | 0.301607% |

Regenerated RoPE is native-bit-exact. The time embedding differs in1 of3072 values; image, text and single-stream modulation differ in49/18432,52/18432 and18/9216 values, respectively. These differences matter to the scope: the predicted-project arm carries its predicted context instead of refreshing it from native values.

All five final images use the actual26-operation suffix and VAE. The native and complete-closure sham have exact final latent and PNG equality. Relative to the native arm:

| Arm | Common joint1 image L2 | Final Z4 error RMS / native RMS | RGB channel MAE, scale0–255 | Maximum channel difference |
|---|---:|---:|---:|---:|
| `native` | 0 | 0% | 0 | 0 |
| `sham` | 0 | 0% | 0 | 0 |
| `latent.predicted` | 7.702568 | 0.860685% | 0.0642980 | 6 |
| `project.predicted` | 7.940002 | 0.889146% | 0.0646591 | 7 |
| `joint.predicted` | 8.328727 | 0.916151% | 0.0656815 | 7 |

Extending the predicted extent slightly increases both final latent and RGB discrepancy in this history. Small RGB MAE is a pixel observation, not an independent semantic score or a general error guarantee. See the [native VAE comparison figure](../../../../artifacts/mdb-flux-reentry-20261104/native-futures.png) and [state-error observations](../../../../artifacts/mdb-flux-reentry-20261104/state-errors.png). The exact sham PNG is retained with the four plotted outputs.

## Forecast boundary and equations

A clean native prefix executes68 operations: two complete28-operation denoising steps, then the step2 projection, five joint blocks and six single blocks. The observed pre6 state supplies full image/text registers, stored Z2, conditioning, position IDs, the timestep/sigma schedule, and step2 time/RoPE context.

The [unchanged single-block reference](../qualification/flux_transition_reference.py) predicts blocks6..19 over the complete concatenated register `[text,image]`, including all queries and keys. Each block's complete predicted I/T output becomes the next block's input. Its attention and MLP features are parallel consumers of the same modulated incoming register; their concatenation passes through the actual output projection, gate and residual addition.

Readout consumes the predicted post19 image and observed step2 time embedding. It applies SiLU and learned adaptive-normalization modulation, population LayerNorm, scale/shift and the actual output projection with explicit BF16 boundaries. The deterministic commit uses the qualified weak-scalar arithmetic:

```text
dt = float32(sigma3 - sigma2)
update = BF16(BF16(dt) * predicted_noise)
predicted_Z3 = BF16(float32(observed_Z2) + float32(update))
```

The native-profile fixture checks the zero-dimensional FP32 coefficient multiplied by a dimensional BF16 tensor on actual CUDA before loading the model. This fixture establishes the declared scalar promotion for its hostile cases; it is not a proof of every CUDA kernel's arithmetic.

The commit increments the logical step to3, sets `phase="project", cursor=0`, clears image/text/context/noise, and preserves conditioning, IDs and schedule. The [new reentry reference](../qualification/flux_reentry_reference.py) then predicts the complete next projection:

- Learned `x_embedder` consumes its own predicted Z3; `context_embedder` consumes the carried actual conditioning.
- The scheduled step3 timestep follows the adapter's BF16 cast, division by1000 and multiplication by1000. The time embedding and all image/text/single-stream modulation tensors are computed from actual learned weights.
- RoPE is regenerated from the carried image/text IDs, actual axes dimensions and theta. Native future context or RoPE is not a forecast input.

The result includes both projected streams and every context field: `temb`, `img_mod`, `txt_mod`, `single_mod` and the FP32 cosine/sine pair.

Joint0 differs from the parallel single-block law. For image and text separately, it normalizes and modulates the incoming stream, forms stream-specific Q/K/V and learned Q/K normalization, and attends over concatenated text and image. Each stream receives its own projected, gated attention residual. **A second LayerNorm/modulation and the stream-specific MLP consume that updated residual.** The joint MLP is serial after the attention write; the single-block parallel A/M formula is not reused.

The full forecast therefore has this dependency chain:

```text
observed pre6 → predicted single6..19 → predicted readout noise
             → predicted Z3 → predicted complete project/context
             → predicted joint0 image AND text
```

The references use FP64 reductions and explicit BF16 operation boundaries, plus declared FP32 timestep/RoPE arithmetic. They are full-parameter numerical programs, not compressed learned circuits or exact CUDA/SDPA emulators. Independent reconstruction agreement and native numerical agreement are separate quantities to be measured.

## Typed complete-state admission

The base Saturn-pub FLUX adapter permits latent/conditioning writes at a project boundary and image/text writes at live transformer boundaries. It forbids replacing context and latent at joint entry. Replacing only predicted I/T there while retaining native latent or future context would give the next native consumer a different state from the forecast.

The public opt-in [FluxForecastClosureAdapter](../src/mdb/flux_forecast_closure.py) adds a distinct execution/surface contract for this experiment. It inherits native advancement unchanged. At actual native joint cursor0 or1, `closure_act` admits exactly one atomic replacement of all four ports:

```text
{latent, context, image, text}
```

The Act supplies the complete recursive context, copies values into the reference tensor layouts without arithmetic casts, and validates shapes, strides, dtypes, devices and finite values. All thirteen source-state ports, including clock, conditioning, IDs and schedule, are bound by digest preconditions. Clock and non-closure fields remain read-only. Partial context/latent admission and forged cursor changes are refused.

The predicted-project and predicted-joint arms carry their own predicted Z3 and complete predicted project context alongside I/T. Native parent cuts supply qualified boundary clocks and unchanged carried state; they do not correct the predicted four-port values. The base adapter source remains unchanged and separately hash-bound. The new opt-in module has its own source identity and execution contract.

## Five native futures

One global `ForecastTransaction` seals the entire remaining-step, project and joint forecast, its input binding and hypotheses before any selected current-step or next-step target runs. Model-module guards cover every native component during prediction. Earlier native prefix operations are permitted inputs; zero selected-target entries before the seal refers to the declared later frontier, not the model class's entire execution history.

After sealing,18 reference transitions execute fourteen single blocks, readout, Euler commit, step3 projection and joint0. All five branches meet at actual logical-step3/joint1, immediately after joint0:

| Arm | Typed admission | Native bridge before the common boundary |
|---|---|---|
| `native` | None | Shared native reference |
| `sham` | Exact native latent/context/I/T, one complete-closure Act | None |
| `latent.predicted` | Predicted Z3, one latent-only Act at step3/project | Native project and joint0 |
| `project.predicted` | Own predicted Z3/context/projected I/T, one complete-closure Act at joint0 entry | Native joint0 |
| `joint.predicted` | Own predicted Z3/context/joint0 I/T, one complete-closure Act at joint1 entry | None |

Every branch then executes the same26 native operations: joint1..4, single0..19, readout and scheduler, followed by actual VAE decoding. The forecast does not independently compute this remaining suffix or final RGB.

The completed physical census is:

```text
68 prefix +18 reference +3 bridges +5×26 suffix =219 native transitions
1 latent Act +3 complete-closure Acts =4 Acts
219 +4 =223 receipt occurrences; 5 VAE/PNG futures
```

There was one model load, prompt encoding, initial noise draw and finite lease, with no retries or per-arm jobs. Receipt occurrences remain distinct from deduplicated receipt files. The accepted audit verifies that after the seal each selected single6..19/readout component ran six times across the current reference and final-step suffixes. Each next-project root ran twice, position embedding ran four times, and joint0 ran three times.

## What the contrasts can resolve

At the common joint1 boundary, let `Y0` be native, `Y1` the latent-predicted arm, `Y2` the project-predicted arm and `Y3` the joint-predicted arm. Saved complete vectors allow the ordered identity:

```text
Y3 - Y0 = (Y3 - Y2) + (Y2 - Y1) + (Y1 - Y0)
```

`Y3−Y2` compares the predicted joint law with native joint0 on the same predicted project input. `Y2−Y1` compares predicted versus native project state after transport through native joint0, conditional on the same predicted Z3. That state includes persistent context, so it is a complete project contrast rather than an image-embedding-only contrast. `Y1−Y0` measures the earlier committed-latent difference transported through native project and joint0.

The identity is an ordered finite-difference account. It does not identify a unique causal attribution, equal `native_joint(error)`, or make scalar error norms additive. Image, text, context, readout noise, committed latent and RGB retain their own units and denominators. Signed cross terms, near misses, counter-trends and disagreement between latent and RGB observations must remain visible. No accuracy threshold or semantic score may erase a mechanically valid branch.

For the saved common image vectors, the inherited, project and joint terms have L2 norms7.702568,5.557266 and7.529549. Their pairwise cosines are−0.317358,−0.161239 and−0.378400, and the vector identity closes exactly. Negative cross terms partially cancel the component discrepancies, yet total image L2 rises7.702568→7.940002→8.328727 as project and joint predictions are added. Cancellation here is numerical; it neither establishes downstream repair nor makes the separate terms unique causal contributions.

## Evidence and qualification

The [worker](../qualification/flux_reentry_campaign.py) archives complete actual BF16 parameter banks, loads one bank at a time for prediction, and releases it before the next stage. Fourteen full single-block outputs, full predicted project/context/joint outputs, readout and commit remain available. Split aliases are omitted from the fourteen-block transaction tree. The forecast budget is768MiB; output bounds are8GiB total and1GiB per member.

Native reference exits and bridge exits retain complete cuts and closure arrays. All five common inputs and final committed cuts are retained. The native final-step reference has full arrays at27 aligned boundaries; intermediate candidate records contain worker-observed full norms, descriptors and64 fixed coordinate witnesses. Those witnesses do not independently establish every full intermediate candidate norm or complete cut.

Source checks distinguish the unchanged base adapter, native normalization/scheduler/project/joint sources and the new closure module. Qualified prior scientific helpers are copied from the immutable earlier payload, rather than a possibly dirty checkout. Mathematical source identities and the original audit source were frozen before execution.

The [collector](../qualification/collect_flux_reentry.py) uses predeclared authenticated SSH manifest collection. It requires the actual succeeded request-bound scheduler job, checks the executed command/payload, brackets the file copy with remote inventories, and hashes every local file. A producer manifest covers every regular file except itself. No stdout archive or fabricated transport report is created. The [auditor](../qualification/analyze_flux_reentry.py) reads saved evidence only, checks typed closure/source/census/custody and independently reconstructs the mathematical stages. Collection and audit do not rerun inference.

The original offline audit stopped on a nested typed-context reader error: unpacked Act context parameters were compared with still-typed StateCut nodes. The [explicit correction receipt](../../../../artifacts/mdb-flux-reentry-20261104/audit-correction-v1/correction.json) binds original auditor `1c6aa030…` and corrected reader `b4cdcc32…`, the [preserved failed attempt](../../../../artifacts/mdb-flux-reentry-20261104/audit-attempt-01/result.json), raw spot checks and regression tests. This was a post-outcome instrument repair. No worker, mathematical recurrence, sealed forecast, native evidence, criterion or model execution changed. The publication binds this corrected authority explicitly.

The [final isolated qualification](../../../../artifacts/mdb-flux-reentry-20261104/combined-qualification-06.json) passes69 tests in12.14s:49 core mechanics/reference/launch/audit tests and20 publication tests. It includes the [tiny five-future campaign](../tests/test_flux_reentry_campaign.py), typed closure tests and hostile source/clock/custody cases. Tiny image consumers are synthetic; the separate real run supplies all five actual VAE outputs.

## Public equation and Atlas publication

The [publisher](../qualification/publish_flux_reentry.py) uses installed public `EquationCatalog`, `compare_equations`, `applicability` and `ComputationAtlas` APIs. It produced10 curated records over eight named field comparisons from this one history. Records bind full observed/predicted arrays through source references and retain signed residual summaries, applicability predicates, actual input cuts, source/model identities, forecast seal and explicit corrected-auditor authority. Some records reuse comparisons; ten records are not ten independent experiments.

The exact-context query returns10 matches. Changing the seed returns0; omitting the input cut returns0 matches with applicability `unknown`. These check supplied predicate matching, not empirical generalization. No family candidate or automatic rule was induced. Historical step-boundary records are links, not pooled replications.

The equation API reports observed-minus-predicted residuals; the numerical audit reports predicted-minus-native errors. Continuation effects explicitly compare the installed branch with native. Norms agree across sign conventions, while signed summaries retain their declared direction. A [live public Atlas query](../../../../artifacts/mdb-flux-reentry-20261104/atlas-query-observation.json) finds both joint-law records without importing Torch or loading a model.

The [standard private evidence sweep](../../../../artifacts/mdb-flux-reentry-20261104/evidence-v1/pipeline-run-summary.json) returned0 but accepted0 claims and found0 address citations from these public MDB receipts. That is an instrument coverage gap, not additional scientific certification or missing experimental evidence. It is recorded in the [public MDB evidence-import request](../../../../artifacts/mdb-flux-reentry-20261104/request-public-mdb-evidence-plane-import.json); the public equation/Atlas publication above did execute successfully.

## Maintainer reproduction

Build and install an MDB wheel containing `mdb.flux_forecast_closure` **before staging**. The launcher packages installed public distributions; editing a checkout after installing an older wheel does not add the new module to a remote payload. Use an isolated environment containing qualified public MDB, Saturn-pub and mrun-pub wheels plus the declared diffusion dependencies. Qualification scripts run from the maintainer source tree; runtime imports resolve installed packages, with no sibling checkout import or `sys.path` modification.

For example, from the Saturn-pub repository root, with `$PY` naming that environment's Python and its build dependencies available:

```sh
rtk proxy "$PY" -m pip wheel --no-deps ./packages/mdb --wheel-dir /absolute/path/to/wheels
rtk proxy "$PY" -m pip install --no-deps --force-reinstall /absolute/path/to/wheels/mdb_runtime-0.1.0-py3-none-any.whl
rtk proxy "$PY" -I -c 'from mdb.flux_forecast_closure import FluxForecastClosureAdapter, closure_act; print("closure module installed")'
```

Retain the qualified earlier step-boundary run, including its request, payload and independent mathematical source snapshots. The launcher verifies that source authority; it does not substitute current legacy helper files. Select a new empty run directory and the exact earlier run:

```sh
export MRUN_URL=http://192.168.68.68:9025
rtk proxy "$PY" -I packages/mdb/qualification/submit_flux_reentry.py /absolute/path/to/new-run --prior-run /absolute/path/to/qualified-step-boundary-run
# Review the staged request, source/wheel pins, actual command, storage and resources.
rtk proxy "$PY" -I packages/mdb/qualification/submit_flux_reentry.py /absolute/path/to/new-run --submit-staged
# Only after the exact admitted job succeeds:
rtk proxy "$PY" -I packages/mdb/qualification/collect_flux_reentry.py /absolute/path/to/new-run job-EXACT
rtk proxy "$PY" -I packages/mdb/qualification/analyze_flux_reentry.py /absolute/path/to/new-run --out /absolute/path/to/audit.json
# Saved-only public equation/Atlas curation, with explicit accepted-auditor authority:
rtk proxy "$PY" -I packages/mdb/qualification/publish_flux_reentry.py /absolute/path/to/new-run /absolute/path/to/audit.json /absolute/path/to/new-publication
```

The earlier lane calibrated the12GiB RAM/13GB VRAM exception; this run's measured peaks are8,654.4MB RSS and9,588MB VRAM. The worker has a900-second budget and the lease a1200-second bound. Model/cache paths and Python requirements are rebound to the sealed worker-native execution intent. Failed preflight, startup, model execution, collection and saved-only audit are separate statuses; preserve each attempt rather than replacing it with a retry. Reproducing this historical corrected audit also requires its explicit correction receipt/source authority, rather than silently substituting that reader for the original pre-outcome snapshot.

This is a conditional numerical composition test across a real commit and reentry. It does not establish raw-prompt-to-image prediction, learned semantic rules, arbitrary-context generalization, automatic mechanism discovery, universal error bounds or an inference speedup. The smallest next contrast is predicted joint1 versus native joint1 from the same predicted joint0 state while holding predicted context and Z3 fixed. Extending through the remaining denoising step to a fully predicted Z4 is a later horizon, not a completed result here.
