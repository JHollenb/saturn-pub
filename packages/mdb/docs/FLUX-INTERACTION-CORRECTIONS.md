# Contextual interaction corrections through native continuation

This workflow follows [contextual eye programs](FLUX-EYE-PROGRAM.md). It separates a correction to one current image attention write from a correction to the program consumed by later blocks. The [blog post](../../../../obsidian/blog/2026-11-04-140929-contextual-programs-and-predictable-image-edits.md) gives the high-level account, images and mathematical interpretation.

## Experiment and arithmetic

Model: FLUX.2 Klein 4B distilled, inventory `dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f`, CUDA BF16, batch one, 512×512, four diffusion steps, seed 60665, step 3/single block 7. The complete receiving image, latent history, modulation and position operands are fixed. Color, shape, additive and joint working text come from explicitly recorded sources.

For fixed receiving I, define attention fields A0, Ac, As, Aa and Aj. The candidate operand is `BF16(Alinear + r*R + c*C)`, rounded once before native projection, with:

```text
Alinear = Ac + As - A0
C = Aa - Alinear
R = Aj - Aa
```

The four corners use `(r,c)` equal to `(0,0), (0,1), (1,0), (1,1)`. Two negative controls use `(-1,0)` and `(0,-1)`. Each main corner is crossed with actual native post7 additive or joint text. Image M, output projection and gate are native observed consumers. All branches continue through the unchanged native suffix and VAE.

The new correction forecasts and branch hypotheses are sealed before new current-block comparisons. Source fields are inherited equation calculations. Some additive/base native measurements and source-context image outcomes were already observed in earlier work; this is a prospective test of new intervention outcomes on a known specimen, not blind first exposure or donor-free prompt interpretation.

Native source blocks run for all five text roles. Only six unique cached image projections are needed for ten correction arms. Four complete full-block A-only substitutions independently check cached endpoints, with their native text output retained. Two same-value image Acts are sham controls. The three native references plus twelve edited branches give fifteen images.

## Results

| Current A | Continuing additive T: RGB MAE to joint | Continuing joint T: RGB MAE to joint |
|---|---:|---:|
| Alinear | 1.744886 | 0.215013 |
| Alinear+C | 1.781527 | 0.307824 |
| Alinear+R | 1.701174 | 0.312700 |
| Alinear+C+R | 1.697821 | 0.130053 |

For each pair, complete post-Act payloads differ only in `text`; the corrected image is identical. Current both-corrected A is therefore insufficient to supply the joint future under additive continuing T. Changing only the continuing text produces the much larger tested effect. The native joint target itself is source-assisted and has collateral relative to the additive image.

Although C and R have negative cosine before the consumer, each has a small positive effect on the historical joint-minus-additive eye-window RGB projection. Under additive T, R at C0 gives +0.014414 and C at R0 gives +0.016380. The matched T changes give +0.9835 to +0.9884 across corners. These are signed effects on a supplied RGB contrast, not semantic percentages. Negative doses decrease that contrast in the tested additive context. Mixed effects are −0.007312 under additive T and −0.002182 under joint T.

The observer is not a calibrated hue classifier: amber native base scores 0.527 on the joint-minus-additive contrast. Review RGB, eye geometry and region distances together. Current R-only improves global target distance while worsening clothing-window error; current C-only improves the eye window while worsening global error. The [full audit](../../../../artifacts/mdb-interaction-corrections-20261104/analysis-v1.json) preserves every target/collateral window and signed response.

Five complete attention fields receive native error censuses: relative RMS errors range 0.124593–0.126008%. The four earlier diagnostic fields now have complete native comparisons. All six candidate fields rebuild exactly from the declared algebra and BF16 cast. All four cached-consumer comparisons, three historical PNG controls and two shams are exact within their stated mechanics scope.

At pre-single11, the three native references retain complete I/T arrays. Candidate observations retain full-tensor descriptors, 64 fixed witnesses and worker-calculated full L2/signed-mean comparisons. Complete candidate intermediate arrays are not retained; online comparison values are source-bound observations rather than independently reconstructed trajectories.

The census is one load, zero encodes, 213 native transitions, 15 Acts, 228 receipts, five direct source blocks, four full-block validation calls, six unique projections and fifteen VAE decodes. Worker time is 49.249 seconds, runner time 94.475 seconds. Custody retains 180,803,484 archive bytes and 220 files. These counts include validation; no setup-inclusive speedup is claimed.

## Execute the workflow

Use the qualified installed-public-package environment and one finite mrun CUDA lease. The launcher hashes the explicit historical source cuts and prediction arrays; selected historical CAS files are hydrated on their source host. Runtime modules resolve installed MDB/Saturn-pub/mrun-pub, and qualification helpers are staged explicitly. No sibling checkout discovery or private Saturn runtime import is used.

```sh
python -I /absolute/MDB/qualification/submit_flux_interaction_corrections.py FRESH_RUN --source PRIOR_EYE_PROGRAM_ROOT
python -I /absolute/MDB/qualification/submit_flux_interaction_corrections.py FRESH_RUN --submit
python -I /absolute/MDB/qualification/submit_flux_interaction_corrections.py FRESH_RUN --collect JOB_ID
python -I /absolute/MDB/qualification/analyze_flux_interaction_corrections.py FRESH_RUN FRESH_AUDIT.json
python -I /absolute/MDB/qualification/publish_flux_contextual_atlas.py FRESH_RUN FRESH_AUDIT.json FRESH_CATALOG_ROOT
```

The November 4 cache profile is `MDB_WORKER_UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache`, `MDB_WORKER_TORCH_HOME=/mnt/hdd8tb/mdb/torch`, `MDB_WORKER_MODEL_ROOTS=/mnt/big/llm-models`. Use `common exec` where scheduler credentials are needed. Strict preflight, no retries, one resident model, batch one and 600-second worker/720-second lease limits apply. The 10GB RAM/13GB VRAM exception is calibrated on the preceding no-encoder lane, not hand-inferred as a performance optimization.

Job: `job-20c6f6c6773c`. [Sealed request](../../../../artifacts/mdb-interaction-corrections-20261104/run-01/request.json), [worker](../qualification/flux_interaction_corrections_campaign.py), [launcher](../qualification/submit_flux_interaction_corrections.py), [auditor](../qualification/analyze_flux_interaction_corrections.py), [correction algebra](../qualification/flux_interaction_corrections.py), [visual review](../../../../artifacts/mdb-interaction-corrections-20261104/visual-review.json).

## Contextual symbol records in the public atlas

The [materializer](../qualification/publish_flux_contextual_atlas.py) uses the existing public `EquationCatalog` and `ComputationAtlas`. Each of five records compares the complete 3,145,728-value predicted image A field with its native observation. It retains the following in existing binding/frontier/evidence fields:

| Field | Meaning |
|---|---|
| Generic law and route | `flux.parallel-attention.reference.v1`, `flux.context-to-image.single7.v1` |
| Program and source | Base/color/shape/additive/joint role, exact entry cut and prediction source hash |
| Receiving state | Common receiving-state anchor, complete actual entry cut and profile |
| Clock | Step 3/block 7; changed clocks do not inherit applicability |
| Writer | Predicted image A; native fixed image M; native projection/gate |
| Consumer and horizon | Native projection/residual, later single blocks, readout, scheduler and VAE |
| Target and collateral | Supplied eye/hair/clothing/background windows, signed RGB contrast and all grid responses |
| Continuation | Three native source-program RGB futures; color/shape-only RGB not newly measured in this panel |
| Limitations and counterexamples | No automatic semantic name, general object segmentation, exact additive joint program or arbitrary-scene guarantee |

The atlas receives five immutable records. All five exact-context queries return one record; ten changed-clock/receiving-state queries return zero. Context matching is a supplied predicate result, not validation at a new context. Original curated records are preserved, and no new catalogue schema or automatic family induction is introduced.

The [public atlas provenance](../../../../artifacts/mdb-interaction-corrections-20261104/public-atlas-v1/provenance.json) binds records and queries. The [offline evidence sweep](../../../../artifacts/mdb-interaction-corrections-20261104/evidence-v1/provenance.json) uses explicit frozen private tooling outside the model runtime: run/verify/query return 0, with twenty numeric rows and five model/input-qualified addresses. Forty-two focused correction/catalogue/atlas checks pass.

## Next discriminating cut

Trace formation and persistence of the continuing T program backward through its incoming residual, attention write and MLP write. Freeze a prediction of a joint representation before its native computation; test that representation through matched future consumers. Consumer-conditioned finite effects are more informative than ranking raw interaction norms. Separate relationship/geometry specimens still need their own controls at the clocks that build those features.
