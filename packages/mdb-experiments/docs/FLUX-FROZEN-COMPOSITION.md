# Frozen composition on a fresh portrait

This completed assay tests a semantic composition rule before the new joint prompt encoding or any new denoiser outcome is available. It keeps semantic edit accuracy separate from numerical projection accuracy. Generic computation names remain useful; the retained record includes the failed transfer prediction.

## Protocol and visibility

Klein4B distilled, CUDA BF16 batch1, 512px/four steps, seed60666. New wording describes an adult anime woman with black hair, a teal shirt and ivory background. The source square changes amber to violet eyes and round/open to narrow/almond eyelids. The preceding portrait used different wording, copper hair, plum clothing, gray background and seed60665.

Only new base/color/shape encodings may enter the candidate rules:

```text
Eadd = B(Ecolor + Eshape - Ebase)
Delta_old = Ejoint_old - Eadd_old
Eplus = B(Eadd + Delta_old)
Eminus = B(Eadd - Delta_old)
Tpred(role) = B(Erole @ W_context.T)
```

The coefficient is fixed at1. Old/new text rows keep their original coordinates; no learned alignment or target-dependent fit is applied. Three complete initialT fields and semantic predictions are sealed before the new joint encode and before any native denoiser transition. Only then does the worker encode new joint, release the encoder, run the base first three steps once, and branch at step3/project. All edited futures share base L3/history. These are conditional final-step edits, not separate full generations from each prompt.

Frozen predictions: additive should produce violet/narrower eyes with coarse composition preserved; positive old correction should reduce eye-window error to new joint, negative correction should worsen it; predictedT native futures should remain below1RGB level MAE versus their measured-projector counterparts. These are scored forecasts, not discovery gates.

## Outcomes

| Candidate | Eye-window MAE to new joint | Whole-image MAE to new joint | PredictedT versus measuredT future MAE |
|---|---:|---:|---:|
| Additive |2.572518|0.743697|0.196051|
| Plus old interaction |2.923967|0.729869|0.187280|
| Minus old interaction |2.988370|0.755995|0.186213|

All errors use0–255RGB. Addition yields violet-looking eyes and retains coarse black hair/teal clothing/ivory background. Both correction signs worsen the frozen eye-window match. The positive correction slightly improves whole-image and clothing error, so the target miss is retained alongside that collateral trend. Narrower eyelids are not confidently established visually, including the late native shape/joint controls. A complete semantic success is not claimed.

The frozen eye box clips lower iris/sclera and includes skin/hair. Its metrics remain unchanged; a larger post-outcome display crop is labeled separately. Diagnostic hue counts distinguish amber versus violet but do not segment irises or measure eyelid geometry. The native color control supports color applicability, not a calibrated semantic score.

All three complete1,572,864-value initialT references independently reconstruct exactly. Native-relative RMS error is0.256120–0.256459%. Their matched native futures meet the prospective average-error forecast, while maximum pixel errors44/46/80 remain. Accurate local computation does not make the semantic composition exact.

A separate post-outcome saved-state diagnostic finds cosine−0.007238 between the old correction and the new required joint-minus-additive encoding. Even the best scalar fit leaves residual norm1997.355 versus1997.408 without it. This rules out reconstructing this new joint encoding by merely rescaling that fixed old direction in these raw coordinates. It does not rule out contextual laws, token alignment, other carriers or useful downstream effects. Diffusion seed does not enter the prompt encoder; wording/attributes/coordinate alignment are upstream differences here.

## Run and audit

Keep the same execution-environment intent at staging and submission. The launcher now refuses a mismatch before scheduler admission. Source-conditioned frames use the existing public `architectural` transaction frontier, covered by a mechanics test.

```sh
export MDB_WORKER_UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache
export MDB_WORKER_TORCH_HOME=/mnt/hdd8tb/mdb/torch
export MDB_WORKER_MODEL_ROOTS=/mnt/big/llm-models
python -I qualification/submit_flux_frozen_composition.py NEW_RUN \
  --source WORKSPACE/artifacts/mdb-conditioning-program-20261104
python -I qualification/submit_flux_frozen_composition.py NEW_RUN --submit
python -I qualification/submit_flux_frozen_composition.py NEW_RUN --collect JOB_ID
python -I qualification/analyze_flux_frozen_composition.py RUN NEW_AUDIT.json \
  --source WORKSPACE/artifacts/mdb-conditioning-program-20261104
python -I qualification/publish_flux_frozen_atlas.py RUN AUDIT.json NEW_ATLAS
```

The paths describe the measured Beast deployment; historical arrays remain host-local and hash-bound. Use installed public MDB/Saturn-pub/mrun packages. Qualification helpers are explicitly staged; there are no private runtime imports or checkout path changes. One finite mrun lease uses strict preflight, declared CUDA, no retries, and the measured full-encoder24GBRAM/13GBVRAM profile. This is not a portable reservation recommendation.

Successful job `job-7d2261c2f5e4` uses one load, four encodes, one noise draw,389native transitions,10Acts,399receipts and11images. Worker time is66.413s and runner108.58s; peakRSS8926.2MB/VRAM9220MB. The189,368,636-byte archive retains190files/23parts. Plain base versus same-E Act is exact in latent/RGB. Two prior failures remain recorded: an environment-intent mismatch before model load, then an unsupported transaction frontier after three source encodes. Neither encoded the new joint or ran the denoiser. Three source encodes/one model load from the second failure are additional physical cost. No setup-inclusive speedup is claimed.

The [independent audit](../../../../artifacts/mdb-frozen-composition-20261104/analysis-v1.json), [visual review](../../../../artifacts/mdb-frozen-composition-20261104/visual-review.json), [post-outcome geometry](../../../../artifacts/mdb-frozen-composition-20261104/interaction-geometry-v1.json), [public atlas](../../../../artifacts/mdb-frozen-composition-20261104/public-atlas-v2/provenance.json) and [offline evidence index](../../../../artifacts/mdb-frozen-composition-20261104/evidence-v1/provenance.json) preserve the result. Three public projection occurrences pass3exact/6changed-context queries; evidence run/verify/query returns0with17numeric rows and3qualified addresses. An initial offline atlas import omitted explicit backend metadata; the partialv1directory is preserved, and correctedv2required no inference.

The [onset follow-up](FLUX-EDIT-ONSET.md) now completes that cut: earlier native and composed programs yield narrower eyelids, while the same predicted initialT stacks through repeated native reads. Precise composition and collateral remain approximate. A separate matched wording/attribute contrast can test whether token alignment or contextual interaction accounts for the failed old correction. Neither result tunes this already scored test into a retrospective success.
