# Step-1 pulses, carrier handoffs and frozen writes

This assay tests whether a shape program leaves a persistent image update when later conditioning returns to base, then separates working image/text and first-joint attention/MLP returns. It uses the public installed MDB/Saturn-pub runtime and a single finite mrun CUDA lease.

The experiment is deliberately scoped to the known seed60666 portrait, FLUX.2 Klein4B distilled, CUDA BF16, batch1,512px,four steps. Start from the same saved base latent at step1/project. Four base/color/shape/joint conditioning programs run for one step, with base E restored before step2. Two controls retain shape/joint E. These reuse observed prompt encodings; there are no new encodes or noise draws.

## What the native images show

The shape/joint pulse leaves mostly round eyes after later base conditioning. Continuing the shape program gives clearer narrow eyelids. This is evidence of repair or dependence on continued program exposure in this context; a self-sustaining narrow-eye edit is not established.

The remaining pulse RGB contrast includes eyelid, shading, face and collar details. Taking pulse.base as0 and pulse.shape as1 gives these image-only/text-only projections:

| Handoff after | I only | T only |
|---|---:|---:|
|joint0|0.005526|0.994901|
|joint4|0.032634|0.991381|
|single6|0.572834|0.636458|
|single18|1.004945|0.004059|

All four I+T controls reproduce pulse.shape exactly. The ordered prediction that image would dominate at single6 misses: both carry contrast there, with T-only slightly closer by eye MAE. These values are RGB-contrast diagnostics, not semantic percentages or a calibrated shape classifier.

At joint0, the incoming image R and gates match between base/shape. Observe full native attention return A and sequential MLP return M. Independently calculate all four frozen-return exits:

```
I_exit = B(B(R + B(g_attention * A_selected)) + B(g_mlp * M_selected))
```

B denotes BF16 rounding at the native edges. The four complete predictions are bit-exact against the independently audited equation and native source/mixed image returns. These are frozen observed returns; M is not recomputed after a mixed A. With base T, A-only and M-only futures stay close to base. With shape T, both retain nearly all the weak pulse contrast. This does not qualify an isolated narrow-eye writer.

## Reproduction

Use a qualified installed environment outside the checkout; source helpers are explicit payload files, not runtime path injection. Stage and submit with identical worker environment:

```
PYTHONDONTWRITEBYTECODE=1
MDB_WORKER_UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache
MDB_WORKER_TORCH_HOME=/mnt/hdd8tb/mdb/torch
MDB_WORKER_MODEL_ROOTS=/mnt/big/llm-models

python -I qualification/submit_flux_step1_writers.py RUN --source ONSET_ROOT
python -I qualification/submit_flux_step1_writers.py RUN --submit
python -I qualification/submit_flux_step1_writers.py RUN --collect JOB
python -I qualification/analyze_flux_step1_writers.py RUN AUDIT --source ONSET_ROOT
python -I qualification/publish_flux_step1_writers.py RUN AUDIT ATLAS
```

The first four lines document environment values; export them or use `env` for both staging and submission. Scheduler credentials come through `common exec`. Strict preflight, one resident load,10GB RAM/13GB VRAM,600s worker/720s lease and no retries bind the measured lane. Do not run a second CUDA assay concurrently.

Job `job-2e09f217e4a4` produced22native images,1646adapter transitions,44Acts and1690receipt emissions, with5separately counted direct block calls. Three tiny-native mechanics tests pass. The saved-only audit verifies source/archive custody, transaction order, complete exit fields, typed Act footprints, receipt paths, reset E and latent/image joins. Four public equation/atlas records retain eight dependent A/M/T outcomes and explicit applicability.

Artifacts: `artifacts/mdb-step1-writers-20261104/analysis-v1.json`, `visual-review.json`, `public-atlas-v1/provenance.json`. The next assay, [common-conditioning single6 overlap](FLUX-STEP1-OVERLAP.md), holds later shape E fixed so the consumer preserves the stronger eye-shape response.
