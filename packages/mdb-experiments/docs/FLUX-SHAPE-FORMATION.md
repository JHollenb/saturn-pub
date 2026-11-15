# Where the early shape response forms

The [single6 RAMT panel](FLUX-STEP1-OVERLAP.md) found substantial accumulated-image and continuing-text effects. This assay moves backward through single blocks0–6 with typed handoffs, stream clamps and rescue. All branches receive the same shape conditioning before denoising step2.

The first specimen is the known seed60666 portrait, FLUX.2 Klein4B distilled, CUDA BF16, batch1,512px,four steps. It starts from matched step1/single0 base/shape cuts. The source encodings and preceding joint-block computation are observed, not independently predicted.

## Panel and findings

Capture both native streams at boundaries0–7. Introduce shape I or T at0,3,6. Transfer I+T at0/6 as exact coherent controls. Then test:

- Shape T at0, followed by base T at6: can early text leave useful image state?
- Base T before every block0–6: does continual text suppression prevent the response?
- Base T before0–5, shape T at6: does late text rescue recover it?
- Base I before every block0–6: can continuing text still influence the future?

All replacements use complete native fields from the corresponding source boundary. They preserve undeclared state and receive the same later shape E. They are sufficient/conditional interventions, not minimal head or neuron claims.

Job `job-19679c58a4a7` gives these eye-region RGB-contrast projections, with its weak reference0 and stronger reference1:

| Intervention | Projection |
|---|---:|
|I-only at0 /3 /6|0.033299 /0.242828 /0.488147|
|T-only at0 /3 /6|0.995167 /0.822319 /0.648591|
|T at0, reset tobase at6|0.491335|
|Base T clamped through6|0.032174|
|Base T through5, shape T rescue6|0.680555|
|Base I clamped through6|0.611562|

Early T-only reaches an image state with full-field projection0.938960 onto the source I difference at pre6. The incoming image at pre0 is exactly base. After Treset6, text is exactly base while image retains that difference, and the final image keeps an intermediate contrast. Repeated base-T clamps leave pre6 image projection0.068474 instead. The rescue and continued-clamp branches have identical incoming I at6; changing only T there reduces final eye MAE tostrong from8.412861 to3.723091.

These are mixed pixel/field contrasts, not percentages of eye shape. Unblinded native-image inspection finds more open/round weak-reference eyelids, narrower upper lids in the strong reference, intermediate appearances after reset/rescue, and coarse portrait appearance retained with fine collateral. Source I handoff authority grows over this interval while T handoff authority declines. The four prospective discovery directions match; no scientific gate discards a case.

## Mathematical interpretation

At each single block j, normalized/time-modulated image state supplies a pointwise MLP, while attention reads joint image/text state:

```
H_I = mod_j(LN(I_j), time)
M_I = SwiGLU(H_I W_m + b_m)
A_I = Attention_I(H_I, H_T, positions)
P_I = B([A_I; M_I] W_out.T + b_out)
I_(j+1) = B(I_j + B(g_j P_I))
```

This is the pinned native architecture. In the early T-only branch, incoming I and modulation are held fixed at0. The first image change therefore has an attention-mediated entry path; the pointwise image MLP has no direct text argument there. **This is an architectural inference**, not a new complete A/M-port capture. Once attention changes I, subsequent MLP inputs can differ. The native causal assay establishes the path into accumulated I and its downstream effect; it does not isolate the responsible heads or independently predict their formation.

## Reproduction and custody

Use the same qualified installed runtime and worker environment as [step1 writers](FLUX-STEP1-WRITERS.md), through `common exec` for scheduler credentials:

```
python -I qualification/submit_flux_shape_formation.py RUN --source WRITERS_ROOT --overlap-source OVERLAP_ROOT
python -I qualification/submit_flux_shape_formation.py RUN --submit
python -I qualification/submit_flux_shape_formation.py RUN --collect JOB
python -I qualification/analyze_flux_shape_formation.py RUN AUDIT --source WRITERS_ROOT --overlap-source OVERLAP_ROOT
```

The first assay uses1load,0encodes/noise,1068native transitions,47Acts,1115receipt emissions and14VAEs, with no extra reconstructed block calls. The audit retains repeated canonical receipt IDs correctly:1103distinct IDs correspond to1115physical emissions. All source/intervention I/T boundaries and42step-boundary latent observations are typed and retained. Two historical PNG references and both coherent I+T controls are exact.

Three tiny-native installed-runtime tests verify recipe counts, typed clamp/rescue footprints and creating a fresh full step0 session from the saved schedule. The same worker can run with a new seed:40additional native transitions,2additional Acts and1noise draw prepare a new common step1 latent. The semantic predictions must be frozen from an audited discovery before that noise or any new native prefix runs. Prompts and observed encodings remain fixed, so this tests receiving-state variation rather than new-wording generalization.

Six source-qualified causal records per specimen use generic occurrence IDs for handoff0/3/6, text-seed/reset, text-clamp/rescue and image-clamp. The offline evidence sweep indexes these explicit records. They are not newly invented mathematical equations or automatically ingested equation-atlas families.

## Frozen replication and its useful miss

Job `job-db9e414ff4d6` tests seed60667 after nine individual claims seal before fresh noise or any new prefix. Eight match. The predicted0.2–0.8final eye-contrast range after Tseed0/reset6 misses at0.138179, despite pre6 image-state projection0.936189, nearly the discovery0.938960. The direction remains positive, but the source-calibrated magnitude does not transfer.

| Case |60666|60667|
|---|---:|---:|
|Early T-only final eye contrast|0.995167|0.985661|
|Early I-only final eye contrast|0.033299|0.003717|
|Early T then reset6|0.491335|0.138179|
|T clamped through6|0.032174|0.005926|
|T rescue6|0.680555|0.503362|
|I clamped through6|0.611562|0.434109|

The saved factorial also exposes a change in I×T interaction. Define the interaction over four **native measured futures** as `J = Y11 - Y10 - Y01 + Y00`, using pre6 I/T handoffs. Its eye-contrast projection changes from−0.136738to+0.385870 across seeds; whole-image projection changes from−0.045640to+0.288399. This is a descriptive interaction of outcomes, not an independently predicted or installed correction. Joint state matters more than the sum of the two separate contrasts in the new context. Nonzero interaction is also present in committed latents before VAE; its precise semantic source is not localized.

The replication uses1load,0encodes,1newnoise draw,1108native transitions,49Acts,1157receipt emissions and14VAEs. Both coherent I+T controls match the new strong reference exactly. Discovery/replication worker times are105.146/112.792seconds; runner177.49/178.40seconds. PeakRSS2091.2/7287.4MB, VRAM9876/9970MB. These are evidence-run costs, not a predictor speedup.

Artifact roots: `artifacts/mdb-shape-formation-20261104/` and `artifacts/mdb-shape-formation-replication-20261104/`. The former retains `replication-predictions-v1.json`, `architecture-inference-v1.json` and the zero-model-call `interaction-diagnostic-v1.json`; the latter retains `replication-scores-v1.json`. Discovery evidence-v1 and replication evidence-v2 index12curated causal records,65numeric rows and24source-qualified addresses with run/verify/query returning0. Replication evidence-v1 remains historical; v2 explicitly attaches the tighter frozen forecast and its counterexample, which the generic four-direction summary alone omitted.

The next useful intervention holds the complete receiving state and program common while testing why the I×T consumer interaction changes, then captures the relevant attention/MLP ports. The two-seed carrier structure is a convergent trend; accurate edit magnitude, precise preservation, new wording and minimum writers remain open.
