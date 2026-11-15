# From receiving writes to head, armor and contextual appearance

Two further native cuts extend [the first receiving-write result](FLUX-ANIME-WRITERS.md). We can change a head/hood appearance while retaining much of the armor, and change the armor, tunic, exposed-leg and sword-hilt appearance while retaining the blond head. The effects depend on the clock, the type of state transplanted and the surrounding scene state. These are conditional causal records in two known scenes, not automatically recovered face, eye or character programs.

## The native experiment

Job `job-72cf2fcc451d` ran22 final images: four natural suffix traces and18 later transplants, with eight exact historical controls. It compared selected-left working image replacements before single10/19 at runtime steps2/3, complete working image+text replacement before single19, and selected-left/whole carried latent replacement at step3 projection. Steps2/3 are zero-based, the last two denoising steps of this four-step schedule. The checkpoint, conditioning, geometry, CUDA BF16 profile and physical batch1 remain fixed.

Job `job-932158fd4416` imported actual step3/single19 frontiers and ran10 more images, including six exact controls. It partitioned the already tested348 selected-left native rows into96 upper rows (native H<11 within that support) and252 complementary lower rows. Untouched, upper, lower and complete selected-left replacements form an exact2×2 factorial. All nonimage state is identical; the two disjoint row replacements compose exactly to the complete replacement. The geometric upper band includes head, hood/shoulder, umbrella and environment. It is not a pure facial mask.

![Actual native factorial outputs: reference, upper96 rows, lower252 rows and both bands](../../../../obsidian/images/flux-anime-20261103/part-factorial.png)

## What became distinguishable

In festival, the upper-band late image edit changes the left appearance from a long silver-haired figure to a short black bob, changed eye/face drawing and burgundy hood. The long blue outer coat and much of the lower costume remain. The lower edit retains the long silver-haired head while changing lower/chest clothing pixels. The umbrella can become a faint outline under the upper edit, so head control includes prop/background collateral.

In bridge, the upper edit gives the left figure violet hair, a mage-like face and a green hood while retaining silver chest, arm and leg armor and a gray/silver sword hilt. The lower edit retains the blond head but changes the costume to green-tinted segmented armor, a shorter tunic with exposed thighs, and a gold sword hilt. Both edits combine those coarse head and lower-appearance cues. The figure continues holding the lowered sword; the right figure retains its staff and coarse mage appearance.

| Geometric RGB measurement versus native | Upper edit | Lower edit |
|---|---:|---:|
| Festival left-hair rectangle MAE |0.20725|0.00298|
| Bridge left-hair rectangle MAE |0.19869|0.00314|
| Bridge armor rectangle MAE |0.02190|0.10897|
| Bridge collar rectangle MAE |0.17242|0.00347|

Values are zero-to-one RGB averages under supplied rectangles, including mixtures/background. They support preferential geometric response, not independent semantic masks. Iris color, exact facial identity and parametrized face shape are not established. Enlarged head sheets use nearest-neighbor2× native pixels without enhancement.

The depth panel reveals another separation. Festival step2 local working-image edits create a short silver bob with a hoodie: hairstyle/face/clothing move without the source black hair color. Step3 local working-image edits produce the black bob. In bridge, an earlier local image edit can change the right figure to blond hair while retaining its green robe and gold-orb staff. Thus hair, costume, prop association and coarse instance appearance need separate records. The edit changes both clock and the naturally evolved source payload; this is not a fixed-vector wrong-time control.

## Two kinds of memory and an exact next-state prediction

Working image I and text T are temporary transformer registers. Carried latent L survives scheduler commits. The native deterministic update is

\[
v_t=R_\theta(F_\theta(I_t,T_t;\text{context})),\quad
L_{t+1}=\operatorname{BF16}\bigl(\operatorname{FP32}(L_t)+\Delta\sigma_t\,v_t\bigr).
\]

The implementation preserves native scalar/tensor promotion: dt times BF16 velocity is evaluated before adding to FP32 carried state. Casting or reassociating the operations differently is not the qualified equation.

For complete I/T restorations, we observed the natural source velocity, then persisted a full next-latent prediction before executing each new transplanted branch. All four predictions match the actual131,072-value next latent exactly. This is observed-source-assisted Euler prediction, not independent RGB prediction from text. Complete I/T restoration is a joint bundle; it does not isolate text from global image effects. The transplanted branch's velocity itself was not separately retained as a tensor; source velocity and predicted/actual next latent are retained.

At step2, source and recipient start with identical selected-left L. Restoring complete source I/T before the final single block therefore produces the exact source next-left latent:44,544 values in each scene, while non-left latent remains different. By contrast, restoring complete I/T at step3 retains the recipient's already different L, so even the next-left latent differs from the source. The equation accounts for both outcomes.

An offline matched-state comparison makes contextual dependence sharper. In festival, the step2 complete-I/T branch at step3 projection and the selected-left L-restoration branch have bit-identical left L and all nonlatent state, but differ outside the left support. Their final left heads have black versus silver bobs. The complement includes environment as well as the companion figure; this is not an isolated right-person intervention. A local state description needs its surrounding receiving state and clock to predict appearance.

## What math happens in the later blocks

The earlier joint block applies attention and then a pointwise MLP to the changed receiving features. The later single blocks use a parallel architecture, as inspected in the installed Diffusers implementation. With Z denoting the time-modulated normalization of concatenated text/image state H,

\[
[Q,K,V,U,V_m]=W_{in}Z,\quad
A=\operatorname{Attention}(Q,K,V;P),\quad
M=\operatorname{SiLU}(U)\odot V_m,
\]
\[
H'=H+g_t\odot W_{out}[A;M].
\]

Q/K normalization and rotary position transformations are included in the attention path. Both branches read the same normalized receiving state; the MLP is not sequentially downstream of attention inside this single block. The native output projection is one fused concatenated GEMM. Splitting it into separately rounded attention/MLP projections would require its own numerical qualification. These equations describe inspected architecture; this assay did not separately intervene on those fused internal operands.

Our edits enter before single19, followed by that native parallel block, normalization/readout, the scheduler and VAE. The [subsequent residual/attention/MLP factorial](FLUX-ANIME-RESIDUAL-FACTORS.md) executes the inward cut at single18, whose exit is writable in the public adapter; single19's readout exit remains read-only. It preserves the fused GEMM and distinguishes carried appearance from local and cross-position branch responses.

## The finite interaction is measured, not assigned a semantic name

Let00 be native,10 upper,01 lower and11 both. We compute

\[
\mathcal I_L=L_{11}-L_{10}-L_{01}+L_{00},\qquad
\mathcal I_Y=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
\]

Final-latent interaction RMS is0.01118 in festival and0.01250 in bridge; RGB interaction RMS is0.00493/0.00815. The interaction is already present before VAE decode. It is finite native numerical nonadditivity, including rounding, not an identified abstract XOR or semantic binding primitive.

A contact-sheet interpretation initially suggested a cloth-length reversal under the combined edit. Full-resolution originals corrected that: the short tunic follows the lower band and remains in the combination. Raw anonymous observations, the [additive correction](../../../../artifacts/mdb-flux-anime-20261103/parts-review-v1/full-resolution-correction.json) and independent numerical measurements are all retained. A separate apparent saturation difference between pixel-identical historical controls also remains explicitly corrected. Informal fine-face/color inspection needs stronger measurement before promoting subtle distinctions.

## Evidence and using MDB

The [worker](../qualification/flux_anime_depth.py) uses installed public MDB/Saturn-pub, actual StateCuts, row-selected Acts, NativeCutNavigator and ForecastTransaction. No candidate-block execution, retraining, compositing or image generation service supplies these outputs. The [submitter](../qualification/submit_flux_anime_depth.py) supports a depth stage followed by `--parts`; both use one finite CUDA lease, strict preflight and no retries. The second stage needs only30 native transitions for10 final images because it imports the saved late frontiers.

```bash
python qualification/submit_flux_anime_depth.py DEPTH_RUN --predecessor AUDITED_SPATIAL_RUN
python qualification/submit_flux_anime_depth.py DEPTH_RUN --collect DEPTH_JOB
# Audit the retained depth evidence before selecting the next bounded cut.
python qualification/submit_flux_anime_depth.py PART_RUN --predecessor DEPTH_RUN --parts
python qualification/submit_flux_anime_depth.py PART_RUN --collect PART_JOB
```

Supply the qualified Linux worker cache settings and installed-wheel launcher from [the earlier workflow](FLUX-ANIME-SCENES.md). Source custody remains explicit host-local hash/size-pinned CAS, not a deployed fleet artifact mount. The worker supports the two declared starting boundaries; broad arbitrary-stage analysis is not implied.

The depth job used582 transitions/22 Acts/604 receipt emissions, worker204.99s and scheduler269.94s, peak RSS7,448MB/VRAM10,472MB. The part job used30 transitions/eight Acts/38 emissions, worker53.53s and scheduler81.17s, peak RSS8,068.1MB/VRAM9,242MB. Timing includes different geometry/output work and does not establish a matched speedup. Both used one load and zero encodes. All14 historical controls and four numerical forecasts passed; both independent receipt/Act/manifest audits are valid. The pinned wheel's legacy CPU-FP32 transaction label/external-call counter limitation remains qualified additively.

Evidence: [depth audit](../../../../artifacts/mdb-flux-anime-20261103/depth-run-01/analysis-v1.json), [part audit](../../../../artifacts/mdb-flux-anime-20261103/parts-run-01/analysis-v1.json), [factorial closure and interactions](../../../../artifacts/mdb-flux-anime-20261103/parts-run-01/factorial-interactions.json), [matched-state contrasts](../../../../artifacts/mdb-flux-anime-20261103/depth-matched-state-contrasts.json), [depth appearance review](../../../../artifacts/mdb-flux-anime-20261103/depth-review-v1/visual-review.json), [part appearance review](../../../../artifacts/mdb-flux-anime-20261103/parts-review-v1/visual-review.json), [depth forecast pin](../../../../artifacts/mdb-flux-anime-20261103/depth-forecast-pin.json), [depth external seal](../../../../artifacts/mdb-flux-anime-20261103/depth-external-seal.json), [part external pin](../../../../artifacts/mdb-flux-anime-20261103/parts-external-pin.json). The part external seal was observed before collection after two known control images already existed remotely; scheduler-bound forecasts predate execution.

Eighteen new [time/depth and part occurrences](../../../../artifacts/mdb-flux-anime-20261103/depth-part-records.json) bring the combined [citation query](../../../../artifacts/mdb-flux-anime-20261103/evidence-v3/xref-query.json) to38 exact model/checkpoint/parent-bound records. Offline evidence run/verify/query/stats all return0. These records preserve mixed responses and counterexamples; automatic semantic rule induction, necessary/minimal facial components and independent eye-color control remain open.
