# Distinguishing carried appearance from attention and MLP writes

The next cut after [time and part localization](FLUX-ANIME-DEPTH-PARTS.md) separates the incoming residual, attention operand and gated MLP operand inside one late FLUX single block. In the tested bridge scene, most of the late head/costume response travels through the incoming residual. Attention and MLP changes refine it; attention carries changes across positions and into working text. This identifies a conditional carrier and response, not where the appearance was first formed.

## The actual native run

Job `job-8505274b3899` used the same FLUX.2 Klein4B distilled checkpoint, CUDA BF16, physical batch1,512×512 and four-step schedule as the earlier anime runs. It imported real step3/single10 frontiers, replayed each to single18, then made two I-only sources with matched incoming text, latent and context. The supplied upper96 and lower252 native image rows partition the previously tested left support. They include mixtures of foreground and background; they are not semantic segmentation masks.

We chose single18, the penultimate block, because its output enters the writable single19 boundary. The final block19 exits at readout, whose image/text ports are currently read-only. No fabricated cut or modified Navigator bypasses that restriction.

Twenty native RGB futures were executed: three source/reference continuations, seven nonzero residual/attention/MLP combinations for each band, two coherent working-text completions and a sham. Four controls are exact: baseline and sham match the historical baseline PNG; each coherent completion matches its own newly observed source future PNG. All receipt chains, source custody, declared operand selections and native block parity checks verify independently.

![Native outputs: base, source residual only, source attention plus MLP only, and coherent band source; upper then lower](../../../../obsidian/images/flux-anime-20261104/factorial-futures.png)

These are native suffix decodes, laid out in a contact sheet. No generated illustration, composited feature replacement or enhancement supplies the scientific images. Full-resolution originals and a task-aware [appearance review](../../../../artifacts/mdb-flux-anime-20261104/factorial-run-01/appearance-review.json) remain available.

## The factorized computation

A single-stream block reads concatenated working text/image state H. Its time-modulated normalized input Z supplies parallel attention and SwiGLU branches:

\[
[Q,K,V,U,V_m]=W_{in}Z,\qquad A=\mathrm{Attention}(Q,K,V),\qquad M=\mathrm{SiLU}(U)\odot V_m.
\]

Normalization and rotary position transforms in the attention path remain native. The output is computed with one fused projection of concatenated operands, not separately rounded branch projections:

\[
P_{a,m}=Q_{\mathrm{BF16}}\!\left(W_{out}[A_a;M_m]+b_{out}\right),\quad
I'_{r,a,m}=Q_{\mathrm{BF16}}\!\left(R_r+Q_{\mathrm{BF16}}(g\odot P_{a,m})\right).
\]

Here r,a,m independently choose recipient or band-source operands. R is the incoming image residual, A has3072 channels and M has9216. The gate g and all incoming nonimage state are matched. The worker preserves the native fused GEMM and BF16 multiply/add ordering. The symbolic expression summarizes the measured implementation; it is not a claim that arbitrary reassociation reproduces CUDA bits.

`R` means source residual plus recipient attention/MLP operands. `AM` means source attention/MLP image operands plus recipient residual. These are operand replacements, not deletion ablations. Source attention changes are installed at all image positions; source M changes are naturally confined to the edited band. Working text remains recipient until the explicit `RAMT` completion.

The worker makes separately counted direct block calculations to capture or substitute operands, then installs the calculated image through a public typed Act at the native block exit. NativeCutNavigator itself executes unmodified. Source block I/T outputs match native Navigator outputs exactly. Complete `RAM` reconstructs source post-block image exactly; adding T reconstructs the coherent source continuation and PNG exactly.

## What changed, and which path carried it

Upper residual replacement changes the left head toward violet hair and a green hood while retaining silver armor and the lowered sword. Lower residual replacement keeps the blond head while changing toward green segmented armor, short tunic, exposed thighs and a gold sword hilt. Residual-only results have some less coherent/doubled contours. Attention-plus-MLP-only replacements retain much more recipient appearance. The right figure retains its coarse robe/staff relationship in the inspected originals.

Let B be baseline RGB, S the corresponding native band-source RGB and Y an intervention output. The signed source projection is

\[
p(Y)=\frac{\langle Y-B,S-B\rangle}{\lVert S-B\rVert^2}.
\]

This is a directional pixel measurement under a supplied rectangle, not a semantic percentage, perceptual identity score or additive attribution.

| Source operands | Upper head rectangle p | Lower armor rectangle p |
|---|---:|---:|
| R |0.7164|0.7041|
| A |0.1578|0.2044|
| M |0.0538|0.0816|
| RA |0.9403|0.9231|
| RM |0.7997|0.8127|
| AM |0.2418|0.3296|
| RAM |1.0000|0.9998|
| RAMT |1.0000|1.0000|

The residual-dominance hypothesis is supported in these two geometric measurements. Attention replacement moves farther toward the source than MLP replacement here. Upstream MLPs can still have created the residual features: this late cut cannot assign their origin or imply that MLP computation is generally unimportant.

The internal locality result is exact. Upper/lower MLP operand changes have zero changed elements outside the edited image rows and zero in text. Attention changes1,599,045/1,579,013 elements outside those rows and799,568/950,914 working-text elements. This distinguishes a positionwise local transform from cross-position mixing at this block; it does not name either branch a particular semantic operation.

Text feedback has a small but measurable late consumer effect. `RAM` leaves recipient T and differs from its source future by mean absolute RGB0.0005713/0.0005839. Replacing T as well makes each PNG exact. This is conditional evidence about this clock and scene, not universal irrelevance of text.

## Forecasts, errors and coverage

The staged hypotheses and source identities were externally pinned before execution; the global seal was observed before any source block calculations or images. After observing three source operand sets, the worker froze numeric predictions before executing the mixed candidate calculations. Predictions use actual source operands and32 retained learned output-weight rows, evaluated in FP64 with explicit BF16 rounding at the declared edges. Eight query positions yield256 predicted image features per combination.

Across14 unique R/A/M combinations,3,570 of3,584 sampled image values match exactly. Fourteen differ; the largest absolute error is1.0 native feature unit. All stored forecast arrays independently reconstruct exactly from their saved inputs. The sampled reference is not an exact CUDA GEMM emulator, and these error magnitudes remain observations rather than discovery gates. RAMT repeats RAM image predictions and is excluded from that total.

The numeric seal is worker-ordered before mixed calculations. Its external pin was obtained after17 remote images already existed, before collection. This qualification remains explicit; the earlier external global seal and staged request do not retroactively make the later numeric seal externally witnessed before all execution.

This is source-assisted prediction of sampled internal updates, plus exact coherent native reconstruction. It does not independently forecast source A/M from raw prompt, the complete image, eye color, face shape or arbitrary novel scenes. Full candidate outputs outside sampled coordinates are bound by executed code, operand descriptors and Act receipts; only sampled outputs were independently recomputed offline.

## Reproduce and inspect with MDB

The [submitter](../qualification/submit_flux_anime_factorial.py) requires the audited depth run and its sibling part-run plan. Use the qualified installed-wheel Linux cache profile from [the scene workflow](FLUX-ANIME-SCENES.md):

```bash
python qualification/submit_flux_anime_factorial.py NEW_RUN --predecessor AUDITED_DEPTH_RUN
# Prefer compact mrun job metadata while running; --status currently downloads full logs.
python qualification/submit_flux_anime_factorial.py NEW_RUN --collect JOB_ID
```

The [worker](../qualification/flux_anime_factorial.py) uses installed public MDB/Saturn-pub, one strict finite mrun CUDA lease and no retry. This run used96 adapter transitions,21 Acts,117 receipt emissions,20 separately counted candidate block calculations, one model load and zero prompt encodes. Worker time was168.31s; scheduler time255.36s, peak RSS6,587.8MB and VRAM9,602MB. Comparison with older jobs is not a matched speed benchmark.

The frozen local [independent auditor](../../../../artifacts/mdb-flux-anime-20261104/analyze_factorial.py) verifies manifests, custody, source preparation, source block/native parity, receipt chains, operand selections, BF16 residual updates, sampled Act binding, forecast reconstruction and exact PNG controls. The pinned transaction package has a legacy CPU-FP32 scope label and does not count external Navigator calls; the additive [metadata qualification](../../../../artifacts/mdb-flux-anime-20261104/factorial-run-01/metadata-qualification.json) preserves raw artifacts and makes the actual CUDA BF16 receipts/counts authoritative.

Evidence: [audit and every numeric error](../../../../artifacts/mdb-flux-anime-20261104/factorial-run-01/analysis-v1.json), [global external seal](../../../../artifacts/mdb-flux-anime-20261104/external-seal.json), [numeric external pin](../../../../artifacts/mdb-flux-anime-20261104/numeric-external-pin.json), [14 new generic occurrences](../../../../artifacts/mdb-flux-anime-20261104/computation-records.json), [52 combined exact model/checkpoint/parent citation hits](../../../../artifacts/mdb-flux-anime-20261104/evidence-v1/xref-query.json). Evidence run/verify/query/stats all return0. The anime campaign now contains86 executed native images including controls; occurrence indexing counts52 selected interventions, not52 independently discovered programs.

## The next causal question

The [subsequent reciprocal depth curve and joint0 factorial](FLUX-ANIME-ORIGIN-COOPERATION.md) execute the next backward cuts:32 depth futures,20 joint futures and exact complete post-block predictions for14 unique combinations. They reveal direction-dependent transfer and strong residual/write cooperation, while first formation and necessary/minimal writers remain open. First-class operand Acts and writable readout-entry image/text ports would remove the candidate calculation workaround; the existing request has been upvoted with this run's evidence. Automatic semantic naming and rule induction remain open.
