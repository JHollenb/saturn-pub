# Discovering and executing contextual responses

The useful unit of discovery is a conditional state transition: how an explicit change to a live state changes a write, and how subsequent computation consumes that write. This joins MDB's symbol records, replay, dependency planning, mathematical predictors and native execution without assuming every object has a fixed neuron or address.

Astra reviewed the numerical evidence alongside the scene-generator, ISA MLP, time-circuit, graph, spectral and model-memory work. The resulting research algorithm is **counterexample-guided response closure**. It is a method for refining programs and applicability, with an initial finite-response audit API. Automatic operand discovery and automatic execution-policy selection remain integration work.

## One transition and three distinct problems

Write the native transition as

\[
S_{q+1}=T^N_{q,\alpha}(S_q,u_q).
\]

The state includes live residual streams and persistent memory; the clock includes depth and denoising/token position. The execution contract \(\alpha\) includes checkpoint, dtype, accumulation policy and operation geometry. A proposed operational chart \(z=C_q(S)\) needs both a transition law and a consumer account:

\[
C_{q+1}(T^N_q(S,u))\approx f_q(C_q(S),u),\qquad O_q(S,u)\approx g_q(C_q(S),u).
\]

The same prediction error can require different repairs:

| Diagnostic | What it distinguishes | Next refinement |
|---|---|---|
| Same complete operands, different numerical implementation | Arithmetic contract discrepancy | Qualify the primitive or use the native primitive |
| Equal retained variables, different omitted state changes retained successors | State closure discrepancy | Add/test an operand, memory read or interaction; revise chart/domain |
| Retained successors agree, later consumer differs | Consumer/readout discrepancy | Describe the consumer and its additional state |
| Native execution on the candidate's current state still differs from the true trajectory | Inherited upstream discrepancy | Restore a compatible cut and replay; native fallback alone does not erase past error |

These are experimental distinctions, not an automatic classifier. A local derivative can miss finite nuisance effects: \(T(x,y)=(x+y^2,y)\) depends on discarded \(y\) even though its derivative at zero vanishes. Preserve finite signed interventions and counterexamples.

## Execute changes from a common parent

For an additive edit, define the finite response

\[
D_q(S,\delta)=T_q(S+\delta)-T_q(S).
\]

Typed replacements generalize this definition to fields without additive coordinates. A candidate trajectory consumes its own predicted state:

\[
\widehat S^1_{q+1}=S^0_{q+1}+\widehat D_q(S^0_q,\widehat S^1_q-S^0_q).
\]

Let \(e_q=\widehat S^1_q-S^1_q\), and define the local numerical/program defect \(r_q=\widehat S^1_{q+1}-T^N_q(\widehat S^1_q)\). Then

\[
e_{q+1}=T^N_q(\widehat S^1_q)-T^N_q(S^1_q)+r_q.
\]

This separates inherited error from a local response defect. It also explains why replacing the next operation with native execution does not restore the true trajectory automatically.

The execution choices are conservative:

1. Reuse an output when its complete relevant inputs and execution contract are unchanged.
2. Use a qualified algebraic update, such as the unchanged-key attention partition, on affected operands.
3. Use a bounded empirical response law where new combinations support its applicability.
4. Execute the affected native operation where the cheaper alternatives remain inaccurate or unqualified.

Use dependency footprints to propagate changes. A two-row text edit can alter every image attention read and therefore every receiving MLP input; sparse source support does not imply sparse downstream work. The [dependency planner](DEPENDENCY-REUSE.md) identifies candidates but does not itself execute or certify a skip. The [attention partition](ATTENTION-RESPONSE.md) supplies one algebraic update, with its own numerical limits.

## A fresh test of numerical anchoring

The November 5 assay constructs working-text dose \(3/8\) directly from original base/donor rows 4 and 11. It crosses that same text with two known image receivers at Klein4B step2/joint1. The native half-dose endpoints are declared calibration anchors. Both new complete image/text exits are unseen when eight forecasts seal: absolute mathematical prediction, receiver-specific anchoring, copy-anchor control and an other-receiver-response control for each target. Then two original CUDA BF16 blocks run. There are no image decodes in this assay.

If \(P\) is the mathematical reference and \(b(S)=P(S)-T^N(S)\), anchoring gives

\[
P_A(S_1)=T^N(S_0)+P(S_1)-P(S_0),\qquad
P_A(S_1)-T^N(S_1)=b(S_1)-b(S_0).
\]

It removes a constant offset; it cannot improve the unrounded within-receiver response law. Final BF16 recasting adds a separately stored residual \(\rho\). The audit verifies the pointwise identity \(C-N_1=(b_1-b_0)+\rho\) for the anchored arrays and retains signed error-power cross terms.

| Measurement | Receiver 0 | Receiver 1 |
|---|---:|---:|
| Absolute image endpoint relative RMS | 0.3266% | 0.3253% |
| Anchored image endpoint relative RMS | 0.3664% | 0.3625% |
| Absolute text endpoint relative RMS | 0.1398% | 0.1271% |
| Anchored text endpoint relative RMS | 0.1262% | 0.1251% |
| Mathematical image dose-response error / native effect RMS | 66.15% | 65.87% |
| Mathematical text dose-response error / native effect RMS | 49.66% | 49.80% |
| Copy-anchor response error / native effect RMS | 100% | 100% |

The anchored stored response beats copying the anchor on each stream, and using the other receiver's response is worse. That preserves a useful, receiver-dependent trend. But unit-offset anchoring worsens image endpoints. The endpoint defects have cosine only 0.374–0.378 across dose on image state, versus 0.536–0.573 on text. Their covariance explains the heterogeneous result.

The receiver-by-dose mixed difference is

\[
I=F_{11}-F_{10}-F_{01}+F_{00}.
\]

Its reference error is 125.34% for image and 131.03% for text. Zero interaction, the copy control, has 100% error by construction and is better in RMS. Anchoring leaves this response essentially unchanged apart from recasting. This does not identify an omitted semantic variable: reduced-precision arithmetic alone can produce finite mixed differences for an affine real law.

The test therefore rejects this offset correction as a reliable fix for these fine responses. It supports qualifying native numerical boundaries before expanding a semantic or dose grid. [Audit](../../../../artifacts/mdb-response-algorithm-20261105/audit-v1.json), [custody](../../../../artifacts/mdb-response-algorithm-20261105/collection-01/report.json), [Astra review](../../../../artifacts/mdb-response-algorithm-20261105/advice/astra-review.md).

## Reusable audit API

### Predicting affected primitives from their operands

The next qualification moves calibration inside the transition. For each primitive, the independent NumPy predictor computes a wide finite response around an explicit historical native input/output pair. An affine return is

\[
\widehat y_1=B\{y_0+(x_1-x_0)W^\top\}.
\]

The widened BF16 endpoints are subtracted without casting their difference back to BF16. Bias cancels. Normalization and attention use corresponding wide law differences; the receiving MLP advances from the candidate's own residual and normalization. This is a numerical hypothesis that the native primitive's local arithmetic defect changes slowly, rather than a reconstruction of CUDA reduction internals. Its zero-edit identity is a mechanics control.

On the saved 3/8 development cases, image-response error improves from 66% to 53%; text-response error improves from 50% to 38–40%. The receiver-by-dose interaction remains near 99% error against the zero-interaction control's 100%. These cases informed development and are not prospective evidence. [Source-bound diagnostic](../../../../artifacts/mdb-flux-local-delta-20261105/diagnostic/audit-v1.json).

The [fresh campaign](../../../../artifacts/mdb-flux-local-delta-20261105/README.md) seals complete writes for four new asymmetric text states, then renders their candidate futures and seals image readouts before releasing any new native target block. Historical native calibration and all native suffix/VAE work remain explicit. The predictor is currently a staged qualification helper; the public APIs supply transactions, typed Acts, replay, archives and response audits. Fresh outcomes must be read from the completed campaign audit, not inferred from the development result.

The completed [prospective audit](../../../../artifacts/mdb-flux-local-delta-20261105/futures/audit-v1.json) verifies four fresh targets, eight forecasts,14images,746native transitions and both pretarget seals. Local predicted writes yield blue sphere/blue cube or red sphere/red cube, with all8signed object-color forecasts correct and the sphere-left-of-cube relation visually retained. RGB edit error is0.93–2.78% versus copying's100%; RGB MAE is0.0995–0.1117/255. Image/text internal edit error improves to24.81–27.50%/27.81–28.57% versus29.25–32.89%/32.24–34.92%. This is a shared contextual color competition rather than independent object-color control. Absolute mathematics also predicts these coarse scenes; primitive calibration is not shown necessary for that success.

Local image endpoints worsen, one RGB edit worsens, and receiver-by-pattern RGB interaction still has129.54%error with cosine0.0083. The native mixed effect is0.595RMS levels and the predicted norm is83.2%of native, so this is an inaccurate interaction rather than simply a zero prediction. Worker260.815s/process347.04s and local calculations2.24–3.23s per target do not establish speedup. [Visual review](../../../../artifacts/mdb-flux-local-delta-20261105/futures/visual-review-v1.json) and all counterexamples remain attached.

`mdb.response_closure` imports NumPy lazily and imports no model framework. Callers supply arrays with their own bound provenance:

```python
from mdb.response_closure import audit_response, audit_factorial

row = audit_response(
    native_anchor, native_target, reference_anchor, reference_target,
    candidates={"anchored": stored_anchored, "copy": native_anchor},
)
effect_error = row["reference_response"]["relative_rms"]

panel = audit_factorial(
    {"00": native_r0_half, "01": native_r0_new,
     "10": native_r1_half, "11": native_r1_new},
    {"00": predicted_r0_half, "01": predicted_r0_new,
     "10": predicted_r1_half, "11": predicted_r1_new},
)
interaction = panel["contrasts"]["receiver_by_dose"]
```

`reference_response` compares \(P_1-P_0\) to \(N_1-N_0\). A candidate's `error_over_native_effect` instead divides its **endpoint error** by the observed effect size; for an anchored candidate this is also its response error from the exact native anchor. Do not call that field a mathematical response error for the absolute candidate. `modification_from_unrounded_anchor_rms` is a recast term only for the anchored candidate; for foils it includes their intended method change.

Zero reference effects produce `None` relative errors. Shape/finite-value violations are mechanics errors. Numerical accuracy has no boolean scientific gate. Four new audit tests and ten existing partition/reuse tests pass from the installed wheel outside the checkout; two separate framework-free tests prove actual cache-body bypass and restoration on failure.

## Qualification of native incremental execution

The completed finite qualification reuses the six unchanged image-prefix modules: image LayerNorm, Q/K/V projections, and Q/K RMSNorm. Each replacement checks the whole actual call, including shape, dtype, device, stride, tensor equality and absence of unexpected kwargs, then returns a fresh clone. Checkpoint, source, clock, nontext receiving state, epsilon and numerical policy bind the cache. Projection archives are restored to the Linear module's original three-dimensional return shape.

The affected text prefix keeps its full native row geometry; attention, output projections and receiving MLPs remain native. Both incremental executions and both interleaved full controls match their archived complete typed payloads and fingerprints exactly. The two incremental blocks genuinely skip six image affine operations, two LayerNorms and four RMSNorms in total. Twelve cache hits and four continuation receipts verify the schedule. The original-body execution count is inferred from module entries minus cache hits, supported by the scoped helper's source and tests proving that a cache hit never calls the original body. [Independent reuse audit](../../../../artifacts/mdb-response-algorithm-20261105/incremental/audit-v1.json).

This is an exact execution baseline on two known cases with six manually selected modules, rather than automatic dependency discovery. Fixed-order navigation times are 3.3732/0.0561 seconds for incremental/full on the first receiver and 0.0664/0.0585 seconds on the second. Equality checks account for 0.00044/0.00052 seconds in the incremental calls. The measured incremental calls are slower; the panel does not establish wall-time savings. It qualifies actual skipped work and complete numerical closure, against which cheaper boundary replacements can now be tested.

The [public atlas publication](../../../../artifacts/mdb-response-algorithm-20261105/public-records-v1/provenance.json) indexes three supplied generic methods with six occurrences over the same two target states. Six exact-context queries resolve and twelve changed-clock/unknown-input queries abstain. Numerical limitations remain attached, and no tensor bytes are copied into the index. Applicability is supplied, not learned; these are two native targets rather than six independent experiments.

## The discovery loop and symbol record

### Qualify the law through its consumer

The [attention/receiver follow-up](../../../../artifacts/mdb-attention-anchor-20261105/README.md) tests two fixed attention laws on corrected owned Q/K/V, then carries them through the receiving block. Direct attention improves raw boundary-response error9.70–9.82%→7.07–7.15%, but complete-response error worsens. The reversal is demonstrated by the output projection: added baseline error outweighs a smaller edited endpoint error. Full-program raw P0 was not retained, so the exact first boundary remains unresolved; its corrected-prefix baseline differs from the raw-boundary test's native-QKV baseline. Use `audit_response` to inspect `defect_power.anchor_mse`, `target_mse`, `signed_cross_twice` and `change_mse` at each writer; the cross term is uncentered, not automatically statistical covariance.

A fixed attention×receiving-LN factorial retains both full P0/P1 programs. Anchored attention plus direct receiving LayerNorm improves all eight ordinary complete responses: image40.42–42.66%→36.34–38.54%, text39.54–40.84%→38.54–40.32%. Mixed text slightly worsens and weak interactions remain unresolved. This is known-state development with0new native calls, rather than a default runtime policy or unseen-input qualification. [Audit](../../../../artifacts/mdb-attention-anchor-20261105/receiver/audit-v1.json), [scripted worker](../qualification/flux_receiver_norm_diagnostic.py) and [independent auditor](../qualification/analyze_flux_receiver_norm.py).

The workflow now retains complete projected attention, MLP input/output and exit fields, reconstructs baseline post-attention residuals from recorded operands, and verifies exact upstream invariance across receiving-LN choices. Content-addressed BF16 archives reduce368logical references to208unique arrays. Six supplied program IDs have24exact-context occurrences and48changed-context abstentions in the [atlas](../../../../artifacts/mdb-attention-anchor-20261105/index-v2/provenance.json). The following fresh assay tests the selected hybrid against the previous program without tuning further combinations on those development cases.

### Fresh state equations and image futures

The [completed fresh comparison](../../../../artifacts/mdb-flux-hybrid-futures-20261105/README.md) crosses new7/8and1/8text-row dose patterns with two known calibrated image receivers. Both complete programs retain corrected prefix normalization, five frozen attention-affine laws and anchored attention; only receiving LN differs. Eight complete writes and four P0 baselines seal before any learned module forward, then candidate image/readout forecasts seal before four native targets. Both image/text predictions are installed through typed Acts and consumed by the unchanged native suffix/VAE. Counts are14images,746native transitions,24Acts,56receipts and676block entries.

The hybrid improves mean image mathematical-response error18.1609%→16.0875%, with all four cases improving. Image MLP-input response improves13.46–14.52%→9.75–10.54%. Both programs predict all8native object-region color signs and visibly retain sphere-left-of-cube. Hybrid RGB MAE worsens3/4, RGB-effect error improves2/4, and one complete text response worsens. Mixed image/text/RGB errors become100.52%/117.58%/140.81%; numerical and consumer countertrends remain evidence, rather than accuracy gates. [Independent audit](../../../../artifacts/mdb-flux-hybrid-futures-20261105/audit/collection-01/extracted/binding-minimal-result/audit-v1.json).

Full P0/P1 raw attention is retained in this fresh comparison.304logical prediction/native references resolve to181unique public-schema BF16 archives; the audit additionally reads30distinct historical arrays. This closes the prior capture omission for this comparison without retroactively resolving the earlier attention-law reversal. An archive-schema incompatibility was caught while the first attempt was queued; it was cancelled before execution. The corrected worker is tested through the actual public writer/reader and retains the failed stage.

A saved-only [MLP cut](../../../../artifacts/mdb-flux-hybrid-futures-20261105/mlp/collection-01/extracted/binding-minimal-result/audit-v1.json) independently extracts the same MLP subprogram and reproduces24stored own P0/P1 writes exactly. Twelve conditional C0/C1 outputs supply actual native MLP inputs as a diagnostic. Let `H=P−C` be surrogate operand propagation and `L=C−N` the conditional law discrepancy. Endpoint, finite-response and mixed accounts retain `H²+L²+2HL`, including cancellation. The hybrid lowers inherited response-error power in all8image/text cases, while conditional image/text MLP-response error remains12.60–15.25%/9.92–10.60%. This does not identify a specific internal primitive or an accuracy floor. Next, capture first-affine, SiLU/product and output-affine native ports on one preselected saved input, require native replay closure and compare each primitive on identical operands.

Full native scientific custody stays on Beast:1.466GBof scientific files plus original-format archive/log. A CPU mrun audit verifies it there and returns a7.04MBscientific projection,18.95MBwith complete local projection custody. [Authenticated retrieval](../qualification/collect_flux_remote_directory.py) restores exact members on demand; the projection labels missing local arrays/stores explicitly. [Storage workflow](SHARED-TENSOR-STORAGE.md), [native worker](../qualification/flux_hybrid_future.py), [near-data audit](../qualification/flux_hybrid_remote_audit.py), [MLP diagnostic](../qualification/flux_hybrid_mlp_cut.py) and [offline proof](../../../../artifacts/mdb-flux-hybrid-futures-20261105/verification-v1.json) are executable references. Native process160.69s, independent audit43.81s and MLP diagnostic45.37s do not establish a matched inference speedup. Two supplied generic laws gain8atlas occurrences with16changed-context abstentions; no automatic semantic naming is claimed.

### Locating the MLP discrepancy

The completed [primitive cut](../../../../artifacts/mdb-flux-mlp-primitives-20261106/README.md) selects one known row4-heavy/base state at step2/joint1. Two source-extracted original MLP calls, image1024rows/text512rows, reproduce archived outputs exactly. Hooks capture first affine, SiLU, product and output affine with actual paired-view geometry and plain boolean numerical flags. No full model, block, attention or image runs. Original live input strides were lost in prior normalization; full contiguous archived inputs and exact output replay qualify this narrower behavior.

On actual operands, SiLU/product/combined SwiGLU are exact for both target and historical anchor. Anchored first-affine response error is12.57%image/11.50%text; output affine11.29%/8.07%. In output space, `C−Y=(C−Z)+(Z−A)+(A−Y)` separates first-affine propagation, nonlinear propagation and local output-affine discrepancy. Image powers in10⁻⁶ are4.232+0+3.505−1.368=6.369; text39.033+0+39.814−13.734=65.113. Ordered terms and signed cross moments are not unique positive causal shares. Exact local nonlinear agreement is conditional on these operands.

An [independent CPU audit](../../../../artifacts/mdb-flux-mlp-primitives-20261106/audit/collection-01/extracted/binding-minimal-result/audit-v1.json) recomputes24saved fields,8own P0/P1 writes and4previous C0/C1 joins exactly. Ten supplied-scope primitive catalog occurrences resolve10exact contexts and abstain on20changed contexts. Full254.95MBraw evidence remains remote; a551KBscientific projection returns locally. A real3.15MBnative output is retrieved and decoded by hash without rerunning it.

The subsequent [fixed anchored/direct affine factorial](../../../../artifacts/mdb-flux-mlp-primitives-20261106/panel/panel-report-v2.json) uses0nativecalls. Direct first-affine response error drops to0.90%image/0.81%text, while direct output slightly worsens its local response. Direct-first/anchored-output improves complete conditional MLP response15.22→11.30%image/10.32→8.08%text. Both P0/P1 and signed composition interactions are retained;321MBof full predictions stay remote. Its [proof](../../../../artifacts/mdb-flux-mlp-primitives-20261106/panel/verification-v2.json) verifies source/controls/custody, without a second independent recomputation of the new panel. This supports operation-specific numerical laws on the selected known state, not a changed default, fresh forecast or better image editor.

The research helpers are [native capture](../qualification/flux_mlp_primitive_cut.py), [independent audit](../qualification/analyze_flux_mlp_primitives.py), [saved factorial](../qualification/flux_mlp_affine_panel.py), [NumPy archive writer](../qualification/flux_numpy_archive.py) and [offline proof](../qualification/verify_flux_mlp_primitives.py). Nineteen focused installed-environment tests pass. An initial CPU export fails the no-framework guard because the native archive helper imports Torch; the corrected writer matches its actual bytes/metadata and remains framework-free. Failed status/raw output/cost remain.37.04s native,55.71s audit,73.91s failed panel and49.22s corrected panel are experiment costs, not an inference speedup. Next, freeze the revised MLP law inside the full program, predict its own receiving operands on fresh states, and test the native consumer.

### Integrating the MLP on its own predicted inputs

The [own-state integration](../../../../artifacts/mdb-flux-own-state-mlp-20261106/README.md) keeps the hybrid's stored predicted MLP input and post-attention residual fixed. It replaces only the first affine with the preceding direct equation, retaining SwiGLU, anchored output affine, original time gate and BF16 residual addition. Both candidates retain their actual P0/P1. Native-input worlds are separate diagnostics. This crosses four known targets, two known anchors and image/text streams; no new native target or image is executed.

Mean image complete-response error improves16.0875→15.2968%, all eight ordinary image/text responses and all eight edited endpoints improve. P0 bias worsens in both receiver charts, inherited-versus-conditional MLP discrepancy power rises in all eight comparisons, and receiver×pattern mixed errors remain97.15%image/114.06%text. Neither a blanket improvement nor interaction closure is established.

Retain `V=B(gM)`, `Y=B(U+V)` and `rho=Y−U−V`. The finite response is `deltaY=deltaU+deltaV+deltaRho`, with all three powers and all signed pairwise cross terms. The finite-addition discrepancy is appreciable relative to the small response; it does not imply faulty addition or a numerical floor. One text row's addition-error power worsens while total error improves. At the complete exit, `inherited_program_state=P−C` includes both residual and MLP-input differences.

The [independent audit](../../../../artifacts/mdb-flux-own-state-mlp-20261106/audit/collection-01/extracted/binding-minimal-result/audit-v1.json) separately composes36MLPs/60writes from canonical operands and original parameter recipes, checks204raw-bit fields including signed zeros, and reproduces the complete signed report. Forty old-source controls, five arithmetic mechanics tests and ten custody tests pass. Eight public catalog occurrences resolve8exact contexts and abstain on16changed contexts. Scientific errors remain observations; applicability is supplied, not induced.

Calculation73.33s/audit84.08s are CPU-only paid dense arithmetic,0native/framework calls/images. Peak RSS~1.5GB supports3GBfuture reservations instead of6GB. Full548.27MBscientific custody remains remote with a1.50MBlocal scientific projection and exact retrieval. [Worker](../qualification/flux_own_state_mlp.py), [equations](../qualification/flux_own_state_mlp_math.py), [analysis](../qualification/analyze_flux_own_state_mlp.py), [separate recomputation](../qualification/flux_own_state_mlp_audit.py) and [offline proof](../qualification/verify_flux_own_state_mlp.py) make the workflow repeatable. Before claiming fresh image improvement, seal both programs on a withheld receiver and render each one's own P0 as well as P1; using only a native baseline would conceal the worsened starting bias.

### Fresh receiver and actual image continuations

The [fresh-receiver study](../../../../artifacts/mdb-flux-fresh-mlp-futures-20261106/README.md) freezes both MLP programs, draws seed 81119 and advances one native prefix to step 2/joint1. A fixed historical two-row displacement is applied directly to the untouched fresh native text at half, .875/.125 and .125/.875 doses. One explicit native half-state calibration is shared by both programs. This tests supplied-edit transfer with new calibration, rather than semantic-address discovery or calibration-free transfer.

Both own P0/P1 writes seal first. Six candidate image futures and fixed-region forecasts seal while both native-target counters remain zero; native N0 and the two targets render afterward. Mean image-state paired-response error improves 16.5991→15.7173%, all four image/text responses and edited-state endpoints improve, and P0 state bias worsens. RGB paired-response error improves 36.3312→35.9069% for row11 but worsens 28.3670→28.8943% for row4; endpoint MAE worsens for both. The visible red sphere/blue cube arrangement and frozen color-ordering signs remain, with small appearance changes.

The [saved-only continuation](../../../../artifacts/mdb-flux-fresh-mlp-futures-20261106/continuation/collection-01/extracted/binding-minimal-result/diagnostic.json) reads 18 actual latent cuts. Both response rankings already reverse by the next denoising commit and remain worse at the final latent. This brackets the first retained reversal between J1 and logical step 3, including remaining transformer computation and scheduler update. Row11's RGB ranking changes again afterward. Coordinate systems and native effect norms differ across these boundaries; these percentages do not establish cross-space amplification.

For `e0=P0−N0`, `e1=P1−N1`, retain `r=e1−e0` and `mean(r²)=mean(e1²)+mean(e0²)−2mean(e1e0)`. The stored `signed_cross_twice` is **subtracted** here. Row11's final-latent edited endpoint MSE improves while larger baseline error and more opposed errors increase paired-response MSE. The old→new program-change vector and signed crossmoment are also retained.

The separate CPU audit reproduces all 72 complete forecast fields at bit equality, verifies 150 arrays/40 typed cuts/37 receipts and joins each named installed/native state and commit to the exact consumer receipt chain. It binds copied source parents to the authenticated source plan and checks conditioning, schedule and clocks. Eight mechanics tests pass. Four occurrences reuse the earlier generic law IDs; four exact queries resolve and eight changed contexts abstain. [Offline proof](../../../../artifacts/mdb-flux-fresh-mlp-futures-20261106/verification-v1.json).

Native process 175.05s, CPU forecast audit 84.02s and saved-latent diagnostic 33.97s are separate costs. The native lease performs 538 transitions/486 block entries/15 Acts/9 images, one load/noise draw and zero encodes. Full 1.094GB scientific custody remains remote with a 4.00MB local scientific projection. Executed requests remain immutable; measured peaks support smaller future RAM reservations. [Stager](../qualification/stage_flux_fresh_mlp_future.py), [pure program](../qualification/flux_fresh_mlp_program.py), [native worker](../qualification/flux_fresh_mlp_future.py), [auditor](../qualification/analyze_flux_fresh_mlp_future.py), [near-data runner](../qualification/flux_fresh_mlp_audit.py) and [saved diagnostic](../qualification/flux_fresh_mlp_continuation.py) are executable references.

The primary generic evidence sweep succeeds with `target-streams-only`; the four-JSON saved diagnostic has `evidence-incomplete-empty-target`, while its dedicated custody/source/vector verification passes. That is a stream-schema limitation, not a scientific null. The next localization should inspect the J1-to-first-commit interval before changing another primitive or starting with VAE arithmetic.

### Tracing the original futures through their consumers

The completed [native consumer trace](../../../../artifacts/mdb-flux-consumer-trace-20261106/README.md) resolves that interval. Both image-response rankings first reverse at **J2's post-attention residual**: row4 old/revised18.2696/18.4567%, row11 19.7816/20.2397%, after the revised J1 exits were better. Text still improves there. Actual finite-write accounting shows the larger realized attention-write response error outweighs the smaller carried error, with weaker cancellation adding a smaller penalty. The same native attention law consumes different predicted states; a first observed boundary is not a complete primitive or causal-path explanation.

The later curves retain recoveries. Row11's projected-noise response improves34.8462→34.7165%, then latent response worsens48.0837→48.2084%. Incoming latent/sigmas are bit-identical across all nine worlds; `h=−.16648530960083008`. Define `eta=z_after−(z_before+h*noise)` in FP64. The exact paired account is `r_latent=h*r_noise+r_eta`; larger finite-remainder error and weaker cancellation outweigh better noise. The captured scheduler upcasts the sample, multiplies/adds and recasts; scalar FP32 `dt` does not by itself establish FP32 multiplication with BF16 noise. The aggregate remainder is not assigned to a single final cast.

The [native helper](../qualification/flux_consumer_trace.py) starts from nine audited J1 exits and runs25 individual operations per branch using `NativeCutNavigator`. It saves every typed endpoint, joint2..4 image/text attention residuals and readout norm/noise/Euler operands. All nine **complete next-commit payloads and execution points** reproduce the original archived futures. Counts are225 transitions/receipts,207 block calls,0Acts/VAE calls/images/prefixes/noise/encodes. This is a retrospective microscope of existing predictions, not nine fresh forecasts. Single blocks retain complete-block resolution.

The separate [CPU analyzer](../qualification/analyze_flux_consumer_trace.py) verifies234 cuts/484 pages/457 blobs,63 intermediate arrays,130 original parameter recipes and all named operation/receipt/source/clock joins. It emits228 observations,208 same-coordinate finite-write accounts and4 Euler accounts. Normalization/projection are separate maps rather than residual-write edges. Four-world mode comparisons and six-world old/new rankings carry explicit per-world replay qualification. Unqualified values remain trace diagnostics. Fourteen installed-environment mechanics tests pass; [offline verification](../../../../artifacts/mdb-flux-consumer-trace-20261106/verification-v1.json) closes the compact projection to its full remote evidence.

The native process takes177.95s, CPU audit188.34s, full custody206.78s; a failed89.96s setup attempt is retained separately. Full2.375GBscientific files remain remote and a2.75MBscalar/provenance projection returns locally. Pretrained arrays are original retrieval recipes, with0persistent parameter bytes. The observation stream makes the standard sweep applicable:238records/two streams, run/verify0`target-streams-only`; this is mechanics coverage rather than automatic rule discovery or semantic certification.

The [stager](../qualification/stage_flux_consumer_trace.py), [CPU runner](../qualification/flux_consumer_trace_audit.py), [finite equations](../qualification/flux_consumer_trace_math.py), [plotter](../qualification/plot_flux_consumer_trace.py) and [workflow](../../../../artifacts/mdb-flux-consumer-trace-20261106/README.md) make this repeatable without rerendering images. The next causal cut fills old-image/new-text and new-image/old-text input corners at J2, across the half baseline and both edits, then follows their native continuation. Own baselines are required for every corner.

### Crossing incoming image and text

The completed [J2 stream assay](../../../../artifacts/mdb-flux-stream-cross-20261106/README.md) executes those six missing corners plus three old/old shams. One typed Act changes only image or text; all other payload descriptors and the J2 clock remain equal. Each corner retains its own half baseline. Cached new/new and native controls reuse the prior qualified trace. The three shams close exactly at both attention midpoints and complete J2 exit/noise/commit payloads:15 checks.

Both isolated switches worsen J2's image-response error in both edits. For row4, old/old18.2696%, image-only18.5013%, text-only18.3928%, both18.4567%; row11 19.7816/20.1505/19.9584/20.2397%. Text-only holds incoming image exactly fixed yet changes its write, identifying a causal effect through the joint reader. Its immediate text response improves most, but row4's text exit worsens. Text-only noise/latent improves for row11,34.8462→34.3130%/48.0837→47.9699%, and worsens for row4,31.2319→33.0591%/43.8790→45.3574%. It is not a generally better mixed program.

For `R_ab=Y_ab(case)−Y_ab(half)`, retain `A=R10−R00`, `B=R01−R00`, `H=R11−R10−R01+R00` and `r11=r00+A+B+H`. Row4 image-attention powers in10⁻⁶ are41.223/9.220/16.121, with cross terms+7.921/−15.920/−16.185. Strong cancellation makes the total change42.379 and the overall error increase only1.083. These finite terms are not positive causal shares. The [CPU report](../../../../artifacts/mdb-flux-stream-cross-20261106/audit/collection-01/extracted/binding-minimal-result/audit-v1.json) retains64 responses,16 factorial bundles and16 actual attention-write accounts with zero vector reconstruction residuals.

The [native helper](../qualification/flux_stream_cross.py) groups1J2+23readout-bound+1Euler transitions per branch:225 native transitions/207blocks/9Acts/36receipts,0VAE/images. Sparse capture retains39cuts/18midpoint arrays and328.51MBscientific files. [Consumed-source verification](../qualification/flux_qualified_source.py) inherits the original full-custody qualification and freshly hashes every consumed source member; it does not claim another complete verification of the old archive/log. Independent current custody/store/receipt/parameter verification returns a1.27MBprojection. Native94.24s, CPU47.40s, custody26.71s are separate measured costs; changed capture scope prevents a controlled inference-speed claim. Three new instrument tests pass; standard evidence run/verify0covers74records/two streams. Observation clocks correctly distinguish the attention input, next-block entry, scheduler and next denoising commit. [Offline proof](../../../../artifacts/mdb-flux-stream-cross-20261106/verification-v1.json).

The [stager](../qualification/stage_flux_stream_cross.py), [CPU runner](../qualification/flux_stream_cross_audit.py), [analyzer](../qualification/analyze_flux_stream_cross.py), [factorial equations](../qualification/flux_stream_cross_math.py) and [workflow](../../../../artifacts/mdb-flux-stream-cross-20261106/README.md) are qualification helpers using public runtime APIs. The next smaller mechanism cut crosses actual positioned text K and V while holding image Q/K/V fixed, qualifying native diagonal midpoint closure before interpreting allocation versus content. The existing [attention partition](ATTENTION-RESPONSE.md) explains the real-arithmetic distinction; native BF16 dispatch remains a separate numerical contract.

### A measured numerical refinement

The [attention-arithmetic study](../../../../artifacts/mdb-attention-arithmetic-20261105/README.md) selects a fixed1,024-coordinate carried BF16 accumulation rule from saved text-Q evidence, checks disjoint Q/K/V coordinates, and tests the unchanged rule on12affines. It nearly reproduces sampled attention affines while performing worse on the tested MLP shapes. The program therefore binds the rule to five active attention ports; module geometry and numerical profile are applicability conditions.

Four new asymmetric states then receive eight complete forecasts sealed before native target execution. Image installed-edit error improves42.63–45.21%→40.63–42.63%, text45.20–47.25%→39.81–41.51%; small receiver interactions remain inaccurate. The [verified audit](../../../../artifacts/mdb-attention-arithmetic-20261105/fresh/audit-v1.json) distinguishes forecast operands from post-hoc actual-input diagnostics and native attention controls. No images are produced in this numerical cut.

On actual native modulated operands, the frozen Q/K/V rule matches99.9593% of36,864 changed-row values across two distinct text inputs. Owned modulated inputs already have about0.295% error: a near-exact affine primitive cannot repair upstream operands. The direct LayerNorm/modulation equation matches those saved inputs exactly. Anchoring can transport a baseline BF16 rounding residual into a different input, so it is a qualified approximation rather than a universally better law. A separate saved-development assay tests fixed absolute-normalization variants, retaining P0/P1 arrays and reporting mathematical response separately from the installed edit. Fresh validation of a selected revision remains necessary.

The completed [saved normalization audit](../../../../artifacts/mdb-attention-arithmetic-20261105/normalization/audit-v1.json) verifies144arrays and reproduces the original arithmetic-control exits exactly. Direct LN makes changed-row modulation exact; direct Q/K RMS reduces about3,200positioned mismatches perfield to2–6/6,144values. Whole-block mathematical-response errors remain40.38–42.66%image and38.90–40.84%text across the two new variants. LN alone improves every comparison, while adding RMS has mixed effects; mixed receiver errors remain above100%. The run uses0nativecalls/0frameworks/0VRAM,74.478worker/131.27process seconds. This localizes prefix arithmetic without closing the attention/MLP response or adding new image evidence.

Two generic programs enter the public atlas with eight exact-context occurrences and explicit counterexamples;16changed-clock/unknown-input queries abstain. This does not learn applicability or semantic names. Seven instrument tests pass. Import, checkpoint hydration, preparation, native controls, collection and auditing remain costs; the fresh process takes702.91seconds and establishes no speedup.

```text
bind parent, clock, edit and downstream consumer
build/read the dependency program
reuse unchanged writes
execute affected writes at the cheapest qualified level
freeze new-case forecasts and run selected native comparisons
find the first consumer-relevant discrepancy
distinguish numerical, closure, consumer and inherited error
refine an operand, primitive, chart or domain
test a fresh combination and retain its counterexamples
```

A computation record needs a resolver/address, current operands, clock, checkpoint/numerical profile, receiving-state conditions, law, reads/writes/invalidations, consumer, observed target/collateral/continuation effects, cost and counterexamples. A generic operation ID can remain stable while live addresses and payloads change. Adding a receiver ID is bookkeeping; an explanatory refinement identifies an operand or interaction that accounts for more receivers.

Graph cuts rank candidate experiment boundaries under a declared objective. Spectral directions propose charts; they do not establish semantic meaning or closure. Shared tensor storage and model virtual memory preserve common parents and immutable payloads, but attention summaries still depend on their queries. Dynamic circuits describe the active dependency routes; time circuits compose their writes and persistent state across clocks. These become different views of the same state-transition program.

Native baseline, preparation, equality checks, parameter hydration, seals, validation, collection and failures all count toward cost. The present work establishes a discovery method and bounded instruments, rather than a measured general speedup or a complete semantic compiler.
