# Following appearance backward: conditional state and write cooperation

Two further cuts extend [the late residual factorial](FLUX-ANIME-RESIDUAL-FACTORS.md). A reciprocal depth curve shows direction-dependent head/costume transfer, and a joint0 factorial shows that weak isolated writes can have strong effects when paired with their matching residual. We also forecast complete post-block tensors exactly from observed source writes, then independently execute their candidate calculations and native image continuations.

These results support a computation record that names a write together with its receiving state, clock and consumer. They do not identify an isolated universal “blond hair MLP,” a first concept-formation layer or complete prompt-to-image understanding.

## The reciprocal curve

Job `job-a19dc0ac9245` imports actual step3 projection frontiers from the known bridge scene. The model is FLUX.2 Klein4B distilled, checkpoint inventory SHA `dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f`, CUDA BF16, physical batch1,512×512 and four denoising steps. Step3 is the last step, zero-based. No prompt is encoded again.

Two natural histories supply recipient/source states. At joint0, joint4 and single0/6/10/14/18, the worker replaces either the upper96 or lower252 selected-left image rows in both directions. Every replacement preserves its own recipient's latent, working text, conditioning/context and outside image rows exactly. The histories themselves differ beyond the band; they are not isolated semantic single-trait examples.

The panel executes28 interventions, two natural continuations and two shams. Six independent historical comparisons are exact: both natural PNGs, both shams and the two forward single18 band cases from the previous experiment. All528 native transitions,30 Acts and558 receipt emissions close under independent custody/lineage audit.

![Upper-band reciprocal head responses: native then seven depth sites; recipient directions occupy separate rows](../../../../obsidian/images/flux-anime-origin-20261104/origin-heads.png)

Violet-haired hooded head features transfer into the blond recipient even before joint0. In the reverse direction, early upper edits leave short pale-violet hair without the full original hood; from single6 onward, the reversed result is blond. Upper forward edits can also change the right figure's hair while its robe and staff remain. Lower edits predominantly alter the costume response while retaining coarse recipient head appearance.

This gives a candidate response interval between pre-single0 and pre-single6. It does not isolate a necessary repair block: the naturally evolved donor tensor changes together with the intervention clock. A paired branch trace and same-payload restoration are needed to separate changing payload from downstream repair.

The pixel instrument also exposes its limit. An early reverse head rectangle has about0.64 directional progress toward the other native image while the hair still looks violet. That rectangle includes face, hood, shoulder and background. Its scalar is not a hair-color or identity decoder. [Native review and limitations](../../../../artifacts/mdb-flux-anime-origin-20261104/scout-run-03/appearance-review.json).

## The next inward cut: sequential joint0

Job `job-15fbdff68553` imports the scout's actual joint0 parents. It prepares upper/lower sources by replacing only the declared incoming image rows, with all incoming nonimage state matched to the baseline. It then captures the native residual R, image attention return A, MLP return M, time gates and actual block exit.

The joint block is sequential:

\[
A=\operatorname{Attention}(N_I(I),N_T(T);P)_I,
\quad U=Q\bigl(R+Q(g_A\odot A)\bigr),
\]
\[
M=\operatorname{FF}(N_2(U)),\qquad
I'=Q\bigl(U+Q(g_M\odot M)\bigr).
\]

Q denotes native BF16 rounding at the multiply/add edges; the normalizers include the time modulation. The MLP acts separately on each position, but reads a position state that attention has already changed.

That ordering is visible in the measurements. The upper source has zero changed residual elements outside its96 input rows, but2,119,343 changed attention-return elements and2,257,610 changed MLP-return elements outside them. Lower-source counts are0,2,056,627 and2,056,698. The pointwise MLP need not mix positions to respond to information received from other positions through attention.

The earlier single18 assay instead used parallel attention and MLP features feeding one fused projection. Its MLP features stayed exactly local to the input edit. These are different architectures and port types: joint0 M is a3072-channel projected MLP return, while single18 M was a9216-channel pre-fused operand. Their raw values and energies are not interchangeable semantic measures.

## A weak write becomes effective in a matching receiving state

For each band, R, A and M independently select recipient or source values. Both attention and MLP image returns are explicitly frozen to the selected observed ports in the mixed calculation. An A-only intervention therefore does not silently recompute a new MLP return and call it recipient M. R chooses the direct candidate's true incoming image state. Working text stays recipient until the coherent RAMT completion.

The native joint helper executes those separately counted candidate blocks and checks its sequential equation. Public typed image Acts then install the calculated exits after an unmodified native baseline step. Every suffix is executed by the unchanged NativeCutNavigator. The worker records the calculation parent separately from the continuation parent.

![Native outputs: recipient, residual only, MLP only, residual plus MLP, coherent source](../../../../obsidian/images/flux-anime-origin-20261104/joint-cooperation.png)

In full-resolution originals, upper R alone changes the left head toward violet hair and a hood, while the right hair remains predominantly violet. M alone retains a blond left head and violet right head. RM changes the right hair to blond, while its robe and gold-orb staff remain; RA gives a similar coarse response. The lowered left sword and substantial silver armor also remain. These are conditional trait/association changes, not complete character transfers.

For a supplied RGB rectangle, define directional source progress

\[
p(Y)=\frac{\langle Y-B,S-B\rangle}{\|S-B\|^2}.
\]

| Source ports | Upper left-head rectangle | Upper right-head rectangle |
|---|---:|---:|
| R |0.6358|0.3792|
| A |0.0137|0.0277|
| M |0.0461|0.0401|
| AM |0.0936|0.1042|
| RA |0.9219|0.8238|
| RM |0.9300|0.8670|
| RAM |0.9976|0.9980|

These are signed pixel projections, not semantic percentages. The hypothesis that an isolated current attention/MLP write would explain more of the upper companion response than R is unsupported. The stronger observation is cooperation with the receiving residual. Neither the input residual nor the write has a standalone fixed semantic interpretation in this panel.

## The cooperation precedes image decoding

For a write choice W, retain the finite interaction at each plane:

\[
\mathcal I=Y_{RW}-Y_R-Y_W+Y_0.
\]

The upper R×M interaction contributes+0.4477 to the right-head directional projection. Its RMS is0.000652 at the block's image exit,0.10246 in the final carried latent and0.03735 in RGB. The interaction already exists before VAE decoding. These values use different spaces and scales and are not an amplification ratio or a decomposition of semantic responsibility.

The exit nonadditivity includes native rounding; the following transformer/readout/scheduler computations are nonlinear. This identifies a finite conditional interaction, not an abstract semantic AND gate. Results are heterogeneous: lower-band R×M and R×AM right-head interactions are negative. [All signed interactions and planes](../../../../artifacts/mdb-flux-anime-origin-20261104/joint-run-01/composition-v1.json).

## Complete numerical forecasts, with explicit coverage

After three source cases, the worker seals complete predictions for each new R/A/M combination:

\[
\widehat I'_{r,a,m}=Q\!\left(Q(R_r+Q(g_A A_a))+Q(g_M M_m)\right).
\]

It stores lossless BF16 bits for all1024×3072 image features per case. Fourteen unique mixed cases match their independently executed candidate outputs exactly:44,040,192 values, maximum error0. A separate [bitwise audit](../../../../artifacts/mdb-flux-anime-origin-20261104/joint-run-01/bitwise-forecast-v1.json) checks all uint16 representations, including signed zero, with zero mismatches. Two RAMT cases reuse RAM's image prediction and are excluded from that total. The offline auditor reconstructs every complete prediction independently and checks it against the actual typed image Act payload.

The inputs to those forecasts are observed native R/A/M returns and gates. We have not independently predicted attention or MLP returns from raw prompt/state, nor predicted the final RGB without native continuation. Exact local accounting and semantic scene explanation have different coverage.

All six joint controls are exact: three source/reference PNGs, sham and two coherent RAMT reconstructions. RAM alone has source post-block I but recipient T, leaving small final RGB differences; adding source T closes each PNG exactly. This is evidence for a full working-state closure, not a minimal necessary semantic program.

Hypotheses and request identities were externally pinned before these runs' outcomes were collected. Numeric forecasts are ordered and hashed before mixed candidate calls by the immutable worker. Their hashes were observed externally after remote execution, before collection; we do not claim an externally witnessed pre-execution numeric seal. The scout additionally discloses previously viewed previews from an incomplete cancelled attempt. These are exploratory, source-assisted experiments, not held-out semantic rule certification.

## Alignment with the scene generator and editing circuits

A November4 predecessor review narrows the novelty of this campaign. The existing scene-generator work already joins source programs, recipient addresses, clocked writes, carried image history and native consumers. Even attention/MLP decomposition and conditional interactions have close predecessors. These results extend and qualify that architecture through public MDB; they do not originate it. The [September9 prior-art correction](../../../../saturn/results/flux-persistent-object-program/corrections/2026-09-09-prior-art-audit.md) records an earlier instance of this same overlap.

| Existing evidence | Connection to the current assay | Remaining boundary |
|---|---|---|
| [Complete scene source compilation](../../../../research/demos/scene-generator.md):21/21 recorded native RGB outputs across seven specimens/five families; static source sealed before the target render | A complete source drives the native machine from its input boundary. MDB describes selected consumers inside that machine | Exact source substitution does not explain every semantic computation or predict intermediate writes independently |
| [Eye color/shape circuit chart](../../../../saturn/experiments/2026-08-27-flux-rosetta-kv-head-token-coalition/FINDINGS-POPULATION-GEOMETRY-CHART.md): eye core/face ring, contextual text K/V, shared eight-head coalition, parallel attention/MLP decomposition and causal continuation | R/A/M accounting is a finer numerical qualification and an additional sequential-joint contrast, not the first separation of these paths | Historical coordinates and payloads require model/profile/parent qualification before reuse here |
| Same chart: target scheduler register plus final-step text/image pre-state reproduces target RGB exactly; either alone does not | RAM versus RAMT closure fits the established distinction between a local calculation and complete continuation state | Captured-target closure is source-assisted sufficiency, not target-free discovery of the edit |
| [Held-out object algebra](../../../../saturn/experiments/2026-08-18-object-heldout-algebra-isolation2/FINDINGS.md): wrong-address effects, contextual pose, non-setpoint deltas, early writes suppressed by baseline-pinned later readouts | Residual-dependent edits and apparent repair have causal precedents | The reciprocal curve varies both donor payload and landing clock; it has not isolated a repair block |
| [Persistent four-edit program](../../../../saturn/experiments/2026-09-09-flux-persistent-object-program/FINDINGS.md): complete denoising episodes consume preceding generated RGB plus explicit source history | Branch records should retain both image history and source-program history | Host-supplied handles are not automatically discovered neural object identities |

The common structure is `source/context → addressed and clocked computation → working text/image state → scheduler register → later native computation → RGB`. At diffusion-step scale, `L[t+1] = Scheduler(L[t], Transformer(L[t], E, t))`: carried sample L and transformer action are distinct parents. Our BF16 equations account for one block inside that action using observed A/M returns. Neither scope amounts to complete prompt-to-image semantic understanding.

“Symbol table” also covers distinct representations: lexical manifests supply prompt addresses; native reference values/IDs supply an input binding; contextual T/I tensors and L carry evolving neural state; scene-transaction entity/relation registers express host scene semantics. The [scene transaction implementation](../../../../saturn/src/saturn/scene_transaction.py) explicitly does not claim its logical registers are exposed natively inside diffusion. Connecting these representations is the research task. The existing [scene circuit compiler](../../../../saturn/src/saturn/scene_circuit_compiler.py) already separates full-scene generation authority from per-symbol editing authority and retains unresolved symbols.

The next useful integration is a **same-artifact eye-color × eye-shape study**, reusing known source programs, eye/face supports and the register/program factorial. First qualify model inventories, schedules, geometry and native controls; old addresses are not automatically interchangeable with this512-pixel campaign. Bind semantic edit records to MDB module ports and parent-bound occurrences. Freeze a bounded prediction for an unexecuted composition: local tensor response, eye-change direction, named collateral and continuation behavior. Disclose observed source assistance. Use matched-payload restorations and baseline-pinned later readouts to separate formation, transport, overwrite and repair. This integration and prospective test have not been executed here.

## Symbol-table records and what remains

The [42 added occurrences](../../../../artifacts/mdb-flux-anime-origin-20261104/computation-records.json) bring the combined [exact citation query](../../../../artifacts/mdb-flux-anime-origin-20261104/evidence-v1/xref-query.json) to94 model/checkpoint/parent-bound records. They retain direction, clock, support, source/calculation/continuation parents, port selection, target/collateral response, full numeric errors and counterexamples. IDs remain generic. Automatic semantic family discovery and rule induction remain open.

If continuing the current anime branch, trace an early reverse-edit trajectory through single0–6, then restore matched payloads at matched boundaries using the prior editing controls above. Separate image-band, outside-image and working-text paths before naming a repair mechanism. Backward origin work must also cross the previous scheduler commit into carried latent L; this scout starts in the last denoising step. The higher-priority integration is to join these measurements to existing semantic edit programs rather than repeat a broad discovery sweep.

## MDB workflow and execution improvements

Use the installed public packages and the [scout worker](../qualification/flux_anime_origin.py), [scout submitter](../qualification/submit_flux_anime_origin.py), [joint worker](../qualification/flux_anime_origin_joint.py) and [joint submitter](../qualification/submit_flux_anime_origin_joint.py). Run the independent saved-evidence auditors between stages:

```bash
python qualification/submit_flux_anime_origin.py SCOUT_RUN --predecessor AUDITED_DEPTH_RUN
python qualification/submit_flux_anime_origin.py SCOUT_RUN --collect SCOUT_JOB
# Verify retained source/custody, native controls, Acts and complete receipt chains.
python qualification/submit_flux_anime_origin_joint.py JOINT_RUN --predecessor SCOUT_RUN
python qualification/submit_flux_anime_origin_joint.py JOINT_RUN --collect JOINT_JOB
```

The qualified working cache is now `MDB_WORKER_UV_CACHE_DIR=/mnt/big/llm-models/.uv-cache`, with the existing model roots and Torch home from [the original workflow](FLUX-ANIME-SCENES.md). The new launchers disable bytecode writes; workers declare SSD output scratch at `/home/beast/.cache/mdb-research/JOB_ID`. They retain installed public MDB0.1.0/Saturn-pub0.4.1/mrun-pub0.1.2, Python3.14.6, Torch2.13.0+cu130 and Diffusers0.39.0. Full resolved dependency inventories remain in raw logs; incidental cached versions differ from the HDD environment, while all12 native PNG comparisons are exact. This is an observed qualified profile, not proof that arbitrary dependency changes preserve parity.

HDD journal contention caused one cancelled partial model attempt and one cancelled bootstrap-only attempt. Both remain execution-null in the [attempt ledger](../../../../artifacts/mdb-flux-anime-origin-20261104/attempt-ledger.json); their outputs are not counted as completed campaign evidence. The corrected scout took72.71 worker/146.32 scheduler seconds, peak RSS3,045MB/VRAM9,256MB. The joint panel took75.08/133.85 seconds, peak RSS5,538.1MB/VRAM9,342MB. No matched speedup factor is claimed against interrupted runs.

The scout producer also omitted its request SHA from the outer summary. The collector retained its complete log before rejecting that omission. [Offline recovery](../../../../artifacts/mdb-flux-anime-origin-20261104/scout-run-03/collection-qualification.json) verified the unchanged archive, manifest-bound source identity and scheduler request; the raw report remains unchanged. Future producer summaries are fixed. The legacy transaction CPU-FP32 label and exclusion of external Navigator calls remain qualified by actual CUDA BF16 receipts and explicit native/candidate counts.

The completed panels add52 native images including controls, for138 completed campaign images. Independent audits verify1,068 native transitions,51 Acts and1,119 receipt emissions, plus20 explicitly separate candidate block calculations. Evidence run/verify/query/stats return0; the complete BF16 forecast mechanics plus native joint helper tests pass5/5. [Scout audit](../../../../artifacts/mdb-flux-anime-origin-20261104/scout-run-03/analysis-v1.json), [joint audit](../../../../artifacts/mdb-flux-anime-origin-20261104/joint-run-01/analysis-v1.json), [joint appearance review](../../../../artifacts/mdb-flux-anime-origin-20261104/joint-run-01/appearance-review.json).
