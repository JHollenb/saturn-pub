# A self-fed FLUX forecast through joint1

This campaign extends the [reentry forecast](FLUX-REENTRY.md) by one complete joint attention/MLP block and adds a matched native bridge. Its question is whether a discrepancy at the next boundary was transported from earlier predictions or introduced by the current mathematical block.

The follow-up [joint-tail campaign](FLUX-JOINT-TAIL.md) extends this method through all five next-step joint blocks, with three matched local comparisons and six common-boundary continuations.

**Actual run and independent audit completed.** The 19-transition forecast reaches joint2 with image/text relative RMS discrepancies of 0.521577%/0.345148% against native. At this boundary, local joint1 numerical error partly opposes transported upstream error in both streams. After 25 native transitions and VAE, the complete predicted-state arm differs by 0.102547 RGB channel levels out of 255. The remaining suffix is native; this is a mathematical forecast from an observed internal prefix.

Job `job-0cda5797a845` succeeded in 466.71 runner seconds with peak RSS 8482.1 MB/VRAM 9590 MB. The [authenticated collection](../../../../artifacts/mdb-flux-joint-chain-20261104/run-01/collection.json) retains 625 files/5,797,359,286 bytes. Producer, remote-before, remote-after and local inventories agree exactly. The [unchanged pre-outcome auditor](../../../../artifacts/mdb-flux-joint-chain-20261104/audit.json) passed on its first execution in 72.3454s. All 80,920,576 checked output scalars match the separately implemented, independently fed recurrence bit for bit. This comparison qualifies the mathematical reconstruction; native CUDA differences remain nonzero:

| Forecast output | Error RMS / corresponding native RMS |
|---|---:|
| Committed Z3 |0.213888%|
| Next projected image / text |0.294009% /0.256406%|
| Joint0 image / text |0.428283% /0.294086%|
| Joint1 image / text |0.521577% /0.345148%|

The forecast sealed at 266.180444 worker seconds, before the first native reference exit at 268.848758s. All 25 selected-target counters were zero at the seal; the source-qualified census matches exactly. The two regenerated RoPE arrays are native-bit-exact. Time embedding/image modulation/text modulation/single modulation retain 1/49/52/18 native discrepancies respectively; predicted context is carried without native refresh.

The common-joint2 decomposition uses the same native denominator for all three terms:

| Stream | Transported upstream `B−N` | Local joint1 `P−B` | Total `P−N` | Cosine between terms |
|---|---:|---:|---:|---:|
| Image |0.527689%|0.292997%|0.521577%|−0.298361|
| Text |0.346243%|0.121194%|0.345148%|−0.184027|

Both vector identities have zero measured residual. The local terms are substantial and partly opposing, leaving total norm slightly below the transported term. This is numerical cancellation in this tested state, not a semantic repair mechanism or a rule for later blocks. The stored latent has zero local joint1 contribution, as expected for a block that carries it unchanged.

| Future | Final Z4 relative RMS discrepancy | Final RGB channel MAE,0–255 | Maximum channel difference |
|---|---:|---:|---:|
| Native |0%|0|0|
| Sham |0%|0|0|
| Predicted joint0 plus native joint1 |1.231773%|0.103616|8|
| Complete predicted joint1 |1.239763%|0.102547|7|

The complete prediction is slightly closer in RGB and slightly farther in final latent RMS than its native-bridge comparison. These are distinct measured observables. Small mean pixel error and numerical vector cancellation do not establish an object-level explanation. See [actual native VAE futures](../../../../artifacts/mdb-flux-joint-chain-20261104/native-futures-v2.png) and [state/error decomposition](../../../../artifacts/mdb-flux-joint-chain-20261104/state-errors-v2.png).

Qualification passed 134 installed-wheel tests, including the 69 prior reentry checks. A separate [byte-level closure spotcheck](../../../../artifacts/mdb-flux-joint-chain-20261104/independent-postrun-spotcheck.json) also passed. Its first reader rejected a read-only mapping container before examining leaves; the [recorded auxiliary correction](../../../../artifacts/mdb-flux-joint-chain-20261104/spotcheck-correction-v2.json) accepts mapping containers while retaining exact dtype/shape/byte checks. The failed attempt and original source remain saved. Primary audit, forecast and native evidence were unchanged.

The specimen is FLUX.2 Klein 4B distilled, native CUDA BF16, batch size 1, 512×512, four denoising steps. The prompt is `A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph.` Seed 101 is new to this forecast campaign. The forecast starts from the observed step2/pre-single6 state after 68 native transitions.

The mathematical continuation consumes its own complete state through14 single blocks, readout, Euler commit, the next projection, joint0 and joint1:19 native transition equivalents. It carries its own predicted latent, regenerated time context, image stream and text stream. The final25 transitions and VAE remain native. This is a bounded mechanical forecast from an observed prefix.

For one joint block, let `S` contain both working streams and the modulation/rotary context. The attention write is followed by normalization/modulation of the receiving residual and then the MLP write. With separate image and text equations suppressed, the serial structure is

\[
U = X + g_a\odot A_\theta(\operatorname{mod}(\operatorname{LN}(X),s_a,b_a)),\qquad
X' = U + g_m\odot M_\theta(\operatorname{mod}(\operatorname{LN}(U),s_m,b_m)).
\]

Attention mixes the two streams. Each stream's MLP then consumes its updated residual. Joint0 and joint1 use the same mathematical template with different named parameter banks. No target activation supplies either next input.

Let `N` be native joint1 from the native joint0 state, `B` native joint1 from the complete predicted joint0 state, and `P` mathematical joint1 from that same predicted state. All meet at step3/joint2. Then

\[
P-N=(B-N)+(P-B).
\]

`B−N` measures upstream discrepancy after native joint1 transports it. `P−B` measures the mathematical/native discrepancy of joint1 on identical incoming operands. It includes differences between numerical executors; it is not evidence that the architectural equation is absent. Their cross-dot measures reinforcement or opposition:

\[
\|P-N\|^2=\|B-N\|^2+\|P-B\|^2+2\langle B-N,P-B\rangle.
\]

This is an ordered finite-difference identity. It does not uniquely attribute object concepts or partition semantic causal effects.

| Branch | Installation | Bridge to common joint2 | Final native suffix |
|---|---|---|---|
| `native` | Native joint1 exit | None |25 transitions + VAE|
| `sham` | Exact native complete closure | None |25 transitions + VAE|
| `inherited.predicted` | Predicted joint0 latent/context/image/text | One native joint1 |25 transitions + VAE|
| `joint1.predicted` | Predicted joint1 latent/context/image/text | None |25 transitions + VAE|

The opt-in `FluxJointForecastClosureAdapter` admits atomic replacements at live joint cursors1/2 under a distinct execution identity and state schema. Its Act writes all four ports and binds13 source preconditions, including conditioning, IDs, schedule and clock. Native advancement is inherited unchanged. The preceding closure adapter keeps its original source and policy.

The whole forecast is sealed before the selected target occurrences execute. Earlier denoising steps in the observed prefix have already used these module classes; the zero-before-seal claim concerns the selected current/next-step occurrences. Every transformer/VAE module is guarded during prediction. The sealed hypothesis, source bindings, full prediction,19 reference boundaries, full bridge input/output, common/final cuts, receipts and images are retained.

Each branch has26 observation points: common joint2 plus25 exits. Native arrays are retained in full; intermediate candidate observations retain full descriptors, online signed comparisons and64 fixed witnesses. Complete candidate common/final cuts remain independently auditable. A receipt does not imply every intermediate candidate tensor was archived.

The finite census is68 prefix +19 reference +1 bridge +4×25 suffix =188 native transitions,3 atomic Acts,191 receipt emissions and4 VAE calls, in one lease/load/encode/noise draw. CPU mathematical evaluations are additional work and are not native model calls. This design does not measure inference speedup.

The implementation is [the campaign](../qualification/flux_joint_chain_campaign.py), [independent saved-only audit](../qualification/analyze_flux_joint_chain.py), [public closure](../src/mdb/flux_joint_forecast_closure.py) and [public equation/atlas publisher](../qualification/publish_flux_joint_chain.py). The numerical predictor and independent reference are the unchanged, separately implemented reentry functions. Real-model runs use mrun; installed-wheel qualification uses tiny actual native classes and explicitly synthetic consumers.

A computation record describes the update law, its parameter/source bindings, required state, applicability and observed consumer effects. The live state is the tensor payload plus its clock and carried dependencies. The continuation identifies the next consumer. The symbol table indexes these records; it is not a replacement for the state itself. Chaining requires that one record's complete output satisfy the next record's input contract.

The generic law name `flux.joint.attention-then-mlp` describes a recurring computation template. A binding selects the actual block weights and state. Sharing that name across bindings does not establish a universal object concept, a transferable numerical error bound or applicability to untested states. The matched bridge makes a practical debugging distinction: a later record can receive a different state even when its local equation remains a good account of the native computation.

To reproduce the workflow, use an installed public MDB wheel containing the new closure module, Saturn-pub0.4.1 and the source-qualified Diffusers0.39/Torch2.13 profile. The launcher consumes immutable helpers from the preceding admitted reentry run. Staging creates a new directory and fixes sources, model/input geometry, bounds and offline auditor before admission:

```bash
MDB_PY=/path/to/qualified/python
MDB_Q=/path/to/saturn-pub/packages/mdb/qualification
MDB_RUN=/path/to/new-run
"$MDB_PY" -I "$MDB_Q/submit_flux_joint_chain.py" "$MDB_RUN" \
  --prior-run /path/to/qualified-reentry-run
"$MDB_PY" -I "$MDB_Q/submit_flux_joint_chain.py" "$MDB_RUN" --submit-staged
```

Inspect the returned job with `mrun why JOB` and `mrun attach JOB`. The launcher uses strict guarded admission and one finite worker with no automatic retries. After an authenticated succeeded job, collect the declared result files and run the exact frozen auditor:

```bash
"$MDB_PY" -I "$MDB_Q/collect_flux_joint_chain.py" "$MDB_RUN" JOB
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  "$MDB_PY" -I "$MDB_RUN/independent-math-freeze/analyze_flux_joint_chain.py" \
  "$MDB_RUN" --out /path/to/new-audit.json
"$MDB_PY" -I "$MDB_Q/publish_flux_joint_chain.py" \
  "$MDB_RUN" /path/to/new-audit.json /path/to/new-publication
```

The collector verifies producer, remote-before, remote-after and local file inventories. The auditor reads saved arrays and typed receipts; it never executes a learned model. The publisher uses the installed public EquationCatalog and ComputationAtlas to record both streams against clean native and matched native bridge references. These four comparisons reuse two sealed output fields from one history.

The underlying debugger pattern is also available directly:

```python
from saturn_pub import Session
from mdb.flux_joint_forecast_closure import closure_act
from mdb.native_navigation import NativeCutNavigator

nav = NativeCutNavigator(Session.from_cut(adapter, joint1_cut), max_steps=26)
nav.apply(closure_act(joint1_cut, latent=predicted_z3,
    context=predicted_context, image=predicted_joint0_image,
    text=predicted_joint0_text, name="joint1.matched-incoming-state"))
bridge_receipt = nav.step()
native_joint1_on_predicted_state = nav.capture()
```

Here the `predicted_*` operands are complete source-bound values converted into the qualified native layouts without changing their sealed numerical values. The campaign performs those checks, capture and receipt custody explicitly; this small excerpt illustrates navigation, not the whole prospective transaction.

The accepted [public publication](../../../../artifacts/mdb-flux-joint-chain-20261104/equation-publication-v1/result.json) contains four observations over two sealed output fields, comparing each against clean native and matched native bridge. The [explicit cross-run federation](../../../../artifacts/mdb-flux-joint-chain-20261104/cross-run-atlas-v1/result.json) retains 14 observations from this history and seed 97's preceding reentry history. All imported observation fingerprints remain unchanged. Six observations share the joint attention/MLP law name; the ten earlier records also contain projected fields and two dependent composed views.

```python
from mdb.computation_atlas import ComputationAtlas

atlas = ComputationAtlas("/path/to/cross-run-atlas-v1/atlas")
joint = atlas.find(law_id="flux.joint.attention-then-mlp")
assert len(atlas.find()) == 14
assert len(joint) == 6
context = next(row["equation"]["tested_context"] for row in joint
    if row["equation"]["tested_context"]["seed"] == 101)
assert len(atlas.find(applicable_to=context)) == 4
assert atlas.find(applicable_to={**context, "seed": -1}) == []
```

The source-bound federation actually ran without importing Torch. It preserves exact applicability, signed effects, native continuation and numerical counterexamples. Grouping by generic law is a retrieval operation; automatic family discovery and rule induction remain open.

The standard Saturn evidence sweep also ran and verified over 247 event records, but recognized zero public typed claims/address references across 345 JSON files. Its `target-streams-only` status is an instrument coverage limit, separate from the successful public equation/atlas publication and mathematical audit. [Sweep evidence](../../../../artifacts/mdb-flux-joint-chain-20261104/evidence-v1/run.json).

The [prepared-weight request](../../../../artifacts/mdb-flux-joint-chain-20261104/request-prepared-weights-v2.json) now includes this measured625-file/5.8GB workflow. Source-bound parameter reuse with bounded per-bank loading would reduce repeated evidence copies; a reusable paired observer and stable installed forecast-law API would remove campaign-specific glue. No speedup was measured. The next scientific cut is the remaining joint blocks under the same matched-state bridge, followed by closing the remaining denoising pass to predict final latent without a native denoiser suffix.
