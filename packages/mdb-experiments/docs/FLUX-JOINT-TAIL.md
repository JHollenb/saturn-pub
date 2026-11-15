# Chaining a FLUX forecast through all five joint blocks

This campaign extends the [joint1 forecast](FLUX-JOINT-CHAIN.md) through joint2, joint3 and joint4. It tests a complete mathematical prediction from an observed internal prefix, with matched native bridges that distinguish inherited error from each new block's local numerical discrepancy.

The follow-up [final-latent campaign](FLUX-FINAL-LATENT.md) extends the forecast through the remaining denoising suffix and uses the native VAE as its final consumer.

The specimen is FLUX.2 Klein 4B distilled, CUDA BF16, batch size 1, 512×512 and four denoising steps. Seed 103 is fresh to this campaign. The prompt is `A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph.` A native prefix of 68 transitions reaches logical step 2 immediately before single block 6.

The mathematical program then computes 22 transition equivalents: 14 complete single blocks, readout, Euler commit, next-step input projection and all five joint blocks. Each stage consumes the complete output of the preceding mathematical stage. The predicted latent, regenerated context and both working streams travel together. There is no donor or intermediate native activation refresh.

All outputs seal before the selected later native target occurrences execute. Earlier denoising steps in the observed prefix have already used the same module classes. The prospective claim concerns the selected later inputs and outputs.

## Measured result

**The actual run and independent audit completed.** Job `job-7640a7cd67b8` succeeded in 543.02 runner seconds with peak RSS 8176.9 MB and VRAM 9598 MB. The [collection](../../../../artifacts/mdb-flux-joint-tail-20261105/run-01/collection.json) contains 791 files/7,492,981,559 bytes, with producer, remote-before, remote-after and local identities agreeing. The whole forecast sealed at 300.506088 worker seconds with all 28 selected-target counters zero; the first native reference exit followed at 302.934956s.

The [pre-outcome auditor](../../../../artifacts/mdb-flux-joint-tail-20261105/audit.json) passed unchanged on its first execution in 85.345s. Its independently implemented recurrence consumes its own reconstructed predecessors: 95,076,352 checked scalar outputs across 34 fields are bit-exact against the sealed mathematical forecast. Native CUDA remains numerically different:

| Next-step output | Image relative RMS discrepancy | Text relative RMS discrepancy |
|---|---:|---:|
| Projection |0.296034%|0.256406%|
| Joint0 |0.438789%|0.298407%|
| Joint1 |0.527006%|0.338674%|
| Joint2 |0.798186%|0.378262%|
| Joint3 |0.910906%|0.469138%|
| Joint4 |0.493905%|0.759781%|

The joint4 maximum coordinate differences are 4 image units and 1024 text units. The small relative RMS values describe averages in this tested state; they do not establish native bit equality or a pointwise error bound.

All six local image/text cross-cosines are negative, from −0.332196 to −0.254904. Five total norms are below their inherited norms. Joint4 image instead rises from 0.492483% inherited to 0.493905% total despite a negative cosine; its local term adds enough energy to outweigh the opposing cross term. All six local vector identities have zero measured residual. At the common boundary, the telescope contains both positive and negative cross-dots.

The decline in image percentage from joint3 to joint4 is caused by changing scale: absolute discrepancy RMS grows from 0.015602 to 0.032243 while native RMS grows from 1.712832 to 6.528175. Native text RMS grows from 26.300045 to 267.328369. [Direct full-array subtraction](../../../../artifacts/mdb-flux-joint-tail-20261105/native-update-scales.json) measures joint4 image/text output norms at 3.81×/10.16× their input norms. These whole-block observations neither separate attention from MLP nor establish the behavior across clocks. See [relative/error decomposition](../../../../artifacts/mdb-flux-joint-tail-20261105/state-errors.png) and [absolute scales](../../../../artifacts/mdb-flux-joint-tail-20261105/native-scale-changes.png).

| Future | Final latent relative RMS discrepancy | RGB channel MAE,0–255 | Maximum channel difference |
|---|---:|---:|---:|
| Native |0%|0|0|
| Sham |0%|0|0|
| `bridge.j2` |1.095850%|0.105118|10|
| `bridge.j3` |1.083765%|0.109830|9|
| `bridge.j4` |1.113446%|0.109229|11|
| `full.predicted` |1.115785%|0.113383|9|

The final latent and pixel rankings differ. Native and sham latent/PNG are exact. All [six actual images](../../../../artifacts/mdb-flux-joint-tail-20261105/native-futures.png) are retained. A separate [byte-level spotcheck](../../../../artifacts/mdb-flux-joint-tail-20261105/independent-postrun-spotcheck.json) passed on its first attempt, confirming complete installations, bridge inputs, adjacent non-stream fields/clocks and the receipt census. [Installed qualification](../../../../artifacts/mdb-flux-joint-tail-20261105/qualification-02.json) passed 228 tests. Scientific numerical magnitudes remain observations; terminal status is unassessed.

## Matched local comparisons

For k∈{2,3,4}, define N_k as the clean native block output, B_k as the unchanged native block applied to the complete predicted incoming state, and P_k as the sealed mathematical block applied to that same predicted state. Both streams and the time/rotary context are matched; the carried latent and clock agree. Then

\[
P_k-N_k=(B_k-N_k)+(P_k-B_k).
\]

The inherited term includes native transport of the upstream difference. The local term compares implementations on identical operands. Their signed cross-dot determines whether their squared energies reinforce or oppose:

\[
\|P_k-N_k\|^2=\|B_k-N_k\|^2+\|P_k-B_k\|^2+2\langle B_k-N_k,P_k-B_k\rangle.
\]

Use `native_normalized_relative_rms` to compare term magnitudes with the same N_k denominator. The general per-comparison metric for P_k−B_k instead uses B_k as its reference. These denominators are explicit in the audit.

Joint blocks perform attention and then an MLP on the receiving residual. Image and text attention mix both streams. Each stream's MLP consumes its updated residual; this differs from the parallel attention/MLP structure of the preceding single blocks. All five joint instances share a template and bind distinct learned parameter banks. The predictor and independently implemented reconstruction retain the source-qualified BF16 rounding boundaries.

## Six native futures

| Branch | Complete installed state | Bridge to step3 single0 | Native suffix |
|---|---|---|---|
| `native` | Clean native joint4 output | None |22 transitions + VAE|
| `sham` | Exact native closure | None |22 transitions + VAE|
| `bridge.j2` | Predicted joint1 |Native joints2–4|22 transitions + VAE|
| `bridge.j3` | Predicted joint2 |Native joints3–4|22 transitions + VAE|
| `bridge.j4` | Predicted joint3 |Native joint4|22 transitions + VAE|
| `full.predicted` | Predicted joint4 |None|22 transitions + VAE|

Every bridge input and output is retained in full. The first bridge operation supplies B_k. P_k is also installed in the next branch: P2 in `bridge.j3`, P3 in `bridge.j4`, and P4 in `full.predicted`. The audit verifies that B_k and installed P_k differ only in image/text at exactly the same clock. Their later native consumers are identical.

At the common single0 boundary, let X1, X2 and X3 denote the three bridge outputs. The ordered telescope is

\[
P-N=(X_1-N)+(X_2-X_1)+(X_3-X_2)+(P-X_3).
\]

The adjacent contrasts include subsequent native transport of local discrepancies. They provide an ordered decomposition rather than unique module or semantic attribution. The audit retains all six pairwise cross-dots and cosines.

One load, encoding and initial noise draw serve the entire lease. Its finite census is68 prefix +22 reference +6 bridge +6×22 suffix =228 native transitions,5 atomic Acts,233 receipt occurrences and6 actual VAE outputs. These branches share one history; they are not six seed replications.

Each branch has23 observation boundaries. Native trajectory arrays are retained in full. Intermediate candidates retain descriptors, worker-measured full comparisons and64 fixed coordinate witnesses. Complete common/final candidate cuts remain available for independent comparison. The independent auditor distinguishes those storage levels explicitly.

## Running the workflow

Build and install the public MDB package with `FluxJointTailClosureAdapter`, Saturn-pub0.4.1 and the source-qualified Diffusers0.39/Torch2.13 profile. Qualification runs outside the checkout against installed wheels. Real-model execution uses strict mrun admission, one finite lease and no automatic retries.

```bash
MDB_PY=/path/to/qualified/python
MDB_Q=/path/to/saturn-pub/packages/mdb/qualification
MDB_RUN=/path/to/new-run
"$MDB_PY" -I "$MDB_Q/submit_flux_joint_tail.py" "$MDB_RUN" \
  --prior-run /path/to/qualified-joint-chain-run
"$MDB_PY" -I "$MDB_Q/submit_flux_joint_tail.py" "$MDB_RUN" --submit-staged
```

Staging binds the old scientific helpers from the prior admitted payload, the new worker and adapter, installed package entries, input geometry, resource/transport bounds and independent auditor. Inspect the returned job using `mrun why JOB` and `mrun attach JOB`. After authenticated success:

```bash
"$MDB_PY" -I "$MDB_Q/collect_flux_joint_tail.py" "$MDB_RUN" JOB
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  "$MDB_PY" -I "$MDB_RUN/independent-math-freeze/analyze_flux_joint_tail.py" \
  "$MDB_RUN" --out /path/to/new-audit.json
"$MDB_PY" -I "$MDB_Q/publish_flux_joint_tail.py" \
  "$MDB_RUN" /path/to/new-audit.json /path/to/new-publication
```

The collector checks producer, remote-before, remote-after and local inventories within the declared12GiB total/1GiB member limits. The saved-only auditor independently reconstructs every prediction from its own reconstructed predecessor. It also verifies the prospective seal, module counters, typed cuts, atomic replacements, receipt paths, matched bridge operands and actual final outputs. Numerical magnitudes remain observations; mechanics and custody checks protect the instrument.

Direct navigation uses the same public debugger primitives:

```python
from saturn_pub import Session
from mdb.flux_joint_tail_closure import closure_act
from mdb.native_navigation import NativeCutNavigator

nav = NativeCutNavigator(Session.from_cut(adapter, joint3_cut), max_steps=24)
nav.apply(closure_act(joint3_cut, latent=predicted_z3,
    context=predicted_context, image=predicted_joint2_image,
    text=predicted_joint2_text, name="joint3.matched-incoming-state"))
receipt = nav.step()
native_joint3_on_predicted_state = nav.capture()
```

The operands must be complete, source-bound values converted into the qualified native layout without changing their sealed values. This excerpt illustrates navigation; the campaign implements the prospective transaction and evidence custody around it.

The new adapter admits live joint cursors2/3/4 and single cursor0 under its own execution identity. It replaces latent/context/image/text atomically and binds13 source and clock preconditions. Native advancement and both preceding adapter identities remain unchanged.

## Computation records and the symbol table

The [executed public publisher](../../../../artifacts/mdb-flux-joint-tail-20261105/equation-publication-v1/result.json) creates 12 equation/atlas observations: three blocks × two streams × clean-native or matched-bridge reference. They reuse six sealed output fields from one history. Each record binds its block, stream, reference kind/cut, input cut, source/model execution, prompt, seed, frontier and native continuation. Each of the twelve fully specified measured occurrence queries returns one record; changed-seed and missing-input queries return none. EquationCatalog reports observed−predicted while the auditor reports predicted−reference; [publication verification](../../../../artifacts/mdb-flux-joint-tail-20261105/publication-verification.json) confirms the opposing signed means and matching magnitudes/counts.

For a matched local record, the actual final candidate/reference pair is `bridge.j3`/`bridge.j2` for joint2, `bridge.j4`/`bridge.j3` for joint3, and `full.predicted`/`bridge.j4` for joint4. Those pairs carry the respective P_k and B_k through the same subsequent native consumers. Clean-native records use the same candidate future against `native`.

A computation record describes a law and its source bindings, required state, applicability, signed effects and continuation. A live state contains the tensors, carried dependencies and clock. A continuation names the next consumer. The symbol table connects these without replacing the live state with a record.

The generic `flux.joint.attention-then-mlp` name groups a recurring mathematical template. The exact occurrence conditions retain which weights and state were tested. Grouping records does not induce an applicability rule, an object dictionary or a numerical error bound for new contexts. Closing the remaining denoising suffix mathematically is the next extension toward a final-latent forecast from the observed prefix.

The [explicit three-catalog federation](../../../../artifacts/mdb-flux-joint-tail-20261105/cross-run-atlas-v1/result.json) holds 26 observations from three native histories and 14 complete applicability contexts. Eighteen observations use the joint law, six the projection law and two composed own-state views. All source observation fingerprints remain unchanged; the framework-free federation imported no Torch and induced no families.

```python
from mdb.computation_atlas import ComputationAtlas

atlas = ComputationAtlas("/path/to/cross-run-atlas-v1/atlas")
assert len(atlas.find()) == 26
joint = atlas.find(law_id="flux.joint.attention-then-mlp")
assert len(joint) == 18
current = [row for row in joint if row["equation"]["tested_context"]["seed"] == 103]
assert len(current) == 12
context = current[0]["equation"]["tested_context"]
assert len(atlas.find(applicable_to=context)) == 1
assert atlas.find(applicable_to={**context, "seed": -1}) == []
```

The standard [Saturn evidence sweep](../../../../artifacts/mdb-flux-joint-tail-20261105/evidence-v1/run.json) ran and verified over 300 event records and scanned 423 JSON files, but recognized zero public typed claims/address references. That is an instrument coverage gap, separate from the successful equation publication and atlas. The existing [prepared-weight request](../../../../artifacts/mdb-flux-joint-tail-20261105/request-prepared-weights-v3.json) now includes the measured 7.49 GB evidence transport cost; no speedup was measured.
