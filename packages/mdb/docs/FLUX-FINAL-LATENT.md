# Predicting a final FLUX latent from an observed prefix

This campaign extends the [joint-tail forecast](FLUX-JOINT-TAIL.md) through the remaining twenty single blocks, readout and Euler commit. The mathematical program predicts the final latent before the selected native targets execute. The native VAE consumes that predicted latent; the forecast begins at an observed internal state.

The specimen is FLUX.2 Klein 4B distilled, CUDA BF16, batch size 1, four denoising steps and 512×512 output. Fresh seed 107 uses `A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph.` A clean prefix of 68 native transitions reaches logical step 2 immediately before single block 6.

## Complete remaining denoiser prediction

The forecast seals 44 transitions: fourteen complete single blocks, readout, Euler commit to Z3, next-step projection, five joint blocks, twenty final single blocks, readout and Euler commit to Z4. Each stage consumes its own complete predicted predecessor. Both working streams, regenerated time/rotary context and predicted Z3 remain source-bound. The final readout uses the predicted context and image stream; the last Euler step uses predicted Z3 and the supplied σ3→σ4=0 schedule.

Single blocks 6–19 occur at two clocks. Their learned parameters are archived once and reused; the working states and modulation context differ. The six additional parameter banks 0–5 complete the twenty-bank inventory. Reusing constants supplies no native future activation.

The native final commit clears context, image, text and noise prediction. A terminal replacement therefore installs predicted Z4 with the other three writable closure fields explicitly `None`; noise remains `None`. The new adapter preserves those semantics rather than carrying stale working arrays into the VAE boundary.

## Four matched futures

| Branch | State admission | Native denoising after admission | Consumer |
|---|---|---|---|
| `native` | Ordinary final native state |None|Native VAE|
| `sham` | Exact terminal native closure |None|Native VAE|
| `bridge.predicted` | Complete predicted Z3/context/joint4 image/text at final-step single0 |22 transitions|Native VAE|
| `full.predicted` | Sealed predicted Z4 with terminal empty fields |None|Native VAE|

Native reference execution retains all 44 transitions. The bridge retains complete input/output cuts and arrays for its 22 transitions. Both trajectories have 23 observations from single0 to terminal. The mathematical intermediate states are sealed arrays, with no invented native traversal for the full predicted branch.

Let S be the clean state before the final suffix and Ŝ its complete own-state prediction. Native suffix F and mathematical suffix F̂ include twenty single blocks, readout and Euler commit. Define

\[
N=F(S),\qquad B=F(\widehat S),\qquad P=\widehat F(\widehat S).
\]

Then

\[
P-N=(B-N)+(P-B),\qquad
\|P-N\|^2=\|B-N\|^2+\|P-B\|^2+2\langle B-N,P-B\rangle.
\]

The inherited term is the upstream discrepancy after native suffix transport. `P−B` accumulates discrepancies across the whole new suffix. It does not isolate individual blocks. Per-boundary image/text decompositions retain both absolute RMS/L2 and shared-native-normalized terms; readout noise and committed latent use their own field lifetimes and denominators.

The RGB decomposition compares actual VAE outputs. Its `P−B` term is the common native VAE's response to the final-latent discrepancy. It does not describe independently forecast pixels or a VAE kernel error.

One lease/load/encoding/noise draw serves 68 prefix +44 reference +22 bridge =134 native transitions, three atomic Acts, 137 receipts and four VAE calls. Direct native candidate calls remain zero. All 44 predicted transitions seal before the selected post-prefix target entries.

## MDB state admission

`FluxFinalLatentClosureAdapter` has a distinct execution identity. It admits complete four-port replacements only at single0 of the final live step or at terminal project/cursor0. It binds all thirteen source/clock fields, preserves undeclared fields, and inherits native advancement unchanged. Native stepping still rejects completed trajectories.

```python
from saturn_pub import Session
from mdb.flux_final_latent_closure import closure_act
from mdb.native_navigation import NativeCutNavigator

nav = NativeCutNavigator(Session.from_cut(adapter, terminal_cut), max_steps=1)
nav.apply(closure_act(terminal_cut, latent=predicted_z4,
    context=None, image=None, text=None, name="forecast.final-latent"))
predicted_image = adapter.decode(nav.session, height=512, width=512)
```

The predicted value must already have the qualified layout and exactly preserve its sealed BF16 values. The campaign validates that conversion. Its native terminal cut supplies an admission scaffold after sealing; the extra independent spotcheck derives the entire predicted terminal payload from the observed prefix, sealed Z4 and declared clock/reset contract. This experiment does not implement direct serving from the prefix without a native reference scaffold.

## Reproducing the run

Use an installed public MDB wheel containing the new adapter, public Saturn-pub 0.4.1 and the qualified Diffusers 0.39/Torch 2.13 profile. Stage from the preceding accepted joint-tail payload, preserving its scientific helper bytes:

```bash
MDB_PY=/path/to/qualified/python
MDB_Q=/path/to/saturn-pub/packages/mdb/qualification
MDB_RUN=/path/to/new-run
"$MDB_PY" -I "$MDB_Q/submit_flux_final_latent.py" "$MDB_RUN" \
  --prior-run /path/to/accepted-joint-tail-run
"$MDB_PY" -I "$MDB_Q/submit_flux_final_latent.py" "$MDB_RUN" --submit-staged
```

The launcher uses strict mrun admission, a soft Beast preference, CUDA resource guards and no automatic retries. The finite profile declares four CPU threads, a measured 12288 MB RAM/13000 MB VRAM reservation, 14 GB disk, 780 expected seconds, a 1050-second worker budget and 1200-second outer timeout. The transaction cap and single-member cap are each 1 GiB; total collected output is bounded at 12 GiB. Forty full prediction registers have a 754,979,840-byte NPY lower bound, with a declared framing allowance below the transaction cap.

Inspect the job with `mrun why JOB` and `mrun attach JOB`. After authenticated success:

```bash
"$MDB_PY" -I "$MDB_Q/collect_flux_final_latent.py" "$MDB_RUN" JOB
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  "$MDB_PY" -I "$MDB_RUN/independent-math-freeze/analyze_flux_final_latent.py" \
  "$MDB_RUN" --out /path/to/new-audit.json
"$MDB_PY" -I "$MDB_Q/publish_flux_final_latent.py" \
  "$MDB_RUN" /path/to/new-audit.json /path/to/new-publication
```

The collector binds producer, before/after remote and local inventories to the actual succeeded scheduler job. The frozen saved-only auditor independently reconstructs every stage from its own reconstructed predecessor. It checks source/ABI identity, guards/seal, complete typed closure, reference and bridge paths, terminal reset, native images and signed numerical measurements. Accuracy observations remain separate from custody and instrument mechanics.

The [worker](../qualification/flux_final_latent_campaign.py), [auditor](../qualification/analyze_flux_final_latent.py), [adapter](../src/mdb/flux_final_latent_closure.py) and [publisher](../qualification/publish_flux_final_latent.py) implement the workflow.

## Computation records

The publisher records final noise and final Z4 against clean native and matched native suffix references: four comparisons over two sealed fields from one history. The generic law `flux.denoising.own-state-suffix` names a composition. Each occurrence binds output role, reference kind/cut, input cut, model/source/execution, prompt, seed, frontier and horizon.

All four records' candidate future is `full.predicted`, with native VAE rendering and zero denoiser calls after installation. References are `native` or `bridge.predicted`. The noise record's future belongs to the complete final-latent program; it is not an isolated noise replacement experiment. Exact applicability queries distinguish these occurrences. Catalog federation preserves conditions and counterexamples without inducing a semantic family or a new-context error bound.

The four-catalog federation contains 30 observations from four native histories and 18 complete applicability contexts: 18 joint, six project, two prior reentry and four new suffix comparisons. This installed, framework-free query returns the four new occurrences, then selects one by its complete tested context:

```python
from mdb.computation_atlas import ComputationAtlas

atlas = ComputationAtlas("/path/to/cross-run-atlas-v1/atlas")
suffixes = atlas.find(law_id="flux.denoising.own-state-suffix")
assert len(suffixes) == 4
context = suffixes[0]["equation"]["tested_context"]
assert len(atlas.find(applicable_to=context)) == 1
assert atlas.find(applicable_to={**context, "seed": -1}) == []
```

Changing the seed produces no applicable stored record; it does not predict failure of the mathematical law. The catalog currently retrieves exact observed conditions and abstains outside them. The new context's numerical discrepancy must be measured prospectively.

## Measured seed107 execution

Job `job-3a8a9795960a` succeeded under request `458ec595757090a0869edb544948366e6cb69f88d2e28985eebf3ce48941dc03`. Full forecast seal `d0f5adb2c0c1f1ad3ae0c622ef7b509acb0d5db499c6f2f3ebdd038fb2980573` precedes all 44 native reference transitions; all 34 selected-target counters were zero then. Earlier prefix occurrences remain declared.

| Branch | Final latent relative RMS | RGB channel MAE, 0–255 | RGB RMS | Maximum channel difference |
|---|---:|---:|---:|---:|
| `native` |0|0|0|0|
| `sham` |0|0|0|0|
| `bridge.predicted` |1.061867%|0.100101|0.334544|6|
| `full.predicted` |1.047002%|0.099817|0.336575|6|

The full prediction has 88,031 nonidentical latent coordinates out of 131,072; its largest absolute latent difference is 0.169921875. Its rendered PNG differs in 73,964 of 786,432 channel values. Native and sham latent/PNG controls are exact. Averages include the white background and do not bound every coordinate.

Final latent inherited/additional/total shared-native-normalized RMS values are 1.061867% /0.697197% /1.047002%, cosine −0.349461. Their absolute RMS values are 0.008360 /0.005489 /0.008243. Full prediction slightly improves latent RMS and RGB MAE versus the native bridge, while RGB RMS increases. Negative alignment is not semantic repair and need not reduce the total norm: after single9, image total 0.762420% exceeds inherited 0.759239% despite cosine −0.394950.

| Working boundary | Image relative RMS | Image absolute RMS | Text relative RMS | Text absolute RMS |
|---|---:|---:|---:|---:|
| Before single0 |0.491974%|0.031773|0.934045%|2.471992|
| After single4 |0.757802%|0.096453|0.918261%|2.452083|
| After single9 |0.762420%|0.095015|0.972784%|2.633753|
| After single14 |1.453820%|0.205674|1.070770%|2.898728|
| After single19 |1.245970%|0.548065|1.193060%|3.267851|

The final image/text maximum coordinate differences are 18/1536 in their own register units. Readout noise total relative RMS is 0.906603%. The stored suffix curve separates each field's lifetime and denominator; the image percentage drop after single14 coexists with increasing absolute error.

The unchanged frozen primary audit passed on its first execution in 182.426 seconds. Its 56 field comparisons independently reconstruct 189,710,336 output scalars exactly to the sealed mathematical forecast; native CUDA disagreement remains separate. An auxiliary reader's initial projection-address `index=0` assumption failed; public projection addresses use `index=None`, while payload cursor remains 0. The preserved v2 failure and versioned v3 metadata correction leave producer, forecast and primary auditor unchanged. V3 derives all thirteen terminal payload fields from the observed prefix, sealed Z4 and declared reset/clock contract, and verifies exact installation and matched paths.

Qualification passed 316 installed-wheel checks across regression and publisher cohorts. Real execution took 740.32 seconds, peak RSS 8762.2 MB/VRAM 9772 MB. Authenticated collection retains 892 files/9,765,539,505 bytes with equal remote-before/remote-after/local inventories. Of that, 169 distinct explicitly named parameter archives occupy 7,751,110,784 bytes. The deduplicated forecast arrays require 807,150,984 NPY bytes, within the predeclared 1 GiB transaction cap. These costs motivate prepared bank reuse; no speedup was measured.

The workspace evidence root is `artifacts/mdb-flux-final-latent-20261105`, containing `audit.json`, `qualification-summary-v1.json`, `run-01/collection.json`, `independent-postrun-spotcheck-v3.json`, `equation-publication-v1/` and `cross-run-atlas-v1/`. The experiment is mechanically valid with observed numerical trends; terminal status remains unassessed. This single history does not establish arbitrary-prompt coverage or an error guarantee for fresh contexts.
