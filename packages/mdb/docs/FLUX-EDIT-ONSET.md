# Earlier edit onset and repeated predicted text reads

Completed job `job-13cddbd2f9ab` extends the [frozen-composition test](FLUX-FROZEN-COMPOSITION.md) by changing edit onset rather than prompts or rules. It produces25new native futures, reuses8historical step3 images explicitly, and tests the same predicted initial text state across repeated native continuations.

## What changed

The seed60666 portrait, seven conditioning states, additive rule and old interaction doses remain unchanged. One native base replay captures legal project-entry frontiers at steps0–4. At onsets0,1,2, a typed conditioning Act selects base, color, shape, joint, additive, plus-old or minus-old E. It remains active for every subsequent step. Each onset also receives an additive branch whose text is replaced after every remaining native projection with the previously sealed predictedT.

Onset3 images come from the original assay, with hashes/custody checked independently. They are not new executions or independent examples. Hypotheses about earlier shape authority and repeated numerical accuracy seal before any new native project entry. All source encodings and the old predictedT are already observed or calculated; this is a new-clock/future test on a known prompt.

The native computation separates a repeated program read from its dynamic use:

```text
T_k = context_embedder(E)
I_k = x_embedder(Z_k)
Z_(k+1) = scheduler(Z_k, denoiser(I_k, T_k, time_k))
```

The text projection itself reads neither Z nor time. All18captured additive native text returns—nine in native-additive branches and nine in repeated-prediction branches—equal the earlier measured additive field exactly. Its predicted reference retains0.256452%native-relative RMS error. Later attention/MLP computation, readout and scheduler remain native.

## Results

Native shape-only and joint controls produce visibly narrower/almond-like eyelids when introduced at step0, with weaker narrowing at step1. Step2/3 introductions leave much rounder eyes. Color-only controls retain rounder geometry while changing irises to violet. Additive composition also produces violet eyes and earlier narrowing. These are unblinded visual observations, not calibrated geometry scores.

| Edit onset | Remaining predicted text reads | Predicted versus measured additive future RGB MAE | Maximum pixel difference | Additive eye MAE to matched joint |
|---|---:|---:|---:|---:|
|0|4|0.482920|122|5.296933|
|1|3|0.361360|94|3.226945|
|2|2|0.289192|140|2.536570|
|3, historical|1|0.196051|44|2.036776|

All errors use0–255RGB. Repeated predicted text satisfies the prospective average-error forecast below1level at every horizon, while large local errors remain. More accurate pixel matching at a late onset does not mean a better shape edit: its matched native joint reference itself retains rounder eyes.

Earlier onset increases collateral in these examples. Additive clothing MAE versus base is10.0095,5.0055,3.3978,1.8308 at onsets0,1,2,3. Coarse black hair, teal shirt, ivory background and frontal composition persist, but contours, collar details and shading change. The positive old interaction worsens expanded-eye error at onsets0,1,2,3, although onset1 is a near tie3.2312versus3.2269. The negative old interaction slightly improves that error at onset0,5.2935versus5.2969, while worsening whole-image matching. This retained near miss does not establish a portable correction.

Onset changes source exposure, image history and remaining time together. The result does not isolate the time embedding, identify a particular writer, establish necessary/minimal support, or generalize to other prompts or relations. Earlier scene-shape timing work supplied the lead; this qualifies the new portrait and composed edit.

## Execution and audit

Use installed public MDB/Saturn-pub/mrun packages. Explicit qualification helpers use `runpy`; no private runtime imports or checkout path changes are required.

```sh
# Keep the same qualified worker environment at stage and submit.
export MDB_WORKER_UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache
export MDB_WORKER_TORCH_HOME=/mnt/hdd8tb/mdb/torch
export MDB_WORKER_MODEL_ROOTS=/mnt/big/llm-models
python -I qualification/submit_flux_edit_onset.py NEW_RUN \
  --source WORKSPACE/artifacts/mdb-frozen-composition-20261104
python -I qualification/submit_flux_edit_onset.py NEW_RUN --submit
python -I qualification/submit_flux_edit_onset.py NEW_RUN --collect JOB_ID
python -I qualification/analyze_flux_edit_onset.py RUN NEW_AUDIT.json \
  --source WORKSPACE/artifacts/mdb-frozen-composition-20261104
python -I qualification/publish_flux_edit_onset.py RUN AUDIT.json NEW_ATLAS
```

The measured Beast lane uses one finite mrun CUDA BF16 batch1 lease, strict preflight,10GBRAM/13GBVRAM and no retries. Earlier host-local CAS and arrays are size/hash-bound. A tiny native FLUX check qualifies repeated project reads and exact same-text Act closure before the real run. A census check and receipt-collision regression bring the focused suite to3passing tests.

The successful run uses one load, zero encodes/noise draws,2128native transitions,33Acts,2161receipt emissions and25VAEs. Worker time131.527s, runner173.93s; peakRSS5729.9MB/VRAM9298MB. Custody retains186,998,285archive bytes,520files and23parts. Baseline replay reproduces the historical step3 payload and PNG exactly; all three same-E shams match baseline latent/RGB exactly.

Sharing base prefixes saves672adapter transitions relative to the2800needed by the same25branch recipes restarted independently. This is a derived operation-count comparison, not a measured runtime speedup. The native text projector still executes before predictedT is installed for validation; prediction has not yet eliminated that computation.

The auditor verifies101per-step latent observations against retained typed cuts, all receipt/Act paths and full lossless text-return arrays. Twelve of18text returns also join retained post-project typed cuts; the other six remain full native array observations with descriptors and receipts. Complete native candidate histories at every transformer block are not retained. Three identical canonical project receipts recur across physical attempts:2161emissions have2158distinct IDs. They are counted as physical work; canonical identity is not an execution counter. Initial offline audit assumptions about uniqueness and typed-point decoding were repaired without inference reruns.

Nine public atlas occurrences bind onset, clock, receiving state, equation, shared consumer future, collateral and limitations. They share three repeated-prediction futures, not nine independent image predictions. Nine exact-context queries match once;18changed-clock/receiving-state queries match zero. Standard offline evidence run/verify/query returns0with51numeric rows and9qualified addresses.

Evidence: [audit](../../../../artifacts/mdb-edit-onset-20261104/analysis-v1.json), [visual review](../../../../artifacts/mdb-edit-onset-20261104/visual-review.json), [public atlas](../../../../artifacts/mdb-edit-onset-20261104/public-atlas-v1/provenance.json), [offline evidence](../../../../artifacts/mdb-edit-onset-20261104/evidence-v1/provenance.json), [worker](../qualification/flux_edit_onset_campaign.py), [onset/repeated-read helper](../qualification/flux_edit_onset.py).

The next cut can hold the base step1 latent and clock fixed, vary only the shape/joint text program, and trace attention/MLP writes into working image and the next committed latent. That separates conversion of the program into geometry from the already qualified source projection.

The [post-outcome trace diagnostic](../../../../artifacts/mdb-edit-onset-20261104/trajectory-diagnostic-v1.json) additionally retains four-read latent error0→0.5638→0.9773→1.7952→10.4494. The last interval has the largest growth, but block/readout/scheduler responsibility remains unresolved. Its measured-state decomposition `Zj = (Zc+Zs−Zb) + C + R`, with `C=Zadd−(Zc+Zs−Zb)` and `R=Zj−Zadd`, closes exactly at every retained committed clock. These are diagnostic terms, not separately installed corrections or independent forecasts; the analysis uses zero further model calls.
