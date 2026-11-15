# Jacobian reader/writer validation

October 5, 2026. The canonical MDB reader/writer and Qwen Bench adapter pass bounded software mechanics checks. The small specimens are seeded random two-layer, width-16, vocabulary-32 Qwen2 models on CPU BF16, with both an identity transport and a separately fitted non-identity transport. This record establishes runtime and custody behavior; these fixtures supply no pretrained semantic lens or useful memory-writer claim.

The implemented surface is documented in [the API guide](../../docs/JACOBIAN-WORKSPACE.md). Standalone source and lane validation lives in [jacobian_io.py](../../src/mdb/jacobian_io.py); native admission and sidecar publication live in [bench_jacobian.py](../../src/mdb/bench_jacobian.py). Existing serialized native schemas and Act IDs keep their meaning. Each candidate sidecar has a separate attempt identity, allowing repeated edits of the same parent without conflating commit and rejection records.

## Focused and regression tests

- 38 focused tests passed: 15 safe reader/writer tests, six workspace-algebra tests, nine native Bench tests, seven offline verifier tests, and one fitted-worker regression.
- 45 existing Qwen Bench and publication tests passed in the same isolated MDB test process as the earlier focused suite. That combined run had 81 passes before the final additional parameter-rebinding test; the final 37-test focused run includes that correction.
- 29 existing hidden-memory and circuit-measurement tests passed; nine family-specific cases were skipped by their applicability conditions.
- Ruff passes on every new module, focused test, and installed-wheel qualifier. Repository diff checks are clean.

Tests run through the shared `common/bin/common exec` environment with canonical MDB on `PYTHONPATH`. The package's `tests/conftest.py` blocks upstream Saturn, Saturn-pub, and common-harness imports. Coverage includes native identity-lens read parity, signed steering, selected-span edits, BF16 post-cast identities, exact zero-dose behavior, source drift, corruption, module parameter mutation/rebinding, safe artifact reopening, native commit/abort, pending-export refusal, and journal failure rollback. The fitted-worker regression performs actual frozen-model autograd on the tiny fixture, runs all ten arms and both K/V probes, and verifies the export. It also reproduces transient fit-tokenizer truncation and checks that the restored serving tokenizer identity stays fixed throughout the assay. The offline verifier independently checks receipt seals, native install/reject contracts, unique per-attempt event linkage, committed hidden bytes, and readout geometry.

```sh
rtk proxy env PYTHONPATH=/Users/jakeholl/domains/saturn-pub/packages/mdb/src \
  common/bin/common exec python -m pytest -q \
  saturn-pub/packages/mdb/tests/test_jacobian_io.py \
  saturn-pub/packages/mdb/tests/test_jacobian_workspace.py \
  saturn-pub/packages/mdb/tests/test_bench_jacobian.py \
  saturn-pub/packages/mdb/tests/test_bench_jacobian_verify.py \
  saturn-pub/packages/mdb/tests/test_jacobian_real_worker.py
```

## Installed wheel and fresh-process reopen

The canonical Hatchling wheel was installed without dependencies into a fresh target outside the workspace. Separate isolated capture and reopen processes used only that installed MDB plus the existing model libraries. They blocked imports of `saturn`, `saturn_pub`, `common`, `common_harness`, and `mrun`, and verified the imported MDB path belonged to the installed target.

Final wheel SHA-256: `1120c5299ffe096a76db84ceaa71cd44348db8f03a46e6bf17e72e433b96ea2e`.

The [final qualification](installed-wheel/final-qualified-1120c5299ffe/qualification.json) reports valid mechanics, distinct process IDs, exact native source restoration, exact zero-span commit, stale observation refusal, and valid closure over five Jacobian artifacts and seven retained cuts. Safe lens, steering-candidate, and span-candidate save/open/validate also passed. The [two-process qualifier](../../tests/qualify_installed_jacobian.py) reproduces this check; [handles](installed-wheel/final-qualified-1120c5299ffe/handles.json) and the retained export preserve the source and candidate references. The earlier [preliminary check](installed-wheel/preliminary-21501f32f99b/qualification.json) is retained under its different wheel identity.

## Unrelated environment limitation

Seven existing `test_distribution.py` launcher/staging cases failed because the shared environment contains the private `mrun` distribution. MDB's existing `runtime_distributions()` deliberately refuses that conflict and requires `mrun-pub`. The failing cases concern four staged-target checks, two worker-command checks, and worker bootstrap; none executes the new Jacobian reader/writer. Offline import and deterministic-wheel tests passed. No environment packages or existing launcher guards were changed. The separately installed reader/writer qualification needs no scheduler import and passes with `mrun` blocked.

## Scientific scope

The earlier Qwen2.5-1.5B experiment uses a real fitted lens sealed for `cuda-float32-eager-tf32off`. Current Qwen Bench executes BF16 and rejects that artifact. The [Obsidian post](../../../../../obsidian/blog/2026-10-05-203000-j-space-readers-writers-and-memory.md) explains its exploratory read/write mismatch: the complement of two selected J atoms preserves more than 99% of delayed donor behavior. The new [BF16 assay](README.md) fits an independent compatible transport. Broader dictionary support, delayed-query conditioning, and consumer/role panels remain separate scientific work.

## Real-model BF16 attempts

The frozen Qwen2.5-1.5B specimen uses CUDA BF16 eager execution with TF32 disabled, source block 11 / completed boundary 12, and target block 27 before final normalization. The transport is fit from three declared independent prose texts using 576 batched backward invocations. No parameter receives an update. The worker loads the qualified wheel above, checks its installed import root, and blocks upstream runtime imports.

Historical outcomes remain unchanged:

| Job | Outcome | Evidence retained |
| --- | --- | --- |
| [job-c25c2663c1ab](results/job-c25c2663c1ab/job-record.json) | VRAM guard stopped the job after fitting | Plan 3,474 MiB; guard ceiling 3,821.4 MiB; observed peak 4,888 MiB. No consumer rows. |
| [job-2cf2d1202511](results/job-2cf2d1202511/job-record.json) | Tokenizer identity mismatch before the first read | Fit left temporary truncation state; assay encoding changed it. Strict reader refusal exposed the harness bug. |
| [job-4ba3cc18d86a](results/job-4ba3cc18d86a/job-record.json) | 900-second timeout during finalization | All 20 arms and four memory probes returned. No final report or bundle was emitted. |

The second and third jobs use the first job's retained measured resource exception: 6,110 MiB VRAM, with its censored-peak limitation recorded. The third additionally records the worker-only tokenizer repair and unchanged model/protocol/wheel/workload geometry. Automatic retries remain disabled. The [third job log](results/job-4ba3cc18d86a/job.log) contains signed delayed effects and answer outcomes; its original collection remains `complete: false`. Any artifact recovery and independent audit are separate evidence records and do not turn the timed-out job into a successful run.

## Recovered real-model export

The separate model-free CPU recovery [job-d47b087e4124](results/recovery-job-4ba3cc18d86a/job-d47b087e4124/collection-receipt.json) succeeded in 22.01 seconds, with peak RSS 89.7 MiB and no GPU use. It read the retained original workdir without changing it, checked the original worker/protocol/wheel identities, and transported 22,458 inventoried lens/export files. Collection independently checked every file's size and SHA-256, archive identity, protocol, and origin/recovery request joins. The original report was absent; it was not recreated.

An isolated process outside the workspace audited a copy through the independently installed qualified MDB wheel, with upstream imports blocked and no model load. Its [audit receipt](results/recovered-audit-job-4ba3cc18d86a-v2/receipt.json) reports valid closure over 621 cuts, 14 Jacobian artifacts, four memory probes, eight memory writes, 304 VM bundles, and 2,224 JSON artifacts. All 648 payload checksum/geometry checks pass with zero failures or misses. The [runtime identity record](results/recovered-audit-job-4ba3cc18d86a-v2/runtime-identity.json) also checks all 125 installed wheel members, excluding only the installer-modified distribution RECORD; no member differs.

The recovered export contains one committed `export-reopen-native-recall` zero-span candidate and a subsequent close cut. The model-free audit verifies their native linkage; replay equality requires the saved native futures or the worker's execution checks, separately from byte closure. The first local audit passed canonical verification but failed in its optional norm reader because it used the mutable hydration method on a read-only payload store. The corrected audit uses `PayloadStore.audit()` and passed in a separate output directory. No model or scientific panel was rerun.

A separate [read-only future comparison](results/recovered-audit-job-4ba3cc18d86a-v2/future-parity.json) matches 36/36 native/zero pairs: twelve cat in-run cuts, twelve bear in-run cuts, and twelve reopened bear cuts. Token history, pending token, logits payload hash, VM checkpoint reference, model identity, lane, and numerical contract agree. The two bear attempts occupy distinct journal publication segments even though their deterministic endpoint CAS references are identical. The selected close cut preserves the reopened endpoint state. The [pure-stdlib verifier](verify_recovered_futures.py) reproduces this comparison without opening a model or runtime. This qualifies exact source-bound no-op execution in the named native BF16 lane. The original worker reached its equality assertion and close before the timeout, but did not publish its final report; the independent audit supplies the final closure check separately.

The four retained late-K/V probes use layers 22–27, animal row 21:22, keys and values, a fixed supplied query, and two native continuation tokens. J-plus transfer effects are 0 and +0.0625 with recipient answers preserved; full-donor transfer effects are +6.6875 and +10.4375 with fox/lion answers. All four original-store reinstalls have zero effect and exact first logits, answer tokens, endpoint closure, selected-parent restoration, and sibling selections. The [recovered analysis](results/recovered-audit-job-4ba3cc18d86a-v2/recovered-analysis.json) retains these signed observations and post-cast norms. Controls match or exceed the small J effects; this is a bounded memory-carrier observation, not a useful or general semantic-writer qualification.

Actual BF16 edit amplitude is measured as `candidate.float() - source.float()`. The recipe's `algebra.actual_delta_norm` describes work-dtype algebra before the native cast, and the stored BF16 delta rounds the subtraction again. The retained candidates and source bytes are authoritative for post-cast amplitude; those other numbers should not silently substitute for it.

The separate [bounded native qualification](results/recovered-audit-job-4ba3cc18d86a-v2/qualification.json) joins the runtime identity, byte audit, future parity, probe analysis, executed source, and original/recovery jobs. Its five terminal checks certify software mechanics only. The source-bound reader's normalized-head parity and exact preview abort were checked inside the sealed real-model worker; the model-free audit validates custody and native linkage rather than rerunning the output head. General semantic writing, sparse interpretation, learned spectral bindings, compact memory replacement, and cross-lane portability remain unqualified.

The post-job Saturn evidence sweep and read-only verify pass in receipts-only scope when invoked with absolute paths. Its legacy claims resolver leaves five parent-directory dependencies unresolvable; independently resolving all ten qualification dependencies relative to the report verifies every SHA-256. There are no applicable watch streams or model-bound citation claims. Initial format and relative-path invocation findings remain recorded in the [instrument note](index/evidence-instrument-note.json), separate from the valid MDB audit.
