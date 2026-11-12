# WSP Qwen3.5 repair case study

This investigation connects the published Qwen3.5-0.8B edit to actual parameter bytes, local write forecasts and native own-state continuations. It begins with a checkpoint audit, qualifies whole-projection substitution and restoration on short trajectories, and then examines fresh exploratory contexts with longer generation budgets. It is separate from reproducing the published MATH-500 score.

The public source is [Sparks of Weight Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). Sources are pinned to the original Qwen checkpoint `2fc06364715b967f1860aea9cf38778875588b17`, the edited release `d1671a178705bcab7644806d23c535517f35723b`, and the original logs `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. `launch.py fetch` downloads the small sparse-edit artifact and source metadata. Original pretrained weights remain in mrun's selected fleet store.

The implementation uses installed public MDB, Saturn-pub and mrun-pub wheels. Runtime workers import no private Saturn or common modules. The local controller runs through the workspace common launcher for configured scheduler access. Every scientific worker has one guarded finite mrun lease, strict preflight, explicit CPU/CUDA needs, no retry and direct collection. Independent saved-only analysis uses its own CPU lease.

## Stages and artifacts

| Stage | Purpose | Evidence |
|---|---|---|
| Exact source audit | Hash the complete original checkpoint and archive header, reconstruct two sparse BF16 edits, verify published target hashes, measure actual support and singular values | [audit-02](../../../../../artifacts/mdb-wsp-case-study-20261110/audit-02/extracted/wsp-result/summary.json) |
| Short native qualification | Three discovery prompts, four arms, at most 64 greedy own-token choices, six sealed local equation forecasts | [smoke-03](../../../../../artifacts/mdb-wsp-case-study-20261110/smoke-03/extracted/wsp-result/summary.json) |
| Independent saved reader | Verify source/support and independently compare the saved first logits and own-token sequences | [saved-audit-02](../../../../../artifacts/mdb-wsp-case-study-20261110/saved-audit-02/extracted/wsp-result/independent-audit.json) |
| Longer exploratory panel | Three newly authored contexts, four arms, up to 2,048 greedy choices | [discovery-01 results](../../../../../artifacts/mdb-wsp-case-study-20261110/discovery-01/extracted/wsp-result/summary.json) |
| Independent longer-panel reader | Recompute all six 2,048-token comparisons without loading a model | [saved-audit-greedy-01](../../../../../artifacts/mdb-wsp-case-study-20261110/saved-audit-greedy-01/extracted/wsp-result/independent-audit.json) |
| Sampling-policy follow-up | Same exploratory contexts, paired seeds 0/1/2, publication sampling parameters, different backend and 2,048 cap | [recovered observations](../../../../../artifacts/mdb-wsp-case-study-20261110/recovery-think35-01/extracted/wsp-result/native-brief.json) |
| Independent sampled-panel reader | Verify recovered custody and independently recompute all six sequence comparisons and six endpoint/edit-response errors | [saved-audit-think35-02](../../../../../artifacts/mdb-wsp-case-study-20261110/saved-audit-think35-02/extracted/wsp-result/independent-audit.json) |

`retrieve_weights.py` is standalone and is copied into every model-backed result. Its `parameter-artifacts.json` binds required named tensors to exact original file bytes, archive headers, offsets, shapes, dtypes and hashes. The result retains derived changes, activations, predictions, candidate patches and native outcomes. No original pretrained tensor bank is exported into persistent run evidence.

## What the initial stages establish

The exact audit runs as `job-e0a441ea99d3`. The original file matches its pinned HF SHA-256. Both reconstructed edited tensors match the published edited hashes exactly. Layer 15 changes 65,260 entries; layer 19 changes 64,928. Their sum is 130,188 actually changed BF16 values. The public manifest explicitly distinguishes bit changes from the 131,072 addressed positions.

The full checkpoint contains 873,438,784 serialized elements including its vision components. Under that explicitly named denominator, 0.0149052% of values change. This differs from using a nominal 0.8-billion denominator. The audit reconstructs the pinned sparse edit and checks the two published edited hashes; it does not independently download and compare all unchanged tensors in the edited release.

Each actual rounded delta has support on 64 input columns. The best rank-one residual is 0.233668% and 0.293570% of its Frobenius norm for layers 15 and 19. These are descriptive approximations, not rank gates. The independent reader finds an absolute cosine of 0.849005 between the leading output directions. Treat each layer's writer separately when constructing a lower-rank response program.

The short native job `job-99e85a5de24f` uses CUDA BF16 eager Transformers 5.14.1 with Torch 2.13.0. The four arms are original parameters, permanently edited parameters, original parameters with both edited down-projection returns substituted, and edited parameters with both original returns restored. Whole-projection substitutes recompute `F.linear` from the branch's actual current input; they pay that additional original dense arithmetic. This is not a cheap rank-one implementation.

Every arm starts at its own raw prompt with no retained cache. The HF wrapper's mutable RoPE deltas are cleared. Its returned hybrid attention, convolution and recurrent cache is carried through each branch's own generated tokens. The final emitted token remains unconsumed. No durable StateCut export, general public Qwen3.5 debugger adapter or fresh-process hybrid replay is claimed by this worker.

On all three short prompts, substitution matches the edited model and restoration matches the original. All six compared full first-logit vectors are exact, and all six 64-token sequence pairs agree. All twelve trajectories reach the 64-token cap. These are numerical/continuation qualification results rather than demonstrated completed loop repairs. `job-ff7417816e4b` independently recomputes the saved comparisons with zero model loads or forwards.

The first six local FP64 endpoint forecasts have relative RMS differences of approximately 0.158–0.168% against BF16 CUDA projection results. Their original smoke records retain reduction summaries rather than raw native local target vectors, so that numerical endpoint claim has a narrower saved-audit scope. Later worker versions retain both original and edited native local vectors and distinguish endpoint error from edit-response error. Earlier evidence is unchanged.

The independent reader is a separate implementation over retained arrays, run without model forwards. It is not replication by another investigator or on an alternative runtime. The sealed whole-projection prediction is known from the declared intervention algebra; it does not constitute blinded mechanism discovery.

## Prospective boundary and interpretation

Original-prefix observations are supplied source evidence. All local equation forecasts and substitution/restoration program declarations seal before any target arm executes. The whole-projection equivalence is a numerical identity to qualify, not an independently discovered semantic explanation of looping. New behavioral rankings, reduced-rank edits, timing specificity and cross-context applicability remain scientific questions.

The longer exploratory cases are authored pencil arithmetic, rectangle arithmetic and a literal-response instruction. They are frozen before their new target outcomes, with no preassigned looping/nonlooping label. Greedy decoding is explicitly separate from the publication's stochastic `think35` settings. The sampling-policy follow-up uses temperature 0.6, top-k 20, top-p 0.95 and a 1.5 generated-token presence penalty, while retaining its different HF backend and 2,048-token cap. Its outcomes are exploratory, with repeated contexts identified as such.

The completed longer job `job-a97f819ad791` reproduces all six matched first-logit vectors exactly and all six 2,048-token sequence pairs. All twelve trajectories reach the cap without `</think>` or EOS. Maximum repeated-token-trigram counts are 89/113 for original/edited pencil arithmetic, 41/43 for rectangle arithmetic, and 65/105 for the literal instruction. These descriptive counts do not establish a general repetition effect, but they retain the observed adverse direction in this greedy panel. A correct intermediate calculation is not a completed answer. The lease takes 439.47 seconds, with 3,340 MB peak VRAM and 2,249.7 MB RSS.

The saved reader `job-fcea89ac4f49` independently recomputes all six longer-panel matches. The standard Saturn watch/claims/xref sweep and verification both report `evidence-incomplete-empty-target` on the current WSP schema, with no applicable streams or legacy gate receipts. This is an instrument applicability limitation; it supplies no scientific verdict. Those diagnostics are [retained separately](../../../../../artifacts/mdb-wsp-case-study-evidence-20261110/), and `mdb-wsp-evidence-plane-projection` is recorded in the Saturn request registry. The dedicated WSP forecast/outcome verifier and independently pinned file inventories retain their separate custody scope.

The stochastic follow-up `job-566ecbc02f99` uses one seed per context, shared across paired arms. Results are:

| Context (seed) | Original | Edited | Maximum trigram count, original/edited |
|---|---|---|---|
| Pencil inventory (0) | 2,048 cap; no final answer | 2,048 cap; no final answer | 61 / 41 |
| Rectangle area (1) | 2,048 cap; no final answer | 1,450 tokens, EOS, final boxed 126 | 38 / 28 |
| Literal instruction (2) | 248 tokens, EOS, final `azure` | 1,285 tokens, EOS, final `azure` | 11 / 64 |

Both expected comparisons remain exact on all three contexts: all 248,320 first logits and every emitted token match. The edit's rectangle completion is reproduced by whole-projection substitution and removed by whole-projection restoration. These qualify that particular completed continuation, with no claim of semantic discovery, universality, general accuracy or matched-cost advantage. The literal prompt shows substantial extra work under the edit, while the pencil prompt remains unresolved at the cap.

The six prospective FP64 endpoint errors range from 0.155852% to 0.169261% relative RMS. Comparing the predicted change against the BF16 edited-minus-original change on the same original receiver gives 1.421512% to 18.697796% relative RMS. Endpoint agreement alone would hide that larger error relative to the intervention. Both native local target vectors and all forecast vectors are retained here.

The native job's lease takes 498.31 seconds with peak VRAM 3,342 MB and RSS 2,221 MB. All twelve arms complete their declared horizons, after which JSON serialization of nested read-only sampling metadata fails while publishing `native-brief.json`. The job remains failed. CPU recovery `job-87b6853ac400` copies and hashes the existing artifacts, verifies the original forecast/outcome records, reconstructs the brief from saved arrays, and preserves the malformed report and original request in `failed-report/`. It loads no model and executes no forwards. The [recovery receipt](../../../../../artifacts/mdb-wsp-case-study-20261110/recovery-think35-01/extracted/wsp-result/recovery-receipt.json) explicitly declares that first external outcome pins occur at recovery; the source request was pinned before native execution. Source evidence is not rewritten. The writer now handles nested read-only mappings, with a regression test.

The first sampled saved-reader attempt `job-e8594227ed06` completes its analysis and saves its report, then exposes the same nested-metadata problem in the transport envelope. That failed attempt is retained. The worker now builds its transport envelope from the JSON report it actually wrote, so the persisted and transported metadata share the same representation.

The corrected reader `job-6e42718ca052` is collected and transport-verified. It independently reproduces all six exact first-logit/token comparisons and both numerical error measures at every one of the six receivers from the retained arrays. No model is loaded and no native forward executes. Its custody pins bind the recovery's manifest and forecast seal; they do not retroactively claim an earlier external pin for the failed native job's outcomes.

Completion, final boxed literals, internal reasoning boxes, repetition, output length, native equality, numerical errors and work remain distinct. The presence of a correct boxed value inside an unfinished thought is not final-answer correctness. Every source and target forward, extra projection and parameter-validation hash has a declared accounting boundary. Setup, source capture, forecasts, candidate persistence, full-vocabulary readout and transport are paid work.

The next informative causal contrast is to isolate each edited layer, then compare a BF16-aware reduced-rank response against the full edit while recording each scalar read and the later stopping-related consumer. The rectangle completion and literal-response slowdown provide contrasting discovery specimens. Fresh contexts and additional seeds can then test a frozen contextual forecast. Whole-projection algebra, a favorable example or a small endpoint residual cannot substitute for that prediction task, and no MDB-versus-PyTorch search or cost advantage has been measured here.

## Preserved mechanics history

`audit-01` stopped locally during staging because its work-item cap exceeded MDB's finite contract. `audit-02` corrected the declaration without a model forward. The first CUDA attempt `job-baecc401557f` was killed during loading at 2,014 MB against a 1,581 MB ceiling. The corrected envelope allowed `job-b9e950dd2999` to expose a tokenizer `BatchEncoding` handling error before source capture. `smoke-03` fixes that error and completes. Original failure records remain in the two earlier run directories.

The successful smoke peaks at 3,342 MB VRAM and 2,458.2 MB RSS in a 49.44-second lease. The longer panel declares a 4,096 MB VRAM reservation using that measured peak plus headroom for its changed continuation geometry. These resource numbers are measurements of this model/backend/profile and do not certify another workload. `saved-audit-01` also preserves a local staging refusal; its corrected scheduling contract explicitly declares an analysis task.

`launch.py` can fetch sources, stage, submit and collect. `worker.py` executes the finite programs, and `audit_saved.py` is the independent reader. Run real-model phases through mrun rather than directly on the development laptop. The local mechanics tests in `tests/test_wsp_case_study.py` cover sparse BF16 reconstruction, malformed support, exact retrieval, tokenizer return types and sampling-history behavior.

For a new attempt, choose a new run name; staging and submission refuse reuse. Run from the domains workspace with the public wheels available at the locations declared by `wheels()`. For example:

```sh
rtk proxy ./common/bin/common exec python saturn-pub/packages/mdb/qualification/wsp_case_study/launch.py stage --base artifacts/mdb-wsp-case-study-20261110 --name new-audit --phase audit
rtk proxy ./common/bin/common exec python saturn-pub/packages/mdb/qualification/wsp_case_study/launch.py submit --base artifacts/mdb-wsp-case-study-20261110 --name new-audit
# After the finite job completes:
rtk proxy ./common/bin/common exec python saturn-pub/packages/mdb/qualification/wsp_case_study/launch.py collect --base artifacts/mdb-wsp-case-study-20261110 --name new-audit
```

Use `--phase discovery --sampling think35` for the declared stochastic follow-up. A saved-only audit accepts `--phase saved-audit --native-run COLLECTED_RUN_NAME`; it independently pins the selected source files and forecast/outcome identities. The native worker retains raw local targets in later runs so that this reader can recompute both endpoint error and error relative to the much smaller edit response. `--phase recover --native-run FAILED_RUN_NAME` is a separate, zero-forward recovery lease for a reporting failure after saved outcomes exist. Six focused mechanics tests pass, and the case-study sources pass Ruff.
