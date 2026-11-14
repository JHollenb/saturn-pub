# First native-discovery implementation and observations

The automatic analysis loop runs end to end on Qwen2.5-1.5B base without semantic annotations or interventions. Learned sparse coordinates predict some early boundaries better than simple mean writes [claim withdrawn by the 2026-11-12 addendum below], but the tested learned accounts do not show a convincing final-consumer advantage over that baseline. This is an implemented research instrument and a mixed predictive result, not recovered semantic knowledge or a working independent reasoner.

## Specimen and execution

The model is `/mnt/big/llm-models/Qwen2.5-1.5B`, with model identity `6996fe7f165932ebad3c4a159abc7d4f23ea42431d21febfbc4f5a6e23fcf2a9`. Its original safetensors archive is pinned by the parameter manifests and independently verified retrieval scripts in each result.

The measured lane is eager FP32 CUDA, PyTorch 2.13.0+cu130, Transformers 5.14.1, Python 3.14.6, batch one, no cache, raw text without chat templates or added special tokens. Boundaries are layers 0, 9, 19 and 28, ending before final normalization. Each exploration uses 32 features per span, top-k eight, three arms and 60 attempted updates per arm, with development feedback every ten updates. All teacher parameters are frozen. Original tokenizer output IDs and physical offsets are retained; final source additionally records tokenizer asset hashes and prompt-format identity.

Corpus windows come from deterministic hash-ranked existing Markdown and Python files, including historical research scripts. They are ordinary source documents, not a representative broad-language benchmark. Six smoke documents and both twenty-document panels have disjoint source groups [corrected: the panels are less separate than this implies; see the leakage paragraph in the 2026-11-12 addendum below]. Each exploration has 12 discovery, four development and four held-out documents, truncated to 64 tokens. The study uses supplied token histories, not generated-token continuation.

## Results, with panels kept separate

Mean last-position full-vocabulary native-to-predicted KL over four held-out documents; lower is better:

| Account | Initial panel | New-source panel with output-consumer feedback |
| --- | ---: | ---: |
| Hold embedding state | 108.018 | 130.670 |
| Discovery mean writes | **5.406** | 4.710 |
| Linear features plus consumer objectives | 8.449 | 5.545 |
| Sparse reconstruction | 6.333 | **4.703** |
| Sparse features plus consumer objectives | 6.866 | 5.160 |

Different corpora prevent attributing the between-column change to the feedback modification. The second panel is a fresh test of the revised workflow, not a matched causal comparison of objectives. Within that panel, sparse reconstruction and the consumer arm each beat mean writes on three of four documents, while both regress on the remaining document. The small reconstruction aggregate difference is not an established advantage. All learned arms match the native top token on zero of four documents in the second panel.

In the initial panel, sparse reconstruction's first-boundary relative MSE is 0.0400, 0.0781, 1.0183 and 0.0826, versus approximately 0.984 for mean writes on each document. At the final boundary its mean relative MSE is 0.9371 versus 0.5310 for mean writes; the sparse consumer arm reaches 2.5146. This preserves a substantial early-boundary trend alongside later drift. In the second panel, final relative MSE is 0.4867 for sparse reconstruction and 0.5043 for sparse consumers, versus 0.4811 for mean writes.

The initial continuation policy selects 40 linear updates and all 60 updates in each sparse arm. Its development-only diagnosis motivated adding differentiable feedback from predicted final state through the original frozen native norm/head, with development output KL taking first priority in continuation selection. The second panel accepts ten updates per arm and rewinds five of six candidate intervals per arm. Rejected states and their improvements/regressions remain archived. Each accepted arm improves its development output KL from the 8.0886 mean-write initialization, reaching 7.0621, 7.5730 and 7.2643 for linear, reconstruction and sparse consumer respectively. Those development improvements are feedback evidence, not fresh-evaluation claims.

Each exploration exports 96 feature-coordinate candidates, eight statistical role groups, 96 proposed source bindings and 96 observed uses. These are not counts of independent concepts or proven semantic roles. The canonical symbols bind learned source nominations to exact passive observation locations. Native attention mass and top source remain visible as comparison evidence; attention weights do not establish necessity. Native route KL often remains worse than uniform causal attention, so the source resolver is a particularly weak part of this first pass.

## Work, custody and verification

| Run | Job | Worker seconds | Scheduler seconds | Peak RSS MB | Peak VRAM MB |
| --- | --- | ---: | ---: | ---: | ---: |
| Six-document mechanics smoke | `job-3f6ccd1d56a3` | 49.74 | — | — | — |
| Initial twenty-document panel | `job-7a51d642a6dd` | 97.76 | 123.91 | 8,981.2 | 6,352 |
| Consumer-feedback panel | `job-950c44e99d19` | 95.87 | 126.97 | 9,641.6 | 6,360 |

Each exploration loads the teacher once and runs twenty physical document forwards. The initial panel pays 24 last-token norm/head calls; the revised panel pays 456 across fitting, development evaluation and held-out forecasts. Both fit 180 physical student updates and retain 21 candidate closures plus three accepted closures. All runs use one finite admitted lease, strict preflight, no retries and zero interventions. The first pre-model attempt, `job-d339d14fce34`, failed request-digest verification; its failure remains intact.

Directly collected archives match the job markers. The initial archive SHA-256 is `22a34b09d35e9d3c82c94e284f4922cf266c7963c8ef8c5c7328b5bc57eee04e`; the consumer-feedback archive is `ecf6ffc66180e873def8dee52b239010d82862c3a225bd26be2947a537e60575`. Original named parameter retrieval verifies all 338 required tensors from one original archive without persisting another pretrained weight bank.

Artifact-only audits independently recompute 200 sealed state/output metrics per exploration and verify all 24 saved candidate/accepted closures. They perform zero teacher loads or forwards. Each run's corpus allocation, artifact inventory, forecast seal and parameter-custody extension verify. Standard evidence run/verify succeeds on a separate 80-row numerical projection, with 80 model/input-qualified citations and three boundary addresses per panel. This standard sweep verifies the projection; dedicated analysis verification checks the actual run.

The focused Python 3.12 suite passes 100 checks, including knowledge-interchange coverage, and the new pipeline's eight checks also pass against local PyTorch 2.13. An older Python 3.10 environment fails an unchanged directory-fsync monkeypatch test; the same test passes on Python 3.12. Real-model execution uses the sealed installed wheels, not workspace imports. Subsequent source changes add the offline evidence projection, frozen-model admission guard and more explicit tokenizer/profile metadata; measured source/wheel bytes remain pinned in each payload.

Workspace evidence:

- [Initial panel and audit](../../../../../../artifacts/mdb-native-discovery-20261112/main/)
- [Consumer-feedback panel and audit](../../../../../../artifacts/mdb-native-discovery-20261112/consumer/)
- [Workflow and claim boundaries](../../docs/NATIVE-SEMANTIC-DISCOVERY.md)

## What this establishes and what remains open

Mechanics are valid for the named runs. The working inference is that useful intermediate structure can be learned from ordinary execution, while the current representation, shared read map, source resolver and training horizon do not yet retain the distinctions needed by the final consumer. Sparse reconstruction's early performance is a lead to investigate [claim withdrawn by the 2026-11-12 addendum below]; a failed final comparison does not establish that sparse concepts cannot work.

The next useful comparison is a matched discovery/development study of formation versus late cancellation, with checkpoint curves, per-head read diagnostics and explicit positional information, followed by another quarantined source set. This does not require intervention panels. Semantic descriptions, recurring computational subgraphs, cross-refit identity, generated-token continuation, actual FieldLM adaptation and a trained 3D workspace remain open. Typed anonymous context export is implemented; an independent knowledge base is not established.

## Addendum, 2026-11-12 14:42 PST: re-analysis with position-robust metrics and trivial baselines

An independent review flagged that the early-boundary result might be a first-token effect. This addendum re-scores the two saved panels from their artifacts only (no model load, no forward) with the shared `mdb.analysis_metrics` and `mdb.attention_ranking` modules, using `reanalyze_sink.py` in this directory. The original numbers above are unchanged and still verify; what changes is how they should be read. All values below are measured on the saved artifacts.

**Position 0 holds almost all of the residual energy at the early boundaries.** At layers 9 and 19, the first token carries 99.73–99.90% of the target's squared size in three of four held-out documents of the initial panel and in all four of the second panel. The fourth initial-panel document (it starts with the digit `2`) has no such activation (0.01%). The uncentred relative MSE used above is therefore mostly a score for predicting that one large vector.

**A trivial baseline that knows about position 0 beats every learned arm there.** A mean write fit on discovery documents separately for position 0 and for all other positions scores relative MSE 0.0009–0.0022 at layers 9 and 19 on the three documents with the large first-token activation (0.0008–0.0039 on the four second-panel documents). Sparse reconstruction scores 0.038–0.083 on the same three documents. On the document without the activation, the split baseline is worse (1.92) than the plain mean write (0.98), and every learned arm is about 1.0, so nothing learned predicts that document either.

**Excluding position 0, the learned arms are far worse than doing nothing.** On positions 1 onward at layers 9 and 19, relative MSE is 5.1–36 for sparse reconstruction, 8.3–56 for the sparse consumer arm, and 90–567 for linear features in the initial panel, against 1.0 for holding the embedding state and 0.54–0.77 for the split mean write. In the second panel all arms, including the plain mean write, sit at 6.0–16.5 on those positions. The centred variant of relative MSE (error divided by the target's variation across positions) is still dominated by position 0, so it does not settle the question by itself: it is about 1.0 for hold-state and mean writes in both panels and for every arm in the second panel, but 0.04–0.11 for the sparse arms in the initial panel. The position-0-excluded numbers are the decisive ones. The initial-panel ranges "5.1–36 / 8.3–56 / 90–567" exclude the held-out document that starts with the digit `2`, where the same quantities are about 1.0.

**Like-for-like, a small edge for sparse reconstruction survives.** Comparing sparse reconstruction with the plain (global) mean write on positions 1 onward at layers 9 and 19, sparse reconstruction has the lower relative MSE in 8 of 8 cells in the second panel, by 3–6% (for example 15.71 versus 16.16), and in 2 of 8 cells in the initial panel. This leftover edge exists, but it is not established: each panel has only four held-out documents, and both arms are far worse than holding the embedding state (1.0) on these positions.

**So the "early-boundary advantage" of sparse reconstruction is not supported once the first-token activation is accounted for.** It reflects predicting the first token's large vector, which a position split (two mean vectors per span: one for position 0, one for the rest) does an order of magnitude better, while the rest of the sequence is predicted worse than leaving the state unchanged. At the final boundary (layer 28, where position 0 holds 8.5–20% of the energy) the split mean write also has lower relative MSE than sparse reconstruction on all 8 documents across both panels.

**Route nominations lose to "attend to position 0".** At layers 9 and 19, attending to position 0 matches the native top source on 0.68–0.85 of rows in seven of eight documents (0.03–0.04 on the document starting with `2`). Learned arms match on 0.02–0.10 for the sparse arms in the initial panel and up to 0.38 in the second; linear features reaches up to 0.58 on initial-panel documents. At layer 0, attending to the current token matches 0.47–0.52 of rows, against 0.05–0.10 for every learned arm. In none of the 72 document × arm × layer cells does a learned route beat the best trivial baseline.

**Corpus leakage guards.** The second panel's corpus has one byte-identical file under two paths (`.../flux-h18-history-20261106/source/audit/audit_flux_functional_pass.py` and `.../flux-raw-mask-pair-20261106/source/audit-v3/audit_flux_functional_pass.py`): one window is in held-out, the other in discovery. The windows are different parts of the file (word-shingle Jaccard 0.002), so this is file-level, not text-level, overlap, and the earlier statement that panels have disjoint source groups is not correct for that panel. Across panels, three byte-identical files are shared (smoke held-out = consumer-panel discovery, namely `retrieve_vae_weights.py` / `retrieve_qwen_weights.py`; smoke discovery = main-panel development; main-panel discovery = consumer-panel discovery). No run leaks into itself through these, but the panels are less separate than "fresh test" or "quarantined source set" suggests, and a result on one panel is not fully independent of the others. The extractor now groups by file content hash. The visible-input check previously refused only identical whole token windows; it now also refuses cross-split windows that share their first eight tokens. The longest cross-split shared leading run in the saved panels is two tokens in the initial panel and one in the second, so neither would have been refused.

**What changed in the shared code.** `error_metrics` keeps its fields and adds centred and position-0-excluded relative MSE and the position-0 energy fraction. Held-out reports add the position-split mean write (fit on discovery only, scored after the fact and not sealed) and trivial top-source baselines for the native routes. Each learned route also gets a `source_top1` record that always carries position-0, previous-token, current-token and uniform baselines. Training and selection are unchanged.

**What remains open.** Whether learned sparse coordinates carry useful structure on non-first positions is now untested rather than shown: these runs fit to an objective that the first-token vector dominates. The next fit should exclude or separately model position 0, or score positions 1 onward only, and report the split baseline beside every arm. The final-consumer comparison above is unaffected, because it reads only the last position. Why the document starting with `2` has no large first-token activation is not established.

## Addendum, 2026-11-13 21:32 PST: position-robust rerun on a new 80-document panel

**Question.** Does any learned attention route beat the best trivial baseline once the first-token attention sink is split out? **Answer: no.** In this rerun no learned route beats the best trivial baseline in any of 288 held-out cells (16 documents × 3 route layers × 6 learned arms), on all rows or on rows whose native top source is not position 0 (MEASURED). On those non-position-0 rows the learned routes sit at or below chance: 185 of 288 cells score below the expected agreement of uniform attention (MEASURED). The only route predictor that beats the run's four sealed trivial baselines is the learning-free contrast nominator, and it does so because it picks one positional rule per head; a per-head raw-mass rule fitted on the same discovery documents does almost exactly as well (MEASURED, details below).

**Run.** Qwen2.5-1.5B base, eager FP32 CUDA on Beast, the same pipeline and settings as above (64 tokens, layers 0/9/19/28, 32 features, 60 attempted updates per arm, development check every 10). There are six learned arms: the original three plus a `.p0robust` version of each, which uses separate scales and initial writes for position 0 and later positions, selects on position-0-excluded centred error, and keeps the two native KV groups apart in the predicted read. The position-split mean write and the contrast nominator are sealed with the forecasts. The new corpus (`rerun-corpus/main-corpus.json`, fingerprint `325f0c28…`) has 80 source groups, one document per group, split 48 discovery / 16 development / 16 held-out, and shares no group or text hash with the smoke corpus (MEASURED). The nine discovery source hashes printed in both job logs match the committed worktree source (MEASURED).

| Run | Job | Worker s | Scheduler s | Peak VRAM MB | Peak RSS MB | Archive SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 6-document smoke | `job-f65fbc369bca` | 87.8 | 111.5 | 6,358 | 7,593.7 | `20b1ccad…bcc55` (not transferred) |
| 80-document main | `job-635f65272039` | 286.3 | 348.8 | 6,554 | 3,959.5 | `b9f9baeb…4229e` (transferred, hash matches) |

All values in this table are MEASURED from the scheduler record and the result marker. The main job reserved 8,270 MB VRAM (smoke peak × 1.3), priority 70, and waited 129 s in the queue. `audit.py --closures` re-scores 1,488 sealed metrics and verifies 48 saved candidate/accepted closures with zero model loads (MEASURED). `rerun_report.py` and the new `posthoc_route_baselines.py` produce the tables below; the full JSON is next to the archive (`rerun-report.json`, `posthoc-route-baselines.json`).

**First-token sink is present.** Position 0 carries more than half of the native residual energy at layers 9 and 19 in 16 of 16 held-out documents, and in 0 of 16 at layer 28 (MEASURED).

**Route top-source agreement, all rows** (mean over 16 held-out documents; learned arms are the best and worst of the six; MEASURED):

| Layer | position 0 | previous token | current token | uniform | raw mass per head (post hoc) | best trivial per document | learned arms (range of means) | contrast nominator | held-out per-head ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.038 | 0.199 | 0.506 | 0.074 | 0.621 | 0.621 | 0.066–0.086 | 0.632 | 0.641 |
| 9 | 0.795 | 0.069 | 0.059 | 0.074 | 0.795 | 0.795 | 0.124–0.410 | 0.796 | 0.803 |
| 19 | 0.713 | 0.069 | 0.115 | 0.074 | 0.814 | 0.814 | 0.101–0.383 | 0.814 | 0.815 |

The best learned cell is still 0.26 below the best trivial baseline for that document and layer (MEASURED: largest learned margin −0.263 at layer 9, −0.327 at layer 19, −0.457 at layer 0). "Best trivial per document" takes the maximum over position 0, previous token, offsets 0–7 from the end, uniform and raw mass per head, which is generous to the baselines. With only the run's four sealed baselines it is 0.506 / 0.795 / 0.713, and no learned arm reaches those either (MEASURED: `beats_every_trivial_baseline` is false in all 288 sealed records).

**Route top-source agreement, rows whose native top source is not position 0** (the sink split out; these rows all have query ≥ 1, so previous token is always defined; MEASURED):

| Layer | rows (share) | current token | previous token | uniform (expected) | best trivial per document | learned arms (range of means) | raw mass per head | contrast nominator |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11,819 (0.962) | 0.510 | 0.198 | 0.056 | 0.510 | 0.046–0.068 | 0.611 | 0.625 |
| 9 | 2,522 (0.205) | 0.210 | 0.263 | 0.047 | 0.274 | 0.026–0.058 | 0.000 | 0.202 |
| 19 | 3,526 (0.287) | 0.353 | 0.191 | 0.047 | 0.353 | 0.027–0.075 | 0.410 | 0.410 |

Here the learned arms are 5–10 times below the best single trivial rule and mostly below uniform. The `.p0robust` arms do not change this: their non-position-0 means are 0.057 / 0.045 / 0.027 (linear), 0.068 / 0.026 / 0.075 (sparse consumer) and 0.046 / 0.046 / 0.030 (sparse reconstruction) at layers 0 / 9 / 19 (MEASURED). On all rows their layer-9/19 agreement (0.12–0.41) is higher than the original arms' in five of six comparisons, but at most about half of the position-0 baseline (MEASURED).

**The contrast nominator is a per-head positional rule, not a learned route.** It beats the run's sealed trivial set on 16/16 documents at layers 0 and 19 and 8/16 at layer 9 (MEASURED), but its rules differ from the raw-mass per-head rules in one of twelve heads at layer 0, one at layer 9 and none at layer 19 (MEASURED). Its mean gain over raw mass per head is +0.011 at layer 0 (14 of 16 documents above, 1 below), +0.0015 at layer 9 (8 above, 8 below) and exactly 0 at layer 19 (MEASURED). The fair reading is that one positional rule per head (position 0, current token or a short offset) explains about 62–81% of top sources, and that a single rule per layer cannot capture that. Contrast ranking adds at most a small amount on top of raw mass. On non-position-0 rows at layer 9 the contrast nominator (0.202) is below previous token (0.263) on every document (MEASURED).

**Could a learned route have won in this pipeline?** Partly checked. The scorer can register a win: the per-head positional rules (raw mass, contrast) clear the sealed baselines on 16/16 documents at layer 0 with the same `evaluate_route_forecast` code (MEASURED). A held-out per-head ceiling, which reads the answer, scores 0.641 / 0.803 / 0.815, so head-level positional rules leave little headroom on all rows (MEASURED). **Planted-route check (added 2026-11-13, after the audit; CPU, synthetic, MEASURED).** `packages/mdb-lanes/tests/test_native_discovery_planted_route.py` plants a previous-token route (every head; row 0 reads itself) in 28 synthetic documents (T = 10, width 8, 2 heads), whose state is a token embedding plus a fixed sin/cos position code, and fits the same `fit_predictors` (features 16, 10-step intervals). Top-1 agreement with the planted route on 8 held-out documents (uniform expectation 0.29, position 0 0.20, previous token 1.0):

| run | linear_features | sparse_consumer | sparse_reconstruction (no route loss) |
| --- | ---: | ---: | ---: |
| 60 steps (the default budget), 60 accepted updates per arm | 0.56 | 0.41 | 0.34 |
| 200 steps; 200 / 200 / 180 accepted | 1.00 | 0.94 | 0.43 |
| 600 steps; 310 / 320 / 420 accepted | 1.00 | 0.97 | 0.44 |
| 600 steps, position code removed from the state | 0.29 | 0.33 | 0.28 |

So the learned route form can recover a planted previous-token route, but only when its input carries position, the arm has the route loss, and it gets on the order of 200 accepted updates. Without the position code no arm leaves chance even at 600 steps; without the route loss the arm stays below 0.45; at 60 accepted updates no arm reaches 0.6. The Qwen rerun had no position input, route loss on 4 of 6 arms, and 10-40 accepted updates per arm, so **this learner, as run, fails; whether native routes are learnable is not yet known.** The next test is a rerun on the same documents with (1) position input to the route, (2) the route loss on all arms, and (3) at least a few hundred accepted updates per arm, scored against the same sealed and discovery-fixed baselines. This is a synthetic check on a toy route; it shows what the learner needs, not what the native routes are.

Before the check above, the structural reasons were (ASSERTED from reading `native_discovery_learning.py`; items 1-3 are now tested on planted data as above):
1. The predicted route is a rank-4 query·key softmax over the predicted state's feature code, with no position input. Previous token, current token and fixed offsets are not directly expressible.
2. The `sparse_reconstruction` arms have no route loss at all (`auxiliary=False`). Their routes are shaped only through the state write.
3. Each arm fits at most 60 single-document updates and accepts only 10–40 of them (MEASURED from the fit log: accepted 10 / 20 / 10 / 10 / 40 / 20).
4. Layer-9 and layer-19 routes are computed on predicted, not native, states.

**State and output results (secondary; split baseline shown beside each, MEASURED).**
- Position-0-excluded relative MSE at layers 9 and 19: the original arms score 13.8–14.1 and 6.5–6.7, against 0.70 and 0.58 for the split mean write and 1.0 for holding the state.
- The `.p0robust` arms match the split mean write to within about 0.05 (mean paired difference) and beat it, with a bootstrap interval excluding zero, in only one of nine comparisons. Sparse reconstruction `.p0robust` at layer 9 wins 10 of 16 documents by a mean of −0.0015 (bootstrap 95% [−0.0027, −0.0005]). That edge is far too small to matter.
- Last-position output KL: `linear_features.p0robust` 7.425 vs split mean write 8.119, 16 of 16 documents better, mean −0.69 [−0.83, −0.56]. `sparse_consumer.p0robust` scores 7.621, 13 of 16 better.
- Every arm and baseline gets 0 of 16 native top-1 tokens. The KL gain is real on these documents but leaves the forecasts far from the native output.
- At layer 28, `sparse_consumer.p0robust` and `sparse_reconstruction.p0robust` have uncentred relative MSE of 18.0 and 8.4 (split mean write 0.43), while their position-0-excluded error is normal (0.47, 0.46). Their position-0 write extrapolates badly at the last span, where there is no sink. The cause has not been diagnosed.

**What was not tested.**
- A learned route with explicit position input, or with a route loss in the reconstruction arms, on the Qwen documents (the planted check above is synthetic only).
- Longer training, more than one seed, other models or sequence lengths above 64 tokens.
- Generated-token continuation.
- Whether learned routes help the state or output forecasts even though their top source is wrong. Top-source agreement is the only route metric reported here; route KL is in `report.json` but not analysed.
- The smoke archive was not transferred or hash-checked; only its result marker and report were read. `mdb analyze verify` was not run on the main archive; `verify_run` (inside the worker and again in `audit.py`) and the archive hash were.
- Offsets 1–7 from the end clamp early rows to position 0 and get free hits there. This affects only the all-rows table, and it raises the baselines, not the learned numbers.

The answer is therefore not that a learned route cannot beat trivial baselines. In this predictor, at this training budget, none does (MEASURED). ASSERTED from reading the code: the predictor has no direct way to express the positional rules that the native heads mostly follow (no position input to the route). The planted-route check above is consistent with this: with the position code removed, no arm recovers a planted previous-token route (MEASURED on synthetic data, not on Qwen). Conclusion: this learner fails on Qwen; whether routes are learnable here is not yet known; next test: position input, route loss on all arms, more accepted updates.
