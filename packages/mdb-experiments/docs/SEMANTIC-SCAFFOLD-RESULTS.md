# Source-aware semantics, copy, recall and coding

This worktree extends the addition scaffold to four small native-Qwen task families. Supplied task/source labels become token spans and typed value-memory addresses. Native MDB transactions test those sources through unchanged consumers, rather than treating a read score or a token projection as a semantic verdict.

The campaign proposal, raw evidence and results live under `artifacts/mdb-scaffold-semantics-20261111` in the domains workspace. The updated blog is `obsidian/blog/2026-11-11-112926-making-the-time-scaffold-visible-live-symbol-bindings-and-causal-functions.md`. The original addition campaign and its source manifest remain separate under `artifacts/mdb-scaffold-live-20261111`.

## Measured scope

The main panel (`job-7cd1e252a43c`) ran eight pairs; calibration (`job-bcef5f999281`) added three; localization (`job-28b39b928a08`) added one new copy pair and three matched replays. Fifteen executions cover twelve distinct paired contexts, with three-token own-generated suffixes. The one-case smoke (`job-6ae159b26231`) is instrument validation and a repeated discovery example.

- Coding: all native integer completions passed corresponding compiled function/assertion consumers. Source-value swaps at layers 19–27 alone changed `5→6` and `8→9`. The early-plus-middle complement did not flip either. This is source-region sufficiency in two contexts, not native necessity, a minimal circuit or general coding.
- Recall: calibrated word-answer prompts worked; all-depth selected-source values switched `amber→silver` and `cedar→maple`. Narrow frozen carrier effects were weaker. Original unconstrained recall prompts missed the intended response format and are retained.
- Copy: all-depth values switched single-token `amber→violet` and `copper→silver`. The two-row `lumen` versus one-row `nexus` mismatch remained unresolved. Donor-first-row plus tail erasure produced the distractor `marble`.
- Semantics: freezing/boiling and calibrated chilly/scorching prompts produced cold/hot natively. A frozen source route gave specific signed effects against zero-effect grammar controls, but no semantic answer flip. Wider source replacement could copy a description word; the three-token donor `scorching` is not represented by its first source row alone.

## Mechanics and limitations

All model workers use installed public MDB/Saturn-pub wheels, one native BF16 Qwen2.5-1.5B/CUDA/eager resident, strict mrun preflight and no retries. Parameter manifests and standalone retrieval scripts bind the same original weight SHA as the addition panel. Full native session journals, cuts, Acts, raw logit arrays and source payloads are retained.

Saved-only audits verified 32,133 inventoried files, 492 full-vocabulary argmax rows, 328 own-output feedback links, 762 source payloads and fifteen exact sham first-logit/cache results. Fourteen local mechanics checks pass. These are integrity/observer checks, not terminal semantic certification. Cache-change summaries include directly edited source rows; writer-output changes do not establish isolated mediation.

`semantic_scaffold.py` is framework-free and defines declared organisms, source-span resolution and the integer-only Python consumer. `semantic_scaffold_worker.py` uses native Bench sessions and typed memory writes; `semantic_scaffold_launch.py` stages installed-wheel leases; `semantic_scaffold_collect.py` verifies bounded archives; `semantic_scaffold_audit.py` audits arrays/history/payloads and exports an offline occurrence view. Frozen-route files separate discovery from transfer. Later calibration and localization are explicitly additive.

No pretrained weight meanings, arbitrary prompt ontology, general chat/recall/coding competence, inference acceleration or complete semantic circuit have been established. The next cuts are finer late-depth coding coalitions/writer restoration, coherent semantic phrase-versus-later-row swaps, exact span-aware copy controls and recall from generated history.

## Reproduction

From `packages/mdb`, run the model-free tests and tiny CPU port canary:

```sh
rtk proxy /Users/jakeholl/domains/common/bin/common exec /Users/jakeholl/domains/saturn-pub/.venv/bin/python -m pytest -q tests/test_semantic_scaffold.py tests/test_live_scaffold.py tests/test_live_scaffold_native_ports.py
```

Use the launcher's `stage` and `submit` actions with explicit parameter/scaffold helper sources. Modes are `smoke`, `main`, `calibration` and `depth`; the latter two require a frozen-route JSON file. Inspect retained requests and payload seals before reproducing real-model runs. The collector takes one exact staged job root; the auditor takes its retained `results` and a new analysis output directory. Neither audit nor HTML viewing needs model hydration.
