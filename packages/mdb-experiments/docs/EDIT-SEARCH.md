# Scriptable bounded edit search

`mdb edit-search` manages a finite intervention search without loading a model. The runtime remains MDB's typed `Act`, `Session`, `NativeCutNavigator`, `ForecastTransaction` and native consumer. A campaign adapter translates named atoms into typed edits and supplies the native evidence. This separates reusable search bookkeeping from model-specific execution.

Each search binds its complete parent, checkpoint/model, execution contract, consumer and observable. Candidates are subsets of at most twelve named atoms. Freeze a prediction before execution; seal the candidate and plan in the native forecast transaction; execute same-parent replay; audit mechanics and native outputs; attach an explicit consumer verdict. Preserve `uncertain` when the evaluator cannot establish the observable. Image distance alone must not decide whether an object edit succeeded.

```sh
mdb edit-search init evidence/search --plan search-plan.json
mdb edit-search freeze evidence/search --atoms row4.step2 row11.step2 --prediction prediction.json
# Execute the frozen candidate through the model-specific MDB/mrun adapter.
mdb edit-search observe evidence/search CANDIDATE_ID --verdict sufficient --evidence native-audit.json image-review.json
mdb edit-search propose evidence/search --limit 6
mdb edit-search status evidence/search --summary
```

The plan JSON has `binding`, `atoms` and `observable`. `binding` requires `parent`, `model`, `execution` and `consumer`; include donor, runtime, checkpoint, layout, schedule and other dependencies as applicable. The prediction JSON is caller-defined. The public Python API is `mdb.edit_search.EditSearch`; its `create`, `freeze`, `observe`, `propose` and `status` methods expose the same workflow.

Plans, freezes and outcomes are exclusive-create files. Restart with `EditSearch(root)` or the CLI against the existing root. Canonical candidate IDs bind the plan and selected atoms, so atom order does not create duplicate trials. Status verifies freeze identities and every referenced evidence hash. One writer owns each search directory. Pending candidates remain pending after interruption; the ledger does not silently repeat physical work or manufacture outcomes. To revise an uncertain assessment, retain the old ledger and create a separately bound search/review revision.

The proposer visits untested small subsets, including those outside successful branches. It assumes no monotonicity: a failed pair cannot prune either singleton. `smallest_observed` lists successful candidates with the least tested cardinality. `minimum_cardinality_established` becomes true only after every smaller subset has a sufficient/insufficient consumer assessment. It applies solely to the declared finite atoms and observable. It does not prove parameter-, channel-, spatial-, dose- or clock-minimality beyond that lattice, nor fine-pixel preservation.

Use `mdb script` for a focused interactive debugging sequence. For repeated image-edit search, use the Python MDB API inside one resident worker and keep the search ledger outside it. This reuses the admitted model and makes the branch budget explicit. The CLI remains useful for inspecting status, proposing subsets and querying the resulting evidence. A fresh interactive session or fresh model load per edit adds unnecessary work.

The ledger verifies evidence integrity, not whether a human or adapter supplied an honest semantic verdict. The native transaction and custody audit establish that the prediction preceded target execution. Retain both the source of the assessment and instrument limitations in the evidence references. Different parents or execution contracts require a new search; equal-looking images do not authorize cross-run memoization.

## Executed FLUX adapter

The [binding search driver](../qualification/flux_binding_search.py) creates four atoms: donor contextual row 4 or 11, read at project2 or project3. An earlier whole-image pulse stays fixed. The [resident worker](../qualification/flux_binding_minimal.py) restores the exact selected rows from base/donor conditioning at each read, including restoring base at project3 for an early-only edit. Every branch uses the same saved parent and unchanged native suffix/VAE. It counts all native transitions and no-op Acts.

Twelve native futures test the empty set, all four singletons, all six pairs and the full set. Only the early-read pair swaps both colors among tested pairs; no singleton succeeds. A clock-aware prospective rule matches6/6 new cases versus2/6 for the original broader rule. This closes minimum cardinality in this lattice, while leaving four triples, the whole earlier image pulse and fine features unresolved.

The driver prepares six frozen contrasts. After saved-only auditing and explicit native-image review, `--record --run RUN --audit AUDIT --review REVIEW --next round2` records outcomes and freezes the next six smaller untested candidates. Stage/submit/collect remain explicit mrun operations, with one admitted resident CUDA lease and no retries. This is scripted adaptive research with an explicit consumer assessment, rather than an autonomous open-world semantic editor.

Search workers print a small `MDB_DIRECTORY_READY` reference to the completed HDD outbox. [The direct directory collector](../qualification/recover_flux_directory.py) checks the original request-sealed encoder, terminal scheduler record, metadata publication where present, archive, every file and receipt source. It executes no model. After historical recovery and the control-guard fix, job `job-ec8810b4b5d4` qualifies the successful native publication path: five complete branches, 289.96 MB archive, 35.68 seconds collection/validation and zero reruns. The original scheduler log contains metadata only. Fleet-wide mrun CAS/outbox publication remains a separate integration; this collector uses direct SSH. The original scheduler-log collector remains available for historical campaigns.

The second native job generated all six branches, then failed that reporting assertion. Its closed `ForecastTransaction` had already durably stored every row/count and referenced output. A separate index recovery reconstructs unique native receipt paths and port descriptors from original CAS, with independent audit and zero reruns. The original failed job stays failed; the final independent hook counter was not persisted. Future workers save each branch's metadata immediately. The controller accepts only this narrowly identified post-completion error; other failed or incomplete jobs remain refused and retained.

Use [the resumable round controller](../qualification/run_flux_binding_search_round.py) to automate stage → submit → wait → collection → audit with one command:

```sh
python -I qualification/run_flux_binding_search_round.py RUN SIX_AREAS_ROOT PANEL_JSON COLLECTED_RUN AUDIT_JSON
```

Repeated invocation binds the same immutable panel and job, verifies existing custody/audit and avoids resubmission. Queue timeout detaches without cancelling a sound job. Native semantic review and `record_round` remain the explicit assessment step. Its recovery/resume path was exercised against the actual completed evidence.

```sh
python -I qualification/flux_binding_search.py ARTIFACT_ROOT --source SIX_AREAS_ROOT
python -I qualification/flux_binding_minimal.py RUN --source SIX_AREAS_ROOT --panel search --forecasts ARTIFACT_ROOT/round1-panel.json
python -I qualification/flux_binding_minimal.py RUN --submit
python -I qualification/recover_flux_directory.py RUN COLLECTED_RUN
python -I qualification/analyze_flux_binding_search.py COLLECTED_RUN SIX_AREAS_ROOT AUDIT_JSON
python -I qualification/flux_binding_search.py ARTIFACT_ROOT --record --run COLLECTED_RUN --audit AUDIT_JSON --review IMAGE_REVIEW_JSON --next round2
```

Use installed public MDB/Saturn-pub packages. In the domains workspace use `common exec` for mrun credentials and the qualified Python environment; do not add repository paths to Python imports. Actual execution results and limitations are recorded in [the binding report](FLUX-BINDING-MINIMAL.md).

## Recursive carrier and row restorations

[The carrier panel adapter](../qualification/flux_carrier_restoration.py) reuses the same worker/controller. It binds a joint-exit clock, generates a matched first control, then applies typed image/text restorations from that exact source cut. The audit checks each restoration changes only its declared slot, verifies complete payloads separately from lineage, and follows the native future. All source assistance is frozen explicitly.

```sh
python -I qualification/flux_carrier_restoration.py ROOT SIX_AREAS_ROOT
python -I qualification/run_flux_binding_search_round.py ROOT/run-01 SIX_AREAS_ROOT ROOT/panel.json ROOT/collection-01 ROOT/audit-v1.json
# After audit and native-image review, shrink the surviving text carrier.
python -I qualification/flux_carrier_restoration.py ROW_ROOT SIX_AREAS_ROOT --prior ROOT/audit-v1.json
python -I qualification/run_flux_binding_search_round.py ROW_ROOT/run-01 SIX_AREAS_ROOT ROW_ROOT/panel.json ROW_ROOT/collection-01 ROW_ROOT/audit-v1.json
```

Row panels restore baseline image and baseline text outside declared retained edited rows. Groups partition 512 working text rows into the original color positions, a matched nearby pair, other valid positions and padding. The frozen plan binds exact row indices and prior evidence. Success does not imply globally minimal rows or context-independent semantic neurons. Source-custody/import/checkpoint/residency, branch-start and restoration progress events make slow startup and evidence I/O visible on subsequent workers.
