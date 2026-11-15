# Native definitions, uses, spectral identities, and scoped clocks

The symbol table can now retain a context-qualified native definition, its coalition of locations, exact spectral identity, VM descriptors, local clock evidence, and signed downstream use observations in one immutable lifecycle. A pretrained native assay exercised the complete join and durable reopening. Its effects were heterogeneous and did not change the generated token. This establishes the bounded navigation and execution mechanics, while semantic symbol discovery and semantic CoT qualification remain open.

## Why this is the useful join

The [October 22 CoT position assay](../../../../../saturn/experiments/2026-10-22-cot-single-vs-all-position/FINDINGS.md) motivates retaining distributed location sets instead of assuming one scalar or token owns a concept. The [hint-read investigation](../../../../../saturn/experiments/2026-10-15-cot-hint-read/FINDINGS.md) separates observational read annotations from causal interventions. Exact SAS identity supplies symbol/instance/role/frame scope, while the VM bundle supplies sealed execution provenance. Neither supplies semantic validity on its own.

The [Black Mamba clock-specificity organism](../../../../../saturn/experiments/2026-09-18-black-mamba-clock-specificity/FINDINGS.md) motivates address-local domains and reset histories: ticking one bank can affect it while leaving another unchanged. Its expiry grammar was supplied. The [native Mamba retest](../../../../../saturn/experiments/2026-10-02-mamba-clock-retest/FINDINGS.md) separates retention from writing, so a clock is evidence rather than a commit certificate. The implementation consequently preserves vector timestep fields, phase/winding distinctions, invalidations, and supplied policies without silently expiring symbols from host time, LRU counters, or another bank's age.

`mdb.symbol_lifecycle` provides sealed `ClockWitness`, `SymbolOccurrence`, `SymbolUse`, and `SymbolLifecycle` records. `Project` persists them, verifies native cut/context/receipt support, links full-root-binding declared slices, resolves exact unary SAS queries, and reverses native locations to registered symbols. Navigation does not hydrate tensors or select a different native cut. The [API and CLI contract](../../docs/SYMBOL-LIFECYCLE.md) describes the implemented boundaries.

## Pretrained specimen and physical work

The primary run is [job-64a3725e8b10](../../results/symbol-lifecycle/job-64a3725e8b10/job.json): Qwen2.5-0.5B-Instruct, locally materialized weights, HF/eager, CUDA BF16, batch one, seven prompt tokens, and a one-token native continuation. The exact native execution identity includes model bytes, environment, schema, and the experiment-local cache-edit adapter contract. The input was `France Paris The capital of France is`. The `Paris or Rome?` string in the report is contrast metadata; it was not appended to the model input.

The source nomination was supplied from token text: layer 6, committed rows 0 and 1, both KV heads, keys and values. Each source location now selects the entire row vector, for example `[6,0,0,1]`, rather than one scalar dimension. A registered Act replaces present committed-cache rows at a declared pre-embedding boundary; it does not reconstruct arbitrary historical execution. The public native decoder suffix remains unchanged.

One shared baseline and seven candidates used 232 native transitions plus 11 Acts. Two independent prefills and nine restores brought the declared bounded total to 254, below the 256 ceiling. One model was resident in one guarded lease, with strict preflight, CUDA admission, no automatic retry, and an 840-second timeout. The process completed in 51.44 seconds with 1,911.1 MB peak RSS and 1,308 MB peak VRAM. This is a run measurement, not a speedup claim.

## Observations

The baseline Paris-minus-Rome raw-logit margin was 7.40625. Positive changes below favor Paris relative to Rome; these labels are supplied observables, not a correctness evaluator.

| Arm | Change in Paris minus Rome | Generated token changed | Parent restored exactly |
|---|---:|---|---|
| Parent copy | 0 | No | Yes |
| Germany/Berlin donor at nominated rows | 0 | No | Yes |
| Same donor at another target row | +0.0625 | No | Yes |
| Zero source member 0 | -0.5625 | No | Yes |
| Zero source member 1 | +0.125 | No | Yes |
| Zero both source members | -0.25 | No | Yes |
| Half-scale both source members | -0.25 | No | Yes |

All first-token argmax IDs remained 264, outside the three supplied answer labels. The parent-copy arm exactly matched its recorded three logits, margins, argmax, and generated tokens. Full-vocabulary logit equality was not measured by this assay. The donor changed Berlin's recorded logit without changing the Paris-minus-Rome margin; all three pairwise signed contrasts are retained in [observations.jsonl](../../results/symbol-lifecycle/job-64a3725e8b10/observations.jsonl).

The working inference is that a symbol lifecycle should preserve a consumer-specific effect profile rather than one boolean “used” or “valid” value. Member deletion effects oppose each other, the coalition effect is non-additive, and half/zero doses coincide at this measured BF16 readout. One checkpoint/context and one shared baseline cannot establish generality, a unique causal symbol, or a calibrated dose law. The wrong-row arm tests address geometry with the same donor; it is not a wrong-source semantic control. Italy/Rome appears only as unexecuted metadata in the historical report. The final-eight-layer coalition was omitted from this bounded panel.

## Custody and independent reopening

The [raw report](../../results/symbol-lifecycle/job-64a3725e8b10/report.json), [guarded custody](../../results/symbol-lifecycle/job-64a3725e8b10/custody.json), [request](../../results/symbol-lifecycle/job-64a3725e8b10/request.json), and [exact executed worker](../../results/symbol-lifecycle/job-64a3725e8b10/executed-worker.py) are retained. The archive is 5,799,351 bytes with SHA-256 `e45d7e69b713c1cc9ea8173a57e46b392f02a341825def3f392166a1bca27b83`.

The [offline installed-wheel audit](../../results/symbol-lifecycle/job-64a3725e8b10/offline-audit.json) reopened both projects with Torch, Transformers, mrun, private Saturn, and common imports blocked. Each project retained one definition, seven uses, 18 cuts, and 20 receipts. It verified all referenced native pages/blobs, hashed 83 native content files totaling 3,632,496 bytes per project, and repeated forward/reverse lookup. Exact SAS lookup selected the occurrence; wrong-instance and wrong-role lookups abstained. Native selection remained unchanged. Hashing bytes does not decode tensors or perform a fresh native forward.

The [mechanics receipt](../../results/symbol-lifecycle/job-64a3725e8b10/mechanics-receipt.json) additionally verifies 18 VM descriptor files per project and equality of declared, sealed, and executed payload hashes. Those descriptors are pointer-only source/address/operator/renderer records; this does not establish VM page execution or payload equivalence. The historical raw export-summary counts incorrectly read the result wrapper and show zero definitions/uses. Independent sealed-project counts are 1/7; the receipt preserves that diagnostic separately, and the current producer's display-count fix passes the tiny fixture.

The standard evidence sweep and verification completed with `target-streams-and-receipts`, seven observations, one mechanics receipt, no watch findings, and no drift. The xref instrument found no `ar://` citations; it does not certify semantic claims from SAS address strings. See [sweep](../../results/symbol-lifecycle/job-64a3725e8b10/evidence-run.json) and [verification](../../results/symbol-lifecycle/job-64a3725e8b10/evidence-verify.json).

## Preserved predecessors and validation

- [job-977af11c377d](../../results/symbol-lifecycle/job-977af11c377d/job.json) stopped before model loading because a development-host cache path was sent to Beast.
- [job-3f8e7514ecc5](../../results/symbol-lifecycle/job-3f8e7514ecc5/job.json) exposed public worker-bootstrap argument forwarding. The fix restores worker argument semantics and has a regression test.
- [job-7b9a7150f144](../../results/symbol-lifecycle/job-7b9a7150f144/report.json) loaded the model but rejected two differently shaped registry-spec dictionaries. The corrected check compares every declared identity field and retains the richer loaded spec.
- [job-d3092c3d677c](../../results/symbol-lifecycle/job-d3092c3d677c/report.json) completed with representative scalar annotations. The primary run corrected them to full row vectors. Its [annotation replay](../../results/symbol-lifecycle/job-64a3725e8b10/annotation-replay.json) has identical native cut IDs, receipts, and probe results while changing the lifecycle identity. This is same-specimen replay of an annotation correction, not independent semantic replication.

Validation passed 69 focused MDB regression checks, nine distribution/bootstrap checks, the full native tiny fixture, and ten public symbol-path checks run separately. Installed-wheel navigation and the actual collected archive audit passed without framework imports. The three two-bank clock tests are a supplied pure-Python organism, not a new pretrained Mamba result.

Mechanics and instrument status are valid within these bounds. Trend status remains observations-only and terminal status remains not assessed. The next informative research cut is a source-only nominated coalition across several contexts and consumer horizons, with source alignment, wrong-source, and independent donor controls retained per row; it should not turn one observed margin into automatic symbol validity.
