# Symbols with native definitions, uses, and clock evidence

The experimental symbol lifecycle index connects existing read-only `SymbolBinding` records to retained native cuts, declared dataflow slices, spectral identities, VM execution manifests, and consumer observations. It keeps the native model and its typed operations responsible for execution. The index neither hydrates tensors during navigation nor turns an interpretation into write permission.

## Inspect an investigation

```sh
mdb symbols /path/to/project add lifecycle.json --session-id trial
mdb symbols /path/to/project list --symbol subject.country --session-id trial
mdb symbols /path/to/project show LIFECYCLE_SHA256 --session-id trial
mdb symbols /path/to/project at values --cut SOURCE_CUT --selector '[22,0,1,7]' --session-id trial
mdb symbols /path/to/project resolve spectral-query.json --cut SOURCE_CUT --branch main --session-id trial
```

`at` returns all registered matches at that exact cut and selector. For a decoder tensor sequence, the example selector means layer 22, batch 0, KV head 1, row 7; the selected object is the row vector. It does not search arbitrary tensor values for meaning. An empty result describes index coverage rather than absence of a representation.

The corresponding Python methods are `Project.save_symbol_lifecycle`, `symbol_lifecycle`, `symbol_lifecycles`, `symbols_at`, and `resolve_symbol_address`. All are metadata operations. Native hydration remains `Project.load_cut` followed by the compatible public `Session` adapter.

`resolve` takes the existing `SpectralQuery` fields as JSON and requires exact symbol, scene, instance, role, and frame identity. It runs the existing exact SAS read kernel on registered unary addresses at the selected native cut and returns its read trace. Wrong-instance and wrong-role queries abstain; conflicting payload bindings remain ambiguous. Compound ordered bindings remain an explicit gap in this first index. Selected records include validity status and recorded invalidations; lookup does not silently expire or certify them. Address resolution reports no executable or terminal consumer authority.

## Evidence record

`mdb.symbol_lifecycle` provides four sealed records:

- `SymbolOccurrence` records a qualified definition and its location set at a source cut. It can retain a coalition, an exact spectral address, a VM bundle, and CoT observation references.
- `SymbolUse` records a named native consumer, its source and immediate parent/result cuts, receipt, and signed observations. The earlier source cut and immediate transaction input can differ.
- `ClockWitness` records a typed reading, clock domain, provenance, and explicit validity evidence.
- `SymbolLifecycle` joins one original public binding to these events and additive invalidations, with separate mechanics, instrument, trend, and terminal status.

Several lifecycle records can describe the same name at different supported execution points. Their original binding fingerprints remain distinct. A new observation creates a new immutable lifecycle version; an invalidation does not rewrite historical records.

Project publication verifies the original binding seal, model/execution identity, source context, native selector support, cut membership, receipt linkage, and available native consumer declarations. Observed uses require their receipt's exact parent and result. Partial unknown uses retain an explicit incomplete-closure label. Each definition also receives a link to the existing declared dataflow slice for its full root binding; occurrence subsets remain separate. Whole-slot graph dependencies remain distinct from selector-specific causal influence. Missing graph history remains visible in the slice.

Typed spectral and VM seals establish descriptor integrity. The Project join records separately whether a VM bundle's model, execution, source-cut, and context metadata match the native binding. Opaque references remain explicitly unjoined. Neither a seal nor a supplied address-to-location association establishes learned semantics, payload availability, or semantic equivalence.

## The Black Mamba clock boundary

The [Black Mamba clock-specificity organism](../../../../saturn/experiments/2026-09-18-black-mamba-clock-specificity/FINDINGS.md) supports address-local transitions: ticking one selected bank can expire that value while leaving another bank unchanged. Its expiry rule was supplied, and an adaptive gated slot matched the local recurrence. This does not establish a universal exponential clock law.

The later [native Mamba retest](../../../../saturn/experiments/2026-10-02-mamba-clock-retest/FINDINGS.md) separates retention from writing. Releasing retention hurt the tested commit in zero of six cells; the joint timestep effect was write-dominated. A clock witness therefore records temporal evidence without certifying a commit or semantic validity.

The index consequently applies these rules:

- A native execution cursor, layer coordinate, address-local age, selective timestep field, and phase coordinate remain distinct clock kinds.
- Observed execution-point readings must equal retained native metadata, including on invalidations. Observed token/depth/layer coordinates name `metadata.native_point_field` and match that field exactly. This verifies the coordinate; its interpretation remains explicit.
- A local clock belongs to its own domain and reset history. The age of another bank is not substituted for it.
- Host wall time and the VM's paging/LRU counter do not expire a symbol.
- Phase wrapping alone supplies no elapsed-age or winding evidence.
- A native selective timestep may be vector-valued and jointly affect retention and writing. No scalar lifetime is inferred from it.
- Supplied lifetime policies and measured invalidations remain explicit evidence records. This index does not run an automatic expiry controller.

## Qualification

The focused tests cover sealed serialization, native context and receipt refusal, exact source selectors, framework-free reopening, forward/reverse navigation, no tensor hydration during index queries, and preservation of historical invalidations. The native assay and its actual outcomes are recorded in [`experiments/symbol-lifecycle`](../experiments/symbol-lifecycle/README.md). Tiny mechanics fixtures do not establish learned representations or pretrained-model semantic utility.

Legacy native StateCut, address, CAS, and numerical-contract identities are unchanged. The public dependency direction remains `mdb -> saturn_pub`.
