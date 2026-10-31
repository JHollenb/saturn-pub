# Computation atlas

`mdb.computation_atlas` imports explicitly selected, source-qualified
`EquationCatalog` observations across runs. It stores immutable metadata using
public Saturn `Receipt` CAS. It does not execute a model, evaluate equations,
infer semantic names, expand applicability, or promote a candidate to a rule.

An observation keeps the full sealed equation record and report, generic law
and route IDs, declared consumer IDs, tested context, signed effects,
continuation status, counterexamples, and artifact provenance. Original source
catalogs and historical evidence remain unchanged. The atlas snapshot remains
readable when the source catalog is unavailable.

## Executed cross-run example

The [state-transition campaign](STATE-TRANSITION-DISCOVERIES.md) materializes
38 observations: ten earlier matched-context records and twenty-eight new rows.
All 38 exact-context queries match uniquely; changing the implementation source
matches zero. The atlas also retains three caller-selected response-family
candidates comparing the same zero-MLP sites on OR and AND windows.

```python
from mdb.computation_atlas import ComputationAtlas

atlas = ComputationAtlas("artifacts/mdb-state-transition-20261030/atlas")
print(len(atlas.find(consumer_id="regs[6,1]")))  # 38
family = atlas.show("4f1cbe2b89973956f51f5c14274c33f728ce3402ee03146a22415a5b9f7777e1")
print(family["rule_status"])  # not-validated
```

The final-MLP candidate retains negative RSI-margin effects in both windows and
the different hard-output responses: AND changes RDI261 to263 while OR remains3.
Its counterexample metadata preserves the boundary on a proposed hard-output
preservation pattern. Native OR/AND ISA errors are also retained separately from
equation prediction accuracy. Sequence observations carry their actual
own-state continuation chains and explicit host segment boundaries.

The [portable result index](../qualification/state-transition-20261030.json)
embeds the atlas index, selections and source hashes. The
[materializer](../qualification/publish_state_transition_atlas.py) imports
verified, explicitly selected audit scalars. It does not automatically alias
different law IDs across runs, discover groups, induce a rule or expand
applicability. Matched-input and divergent-trajectory effects have distinct
contrast IDs.

## Import and query

The [input-generalization follow-up](INPUT-GENERALIZATION.md) builds a separate
atlas with 42 further observations and four explicitly selected family candidates.
It retains the RSI-effect sign reversals, incomplete/worsening restorations,
post-hoc ISA mismatches and repeated-input checks. Its per-case conditions include
literal programs and before-states; it does not expand the earlier atlas's scope.

```python
from mdb.computation_atlas import ComputationAtlas

atlas = ComputationAtlas("derived/computation-atlas")
records = atlas.ingest(
    "derived/source-qualified-equations",
    [equation_id],
    provenance={"run_id": "job-example", "artifacts": [audit_reference]},
    signed_effects={equation_id: [{
        "consumer_id": "regs[6,1]",
        "contrast_id": "U-restore-minus-zero",
        "value": signed_logit_change,
        "units": "logit",
        "provenance": [audit_reference],
    }]},
)
selected = atlas.find(
    law_id="register-readout/additive-write-response",
    consumer_id="regs[6,1]",
    applicable_to=explicit_context,
)
```

An artifact reference is `{"path": "...", "sha256": "64 lowercase hex"}`;
`json_pointer` and other metadata may be retained. Artifact references are
declarations supplied by the caller. The atlas verifies the selected source
equation file, its fingerprint, its nested comparison fingerprint, and its
qualification predicates. It does **not** fetch external references or certify
their scientific interpretation. The audit ingester below additionally checks
the actual audit-file SHA and reads explicit scalar pointers.

Imports require exact equality predicates matching the bound model, source
identity, execution profile, and horizon. Existing predicate names are retained:
both `checkpoint` and `model_identity` qualify the model binding. A source
predicate may contain the complete bound SHA256 map, its sorted JSON SHA256
with compact separators, or its historical sorted JSON SHA256 with standard
separators. A directly bound SHA256 source identity is also supported. A digest
must match the binding; a descriptive version label is insufficient.

The complete tested context must satisfy **all** original predicates. Queries
use those original predicates, including their spelling and source digest.
Missing fields do not match. A changed source, backend, horizon, or input
condition does not inherit applicability from a nearby observation. Matching
is a supplied predicate result, not empirical validation at a new input.

Supplements are mappings keyed by the selected equation IDs. A signed effect
requires a declared `consumer_id`, `contrast_id`, finite numeric `value`,
`units`, and artifact `provenance`. Continuation metadata requires `status` and
provenance; omitted continuation becomes `not-measured`. Counterexamples remain
a list with `claim_id`, `status`, provenance, and any retained details. Failed
and partial futures can be represented without changing other rows' outcomes.
No per-consumer scalar is inferred from a mixed target-bank mean.

## Compare a proposed response family

```python
family = atlas.compare(
    selected_atlas_ids,
    consumer_id="regs[6,1]",
    contrast_id="U-restore-minus-zero",
)
```

The caller selects the family. Every selected observation must declare the same
law, route, and consumer; measured effects must use the same units. Exact model,
full source identity, execution, and horizon bindings define separate cohorts.
The output retains every row, its signed effect or `not-measured` status,
tested context, continuation, counterexamples, and cohort. Cohort summaries
include raw signed values, sign counts, minimum, maximum, and mean. Values from
different cohorts are not pooled. Counter-trends do not fail a scientific gate.

Two or more measured records yield `family-candidate`; one yields `observation`;
none yields `evidence-incomplete`. Every result remains `not-validated`.
Distinct records or run labels do not establish independent specimens. Selecting
multiple annotated views of the **same source equation record** is refused to
prevent duplicate counting. Reimporting an identical selection is idempotent;
changed annotations create another immutable view, never overwrite history.

Automatic response grouping, transferable rule induction, mechanism discovery,
prospective validation, and rule promotion remain open. This slice provides
source-qualified retrieval and explicit comparisons on which those tools can
build. Caller-supplied law/route labels have no new semantic authority.

## Audit-pointer ingester and standalone CLI

The maintainer helper accepts explicit audit and selection paths, with no
sibling imports or model dependencies:

```sh
python qualification/ingest_computation_atlas.py \
  --atlas derived/atlas --catalog derived/source-qualified-equations \
  --audit saved-analysis.json --selection atlas-selection.json
```

The selection contains `run_id`, `audit_sha256`, and `records`. Each record has
`equation_id` and an optional `effects` list whose elements supply `consumer_id`,
`contrast_id`, `units`, and `value_pointer` (an RFC6901 JSON pointer to a scalar).
Optional `continuation` supplies `status` and `pointer`. Optional
`counterexamples` supplies entries with `claim_id`, `status`, and `pointer`.
The ingester preserves selected payloads and binds both audit and selection
file hashes. It performs no numerical analysis or label inference.

The runtime also exposes `python -m mdb.computation_atlas --atlas DIR` with
`ingest`, `find`, `show`, and `compare` subcommands. `ingest --metadata FILE`
accepts the Python API's provenance and supplement mappings as finite JSON.
Importing and using this metadata API requires neither NumPy nor a model
framework. Each ingestion/comparison accepts at most 256 records, and imported
equation files are limited to 8 MiB. These are metadata integrity bounds.
