# Qualified symbols and branch-specific port slices

`project slice` joins a retained native cut and a supported symbol binding to a
declared dataflow DAG. It reuses Saturn-pub's support checks and MDB's `StateGraph`;
it creates no new executable state or mutation authority. Saturn-pub 0.4.1 adds
the shared metadata support API, and MDB now requires that version.

In a live `mdb public` shell, stop where the carrier exists and name it:

```text
step 2
project symbol-bind carrier --port hidden --context example --context-slot tokens --context-slot hidden --claim "Supplied carrier at this boundary"
project slice --symbol carrier
```

`symbol-bind` uses the original native `SymbolBinding.bind`, including live reads
and exact clock/schema checks. `--port` and `--context-slot` may repeat. A JSON
integer `--selector '[0,0,0]'` can select a coordinate from one port; `--role` can
require the native port role. The default evidence is the current retained cut.
Names are supplied interpretations; binding one does not discover or validate a
semantic mechanism. Native support status records successful binding, not a
causal result.

The slice returns a stable `slice_id`, the full symbol binding, target port
versions, nearest declared writers, the graph snapshot, retained cut/receipt
references, and explicit source/frontier nodes. Outside a live session:

```sh
mdb project ./investigation slice --symbol carrier --cut CUT_ID --context example
mdb project ./investigation slice --port hidden --cut CUT_ID
mdb project ./investigation slice-show SLICE_ID
mdb project ./investigation finding symbol:carrier --cut CUT_ID \
  --title 'Carrier observed' --kind observation --slice-id SLICE_ID
```

The attached shell accepts the same project commands without the directory or
session option. Offline commands infer the session only when exactly one exists;
otherwise supply `--session-id`. `slice symbol:carrier` and `slice port:hidden`
are shorthand forms. `--cut-id` remains an alias of `--cut`.

Offline project commands return JSON on stdout when successful. Expected
refusals (including unsupported bindings, changed seals, and unreadable project
storage) print `mdb project: REASON` to stderr and exit 1, leaving stdout empty.
Argument syntax errors retain argparse's exit code 2.

## Qualification and graph scope

Offline resolution checks the sealed binding against the retained cut's model,
execution, input-context digests, clock, schema, role, and selector bounds. The
result is labelled `metadata-support-verified`. It does not load tensor pages,
check live adapter availability, execute the model, or confer read/write authority.
Actual native reads still pass the original Session and symbol resolver. Both
paths share the model/execution/context support checks in Saturn-pub.

Bindings are archived append-only in project schema 3, including bindings removed
from the current session's symbol table. Original symbol, receipt, annotation and
cut identities are preserved. Multiple supported interpretations under the same
name require explicit `--binding FINGERPRINT`; stale or unsupported bindings are
refused. Component listings check full metadata support before describing a cut
as supporting a symbol. Structural cross-references remain the broader whole-slot
index across recorded transactions.

Each value is qualified by native cut and slot, or by receipt-local transition
boundary for a multi-transition receipt. Intermediate versions are descriptive
nodes and are never presented as retained checkpoints. A selector remains in the
symbol binding, but dependency edges cover whole slots; there is no claim of
coordinate-specific influence. The implemented `--view` is `declared` only.

Backward expansion follows only the selected cut's producer, verified commit
candidate copies, native fork/capture lineage with identical state, and recorded
preservation. Restore and abort select state rather than compute new values.
Refused operations and unrelated sibling futures are excluded. Same-content
replays can share logical cut identities, so this is logical branch ancestry, not
an identification of physical executions.

Declared multi-transition `continue` receipts expand into distinct operation
occurrences and virtual value boundaries. Unknown footprints, missing cuts,
unsupported transaction kinds (including coarse `step` envelopes), and ambiguous
producer receipts stop expansion at explicit frontiers. `complete=false` means
the recorded explanation has gaps, not that an effect is absent. A complete slice
is a declared explanation within recorded scope, not semantic causal proof.
Conflicting copies, receipt execution bindings or discontinuous transition clocks
are integrity errors.

If several receipts produce the same cut, `--receipt RECEIPT_ID` selects the
producer for that requested cut. Without it the result exposes ambiguity rather
than merging explanations. Ambiguities deeper in history remain frontiers.

## Downstream consumers require a bounded future

To inspect consumers of an earlier value, choose an explicit destination:

```sh
mdb project ./investigation slice --symbol carrier --cut SOURCE_CUT \
  --context example --to-cut FUTURE_CUT
```

The future must have the source cut in its logical branch history. This view
follows copies and declared reads toward that destination, includes the side
inputs needed to describe each selected action, and distinguishes them as
`context_inputs`. It reports nearest consumers separately from transitive
downstream operations. A divergent sibling is refused. Unknown portions of the
chosen future remain visible as gaps even when they prevent a downstream path
from being established.

The default upstream slice has no implied future. An empty consumer list there
does not mean that the value has no consumers.

## Saved investigation links

Each slice is persisted as a sealed, immutable result using existing graph
snapshots. `slice-show` returns that exact result even after the session advances.
`Project.slice_record` verifies the slice and graph seals. Findings created with
`--slice-id` retain the slice ID, exact symbol binding and completeness status;
the finding must refer to the same component and source cut. Its evidence may
include the slice's verified source cut and receipt references. Observation,
inference and causal-claim labels remain descriptive and terminally unestablished.

The main project graph connects the symbol source to the saved slice and its
native cut, and relates a finding to its slice. Full port-version graphs remain
independently queryable snapshots rather than being flattened into every query.

Python exposes the same operations:

```python
sliced = project.slice("trial", symbol="carrier", cut_id=cut,
                       context="example", to_cut=future_cut)
assert project.slice_record("trial", sliced["slice_id"]) == sliced
project.finding("trial", "symbol:carrier", "Observed carrier",
                cut_id=cut, slice_id=sliced["slice_id"])
```

No graph database, model framework or private workspace package is needed for
offline use. Native state bytes still belong to Saturn-pub's `LocalStore`.
