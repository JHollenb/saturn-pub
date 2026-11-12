# Computation records and future comparison

`mdb view` opens saved research evidence as a portable interactive HTML file.
Select a generic computation ID, follow its recorded branch states and compare
the native reference, hold continuation and frozen prediction. The exported
page embeds its PNGs and needs no server, network, weights or model framework.

```sh
mdb view RUN/compact \
  --records COMPUTATIONS.jsonl \
  --output computation.html \
  --json-output computation.json
```

The initial measured adapter reads `mdb-explorer-binding-bridge-v1` reports,
including the [frozen Explorer/MDB binding experiment](FLUX-FROZEN-READER-BRIDGE.md).
The same command routes supported token/layer measurements and traces through
the existing [temporal causal viewer](TEMPORAL-CAUSAL-VIEWER.md).

## What to inspect

- **Futures:** choose a computation record, context and branch. Original retained
  native, hold and candidate PNGs appear side by side. The stage buttons select
  velocity, final latent, raw decoder or pixel measurements. They do not execute
  another model or change the displayed image into an activation visualization.
- **Pixel differences:** show absolute RGB differences from the native image at
  a fixed display gain of four. Display clipping does not change recorded metrics.
- **Effects:** switch between complete endpoint error, arm-minus-baseline change
  and the full finite `RW - R - W + Q` interaction. Native and method RMS
  magnitudes remain beside error and cosine. A cosine of 0.99 is alignment,
  not 99% accuracy; amplitude and visible image damage can still disagree.
- **Record and evidence:** read applicability, counterexamples, the computation
  account, actual work and verification scopes. Inspect the source StateCut,
  previous-velocity cut, execution point and tensor content/layout descriptors.
- **Selected-case JSON:** inspect and copy state addresses, metric records,
  image hashes and linked computations. The CLI's normalized JSON export also
  includes the portable image data. No server-side state is changed.

Arrow keys move between the current context's branches when an input is not
focused. Search matches supplied record metadata; an exact generic ID is the
most precise lookup. Clear the record selection to browse every recorded branch.

## How this belongs with the symbol table

Three related units should remain visible together:

1. A **computation record** gives a generic ID, mathematical account, tested
   applicability and counterexamples.
2. An **occurrence** binds that record to a model, execution context, current
   state/address, inputs and receiving computation.
3. A **future** records what was predicted or executed from that occurrence,
   which consumer ran, its continuation and the observed differences.

One computation can have several occurrences and futures. A route or carrier
can have different effects in different receiving states. This view makes those
relationships inspectable without turning the table into a universal dictionary
of objects or operations.

For the current curated-record adapter, a branch link requires an exact request
digest and model identity plus explicitly listed occurrence case IDs in
`results[].endpoint_rows[].case`. Case names, family membership, semantic labels
or a similar-looking image do not establish a link. Mismatched records remain
browsable as **unlinked** rather than silently inheriting applicability.

These are `known-computations.v1` curated observations. Viewing them does not
create an EquationCatalog qualification, a ComputationAtlas promotion or a
writable runtime `SymbolBinding`. The existing lifecycle resolver still requires
its full model/execution/point/schema/context checks before a native write.

## Evidence and artifact handling

The viewer checks the local report against its adjacent collection inventory,
when available, and verifies every embedded PNG against its declared hash and
byte count. It maps absolute image paths only through the collection's declared
original root. It never guesses an image by searching for the same filename.

For an explicitly relocated raw directory, supply its original producer root:

```sh
mdb view COPIED_RAW_DIRECTORY \
  --source-root /original/worker/root \
  --records COMPUTATIONS.jsonl -o computation.html
```

Missing images stay visible as unavailable; they do not erase completed numeric
evidence. A present image with a mismatched hash, unsafe path, duplicate comparison
or inconsistent comparison orientation is refused as an instrument/integrity
error. No image score, cosine or scientific effect threshold controls export.

Local forecast transactions are verified against the report's supplied seal and
event head. External prediction-array verification is displayed separately as a
collector report; the viewer does not reread remote tensor arrays. Source-reference
and native-witness assertions retain their stated scope. The HTML is a derived
view, not a replacement for original receipts, arrays, custody or independent audits.

## Public API and limits

```python
from mdb.evidence_viewer import export_view

bundle = export_view(
    ["run/compact"], "computation.html",
    records=["computation-records.jsonl"],
    json_output="computation.json",
)
```

The sealed normalized format is `mdb-computation-evidence-view-v1`. It keeps
binding, cases, futures, records, comparisons, declared stages, verification
scopes and provenance. It can be rendered again without reopening the original
run. An additional model/report needs an explicit adapter; the initial FLUX
adapter does not imply arbitrary experiment ingestion.

This is an offline research view. It provides no live step, jump, fork, intervention
or breakpoint controls. A future live connection should resolve a selected
occurrence through the existing MDB session and symbol APIs, preserving their
authority checks. Raw-prompt program discovery and automatic symbol-family
induction remain separate research work.
