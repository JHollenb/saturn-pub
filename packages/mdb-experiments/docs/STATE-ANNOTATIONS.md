# State annotations

MDB state annotations give immutable Bench cuts durable, searchable human context without changing
the cut, selected session manifest, CAS object, numerical contract, replay identity, or Hyperion
compiler bytes. An annotation is descriptive evidence. Its label, role, or metadata does not prove
that a semantic circuit exists and does not grant restore or execution authority.

## Bench API

Create an annotation through the resident session so journal publication stays atomic:

```python
row = bench.annotate_state(
    "recipient after frontier write",
    cut_id=postwrite_cut,
    namespace="frontier-reduction-v1",
    role="recipient-postwrite",
    stage="postwrite",
    scientific_source_cut_ids=[source_child_cut],
    recipient_parent_cut_id=recipient_parent_cut,
    tags=["frontier", "kv"],
    native_act_refs=[memory_act_receipt_ref],
    consumer_refs=["bench.qwen2.native"],
    geometry={
        "components": ["key", "value"],
        "layers": list(range(28)),
        "rows": {"start": 48, "stop": 75},
    },
    details={"arm": "coherent-frontier"},
    producer={"kind": "experiment", "id": "frontier-reduction-v1"},
)
```

MDB derives the session/model binding, immutable parent, native writer clock, and model-input
binding from the registered cut. Callers cannot override them. `geometry` has a strict bounded
schema for components, layers, rows or row ranges, and physical addresses. `details` is bounded
JSON for additional descriptive provenance. Neither field has semantic authority.

The read API is:

```python
bench.list_state_annotations(namespace="frontier-reduction-v1", stage="postwrite")
bench.show_state_annotation(row["qualified_label"])
bench.state_lineage(postwrite_cut)
bench.verify_state_annotations()
```

Human labels may repeat. Every annotation also has a content-derived `annotation_id` and a fully
qualified label of the form `namespace::label@sha256:<digest>`. Looking up an ambiguous plain label
fails and returns the qualified choices. Repeating an identical annotation is idempotent.

## Offline inspection

The same sidecar is available without a model:

```bash
mdb states --root SESSION list --namespace frontier-reduction-v1
mdb states --root SESSION show 'frontier-reduction-v1::recipient after frontier write@sha256:...'
mdb states --root SESSION lineage cas://sha256/...
mdb states --root SESSION verify
```

`list` also accepts `--label`, `--role`, `--stage`, `--cut-id`, `--tag`, and `--query`.
Offline stores are read-only; annotation writes must use the resident `BenchSession` API and its
single-writer lock.

## Storage and recovery

Annotations live under `state-annotations/`:

- `objects/<sha256>.json` is an immutable canonical annotation object.
- `index.json` is a sealed session-bound index over those objects.

Journal sessions stage the object and index in the existing Bench publication transaction and
reselect the same session manifest digest. Legacy sessions write the immutable object first, fsync
it, then atomically replace and fsync the index. An interruption may leave an unreferenced object;
`verify` reports its count, and no incomplete object becomes indexed. Export copies the sidecar,
and reopen reads it independently of the model.

Verification checks the index and object seals, cut/session binding, immutable parent, native
clock, model-input binding, source and recipient cut references, duplicate identities, and unsafe
namespace members. Tampering is a mechanics failure. Absence of annotations is valid and leaves
legacy sessions unchanged.

`native_act_refs` and `consumer_refs` are descriptive links supplied by the producer. The sidecar
seals them but does not resolve or qualify the referenced Act or consumer. Its verification report
therefore states `external_references_verified: false`.
