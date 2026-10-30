# Source candidates and native symbol joins

MDB has two separate steps for source labels and native symbols:

1. `mdb.symbol_sources` records a finite, provenance-bearing candidate from an
   explicit source.
2. `mdb.symbol_producers` joins that record to an existing sealed
   `saturn_pub.SymbolBinding` at an exact retained public cut.

The source reader never chooses a native address. The join never invents one.
The resulting lifecycle can then be read by [`SYMBOL-TRACE.md`](SYMBOL-TRACE.md),
which keeps binding groups, occurrences, uses, native cut ancestry, receipts,
and producer provenance in one read-only trace.

## Source readers

First create a sealed vocabulary. Labels are supplied by the caller and are
kept in their declared order:

```python
from mdb.symbol_sources import make_vocabulary, nominate_scores

vocabulary = make_vocabulary(
    ["value", "other"],
    namespace="counter",
    evidence_refs=["fixture://vocabulary"],
)
candidate = nominate_scores(
    [0.5, 0.0],
    vocabulary,
    model_identity="public-counter-v1",
    execution=execution_digest,
    source_cut=source_cut,
    context="producer-context",
    evidence_refs=["cas://sha256/source-report"],
    producer_id="decoder-readout",
    top_k=1,
)
```

The readers are:

- `nominate_prompt` records literal vocabulary-token matches from supplied
  parser tokens.
- `nominate_scores` records finite scores in the frozen vocabulary order,
  retaining logits, probabilities, ties, and the selected label.
- `nominate_history` records ordered model-predicted decisions with native
  source-cut and receipt references. Each event has `commit_status` of
  `committed`, `rejected`, or `proposed`. All events remain in the record, and
  only committed events can select a label.

Every candidate is sealed as `mdb-symbol-source-candidate-v1`. Its identity
includes the model identity, execution digest, source cut, context, vocabulary
fingerprint, producer ID, source payload digest, proposals, and evidence
references. It carries `semantic_qualified: false`,
`causal_dependency_tested: false`, and an explicit
`source_trust` value stating that native-witness verification is separate.
`verify_candidate` recomputes the proposals and selection from the source, so
changing a winner and merely re-signing the JSON is rejected. Source readers
also reject evaluator fields such as expected answers, gold labels, teachers,
and oracles.

The source kinds are `prompt`, `residual`, and `generated-history`. An abstained
or unknown candidate remains a valid source observation with its proposals and
reason; it has no selected native symbol and therefore cannot be attached to a
binding. Preserve it as source evidence until a later native binding exists.

## Joining a producer to native execution

`join_symbol_producers` accepts one or more producer records and requires an
already registered exact binding:

```python
from mdb.symbol_producers import join_symbol_producers

saved = join_symbol_producers(
    project,
    session_id,
    registered_binding,
    {
        "producer": candidate,
        "source_cut": source_cut,
        "binding_context": "producer-context",
    },
)
```

The binding must already be present in the public symbol registry for that
project session. A normal public-adapter registration is:

```python
binding = SymbolBinding.bind(
    adapter.session,
    "counter.value",
    (StateAddress("value"),),
    context="producer-context",
    context_slots=(),
    evidence=(f"cas://sha256/{source_cut}",),
    claim="Supplied native location; source labels remain descriptive.",
)
adapter.symbols = SymbolTable((binding,))
adapter.symbol_context = binding.context
adapter._persist()
```

The join checks the binding seal and registry fingerprint, then checks every
declared producer model identity, execution digest, context, and source cut.
`source_cut`, `parent_cut`, and `result_cut` must be retained public project
cuts. A Bench cut or an unrelated runtime identifier cannot be substituted.
Native uses may be supplied through `uses=`; their retained receipts must close
the declared parent and result cuts.

Each lifecycle occurrence retains the original producer record under
`metadata.producer_join.raw_record`, along with its schema, fingerprint,
verification result, statuses, and evidence references. Envelope fields remain
separate from that raw record. Loading the lifecycle or tracing the symbol
does not load tensors or rewrite the producer bytes.

The adapter also accepts sealed Hyperion reports, `SemanticBinding` records,
native observation reports, CoT register observations, and future ISA source
candidates. Hyperion and SemanticBinding records use their existing seal
verifiers. Native observation reports can be checked against a supplied
artifact store. Candidate source records are checked with
`mdb.symbol_sources.verify_candidate`; their selected label must match the
provided native binding exactly.

## CoT observations

The accepted CoT observation schema is
`mdb-cot-register-observation-v1`. A record must retain `config_ref`,
`observation_ref`, `source_cut`, `lens_ref`, and `session_id`; use the
`decoded-candidates` interpretation; and keep
`causal_dependency_tested`, `native_mutation`, and
`semantic_cot_qualified` explicitly false. The Python join can perform the
full artifact check when both `producer_artifacts` and `producer_manifest` are
provided:

```python
join_symbol_producers(
    project,
    session_id,
    registered_binding,
    {"producer": cot_observation, "source_cut": source_cut},
    producer_artifacts=artifacts,
    producer_manifest=manifest,
)
```

Without those retained verifier inputs, the join keeps the observation as
`schema-verified-fields-only` with `verification_status: evidence-incomplete`.
CoT references remain locator evidence. A decoded label does not become a
semantic identity or a causal dependency.

## Clock, spectral, and VM metadata

An explicit producer envelope may carry typed metadata beside the producer:

```json
{
  "producer": {"schema": "..."},
  "source_cut": "RETAINED_CUT",
  "clock": {"schema": "mdb-symbol-clock-witness-v1"},
  "spectral_address": {"schema": "saturn-spectral-address-abi-v1"},
  "vm_bundle": {"schema": "saturn-vm-cache-execution-bundle-v1"}
}
```

The adapter passes `clock`, `spectral_address`, and `vm_bundle` through to the
sealed occurrence. The existing project lifecycle validator checks native
model/execution identity, exact source-cut references, clock readings and
metadata fields, spectral symbol identity, and VM model/ABI/input identity.
Opaque references remain explicitly unjoined. These fields describe supplied
evidence; they do not grant payload access or semantic authority.

## Authority boundaries

| Evidence step | Establishes | Leaves open |
| --- | --- | --- |
| `source-recomputed` | The candidate's proposals and selection agree with its sealed source. | Native consumption, semantic equivalence, and causal necessity. |
| `artifact-verified` | The supplied producer artifact matches its retained verifier inputs. | Semantic or causal interpretation. |
| Native lifecycle join | Exact registered binding, model/execution/context, retained cuts, locations, receipts, and typed metadata references. | Learned meaning, semantic equivalence, and write authority. |
| Symbol trace | Read-only provenance, immutable lifecycle snapshots, native chronology, and exact source/binding use links. | Semantic identity and causal effect; both remain `not-established`. |

`source-recomputed` may make the lifecycle instrument status mechanically
`valid`, while `source_trust` still records that native-witness verification is
separate. CoT schema-only records remain evidence-incomplete. A producer's
`validated` or `candidate` status is retained as provenance and is never
converted into a semantic pass.

## CLI

The module CLI resolves the binding by its exact registered fingerprint and
accepts one or more producer JSON files:

```sh
python -m mdb.symbol_producers PROJECT_ROOT \
  --binding BINDING_FINGERPRINT \
  --session-id SESSION_ID \
  PRODUCER_ENVELOPE.json [ANOTHER_PRODUCER.json ...] \
  [--uses USES.json]
```

When a project has exactly one session, `--session-id` may be omitted. A
producer JSON file contains either the producer record itself or an envelope
with `producer`, `source_cut`, optional `parent_cut`/`result_cut`,
`binding_context`, and the typed metadata fields above. `--uses` accepts one
native-use object or an array of them. The command prints the sealed lifecycle
join; it does not resolve a symbol name into an address.

The saved lifecycle can be inspected with `mdb symbol-trace` or the project
`symbol-history` command. See [`SYMBOL-TRACE.md`](SYMBOL-TRACE.md) for ancestry,
branch filtering, slice joins, and trace verification.
