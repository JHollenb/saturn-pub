# Typed knowledge-bank/operator interchange

`mdb.knowledge_interchange` is a model-free, finite mechanics harness for the
typed-bank/operator interchange request.  It is deliberately importable in an
installed MDB wheel without importing `saturn`, `saturn_pub`, Torch, mrun, or a
model backend.

The surface separates three things that are easy to accidentally merge:

1. A logical address (`subject`, `relation`, and ordered/scope qualifiers).
2. An immutable value or context record at that address.
3. A recipient-local operator that lowers the selected record into a declared
   recipient state and calls a declared consumer.

The lookup key is `LogicalAddress`, not the answer.  `LogicalTuple` is the
full claim tuple and is used when creating a `KnowledgeRecord`; its object is
ignored when a caller passes it as a lookup for compatibility.  Prefer
`KnowledgeQuery(LogicalAddress(...))` or `LogicalAddress(...)` for new code.
Two records with the same address and different values therefore produce
`ambiguous`; a caller cannot make the lookup appear successful by supplying a
desired answer.

## Minimal API

```python
from mdb.knowledge_interchange import (
    ArtifactBinding, BindingExpectation, InterchangePanel, KnowledgeBank,
    KnowledgeOperator, KnowledgeRecord, LogicalAddress, LogicalTuple,
    OperatorContract, RecordProvenance, VALUE,
)

source = ArtifactBinding("fixture-source", "model-abi-v1", "bank-a-payload", "r1")
bank = KnowledgeBank(
    "bankA",
    source,
    [KnowledgeRecord(
        LogicalTuple("Paris", "country", "France"),
        kind=VALUE,
        revision="r1",
        provenance=RecordProvenance(
            source_id="fixture-source", model_abi="model-abi-v1",
            payload_digest="bank-a-payload", source_revision="r1",
        ),
    )],
)

contract = OperatorContract(
    operator_id="compiled",
    mode="compiled",
    binding=ArtifactBinding("fixture-source", "model-abi-v1", "operator-payload"),
    bank_binding=BindingExpectation(source_id="fixture-source", model_abi="model-abi-v1"),
    recipient_id="recipient",
    lowerer_id="compiled-lowerer",
)
operator = KnowledgeOperator(contract, lambda request: request.value)
panel = InterchangePanel({"bankA": bank}, {"compiled": operator})
receipt = panel.run_matrix(LogicalAddress("Paris", "country"), recipient="recipient")
assert receipt.rows[0].status == "ok"
```

The main types are:

* `ArtifactBinding` binds a source id, model ABI, payload digest, and revision.
  Bank provenance carries the same fields.  A record's
  `value_content_digest` is computed from its actual value/context payload and
  is independent of the declared source payload digest.
* `LogicalAddress` is the value-free lookup key. `LogicalTuple` is the full
  subject/relation/object/qualifier claim. `KnowledgeRecord` adds `kind`
  (`VALUE` or `CONTEXT`), revision, provenance, and record mass.
* `KnowledgeBank.resolve` returns a `Resolution` with a typed status and
  `PartitionAccounting` containing selected and omitted record ids and mass.
  It reports `missing`, `ambiguous`, `stale`, and `context-as-value` distinctly.
* `OperatorContract` declares the operator mode (`native` or `compiled`), its
  own ABI/payload binding, the accepted bank binding, recipient, state reads and
  writes, lowerer id, consumer id, and continuation horizon.
* `RecipientLowerer`/`RecipientLowering` expose only a `LoweringRequest`.
  `request.state` is a read-only declared-state view; undeclared keys raise
  `RuntimeBlockedError`.  This is an interface admission check, not a Python
  sandbox: a callback's closure or global imports remain outside the proof
  boundary and must be audited separately.
* `KnowledgeOperator.execute` returns an `OperatorExecution` for every row.
  It never turns a blocked arm into a panel-level exception.  The execution
  retains query-local merge, SPF-origin transport metadata, optional numeric
  mixing, optional exact-law metadata, address/page/operator/commit/consumer
  identity, continuation trace, and the consumer result.
* `PanelReceipt` stores every `RowReceipt` from the Bank-A/Bank-B ×
  native/compiled cross.  It reports separate `mechanics_status`,
  `instrument_status`, `trend_status`, and `terminal_status` fields.  There is
  no `all-gates` science boolean and no panel-level claim that all rows are
  equivalent.

## Branches and continuation

`InterchangePanel.common_parent()`, `fork()`, and `restore()` provide a small
copy-on-write Python closure for model-free mechanics.  A `BranchCut` is a
content-digested state snapshot; sibling updates cannot mutate the parent or
one another.  This is not automatically a Model OS `StateCut`.  If a caller
already owns a compatible `NativeResearchSession`, it may pass a
`NativeResearchSessionBridge`, which calls that session's public `capture` and
`restore` methods.  The caller remains responsible for supplying the session's
declared tensor state and native identity; this module does not import or create
the session.

`LoweredResult.own_predicted_state` and the `supplied_execution` argument are
always recorded separately.  When supplied execution is present, the consumer
uses it and the receipt says `execution_state_source="supplied-execution"`.
Otherwise it uses the operator's own predicted state, when present.  A
continuation callback can append a trace.  `InterchangePanel.run_chain` adds a
bounded read loop when the experiment supplies an explicit `next_query` rule:
the rule receives only the previous operator's own predicted state, derives a
value-free next address, and the next bank lookup/operator/consumer step gets
its own receipt.  Supplied execution is never used to invent that address.
Missing, ambiguous, stale, and wrong-route next reads stop the chain while
retaining the terminal row.  A short finite chain is still a mechanics result;
it does not establish open-ended recursive continuation.

## Query-local merge, SPF, and exact laws

`QueryLocalMerge` and `PartitionAccounting` are selection records only.  They
do not implement or imply SPF routing.  `SPFOriginTransport` is a separate
metadata plane for origin, route, and page identity.  `NumericMixing` is a
separate declaration for a numeric operation, while `ExactLaw` identifies a
distinct exact operation.  The finite panel records these fields when supplied
by an operator, but does not implement SPF transport, numeric mixing, or exact
law execution.  This keeps a future SPF backend from smuggling a route bucket
into semantic lookup or treating approximate numeric mixing as an exact law.

When a route candidate is available, pass its explicit record id(s) as
`candidate_record_ids`.  `KnowledgeBank.resolve_candidates` verifies each
candidate's complete logical address before loading its payload and emits
`wrong-route` when the route points at a different address.  This is a
route-control diagnostic, not semantic route authority.

Claim extraction is similarly per-arm: pass `claim_extractor` to
`run_matrix` or the operator.  Extracted claims remain on the individual
`OperatorExecution.claims` and are included in neither a gate nor a summary
score.

`ContentSignedConsumerResult` (also exported as the compatibility alias
`SignedConsumerResult`) uses a deterministic SHA-256 content signature over the
consumer payload and signer id.  It is a reproducibility checksum, not a
cryptographic authenticity signature or an authorization mechanism.

The focused tests cover conflicting values, disjoint banks, missing and stale
records, wrong-bank and wrong-operator bindings, context-as-value, recipient
mismatch, runtime-blocked reads, undeclared state writes, branch isolation,
continuation source separation, selected/omitted mass, claim extraction, and
per-row receipt custody.  Model-backed qualification and performance are
outside this package's mechanics panel.
