"""Finite mechanics checks for the typed knowledge interchange surface."""

from __future__ import annotations

from mdb.knowledge_interchange import (
    AMBIGUOUS,
    CONTEXT,
    CONTEXT_AS_VALUE,
    MISSING,
    OK,
    RECIPIENT_MISMATCH,
    RUNTIME_BLOCKED,
    STALE,
    STATE_LEAKAGE,
    ArtifactBinding,
    BindingExpectation,
    ChainReceipt,
    ExactLaw,
    InterchangePanel,
    KnowledgeBank,
    KnowledgeOperator,
    KnowledgeRecord,
    LogicalAddress,
    LogicalTuple,
    LoweredResult,
    NativeResearchSessionBridge,
    NumericMixing,
    OperatorContract,
    RecordProvenance,
    SPFOriginTransport,
    VALUE,
    WRONG_BANK,
    WRONG_OPERATOR,
    WRONG_ROUTE,
    build_finite_panel,
    content_digest,
)


SOURCE = "fixture-source"
ABI = "fixture-model-abi-v1"
QUERY = LogicalAddress("Paris", "country")


def binding(payload: str, revision: str = "current") -> ArtifactBinding:
    return ArtifactBinding(SOURCE, ABI, payload, revision)


def record(value: str, *, payload: str, kind: str = VALUE, revision: str = "r1") -> KnowledgeRecord:
    return KnowledgeRecord(
        LogicalTuple("Paris", "country", value),
        kind=kind,
        revision=revision,
        provenance=RecordProvenance(
            source_id=SOURCE,
            model_abi=ABI,
            payload_digest=payload,
            source_revision=revision,
            locator=f"fixture://{value}",
        ),
    )


def bank(name: str, payload: str, value: str) -> KnowledgeBank:
    return KnowledgeBank(name, binding(payload), [record(value, payload=payload)])


def operator(
    name: str,
    mode: str,
    payload: str,
    *,
    fn=None,
    state_reads=(),
    state_writes=("output",),
    continuation=None,
    horizon=0,
    recipient="denoiser",
):
    contract = OperatorContract(
        operator_id=name,
        mode=mode,
        binding=binding(payload),
        bank_binding=BindingExpectation(source_id=SOURCE, model_abi=ABI),
        recipient_id=recipient,
        state_reads=state_reads,
        state_writes=state_writes,
        lowerer_id=f"{name}-lowerer",
        consumer_id=f"{name}-consumer",
        continuation_horizon=horizon,
        numeric_mixing=NumericMixing("query-local-demo", terms=("value",), exact=False),
        exact_law=ExactLaw("fixture-copy-law"),
    )
    return KnowledgeOperator(contract, fn or (lambda request: request.value), continuation=continuation)


def test_bank_a_b_cross_native_compiled_and_receipt_fields():
    native = operator("native", "native", "native-payload", fn=lambda request: request.value)
    compiled = operator("compiled", "compiled", "compiled-payload", fn=lambda request: request.value)
    panel = build_finite_panel(
        bank_a=bank("bankA", "bank-a-payload", "France"),
        bank_b=bank("bankB", "bank-b-payload", "Belgium"),
        native_operator=native,
        compiled_operator=compiled,
        initial_state={"seed": "neutral"},
        panel_id="fixture-panel",
    )
    receipt = panel.run_matrix(
        QUERY,
        recipient="denoiser",
        transport=SPFOriginTransport(origin=7, route="secondary", page_id="page-3", status="recorded"),
        commit=True,
        claim_extractor=lambda observation: ({"claim": observation},),
    )
    assert len(receipt.rows) == 4
    assert [row.status for row in receipt.rows] == [OK, OK, OK, OK]
    assert receipt.mechanics_status == "valid"
    assert receipt.instrument_status == "observed"
    assert receipt.terminal_status == "not-requested"
    assert {row.execution.output for row in receipt.rows} == {"France", "Belgium"}
    for row in receipt.rows:
        execution = row.execution
        assert execution.signed_consumer_result is not None
        assert execution.signed_consumer_result.signature
        assert execution.claims == ({"claim": row.execution.output},)
        assert execution.arm is not None
        assert execution.arm.address == row.execution.arm.address
        assert execution.arm.page_id == "page-3" or execution.arm.page_id.startswith("page-")
        assert execution.query_local_merge is not None
        assert execution.spf_origin_transport.status == "recorded"
        # Numeric mixing and exact law are explicit, independent receipt planes.
        assert execution.numeric_mixing is not None
        assert execution.exact_law is not None
    assert receipt.receipt_sha256 == receipt.to_payload()["receipt_sha256"]


def test_value_context_missing_ambiguous_and_stale_are_distinct():
    value_bank = KnowledgeBank("bank", binding("bank-payload"))
    value_bank.add(record("France", payload="bank-payload"))
    value_bank.add(record("France", payload="bank-payload", revision="r2"))
    assert value_bank.resolve(QUERY).status == AMBIGUOUS
    assert value_bank.resolve(QUERY, revision="r3").status == STALE
    assert value_bank.resolve(LogicalTuple("Rome", "country", "Italy")).status == MISSING

    context_bank = KnowledgeBank(
        "context-bank",
        binding("context-payload"),
        [record("France", payload="context-payload", kind=CONTEXT)],
    )
    assert context_bank.resolve(QUERY, kind=VALUE).status == CONTEXT_AS_VALUE
    assert context_bank.resolve(QUERY, kind=CONTEXT).status == "resolved"


def test_binding_wrong_bank_and_stale_payload_block_before_lowerer():
    calls = []
    op = operator("compiled", "compiled", "operator-payload", fn=lambda request: calls.append(1))
    foreign = KnowledgeBank(
        "foreign",
        ArtifactBinding("different-source", ABI, "foreign-payload", "r1"),
        [
            KnowledgeRecord(
                LogicalTuple("Paris", "country", "France"),
                VALUE,
                "r1",
                RecordProvenance("different-source", ABI, "foreign-payload", "r1"),
            )
        ],
    )
    result = op.execute(foreign, QUERY)
    assert result.status == WRONG_BANK
    assert calls == []

    stale = KnowledgeBank(
        "stale",
        ArtifactBinding(SOURCE, ABI, "stale-payload", "r1"),
        [record("France", payload="stale-payload")],
    )
    strict = KnowledgeOperator(
        OperatorContract(
            "strict",
            "compiled",
            binding("strict-op"),
            recipient_id="denoiser",
            bank_binding=BindingExpectation(
                source_id=SOURCE, model_abi=ABI, payload_digest="fresh-payload"
            ),
            lowerer_id="strict-lowerer",
        ),
        lambda request: request.value,
    )
    assert strict.execute(stale, QUERY).status == STALE


def test_wrong_operator_and_recipient_mismatch_are_per_row_statuses():
    op = operator("compiled", "compiled", "operator-payload")
    fixture = bank("bank", "bank-payload", "France")
    wrong_id = op.execute(fixture, QUERY, expected_operator_id="native")
    assert wrong_id.status == WRONG_OPERATOR
    wrong_recipient = op.execute(fixture, QUERY, expected_recipient="other-recipient")
    assert wrong_recipient.status == RECIPIENT_MISMATCH


def test_runtime_blocked_access_and_state_leakage_are_not_silent():
    fixture = bank("bank", "bank-payload", "France")
    blocked = operator(
        "blocked",
        "compiled",
        "operator-payload",
        fn=lambda request: request.state["secret"],
        state_reads=(),
    )
    blocked_result = blocked.execute(fixture, QUERY, state={"secret": 99})
    assert blocked_result.status == RUNTIME_BLOCKED

    leaking = operator(
        "leaking",
        "compiled",
        "operator-payload",
        fn=lambda request: LoweredResult(output=request.value, state_updates={"secret": 1}),
        state_writes=("output",),
    )
    leaking_result = leaking.execute(fixture, QUERY, state={"secret": 99})
    assert leaking_result.status == STATE_LEAKAGE


def test_continuation_keeps_own_prediction_and_supplied_execution_separate():
    fixture = bank("bank", "bank-payload", "France")

    def continuation(consumer_input):
        return LoweredResult(
            output=consumer_input.lowered_output,
            own_predicted_state={"next": "predicted"},
            continuation_trace=({"step": 1, "source": "own"},),
        )

    op = operator(
        "native",
        "native",
        "operator-payload",
        continuation=continuation,
        horizon=1,
    )
    result = op.execute(fixture, QUERY, supplied_execution={"next": "supplied"})
    assert result.status == OK
    assert result.own_predicted_state == {"next": "predicted"}
    assert result.supplied_execution == {"next": "supplied"}
    assert result.effective_state == {"next": "supplied"}
    assert result.execution_state_source == "supplied-execution"
    assert result.continuation_trace == ({"step": 1, "source": "own"},)


def test_common_parent_fork_restore_keeps_sibling_state_isolated():
    op = operator("native", "native", "operator-payload")
    panel = InterchangePanel({"A": bank("A", "a", "France")}, {"native": op}, initial_state={"x": 1})
    parent = panel.common_parent()
    left = panel.fork("left", updates={"x": 2})
    right = panel.fork("right", updates={"x": 3})
    assert parent.state == {"x": 1}
    assert left.state == {"x": 2}
    assert right.state == {"x": 3}
    assert panel.restore("left").digest == left.digest
    assert panel.run_matrix(QUERY, cut="right").branch.digest == right.digest


def test_nested_record_payloads_are_defensively_frozen_and_fork_safe():
    nested = {"scope": {"tags": ["a", "b"]}}
    tuple_value = LogicalTuple("entity", "has", nested, qualifiers={"window": [1, 2]})
    provenance_metadata = {"source": {"line": 4}}
    rec = KnowledgeRecord(
        tuple_value,
        VALUE,
        "r1",
        RecordProvenance(SOURCE, ABI, "payload", metadata=provenance_metadata),
        metadata={"labels": ["stable"]},
    )
    before = rec.fingerprint
    nested["scope"]["tags"].append("caller-mutation")
    nested["scope"] = {"tags": ["replaced"]}
    provenance_metadata["source"]["line"] = 99
    assert rec.fingerprint == before
    assert rec.value_content_digest
    assert rec.validate_content()
    object.__setattr__(rec, "value_digest", "tampered")
    assert not rec.validate_content()
    object.__setattr__(
        rec,
        "value_digest",
        content_digest({"kind": rec.kind, "value": rec.value}),
    )
    fork = KnowledgeBank("bank", binding("payload"), [rec]).fork(bank_id="sibling")
    assert fork.records[0].fingerprint == before
    assert tuple_value.qualifiers["window"] == (1, 2)
    nested_op = operator("nested", "native", "operator-payload", fn=lambda request: request.value)
    nested_bank = KnowledgeBank("nested", binding("payload"), [rec])
    nested_result = nested_op.execute(
        nested_bank,
        LogicalAddress("entity", "has", {"window": [1, 2]}),
    )
    assert nested_result.status == OK
    assert nested_result.output == {"scope": {"tags": ["a", "b"]}}


def test_explicit_route_candidates_check_address_before_payload_load():
    good = record("France", payload="bank-payload")
    wrong = KnowledgeRecord(
        LogicalTuple("Paris", "continent", "Europe"),
        VALUE,
        "r1",
        RecordProvenance(SOURCE, ABI, "bank-payload"),
    )
    fixture = KnowledgeBank("bank", binding("bank-payload"), [good, wrong])
    assert fixture.resolve_candidates(QUERY, [wrong.record_id]).status == WRONG_ROUTE
    op = operator("compiled", "compiled", "operator-payload")
    result = op.execute(fixture, QUERY, candidate_record_ids=[wrong.record_id])
    assert result.status == WRONG_ROUTE
    assert result.spf_origin_transport.status == "not-provided"


def test_run_chain_reads_value_free_next_address_from_own_predicted_state():
    first = KnowledgeRecord(
        LogicalTuple("start", "points-to", "middle"),
        VALUE,
        "r1",
        RecordProvenance(SOURCE, ABI, "chain-payload"),
    )
    second = KnowledgeRecord(
        LogicalTuple("middle", "answer", "done"),
        VALUE,
        "r1",
        RecordProvenance(SOURCE, ABI, "chain-payload"),
    )
    fixture = KnowledgeBank("bank", binding("chain-payload"), [first, second])

    def lower(request):
        return LoweredResult(
            output=request.value,
            own_predicted_state={
                "next": {"subject": request.value, "relation": "answer"}
            },
            continuation_trace=({"read": request.logical_tuple.subject},),
        )

    op = operator("native", "native", "chain-operator", fn=lower)
    panel = InterchangePanel({"bank": fixture}, {"native": op})
    seen = []

    def next_query(own_state, execution):
        seen.append((own_state, execution.execution_state_source))
        return own_state["next"]

    receipt = panel.run_chain(
        LogicalAddress("start", "points-to"),
        bank_id="bank",
        operator_ids=("native",),
        next_query=next_query,
        max_steps=2,
    )
    assert isinstance(receipt, ChainReceipt)
    assert receipt.status == OK
    assert len(receipt.steps) == 2
    assert [step.row.execution.output for step in receipt.steps] == ["middle", "done"]
    assert seen == [
        ({"next": {"subject": "middle", "relation": "answer"}}, "own-predicted-state"),
        ({"next": {"subject": "done", "relation": "answer"}}, "own-predicted-state"),
    ]


def test_run_chain_retains_ambiguous_second_read_and_does_not_claim_closure():
    first = KnowledgeRecord(
        LogicalTuple("start", "points-to", "middle"), VALUE, "r1",
        RecordProvenance(SOURCE, ABI, "chain-payload"),
    )
    second_a = KnowledgeRecord(
        LogicalTuple("middle", "answer", "a"), VALUE, "r1",
        RecordProvenance(SOURCE, ABI, "chain-payload"),
    )
    second_b = KnowledgeRecord(
        LogicalTuple("middle", "answer", "b"), VALUE, "r1",
        RecordProvenance(SOURCE, ABI, "chain-payload"),
    )
    fixture = KnowledgeBank("bank", binding("chain-payload"), [first, second_a, second_b])
    op = operator(
        "native", "native", "chain-operator",
        fn=lambda request: LoweredResult(
            request.value,
            own_predicted_state={"next": {"subject": request.value, "relation": "answer"}},
        ),
    )
    receipt = InterchangePanel({"bank": fixture}, {"native": op}).run_chain(
        LogicalAddress("start", "points-to"),
        bank_id="bank",
        next_query=lambda state: state["next"],
        max_steps=3,
    )
    assert receipt.status == AMBIGUOUS
    assert len(receipt.steps) == 2
    assert receipt.steps[-1].row.status == AMBIGUOUS
    assert receipt.terminal_status == "not-requested"


def test_native_session_bridge_is_duck_typed_to_the_public_capture_restore_shape():
    class Session:
        def __init__(self):
            self.events = []

        def capture(self, name, state, *, parent="parent", metadata=None):
            self.events.append(("capture", name, parent, metadata))

        def restore(self, name):
            self.events.append(("restore", name))

    session = Session()
    bridge = NativeResearchSessionBridge(session)
    panel = InterchangePanel(
        {"bank": bank("bank", "bank-payload", "France")},
        {"native": operator("native", "native", "operator-payload")},
        session=bridge,
    )
    panel.fork("candidate", updates={"x": 1})
    panel.restore("candidate")
    assert session.events[0][0] == "capture"
    assert session.events[-1] == ("restore", "candidate")
