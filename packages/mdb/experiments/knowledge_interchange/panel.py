"""Finite supplied-bank/operator example; no neural model or learned law."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mdb.knowledge_interchange import (
    ArtifactBinding,
    BindingExpectation,
    ExactLaw,
    InterchangePanel,
    KnowledgeBank,
    KnowledgeOperator,
    KnowledgeRecord,
    LogicalAddress,
    LogicalTuple,
    LoweredResult,
    OperatorContract,
    RecordProvenance,
    VALUE,
    content_digest,
)


ABI = "supplied-host-law-example-v1"
SCOPE = {"scope": "fictional"}


def make_bank(name, triples):
    digest = content_digest(triples)
    binding = ArtifactBinding("fixture", ABI, digest, "r1")
    records = [KnowledgeRecord(
        LogicalTuple(subject, relation, value, SCOPE), VALUE, "r1",
        RecordProvenance("fixture", ABI, digest, "r1", locator=f"{name}/{i}"),
        record_id=f"{name}-{i}",
    ) for i, (subject, relation, value) in enumerate(triples)]
    return KnowledgeBank(name, binding, records, revision="r1")


def reference(request):
    relation = request.query.relation
    if relation == "friend":
        next_relation = "habitat"
    elif relation == "habitat":
        next_relation = "country"
    else:
        next_relation = None
    return LoweredResult(request.value, own_predicted_state={
        "subject": request.value, "relation": next_relation,
    })


def compiled(request):
    # A supplied exact host law compiled to a dispatch table. This is not an
    # extracted neural operator; both implementations have the same public ABI.
    dispatch = {"friend": "habitat", "habitat": "country", "country": None}
    return LoweredResult(request.value, own_predicted_state={
        "subject": request.value, "relation": dispatch[request.query.relation],
    })


def make_operator(name, mode, lowerer):
    contract = OperatorContract(
        name, mode, ArtifactBinding("fixture", ABI, content_digest(name)),
        recipient_id="host-example", bank_binding=BindingExpectation(model_abi=ABI),
        lowerer_id=name, consumer_id="host-observation",
        exact_law=ExactLaw("supplied-friend-habitat-country-dispatch"),
    )
    return KnowledgeOperator(contract, lowerer)


def next_address(state):
    if state["relation"] is None:
        return None
    return LogicalAddress(state["subject"], state["relation"], SCOPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    banks = {
        "A": make_bank("A", [("Bob", "friend", "Carol"),
                              ("Carol", "habitat", "Rome"), ("Rome", "country", "Norway")]),
        "B": make_bank("B", [("Bob", "friend", "Dave"),
                              ("Dave", "habitat", "Oslo"), ("Oslo", "country", "Italy")]),
    }
    operators = {"reference": make_operator("reference", "native", reference),
                 "compiled": make_operator("compiled", "compiled", compiled)}
    panel = InterchangePanel(banks, operators, initial_state={"cursor": 0},
                             panel_id="supplied-bank-operator-example")
    start = LogicalAddress("Bob", "friend", SCOPE)
    consumer = lambda request: {"value": request.lowered_output,
                                "state_source": request.execution_state_source}
    matrix = panel.run_matrix(start, recipient="host-example", consumer=consumer)
    chains = {f"{bank}/{operator}": panel.run_chain(
        start, bank_id=bank, operator_ids=[operator], max_steps=3,
        next_query=next_address, recipient="host-example", consumer=consumer,
    ).to_payload() for bank in banks for operator in operators}
    controls = {
        "missing": panel.run_matrix(LogicalAddress("Nobody", "friend", SCOPE)),
        "stale": panel.run_matrix(start, revision="r2"),
        "wrong-route": panel.run_matrix(start, bank_ids=["A"], candidate_record_ids=["A-1"]),
    }
    ambiguous = banks["A"].with_record(KnowledgeRecord(
        LogicalTuple("Bob", "friend", "Eve", SCOPE), VALUE, "r1",
        banks["A"].records[0].provenance, record_id="A-conflict"), bank_id="ambiguous")
    controls["ambiguous"] = InterchangePanel({"ambiguous": ambiguous}, operators).run_matrix(start)
    # A supplied consumer state cannot choose the next address. The predicted
    # Carol -> Rome path remains visible even under a wrong execution state.
    oracle = panel.run_chain(start, bank_id="A", operator_ids=["compiled"], max_steps=3,
        next_query=next_address, supplied_executions=[{"subject": "Dave", "relation": "habitat"}],
        recipient="host-example", consumer=consumer).to_payload()
    payload = {"schema": "mdb-supplied-interchange-example-v1",
               "scope": "host mechanics; supplied banks and laws; no neural execution",
               "matrix": matrix.to_payload(), "chains": chains,
               "controls": {name: receipt.to_payload() for name, receipt in controls.items()},
               "supplied_execution_control": oracle}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out.exists() and args.out.read_text() != raw:
        raise ValueError("refusing to replace previous example evidence")
    args.out.write_text(raw)
    print(json.dumps({"out": str(args.out.resolve()), "matrix_rows": len(matrix.rows),
                      "chain_steps": {name: len(value["steps"]) for name, value in chains.items()},
                      "controls": {name: value.status_counts for name, value in controls.items()}}))


if __name__ == "__main__":
    main()
