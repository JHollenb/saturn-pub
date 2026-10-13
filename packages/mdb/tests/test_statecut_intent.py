from copy import deepcopy

import pytest

from mdb.authoritative_dependency import DependencyContract, DependencyDeclaration
from mdb.execution_skeleton import content_digest
from mdb.statecut_intent import (
    HISTORICAL_COUNTERFACTUAL_RECONSTRUCTION,
    PRESENT_STATE_INTERVENTION,
    ExecutionEvidence,
    ParityEvidence,
    ReplayScheduleItem,
    StateCutIntent,
    StateCutIntentError,
    StateCutRef,
    plan_statecut_intent,
    record_statecut_intent,
    verify_statecut_intent_receipt,
)


def digest(label):
    return content_digest({"label": label})


def cut(label, parent=None, *, authenticated=False):
    return StateCutRef(
        cut_ref=f"cas://sha256/{digest(label)}",
        cut_fingerprint=digest(label),
        parent_cut_ref=None if parent is None else parent.cut_ref,
        parent_cut_fingerprint=None if parent is None else parent.cut_fingerprint,
        parent_receipt_fingerprint=(digest(f"parent-receipt:{label}") if authenticated else None),
        boundary={"kind": "token", "label": label},
    )


@pytest.fixture
def contract():
    return DependencyContract(
        contract_id="reasoning-history",
        input_addresses=("prompt",),
        skeleton_fingerprint=digest("skeleton"),
        specimen_fingerprint=digest("specimen"),
        nodes=(
            DependencyDeclaration(
                node_id="prefix",
                dependencies=(),
                reads=("prompt",),
                writes=("prefix-state",),
                invalidates=("prefix-state",),
                work_units=1,
            ),
            DependencyDeclaration(
                node_id="writer",
                dependencies=("prefix",),
                reads=("prefix-state",),
                writes=("reasoning-source",),
                invalidates=("reasoning-source",),
                work_units=2,
            ),
            DependencyDeclaration(
                node_id="store",
                dependencies=("writer",),
                reads=("reasoning-source",),
                writes=("reasoning-kv",),
                invalidates=("reasoning-kv",),
                work_units=3,
            ),
            DependencyDeclaration(
                node_id="reader",
                dependencies=("store",),
                reads=("reasoning-kv",),
                writes=("answer-logits",),
                invalidates=("answer-logits",),
                work_units=4,
            ),
        ),
        outputs=("reader",),
    )


def present_intent(contract):
    observation = cut("observation")
    return StateCutIntent(
        intent_id="present-reasoning-kv",
        intent_kind=PRESENT_STATE_INTERVENTION,
        edit_application_cut=observation,
        observation_cut=observation,
        model_identity_fingerprint=digest("model"),
        state_schema_fingerprint=digest("state-schema"),
        numerical_contract="qwen-bf16-eager-v1",
        dependency_contract_fingerprint=contract.fingerprint,
        skeleton_fingerprint=contract.skeleton_fingerprint,
        specimen_fingerprint=contract.specimen_fingerprint,
        changed_cells=("writer",),
        changed_addresses=("reasoning-source",),
        execution_history_fingerprint=digest("visible-token-history"),
        intervention_lineage=("act/source-edit",),
        provenance={"donor_cut": "cas://sha256/donor", "role": "reasoning-content"},
        native_consumer="qwen.greedy",
        continuation_horizon=4,
    )


def historical_intent(contract, *, authenticated=True, rewind_is_source=False):
    rewind = cut("rewind")
    source = rewind if rewind_is_source else cut("source", rewind, authenticated=authenticated)
    middle = cut("middle", source, authenticated=authenticated)
    observation = cut("observation-history", middle, authenticated=authenticated)
    path = [rewind, source, middle, observation]
    if rewind_is_source:
        path = [rewind, middle, observation]
    schedule = tuple(
        ReplayScheduleItem(
            step_id=f"step-{index}",
            sequence=index,
            input_authority="caller-supplied" if index == 0 else "model-generated",
            input_fingerprint=digest(f"input-{index}"),
            parent_cut=parent,
            result_cut=result,
            dependency_cells=("writer",) if index == 0 else ("store", "reader"),
            input_metadata={"token_position": index},
        )
        for index, (parent, result) in enumerate(zip(path, path[1:]))
    )
    return StateCutIntent(
        intent_id="historical-reasoning-reconstruction",
        intent_kind=HISTORICAL_COUNTERFACTUAL_RECONSTRUCTION,
        edit_application_cut=source,
        observation_cut=observation,
        rewind_cut=rewind,
        replay_schedule=schedule,
        model_identity_fingerprint=digest("model"),
        state_schema_fingerprint=digest("state-schema"),
        numerical_contract="qwen-bf16-eager-v1",
        dependency_contract_fingerprint=contract.fingerprint,
        skeleton_fingerprint=contract.skeleton_fingerprint,
        specimen_fingerprint=contract.specimen_fingerprint,
        changed_cells=("writer",),
        changed_addresses=("reasoning-source",),
        execution_history_fingerprint=digest("visible-token-history"),
        intervention_lineage=("act/source-edit", "act/replay"),
        provenance={"donor_cut": "cas://sha256/donor", "role": "reasoning-content"},
        native_consumer="qwen.greedy",
        continuation_horizon=4,
    )


def execution_rows(plan):
    return tuple(
        ExecutionEvidence(
            cell_id=cell_id,
            receipt_fingerprint=digest(f"cell:{cell_id}"),
            model_identity_fingerprint=plan["model_identity_fingerprint"],
            state_schema_fingerprint=plan["state_schema_fingerprint"],
            numerical_contract=plan["numerical_contract"],
        )
        for cell_id in plan["execution_cells"]
    )


def test_present_and_historical_modes_retain_distinct_semantics_and_same_declared_closure(contract):
    present = present_intent(contract)
    historical = historical_intent(contract)
    present_plan = plan_statecut_intent(present, contract)
    historical_plan = plan_statecut_intent(historical, contract)

    assert StateCutIntent.from_dict(present.to_dict()) == present
    assert present_plan["intent_kind"] == PRESENT_STATE_INTERVENTION
    assert historical_plan["intent_kind"] == HISTORICAL_COUNTERFACTUAL_RECONSTRUCTION
    assert present_plan["declared_scope"]["semantics"].startswith("edit-current-state")
    assert historical_plan["declared_scope"]["semantics"].startswith("restore-rewind-cut")
    assert present_plan["execution_cells"] == ["writer", "store", "reader"]
    assert present_plan["execution_cells"] == historical_plan["execution_cells"]
    assert present_plan["reused_cells"] == ["prefix"]
    assert historical_plan["verified_scope"]["ancestry_status"] == "receipt-bound-parent-links"
    assert historical_plan["claims"] == {
        "exact_reconstruction": False,
        "dependency_graph_complete": False,
        "speedup": "not-claimed",
    }


def test_historical_replay_can_begin_at_source_but_declared_links_are_not_authenticated(contract):
    intent = historical_intent(contract, authenticated=False, rewind_is_source=True)
    plan = plan_statecut_intent(intent, contract)

    assert intent.rewind_cut == intent.edit_application_cut
    assert plan["verified_scope"]["ancestry_status"] == "declared-only"
    assert plan["receipt_bound_parent_linked_closure"] == []
    assert len(plan["declared_parent_linked_closure"]) == 2


def test_historical_mode_refuses_missing_or_noncontiguous_replay(contract):
    base = historical_intent(contract)
    common = dict(
        intent_id="bad-history",
        intent_kind=HISTORICAL_COUNTERFACTUAL_RECONSTRUCTION,
        edit_application_cut=base.edit_application_cut,
        observation_cut=base.observation_cut,
        rewind_cut=base.rewind_cut,
        model_identity_fingerprint=base.model_identity_fingerprint,
        state_schema_fingerprint=base.state_schema_fingerprint,
        numerical_contract=base.numerical_contract,
        dependency_contract_fingerprint=base.dependency_contract_fingerprint,
        changed_cells=base.changed_cells,
        execution_history_fingerprint=base.execution_history_fingerprint,
    )
    with pytest.raises(StateCutIntentError, match="explicit replay schedule"):
        StateCutIntent(**common)

    broken = list(base.replay_schedule)
    foreign = cut("foreign")
    broken[1] = ReplayScheduleItem(
        step_id=broken[1].step_id,
        sequence=broken[1].sequence,
        input_authority=broken[1].input_authority,
        input_fingerprint=broken[1].input_fingerprint,
        parent_cut=foreign,
        result_cut=cut("foreign-result", foreign),
    )
    with pytest.raises(StateCutIntentError, match="contiguous declared lineage"):
        StateCutIntent(**common, replay_schedule=tuple(broken))


def test_planner_refuses_mismatched_dependency_and_specimen_bindings(contract):
    intent = present_intent(contract)
    other = DependencyContract(
        contract_id="other",
        input_addresses=contract.input_addresses,
        nodes=contract.nodes,
        outputs=contract.outputs,
        skeleton_fingerprint=contract.skeleton_fingerprint,
        specimen_fingerprint=contract.specimen_fingerprint,
    )
    with pytest.raises(StateCutIntentError, match="dependency contract fingerprints differ"):
        plan_statecut_intent(intent, other)

    wrong = deepcopy(intent.to_dict())
    wrong.pop("fingerprint")
    wrong["dependency_contract_fingerprint"] = contract.fingerprint
    wrong["skeleton_fingerprint"] = digest("wrong-skeleton")
    wrong = StateCutIntent(
        **{
            "intent_id": wrong["intent_id"],
            "intent_kind": wrong["intent_kind"],
            "edit_application_cut": intent.edit_application_cut,
            "observation_cut": intent.observation_cut,
            "model_identity_fingerprint": wrong["model_identity_fingerprint"],
            "state_schema_fingerprint": wrong["state_schema_fingerprint"],
            "numerical_contract": wrong["numerical_contract"],
            "dependency_contract_fingerprint": wrong["dependency_contract_fingerprint"],
            "skeleton_fingerprint": wrong["skeleton_fingerprint"],
            "changed_cells": wrong["changed_cells"],
            "changed_addresses": wrong["changed_addresses"],
            "execution_history_fingerprint": wrong["execution_history_fingerprint"],
        }
    )
    with pytest.raises(StateCutIntentError, match="skeletons differ"):
        plan_statecut_intent(wrong, contract)

    historical = historical_intent(contract)
    schedule = list(historical.replay_schedule)
    first = schedule[0]
    schedule[0] = ReplayScheduleItem(
        step_id=first.step_id,
        sequence=first.sequence,
        input_authority=first.input_authority,
        input_fingerprint=first.input_fingerprint,
        parent_cut=first.parent_cut,
        result_cut=first.result_cut,
        dependency_cells=("invented-cell",),
    )
    hostile = StateCutIntent(
        intent_id="hostile-schedule-cell",
        intent_kind=historical.intent_kind,
        edit_application_cut=historical.edit_application_cut,
        observation_cut=historical.observation_cut,
        rewind_cut=historical.rewind_cut,
        replay_schedule=schedule,
        model_identity_fingerprint=historical.model_identity_fingerprint,
        state_schema_fingerprint=historical.state_schema_fingerprint,
        numerical_contract=historical.numerical_contract,
        dependency_contract_fingerprint=historical.dependency_contract_fingerprint,
        changed_cells=historical.changed_cells,
        execution_history_fingerprint=historical.execution_history_fingerprint,
    )
    with pytest.raises(StateCutIntentError, match="outside the dependency contract"):
        plan_statecut_intent(hostile, contract)


def test_receipt_binds_declared_execution_order_contract_and_separate_parity(contract):
    plan = plan_statecut_intent(historical_intent(contract), contract)
    rows = execution_rows(plan)
    step_receipts = {
        item["step_id"]: item["result_cut"]["parent_receipt_fingerprint"]
        for item in plan["replay_schedule"]
    }
    restore = ParityEvidence(
        parity_kind="restore",
        reference_sha256=digest("restored-state"),
        candidate_sha256=digest("restored-state"),
        receipt_fingerprint=digest("restore-receipt"),
        scope={"state": "selected-boundary-bytes-and-pointers"},
    )
    replay = ParityEvidence(
        parity_kind="replay",
        reference_sha256=digest("native-future"),
        candidate_sha256=digest("changed-native-future"),
        receipt_fingerprint=digest("replay-receipt"),
        scope={"future": ["token_ids", "logits", "eos"]},
    )
    receipt = record_statecut_intent(
        plan,
        result_cut=cut("candidate-result"),
        execution_evidence=rows,
        replay_step_receipts=step_receipts,
        native_consumer_receipt_fingerprint=digest("native-consumer"),
        restore_parity=restore,
        replay_parity=replay,
    )

    assert verify_statecut_intent_receipt(receipt, plan)
    assert receipt["verified_scope"]["native_restore_parity"] == "exact"
    assert receipt["verified_scope"]["native_replay_parity"] == "different"
    assert receipt["claims"]["dependency_graph_complete"] is False

    wrong_step_receipts = dict(step_receipts)
    wrong_step_receipts[next(iter(wrong_step_receipts))] = digest("unrelated-replay-receipt")
    with pytest.raises(StateCutIntentError, match="schedule parent-link receipt"):
        record_statecut_intent(
            plan,
            result_cut=cut("wrong-replay-receipt"),
            execution_evidence=rows,
            replay_step_receipts=wrong_step_receipts,
            native_consumer_receipt_fingerprint=digest("native-consumer"),
        )

    resealed = deepcopy(receipt)
    resealed.pop("fingerprint")
    resealed["replay_step_receipts"][next(iter(step_receipts))] = digest(
        "resealed-unrelated-replay-receipt"
    )
    resealed["fingerprint"] = content_digest(resealed)
    with pytest.raises(StateCutIntentError, match="schedule parent-link receipt"):
        verify_statecut_intent_receipt(resealed, plan)

    with pytest.raises(StateCutIntentError, match="exact order"):
        record_statecut_intent(
            plan,
            result_cut=cut("bad-order"),
            execution_evidence=tuple(reversed(rows)),
            replay_step_receipts=step_receipts,
            native_consumer_receipt_fingerprint=digest("native-consumer"),
        )
    wrong_contract = list(rows)
    wrong_contract[0] = ExecutionEvidence(
        cell_id=wrong_contract[0].cell_id,
        receipt_fingerprint=wrong_contract[0].receipt_fingerprint,
        model_identity_fingerprint=wrong_contract[0].model_identity_fingerprint,
        state_schema_fingerprint=wrong_contract[0].state_schema_fingerprint,
        numerical_contract="fp64-other-lane",
    )
    with pytest.raises(StateCutIntentError, match="numerical contract mismatch"):
        record_statecut_intent(
            plan,
            result_cut=cut("wrong-contract"),
            execution_evidence=wrong_contract,
            replay_step_receipts=step_receipts,
            native_consumer_receipt_fingerprint=digest("native-consumer"),
        )
    for field_name in ("model_identity_fingerprint", "state_schema_fingerprint"):
        wrong_identity = list(rows)
        values = {
            "cell_id": wrong_identity[0].cell_id,
            "receipt_fingerprint": wrong_identity[0].receipt_fingerprint,
            "model_identity_fingerprint": wrong_identity[0].model_identity_fingerprint,
            "state_schema_fingerprint": wrong_identity[0].state_schema_fingerprint,
            "numerical_contract": wrong_identity[0].numerical_contract,
        }
        values[field_name] = digest(f"wrong:{field_name}")
        wrong_identity[0] = ExecutionEvidence(**values)
        with pytest.raises(StateCutIntentError, match="identity or numerical contract mismatch"):
            record_statecut_intent(
                plan,
                result_cut=cut(f"wrong-{field_name}"),
                execution_evidence=wrong_identity,
                replay_step_receipts=step_receipts,
                native_consumer_receipt_fingerprint=digest("native-consumer"),
            )
    with pytest.raises(StateCutIntentError, match="explicit schedule exactly"):
        record_statecut_intent(
            plan,
            result_cut=cut("missing-step"),
            execution_evidence=rows,
            replay_step_receipts={},
            native_consumer_receipt_fingerprint=digest("native-consumer"),
        )


def test_receipt_and_intent_tamper_fail_closed(contract):
    intent = present_intent(contract)
    plan = plan_statecut_intent(intent, contract)
    receipt = record_statecut_intent(
        plan,
        result_cut=cut("present-result"),
        execution_evidence=execution_rows(plan),
        native_consumer_receipt_fingerprint=digest("native-consumer"),
    )
    tampered_intent = deepcopy(intent.to_dict())
    tampered_intent["numerical_contract"] = "different"
    with pytest.raises(StateCutIntentError, match="schema or fingerprint mismatch"):
        StateCutIntent.from_dict(tampered_intent)
    tampered_receipt = deepcopy(receipt)
    tampered_receipt["result_cut"]["cut_ref"] = "cas://sha256/tampered"
    with pytest.raises(StateCutIntentError, match="schema or fingerprint mismatch"):
        verify_statecut_intent_receipt(tampered_receipt, plan)

    inconsistent_plan = deepcopy(plan)
    inconsistent_plan["numerical_contract"] = "re-sealed-but-inconsistent"
    inconsistent_plan.pop("fingerprint")
    inconsistent_plan["fingerprint"] = content_digest(inconsistent_plan)
    with pytest.raises(StateCutIntentError, match="disagrees with embedded intent"):
        record_statecut_intent(
            inconsistent_plan,
            result_cut=cut("inconsistent-plan"),
            execution_evidence=execution_rows(plan),
            native_consumer_receipt_fingerprint=digest("native-consumer"),
        )
