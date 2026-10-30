from __future__ import annotations

import pytest
import torch

from mdb.forward_interventions import (
    ForwardInterventionError,
    branch_forward,
    branch_forward_many,
)
from mdb.forward_recording import record_forward


class TinyChain(torch.nn.Module):
    def forward(self, value):
        doubled = value * 2
        hidden = doubled + 1
        logits = hidden * 3
        return {"logits": logits, "hidden": hidden}


def capture(value=None):
    value = torch.tensor([[1.0, 2.0]]) if value is None else value
    result = record_forward(
        TinyChain().eval(),
        (value,),
        model_identity="tiny-intervention-test",
        execution={"backend": "cpu-fp32"},
        input_identity={"case": "anonymous-tiny"},
    )
    return result


def first_outputs(recorded, count=2):
    return [
        value_id
        for event in recorded.trace["events"]
        for value_id in event["outputs"]
    ][:count]


def test_single_site_branch_replays_native_prefix_and_records_signed_output_delta():
    recorded = capture()
    value_id = first_outputs(recorded, 1)[0]
    root_id, root = next(iter(recorded.initial_values.items()))
    before = root.clone()

    result = branch_forward(
        recorded.trace,
        recorded.initial_values,
        value_id,
        torch.full_like(root, 4.0),
        native_output=recorded.output,
        source_identity={
            "donor_trace_fingerprint": "a" * 64,
            "case": "tiny-donor",
            "kind": "decay",
        },
    )

    assert torch.equal(result.output["logits"], torch.tensor([[15.0, 15.0]]))
    assert torch.equal(result.output["hidden"], torch.tensor([[5.0, 5.0]]))
    assert torch.equal(recorded.initial_values[root_id], before)
    assert result.receipt.to_dict()["trace_fingerprint"] == recorded.trace["fingerprint"]
    assert result.receipt.to_dict()["native_prefix"]["exact"] is True
    assert result.receipt.to_dict()["parent_fingerprint"]
    assert result.receipt.to_dict()["source_identity"]["case"] == "tiny-donor"
    assert result.receipt.to_dict()["intervention_sites"][0]["replacement_digest"]
    assert any(row["changed"] for row in result.output_differences)
    logits = next(row for row in result.output_differences if row["path"] == "$.logits")
    assert logits["mean_delta"] == pytest.approx(3.0)
    assert any(not row["exact_outputs"] for row in result.step_observations)


def test_opt_in_observations_return_candidate_tensors_and_bind_hashes_in_receipt():
    recorded = capture()
    site_id, hidden_id = first_outputs(recorded, 2)
    logits_id = recorded.trace["outputs"]["logits"]["$value"]

    result = branch_forward(
        recorded.trace,
        recorded.initial_values,
        site_id,
        torch.full_like(recorded.observed_values[site_id], 4.0),
        native_output=recorded.output,
        observe_values=(site_id, hidden_id, logits_id),
    )

    assert set(result.selected_values) == {site_id, hidden_id, logits_id}
    assert torch.equal(result.selected_values[hidden_id], result.output["hidden"])
    observations = result.receipt.to_dict()["selected_value_observations"]
    assert [row["value_id"] for row in observations] == [site_id, hidden_id, logits_id]
    assert observations[0]["candidate_digest"] != observations[0]["native_digest"]
    descendants = {row["value_id"]: row["candidate_digest"]
                   for row in result.receipt.to_dict()["descendant_observations"]}
    assert observations[1]["candidate_digest"] == descendants[hidden_id]


@pytest.mark.parametrize("observe_values", [("missing",), ("",), ("v000000", "v000000")])
def test_opt_in_observation_ids_are_validated(observe_values):
    recorded = capture()
    value_id = first_outputs(recorded, 1)[0]
    with pytest.raises(ForwardInterventionError, match="observed SSA value|unique non-empty"):
        branch_forward(
            recorded.trace,
            recorded.initial_values,
            value_id,
            recorded.observed_values[value_id],
            native_output=recorded.output,
            observe_values=observe_values,
        )


def test_noop_copy_is_exact_and_branches_share_the_same_parent_fingerprint():
    recorded = capture()
    value_id = first_outputs(recorded, 1)[0]
    native_value = recorded.observed_values[value_id]

    first = branch_forward(
        recorded.trace,
        recorded.initial_values,
        value_id,
        torch.full_like(native_value, 9.0),
        dose=0,
        native_output=recorded.output,
    )
    second = branch_forward(
        recorded.trace,
        recorded.initial_values,
        value_id,
        native_value.clone(),
        dose=1,
        native_output=recorded.output,
    )

    assert torch.equal(first.output["logits"], recorded.output["logits"])
    assert torch.equal(second.output["logits"], recorded.output["logits"])
    assert all(not row["changed"] for row in first.output_differences)
    assert all(not row["changed"] for row in second.output_differences)
    assert first.receipt.to_dict()["changed_values"] == []
    assert first.receipt.to_dict()["parent_fingerprint"] == second.receipt.to_dict()["parent_fingerprint"]


def test_two_site_branch_applies_values_at_each_native_producer_boundary():
    recorded = capture()
    first, second = first_outputs(recorded, 2)

    result = branch_forward_many(
        recorded.trace,
        recorded.initial_values,
        {
            first: torch.full((1, 2), 4.0),
            second: torch.full((1, 2), 7.0),
        },
        dose={first: 1.0, second: 0.5},
        native_output=recorded.output,
    )

    assert torch.equal(result.output["hidden"], torch.full((1, 2), 6.0))
    assert torch.equal(result.output["logits"], torch.full((1, 2), 18.0))
    sites = result.receipt.to_dict()["intervention_sites"]
    assert [row["value_id"] for row in sites] == [first, second]
    assert all(row["act"]["operation"]["authority"] == "registered" for row in sites)
    assert result.receipt.to_dict()["changed_descendants"]


@pytest.mark.parametrize(
    "replacement, message",
    [
        (torch.ones(1, 3), "shape"),
        (torch.ones(1, 2, dtype=torch.float64), "dtype"),
        (torch.tensor([[float("nan"), 1.0]]), "non-finite"),
        (torch.empty((1, 2), device="meta"), "CPU device"),
    ],
)
def test_replacement_contract_rejects_shape_dtype_nonfinite_and_device_mismatches(
    replacement, message
):
    recorded = capture()
    value_id = first_outputs(recorded, 1)[0]
    with pytest.raises(ForwardInterventionError, match=message):
        branch_forward(
            recorded.trace,
            recorded.initial_values,
            value_id,
            replacement,
            native_output=recorded.output,
        )


def test_roots_are_exact_and_observed_intermediates_cannot_be_supplied_as_replay_roots():
    recorded = capture()
    value_id = first_outputs(recorded, 1)[0]
    with pytest.raises(ForwardInterventionError, match="intermediate oracles"):
        branch_forward(
            recorded.trace,
            recorded.observed_values,
            value_id,
            torch.ones(1, 2),
            native_output=recorded.output,
        )

    changed_roots = {key: value.clone() for key, value in recorded.initial_values.items()}
    root_id = next(iter(changed_roots))
    changed_roots[root_id] = changed_roots[root_id] + 1
    with pytest.raises(ForwardInterventionError, match="root bytes"):
        branch_forward(
            recorded.trace,
            changed_roots,
            value_id,
            torch.ones(1, 2),
            native_output=recorded.output,
        )


def test_observation_output_must_match_trace_and_root_sites_are_read_only():
    recorded = capture()
    value_id = first_outputs(recorded, 1)[0]
    with pytest.raises(ForwardInterventionError, match="sealed trace"):
        branch_forward(
            recorded.trace,
            recorded.initial_values,
            value_id,
            torch.ones(1, 2),
            native_output={"logits": torch.zeros(1, 2), "hidden": recorded.output["hidden"]},
        )
    root_id = next(iter(recorded.initial_values))
    with pytest.raises(ForwardInterventionError, match="read-only"):
        branch_forward(
            recorded.trace,
            recorded.initial_values,
            root_id,
            torch.ones(1, 2),
            native_output=recorded.output,
        )


def test_replacing_a_view_ssa_value_rebinds_only_that_handle_not_its_root_alias():
    class ViewAlias(torch.nn.Module):
        def forward(self, value):
            view = value.view_as(value)
            return {"view": view, "root": value, "next": view + 1}

    value = torch.tensor([[1.0, 2.0]])
    recorded = record_forward(
        ViewAlias().eval(),
        (value,),
        model_identity="tiny-view-alias-test",
        execution={"backend": "cpu-fp32"},
        input_identity={"case": "alias"},
    )
    value_id = first_outputs(recorded, 1)[0]
    root_id, root = next(iter(recorded.initial_values.items()))

    result = branch_forward(
        recorded.trace,
        recorded.initial_values,
        value_id,
        torch.full_like(root, 9.0),
        native_output=recorded.output,
    )

    assert torch.equal(result.output["view"], torch.full_like(root, 9.0))
    assert torch.equal(result.output["root"], root)
    assert torch.equal(result.output["next"], torch.full_like(root, 10.0))
    assert result.output["view"].data_ptr() != result.output["root"].data_ptr()
    assert result.receipt.to_dict()["alias_scope"].startswith("selected-SSA-handle-only")
    assert torch.equal(recorded.initial_values[root_id], root)
