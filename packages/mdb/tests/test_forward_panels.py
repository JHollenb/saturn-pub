from __future__ import annotations

import copy

import pytest
import torch
from saturn_pub import Receipt
from saturn_pub.values import digest

from mdb.forward_interventions import (
    ForwardInterventionError,
    PreparedForwardParent,
    branch_forward_many,
)
from mdb.forward_panels import ExperimentPanel
from mdb.forward_recording import record_forward


class Organism(torch.nn.Module):
    def forward(self,x):
        a = x*2
        b = a+1
        c = b.square()
        return {"y":c+b,"b":b}


def capture():
    return record_forward(Organism().eval(),(torch.tensor([[1.,2.]]),),
        model_identity="prepared-toy",execution={"backend":"cpu-fp32"},input_identity={"x":"fixed"})


def produced(c,index):
    return c.trace["events"][index]["outputs"][0]


def test_prepared_branches_match_root_replay_multisite_doses_and_observations():
    c = capture()
    b,y = produced(c,1),produced(c,2)
    with PreparedForwardParent(c.trace,c.initial_values,through_value=b,native_output=c.output,
                               observe_values=(b,y)) as parent:
        for delta in (0.,2.,-3.):
            patches = {b:c.observed_values[b]+delta,y:c.observed_values[y]-.5}
            kwargs = {"dose":{b:1.,y:.25},"observe_values":(b,y)}
            expected = branch_forward_many(c.trace,c.initial_values,patches,native_output=c.output,**kwargs)
            got = parent.branch(patches,**kwargs)
            assert all(torch.equal(got.output[k],v) for k,v in expected.output.items())
            assert all(torch.equal(got.selected_values[k],v) for k,v in expected.selected_values.items())
            assert got.output_differences==expected.output_differences
            receipt = got.receipt.to_dict()
            assert Receipt.make(**{k:v for k,v in receipt.items() if k not in {"schema","fingerprint"}}).to_dict()==receipt
            reuse = receipt["replay_reuse"]
            assert digest(reuse["state_cut_manifest"])==reuse["state_cut_fingerprint"]
            assert reuse["operators_executed"]==2
            assert reuse["prefix_operators_reused"]==2
        assert parent.manifest()["completed_branches"]==3
    with pytest.raises(ForwardInterventionError,match="closed"):
        parent.branch({b:c.observed_values[b]})


def test_preparation_is_isolated_from_caller_trace_roots_and_returned_tensors():
    c = capture()
    b = produced(c,1)
    parent = PreparedForwardParent(c.trace,c.initial_values,through_value=b,native_output=c.output)
    expected = copy.deepcopy(c.output)
    next(iter(c.initial_values.values())).add_(100)
    c.trace["execution"]["backend"]="tampered"
    one = parent.branch({b:c.observed_values[b]})
    one.output["y"].zero_()
    two = parent.branch({b:c.observed_values[b]})
    assert all(torch.equal(two.output[k],v) for k,v in expected.items())
    parent.close()


def test_view_storage_geometry_and_aliases_survive_native_prefix_fork():
    class Views(torch.nn.Module):
        def forward(self,x):
            a = x*2
            v = a[:,1:]
            w = torch.as_strided(v,(2,2),(3,1),0)
            return {"result":w+v,"a":a}
    c = record_forward(Views().eval(),(torch.arange(6,dtype=torch.float32).reshape(2,3),),
        model_identity="views",execution={},input_identity={})
    v = next(e["outputs"][0] for e in reversed(c.trace["events"]) if e["operator"]=="aten.slice.Tensor")
    # A later site leaves the offset view live in the prefix closure.
    w = next(e["outputs"][0] for e in c.trace["events"] if e["operator"]=="aten.as_strided.default")
    with PreparedForwardParent(c.trace,c.initial_values,through_value=v) as parent:
        patches = {w:c.observed_values[w]+1}
        actual = parent.branch(patches)
        expected = branch_forward_many(c.trace,c.initial_values,patches,native_output=c.output)
        assert all(torch.equal(actual.output[k],val) for k,val in expected.output.items())


def test_tampered_private_frontier_is_detected_before_continuation():
    c = capture()
    b = produced(c,1)
    parent = PreparedForwardParent(c.trace,c.initial_values,through_value=b)
    parent._PreparedForwardParent__environment[b].add_(1)
    with pytest.raises(ForwardInterventionError,match="frontier changed"):
        parent.branch({b:c.observed_values[b]})


def test_panel_seals_copies_before_execution_and_matches_retained_reference():
    c = capture()
    b = produced(c,1)
    with PreparedForwardParent(c.trace,c.initial_values,through_value=b,observe_values=(b,)) as parent:
        replacement = c.observed_values[b]+1
        panel = ExperimentPanel(parent,{"copy":{"replacements":{b:c.observed_values[b]}},
            "edit":{"replacements":{b:replacement},"metadata":{"forecast":"prior-only"}}},observe_values=(b,))
        manifest = panel.manifest()
        assert digest({k:v for k,v in manifest.items() if k!="fingerprint"})==panel.fingerprint
        replacement.add_(1000)
        results = panel.run()
        assert all(row.error is None for row in results)
        assert torch.equal(results[0].result.output["y"],c.output["y"])
        expected = branch_forward_many(c.trace,c.initial_values,{b:c.observed_values[b]+1},native_output=c.output)
        assert torch.equal(results[1].result.output["y"],expected.output["y"])
        assert results[1].result.receipt.to_dict()["source_identity"]["panel_fingerprint"]==panel.fingerprint
        with pytest.raises(ForwardInterventionError,match="already executed"):
            panel.run()


def test_panel_rejects_bad_geometry_and_preceding_sites_before_any_branch():
    c = capture()
    a,b = produced(c,0),produced(c,1)
    with PreparedForwardParent(c.trace,c.initial_values,through_value=b) as parent:
        for patches,match in (({a:c.observed_values[a]},"precedes"),({b:torch.zeros(4)},"geometry")):
            with pytest.raises(ForwardInterventionError,match=match):
                ExperimentPanel(parent,{"bad":{"replacements":patches}})
        assert parent.manifest()["completed_branches"]==0


def test_unretained_old_observation_and_unrelated_owner_are_refused():
    c = capture()
    a,b = produced(c,0),produced(c,1)
    with PreparedForwardParent(c.trace,c.initial_values,through_value=b) as parent:
        with pytest.raises(ForwardInterventionError,match="not retained"):
            parent.branch({b:c.observed_values[b]},observe_values=(a,))
        with pytest.raises(ForwardInterventionError,match="different trace/root owner"):
            branch_forward_many(c.trace,c.initial_values,{b:c.observed_values[b]},_prepared_parent=parent)


def test_returned_manifests_cannot_mutate_nested_native_cut_metadata():
    c=capture()
    from mdb.forward_symbols import seal_trace
    c.trace=seal_trace({k:v for k,v in c.trace.items() if k!="fingerprint"}|{
        "execution":{"backend":"cpu-fp32","nested":{"list":["original"]}}})
    b=produced(c,1)
    with PreparedForwardParent(c.trace,c.initial_values,through_value=b) as parent:
        manifest=parent.manifest()
        manifest["state_cut"]["execution"]["nested"]["list"][0]="changed"
        got=parent.branch({b:c.observed_values[b]})
        reuse=got.receipt.to_dict()["replay_reuse"]
        assert digest(reuse["state_cut_manifest"])==reuse["state_cut_fingerprint"]
        assert reuse["state_cut_manifest"]["execution"]["nested"]["list"]==["original"]


def test_seeded_random_replay_is_rejected_before_rng_changes():
    from mdb.forward_recording import ATenStepper, ForwardRecordingError
    from mdb.forward_symbols import seal_trace
    c=capture()
    trace=copy.deepcopy(c.trace)
    trace.pop("fingerprint")
    trace["events"][0]["operator"]="aten.rand_like.default"
    trace=seal_trace(trace)
    machine=ATenStepper(trace,c.initial_values)
    rng=torch.get_rng_state().clone()
    with pytest.raises(ForwardRecordingError,match="RNG closure"):
        machine.step()
    assert torch.equal(torch.get_rng_state(),rng)


def test_public_catalog_verifies_prepared_receipts_and_refuses_resealed_clock_lies():
    from mdb.circuit_explorer import (
        CircuitCatalog,
        CircuitCatalogError,
        verify_branch_forward_result,
    )
    c=capture()
    b=produced(c,1)
    catalog=CircuitCatalog()
    record=catalog.register(circuit_id="prepared-toy",trace=c.trace,
        route_candidates=[{"route_id":"b","source_values":[b],"target_values":[c.trace["outputs"]["y"]["$value"]]}],persist=False)
    replacement=c.observed_values[b]+1
    plan=catalog.plan("prepared-toy",route_id="b",intervention={"value_ids":[b],"replacement_digests":{b:describe_digest(replacement)},"dose":1.},persist=False)
    with PreparedForwardParent(c.trace,c.initial_values,through_value=b) as parent:
        result=parent.branch({b:replacement})
        context=record.binding
        kwargs={"roots":c.initial_values,"replacements":{b:replacement},"dose":1.}
        verify_branch_forward_result(plan=plan.to_dict(),record=record.to_dict(),context=context,trace=c.trace,result=result,kwargs=kwargs)
        receipt=result.receipt.to_dict()
        receipt.pop("schema")
        receipt.pop("fingerprint")
        receipt["replay_reuse"]["through_event"]=0
        receipt["replay_reuse"]["prefix_operators_reused"]=1
        receipt["replay_reuse"]["operators_executed"]=3
        bad={"receipt":Receipt.make(**receipt),"output":result.output}
        with pytest.raises(CircuitCatalogError,match="cut clock"):
            verify_branch_forward_result(plan=plan.to_dict(),record=record.to_dict(),context=context,trace=c.trace,result=bad,kwargs=kwargs)


def describe_digest(value):
    from mdb.forward_recording import tensor_descriptor
    return tensor_descriptor(value)["sha256"]


def test_failed_late_site_does_not_contaminate_the_next_panel_candidate():
    class IntegerSuffix(torch.nn.Module):
        def forward(self,x):
            a=x*2
            b=a.to(torch.int64)
            return {"y":b+4}
    c=record_forward(IntegerSuffix().eval(),(torch.tensor([1.,2.]),),
        model_identity="late-failure",execution={},input_identity={})
    a,b=produced(c,0),produced(c,1)
    with PreparedForwardParent(c.trace,c.initial_values,through_value=a) as parent:
        panel=ExperimentPanel(parent,{
            "partial-failure":{"replacements":{a:c.observed_values[a]+3,b:c.observed_values[b]},"dose":{a:1.,b:.5}},
            "good":{"replacements":{a:c.observed_values[a]}}})
        rows=panel.run()
        assert rows[0].result is None and "non-binary dose" in rows[0].error
        assert rows[1].error is None and torch.equal(rows[1].result.output["y"],c.output["y"])
        assert parent.manifest()["completed_branches"]==1
