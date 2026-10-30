from __future__ import annotations

import copy

import pytest
import torch
from saturn_pub import Receipt
from saturn_pub.values import describe

from mdb.circuit_explorer import (
    BRANCH_FORWARD_CONTRACT,
    CircuitCatalog,
    CircuitCatalogError,
    TraceCircuitAdapter,
    verify_branch_forward_result,
)
from mdb.forward_interventions import branch_forward_many
from mdb.forward_recording import record_forward


class _Toy(torch.nn.Module):
    def forward(self, value):
        return {"y": value + torch.ones_like(value)}


def _fixture():
    model = _Toy().eval()
    value = torch.tensor([1.0, 2.0])
    capture = record_forward(
        model,
        (value,),
        model_identity={"model_id": "toy", "weights_sha256": "a" * 64},
        execution={"backend": "cpu", "dtype": "float32"},
        input_identity={"input_sha256": "b" * 64},
    )
    trace = capture.trace
    target = trace["outputs"]["y"]["$value"]
    replacement = torch.tensor([3.0, 4.0])
    replacement_digest = describe(replacement)["sha256"]
    catalog = CircuitCatalog()
    record = TraceCircuitAdapter.from_trace(trace).to_record(
        circuit_id="circuit://toy/branch",
        route_candidates=[{"route_id": "readout", "target_values": [target]}],
        supported_interventions=[{
            "intervention_id": "replace-output",
            "value_ids": [target],
            "dose": 1.0,
        }],
        replay_handle={"backend": "branch"},
    )
    catalog.register(record, persist=False)
    catalog.attach_trace(trace)
    plan = catalog.plan(
        record.circuit_id,
        route_id="readout",
        intervention={
            "intervention_id": "replace-output",
            "value_ids": [target],
            "dose": 1.0,
            "replacement_digests": {target: replacement_digest},
        },
        persist=False,
    )
    return catalog, trace, capture, plan, target, replacement


def _valid_backend(*, plan, trace, roots, replacements, native_output, **kwargs):
    del plan, kwargs
    result = branch_forward_many(
        trace,
        roots,
        replacements,
        native_output=native_output,
        source_identity={"fixture": "adversarial"},
    )
    return {"receipt": result.receipt, "output": result.output}


def _register(catalog, callback=_valid_backend, verifier=verify_branch_forward_result):
    catalog.register_backend(
        "branch", callback, verifier=verifier, contract=BRANCH_FORWARD_CONTRACT
    )


def _run(catalog, trace, capture, plan, replacement, *, roots=None, callback=None):
    _register(catalog, callback=callback or _valid_backend)
    return catalog.execute(
        plan,
        trace=trace,
        roots=capture.initial_values if roots is None else roots,
        replacements={plan.to_dict()["intervention"]["value_ids"][0]: replacement},
        native_output=capture.output,
    )


def _receipt_with(receipt: Receipt, mutate):
    body = copy.deepcopy(receipt.to_dict())
    body.pop("schema", None)
    body.pop("fingerprint", None)
    mutate(body)
    return Receipt.make(**body)


def test_unverified_generic_backend_is_refused_before_callback():
    catalog, trace, capture, plan, _, replacement = _fixture()
    called = []

    def backend(**kwargs):
        called.append(kwargs)
        return {"receipt": Receipt.make(operation="fake"), "output": {"y": 0}}

    catalog.register_backend("unverified", backend)
    with pytest.raises(CircuitCatalogError, match="no evidence verifier"):
        catalog.execute(
            plan,
            backend="unverified",
            trace=trace,
            roots=capture.initial_values,
            replacements={plan.to_dict()["intervention"]["value_ids"][0]: replacement},
            native_output=capture.output,
        )
    assert called == []


@pytest.mark.parametrize(
    "field, value",
    [
        ("trace_fingerprint", "0" * 64),
        ("model_identity", {"model_id": "wrong"}),
        ("execution", {"backend": "wrong"}),
        ("input_identity", {"input_sha256": "wrong"}),
    ],
)
def test_branch_receipt_context_is_bound(field, value):
    catalog, trace, capture, plan, target, replacement = _fixture()

    def backend(*, plan, trace, roots, replacements, native_output, **kwargs):
        valid = _valid_backend(
            plan=plan, trace=trace, roots=roots, replacements=replacements,
            native_output=native_output, **kwargs,
        )
        valid["receipt"] = _receipt_with(valid["receipt"], lambda body: body.__setitem__(field, value))
        return valid

    with pytest.raises(CircuitCatalogError, match="receipt"):
        _run(catalog, trace, capture, plan, replacement, callback=backend)
    assert target in plan.to_dict()["intervention"]["value_ids"]


def test_branch_verifier_rejects_wrong_roots_even_if_backend_ignores_them():
    catalog, trace, capture, plan, target, replacement = _fixture()
    valid_roots = capture.initial_values

    def backend(*, plan, trace, roots, replacements, native_output, **kwargs):
        del roots
        return _valid_backend(
            plan=plan, trace=trace, roots=valid_roots, replacements=replacements,
            native_output=native_output, **kwargs,
        )

    bad_roots = dict(capture.initial_values)
    bad_roots[next(iter(bad_roots))] = torch.tensor([9.0, 9.0])
    with pytest.raises(CircuitCatalogError, match="root"):
        _run(catalog, trace, capture, plan, replacement, roots=bad_roots, callback=backend)
    assert target in plan.to_dict()["intervention"]["value_ids"]


def test_branch_verifier_rejects_replacement_digest_mismatch():
    catalog, trace, capture, plan, target, replacement = _fixture()
    _register(catalog)
    with pytest.raises(CircuitCatalogError, match="replacement digest"):
        catalog.execute(
            plan,
            trace=trace,
            roots=capture.initial_values,
            replacements={target: replacement + 1},
            native_output=capture.output,
        )


@pytest.mark.parametrize("mutation, message", [
    (lambda body: body["intervention_sites"][0].__setitem__("dose", 0.5), "dose"),
    (lambda body: body["intervention_sites"][0].__setitem__("value_id", "v000000"), "site"),
    (lambda body: body["intervention_sites"][0].__setitem__("replacement_digest", "0" * 64), "replacement digest"),
    (lambda body: body["native_prefix"]["steps"][0].__setitem__("operator", "aten.mul.Tensor"), "prefix"),
    (lambda body: body["suffix_observations"].append({}), "suffix"),
    (lambda body: body["output_differences"][0].__setitem__("candidate_digest", "0" * 64), "output claims"),
])
def test_branch_verifier_rejects_tampered_receipt(mutation, message):
    catalog, trace, capture, plan, _, replacement = _fixture()

    def backend(*, plan, trace, roots, replacements, native_output, **kwargs):
        valid = _valid_backend(
            plan=plan, trace=trace, roots=roots, replacements=replacements,
            native_output=native_output, **kwargs,
        )
        valid["receipt"] = _receipt_with(valid["receipt"], mutation)
        return valid

    with pytest.raises(CircuitCatalogError, match=message):
        _run(catalog, trace, capture, plan, replacement, callback=backend)


def test_branch_verifier_binds_the_callback_dose():
    catalog, trace, capture, plan, _, replacement = _fixture()
    _register(catalog)
    with pytest.raises(CircuitCatalogError, match="backend dose"):
        catalog.execute(
            plan,
            trace=trace,
            roots=capture.initial_values,
            replacements={plan.to_dict()["intervention"]["value_ids"][0]: replacement},
            native_output=capture.output,
            dose=0.5,
        )


def test_record_outcome_rejects_a_fake_branch_output_claim():
    catalog, trace, capture, plan, _, replacement = _fixture()
    valid = _valid_backend(
        plan=plan.to_dict(), trace=trace, roots=capture.initial_values,
        replacements={plan.to_dict()["intervention"]["value_ids"][0]: replacement},
        native_output=capture.output,
    )
    fake = _receipt_with(
        valid["receipt"],
        lambda body: body["output_differences"][0].__setitem__("candidate_digest", "0" * 64),
    )
    with pytest.raises(CircuitCatalogError, match="output claims"):
        catalog.record_outcome(plan, receipt=fake, output=valid["output"], persist=False)


def test_plan_rejects_root_or_unknown_site_and_nonfinite_dose():
    catalog, trace, capture, plan, target, replacement = _fixture()
    del plan, replacement
    with pytest.raises(CircuitCatalogError, match="intervention sites differ"):
        catalog.plan(
            "circuit://toy/branch",
            route_id="readout",
            intervention={
                "intervention_id": "replace-output",
                "value_ids": [next(iter(capture.initial_values))],
                "dose": 1.0,
                "replacement_digests": {target: "0" * 64},
            },
            persist=False,
        )
    with pytest.raises(CircuitCatalogError, match="intervention sites differ"):
        catalog.plan(
            "circuit://toy/branch",
            route_id="readout",
            intervention={
                "intervention_id": "replace-output",
                "value_ids": ["v999999"],
                "dose": 1.0,
                "replacement_digests": {"v999999": "0" * 64},
            },
            persist=False,
        )
    with pytest.raises(CircuitCatalogError, match="finite"):
        catalog.plan(
            "circuit://toy/branch",
            route_id="readout",
            intervention={
                "intervention_id": "replace-output",
                "value_ids": [target],
                "dose": float("nan"),
                "replacement_digests": {target: "0" * 64},
            },
            persist=False,
        )
