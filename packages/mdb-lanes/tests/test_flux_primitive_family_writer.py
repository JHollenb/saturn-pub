"""Installed public graph + small fixed-matrix native modules, no model assets."""
from __future__ import annotations

import copy
import hashlib
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from saturn_pub.core import Session
from saturn_pub.values import describe

from mdb_lanes.computation_protocol import compile_protocol

HERE = Path(__file__).resolve().parent
W = runpy.run_path(str(HERE.parent / "qualification/flux_primitive_family_writer.py"))
FIXTURE = HERE / "test_flux_parallel_family_writer.py"
assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == "157385f95ffee58b85197dd86054798e478a568686c8c76da9e04ad703083364"
TOY = runpy.run_path(str(FIXTURE))
bits = TOY["bits"]


def plan(family, identity):
    if family == "sdpa":
        ports = {name: dict(shape=[1, 2, 5, 4], dtype="torch.bfloat16", stage="consumed original BHTD") for name in ("q", "k", "v")}
        factors = [dict(id="route", ports=["q", "k"]), dict(id="value", ports=["v"])]
        nodes, output = [dict(id="attention", op="sdpa", inputs=["q", "k", "v"])], "attention"
    else:
        ports = {name: dict(shape=[1, 5, 16], dtype="torch.bfloat16", stage="raw preactivation half") for name in ("G", "U")}
        factors = [dict(id="gate", ports=["G"]), dict(id="value", ports=["U"])]
        nodes = [dict(id="activation", op="silu", inputs=["G"]), dict(id="product", op="multiply", inputs=["activation", "U"])]
        output = "product"
    receiver = dict(R="P", gate="Q", postblock_text="Q", conditioning="Q", latent="Q")
    receiver["M" if family == "sdpa" else "A"] = "Q"
    return compile_protocol(dict(schema="mdb-computation-family-spec-v2", family_id="fixture-" + family,
        model_identity=identity, endpoint_adapter="single4-original-primitives", source_policy="synthetic owned matrices",
        ports=ports, factors=factors, nodes=nodes, output=output,
        receiver=receiver,
        clock=dict(step=0, phase="single", index=4), max_contexts=1,
        continuation=dict(consumer="fixture next block", transitions=1),
        metric=dict(observable="supplied axis", units="dimensionless", reference="natural P/Q", scope="fixture", direction="minimize")))


def captures(family, native=False):
    adapter, qs, ps = TOY["pair"]()
    values = []
    for state in (qs, ps):
        if native:
            with W["native_single"](adapter, state, family=family, expected_model_identity=adapter.model_identity,
                    allow_cpu_fixture=True) as result:
                session = Session(adapter, state)
                session.continue_()
            assert describe(result["state"]) == describe(session.capture(retain=False).payload)
        else:
            result = W["functional_single"](adapter, state, family=family, expected_model_identity=adapter.model_identity,
                allow_cpu_fixture=True)
        values.append(result)
    return adapter, values, plan(family, adapter.model_identity)


def test_definitions_do_not_import_frameworks():
    code = "import runpy,sys;runpy.run_path(sys.argv[1]);assert not {'torch','diffusers','saturn_pub','numpy','mdb'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(HERE.parent / "qualification/flux_primitive_family_writer.py")], check=True)


@pytest.mark.parametrize("family", ["sdpa", "swiglu"])
def test_original_source_capture_and_dispatch_layout_restoration(family):
    _, functional, _ = captures(family)
    _, native, _ = captures(family, native=True)
    for predicted, target in zip(functional, native, strict=True):
        assert set(predicted["ports"]) == set(target["ports"]) == set(W["source_ports"](family))
        for name in predicted["ports"]:
            assert bits(predicted["ports"][name], target["ports"][name]), name
        name = "v_consumed" if family == "sdpa" else "mlp_input"
        source = target["ports"]
        assert source[name].stride() != tuple(source.descriptors[name]["stride"]), "fixture must expose padded fused view"
        restored = W["restore_layout"](source[name], source.descriptors[name])
        assert W["source_writer"]()["descriptor"](restored) == source.descriptors[name]
        assert bits(restored, source[name]) and restored.data_ptr() != source[name].data_ptr()
        bad = copy.deepcopy(source.descriptors[name])
        bad["stride"] = [1] * restored.ndim
        with pytest.raises(ValueError, match="overlapping"):
            W["restore_layout"](source[name], bad)


@pytest.mark.parametrize("family", ["sdpa", "swiglu"])
def test_generated_graph_endpoints_use_original_native_primitives_and_full_projection(family):
    adapter, results, compiled = captures(family)
    q, p = [row["ports"] for row in results]
    ntext = q.state["text"].shape[1]
    attn = adapter.model.single_transformer_blocks[4].attn
    seen = []
    handle = attn.to_out.register_forward_pre_hook(lambda _module, args: seen.append(tuple(args[0].shape)))
    values = {}
    try:
        for action in compiled["actions"]:
            predicted = W["construct_endpoint"](q, p, action, family=family, plan=compiled)
            target = W["construct_endpoint"](q, p, action, family=family, plan=compiled, native=True)
            assert bits(predicted["image"], target["image"]) and bits(predicted["text"], target["text"])
            if action["role"] == "natural-reference":
                reference = {"P": p, "Q": q}[action["source"]]
                assert describe(target["state"]) == describe(reference.state)
                assert target["construction"]["native_projection_calls"] == 0
            else:
                assert bits(target["text"], q["text_out"]) and bits(target["state"]["conditioning"], q.state["conditioning"])
                assert bits(target["ports"]["out"][:, ntext:], target["image"])
                expected_r = q["R"].clone()
                expected_r[:, ntext:] = p["R"][:, ntext:]
                assert bits(expected_r + q["gate"] * target["ports"]["projection_return"], target["ports"]["out"])
                assert target["construction"]["full_projection_shape"] == [1, 5, 24]
                receipt = target["construction"]["primitive_receipt"]
                assert receipt["calls"] == 1 and target["construction"]["native_projection_calls"] == 1
                if family == "sdpa":
                    for key, factor, name in (("query", "route", "q_consumed"), ("key", "route", "k_consumed"), ("value", "value", "v_consumed")):
                        assert receipt["consumed"][key] == {"P": p, "Q": q}[action["bindings"][factor]].descriptors[name]
                else:
                    assert receipt["consumed"]["mlp_input"]["stride"] == q.descriptors["mlp_input"]["stride"]
            values[action["id"]] = predicted
    finally:
        handle.remove()
    assert seen == [(1, 5, 24)] * 4
    assert bits(values["both"]["ports"]["primitive_return"], p["A" if family == "sdpa" else "M"])
    if family == "sdpa":
        prior = W["source_writer"]()["construct_endpoint"](q, p, "RA")
        assert bits(values["both"]["image"], prior["image"]), "SDPA PP must close the earlier useful RA construction"
    assert not bits(values["baseline"]["image"], q["image_out"]), "fixed R_P baseline must remain distinct from natural Q"
    assert not bits(values[compiled["candidate_menu"][0]]["ports"]["primitive_return"], values[compiled["candidate_menu"][1]]["ports"]["primitive_return"])


@pytest.mark.parametrize("family", ["sdpa", "swiglu"])
def test_functional_graph_uses_no_module_and_rejects_changed_identity_or_receiver(family, monkeypatch):
    _, rows, compiled = captures(family)
    q, p = [row["ports"] for row in rows]
    monkeypatch.setattr(torch.nn.Module, "__call__", lambda *_args, **_kwargs: pytest.fail("functional graph called a native module"))
    for action in compiled["actions"]:
        W["construct_endpoint"](q, p, action, family=family, plan=compiled)
    changed = copy.deepcopy(compiled["specification"])
    changed["receiver"]["R"] = "Q"
    other = compile_protocol(changed)
    with pytest.raises(ValueError, match="receiving closure"):
        W["construct_endpoint"](q, p, other["actions"][2], family=family, plan=other)
    name = "v_consumed" if family == "sdpa" else "mlp_input"
    with torch.inference_mode():
        p[name][(0,) * p[name].ndim] += 1
    with pytest.raises(ValueError, match="source mutated"):
        W["construct_endpoint"](q, p, compiled["actions"][2], family=family, plan=compiled)


def test_native_sdpa_failure_restores_global_dispatch_and_module_hooks(monkeypatch):
    _, rows, compiled = captures("sdpa")
    q, p = [row["ports"] for row in rows]
    original = F.scaled_dot_product_attention

    def fail(*_args, **_kwargs):
        raise RuntimeError("controlled primitive failure")

    monkeypatch.setattr(F, "scaled_dot_product_attention", fail)
    with pytest.raises(RuntimeError, match="controlled primitive failure"):
        W["construct_endpoint"](q, p, compiled["actions"][2], family="sdpa", plan=compiled, native=True)
    assert F.scaled_dot_product_attention is fail
    monkeypatch.setattr(F, "scaled_dot_product_attention", original)
    W["construct_endpoint"](q, p, compiled["actions"][2], family="sdpa", plan=compiled, native=True)


def test_natural_references_are_owned_and_unknown_roles_or_ports_do_not_run():
    _, rows, compiled = captures("swiglu")
    q, p = [row["ports"] for row in rows]
    result = W["construct_endpoint"](q, p, compiled["actions"][1], family="swiglu", plan=compiled)
    with torch.inference_mode():
        result["state"]["image"].zero_()
        result["state"]["conditioning"].zero_()
    assert torch.count_nonzero(p.state["image"]) and torch.count_nonzero(p.state["conditioning"])
    invalid = dict(id="opaque", role="candidate", bindings=dict(gate="P", value="P"))
    with pytest.raises(ValueError, match="binding count"):
        W["construct_endpoint"](q, p, invalid, family="swiglu", plan=compiled)
