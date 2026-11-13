"""Qualified nonlinear primitive substitutions at an observed single-block input.

SDPA route/value and SwiGLU gate/value are different typed factor families.
The source graph is independently orchestrated from original parameters; the
native constructions call the original primitive and full output projection.
Every component endpoint holds P's image residual and Q's remaining closure.
Natural P/Q references retain their complete source states instead.
"""
from __future__ import annotations

import hashlib
import runpy
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from functools import cache
from pathlib import Path

SCHEMA = "mdb-flux-primitive-family-writer-v1"
WRITER = "flux_parallel_family_writer.py"
HELPERS = {
    WRITER: "23c02673a04bfa8e0f28e6fd70a2f104395f3c22434acd0de1f550c8da0a372f",
    "flux_functional_denoiser.py": "cb794daa5818ec3f9107f23839a41010f9569148da3708e39cedd364dff21923",
}
CORE = ("R", "A", "M", "gate", "projection_return", "out", "image_in", "text_in", "image_out", "text_out")
FAMILIES = ("sdpa", "swiglu")
_NATIVE_LOCK = threading.Lock()


def require(value, message):
    if not value:
        raise ValueError(message)


@cache
def source_writer():
    for name, digest in HELPERS.items():
        with Path(__file__).with_name(name).open("rb") as stream:
            require(hashlib.file_digest(stream, "sha256").hexdigest() == digest, "frozen primitive dependency differs: " + name)
    return runpy.run_path(str(Path(__file__).with_name(WRITER)))


def source_ports(family):
    require(family in FAMILIES, "unsupported nonlinear family")
    return CORE + (("q_consumed", "k_consumed", "v_consumed") if family == "sdpa" else ("mlp_input",))


def factor_specs(family):
    """Typed source ports; modulation gate is distinct from the SwiGLU half."""
    require(family in FAMILIES, "unsupported nonlinear family")
    if family == "sdpa":
        return dict(route=dict(ports=["q_consumed", "k_consumed"], stage="positioned and normalized BHTD operands", coherent_bundle=True),
            value=dict(ports=["v_consumed"], stage="consumed BHTD value operand"))
    return dict(gate=dict(ports=["mlp_input"], stage="pre-SiLU first half of original fused MLP operand", half=0),
        value=dict(ports=["mlp_input"], stage="multiplicative second half of original fused MLP operand", half=1))


def functional_single(adapter, state, index=4, *, family, expected_model_identity,
                      allow_cpu_fixture=False, observe=None):
    return source_writer()["functional_single"](adapter, state, index,
        expected_model_identity=expected_model_identity, retain_ports=source_ports(family),
        allow_cpu_fixture=allow_cpu_fixture, observe=observe)


@contextmanager
def native_single(adapter, state, index=4, *, family, expected_model_identity,
                  allow_cpu_fixture=False, observe=None):
    with source_writer()["native_single"](adapter, state, index,
            expected_model_identity=expected_model_identity, retain_ports=source_ports(family),
            allow_cpu_fixture=allow_cpu_fixture, observe=observe) as result:
        yield result


def restore_layout(value, declared):
    """Restore a consumed view's actual dense/padded strides, preserving bits.

    NPY and snapshot layouts are not the original dispatch layout. This routine
    does no dtype/device conversion and records no invented storage offset.
    Padding is unconsumed; only logical element bytes are authenticated.
    """
    import torch

    writer = source_writer()
    actual = writer["descriptor"](value)
    require(isinstance(declared, Mapping) and set(actual) <= set(declared), "missing declared live layout")
    for key in ("shape", "dtype", "device", "content_sha256", "hash_order"):
        require(actual[key] == declared[key], "retained operand differs from declared source: " + key)
    stride = declared["stride"]
    require(isinstance(stride, list) and len(stride) == value.ndim
        and all(type(item) is int and item > 0 for item in stride), "unsupported source operand stride")
    # Sorted physical strides must describe a nonoverlapping dense or padded
    # view. Batch-one strides are retained although they do not affect bytes.
    extent = 1
    for axis in sorted(range(value.ndim), key=lambda item: stride[item]):
        if value.shape[axis] > 1:
            require(stride[axis] >= extent, "overlapping source operand layout")
            extent = stride[axis] * (value.shape[axis] - 1) + extent
    result = torch.empty_strided(tuple(value.shape), tuple(stride), dtype=value.dtype, device=value.device)
    with torch.inference_mode():
        result.copy_(value)
    require(writer["descriptor"](result) == dict(declared), "restored live operand layout differs")
    return result


def _qualify_pair(q, p, family):
    writer = source_writer()
    require(isinstance(q, writer["Ports"]) and isinstance(p, writer["Ports"])
        and q.state is not None and p.state is not None, "live qualified source ports required")
    for key in ("model_identity", "point", "execution", "clock", "domain", "index", "sdpa_profile"):
        require(q.profile[key] == p.profile[key], "P/Q primitive qualification differs: " + key)
    for name in ("single_mod", "rope_cos", "rope_sin"):
        require(q.profile["interfaces"][name] == p.profile["interfaces"][name], "P/Q modulation/RoPE differs: " + name)
    for source in (q, p):
        source.adapter.validate_execution()
        require(set(source_ports(family)) <= set(source), "primitive source retention incomplete")
        for name in source_ports(family):
            require(writer["descriptor"](source[name])["content_sha256"] == source.descriptors[name]["content_sha256"],
                "retained primitive source mutated: " + name)
        require(writer["_same"](source["R"] + source["gate"] * source["projection_return"], source["out"]),
            "source finite gate/add closure differs")
    require(writer["_same"](q["gate"], p["gate"]), "P/Q same-clock modulation gates differ")
    return writer


def validate_action(action, family):
    require(family in FAMILIES and isinstance(action, Mapping), "unsupported primitive action")
    require(isinstance(action.get("id"), str) and action["id"], "primitive action requires an opaque id")
    role = action.get("role")
    require(role in ("natural-reference", "baseline", "candidate", "component-reference"), "unknown primitive action role")
    bindings = action.get("bindings", {})
    if role == "natural-reference":
        require(action.get("source") in ("P", "Q") and bindings == {}, "natural reference requires one full-state source")
    else:
        require(isinstance(bindings, Mapping) and set(bindings) == set(factor_specs(family))
            and all(value in ("P", "Q") for value in bindings.values()), "primitive factor bindings differ")
        selected = sum(value == "P" for value in bindings.values())
        require(selected == {"baseline": 0, "candidate": 1, "component-reference": 2}[role], "factor role differs from binding count")
    return dict(action)


def _native_sdpa(attn, values, descriptors):
    import torch.nn.functional as F
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    writer = source_writer()
    original = F.scaled_dot_product_attention
    receipt = dict(calls=0, consumed={}, returned=None)
    owner = threading.get_ident()

    def capture(*args, **kwargs):
        require(threading.get_ident() == owner and receipt["calls"] == 0, "native primitive SDPA occurrence differs")
        receipt["calls"] += 1
        for index, key in enumerate(("query", "key", "value")):
            value = kwargs[key] if key in kwargs else args[index]
            descriptor = writer["descriptor"](value)
            require(descriptor == descriptors[index], "actual native primitive consumed layout differs: " + key)
            receipt["consumed"][key] = descriptor
        expected = dict(attn_mask=None, dropout_p=0., is_causal=False, scale=None, enable_gqa=False)
        for key, default in expected.items():
            index = {"attn_mask": 3, "dropout_p": 4, "is_causal": 5}.get(key)
            actual = kwargs.get(key, args[index] if index is not None and len(args) > index else default)
            require(actual == default, "native primitive SDPA flag differs: " + key)
        receipt["flags"] = expected
        result = original(*args, **kwargs)
        receipt["returned"] = writer["descriptor"](result)
        return result

    require(_NATIVE_LOCK.acquire(blocking=False), "native primitive capture already active")
    try:
        require(not getattr(original, "_primitive_family_capture", False), "nested native primitive SDPA capture")
        capture._primitive_family_capture = True
        F.scaled_dot_product_attention = capture
        output = dispatch_attention_fn(*(value.permute(0, 2, 1, 3) for value in values), attn_mask=None,
            backend=attn.processor._attention_backend, parallel_config=attn.processor._parallel_config)
        require(receipt["calls"] == 1, "native dispatcher did not consume original SDPA")
        return output.permute(0, 2, 1, 3), receipt
    finally:
        F.scaled_dot_product_attention = original
        _NATIVE_LOCK.release()


def _primitive(q, p, action, family, native, plan):
    import torch
    import torch.nn.functional as F

    from mdb.computation_protocol import run_graph, verify_plan

    writer = source_writer()
    verify_plan(plan)
    specification = plan["specification"]
    receiver = dict(R="P", gate="Q", postblock_text="Q", conditioning="Q", latent="Q")
    receiver["M" if family == "sdpa" else "A"] = "Q"
    require(action in plan["actions"] and specification["receiver"] == receiver,
        "compiled action or receiving closure differs")
    sources = dict(Q=q, P=p)
    attn = q.adapter.model.single_transformer_blocks[q.profile["index"]].attn
    bindings, operands = action["bindings"], {}
    if family == "sdpa":
        require(specification["factors"] == [dict(id="route", ports=["q", "k"]), dict(id="value", ports=["v"])]
            and len(specification["nodes"]) == 1 and specification["nodes"][0]["op"] == "sdpa"
            and specification["nodes"][0]["inputs"] == ["q", "k", "v"], "compiled SDPA graph differs")
        graph_sources = dict(P={}, Q={})
        for name, factor in (("q_consumed", "route"), ("k_consumed", "route"), ("v_consumed", "value")):
            source = sources[bindings[factor]]
            operands[name] = restore_layout(source[name], source.descriptors[name])
            graph_sources[bindings[factor]][name[0]] = operands[name]
        inputs = [operands[name] for name in ("q_consumed", "k_consumed", "v_consumed")]
        declared = [writer["descriptor"](value) for value in inputs]
        receipts = []

        def sdpa(*values):
            require(all(writer["descriptor"](value) == expected for value, expected in zip(values, declared, strict=True)), "generated SDPA operand differs")
            if native:
                raw, record = _native_sdpa(attn, values, declared)
            else:
                raw = F.scaled_dot_product_attention(*values, attn_mask=None, dropout_p=0., is_causal=False, scale=None, enable_gqa=False)
                record = dict(calls=1, consumed=dict(zip(("query", "key", "value"), declared, strict=True)), returned=writer["descriptor"](raw),
                    flags=q.profile["sdpa_profile"])
            receipts.append(record)
            return raw

        raw = run_graph(plan, graph_sources, bindings, dict(sdpa=sdpa), check=_graph_check)
        require(len(receipts) == 1, "generated SDPA execution census differs")
        receipt = receipts[0]
        result = raw.permute(0, 2, 1, 3).flatten(2, 3).to(inputs[0].dtype)
    else:
        nodes = {node["op"]: node for node in specification["nodes"]}
        require(specification["factors"] == [dict(id="gate", ports=["G"]), dict(id="value", ports=["U"])]
            and len(specification["nodes"]) == 2
            and set(nodes) == {"silu", "multiply"}
            and nodes["silu"]["inputs"] == ["G"]
            and nodes["multiply"]["inputs"] == [nodes["silu"]["id"], "U"]
            and specification["output"] == nodes["multiply"]["id"], "compiled SwiGLU graph differs")
        original = restore_layout(q["mlp_input"], q.descriptors["mlp_input"])
        require(original.shape[-1] % 2 == 0, "SwiGLU operand width is odd")
        half = original.shape[-1] // 2
        for factor, index in (("gate", 0), ("value", 1)):
            source = sources[bindings[factor]]
            require(source.descriptors["mlp_input"]["stride"] == q.descriptors["mlp_input"]["stride"], "SwiGLU source layout differs")
            original[..., index*half:(index+1)*half] = source["mlp_input"][..., index*half:(index+1)*half]
        operands["mlp_input"] = original
        consumed = writer["descriptor"](original)
        gate_input, value_input = original[..., :half], original[..., half:]
        graph_sources = dict(P={}, Q={})
        graph_sources[bindings["gate"]]["G"] = gate_input
        graph_sources[bindings["value"]]["U"] = value_input
        if native:
            seen, gate_returns, actual_returns = [], [], []

            def silu(value):
                require(writer["descriptor"](value) == writer["descriptor"](gate_input), "generated native SiLU operand differs")
                handles = [attn.mlp_act_fn.register_forward_pre_hook(lambda _module, args: seen.append(writer["descriptor"](args[0]))),
                    attn.mlp_act_fn.gate_fn.register_forward_hook(lambda _module, _args, result: gate_returns.append(result))]
                try:
                    actual_returns.append(attn.mlp_act_fn(original))
                finally:
                    for handle in reversed(handles):
                        handle.remove()
                require(seen == [consumed] and len(gate_returns) == 1, "native SwiGLU module receipts differ")
                return gate_returns[0]

            def multiply(left, right):
                require(len(actual_returns) == 1 and left is gate_returns[0] and right is value_input,
                    "generated native product dependency differs")
                require(writer["_same"](left * right, actual_returns[0]), "actual native SwiGLU product closure differs")
                return actual_returns[0]

            result = run_graph(plan, graph_sources, bindings, dict(silu=silu, multiply=multiply), check=_graph_check)
        else:
            result = run_graph(plan, graph_sources, bindings, dict(silu=F.silu, multiply=torch.mul), check=_graph_check)
        receipt = dict(calls=1, consumed=dict(mlp_input=consumed), returned=writer["descriptor"](result),
            slices=dict(gate=[0, half], value=[half, 2*half]))
    require(bool(torch.isfinite(result).all()), "nonfinite primitive result")
    return result, operands, receipt


def _graph_check(name, value, port):
    require(list(value.shape) == port["shape"] and str(value.dtype) == port["dtype"], "generated typed operand differs: " + name)


def construct_endpoint(q_ports, p_ports, action, *, family, plan, native=False):
    """Execute one declared factor action with full primitive/projection geometry.

    Both-source primitive is a component reference, not natural P. P's image
    residual is fixed even for the QQ-factor baseline. Q supplies modulation,
    the other branch, postblock text, conditioning and all remaining state.
    """
    import torch

    from mdb.computation_protocol import verify_plan

    require(type(native) is bool, "native lane flag must be explicit bool")
    action = validate_action(action, family)
    writer = _qualify_pair(q_ports, p_ports, family)
    verify_plan(plan)
    require(action in plan["actions"] and plan["specification"]["model_identity"] == q_ports.profile["model_identity"],
        "compiled action/model identity differs")
    if action["role"] == "natural-reference":
        source = {"P": p_ports, "Q": q_ports}[action["source"]]
        state = writer["_copy"](source.state)
        return dict(image=writer["_snapshot"](state["image"]), text=writer["_snapshot"](state["text"]), state=state,
            ports={}, descriptors={}, construction=dict(schema=SCHEMA, action=action, family=family,
                origin="complete natural source state", native_primitive_calls=0, native_projection_calls=0,
                functional_primitive_calls=0, functional_projection_calls=0))
    with torch.inference_mode():
        primitive, operands, receipt = _primitive(q_ports, p_ports, action, family, native, plan)
        target = "A" if family == "sdpa" else "M"
        if action["role"] in ("baseline", "component-reference"):
            source = q_ports if action["role"] == "baseline" else p_ports
            require(writer["_same"](primitive, source[target]), "primitive all-source copy closure differs")
        ntext = q_ports.state["text"].shape[1]
        r, a, m = (writer["_snapshot"](q_ports[name]) for name in ("R", "A", "M"))
        r[:, ntext:] = p_ports["R"][:, ntext:]
        (a if family == "sdpa" else m)[:, ntext:] = primitive[:, ntext:]
        merge = torch.cat([a, m], -1)
        module = q_ports.adapter.model.single_transformer_blocks[q_ports.profile["index"]].attn.to_out
        seen = []
        handle = module.register_forward_pre_hook(lambda _module, args: seen.append(writer["descriptor"](args[0]))) if native else None
        try:
            projection = module(merge) if native else writer["equations"]()["_linear"](module, merge)
        finally:
            if handle is not None:
                handle.remove()
        require(not native or seen == [writer["descriptor"](merge)], "native full projection operand differs")
        write = q_ports["gate"] * projection
        out = r + write
        state = writer["_copy"](q_ports.state)
        state["image"] = writer["_snapshot"](out[:, ntext:])
        values = dict(primitive_return=primitive, merge_input=merge, projection_return=projection, write=write, out=out, **operands)
        result = dict(image=writer["_snapshot"](state["image"]), text=writer["_snapshot"](state["text"]), state=state,
            ports=values, descriptors={name: writer["descriptor"](value) for name, value in values.items()},
            construction=dict(schema=SCHEMA, action=action, family=family, factors=factor_specs(family),
                fixed_bindings=dict(R="P image rows", A="Q" if family == "swiglu" else "selected primitive image rows; Q text rows",
                    M="Q" if family == "sdpa" else "selected primitive image rows; Q text rows", gate="Q", postblock_text="Q", conditioning="Q"),
                full_primitive_shape=list(primitive.shape), full_projection_shape=list(merge.shape), primitive_receipt=receipt,
                native_primitive_calls=int(native), native_projection_calls=int(native),
                functional_primitive_calls=int(not native), functional_projection_calls=int(not native), gate_multiply=1, residual_add=1))
    q_ports.adapter.validate_execution()
    return result
