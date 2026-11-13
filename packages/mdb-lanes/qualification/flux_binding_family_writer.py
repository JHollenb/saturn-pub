"""Explicit joint0 writer ports and recipient-gated text-return interventions.

The functional lane uses the frozen original shared primitive equations. The
native lane only installs hooks around one caller-executed original joint0.
Both replace ungated text returns on declared rows, then retain the recipient's
gates and finite additions. Changing attention recomputes the downstream MLP.
"""
from __future__ import annotations

import hashlib
import json
import runpy
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from functools import cache
from pathlib import Path

SCHEMA = "mdb-flux-binding-family-writer-v1"
GRAPH = "flux_functional_denoiser.py"
HELPERS = {GRAPH: "cb794daa5818ec3f9107f23839a41010f9569148da3708e39cedd364dff21923"}
RETURN_NAMES = ("attention_return", "mlp_return")
TEXT_PORTS = dict(R="text_in", A="text_attention_return_used", gate_A="text_attention_gate",
                  U="text_after_attention", x="text_mlp_input", M="text_mlp_return_used",
                  gate_M="text_mlp_gate", T="text_out")


def require(value, message):
    if not value:
        raise ValueError(message)


@cache
def equations():
    path = Path(__file__).with_name(GRAPH)
    with path.open("rb") as stream:
        require(hashlib.file_digest(stream, "sha256").hexdigest() == HELPERS[GRAPH], "frozen functional graph differs")
    return runpy.run_path(str(path))


def descriptor(value):
    """A JSON descriptor of the actual tensor, including stride and bytes."""
    import torch

    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                device=str(value.device), content_sha256=hashlib.sha256(raw).hexdigest())


def _same(left, right):
    import torch

    return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
            and left.shape == right.shape and left.dtype == right.dtype and left.device == right.device
            and left.stride() == right.stride()
            and torch.equal(left.detach().contiguous().view(torch.uint8), right.detach().contiguous().view(torch.uint8)))


def _snapshot(value):
    import torch

    result = value.detach().clone(memory_format=torch.preserve_format)
    require(result.stride() == value.stride(), "unsupported nondense tensor layout")
    return result


def qualify(adapter, state, *, expected_model_identity, allow_cpu_fixture=False):
    """Validate a caller-bound original model/profile and the actual joint0 cut.

    The explicit CPU fixture lane is arithmetic mechanics evidence only. The
    production domain is original B1 CUDA BF16 512-text/1024-image Klein4B.
    No model identity is assigned or replaced here.
    """
    import torch

    require(type(allow_cpu_fixture) is bool, "explicit fixture flag required")
    require(isinstance(expected_model_identity, str) and expected_model_identity
            and adapter.model_identity == expected_model_identity, "writer model identity differs")
    adapter.validate_execution()
    adapter.validate(state)
    point = adapter.point(state).to_dict()
    require(point["family"] == "flux2-klein" and point["phase"] == "joint"
            and point["index"] == 0 and point["next_operation"] == "joint_block.0"
            and state["phase"] == "joint" and state["cursor"] == 0, "writer requires exact joint0 entry")
    image, text, context = (state[name] for name in ("image", "text", "context"))
    require(isinstance(context, Mapping) and {"img_mod", "txt_mod", "rope"} <= set(context), "incomplete writer context")
    require(image.ndim == text.ndim == 3 and image.shape[0] == text.shape[0] == 1
            and image.shape[-1] == text.shape[-1] and image.dtype == text.dtype == torch.bfloat16
            and image.device == text.device, "writer requires paired batch-one BF16 streams")
    device = "cpu" if allow_cpu_fixture else "cuda"
    require(image.device.type == device, "writer device/profile differs")
    require(not adapter.model.training, "writer requires frozen eval model")
    if not allow_cpu_fixture:
        from saturn_pub.adapters.flux2 import Flux2KleinAdapter

        require(isinstance(adapter, Flux2KleinAdapter), "original public Klein adapter required")
        require(tuple(image.shape) == (1, 1024, 3072) and tuple(text.shape) == (1, 512, 3072)
                and len(adapter.model.transformer_blocks) == 5 and len(adapter.model.single_transformer_blocks) == 20,
                "unqualified original writer geometry")
        equations()["validate_model"](adapter.model)
        require(str(torch.__version__).split("+")[0] == "2.13.0", "unqualified Torch writer profile")
    width = text.shape[-1]
    values = dict(image=image, text=text)
    for name in ("img_mod", "txt_mod"):
        value = context[name]
        require(value.dtype == image.dtype and value.device == image.device
                and tuple(value.shape) in ((1, 6 * width), (1, 1, 6 * width)), "writer modulation profile differs")
        values[name] = value
    rope = context["rope"]
    require(isinstance(rope, tuple) and len(rope) == 2, "writer requires explicit RoPE pair")
    for name, value in zip(("rope_cos", "rope_sin"), rope, strict=True):
        require(value.ndim == 2 and value.shape[0] == image.shape[1] + text.shape[1]
                and value.dtype == torch.float32 and value.device == image.device, "writer RoPE profile differs")
        values[name] = value
    require(all(bool(torch.isfinite(value).all()) for value in values.values()), "nonfinite writer state")
    return dict(schema=SCHEMA, model_identity=adapter.model_identity,
        execution=json.loads(json.dumps(dict(adapter.execution), sort_keys=True, allow_nan=False)),
        point=point, interfaces={name: descriptor(value) for name, value in values.items()},
        domain="CPU arithmetic fixture only" if allow_cpu_fixture else "original Klein4B B1 CUDA BF16 512-text/1024-image joint0",
        raw_prompt_prediction=False, independent_kernel_prediction=False,
        intervention="ungated return row SET before unchanged recipient gate and finite addition", port_names=TEXT_PORTS,
        gated_write_origin="functional lane: consumed primitive result; native lane: derived same-profile product",
        landed_state_origin="functional own arithmetic or actual native norm2 prehook/block-return")


def _interventions(text, rows, attention_return, mlp_return):
    import torch

    require(isinstance(rows, (tuple, list)) and list(rows) == sorted(set(rows))
            and all(type(row) is int and 0 <= row < text.shape[1] for row in rows), "invalid declared text rows")
    payloads, receipts = {}, {}
    for name, value in zip(RETURN_NAMES, (attention_return, mlp_return), strict=True):
        if value is not None:
            require(isinstance(value, torch.Tensor) and value.shape == text.shape and value.dtype == text.dtype
                    and value.device == text.device and bool(torch.isfinite(value).all()), "replacement return profile differs: " + name)
            payloads[name] = _snapshot(value)
        receipts[name] = dict(enabled=value is not None, rows=list(rows),
            operator="row SET of ungated return, no normalization or recast",
            replacement=None if value is None else descriptor(value))
    return payloads, receipts


def _replace(computed, payloads, name, rows):
    if name not in payloads or not rows:
        return computed
    used = _snapshot(computed)
    used[:, rows] = payloads[name][:, rows]
    return used


def functional_joint0(adapter, state, *, expected_model_identity, text_rows=(),
                      attention_return=None, mlp_return=None, allow_cpu_fixture=False, observe=None):
    """Return {state, ports, profile, interventions}; call no native module.

    Replacements are complete text-return tensors. Only declared rows are read
    into the computed return. All other rows preserve computed return bits.
    """
    import torch

    profile = qualify(adapter, state, expected_model_identity=expected_model_identity, allow_cpu_fixture=allow_cpu_fixture)
    graph = equations()
    image, text, context = (state[name] for name in ("image", "text", "context"))
    payloads, interventions = _interventions(text, text_rows, attention_return, mlp_return)
    ports = {}

    def record(name, value):
        ports[name] = _snapshot(value)
        if observe is not None:
            observe(name, _snapshot(value))

    with torch.inference_mode():
        block = adapter.model.transformer_blocks[0]
        (shift, scale, gate), (shift_ff, scale_ff, gate_ff) = graph["_mods"](context["img_mod"], 2)
        (cs, cc, cg), (cs_ff, cc_ff, cg_ff) = graph["_mods"](context["txt_mod"], 2)
        record("image_in", image)
        record("text_in", text)
        for name, value in dict(image_attention_gate=gate, image_mlp_gate=gate_ff,
                text_attention_gate=cg, text_mlp_gate=cg_ff,
                image_attention_shift=shift, image_attention_scale=scale,
                image_mlp_shift=shift_ff, image_mlp_scale=scale_ff,
                text_attention_shift=cs, text_attention_scale=cc,
                text_mlp_shift=cs_ff, text_mlp_scale=cc_ff).items():
            record(name, value)
        x = (1 + scale) * graph["_norm"](block.norm1, image) + shift
        c = (1 + cc) * graph["_norm"](block.norm1_context, text) + cs
        record("image_attention_input", x)
        record("text_attention_input", c)
        attn = block.attn
        q, k, v = (graph["_linear"](getattr(attn, name), x).unflatten(-1, (-1, attn.head_dim)) for name in ("to_q", "to_k", "to_v"))
        cq, ck, cv = (graph["_linear"](getattr(attn, name), c).unflatten(-1, (-1, attn.head_dim)) for name in ("add_q_proj", "add_k_proj", "add_v_proj"))
        q = torch.cat([graph["_rms"](attn.norm_added_q, cq), graph["_rms"](attn.norm_q, q)], dim=1)
        k = torch.cat([graph["_rms"](attn.norm_added_k, ck), graph["_rms"](attn.norm_k, k)], dim=1)
        out = graph["_attention"](q, k, torch.cat([cv, v], dim=1), context["rope"])
        cout, out = out[:, :text.shape[1]], out[:, text.shape[1]:]
        image_attention_return = graph["_linear"](attn.to_out[0], out)
        computed = graph["_linear"](attn.to_add_out, cout)
        used = _replace(computed, payloads, "attention_return", text_rows)
        record("image_attention_return", image_attention_return)
        record("text_attention_return_computed", computed)
        record("text_attention_return_used", used)
        image_write, text_write = gate * image_attention_return, cg * used
        record("image_attention_write", image_write)
        record("text_attention_write", text_write)
        image = image + image_write
        image_mlp_input = graph["_norm"](block.norm2, image) * (1 + scale_ff) + shift_ff
        image_mlp_return = graph["_ff"](block.ff, image_mlp_input)
        record("image_after_attention", image)
        record("image_mlp_input", image_mlp_input)
        record("image_mlp_return", image_mlp_return)
        image_write = gate_ff * image_mlp_return
        record("image_mlp_write", image_write)
        image = image + image_write
        text = text + text_write
        record("text_after_attention", text)
        text_mlp_input = graph["_norm"](block.norm2_context, text) * (1 + cc_ff) + cs_ff
        record("text_mlp_input", text_mlp_input)
        computed = graph["_ff"](block.ff_context, text_mlp_input)
        used = _replace(computed, payloads, "mlp_return", text_rows)
        record("text_mlp_return_computed", computed)
        record("text_mlp_return_used", used)
        text_write = cg_ff * used
        record("text_mlp_write", text_write)
        text = text + text_write
        record("image_out", image)
        record("text_out", text)
    _landings(ports, record)
    after = dict(state, image=image, text=text, cursor=1)
    if len(adapter.model.transformer_blocks) == 1:
        after.update(phase="single", cursor=0)
    return dict(state=after, ports=ports, profile=profile, interventions=interventions)


def _landings(ports, record):
    for stream in ("image", "text"):
        before, intermediate, after = (ports[stream + suffix] for suffix in ("_in", "_after_attention", "_out"))
        record(stream + "_attention_landed_delta", intermediate.double() - before.double())
        record(stream + "_mlp_landed_delta", after.double() - intermediate.double())


def _source_ports(q_ports, p_ports, rows):
    import torch

    q = q_ports["text_in"]
    _interventions(q, rows, None, None)
    for source in (q_ports, p_ports):
        for name in ("text_in", "text_attention_return_used", "text_after_attention", "text_mlp_return_used", "text_out"):
            value = source[name]
            require(value.shape == q.shape and value.dtype == q.dtype == torch.bfloat16
                    and value.device == q.device and bool(torch.isfinite(value).all()), "source writer port profile differs: " + name)
        for name in ("text_attention_gate", "text_mlp_gate"):
            gate = source[name]
            require(gate.shape == (1, 1, q.shape[-1]) and gate.dtype == q.dtype
                    and gate.device == q.device and bool(torch.isfinite(gate).all()), "source writer gate profile differs")
            require(_same(gate, q_ports[name]), "P/Q recipient writer gates differ")
        u = source["text_in"] + source["text_attention_gate"] * source["text_attention_return_used"]
        t = u + source["text_mlp_gate"] * source["text_mlp_return_used"]
        require(_same(u, source["text_after_attention"]) and _same(t, source["text_out"]), "saved writer ports do not close under native gate/add order")


def frozen_endpoint(q_ports, p_ports, rows, r_source, a_source, m_source):
    """Full Q-shaped endpoint for one frozen R/A/M corner; no model execution.

    P/Q port provenance is bound by their capture profiles and the caller's
    source custody. This tensor function validates finite gate/add closure. It
    computes complete-shaped arithmetic, then copies declared rows into Qout.
    """
    import torch

    with torch.inference_mode():
        _source_ports(q_ports, p_ports, rows)
        require(all(name in ("Q", "P") for name in (r_source, a_source, m_source)), "unknown frozen writer source")
        sources = dict(Q=q_ports, P=p_ports)
        r = sources[r_source]["text_in"]
        a = sources[a_source]["text_attention_return_used"]
        m = sources[m_source]["text_mlp_return_used"]
        u = r + q_ports["text_attention_gate"] * a
        value = u + q_ports["text_mlp_gate"] * m
        endpoint = _snapshot(q_ports["text_out"])
        endpoint[:, rows] = value[:, rows]
        if r_source == a_source == m_source:
            expected = _snapshot(q_ports["text_out"])
            expected[:, rows] = sources[r_source]["text_out"][:, rows]
            require(_same(endpoint, expected), "all-Q/all-P writer corner does not reproduce source endpoint bits")
        return endpoint


def _recomputed_endpoint(adapter, state, q_ports, p_ports, rows, r_source, a_source, *,
                         expected_model_identity, allow_cpu_fixture, native, observe):
    import torch

    profile = qualify(adapter, state, expected_model_identity=expected_model_identity, allow_cpu_fixture=allow_cpu_fixture)
    require(r_source in ("Q", "P") and a_source in ("Q", "P"), "unknown recomputed writer source")
    ports = {}

    def record(name, value):
        ports[name] = _snapshot(value)
        if observe is not None:
            observe(name, _snapshot(value))

    with torch.inference_mode():
        _source_ports(q_ports, p_ports, rows)
        require(_same(state["text"], q_ports["text_in"]), "recomputed writer parent must be the Q input")
        if "image_in" in q_ports:
            require(_same(state["image"], q_ports["image_in"]), "recomputed writer image parent differs")
        graph = equations()
        (_, _, gate), (shift, scale, gate_ff) = graph["_mods"](state["context"]["txt_mod"], 2)
        require(_same(gate, q_ports["text_attention_gate"]) and _same(gate_ff, q_ports["text_mlp_gate"]), "recomputed writer recipient modulation differs")
        sources = dict(Q=q_ports, P=p_ports)
        r = _snapshot(q_ports["text_in"])
        a = _snapshot(q_ports["text_attention_return_used"])
        r[:, rows] = sources[r_source]["text_in"][:, rows]
        a[:, rows] = sources[a_source]["text_attention_return_used"][:, rows]
        write = gate * a
        u = r + write
        block = adapter.model.transformer_blocks[0]
        normalized = block.norm2_context(u) if native else graph["_norm"](block.norm2_context, u)
        operand = normalized * (1 + scale) + shift
        m = block.ff_context(operand) if native else graph["_ff"](block.ff_context, operand)
        final_write = gate_ff * m
        t = u + final_write
        for name, value in dict(text_in=r, text_attention_gate=gate, text_attention_return_used=a,
                text_attention_write=write, text_after_attention=u, text_mlp_gate=gate_ff,
                text_mlp_shift=shift, text_mlp_scale=scale, text_mlp_input=operand,
                text_mlp_return_computed=m, text_mlp_return_used=m,
                text_mlp_write=final_write, text_out=t,
                text_attention_landed_delta=u.double()-r.double(), text_mlp_landed_delta=t.double()-u.double()).items():
            record(name, value)
        endpoint = _snapshot(q_ports["text_out"])
        endpoint[:, rows] = t[:, rows]
    adapter.validate_execution()
    profile = dict(profile, continuation_scope="Full text-shape original MLP from frozen R/A hybrid U; exact Qout copied outside selected rows",
                   lane="native-MLP-target" if native else "functional-MLP-forecast")
    return dict(endpoint=endpoint, ports=ports, profile=profile,
                interventions=dict(rows=list(rows), r_source=r_source, a_source=a_source, m_source="recomputed-own-MLP"),
                native_calls=dict(norm2_context=1, ff_context=1) if native else {})


def functional_recomputed_endpoint(adapter, state, q_ports, p_ports, rows, r_source, a_source, *,
                                   expected_model_identity, allow_cpu_fixture=False, observe=None):
    """Recompute only the original full-shaped text MLP by shared primitives."""
    return _recomputed_endpoint(adapter, state, q_ports, p_ports, rows, r_source, a_source,
        expected_model_identity=expected_model_identity, allow_cpu_fixture=allow_cpu_fixture, native=False, observe=observe)


def native_recomputed_endpoint(adapter, state, q_ports, p_ports, rows, r_source, a_source, *,
                               expected_model_identity, allow_cpu_fixture=False, observe=None):
    """One actual original norm2_context+ff_context target; no attention replay."""
    return _recomputed_endpoint(adapter, state, q_ports, p_ports, rows, r_source, a_source,
        expected_model_identity=expected_model_identity, allow_cpu_fixture=allow_cpu_fixture, native=True, observe=observe)


def _argument(args, kwargs, name, index):
    if name in kwargs:
        return kwargs[name]
    require(len(args) > index, "missing actual native joint0 operand: " + name)
    return args[index]


@contextmanager
def native_joint0(adapter, state, *, expected_model_identity, text_rows=(),
                  attention_return=None, mlp_return=None, allow_cpu_fixture=False, observe=None):
    """Capture/intervene one caller-executed original native joint0, no replay.

    Use ``with native_joint0(...) as captured: session.continue_()``. Hook cleanup
    is unconditional. The actual block entry must match the supplied receiver in
    full consumed bytes/layout; no other joint/single block may run in scope.
    """
    profile = qualify(adapter, state, expected_model_identity=expected_model_identity, allow_cpu_fixture=allow_cpu_fixture)
    block = adapter.model.transformer_blocks[0]
    payloads, interventions = _interventions(state["text"], text_rows, attention_return, mlp_return)
    expected = {name: _snapshot(state[name]) for name in ("image", "text")}
    context = state["context"]
    expected.update({name: _snapshot(context[name]) for name in ("img_mod", "txt_mod")})
    expected["rope"] = tuple(_snapshot(value) for value in context["rope"])
    ports, calls = {}, dict(block=0, attention=0, mlp=0, image_mlp=0)
    capture = dict(state=None, ports=ports, profile=profile, interventions=interventions, native_calls=calls)
    owner = threading.get_ident()

    def record(name, value):
        require(threading.get_ident() == owner, "native writer crossed threads")
        ports[name] = _snapshot(value)
        if observe is not None:
            observe(name, _snapshot(value))

    def entry(_module, args, kwargs):
        require(calls["block"] == 0, "native writer allows one joint0 invocation")
        calls["block"] += 1
        for field, name, index in (("image", "hidden_states", 0), ("text", "encoder_hidden_states", 1),
                                   ("img_mod", "temb_mod_img", 2), ("txt_mod", "temb_mod_txt", 3)):
            value = _argument(args, kwargs, name, index)
            require(_same(value, expected[field]), "actual native writer input differs: " + name)
        rope = _argument(args, kwargs, "image_rotary_emb", 4)
        require(isinstance(rope, tuple) and len(rope) == 2 and all(_same(a, b) for a, b in zip(rope, expected["rope"], strict=True)), "actual native writer RoPE differs")
        options = kwargs.get("joint_attention_kwargs", args[5] if len(args) > 5 else None)
        require(options in (None, {}), "native writer custom attention kwargs unsupported")
        record("image_in", expected["image"])
        record("text_in", expected["text"])
        graph = equations()
        (shift, scale, gate), (shift_ff, scale_ff, gate_ff) = graph["_mods"](expected["img_mod"], 2)
        (cs, cc, cg), (cs_ff, cc_ff, cg_ff) = graph["_mods"](expected["txt_mod"], 2)
        for name, value in dict(image_attention_gate=gate, image_mlp_gate=gate_ff,
                text_attention_gate=cg, text_mlp_gate=cg_ff,
                image_attention_shift=shift, image_attention_scale=scale,
                image_mlp_shift=shift_ff, image_mlp_scale=scale_ff,
                text_attention_shift=cs, text_attention_scale=cc,
                text_mlp_shift=cs_ff, text_mlp_scale=cc_ff).items():
            record(name, value)

    def attention_inputs(_module, args, kwargs):
        record("image_attention_input", _argument(args, kwargs, "hidden_states", 0))
        record("text_attention_input", _argument(args, kwargs, "encoder_hidden_states", 1))

    def attention(_module, _args, result):
        require(calls["block"] == 1 and calls["attention"] == 0, "native attention occurrence differs")
        calls["attention"] += 1
        require(isinstance(result, tuple) and len(result) == 2, "native attention return ABI differs")
        image, text = result
        used = _replace(text, payloads, "attention_return", text_rows)
        record("image_attention_return", image)
        record("text_attention_return_computed", text)
        record("text_attention_return_used", used)
        record("image_attention_write", ports["image_attention_gate"] * image)
        record("text_attention_write", ports["text_attention_gate"] * used)
        return image, used

    def image_mlp(_module, _args, result):
        require(calls["image_mlp"] == 0, "native image MLP occurrence differs")
        calls["image_mlp"] += 1
        record("image_mlp_return", result)
        record("image_mlp_write", ports["image_mlp_gate"] * result)

    def text_mlp(_module, _args, result):
        require(calls["attention"] == 1 and calls["mlp"] == 0, "native text MLP occurrence differs")
        calls["mlp"] += 1
        used = _replace(result, payloads, "mlp_return", text_rows)
        record("text_mlp_return_computed", result)
        record("text_mlp_return_used", used)
        record("text_mlp_write", ports["text_mlp_gate"] * used)
        return used

    def finish(_module, _args, result):
        require(isinstance(result, tuple) and len(result) == 2, "native block return ABI differs")
        text, image = result
        record("text_out", text)
        record("image_out", image)
        _landings(ports, record)
        after = dict(state, image=_snapshot(image), text=_snapshot(text), cursor=1)
        if len(adapter.model.transformer_blocks) == 1:
            after.update(phase="single", cursor=0)
        capture["state"] = after

    def forbidden(_module, _args):
        raise ValueError("native writer scope cannot execute another transformer block")

    handles = []
    try:
        handles.extend((block.register_forward_pre_hook(entry, with_kwargs=True),
            block.attn.register_forward_pre_hook(attention_inputs, with_kwargs=True),
            block.attn.register_forward_hook(attention),
            block.norm2.register_forward_pre_hook(lambda _m, args: record("image_after_attention", args[0])),
            block.norm2_context.register_forward_pre_hook(lambda _m, args: record("text_after_attention", args[0])),
            block.ff.register_forward_pre_hook(lambda _m, args: record("image_mlp_input", args[0])),
            block.ff_context.register_forward_pre_hook(lambda _m, args: record("text_mlp_input", args[0])),
            block.ff.register_forward_hook(image_mlp), block.ff_context.register_forward_hook(text_mlp),
            block.register_forward_hook(finish)))
        handles.extend(other.register_forward_pre_hook(forbidden) for other in (
            *adapter.model.transformer_blocks[1:], *adapter.model.single_transformer_blocks))
        yield capture
        require(calls == dict(block=1, attention=1, mlp=1, image_mlp=1) and capture["state"] is not None,
                "native joint0 capture incomplete")
        adapter.validate_execution()
    finally:
        for handle in reversed(handles):
            handle.remove()
