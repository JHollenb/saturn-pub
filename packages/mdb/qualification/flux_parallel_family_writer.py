"""Full-concatenation single-stream writer ports from original parameters.

The equation lane shares the pinned Torch primitive graph. The native lane
observes one caller-executed public adapter transition, including the actual
SDPA operands and module returns. Neither lane predicts a raw prompt. Retained
values and live layout descriptors are separate: cloning a padded QKV view can
change its stride, and that change is reported rather than hidden.
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

SCHEMA = "mdb-flux-parallel-family-writer-v1"
GRAPH = "flux_functional_denoiser.py"
HELPERS = {GRAPH: "cb794daa5818ec3f9107f23839a41010f9569148da3708e39cedd364dff21923"}
PORT_NAMES = (
    "image_in", "text_in", "R", "shift", "scale", "gate", "norm", "x",
    "fused_input", "fused_return", "q_raw", "k_raw", "v_raw", "q_norm", "k_norm",
    "q_consumed", "k_consumed", "v_consumed", "sdpa_return", "A", "mlp_input", "M",
    "merge_input", "projection_return", "write", "out", "image_out", "text_out",
)
_CAPTURE_LOCK = threading.Lock()


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
    """Hash logical element-order bytes; retain the actual live tensor layout."""
    import torch

    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                device=str(value.device), content_sha256=hashlib.sha256(raw).hexdigest(),
                hash_order="contiguous logical element order, dtype bytes")


def _snapshot(value):
    import torch

    return value.detach().clone(memory_format=torch.preserve_format)


def _copy(value):
    import torch

    if isinstance(value, torch.Tensor):
        return _snapshot(value)
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy(item) for item in value)
    if isinstance(value, list):
        return [_copy(item) for item in value]
    return value


def _same(left, right):
    import torch

    return (left.shape == right.shape and left.dtype == right.dtype and left.device == right.device
            and torch.equal(left.detach().contiguous().view(torch.uint8), right.detach().contiguous().view(torch.uint8)))


class Ports(dict):
    """Tensor-only mapping with a transient original-projection binding.

    Attributes are not part of the serializable tensor table. Save descriptors
    alongside tensor records; constructing a hybrid requires this live object.
    No parameters are copied or returned as ports.
    """
    def __init__(self, adapter, profile, retain_ports, observe):
        super().__init__()
        requested = PORT_NAMES if retain_ports is None else retain_ports
        require(isinstance(requested, (tuple, list, set, frozenset))
                and set(requested) <= set(PORT_NAMES), "unknown retained writer port")
        self.requested = frozenset(requested)
        self.descriptors, self.retained_descriptors = {}, {}
        self.adapter, self.profile, self.observe = adapter, profile, observe
        self.state = None

    def record(self, name, value):
        require(name in PORT_NAMES and name not in self.descriptors, "duplicate or unknown writer port: " + name)
        self.descriptors[name] = descriptor(value)
        if name in self.requested:
            self[name] = _snapshot(value)
            self.retained_descriptors[name] = descriptor(self[name])
        if self.observe is not None:
            self.observe(name, _snapshot(value))

    def result(self, **extra):
        require(set(self.descriptors) == set(PORT_NAMES), "writer port capture incomplete")
        return dict(state=self.state, ports=self, profile=self.profile,
            descriptors=self.descriptors, retained_descriptors=self.retained_descriptors,
            retention=dict(full_arrays=sorted(self), hash_and_layout_only=sorted(set(self.descriptors) - set(self)),
                           snapshot_layout="retained_descriptors; live layout is in descriptors"), **extra)


def qualify(adapter, state, index=4, *, expected_model_identity, allow_cpu_fixture=False):
    """Qualify a fixed original model, clock, block and numerical profile."""
    import torch

    require(type(allow_cpu_fixture) is bool, "explicit fixture flag required")
    require(isinstance(expected_model_identity, str) and expected_model_identity
            and adapter.model_identity == expected_model_identity, "writer model identity differs")
    adapter.validate_execution()
    adapter.validate(state)
    point = adapter.point(state).to_dict()
    require(type(index) is int and 0 <= index < len(adapter.model.single_transformer_blocks)
            and point["family"] == "flux2-klein" and point["phase"] == "single"
            and point["index"] == index and point["next_operation"] == f"single_block.{index}"
            and state["phase"] == "single" and state["cursor"] == index
            and state["step"] == point["logical_step"] == 0, "writer requires matching step0 single-block entry")
    image, text, context = (state[name] for name in ("image", "text", "context"))
    require(isinstance(context, Mapping) and {"single_mod", "rope"} <= set(context), "incomplete single-block context")
    require(image.ndim == text.ndim == 3 and image.shape[0] == text.shape[0] == 1
            and image.shape[-1] == text.shape[-1] and image.dtype == text.dtype == torch.bfloat16
            and image.device == text.device, "writer requires paired batch-one BF16 streams")
    require(image.device.type == ("cpu" if allow_cpu_fixture else "cuda"), "writer device/profile differs")
    require(not adapter.model.training, "writer requires frozen eval model")
    if not allow_cpu_fixture:
        import diffusers
        from saturn_pub.adapters.flux2 import Flux2KleinAdapter

        require(isinstance(adapter, Flux2KleinAdapter), "original public Klein adapter required")
        require(tuple(image.shape) == (1, 1024, 3072) and tuple(text.shape) == (1, 512, 3072)
                and len(adapter.model.transformer_blocks) == 5 and len(adapter.model.single_transformer_blocks) == 20,
                "unqualified original writer geometry")
        require(str(torch.__version__).split("+")[0] == "2.13.0", "unqualified Torch writer profile")
        require(str(diffusers.__version__) == "0.39.0", "unqualified Diffusers writer profile")
    equations()["validate_model"](adapter.model)
    attn = adapter.model.single_transformer_blocks[index].attn
    require(attn.heads * attn.head_dim == attn.inner_dim and attn.mlp_mult_factor == 2,
            "unsupported fused attention/MLP geometry")
    width = image.shape[-1]
    mod = context["single_mod"]
    require(mod.dtype == image.dtype and mod.device == image.device
            and tuple(mod.shape) in ((1, 3 * width), (1, 1, 3 * width)), "single modulation profile differs")
    values = dict(image=image, text=text, single_mod=mod)
    rope = context["rope"]
    require(isinstance(rope, tuple) and len(rope) == 2, "writer requires explicit RoPE pair")
    for name, value in zip(("rope_cos", "rope_sin"), rope, strict=True):
        require(tuple(value.shape) == (image.shape[1] + text.shape[1], attn.head_dim)
                and value.dtype == torch.float32 and value.device == image.device, "writer RoPE profile differs")
        values[name] = value
    require(all(bool(torch.isfinite(value).all()) for value in values.values()), "nonfinite writer state")
    return dict(schema=SCHEMA, model_identity=str(adapter.model_identity), index=index, point=point,
        execution=json.loads(json.dumps(dict(adapter.execution), sort_keys=True, allow_nan=False)),
        interfaces={name: descriptor(value) for name, value in values.items()},
        clock={name: descriptor(state[name]) for name in ("timesteps", "sigmas") if name in state},
        domain="CPU arithmetic fixture only" if allow_cpu_fixture else "original Klein4B step0 B1 CUDA BF16 512-text/1024-image",
        raw_prompt_prediction=False, independent_kernel_prediction=False,
        projection_geometry="full concatenated text+image sequence and full attention+MLP feature width",
        attention_layout="q_raw/k_raw/v_raw BTHD; q_consumed/k_consumed/v_consumed and sdpa_return BHTD",
        port_origins=dict(A="functional: flattened SDPA return; native: attention feature slice of actual to_out operand",
            M="functional: original SwiGLU equation; native: actual mlp_act_fn return",
            write="same-profile product of consumed gate and projection return; native landing is actual block return"),
        sdpa_profile=dict(attn_mask=None, dropout_p=0., is_causal=False, scale=None, enable_gqa=False),
        port_names=list(PORT_NAMES))


def _after(adapter, state, out):
    ntext = state["text"].shape[1]
    after = _copy(state)
    after.update(image=_snapshot(out[:, ntext:]), text=_snapshot(out[:, :ntext]), cursor=state["cursor"] + 1)
    if after["cursor"] == len(adapter.model.single_transformer_blocks):
        after["phase"] = "readout"
    return after


def _streams(ports, state):
    ports.record("image_in", state["image"])
    ports.record("text_in", state["text"])
    shift, scale, gate = equations()["_mods"](state["context"]["single_mod"], 1)[0]
    for name, value in dict(shift=shift, scale=scale, gate=gate).items():
        ports.record(name, value)
    return shift, scale, gate


def functional_single(adapter, state, index=4, *, expected_model_identity,
                      retain_ports=None, allow_cpu_fixture=False, observe=None):
    """One independently orchestrated original-weight block; no Module call."""
    import torch
    import torch.nn.functional as F

    profile = qualify(adapter, state, index, expected_model_identity=expected_model_identity, allow_cpu_fixture=allow_cpu_fixture)
    graph = equations()
    ports = Ports(adapter, profile, retain_ports, observe)
    with torch.inference_mode():
        block = adapter.model.single_transformer_blocks[index]
        attn = block.attn
        shift, scale, gate = _streams(ports, state)
        r = torch.cat([state["text"], state["image"]], 1)
        ports.record("R", r)
        norm = graph["_norm"](block.norm, r)
        ports.record("norm", norm)
        x = (1 + scale) * norm + shift
        ports.record("x", x)
        ports.record("fused_input", x)
        fused = graph["_linear"](attn.to_qkv_mlp_proj, x)
        ports.record("fused_return", fused)
        qkv, mlp = fused.split([3 * attn.inner_dim, 2 * attn.mlp_hidden_dim], -1)
        q, k, v = (value.unflatten(-1, (attn.heads, attn.head_dim)) for value in qkv.chunk(3, -1))
        for name, value in dict(q_raw=q, k_raw=k, v_raw=v).items():
            ports.record(name, value)
        q, k = graph["_rms"](attn.norm_q, q), graph["_rms"](attn.norm_k, k)
        ports.record("q_norm", q)
        ports.record("k_norm", k)
        q, k = (graph["_rotate"](value, state["context"]["rope"]) for value in (q, k))
        q, k, v = (value.permute(0, 2, 1, 3) for value in (q, k, v))
        for name, value in dict(q_consumed=q, k_consumed=k, v_consumed=v).items():
            ports.record(name, value)
        sdpa = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
        ports.record("sdpa_return", sdpa)
        a = sdpa.permute(0, 2, 1, 3).flatten(2, 3).to(q.dtype)
        ports.record("A", a)
        ports.record("mlp_input", mlp)
        m = graph["_swiglu"](mlp)
        ports.record("M", m)
        merge = torch.cat([a, m], -1)
        ports.record("merge_input", merge)
        projection = graph["_linear"](attn.to_out, merge)
        ports.record("projection_return", projection)
        write = gate * projection
        ports.record("write", write)
        out = r + write
        ports.record("out", out)
        ports.record("text_out", out[:, :state["text"].shape[1]])
        ports.record("image_out", out[:, state["text"].shape[1]:])
        ports.state = _after(adapter, state, out)
    adapter.validate_execution()
    return ports.result(operation_counts=dict(linear=2, layer_norm=1, rms_norm=2, sdpa=1,
        silu=1, rope=2, gate_multiply=1, residual_add=1), lane="original-weight-equations")


def _argument(args, kwargs, name, index, default=...):
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    require(default is not ..., "missing native single-block operand: " + name)
    return default


@contextmanager
def native_single(adapter, state, index=4, *, expected_model_identity,
                  retain_ports=None, allow_cpu_fixture=False, observe=None):
    """Observe exactly one caller Session.continue_(), restoring every hook.

    The temporary SDPA wrapper delegates unchanged to Torch and records actual
    BHTD operands/return. This scope is serialized and rejects nested captures.
    Existing read-only module hooks are permitted.
    """
    import torch
    import torch.nn.functional as F

    profile = qualify(adapter, state, index, expected_model_identity=expected_model_identity, allow_cpu_fixture=allow_cpu_fixture)
    original_sdpa = F.scaled_dot_product_attention
    handles = []
    ports = Ports(adapter, profile, retain_ports, observe)
    expected_combined = descriptor(torch.cat([state["text"], state["image"]], 1))
    require(_CAPTURE_LOCK.acquire(blocking=False), "native single-block capture already active")
    calls = dict(block=0, norm=0, attention=0, fused=0, norm_q=0, norm_k=0, mlp=0, projection=0, sdpa=0)
    capture = dict(state=None, ports=ports, profile=profile, descriptors=ports.descriptors,
                   retained_descriptors=ports.retained_descriptors, native_calls=calls)
    block = adapter.model.single_transformer_blocks[index]
    attn = block.attn
    owner = threading.get_ident()
    live = dict(active=False)

    def check_thread():
        require(threading.get_ident() == owner, "native writer crossed threads")

    def count(name):
        check_thread()
        require(live["active"] and calls[name] == 0, "native writer occurrence differs: " + name)
        calls[name] += 1

    def entry(_module, args, kwargs):
        check_thread()
        require(calls["block"] == 0, "native writer allows one selected block")
        calls["block"] += 1
        value = _argument(args, kwargs, "hidden_states", 0)
        require(descriptor(value) == expected_combined, "actual native combined input differs")
        for name in ("image", "text"):
            require(descriptor(state[name]) == profile["interfaces"][name], "supplied source state mutated: " + name)
        require(descriptor(state["context"]["single_mod"]) == profile["interfaces"]["single_mod"], "supplied source modulation mutated")
        require(_argument(args, kwargs, "encoder_hidden_states", 1) is None
                and _argument(args, kwargs, "split_hidden_states", 5, False) is False,
                "native public adapter concatenation ABI differs")
        mod = _argument(args, kwargs, "temb_mod", 2)
        rope = _argument(args, kwargs, "image_rotary_emb", 3)
        require(descriptor(mod) == profile["interfaces"]["single_mod"], "actual native modulation differs")
        require(isinstance(rope, tuple) and len(rope) == 2 and all(
            descriptor(value) == profile["interfaces"][name] for value, name in zip(rope, ("rope_cos", "rope_sin"), strict=True)),
            "actual native RoPE differs")
        require(_argument(args, kwargs, "joint_attention_kwargs", 4, None) in (None, {}), "custom native attention kwargs unsupported")
        live.update(active=True, R=value, gate=_streams(ports, state)[2])
        ports.record("R", value)

    def norm(_module, _args, result):
        count("norm")
        ports.record("norm", result)

    def norm_input(_module, args):
        require(descriptor(args[0]) == ports.descriptors["R"], "actual native norm residual differs")

    def attention_input(_module, args, kwargs):
        value = _argument(args, kwargs, "hidden_states", 0)
        ports.record("x", value)

    def attention(_module, _args, result):
        count("attention")
        require(_same(result, live["projection"]), "native attention return differs from actual projection")

    def fused_input(_module, args):
        require(descriptor(args[0]) == ports.descriptors["x"], "actual fused projection operand differs")
        ports.record("fused_input", args[0])

    def fused(_module, _args, result):
        count("fused")
        ports.record("fused_return", result)
        qkv, mlp = result.split([3 * attn.inner_dim, 2 * attn.mlp_hidden_dim], -1)
        for name, value in zip(("q_raw", "k_raw", "v_raw"), qkv.chunk(3, -1), strict=True):
            ports.record(name, value.unflatten(-1, (attn.heads, attn.head_dim)))
        live["mlp_input_descriptor"] = descriptor(mlp)

    def rms_input(name):
        def hook(_module, args):
            require(descriptor(args[0]) == ports.descriptors[name + "_raw"], "actual native raw Q/K operand differs")
        return hook

    def rms(name):
        def hook(_module, _args, result):
            count("norm_" + name)
            ports.record(name + "_norm", result)
        return hook

    def mlp(_module, _args, result):
        count("mlp")
        ports.record("M", result)

    def mlp_input(_module, args):
        require(descriptor(args[0]) == live["mlp_input_descriptor"], "actual native MLP operand differs")
        ports.record("mlp_input", args[0])

    def projection_input(_module, args):
        value = args[0]
        ports.record("merge_input", value)
        ports.record("A", value[..., :attn.inner_dim])
        require(descriptor(value[..., attn.inner_dim:])["content_sha256"] == ports.descriptors["M"]["content_sha256"],
                "native merge MLP operand differs from actual MLP return")

    def projection(_module, _args, result):
        count("projection")
        ports.record("projection_return", result)
        live["projection"] = result

    def sdpa(*args, **kwargs):
        count("sdpa")
        for name, field, offset in (("q_consumed", "query", 0), ("k_consumed", "key", 1), ("v_consumed", "value", 2)):
            ports.record(name, _argument(args, kwargs, field, offset))
        require(_argument(args, kwargs, "attn_mask", 3, None) is None
                and _argument(args, kwargs, "dropout_p", 4, 0.) == 0.
                and _argument(args, kwargs, "is_causal", 5, False) is False
                and kwargs.get("scale") is None and kwargs.get("enable_gqa", False) is False,
                "native SDPA profile differs")
        result = original_sdpa(*args, **kwargs)
        ports.record("sdpa_return", result)
        return result

    def finish(_module, _args, result):
        require(isinstance(result, torch.Tensor), "native combined block return ABI differs")
        write = live["gate"] * live["projection"]
        ports.record("write", write)
        ports.record("out", result)
        ntext = state["text"].shape[1]
        ports.record("text_out", result[:, :ntext])
        ports.record("image_out", result[:, ntext:])
        ports.state = _after(adapter, state, result)
        capture["state"] = ports.state
        live.clear()
        live["active"] = False

    def forbidden(_module, _args):
        raise ValueError("native writer scope cannot execute another transformer block")

    try:
        require(not getattr(original_sdpa, "_parallel_writer_capture", False), "nested SDPA capture unsupported")
        sdpa._parallel_writer_capture = True
        handles.extend((block.register_forward_pre_hook(entry, with_kwargs=True),
            block.norm.register_forward_pre_hook(norm_input), block.norm.register_forward_hook(norm),
            block.attn.register_forward_pre_hook(attention_input, with_kwargs=True),
            block.attn.register_forward_hook(attention), attn.to_qkv_mlp_proj.register_forward_pre_hook(fused_input),
            attn.to_qkv_mlp_proj.register_forward_hook(fused), attn.norm_q.register_forward_pre_hook(rms_input("q")),
            attn.norm_k.register_forward_pre_hook(rms_input("k")), attn.norm_q.register_forward_hook(rms("q")),
            attn.norm_k.register_forward_hook(rms("k")), attn.mlp_act_fn.register_forward_pre_hook(mlp_input),
            attn.mlp_act_fn.register_forward_hook(mlp),
            attn.to_out.register_forward_pre_hook(projection_input), attn.to_out.register_forward_hook(projection),
            block.register_forward_hook(finish)))
        handles.extend(other.register_forward_pre_hook(forbidden) for other in (
            *adapter.model.transformer_blocks, *(value for position, value in enumerate(adapter.model.single_transformer_blocks) if position != index)))
        F.scaled_dot_product_attention = sdpa
        yield capture
        require(all(value == 1 for value in calls.values()) and capture["state"] is not None, "native single-block capture incomplete")
        adapter.validate_execution()
        capture.update(ports.result(native_calls=calls, lane="native-reference"))
    finally:
        F.scaled_dot_product_attention = original_sdpa
        for handle in reversed(handles):
            handle.remove()
        live.clear()
        _CAPTURE_LOCK.release()


def construct_endpoint(q_ports, p_ports, arm, *, native=False):
    """Image R/A hybrid, full merged projection, Q MLP/gate/postblock text.

    P is an explicitly separate natural full-state reference. Other arms change
    image rows only. ``native=True`` executes the original output-projection
    module once; the equation lane uses its original parameters with F.linear.
    """
    import torch

    require(type(native) is bool and arm in ("P", "Q", "R", "A", "RA"), "unknown parallel endpoint arm")
    require(isinstance(q_ports, Ports) and isinstance(p_ports, Ports)
            and q_ports.state is not None and p_ports.state is not None, "live qualified source ports required")
    q, p = q_ports.profile, p_ports.profile
    for key in ("model_identity", "point", "execution", "clock", "domain"):
        require(q[key] == p[key], "P/Q writer qualification differs: " + key)
    for name in ("single_mod", "rope_cos", "rope_sin"):
        require(q["interfaces"][name] == p["interfaces"][name], "P/Q clock/modulation/RoPE differs: " + name)
    required = {"R", "A", "M", "gate", "out", "image_in", "text_in", "image_out", "text_out", "projection_return"}
    for source in (q_ports, p_ports):
        source.adapter.validate_execution()
        require(required <= set(source), "endpoint requires retained core source ports")
        for name in required:
            require(descriptor(source[name])["content_sha256"] == source.descriptors[name]["content_sha256"], "retained source port mutated: " + name)
        require(_same(source["R"] + source["gate"] * source["projection_return"], source["out"]), "source finite gate/add closure differs")
    require(_same(q_ports["gate"], p_ports["gate"]), "P/Q recipient gates differ")
    if arm == "P":
        return dict(image=_snapshot(p_ports.state["image"]), text=_snapshot(p_ports.state["text"]), state=_copy(p_ports.state),
            ports={}, descriptors={}, construction=dict(arm=arm, origin="natural full P state including P conditioning and text", native_projection_calls=0))
    graph = equations()
    ntext = q_ports.state["text"].shape[1]
    with torch.inference_mode():
        r, a = _snapshot(q_ports["R"]), _snapshot(q_ports["A"])
        if arm in ("R", "RA"):
            r[:, ntext:] = p_ports["R"][:, ntext:]
        if arm in ("A", "RA"):
            a[:, ntext:] = p_ports["A"][:, ntext:]
        merge = torch.cat([a, q_ports["M"]], -1)
        projection_module = q_ports.adapter.model.single_transformer_blocks[q["index"]].attn.to_out
        projection = projection_module(merge) if native else graph["_linear"](projection_module, merge)
        write = q_ports["gate"] * projection
        out = r + write
        if arm == "Q":
            require(_same(out, q_ports["out"]), "Q full-concatenation selfcopy closure differs")
        after = _copy(q_ports.state)
        after["image"] = _snapshot(out[:, ntext:])
        values = dict(merge_input=merge, projection_return=projection, write=write, out=out)
        result = dict(image=_snapshot(after["image"]), text=_snapshot(after["text"]), state=after,
            ports=values, descriptors={name: descriptor(value) for name, value in values.items()},
            construction=dict(arm=arm, image_residual_source="P" if arm in ("R", "RA") else "Q",
                image_attention_source="P" if arm in ("A", "RA") else "Q", mlp_source="Q", gate_source="Q",
                text_and_conditioning_source="Q postblock text and complete Q state",
                full_projection_shape=list(merge.shape), native_projection_calls=int(native),
                functional_projection_calls=int(not native), gate_multiply=1, residual_add=1))
    q_ports.adapter.validate_execution()
    return result
