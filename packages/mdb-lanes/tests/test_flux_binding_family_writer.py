"""Controlled arithmetic toy qualification; no pretrained model or weights."""
from __future__ import annotations

import copy
import itertools
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from saturn_pub.contracts import ExecutionPoint
from saturn_pub.core import Adapter, Session

FILE = Path(__file__).parents[1] / "qualification/flux_binding_family_writer.py"
W = runpy.run_path(str(FILE))


def bits(left, right):
    return left.shape == right.shape and left.dtype == right.dtype and torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def linear(rows=4, cols=4, offset=0):
    values = (((torch.arange(rows * cols).reshape(rows, cols) + offset) % 7) - 3) / 8
    return SimpleNamespace(weight=values.to(torch.bfloat16), bias=None)


class Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.normalized_shape, self.weight, self.bias, self.eps = (4,), None, None, 1e-6

    def forward(self, value):
        return F.layer_norm(value, self.normalized_shape, self.weight, self.bias, self.eps)


class FF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_in, self.linear_out = linear(8, 4, 2), linear(4, 4, 5)

    def forward(self, value):
        a, b = F.linear(value, self.linear_in.weight).chunk(2, -1)
        return F.linear(F.silu(a) * b, self.linear_out.weight)


class Attention(torch.nn.Module):
    """Fixed synthetic matrices and native primitive attention, no learned model."""
    def __init__(self):
        super().__init__()
        self.head_dim = 4
        for offset, name in enumerate(("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj")):
            setattr(self, name, linear(offset=offset))
        self.to_out, self.to_add_out = [linear(offset=4)], linear(offset=6)
        for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            setattr(self, name, SimpleNamespace(normalized_shape=(4,), weight=torch.ones(4, dtype=torch.bfloat16), eps=1e-6))

    def forward(self, hidden_states, encoder_hidden_states, image_rotary_emb):
        values = {}
        for prefix, value, names in (("", hidden_states, ("to_q", "to_k", "to_v")),
                                      ("c", encoder_hidden_states, ("add_q_proj", "add_k_proj", "add_v_proj"))):
            for field, name in zip(("q", "k", "v"), names, strict=True):
                values[prefix + field] = F.linear(value, getattr(self, name).weight).unflatten(-1, (1, 4))
        q = torch.cat((F.rms_norm(values["cq"], (4,), self.norm_added_q.weight, 1e-6),
                       F.rms_norm(values["q"], (4,), self.norm_q.weight, 1e-6)), 1)
        k = torch.cat((F.rms_norm(values["ck"], (4,), self.norm_added_k.weight, 1e-6),
                       F.rms_norm(values["k"], (4,), self.norm_k.weight, 1e-6)), 1)
        v = torch.cat((values["cv"], values["v"]), 1)
        cos, sin = (value[None, :, None, :] for value in image_rotary_emb)

        def rotate(value):
            even, odd = value.reshape(*value.shape[:-1], 2, 2).unbind(-1)
            perpendicular = torch.stack((-odd, even), -1).flatten(3)
            return (value.float() * cos + perpendicular.float() * sin).to(value.dtype)

        result = F.scaled_dot_product_attention(rotate(q).transpose(1, 2), rotate(k).transpose(1, 2),
                                                v.transpose(1, 2), dropout_p=0., is_causal=False)
        result = result.transpose(1, 2).flatten(2, 3).to(q.dtype)
        text, image = result[:, :encoder_hidden_states.shape[1]], result[:, encoder_hidden_states.shape[1]:]
        return F.linear(image, self.to_out[0].weight), F.linear(text, self.to_add_out.weight)


def mods(value):
    parts = value[:, None].chunk(6, -1)
    return parts[:3], parts[3:]


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm1_context, self.norm2, self.norm2_context = (Norm() for _ in range(4))
        self.ff, self.ff_context, self.attn = FF(), FF(), Attention()

    def forward(self, hidden_states, encoder_hidden_states, temb_mod_img, temb_mod_txt,
                image_rotary_emb, joint_attention_kwargs=None):
        assert joint_attention_kwargs in (None, {})
        (shift, scale, gate), (shift_ff, scale_ff, gate_ff) = mods(temb_mod_img)
        (cs, cc, cg), (cs_ff, cc_ff, cg_ff) = mods(temb_mod_txt)
        x = (1 + scale) * self.norm1(hidden_states) + shift
        c = (1 + cc) * self.norm1_context(encoder_hidden_states) + cs
        a, ca = self.attn(hidden_states=x, encoder_hidden_states=c, image_rotary_emb=image_rotary_emb)
        image = hidden_states + gate * a
        image = image + gate_ff * self.ff(self.norm2(image) * (1 + scale_ff) + shift_ff)
        text = encoder_hidden_states + cg * ca
        text = text + cg_ff * self.ff_context(self.norm2_context(text) * (1 + cc_ff) + cs_ff)
        return text, image


class Toy(Adapter):
    model_identity = "controlled-nonlearned-writer-matrices"
    execution = dict(family="flux2-klein", device="cpu", dtype="torch.bfloat16", scope="arithmetic-fixture")

    def __init__(self):
        self.model = torch.nn.Module()
        self.model.transformer_blocks = torch.nn.ModuleList([Block()])
        self.model.single_transformer_blocks = torch.nn.ModuleList([])
        self.model.eval()
        self.validations = 0

    def validate_execution(self):
        self.validations += 1

    def validate(self, state):
        assert state["image"].shape == (1, 2, 4) and state["text"].shape == (1, 3, 4)

    def addresses(self, state):
        return ("image", "text")

    def boundary(self, state):
        return f"step:{state['step']}/{state['phase']}:{state['cursor']}"

    def point(self, state):
        return ExecutionPoint(family="flux2-klein", logical_step=state["step"], phase=state["phase"],
            operator=state["phase"], index=state["cursor"], edge="before", local_clock=state["cursor"],
            next_operation=f"joint_block.{state['cursor']}")

    def advance(self, state):
        result = self.model.transformer_blocks[0](hidden_states=state["image"], encoder_hidden_states=state["text"],
            temb_mod_img=state["context"]["img_mod"], temb_mod_txt=state["context"]["txt_mod"],
            image_rotary_emb=state["context"]["rope"], joint_attention_kwargs=None)
        return dict(state, text=result[0], image=result[1], phase="single", cursor=0)


def fixture():
    adapter = Toy()
    image = torch.tensor([[[1., -.5, .25, 2.], [.75, -1., .5, .125]]], dtype=torch.bfloat16)
    text = torch.tensor([[[.5, -.25, 1., 2.], [1., 2., -2., -.5], [-1., .25, 2., .5]]], dtype=torch.bfloat16)
    im = torch.tensor([.125] * 4 + [.25] * 4 + [.75] * 4 + [.25] * 4 + [-.125] * 4 + [.5] * 4, dtype=torch.bfloat16)[None]
    tm = torch.tensor([-.125] * 4 + [.125] * 4 + [1.25] * 4 + [.125] * 4 + [.25] * 4 + [-.5] * 4, dtype=torch.bfloat16)[None]
    angles = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 20
    state = dict(image=image, text=text, context=dict(img_mod=im, txt_mod=tm, rope=(angles.cos(), angles.sin())),
                 step=2, phase="joint", cursor=0, immutable="same nontext state")
    return adapter, state


def predict(adapter, state, **kwargs):
    return W["functional_joint0"](adapter, state, expected_model_identity=adapter.model_identity, allow_cpu_fixture=True, **kwargs)


def native(adapter, state, **kwargs):
    with W["native_joint0"](adapter, state, expected_model_identity=adapter.model_identity, allow_cpu_fixture=True, **kwargs) as capture:
        session = Session(adapter, state)
        session.continue_()
        actual = session.capture(retain=False).payload
    assert bits(actual["text"], capture["state"]["text"]) and bits(actual["image"], capture["state"]["image"])
    return capture


def test_standalone_definitions_are_framework_free():
    code = "import runpy,sys; runpy.run_path(sys.argv[1]); assert not {'torch','saturn_pub','numpy'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(FILE)], check=True)


def test_functional_baseline_matches_frozen_graph_without_any_module_call(monkeypatch):
    adapter, state = fixture()
    before = copy.deepcopy(state)
    monkeypatch.setattr(torch.nn.Module, "__call__", lambda *_args, **_kwargs: pytest.fail("functional path entered a module"))
    result = predict(adapter, state)
    expected_image, expected_text = W["equations"]()["functional_joint"](adapter.model, 0, state["image"], state["text"], state["context"])
    assert bits(result["state"]["image"], expected_image) and bits(result["state"]["text"], expected_text)
    assert bits(state["text"], before["text"]) and bits(state["image"], before["image"])
    p = result["ports"]
    assert bits(p["text_after_attention"], p["text_in"] + p["text_attention_gate"] * p["text_attention_return_used"])
    assert bits(p["text_out"], p["text_after_attention"] + p["text_mlp_gate"] * p["text_mlp_return_used"])
    assert result["profile"]["domain"] == "CPU arithmetic fixture only"


def test_actual_native_hooks_match_functional_ports_and_remove_every_hook():
    adapter, state = fixture()
    baseline = predict(adapter, state)
    modules = list(adapter.model.modules())
    initial = [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in modules]
    result = native(adapter, state)
    assert result["native_calls"] == dict(block=1, attention=1, mlp=1, image_mlp=1)
    assert baseline["ports"].keys() == result["ports"].keys()
    for name, value in baseline["ports"].items():
        assert bits(value, result["ports"][name]), name
    assert initial == [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in modules]


def test_attention_return_set_recomputes_own_mlp_and_preserves_unselected_rows():
    adapter, state = fixture()
    baseline = predict(adapter, state)
    donor = baseline["ports"]["text_attention_return_used"].clone()
    donor[:, 1] = torch.tensor([1., -2., .5, 3.], dtype=torch.bfloat16)
    changed = predict(adapter, state, text_rows=[1], attention_return=donor)
    target = native(adapter, state, text_rows=[1], attention_return=donor)
    for name, value in changed["ports"].items():
        assert bits(value, target["ports"][name]), name
    assert bits(target["ports"]["text_attention_return_computed"], baseline["ports"]["text_attention_return_computed"])
    assert bits(target["ports"]["text_attention_return_used"][:, 1], donor[:, 1])
    assert bits(target["ports"]["text_attention_return_used"][:, [0, 2]], baseline["ports"]["text_attention_return_used"][:, [0, 2]])
    assert not bits(changed["ports"]["text_mlp_input"][:, 1], baseline["ports"]["text_mlp_input"][:, 1])
    assert not bits(changed["ports"]["text_mlp_return_computed"][:, 1], baseline["ports"]["text_mlp_return_computed"][:, 1])
    assert bits(target["ports"]["image_out"], baseline["ports"]["image_out"])


def test_mlp_return_replacement_retains_actual_computed_port_and_signed_zero():
    adapter, state = fixture()
    baseline = predict(adapter, state)
    donor = baseline["ports"]["text_mlp_return_used"].clone()
    donor[:, 1] = -0.
    changed = predict(adapter, state, text_rows=[1], mlp_return=donor)
    target = native(adapter, state, text_rows=[1], mlp_return=donor)
    assert bits(target["ports"]["text_mlp_return_computed"], baseline["ports"]["text_mlp_return_computed"])
    assert torch.signbit(target["ports"]["text_mlp_return_used"][:, 1]).all()
    assert bits(changed["ports"]["text_out"], target["ports"]["text_out"])
    no_op = native(adapter, state, text_rows=[], mlp_return=donor)
    assert bits(no_op["ports"]["text_out"], baseline["ports"]["text_out"])


def test_frozen_eight_corners_preserve_native_add_order_and_q_complement():
    adapter, state = fixture()
    other = copy.deepcopy(state)
    other["text"][:, 1] += torch.tensor([.5, -.5, 1., -1.], dtype=torch.bfloat16)
    q, p = predict(adapter, state)["ports"], predict(adapter, other)["ports"]
    for r, a, m in itertools.product(("Q", "P"), repeat=3):
        result = W["frozen_endpoint"](q, p, [1], r, a, m)
        sources = dict(Q=q, P=p)
        expected = sources[r]["text_in"] + q["text_attention_gate"] * sources[a]["text_attention_return_used"]
        expected = expected + q["text_mlp_gate"] * sources[m]["text_mlp_return_used"]
        assert bits(result[:, 1], expected[:, 1])
        assert bits(result[:, [0, 2]], q["text_out"][:, [0, 2]])
    assert bits(W["frozen_endpoint"](q, p, [1], "Q", "Q", "Q"), q["text_out"])
    all_p = q["text_out"].clone()
    all_p[:, 1] = p["text_out"][:, 1]
    assert bits(W["frozen_endpoint"](q, p, [1], "P", "P", "P"), all_p)
    # A controlled BF16 midpoint distinguishes native sequential adds from fusion.
    r = torch.ones(1, 3, 4, dtype=torch.bfloat16)
    a = torch.full_like(r, 2**-8)
    gate = torch.ones(1, 1, 4, dtype=torch.bfloat16)
    u = r + gate * a
    t = u + gate * a
    ports = dict(text_in=r, text_attention_gate=gate, text_attention_return_used=a,
                 text_after_attention=u, text_mlp_gate=gate, text_mlp_return_used=a, text_out=t)
    result = W["frozen_endpoint"](ports, ports, [1], "Q", "Q", "Q")
    assert bits(result, t) and not bits(result[:, 1], (r.double() + 2*a.double()).to(torch.bfloat16)[:, 1])


def test_recomputed_crossings_use_full_shape_original_mlp_without_attention(monkeypatch):
    adapter, state = fixture()
    other = copy.deepcopy(state)
    other["text"][:, 1] += torch.tensor([.5, -.5, 1., -1.], dtype=torch.bfloat16)
    q, p = predict(adapter, state)["ports"], predict(adapter, other)["ports"]
    calls = []
    block = adapter.model.transformer_blocks[0]
    handles = [block.ff_context.register_forward_pre_hook(lambda _module, args: calls.append(tuple(args[0].shape)))]
    monkeypatch.setattr(block.attn, "forward", lambda *_a, **_kw: pytest.fail("recomputed MLP replayed attention"))
    for r, a in (("Q", "P"), ("P", "Q")):
        functional = W["functional_recomputed_endpoint"](adapter, state, q, p, [1], r, a,
            expected_model_identity=adapter.model_identity, allow_cpu_fixture=True)
        target = W["native_recomputed_endpoint"](adapter, state, q, p, [1], r, a,
            expected_model_identity=adapter.model_identity, allow_cpu_fixture=True)
        assert bits(functional["endpoint"], target["endpoint"])
        assert bits(target["endpoint"][:, [0, 2]], q["text_out"][:, [0, 2]])
        for name, value in functional["ports"].items():
            assert bits(value, target["ports"][name]), name
        assert target["native_calls"] == dict(norm2_context=1, ff_context=1)
    for handle in handles:
        handle.remove()
    assert calls == [(1, 3, 4), (1, 3, 4)]


def test_hostile_profiles_actual_inputs_and_multiple_calls_reject_with_cleanup():
    adapter, state = fixture()
    with pytest.raises(ValueError, match="model identity"):
        W["qualify"](adapter, state, expected_model_identity="foreign", allow_cpu_fixture=True)
    with pytest.raises(ValueError, match="device/profile"):
        W["qualify"](adapter, state, expected_model_identity=adapter.model_identity)
    with pytest.raises(ValueError, match="joint0 entry"):
        predict(adapter, dict(state, cursor=1))
    with pytest.raises(ValueError, match="declared text rows"):
        predict(adapter, state, text_rows=[1, 1])
    with pytest.raises(ValueError, match="replacement return profile"):
        predict(adapter, state, text_rows=[1], mlp_return=torch.ones(1, 1, 4))
    block = adapter.model.transformer_blocks[0]
    initial = len(block._forward_pre_hooks)
    changed = copy.deepcopy(state)
    changed["text"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="actual native writer input differs"):
        with W["native_joint0"](adapter, state, expected_model_identity=adapter.model_identity, allow_cpu_fixture=True):
            adapter.advance(changed)
    assert len(block._forward_pre_hooks) == initial
    with pytest.raises(ValueError, match="one joint0 invocation"):
        with W["native_joint0"](adapter, state, expected_model_identity=adapter.model_identity, allow_cpu_fixture=True):
            adapter.advance(state)
            adapter.advance(state)
    assert len(block._forward_pre_hooks) == initial
    with pytest.raises(ValueError, match="capture incomplete"):
        with W["native_joint0"](adapter, state, expected_model_identity=adapter.model_identity, allow_cpu_fixture=True):
            pass
    assert len(block._forward_pre_hooks) == initial
