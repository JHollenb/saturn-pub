"""Installed-package, fixed synthetic matrices; no pretrained model or weights."""
from __future__ import annotations

import copy
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock
from saturn_pub.contracts import ExecutionPoint
from saturn_pub.core import Adapter, Session
from saturn_pub.values import describe

FILE = Path(__file__).parents[1] / "qualification/flux_parallel_family_writer.py"
W = runpy.run_path(str(FILE))
CORE = ("R", "A", "M", "gate", "projection_return", "out", "image_in", "text_in", "image_out", "text_out")


def bits(left, right):
    return left.shape == right.shape and left.dtype == right.dtype and torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


class Toy(Adapter):
    """Public Session adapter over a small original block with supplied matrices."""
    model_identity = "controlled-fixed-matrix-parallel-writer"
    execution = dict(family="flux2-klein", device="cpu", dtype="torch.bfloat16", scope="arithmetic-fixture")

    def __init__(self):
        self.model = torch.nn.Module()
        self.model.config = SimpleNamespace(guidance_embeds=False)
        self.model.transformer_blocks = torch.nn.ModuleList([])
        self.model.single_transformer_blocks = torch.nn.ModuleList([
            Flux2SingleTransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4, mlp_ratio=2.)
            for _ in range(6)])
        with torch.no_grad():
            for index, (name, parameter) in enumerate(self.model.named_parameters()):
                if "norm_" in name:
                    parameter.fill_(1.)
                else:
                    values = ((torch.arange(parameter.numel()).reshape(parameter.shape) * 3 + index * 5) % 29 - 14) / 32
                    parameter.copy_(values)
        self.model.to(torch.bfloat16).eval().requires_grad_(False)
        self.bound = self.signature()

    def signature(self):
        return (self.model.training, tuple((id(value), value._version, value.dtype, value.device) for value in self.model.parameters()))

    def validate_execution(self):
        if self.signature() != self.bound:
            raise ValueError("fixture execution guard invalidated; rewrap")

    def validate(self, state):
        assert state["image"].shape == (1, 3, 8) and state["text"].shape == (1, 2, 8)

    def addresses(self, _state):
        return ("image", "text")

    def boundary(self, state):
        return f"step:{state['step']}/{state['phase']}:{state['cursor']}"

    def point(self, state):
        return ExecutionPoint(family="flux2-klein", logical_step=state["step"], phase=state["phase"],
            operator=state["phase"], index=state["cursor"], edge="before", local_clock=state["cursor"],
            next_operation=f"single_block.{state['cursor']}")

    def advance(self, state):
        self.validate_execution()
        out = self.model.single_transformer_blocks[state["cursor"]](
            hidden_states=torch.cat([state["text"], state["image"]], 1), encoder_hidden_states=None,
            temb_mod=state["context"]["single_mod"], image_rotary_emb=state["context"]["rope"],
            joint_attention_kwargs=None)
        ntext = state["text"].shape[1]
        cursor = state["cursor"] + 1
        return dict(state, image=out[:, ntext:], text=out[:, :ntext], cursor=cursor,
                    phase="readout" if cursor == len(self.model.single_transformer_blocks) else "single")


def fixture(index=4):
    adapter = Toy()
    image = ((torch.arange(24).reshape(1, 3, 8) % 11) - 5).to(torch.bfloat16) / 4
    text = ((torch.arange(16).reshape(1, 2, 8) % 7) - 3).to(torch.bfloat16) / 4
    image[0, 0, 0] = -0.
    mod = torch.tensor([.125] * 8 + [-.25] * 8 + [.75] * 8, dtype=torch.bfloat16)[None]
    angles = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 16
    state = dict(image=image, text=text, context=dict(single_mod=mod, rope=(angles.cos(), angles.sin())),
        conditioning=torch.tensor([[[1., -2.]]], dtype=torch.bfloat16), latent=torch.zeros(1, 3, 2, dtype=torch.bfloat16),
        timesteps=torch.tensor([1., .5]), sigmas=torch.tensor([1., .5, 0.]), step=0, phase="single", cursor=index,
        immutable={"token": "unchanged public payload"})
    return adapter, state


def functional(adapter, state, **kwargs):
    return W["functional_single"](adapter, state, index=state["cursor"], expected_model_identity=adapter.model_identity,
                                   allow_cpu_fixture=True, **kwargs)


def native(adapter, state, **kwargs):
    with W["native_single"](adapter, state, index=state["cursor"], expected_model_identity=adapter.model_identity,
                            allow_cpu_fixture=True, **kwargs) as capture:
        session = Session(adapter, state)
        session.continue_()
        actual = session.capture(retain=False).payload
    assert bits(actual["image"], capture["state"]["image"]) and bits(actual["text"], capture["state"]["text"])
    assert actual.keys() == capture["state"].keys()
    assert describe(actual) == describe(capture["state"])
    for name in ("conditioning", "latent", "timesteps", "sigmas"):
        assert bits(actual[name], capture["state"][name])
    assert actual["phase"] == capture["state"]["phase"] and actual["cursor"] == capture["state"]["cursor"]
    return capture


def pair():
    adapter, q = fixture()
    p = copy.deepcopy(q)
    p["image"][:, 0] += torch.tensor([.25, -.5, .75, -.25, 1., -.5, .25, .5], dtype=torch.bfloat16)
    p["text"][:, 1] -= .5
    p["conditioning"] += 1
    return adapter, q, p


def test_definitions_are_framework_free_and_graph_is_authenticated():
    code = "import runpy,sys; runpy.run_path(sys.argv[1]); assert not {'torch','diffusers','saturn_pub','numpy'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(FILE)], check=True)
    assert W["equations"]()["PROFILE"] == "flux2-klein-functional-diffusers-0.39-native-attention-v1"


def test_equations_match_frozen_graph_without_module_calls_and_keep_inputs_owned(monkeypatch):
    adapter, state = fixture()
    before = copy.deepcopy(state)
    monkeypatch.setattr(torch.nn.Module, "__call__", lambda *_a, **_kw: pytest.fail("equation lane called a module"))
    result = functional(adapter, state)
    image, text = W["equations"]()["functional_single"](adapter.model, 4, state["image"], state["text"], state["context"])
    assert bits(image, result["state"]["image"]) and bits(text, result["state"]["text"])
    assert set(result["ports"]) == set(W["PORT_NAMES"]) == set(result["descriptors"])
    assert result["profile"]["domain"] == "CPU arithmetic fixture only"
    assert bits(state["image"], before["image"]) and bits(state["text"], before["text"])
    with torch.inference_mode():
        result["ports"]["R"].zero_()
        result["state"]["conditioning"].zero_()
    assert bits(state["image"], before["image"]) and bits(state["conditioning"], before["conditioning"])


def test_actual_native_operand_returns_public_state_and_hook_cleanup():
    adapter, state = fixture()
    predicted = functional(adapter, state)
    observed = []
    handle = adapter.model.single_transformer_blocks[4].register_forward_pre_hook(lambda *_: observed.append("read-only"))
    modules = list(adapter.model.modules())
    initial = [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in modules]
    original_sdpa = F.scaled_dot_product_attention
    target = native(adapter, state)
    assert observed == ["read-only"]
    assert target["native_calls"] == dict(block=1, norm=1, attention=1, fused=1, norm_q=1, norm_k=1, mlp=1, projection=1, sdpa=1)
    assert initial == [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in modules]
    assert F.scaled_dot_product_attention is original_sdpa
    for name, value in predicted["ports"].items():
        assert bits(value, target["ports"][name]), name
        assert predicted["descriptors"][name]["content_sha256"] == target["descriptors"][name]["content_sha256"]
    assert bits(target["ports"]["out"], target["ports"]["R"] + target["ports"]["gate"] * target["ports"]["projection_return"])
    handle.remove()


def test_layout_retention_is_explicit_without_padded_view_duplication():
    adapter, state = fixture()
    full = native(adapter, state)
    raw = full["descriptors"]["q_raw"]
    kept = full["retained_descriptors"]["q_raw"]
    assert raw["shape"] == [1, 5, 2, 4] and raw["stride"][1] > 8
    assert raw["content_sha256"] == kept["content_sha256"] and raw["stride"] != kept["stride"]
    filtered = native(adapter, state, retain_ports=list(CORE))
    assert set(filtered["ports"]) == set(CORE) and set(filtered["descriptors"]) == set(W["PORT_NAMES"])
    assert "q_raw" in filtered["retention"]["hash_and_layout_only"]
    assert all(isinstance(value, torch.Tensor) for value in filtered["ports"].values())
    with pytest.raises(ValueError, match="unknown retained"):
        functional(adapter, state, retain_ports=["invented"])


def test_hybrids_use_one_full_concat_projection_and_keep_q_text_mlp_gates(monkeypatch):
    adapter, qs, ps = pair()
    q, p = functional(adapter, qs, retain_ports=CORE)["ports"], functional(adapter, ps, retain_ports=CORE)["ports"]
    original = F.linear
    calls = []

    def linear(value, weight, bias=None):
        calls.append(tuple(value.shape))
        return original(value, weight, bias)

    monkeypatch.setattr(F, "linear", linear)
    values = {}
    ntext = qs["text"].shape[1]
    for arm in ("Q", "R", "A", "RA"):
        result = W["construct_endpoint"](q, p, arm)
        r, a = q["R"].clone(), q["A"].clone()
        if arm in ("R", "RA"):
            r[:, ntext:] = p["R"][:, ntext:]
        if arm in ("A", "RA"):
            a[:, ntext:] = p["A"][:, ntext:]
        merged = torch.cat([a, q["M"]], -1)
        expected = r + q["gate"] * original(merged, adapter.model.single_transformer_blocks[4].attn.to_out.weight)
        assert bits(result["image"], expected[:, ntext:]) and bits(result["text"], q["text_out"])
        assert bits(result["state"]["conditioning"], qs["conditioning"])
        assert bits(result["ports"]["merge_input"], merged)
        assert result["construction"]["full_projection_shape"] == [1, 5, 24]
        if arm == "Q":
            assert describe(result["state"]) == describe(q.state)
        values[arm] = result["image"]
    assert calls == [(1, 5, 24)] * 4
    assert bits(values["Q"], q["image_out"])
    weight = adapter.model.single_transformer_blocks[4].attn.to_out.weight
    split_projection = original(q["A"], weight[:, :8]) + original(q["M"], weight[:, 8:])
    assert not bits(split_projection, q["projection_return"]), "toy must detect the unqualified split-GEMM rewrite"
    assert torch.isfinite(values["RA"].double() - values["R"].double() - values["A"].double() + values["Q"].double()).all()
    natural = W["construct_endpoint"](q, p, "P")
    assert bits(natural["state"]["conditioning"], ps["conditioning"]) and bits(natural["text"], p["text_out"])
    assert not bits(natural["state"]["conditioning"], qs["conditioning"])
    assert calls == [(1, 5, 24)] * 4
    with torch.inference_mode():
        natural["state"]["conditioning"].zero_()
    assert bits(p.state["conditioning"], ps["conditioning"])


def test_native_endpoint_is_actual_original_projection_consumer():
    adapter, qs, ps = pair()
    q, p = native(adapter, qs, retain_ports=CORE)["ports"], native(adapter, ps, retain_ports=CORE)["ports"]
    seen = []
    module = adapter.model.single_transformer_blocks[4].attn.to_out
    handle = module.register_forward_pre_hook(lambda _m, args: seen.append(args[0].clone()))
    for arm in ("Q", "R", "A", "RA"):
        expected = W["construct_endpoint"](q, p, arm)
        target = W["construct_endpoint"](q, p, arm, native=True)
        assert bits(expected["image"], target["image"]) and bits(target["text"], q["text_out"])
        assert bits(seen[-1], target["ports"]["merge_input"])
        assert target["construction"]["native_projection_calls"] == 1
    assert len(seen) == 4 and all(tuple(value.shape) == (1, 5, 24) for value in seen)
    handle.remove()


def test_matching_index_five_and_hostile_source_clock_identity_mutations():
    adapter, state = fixture(5)
    expected, target = functional(adapter, state), native(adapter, state)
    assert expected["state"]["phase"] == target["state"]["phase"] == "readout"
    assert bits(expected["state"]["image"], target["state"]["image"])
    with pytest.raises(ValueError, match="model identity"):
        W["qualify"](adapter, state, 5, expected_model_identity="foreign", allow_cpu_fixture=True)
    with pytest.raises(ValueError, match="matching step0"):
        W["functional_single"](adapter, state, 4, expected_model_identity=adapter.model_identity, allow_cpu_fixture=True)
    with pytest.raises(ValueError, match="matching step0"):
        functional(adapter, dict(state, step=1))
    with pytest.raises(ValueError, match="device/profile"):
        W["qualify"](adapter, state, 5, expected_model_identity=adapter.model_identity)
    bad = copy.deepcopy(state)
    bad["context"]["single_mod"][0, -1] += .5
    q, p = functional(adapter, state)["ports"], functional(adapter, bad)["ports"]
    with pytest.raises(ValueError, match="clock/modulation"):
        W["construct_endpoint"](q, p, "RA")
    with torch.inference_mode():
        q["R"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="mutated"):
        W["construct_endpoint"](q, q, "Q")
    with torch.no_grad():
        adapter.model.single_transformer_blocks[5].attn.to_out.weight[0, 0] += 1
    with pytest.raises(ValueError, match="invalidated"):
        functional(adapter, state)


def test_hostile_actual_operands_extra_calls_and_exception_cleanup():
    adapter, state = fixture()
    original_sdpa = F.scaled_dot_product_attention
    modules = list(adapter.model.modules())
    initial = [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in modules]
    changed = copy.deepcopy(state)
    changed["image"][0, 0, 0] = 1.
    cases = [("actual native combined input", lambda: adapter.advance(changed)),
             ("another transformer block", lambda: adapter.advance(dict(state, cursor=5))),
             ("one selected block", lambda: (adapter.advance(state), adapter.advance(state)))]
    for message, action in cases:
        with pytest.raises(ValueError, match=message):
            with W["native_single"](adapter, state, expected_model_identity=adapter.model_identity, allow_cpu_fixture=True):
                action()
        assert F.scaled_dot_product_attention is original_sdpa
        assert initial == [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in modules]
    with pytest.raises(ValueError, match="capture incomplete"):
        with W["native_single"](adapter, state, expected_model_identity=adapter.model_identity, allow_cpu_fixture=True):
            pass
    assert F.scaled_dot_product_attention is original_sdpa
    with pytest.raises(RuntimeError, match="caller stopped"):
        with W["native_single"](adapter, state, expected_model_identity=adapter.model_identity, allow_cpu_fixture=True):
            raise RuntimeError("caller stopped")
    assert F.scaled_dot_product_attention is original_sdpa
    assert native(adapter, state)["native_calls"]["block"] == 1
