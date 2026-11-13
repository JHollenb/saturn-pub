"""Tiny CPU contracts for the public second-order Qwen read producer."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from mdb_lanes.read_weight_taylor import (
    compile_taylor_program,
    evaluate_taylor_program,
    smooth_native_logweights,
)
from mdb_lanes.read_weight_transport import SmoothQwenReadTransport, compile_score_program


class NoForwardLinear:
    def __init__(self, output: int, input_width: int, *, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(output, input_width, generator=generator)
        self.bias = None

    def __call__(self, *_args, **_kwargs):  # pragma: no cover - proves functional execution
        raise AssertionError("Taylor transport must not call module forward")


class NoForwardNorm:
    def __init__(self, width: int, *, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(width, generator=generator)
        self.variance_epsilon = 1e-5

    def __call__(self, *_args, **_kwargs):  # pragma: no cover - proves functional execution
        raise AssertionError("Taylor transport must not call module forward")


class FakeAttention:
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, *, seed: int):
        self.num_heads = heads
        self.num_key_value_heads = kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.q_proj = NoForwardLinear(heads * head_dim, hidden, seed=seed + 1)
        self.k_proj = NoForwardLinear(kv_heads * head_dim, hidden, seed=seed + 2)
        self.v_proj = NoForwardLinear(kv_heads * head_dim, hidden, seed=seed + 3)
        self.o_proj = NoForwardLinear(heads * head_dim, hidden, seed=seed + 4)


class FakeMLP:
    def __init__(self, hidden: int, intermediate: int, *, seed: int):
        self.gate_proj = NoForwardLinear(intermediate, hidden, seed=seed + 1)
        self.up_proj = NoForwardLinear(intermediate, hidden, seed=seed + 2)
        self.down_proj = NoForwardLinear(hidden, intermediate, seed=seed + 3)


class FakeLayer:
    def __init__(self, hidden: int, intermediate: int, heads: int, kv_heads: int, head_dim: int, *, seed: int):
        self.input_layernorm = NoForwardNorm(hidden, seed=seed + 10)
        self.post_attention_layernorm = NoForwardNorm(hidden, seed=seed + 11)
        self.self_attn = FakeAttention(hidden, heads, kv_heads, head_dim, seed=seed + 20)
        self.mlp = FakeMLP(hidden, intermediate, seed=seed + 30)


def make_transport(*, hidden: int = 8, heads: int = 4, kv_heads: int = 2, head_dim: int = 2):
    layers = [
        FakeLayer(hidden, 12, heads, kv_heads, head_dim, seed=1000 + index * 100)
        for index in range(23)
    ]
    model = SimpleNamespace(
        model=SimpleNamespace(layers=layers),
        config=SimpleNamespace(
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
        ),
    )
    generator = torch.Generator().manual_seed(81)
    incoming = torch.randn(1, 1, hidden, generator=generator)
    baseline_attention = torch.randn(1, 1, hidden, generator=generator)
    history = {
        index: (
            torch.randn(1, kv_heads, 3, head_dim, generator=generator),
            torch.randn(1, kv_heads, 3, head_dim, generator=generator),
        )
        for index in (20, 21)
    }
    cos = torch.randn(1, 1, head_dim, generator=generator)
    sin = torch.randn(1, 1, head_dim, generator=generator)
    transport = SmoothQwenReadTransport(
        model,
        incoming=incoming,
        baseline_attention=baseline_attention,
        history=history,
        cos=cos,
        sin=sin,
    )
    return transport


def score_fixture():
    q0 = torch.tensor(
        [[[[0.5, -0.4]], [[0.2, 0.7]], [[-0.8, 0.3]], [[0.6, 0.1]]]], dtype=torch.float32
    )
    k_full = torch.tensor(
        [[
            [[0.4, 0.2], [0.1, -0.7], [0.5, 0.8], [-0.3, 0.6]],
            [[-0.2, 0.9], [0.3, -0.4], [0.6, 0.1], [0.7, -0.5]],
        ]],
        dtype=torch.float32,
    )
    scaling = 0.5
    repeated = k_full.repeat_interleave(2, dim=1)
    native_score0 = torch.matmul(q0.to(torch.bfloat16), repeated.to(torch.bfloat16).transpose(-2, -1))
    native_score0 = native_score0.float() * scaling
    return q0, k_full, native_score0, scaling


def compile_fixture():
    transport = make_transport()
    q0, k_full, native_score0, scaling = score_fixture()
    direction = torch.randn_like(transport.baseline_attention) * 0.05
    program = compile_taylor_program(
        transport,
        q0,
        k_full,
        native_score0,
        direction,
        kv_head=1,
        source_row=1,
        selected_rows=(0, 1, 2, 3),
        scaling=scaling,
    )
    return transport, q0, k_full, native_score0, direction, program, scaling


def test_nested_jvp_coefficients_match_symmetric_finite_difference():
    transport, q0, k_full, native_score0, direction, program, scaling = compile_fixture()
    epsilon = 5e-2
    plus = smooth_native_logweights(
        transport,
        q0,
        k_full,
        native_score0,
        direction,
        epsilon,
        kv_head=1,
        source_row=1,
        selected_rows=(0, 1, 2, 3),
        scaling=scaling,
    )
    minus = smooth_native_logweights(
        transport,
        q0,
        k_full,
        native_score0,
        direction,
        -epsilon,
        kv_head=1,
        source_row=1,
        selected_rows=(0, 1, 2, 3),
        scaling=scaling,
    )
    anchor = smooth_native_logweights(
        transport,
        q0,
        k_full,
        native_score0,
        direction,
        0.0,
        kv_head=1,
        source_row=1,
        selected_rows=(0, 1, 2, 3),
        scaling=scaling,
    )
    finite_first = (plus - minus) / (2 * epsilon)
    finite_second = (plus - 2 * anchor + minus) / (epsilon**2)
    torch.testing.assert_close(torch.tensor(program["c1"]), finite_first[0], rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(torch.tensor(program["c2"]), finite_second[0], rtol=3e-2, atol=3e-2)


def test_first_order_coefficient_matches_existing_score_program_and_route_split():
    transport, q0, k_full, native_score0, direction, program, scaling = compile_fixture()
    direction_receipt = transport.compile_direction(direction)
    score_program = compile_score_program(
        q0,
        k_full,
        native_score0,
        direction_receipt["q_dot"],
        direction_receipt["k_dot"],
        kv_head=1,
        source_row=1,
        scaling=scaling,
    )
    probability = score_program["logp0"].exp()
    expected = score_program["a"] - (probability * score_program["a"]).sum(dim=-1, keepdim=True)
    selected = expected.index_select(-1, torch.tensor(program["selected_rows"]))
    torch.testing.assert_close(torch.tensor(program["c1_score"]), selected[0])
    torch.testing.assert_close(torch.tensor(program["c1"]), torch.tensor(program["c1_score"]), atol=2e-4, rtol=2e-4)
    assert len(program["c2_route"]) == 2
    assert len(program["c2_route"][0]) == 4


@pytest.mark.parametrize(
    "method", ["order2", "order1", "stale", "routing-only", "wrong-curvature", "wrong-sign"]
)
def test_scalar_evaluator_keeps_native_anchor_and_does_not_need_torch(monkeypatch, method):
    _, _, _, _, _, program, _ = compile_fixture()
    import mdb_lanes.read_weight_taylor as taylor

    def forbidden_torch():  # pragma: no cover - called only if evaluator regresses
        raise AssertionError("framework-free evaluator imported torch")

    monkeypatch.setattr(taylor, "_torch", forbidden_torch)
    result = evaluate_taylor_program(program, 0.0, method=method)
    assert result["pred_logpB"] == program["logp0"]
    assert all(value == 0.0 for row in result["delta_logpB"] for value in row)
    for p_row, logp_row in zip(result["pB"], result["logpB"]):
        assert p_row == pytest.approx([math.exp(value) for value in logp_row])


def test_taylor_methods_have_distinct_curvature_controls():
    _, _, _, _, _, program, _ = compile_fixture()
    order2 = evaluate_taylor_program(program, 0.6, method="order2")
    order1 = evaluate_taylor_program(program, 0.6, method="order1")
    wrong = evaluate_taylor_program(program, 0.6, method="wrong-curvature")
    assert order2["pred_logpB"] != order1["pred_logpB"]
    assert wrong["pred_logpB"] != order2["pred_logpB"]


def test_scalar_evaluator_rejects_malformed_routing_curvature():
    _, _, _, _, _, program, _ = compile_fixture()
    malformed = dict(program)
    malformed["c2_route"] = [[0.0]]
    with pytest.raises(ValueError, match="curvature matrix"):
        evaluate_taylor_program(malformed, 0.2, method="routing-only")
