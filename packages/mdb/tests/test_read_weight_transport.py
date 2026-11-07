"""Mechanics for the baseline-only smooth Qwen read transport producer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mdb.read_weight_transport import (
    SmoothQwenReadTransport,
    compile_score_program,
    evaluate_score_program,
)


class NoForwardLinear:
    def __init__(self, output: int, input_width: int, *, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(output, input_width, generator=generator)
        self.bias = None

    def __call__(self, *_args, **_kwargs):  # pragma: no cover - a failure proves no module call
        raise AssertionError("transport must use functional linears, not module forward")


class NoForwardNorm:
    def __init__(self, width: int, *, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(width, generator=generator)
        self.variance_epsilon = 1e-5

    def __call__(self, *_args, **_kwargs):  # pragma: no cover - a failure proves no module call
        raise AssertionError("transport must use functional RMSNorm, not module forward")


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
    return SmoothQwenReadTransport(
        model,
        incoming=incoming,
        baseline_attention=baseline_attention,
        history=history,
        cos=cos,
        sin=sin,
    )


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
    q_dot = torch.tensor(
        [[[[0.3, 0.2]], [[-0.1, 0.4]], [[0.5, -0.6]], [[0.2, 0.7]]]], dtype=torch.float32
    )
    k_dot = torch.tensor([[[[0.4, -0.2]], [[-0.3, 0.5]]]], dtype=torch.float32)
    return q0, k_full, native_score0, q_dot, k_dot, scaling


def test_smooth_shadow_jvp_matches_finite_difference_without_module_forwards():
    transport = make_transport()
    direction = torch.randn_like(transport.baseline_attention) * 0.05
    compiled = transport.compile_direction(direction)

    epsilon = 1e-3
    plus_q, plus_k = transport.shadow(transport.baseline_attention + epsilon * direction)
    minus_q, minus_k = transport.shadow(transport.baseline_attention - epsilon * direction)
    finite_q = (plus_q - minus_q) / (2 * epsilon)
    finite_k = (plus_k - minus_k) / (2 * epsilon)

    assert compiled["shadow_q0"].shape == (1, 4, 1, 2)
    assert compiled["shadow_k0"].shape == (1, 2, 1, 2)
    assert compiled["logical_linears"] == 19
    assert compiled["logical_cost"]["native_forward_calls"] == 0
    assert compiled["host_enqueue_seconds"] >= 0.0
    assert compiled["q_dot"].dtype is torch.float32
    assert compiled["k_dot"].dtype is torch.float32
    torch.testing.assert_close(compiled["q_dot"], finite_q, rtol=4e-3, atol=4e-3)
    torch.testing.assert_close(compiled["k_dot"], finite_k, rtol=4e-3, atol=4e-3)


def test_score_transport_uses_exact_exponential_identity_and_zero_anchor():
    q0, k_full, native_score0, q_dot, k_dot, scaling = score_fixture()
    program = compile_score_program(
        q0,
        k_full,
        native_score0,
        q_dot,
        k_dot,
        kv_head=1,
        source_row=1,
        scaling=scaling,
    )
    dose = 0.25
    result = evaluate_score_program(program, dose)
    delta = dose * program["a"]
    delta = delta.clone()
    delta[..., -1] += dose**2 * program["b"]
    expected_all = program["logp0"] + delta
    expected_all = expected_all - torch.logsumexp(expected_all, dim=-1, keepdim=True)
    torch.testing.assert_close(result["pred_logpB"], expected_all[..., 1])
    torch.testing.assert_close(result["delta_logpB"], expected_all[..., 1] - program["logp0"][..., 1])
    torch.testing.assert_close(result["pB"], result["pred_logpB"].exp())

    anchor = evaluate_score_program(program, 0.0)
    assert torch.equal(anchor["pred_logpB"], program["logp0"][..., 1])
    assert torch.equal(anchor["delta_logpB"], torch.zeros_like(anchor["delta_logpB"]))


def test_current_key_cross_term_and_query_only_controls_are_separate():
    q0, k_full, native_score0, q_dot, k_dot, scaling = score_fixture()
    program = compile_score_program(
        q0,
        k_full,
        native_score0,
        q_dot,
        k_dot,
        kv_head=1,
        source_row=1,
        scaling=scaling,
    )
    selected_q0 = q0[:, 2:, 0, :]
    selected_qdot = q_dot[:, 2:, 0, :]
    current_kdot = k_dot[:, 1, 0, :]
    expected_b = (selected_qdot * current_kdot[:, None, :]).sum(dim=-1) * scaling
    torch.testing.assert_close(program["b"], expected_b)
    expected_current_key = (selected_q0 * current_kdot[:, None, :]).sum(dim=-1) * scaling
    torch.testing.assert_close(
        program["a"][..., -1] - program["a_query"][..., -1], expected_current_key
    )

    dose = 0.5
    query_only = evaluate_score_program(program, dose, method="query-only")
    query_delta = dose * program["a_query"]
    query_logits = program["logp0"] + query_delta
    query_logits = query_logits - torch.logsumexp(query_logits, dim=-1, keepdim=True)
    torch.testing.assert_close(query_only["pred_logpB"], query_logits[..., 1])

    linear = evaluate_score_program(program, dose, method="linear-logweight")
    expected_linear_delta = dose * (
        program["a"][..., 1] - (program["logp0"].exp() * program["a"]).sum(dim=-1)
    )
    torch.testing.assert_close(linear["delta_logpB"], expected_linear_delta)

    wrong = evaluate_score_program(program, dose, method="wrong-sign")
    assert not torch.equal(wrong["pred_logpB"], evaluate_score_program(program, dose)["pred_logpB"])


@pytest.mark.parametrize("method", ["stale", "query-only", "linear-logweight"])
def test_all_score_methods_keep_zero_dose_baseline_anchor(method):
    q0, k_full, native_score0, q_dot, k_dot, scaling = score_fixture()
    program = compile_score_program(
        q0,
        k_full,
        native_score0,
        q_dot,
        k_dot,
        kv_head=0,
        source_row=2,
        scaling=scaling,
    )
    result = evaluate_score_program(program, 0.0, method=method)
    assert torch.equal(result["pred_logpB"], program["logp0"][..., 2])
    assert torch.equal(result["pB"], program["logp0"][..., 2].exp())
