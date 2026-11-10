"""Tiny CPU contracts for centered second-order Qwen read forecasts."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch

from mdb.read_weight_centered import (
    compile_at_center,
    compile_centered_path,
    evaluate_centered_program,
    prepare_centered_path,
)
from mdb.read_weight_taylor import compile_taylor_program
from mdb.read_weight_transport import SmoothQwenReadTransport


class NoForwardLinear:
    def __init__(self, output: int, input_width: int, *, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(output, input_width, generator=generator)
        self.bias = None

    def __call__(self, *_args, **_kwargs):  # pragma: no cover - proves functional execution
        raise AssertionError("centered transport must not call module forward")


class NoForwardNorm:
    def __init__(self, width: int, *, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(width, generator=generator)
        self.variance_epsilon = 1e-5

    def __call__(self, *_args, **_kwargs):  # pragma: no cover - proves functional execution
        raise AssertionError("centered transport must not call module forward")


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
    return q0, k_full, native_score0, scaling


def inputs():
    transport = make_transport()
    q0, k_full, native_score0, scaling = score_fixture()
    write_delta = torch.randn_like(transport.baseline_attention) * 0.05
    return transport, q0, k_full, native_score0, write_delta, scaling


def make_prepared():
    transport, q0, k_full, native_score0, write_delta, scaling = inputs()
    path = prepare_centered_path(
        transport,
        q0,
        k_full,
        native_score0,
        write_delta,
        kv_head=1,
        source_row=1,
        selected_rows=(0, 1, 2, 3),
        scaling=scaling,
    )
    return transport, q0, k_full, native_score0, write_delta, scaling, path


def compile_fixture(*, center: float = 0.5):
    transport, q0, k_full, native_score0, write_delta, scaling = inputs()
    program = compile_at_center(
        transport,
        q0,
        k_full,
        native_score0,
        write_delta,
        center=center,
        kv_head=1,
        source_row=1,
        selected_rows=(0, 1, 2, 3),
        scaling=scaling,
    )
    return transport, q0, k_full, native_score0, write_delta, scaling, program


def test_centered_nested_jvp_matches_finite_differences_at_midpoint():
    _, _, _, _, _, _, path = make_prepared()
    program = compile_centered_path(path, center=0.5)
    epsilon = 5e-2
    plus = path(0.5 + epsilon)
    minus = path(0.5 - epsilon)
    anchor = path(0.5)
    finite_first = (plus - minus) / (2 * epsilon)
    finite_second = (plus - 2 * anchor + minus) / (epsilon**2)
    torch.testing.assert_close(torch.tensor(program["c1"]), finite_first[0], rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(torch.tensor(program["c2"]), finite_second[0], rtol=3e-2, atol=3e-2)


def test_center_zero_matches_existing_taylor_v1_coefficients():
    transport, q0, k_full, native_score0, write_delta, scaling = inputs()
    old = compile_taylor_program(
        transport,
        q0,
        k_full,
        native_score0,
        write_delta,
        kv_head=1,
        source_row=1,
        selected_rows=(0, 1, 2, 3),
        scaling=scaling,
    )
    new = compile_at_center(
        transport,
        q0,
        k_full,
        native_score0,
        write_delta,
        center=0.0,
        kv_head=1,
        source_row=1,
        selected_rows=(0, 1, 2, 3),
        scaling=scaling,
    )
    torch.testing.assert_close(torch.tensor(new["logp_native0"]), torch.tensor(old["logp0"]))
    torch.testing.assert_close(torch.tensor(new["g_center"]), torch.tensor(old["logp0"]))
    torch.testing.assert_close(torch.tensor(new["c1"]), torch.tensor(old["c1"]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(torch.tensor(new["c2"]), torch.tensor(old["c2"]), rtol=1e-6, atol=1e-6)


def test_centered_zero_residual_is_retained_and_baseline_correction_is_exact():
    *_, program = compile_fixture(center=0.5)
    centered = evaluate_centered_program(program, 0.0, method="centered")
    assert centered["delta_logpB"] == program["zero_centered_residual"]
    assert any(abs(value) > 1e-8 for row in centered["centered_zero_residual"] for value in row)

    corrected = evaluate_centered_program(program, 0.0, method="baseline-corrected")
    assert corrected["pred_logpB"] == program["logp_native0"]
    assert corrected["delta_logpB"] == [[0.0] * len(row) for row in program["logp_native0"]]
    assert corrected["zero_delta_logpB"] == corrected["delta_logpB"]


def test_prepared_callable_is_reused_and_has_consistent_python_readout(monkeypatch):
    _, _, _, _, _, _, path = make_prepared()
    expected = path(0.4).detach().to(device="cpu", dtype=torch.float32).tolist()[0]
    assert path.to_python_matrix(0.4) == expected

    import mdb.read_weight_taylor as taylor

    def forbidden_prepare(*_args, **_kwargs):  # pragma: no cover - called only on regression
        raise AssertionError("compiling a prepared path must not prepare it again")

    monkeypatch.setattr(taylor, "_prepare", forbidden_prepare)
    program = compile_centered_path(path, center=0.5)
    assert program["selected_rows"] == [0, 1, 2, 3]
    assert program["query_heads"] == [2, 3]


def test_centered_program_is_json_serializable_and_evaluator_is_framework_free(monkeypatch):
    *_, program = compile_fixture(center=0.5)
    encoded = json.dumps(program, allow_nan=False)
    assert "mdb-smooth-qwen-centered-taylor-program-v1" in encoded

    import mdb.read_weight_centered as centered

    def forbidden_torch():  # pragma: no cover - called only if evaluator regresses
        raise AssertionError("centered scalar evaluator imported torch")

    monkeypatch.setattr(centered, "_torch", forbidden_torch)
    result = evaluate_centered_program(program, 0.25, method="centered")
    assert result["schema"] == "mdb-smooth-qwen-centered-taylor-evaluation-v1"
    assert result["pB"] == [[math.exp(value) for value in row] for row in result["logpB"]]


def test_wrong_sign_uses_negative_dose_and_curvature_control_is_distinct():
    *_, program = compile_fixture(center=0.5)
    negative = evaluate_centered_program(program, -0.3, method="centered")
    wrong_sign = evaluate_centered_program(program, 0.3, method="wrong-sign")
    wrong_curvature = evaluate_centered_program(program, 0.3, method="wrong-curvature")
    assert wrong_sign["pred_logpB"] == negative["pred_logpB"]
    assert wrong_curvature["pred_logpB"] != evaluate_centered_program(
        program, 0.3, method="centered"
    )["pred_logpB"]


def test_centered_compile_and_evaluator_reject_nonfinite_or_malformed_inputs():
    _, _, _, _, _, _, path = make_prepared()
    with pytest.raises(ValueError, match="center must be finite"):
        compile_centered_path(path, center=float("nan"))

    *_, program = compile_fixture(center=0.5)
    with pytest.raises(ValueError, match="dose must be finite"):
        evaluate_centered_program(program, float("inf"))

    malformed = dict(program)
    malformed["c2"] = [[0.0]]
    with pytest.raises(ValueError, match="inconsistent shapes"):
        evaluate_centered_program(malformed, 0.2)

    malformed = dict(program)
    malformed["c1"] = [row[:] for row in program["c1"]]
    malformed["c1"][0][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite scalar"):
        evaluate_centered_program(malformed, 0.2)
