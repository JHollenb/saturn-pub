"""Exact product algebra, row structure and the public native recorder lane."""
from pathlib import Path
import runpy

import numpy as np
import pytest

LAW = runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification/flux_mlp_state_law.py"))


def test_product_difference_keeps_gate_value_interaction_and_rounding():
    rng = np.random.default_rng(17)
    x0 = LAW["quantize"](rng.normal(size=(1, 4, 8)))
    x1 = LAW["quantize"](x0 + rng.normal(size=x0.shape) * .1)
    incoming = LAW["quantize"](rng.normal(size=(24, 8)))
    outgoing = LAW["quantize"](rng.normal(size=(8, 12)))
    low, high = [LAW["forecast"](x, incoming, outgoing) for x in (x0, x1)]
    parts = LAW["decompose"](low, high, outgoing)
    assert parts["closure_max_abs"] == 0
    assert np.array_equal(sum(parts["components"].values()), parts["product_delta"])
    assert np.any(parts["components"]["interaction"] != 0)
    assert np.any(parts["components"]["product_rounding"] != 0)


def test_midpoint_is_not_assumed_to_be_an_affine_operation():
    incoming = np.array([[1], [1]], np.float32)
    outgoing = np.ones((1, 1), np.float32)
    x0, x1 = np.zeros((1, 1, 1), np.float32), np.full((1, 1, 1), 2, np.float32)
    ys = [LAW["forecast"](x, incoming, outgoing)["output"] for x in (x0, x1, (x0+x1)/2)]
    assert not np.array_equal(ys[2], (ys[0]+ys[1])/2)


def test_frozen_new_cases_preserve_declared_row_locality():
    low = np.arange(24, dtype=np.float32).reshape(1, 3, 8) / 8
    high = low + 1
    cases = LAW["cases"](low, high)
    assert np.array_equal(cases["single_row"][:, 1:], low[:, 1:])
    assert np.array_equal(cases["single_row"][:, 0], high[:, 0])
    assert np.array_equal(cases["permuted_midpoint"], cases["midpoint"][:, ::-1])
    assert np.array_equal(low, np.arange(24, dtype=np.float32).reshape(1, 3, 8)/8)


def test_public_recorder_captures_cpu_bf16_named_substeps():
    torch = pytest.importorskip("torch")
    from mdb_lanes.forward_recording import record_forward
    from mdb_lanes.forward_ports import NativePortCatalog

    class Gate(torch.nn.Module):
        def forward(self, x):
            a, b = x.chunk(2, -1)
            return torch.nn.functional.silu(a) * b

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_in = torch.nn.Linear(8, 24, bias=False)
            self.act_fn = Gate()
            self.linear_out = torch.nn.Linear(12, 8, bias=False)

        def forward(self, x):
            return self.linear_out(self.act_fn(self.linear_in(x)))

    model = MLP().eval().to(torch.bfloat16)
    x = torch.arange(32, dtype=torch.bfloat16).reshape(1, 4, 8).clone()
    capture = record_forward(model, (x,), model_identity="tiny-MLP", input_identity="literal-arange32",
        execution=dict(backend="cpu", dtype="bfloat16"), capture_ports=True)
    ports = NativePortCatalog(capture.trace)
    for path in ("linear_in", "act_fn", "linear_out"):
        binding = ports.resolve("module:"+path+":return")
        assert binding["value_id"] in capture.observed_values
    assert sum(event["operator"] == "aten.silu.default" for event in capture.trace["events"]) == 1
    assert ports.resolve("parameter:linear_in.weight")["value_id"] in capture.initial_values


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_operands_are_instrument_errors(value):
    with pytest.raises(ValueError, match="finite"):
        LAW["quantize"]([value])
