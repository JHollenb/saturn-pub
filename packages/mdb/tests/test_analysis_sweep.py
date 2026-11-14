"""Batched cell sweeps (``mdb.analysis.sweep.cell_sweep``) against the stepped engine: every arm
equals ``Run.set_cell`` + ``until logits`` on its own run, the null arm equals the stepped native
logits, and the sweep refuses a moved cursor or an edited run."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402
from test_engine_ports import PROMPT, build_model  # noqa: E402

from mdb.analysis.forecast import Refusal  # noqa: E402
from mdb.analysis.sweep import Cell, cell_sweep  # noqa: E402
from mdb.engine import Run  # noqa: E402

ROW = len(PROMPT) - 1
CASES = [("qwen2", "fp32"), ("qwen3", "fp32"), ("qwen3", "bf16")]
TOL = {"fp32": 1e-5, "bf16": 0.07}   # bf16: a batch of B rounds with another matmul shape


@pytest.fixture(scope="module", params=CASES, ids=lambda case: "-".join(case))
def setup(request):
    model = build_model(*request.param)
    return DecoderAdapter(model, granularity="operation"), request.param[1]


def _stepped(adapter, cells=()):
    run = Run(adapter, PROMPT)
    run.until(f"t{ROW}.L0.h")
    for cell in cells:
        run.set_cell(cell.layer, cell.tensor, cell.value.reshape(1, 1, 1, -1), rows=[cell.row],
                     kv=[cell.kv])
    run.until(f"t{ROW}.logits")
    return run.read().reshape(-1).float()


def _arms(adapter):
    generator = torch.Generator().manual_seed(5)
    d = adapter.head_dim

    def value():
        return torch.randn(d, generator=generator)

    return [[Cell(1, "v", 2, 1, value())], [Cell(0, "k", 4, 0, value())],
            [Cell(2, "v", 5, 0, value()), Cell(2, "k", 5, 0, value())],
            [Cell(1, "v", 2, 1, value()), Cell(0, "v", 0, 1, value())], []]


def test_each_arm_equals_set_cell_on_a_stepped_run(setup):
    adapter, dtype = setup
    arms = _arms(adapter)
    run = Run(adapter, PROMPT)
    run.until(f"t{ROW}.L0.h")
    native = _stepped(adapter)
    out = cell_sweep(run, arms, batch=2, stepped_logits=native)
    assert out["logits"].shape == (len(arms), native.numel()) and out["batches"] == 3
    assert out["null_vs_stepped"] <= TOL[dtype]
    for index, arm in enumerate(arms):
        expected = _stepped(adapter, arm)
        assert torch.allclose(out["logits"][index], expected, atol=TOL[dtype], rtol=0)
        assert int(out["argmax"][index]) == int(out["logits"][index].argmax())
    assert run.where == f"t{ROW}.L0.h" and not run.cells   # the run is not moved or edited
    # the edits reach the logits: arm 0 differs from native, the empty arm does not
    assert not torch.allclose(out["logits"][0], native, atol=TOL[dtype], rtol=0)
    assert torch.allclose(out["logits"][-1], native, atol=TOL[dtype], rtol=0)


def test_token_subset_and_fp32_batch_is_exact():
    adapter = DecoderAdapter(build_model("qwen2", "fp32"), granularity="operation")
    run = Run(adapter, PROMPT)
    run.until(f"t{ROW}.L0.h")
    arms = _arms(adapter)
    full = cell_sweep(run, arms, batch=8)
    sub = cell_sweep(run, arms, tokens=[3, 9, 17], batch=1)
    assert torch.allclose(sub["logits"], full["logits"][:, [3, 9, 17]], atol=1e-5, rtol=0)
    assert torch.equal(sub["argmax"], full["argmax"])


def test_refuses_a_moved_cursor_an_edited_run_and_bad_cells():
    adapter = DecoderAdapter(build_model("qwen3", "fp32"), granularity="operation")
    run = Run(adapter, PROMPT)
    run.until(f"t{ROW}.L1.q")
    with pytest.raises(Refusal, match="first port"):
        cell_sweep(run, [[]])
    run = Run(adapter, PROMPT)
    run.until(f"t{ROW}.L0.h")
    with pytest.raises(Refusal, match="before t"):
        cell_sweep(run, [[Cell(0, "v", ROW, 0, torch.zeros(adapter.head_dim))]])
    with pytest.raises(Refusal, match="values"):
        cell_sweep(run, [[Cell(0, "v", 1, 0, torch.zeros(3))]])
    run.set_cell(0, "v", 0.5, "add", rows=[1])
    with pytest.raises(Refusal, match="carries edits"):
        cell_sweep(run, [[]])
