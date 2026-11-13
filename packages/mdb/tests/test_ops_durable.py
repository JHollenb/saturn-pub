"""REPL checkpoints become durable when the debugger is attached to a project."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import mdb  # noqa: E402
from mdb.core.errors import Refused  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops import dispatch  # noqa: E402

TOKENS = [1, 5, 9, 3]


def test_checkpoint_op_stores_and_restart_op_restores_without_a_live_run(tmp_path):
    first = mdb.attach(tiny_model("qwen2", "fp32"), project=tmp_path / "p")
    run = first.run(TOKENS)
    run.until("L1.sg")
    run.set("L1.u", torch.full(run.port_shape("u"), 0.125), "add")
    ctx = first.ops(run)
    made = dispatch(ctx, "checkpoint", {"name": "pre"})
    assert made["durable"]["blobs"] == 1 and made["where"] == run.where
    cut = run.checkpoint()["cut"]

    # A later process: same model, same project, no run yet.
    second = mdb.attach(tiny_model("qwen2", "fp32"), project=tmp_path / "p")
    later = second.ops(None)
    assert dispatch(later, "restart", {"name": "pre"})["where"] == run.where
    assert later.run.checkpoint()["cut"] == cut
    assert [edit["port"] for edit in later.run.edits] == ["u"]
    with pytest.raises(Refused, match="no durable checkpoint 'nope'"):
        dispatch(later, "restart", {"name": "nope"})


def test_without_a_project_checkpoints_stay_in_memory():
    dbg = mdb.attach(tiny_model("qwen2", "fp32"))
    with pytest.raises(Refused, match="no run to rewind"):
        dispatch(dbg.ops(None), "restart", {"name": "pre"})
    run = dbg.run(TOKENS)
    run.until("L0.p")
    assert "durable" not in dispatch(dbg.ops(run), "checkpoint", {"name": "pre"})
