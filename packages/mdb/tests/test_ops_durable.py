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


def test_a_checkpoint_name_is_at_most_64_characters(tmp_path):
    dbg = mdb.attach(tiny_model("qwen2", "fp32"), project=tmp_path / "p")
    run = dbg.run(TOKENS)
    run.until("L0.p")
    ctx = dbg.ops(run)
    with pytest.raises(Refused, match="at most 64 characters .this one has 65"):
        dispatch(ctx, "checkpoint", {"name": "n" * 65})
    assert ctx.checkpoints == {} and dbg.checkpoints() == []  # nothing stored
    assert dispatch(ctx, "checkpoint", {"name": "n" * 64})["durable"]["blobs"] == 0


@pytest.mark.parametrize("where", ["save", "record", "select"])
def test_an_interrupted_durable_checkpoint_is_durable_or_absent_never_memory_only(
    tmp_path, monkeypatch, where
):
    """The durable write runs first; the in-memory entry is added only after it returned. An
    interrupt at any point of the durable write leaves either no checkpoint or a durable one
    that ``restart`` finds."""
    from mdb.core import recipe as recipes

    dbg = mdb.attach(tiny_model("qwen2", "fp32"), project=tmp_path / "p")
    run = dbg.run(TOKENS)
    run.until("L0.p")
    ctx = dbg.ops(run)
    target = {"save": (type(dbg), "_save_recipe"), "record": (recipes, "record_checkpoint"),
              "select": (type(dbg.project), "select")}[where]
    original = getattr(*target)

    def interrupted(*args, **kwargs):
        if where != "save":
            original(*args, **kwargs)  # the step completed, then Ctrl-C landed
        raise KeyboardInterrupt

    monkeypatch.setattr(*target, interrupted)
    with pytest.raises(KeyboardInterrupt):
        dispatch(ctx, "checkpoint", {"name": "cp"})
    monkeypatch.undo()
    assert "cp" not in ctx.checkpoints
    durable = [c["name"] for c in dbg.checkpoints()]
    assert durable == ([] if where == "save" else ["cp"])
    if durable:
        run.until("L1.p")
        assert dispatch(ctx, "restart", {"name": "cp"})["where"] == f"t{len(TOKENS) - 1}.L0.p"


def test_a_missing_durable_checkpoint_names_the_session_not_a_filesystem_path(tmp_path):
    dbg = mdb.attach(tiny_model("qwen2", "fp32"), project=tmp_path / "p", session="mine")
    with pytest.raises(Refused, match="no durable checkpoint 'nope'") as raised:
        dispatch(dbg.ops(None), "restart", {"name": "nope"})
    assert str(tmp_path) not in str(raised.value) and "mine" in str(raised.value)
