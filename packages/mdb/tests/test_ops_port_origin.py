"""Activation patching of a port from another run (M28): ``set ADDR = ORIGIN:ADDR`` on a port,
in ``set``, ``whatif`` and ``compare``.

A clean/corrupted pair on tiny Qwen2 (bf16) and Qwen3 (fp32): a whole residual patched from the
clean run reproduces the clean logits bit for bit; a one-head ``A[h1]`` patch equals the same
masked edit with the clean tensor supplied directly; the manifest, the receipt and checkpoints
carry the origin pointer; the clean run's cursor and state digest are unchanged by the fetch;
with a project the edited run's pointer resolves after the clean run is gone; the refusals.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_ops_predict import unlabelled  # noqa: E402

import mdb  # noqa: E402
from mdb.analysis.propagate import state_digest  # noqa: E402
from mdb.core import recipe as recipes  # noqa: E402
from mdb.engine import Run  # noqa: E402
from mdb.engine.ports import tensor_digest  # noqa: E402
from mdb.ops import Refused, dispatch  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402
from mdb.ops.predict import session_digest  # noqa: E402

CLEAN, CORRUPT = [1, 7, 9, 3], [1, 5, 9, 3]   # row 1 differs
LAST = [1, 7, 9, 5]                            # differs from CLEAN at the last row only
CONFIGS = [("qwen2", "bf16"), ("qwen3", "fp32")]


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda c: "-".join(c))
def tiny(request):
    return open_tiny(*request.param)


def _ctx(debugger, corrupt=CORRUPT):
    ctx = debugger.ops()
    dispatch(ctx, "start", {"prompt": CLEAN, "name": "clean"})
    dispatch(ctx, "until", {"target": "t3.logits"})
    dispatch(ctx, "start", {"prompt": corrupt})
    return ctx


def _logits(run):
    run.until("t3.logits")
    return run.read("t3.logits")


def _head(debugger, head):
    d = debugger.adapter.head_dim
    return slice(head * d, (head + 1) * d)


def _direct(debugger, prompt, address, value, mask=None):
    """A fresh run with ``value`` written by ``Run.set`` directly (no op code, no origin)."""
    run = Run(debugger.adapter, prompt)
    run.until(address.rsplit(".", 1)[0] + ".h")
    run.set(address, value, "replace", mask=mask)
    return _logits(run)


def test_whole_residual_from_the_clean_run_is_the_clean_run(tiny):
    """The one differing row's residual ``h`` at any layer patched from clean: every later value
    of that row is the clean one, so the logits are the clean logits bit for bit."""
    ctx = _ctx(tiny, LAST)
    clean = tiny.runs["clean"]
    want = clean.read("t3.logits")
    assert not torch.equal(_logits(Run(tiny.adapter, LAST)), want)
    for layer in range(tiny.adapter.layers):
        dispatch(ctx, "start", {"prompt": LAST})
        dispatch(ctx, "until", {"target": f"t3.L{layer}.h"})
        result = dispatch(ctx, "set", {"target": f"L{layer}.h", "value": f"clean:L{layer}.h"})
        assert result["source"] == f"clean:t3.L{layer}.h"
        assert torch.equal(_logits(ctx.run), want), layer


def test_every_position_patched_cells_for_the_prompt_rows_and_the_port_for_the_last(tiny):
    """With an earlier row differing, the full patch is the prompt rows' cache cells (every
    layer, rows 0..2) plus the last row's residual port: again the clean logits bit for bit."""
    ctx = _ctx(tiny)
    want = tiny.runs["clean"].read("t3.logits")
    for layer in range(tiny.adapter.layers):
        for tensor in ("k", "v"):
            cell = f"cache.L{layer}.{tensor}[row=0:3]"
            dispatch(ctx, "set", {"target": cell, "value": f"clean:{cell}"})
    dispatch(ctx, "set", {"target": "L0.h", "value": "clean:t3.L0.h"})
    assert torch.equal(_logits(ctx.run), want)


def test_one_head_patch_equals_the_same_edit_supplied_directly(tiny):
    ctx = _ctx(tiny)
    clean = tiny.runs["clean"]
    digest, where = state_digest(clean), clean.where
    result = dispatch(ctx, "set", {"target": "L0.A[h1]", "value": "clean:L0.A[h1]"})
    assert unlabelled(result) == []
    block = clean.read("t3.L0.A")[..., _head(tiny, 1)]
    edit = result["edits"][0]
    assert edit["selection"] == "h1" and edit["mask_count"] == block.numel()
    pointer = edit["origin"]
    assert pointer["origin"] == "clean" and pointer["via"] == "live"
    assert pointer["address"] == "clean:t3.L0.A[h1]" and pointer["where"] == "t3.logits"
    assert pointer["sha256"] == tensor_digest(block) and pointer["shape"] == list(block.shape)
    assert result["origin"] == pointer
    # the fetch left the clean run where it was, state digest included
    assert state_digest(clean) == digest and clean.where == where
    patched = _logits(ctx.run)
    full = torch.zeros_like(clean.read("t3.L0.A"))
    full[..., _head(tiny, 1)] = block
    mask = torch.zeros(full.shape, dtype=torch.bool)
    mask[..., _head(tiny, 1)] = True
    assert torch.equal(patched, _direct(tiny, CORRUPT, "t3.L0.A", full, mask))
    assert not torch.equal(patched, _logits(Run(tiny.adapter, CORRUPT)))
    # the receipt of the committed transition, the checkpoint and info edits name the origin
    committed = ctx.run.committed_edits()
    assert [e["origin"]["sha256"] for e in committed if "origin" in e] == [pointer["sha256"]]
    path = ctx.run.checkpoint(pin=False)["path_edits"]
    assert path[0]["edits"][0]["origin"]["origin"] == "clean"
    info = dispatch(ctx, "info", {"subject": "edits"})
    assert info["committed"][0]["edits"][0]["origin"]["address"] == "clean:t3.L0.A[h1]"
    assert state_digest(clean) == digest


def test_another_row_and_layer_of_the_clean_run(tiny):
    """ADDR2 may name another occurrence: the clean run's L1 MLP output into this run's L0."""
    ctx = _ctx(tiny)
    clean = tiny.runs["clean"]
    dispatch(ctx, "set", {"target": "L0.m", "value": "clean:t3.L1.m"})
    assert torch.equal(_logits(ctx.run), _direct(tiny, CORRUPT, "t3.L0.m",
                                                 clean.read("t3.L1.m")))


def test_whatif_and_compare_take_a_value_of_another_run(tiny):
    ctx = _ctx(tiny)
    before = session_digest(ctx)
    clean = tiny.runs["clean"]
    digest = state_digest(clean)
    result = dispatch(ctx, "whatif", {"target": "L0.A[h1]", "value": "clean:L0.A[h1]",
                                      "consumer": "M[4,9]"})
    assert unlabelled(result) == [] and session_digest(ctx) == before
    assert result["edit"]["origin"]["origin"] == "clean"
    assert result["edit"]["source"] == "clean:t3.L0.A[h1]"
    assert result["prediction"]["state"] in ("resolved", "refused")
    if result["prediction"]["state"] == "resolved":
        assert result["prediction"]["exact"]["value"]["authority"] == "derived-exact"
    full = torch.zeros_like(clean.read("t3.L0.A"))
    full[..., _head(tiny, 1)] = clean.read("t3.L0.A")[..., _head(tiny, 1)]
    mask = torch.zeros(full.shape, dtype=torch.bool)
    mask[..., _head(tiny, 1)] = True
    edited = _direct(tiny, CORRUPT, "t3.L0.A", full, mask).reshape(-1).to(torch.float64)
    base = _logits(Run(tiny.adapter, CORRUPT)).reshape(-1).to(torch.float64)
    assert result["measured"]["edited"]["value"] == float(edited[4] - edited[9])
    assert result["measured"]["delta"]["value"] == float((edited[4] - edited[9])
                                                         - (base[4] - base[9]))
    found = dispatch(ctx, "compare", {"a": "L0.A[h1]=clean:L0.A[h1]", "b": "L1.m=clean:L1.m",
                                      "consumer": "M[4,9]"})
    assert unlabelled(found) == [] and session_digest(ctx) == before
    assert state_digest(clean) == digest


def test_a_project_stores_the_value_so_the_edited_pointer_outlives_the_clean_run(tiny, tmp_path):
    dbg = mdb.attach(tiny.model, project=tmp_path / "p")
    ctx = _ctx(dbg)
    clean = dbg.runs["clean"]
    block = clean.read("t3.L0.A")[..., _head(tiny, 1)]
    value = clean.read("t3.L0.v")
    dispatch(ctx, "frame", {"address": "clean:"})
    dispatch(ctx, "checkpoint", {"name": "hot"})
    dispatch(ctx, "frame", {"address": "main:"})
    del dbg.runs["clean"]
    # a durable checkpoint as the origin: restored from its cut it starts at its cursor, so the
    # occurrence before it comes from a replay (a cell occurrence the same way)
    staged = dispatch(ctx, "set", {"target": "L0.A[h1]", "value": "hot:L0.A[h1]"})
    assert staged["origin"]["via"] == "replay" and staged["origin"]["recipe"] is not None
    assert staged["origin"]["sha256"] == tensor_digest(block)
    found, pointer = dbg.cell_value("hot:t3.L0.v[kv0]", run=ctx.run)
    assert pointer["via"] == "replay"
    assert torch.equal(found.reshape(-1), value[..., :tiny.adapter.head_dim].reshape(-1))
    dispatch(ctx, "abort", {})
    dbg.runs["clean"] = clean
    dispatch(ctx, "set", {"target": "L0.A[h1]", "value": "clean:L0.A[h1]"})
    dispatch(ctx, "commit", {})
    dispatch(ctx, "until", {"target": "t3.L1.a"})
    want = ctx.run.read("t3.L1.a")
    pointer = dispatch(ctx, "print", {"address": "L1.a"})["pointer"]
    assert pointer["recipe"] is not None
    recipe = recipes.load(dbg.project, pointer["recipe"])
    stored = recipe["transitions"][0]["edits"][0]
    assert stored["origin"]["origin"] == "clean" and stored["value"]["kind"] == "tensor"
    assert dbg.project.store.has_tensor(stored["value"]["ref"])
    del dbg.runs["clean"]
    for tier in ("replay", None):
        found = mdb.resolve(pointer, session=dbg, **({"tiers": (tier,)} if tier else {}))
        assert found.state == "resolved" and torch.equal(found.value, want), tier
    # a durable checkpoint replays the edit from its blob, the clean run gone
    dispatch(ctx, "checkpoint", {"name": "patched"})
    again = dbg.restart("patched")
    assert torch.equal(again.read("t3.L1.a"), want)


def test_without_a_project_the_edited_pointer_says_what_would_resolve_it(tiny, tmp_path):
    ctx = _ctx(tiny)
    dispatch(ctx, "set", {"target": "L0.A[h1]", "value": "clean:L0.A[h1]"})
    dispatch(ctx, "until", {"target": "t3.L0.o"})
    pointer = dispatch(ctx, "print", {"address": "L0.o"})["pointer"]
    assert pointer["recipe"] is None and len(pointer["address"]["origin"]["edits"]) == 1
    mdb.attach(tiny.model, project=tmp_path / "p")
    found = mdb.resolve(pointer, project=tmp_path / "p")
    assert found.state == "refused"
    assert "another run" in found.reason and "project=" in found.reason


def test_refusals_say_what_to_do_and_stage_nothing(tiny):
    ctx = _ctx(tiny)
    run = ctx.run
    cases = [
        ("L0.A[h1]", "clean:L0.A[h0:2]", r"selects shape \[1, 1, \d+\]; t3.L0.A\[h1\] selects "
                                         r".*select the same block on both sides"),
        ("L0.A", "nowhere:L0.A", "unknown origin 'nowhere'"),
        ("L0.A", "clean:t4.L0.A", r"clean has not computed t4.L0.A .*until t4.L0.A"),
        ("L0.A", "clean:t2.L0.A", r"before the start.*cache cells"),
        ("L0.A", "clean:cache.L0.v", "not a cache cell"),
        ("L0.A", "@" + "0" * 64 + ":L0.A", "not a run"),
    ]
    for target, value, message in cases:
        with pytest.raises(Refused, match=message):
            dispatch(ctx, "set", {"target": target, "value": value})
        with pytest.raises(Refused, match=message):
            dispatch(ctx, "whatif", {"target": target, "value": value, "consumer": "M[4,9]"})
    assert run.edits == [] and run.cells == []
