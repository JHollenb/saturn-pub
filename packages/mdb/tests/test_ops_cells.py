"""State-cell edits through the ops: named runs, values from another run (``ORIGIN:ADDR``), the
assignment sugar, the four-arm compare (base, A, B, A+B, interaction) against independent runs,
``whatif`` of a cell edit, and the refusals."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_ops_predict import unlabelled  # noqa: E402

import mdb  # noqa: E402
from mdb.analysis import compare, propagate  # noqa: E402
from mdb.analysis.forecast import Margin  # noqa: E402
from mdb.engine import Run  # noqa: E402
from mdb.engine.ports import tensor_digest  # noqa: E402
from mdb.ops import BadArgs, Refused, dispatch  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402
from mdb.ops.predict import build_edit, parse_assignment, session_digest  # noqa: E402
from mdb.repl.render import render  # noqa: E402
from mdb.repl.shell import _compare_args, _set_args  # noqa: E402

PROMPT, DONOR = [1, 5, 9, 3], [1, 7, 9, 3]
CONFIGS = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16"), ("qwen3", "fp32")]
A = "cache.L0.v[kv0,row=1]"
B = "cache.L1.k[kv0,row=1]"


@pytest.fixture(params=CONFIGS, ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


def _ctx(debugger, donor_at="t3.logits"):
    ctx = debugger.ops()
    dispatch(ctx, "start", {"prompt": DONOR, "name": "boiling"})
    dispatch(ctx, "until", {"target": donor_at})
    dispatch(ctx, "start", {"prompt": PROMPT})
    return ctx


def _block(run, layer, tensor, kv, row):
    return run.read_cache(layer, tensor)[:, kv:kv + 1, row:row + 1, :]


# --- values from another run ----------------------------------------------------------------


def test_each_origin_form_resolves_to_the_donor_tensor_bitwise(debugger, tmp_path):
    ctx = _ctx(debugger, "t4.L1.h")  # the donor committed t3 and its L0 of t4
    donor, run = debugger.runs["boiling"], ctx.run
    value, pointer = debugger.cell_value(f"boiling:{A}", run=run)
    assert torch.equal(value, _block(donor, 0, "v", 0, 1))
    assert pointer["via"] == "live" and pointer["sha256"] == tensor_digest(value)
    assert pointer["origin"] == "boiling" and pointer["where"] == donor.where
    # occurrences: the value row 3 wrote at v, and its post-RoPE key kr (the cache rows)
    v, _ = debugger.cell_value("boiling:t3.L0.v[kv0]", run=run)
    assert torch.equal(v, _block(donor, 0, "v", 0, 3))
    k, _ = debugger.cell_value("boiling:t3.L1.kr[kv0]", run=run)
    assert torch.equal(k, _block(donor, 1, "k", 0, 3))
    with pytest.raises(Refused, match="before RoPE"):
        debugger.cell_value("boiling:t3.L0.k[kv0]", run=run)
    # project forms: a durable checkpoint name (stored cut) and a session id (its cursor)
    durable = mdb.attach(debugger.model, project=tmp_path / "p")
    stored = durable.run(DONOR, name="donor")
    stored.until("t3.logits")
    durable.checkpoint(stored, "hot")
    here = durable.run(PROMPT)
    for origin, via in (("hot", "cut"), (durable.session_id, "cut")):
        found, pointer = durable.cell_value(f"{origin}:{A}", run=here)
        assert torch.equal(found, _block(stored, 0, "v", 0, 1)) and pointer["via"] == via
        assert pointer["recipe"] is not None
    del durable.runs["donor"]
    found, _ = durable.cell_value(f"hot:{A}", run=here)
    assert torch.equal(found, _block(stored, 0, "v", 0, 1))


def test_cross_run_refusals_name_what_differs(debugger, tmp_path, monkeypatch):
    ctx = _ctx(debugger, "t3.L0.h")
    run, donor = ctx.run, debugger.runs["boiling"]
    with pytest.raises(Refused, match="unknown origin 'nowhere'"):
        dispatch(ctx, "set", {"target": A, "value": f"nowhere:{A}"})
    with pytest.raises(Refused, match="has not computed row 3"):
        debugger.cell_value("boiling:cache.L0.v[row=3]", run=run)
    with pytest.raises(Refused, match="has not computed t3.L0.v"):
        debugger.cell_value("boiling:t3.L0.v", run=run)
    with pytest.raises(Refused, match="row 2 vs 1"):
        dispatch(ctx, "set", {"target": A, "value": "boiling:cache.L0.v[kv0,row=1:3]"})
    monkeypatch.setattr(donor, "_numeric_scope", lambda: {"torch_num_threads": 99})
    with pytest.raises(Refused, match="numeric scope.*torch_num_threads"):
        debugger.cell_value(f"boiling:{A}", run=run)
    monkeypatch.undo()
    # another model: a durable checkpoint of another identity in the same project
    other = mdb.attach(open_tiny(*_config(debugger), seed=5).model, project=tmp_path / "p")
    elsewhere = other.run(DONOR)
    other.checkpoint(elsewhere, "foreign")
    mine = mdb.attach(debugger.model, project=tmp_path / "p")
    with pytest.raises(Refused, match="model"):
        mine.cell_value(f"foreign:{A}", run=mine.run(PROMPT))
    assert run.cells == []


def _config(debugger):
    return debugger.family, {"torch.bfloat16": "bf16",
                             "torch.float32": "fp32"}[str(debugger.adapter.dtype)]


def test_set_from_another_run_stages_a_supplied_cell_with_its_pointer(debugger):
    ctx = _ctx(debugger)
    result = dispatch(ctx, "set", {"target": A, "value": f"boiling:{A}"})
    assert unlabelled(result) == []
    edit = result["edits"][0]
    assert result["cell"]["timing"] == "before" and result["staged_cells"] == 1
    assert edit["origin"]["origin"] == "boiling" and edit["sha256"] == edit["origin"]["sha256"]
    shown = dispatch(ctx, "print", {"address": A})
    assert shown["provenance"]["authority"] == "supplied"
    want = _block(debugger.runs["boiling"], 0, "v", 0, 1)
    assert torch.equal(_block(ctx.run, 0, "v", 0, 1), want)
    info = dispatch(ctx, "info", {"subject": "edits"})
    assert info["cells"]["staged_total"] == 1 and unlabelled(info) == []
    committed = dispatch(ctx, "commit", {})
    assert committed["committed_cells"] == 1
    assert dispatch(ctx, "print", {"address": A})["provenance"]["authority"] == "supplied"


# --- named runs and the REPL syntax -----------------------------------------------------------


def test_named_runs_switch_and_keep_their_checkpoints(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "checkpoint", {"name": "here"})
    runs = dispatch(ctx, "info", {"subject": "runs"})
    assert [(r["name"], r["current"]) for r in runs["runs"]] == [("boiling", False),
                                                                 ("main", True)]
    switched = dispatch(ctx, "frame", {"address": "boiling:"})
    assert switched["run"] == "boiling" and ctx.run is debugger.runs["boiling"]
    assert ctx.run.where == "t3.logits" and "here" not in ctx.checkpoints
    dispatch(ctx, "frame", {"address": "main:"})
    assert "here" in ctx.checkpoints and ctx.run is debugger.runs["main"]
    with pytest.raises(Refused, match="no run named"):
        dispatch(ctx, "frame", {"address": "cold:"})
    with pytest.raises(Refused, match="not a branch name"):
        dispatch(ctx, "start", {"prompt": PROMPT, "name": "9lives"})


def test_assignment_sugar_is_parsed_without_eval():
    assert parse_assignment(f"{A} += 0.5") == (A, 0.5, "add", None)
    assert parse_assignment(f"{A}-=2") == (A, -2.0, "add", None)
    assert parse_assignment(f"{A} *= 3") == (A, 3.0, "scale", None)
    assert parse_assignment(f"{A} /= 4") == (A, 0.25, "scale", None)
    assert parse_assignment(f"{A} = boiling:{A} mix 0.5") == (A, f"boiling:{A}", "mix", 0.5)
    assert parse_assignment("L0.q[h1]=-1 add") == ("L0.q[h1]", -1.0, "add", None)
    for bad in (f"{A} -= boiling:{A}", f"{A} /= 0", f"{A} = 1 mix", f"{A} += 1 add",
                "__import__('os') += 1 add 2"):
        with pytest.raises(BadArgs):
            parse_assignment(bad)
    assert _set_args(f"{A} = boiling:{A} mix 0.25") == {"target": A, "value": f"boiling:{A}",
                                                          "op": "mix", "dose": 0.25}
    assert _set_args("q 2 add") == {"target": "q", "value": 2.0, "op": "add"}
    assert _compare_args(f"{A} = boiling:{A} mix 0.5 {B} += 1 --consumer M[4,9]") == {
        "a": f"{A}=boiling:{A} mix 0.5", "b": f"{B}+=1", "consumer": "M[4,9]"}


# --- the four-arm experiment ---------------------------------------------------------------


def _independent(debugger, cells):
    """A fresh run with the donor blocks written by ``Run.set_cell`` directly (no op code, no
    origin resolver), run to t3.logits: the margin M[4,9] in float64."""
    donor = debugger.runs["boiling"]
    run = Run(debugger.adapter, PROMPT)
    run.until("t3.L0.h")
    for layer, tensor in cells:
        run.set_cell(layer, tensor, _block(donor, layer, tensor, 0, 1).clone(), rows=[1], kv=[0])
    run.until("t3.logits")
    logits = run.read().reshape(-1).to(torch.float64)
    return logits[4] - logits[9]


def test_four_arm_compare_equals_four_independent_runs(debugger):
    ctx = _ctx(debugger)
    before = session_digest(ctx)
    a = compare.Side(f"{A}=boiling:{A}", edits=[build_edit(ctx, A, f"boiling:{A}", "replace")])
    b = compare.Side(f"{B}=boiling:{B}", edits=[build_edit(ctx, B, f"boiling:{B}", "replace")])
    found = compare.compare(ctx.run, a, b, Margin(3, 4, 9))
    values = found["values"]
    arms = {"base": [], "A": [(0, "v")], "B": [(1, "k")], "A+B": [(0, "v"), (1, "k")]}
    for name, cells in arms.items():
        assert torch.equal(values[name], _independent(debugger, cells)), name
    assert torch.equal(values["interaction"],
                       values["A+B"] - values["A"] - values["B"] + values["base"])
    assert session_digest(ctx) == before
    result = dispatch(ctx, "compare", {"a": f"{A}=boiling:{A}", "b": f"{B}=boiling:{B}",
                                       "consumer": "M[4,9]"})
    assert unlabelled(result) == [] and result["interaction"]["value"]["authority"] == "measured"
    assert session_digest(ctx) == before


def test_overlapping_cell_sides_that_do_not_commute_refuse_a_joint_arm(debugger):
    ctx = _ctx(debugger)
    result = dispatch(ctx, "compare", {"a": f"{A}=boiling:{A} mix 0.5",
                                       "b": "cache.L0.v[row=1] += 1", "consumer": "M[4,9]"})
    assert result["interaction"]["state"] == "refused" and "order" in result["interaction"]["reason"]
    same = dispatch(ctx, "compare", {"a": f"{A} += 0.5", "b": f"{A} += 0.25",
                                     "consumer": "M[4,9]"})
    assert same["interaction"]["value"]["authority"] == "measured"


# --- whatif of a cell edit ---------------------------------------------------------------


def test_whatif_of_a_cell_edit_predicts_measures_and_restores(debugger):
    ctx = _ctx(debugger)
    before = session_digest(ctx)
    result = dispatch(ctx, "whatif", {"target": A, "value": f"boiling:{A}",
                                      "consumer": "M[4,9]"})
    assert unlabelled(result) == [] and session_digest(ctx) == before
    edit = result["edit"]
    assert edit["first_read"] == "t3.L0.A" and edit["stage_at"] == "t3.L0.h"
    assert edit["origin"]["origin"] == "boiling" and result["mode"] == "pending"
    measured = result["measured"]["delta"]["value"]
    native = float(_independent(debugger, [(0, "v")]) - _independent(debugger, []))
    assert measured == native
    assert result["prediction"]["state"] == "resolved"
    terms = sum(t["value"]["value"] for t in result["split"]["terms"])
    assert abs(terms - measured) <= 1e-12 * max(1.0, abs(measured))
    assert result["difference"]["native_minus_exact"].get("within_bound") in (True, None)
    assert result["state"] == result["effect"]["state"]


def test_whatif_of_a_cell_rewinds_when_the_consumer_is_computed(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "t3.logits"})
    before = session_digest(ctx)
    result = dispatch(ctx, "whatif", {"target": "cache.L1.v[row=1]", "value": 0.5, "op": "mix",
                                      "dose": 0.5, "consumer": "M[4,9]"})
    assert result["mode"] == "rewind" and result["edit"]["stage_at"] == "t3.L1.h"
    assert session_digest(ctx) == before
    edit = propagate.Edit("cache.L1.v[row=1]", 0.5, "mix", cell={
        "layer": 1, "tensor": "v", "rows": [1], "kv": None, "d": None}, dose=0.5)
    found = compare.measure(ctx.run, [edit], Margin(3, 4, 9))
    assert result["measured"]["edited"]["value"] == float(found["value"])
    with pytest.raises(Refused, match="cannot reach it|not downstream"):
        dispatch(ctx, "whatif", {"target": "cache.L1.v[row=3]", "value": 0.0,
                                 "consumer": "M[4,9]"})


# --- seek switches runs (M9) -------------------------------------------------------------------


def test_seek_with_a_run_name_switches_to_that_run_and_moves_it(debugger):
    ctx = _ctx(debugger)
    donor = debugger.runs["boiling"]
    assert ctx.run is not donor
    start = donor.where
    result = dispatch(ctx, "seek", {"target": "boiling:t3.L0.p"})
    assert ctx.run is donor and result["run"] == "boiling" and result["via"] == "run"
    assert donor.where == "t3.L0.p" != start and result["where"] == "t3.L0.p"
    # a named run seeks backwards too, and switching back finds the other run where it was
    dispatch(ctx, "seek", {"target": "boiling:t3.L0.h"})
    assert ctx.run.where == "t3.L0.h"
    with pytest.raises(Refused, match=r"no run named 'nobody'.*have: .*boiling"):
        dispatch(ctx, "seek", {"target": "nobody:t3.L0.p"})
    assert ctx.run is donor


def test_info_edits_says_how_many_cell_edits_it_left_out(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "set", {"target": A, "value": f"boiling:{A}"})
    dispatch(ctx, "set", {"target": B, "value": f"boiling:{B}"})
    info = dispatch(ctx, "info", {"subject": "edits", "limit": 1})
    assert len(info["cells"]["staged"]) == 1 and info["cells"]["staged_total"] == 2
    assert info["cells"]["omitted"] == {"staged": 1}
    full = dispatch(ctx, "info", {"subject": "edits", "limit": 2})
    assert "omitted" not in full["cells"] and unlabelled(info) == []
    shown = render(info).splitlines()
    assert "staged cell: +1 more of 2 (use limit= for the rest)" in shown
    assert not any("more of" in line for line in render(full).splitlines())
