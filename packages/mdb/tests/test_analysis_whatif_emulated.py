"""The emulated change in whatif (M51): sealed next to the linear and exact changes, it predicts
the native change; the verdict holds the native change against it within one dtype spacing at
the consumer (``verdict.FLOOR_SPACINGS``, supplied, the constant verify uses).

Measured on tiny models (no tolerance): the emulated change equals the native change bit for bit
for port, cross-run, head-selected and cache-cell edits at cursors ahead of the edit, between
the edit and the consumer and past the consumer, and on a model whose rounded scores tie (the
exact chain is tens of spacings off there). A wrong identity in the emulation is not confirmed
and names the stop where emulated and native part.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_analysis_forecast import TIE_PROMPT, _tie_model  # noqa: E402

from mdb.analysis.forecast import EmulatedRowModel, Margin  # noqa: E402
from mdb.analysis.propagate import Edit, whatif, whatif_many  # noqa: E402
from mdb.analysis.verdict import EFFECT, NO_EFFECT, NOT_CONFIRMED  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops.debugger import attach  # noqa: E402

CLEAN = [1, 7, 9, 3, 11, 2]
MAIN = [1, 5, 9, 3, 11, 2]
ROW = 5
CURSORS = (f"t{ROW}.L0.h", f"t{ROW}.L3.h", f"t{ROW}.L5.h", f"t{ROW}.logits")
CONSUMER = Margin(ROW, 6, 7)


@pytest.fixture(scope="module", params=["bf16", "fp32"])
def debugger(request):
    return attach(tiny_model("qwen2", request.param, layers=6))


def _edits(debugger):
    clean = debugger.run(CLEAN)
    clean.until(f"t{ROW}.logits")
    head = torch.zeros(1, 1, debugger.adapter.hidden_size, dtype=torch.bool)
    head[..., debugger.adapter.head_dim:2 * debugger.adapter.head_dim] = True
    a = f"t{ROW}.L4.A"
    cell = {"layer": 5, "tensor": "v", "rows": [1], "kv": None, "d": None}
    return {
        "port add": Edit(f"t{ROW}.L4.m", 0.5, "add"),
        "cross-run value": Edit(f"t{ROW}.L4.m", clean.read(f"t{ROW}.L4.m").clone()),
        "head-selected": Edit(a, clean.read(a).clone(), mask=head, selection="h1"),
        "cell add": Edit("cache.L5.v[row=1]", 4.0, "add", cell=dict(cell)),
        "cross-run cell": Edit("cache.L5.v[row=1]",
                               clean.read_cache(5, "v")[:, :, 1:2, :].clone(), cell=dict(cell)),
    }


def _emulated_equals_native(report, result):
    effect = report["effect"]
    assert report["prediction"]["emulated"]["value"]["authority"] == "derived-exact"
    assert effect["emulated"]["value"] == effect["native"]["value"], effect
    assert torch.equal(result.tensors["emulated"].reshape(-1),
                       result.tensors["native_delta"].reshape(-1))
    assert effect["residual"]["value"] == 0.0 and effect["residual_spacings"]["value"] == 0.0
    assert effect["floor"]["value"] == effect["spacing"]["value"]
    assert effect["floor"]["authority"] == "supplied"
    assert effect["verdict"] == (EFFECT if effect["native"]["value"] != 0 else NO_EFFECT)
    terms = {t["name"]: t["value"]["value"] for t in report["split"]["terms"]}
    assert terms["residual"] == 0.0 and "rounding effect" in terms
    native = report["measured"]["delta"]["value"]
    assert abs(sum(terms.values()) - native) <= 1e-9 * max(1.0, abs(native)), terms


@pytest.mark.parametrize("name", ["port add", "cross-run value", "head-selected", "cell add",
                                  "cross-run cell"])
def test_the_emulated_change_is_the_native_change_at_every_cursor(debugger, name):
    edit = _edits(debugger)[name]
    for cursor in CURSORS:
        run = debugger.run(MAIN)
        run.until(cursor)
        result = whatif(run, edit, CONSUMER)
        _emulated_equals_native(result.report, result)


def test_many_whatifs_share_the_emulated_base_and_each_equals_its_own(debugger):
    """``whatif_many`` (reads) shares one emulated base per (row, consumer, cursor) and keeps the
    edited port's value per port: edits at different layers each get their own."""
    run = debugger.run(MAIN)
    run.until(f"t{ROW}.logits")
    requests = [(Edit(f"t{ROW}.L{layer}.m", 0.5, "add"), CONSUMER) for layer in (1, 3, 4)]
    many = whatif_many(run, requests)
    for (edit, consumer), result in zip(requests, many):
        alone = whatif(run, edit, consumer)
        assert torch.equal(result.tensors["emulated"], alone.tensors["emulated"])
        _emulated_equals_native(result.report, result)


@pytest.mark.parametrize("cursor", ["ahead", "rewind"])
def test_score_ties_the_exact_chain_misses_are_emulated_bit_for_bit(cursor):
    """The M42 tie model: the native L0 scores tie in bf16, so the exact change is far from the
    native one; the emulated change is the native one and the verdict is an effect."""
    run = _tie_model().run(TIE_PROMPT)
    row = run.parse(run.where)[0]
    run.until(f"t{row}.L0.h" if cursor == "ahead" else f"t{row}.logits")
    for edit in (Edit(f"t{row}.L0.h", 0.5, "add"), Edit(f"t{row}.L0.A", 0.0, "scale")):
        result = whatif(run, edit, Margin(row, 4, 9))
        report = result.report
        _emulated_equals_native(report, result)
        assert report["mode"] == ("pending" if cursor == "ahead" else "rewind")


def test_a_wrong_identity_is_not_confirmed_and_names_where_it_parts(debugger, monkeypatch):
    """Both chains use a wrong RMSNorm (x 1.5; the tiny models' MLPs barely move the margin, so
    a wrong SiLU parts by one spacing only): the emulated change parts from the native one
    beyond the floor, and the stage check finds the native norm outside the bound of the
    (wrong) identity on its native operand."""
    from mdb.analysis.forecast import RowModel

    def wrong(original):
        def rms(self, *args):
            value, bound = original(self, *args)
            return value * 1.5, bound
        return rms

    monkeypatch.setattr(RowModel, "_rms", wrong(RowModel._rms))
    monkeypatch.setattr(EmulatedRowModel, "_rms", wrong(EmulatedRowModel._rms))
    run = debugger.run(MAIN)
    run.until(f"t{ROW}.L0.h")
    effect = whatif(run, Edit(f"t{ROW}.L0.h_out", 4.0, "add"), CONSUMER).report["effect"]
    assert effect["residual_spacings"]["value"] > 1, effect
    assert (effect["verdict"], effect["state"]) == (NOT_CONFIRMED, "resolved"), effect
    assert f"t{ROW}.L0.a: the native value lies" in effect["reason"], effect["reason"]
    assert "outside the rounding bound of its identity" in effect["reason"]


def test_a_sub_resolution_edit_reports_the_exact_change_the_dtype_cannot_show(debugger):
    run = debugger.run(MAIN)
    run.until(f"t{ROW}.L0.h")
    tiny = 1e-7 if debugger.adapter.dtype == torch.bfloat16 else 1e-14
    effect = whatif(run, Edit(f"t{ROW}.L4.m", tiny, "add"), CONSUMER).report["effect"]
    assert effect["native"]["value"] == effect["emulated"]["value"] == 0.0
    assert (effect["verdict"], effect["state"]) == (NO_EFFECT, "resolved")
    assert f"no change at {effect['dtype']} resolution" in effect["note"]
    assert "exact" in effect["note"]


def test_a_refused_emulation_decides_on_the_native_change_alone(debugger):
    """A staged edit ahead of the cursor rewrites an operand the chains recompute: the emulated
    chains are refused and the verdict is native only (edited against base)."""
    run = debugger.run(MAIN)
    run.until(f"t{ROW}.L0.h")
    run.set(f"t{ROW}.L0.q", 0.01, "add")
    report = whatif(run, Edit(f"t{ROW}.L4.m", 0.5, "add"), CONSUMER).report
    assert report["prediction"]["emulated"]["state"] == "refused"
    effect = report["effect"]
    assert effect["basis"].startswith("native only") and "emulated" not in effect
    assert effect["verdict"] == (EFFECT if effect["native"]["value"] != 0 else NO_EFFECT)
