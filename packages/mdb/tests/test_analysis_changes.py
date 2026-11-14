"""Native endpoint checks independent of the finite-change helpers."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis.changes import explain_changes  # noqa: E402
from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
from mdb.analysis.propagate import state_digest  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402


@pytest.mark.parametrize("family,dtype", [("qwen2", "fp32"), ("qwen2", "bf16"),
                                         ("qwen3", "fp32"), ("qwen3", "bf16")])
@pytest.mark.parametrize("kind", ["v", "k", "both", "sham"])
def test_native_branch_differences_and_signed_tails(family, dtype, kind):
    dbg = open_tiny(family, dtype)
    base = dbg.run([1, 5, 9, 3], name="base")
    base.until("logits")
    edited = dbg.run([1, 5, 9, 3], name="edited")
    if kind != "sham":
        for port in (("k", "v") if kind == "both" else (kind,)):
            edited.set_cell(0, port, 0.75, "add", rows=[1])
    edited.until("logits")
    before = [state_digest(r) for r in (base, edited)]
    result = explain_changes(base, edited, Margin(3, 4, 9), layer=0, top=1)
    assert before == [state_digest(r) for r in (base, edited)]
    report, tensors = result.report, result.tensors
    layer = dbg.adapter.backbone.layers[0]
    c = tensors["readout"]
    for section, port, weight in (("attention", "o", layer.self_attn.o_proj.weight),
                                   ("mlp", "m", layer.mlp.down_proj.weight)):
        native = (edited.read(f"L0.{port}").double() - base.read(f"L0.{port}").double()).flatten()
        reconstructed = sum(tensors[f"{section}_write"].values()) + tensors[f"{section}_gap"]
        torch.testing.assert_close(reconstructed, native, rtol=1e-12, atol=1e-12)
        rank = report[section]["sources" if section == "attention" else "neurons"]
        for name, value in rank["tail"].items():
            accounted = sum(row["parts"][name]["value"] for row in rank["rows"]) + value["value"]
            expected = float(tensors["source_parts" if section == "attention" else "neuron_parts"][name].sum())
            assert accounted == pytest.approx(expected, abs=1e-12)
        accounted = rank["total"]["value"] + report[section]["write_discrepancy_projection"]["value"]
        assert accounted == pytest.approx(float(c @ native), abs=1e-10)
    logits = [r.read("logits").double().flatten() for r in (base, edited)]
    delta = float(logits[1][4] - logits[1][9] - logits[0][4] + logits[0][9])
    consumer = report["consumer"]
    assert consumer["delta"]["value"] == delta
    assert sum(consumer[k]["value"] for k in (
        "residual_change", "normalization_scale_change", "native_discrepancy")) == pytest.approx(delta)
    # Independent expansion of the product from captured native gate and up operands.
    g0, g1 = [r.read("L0.g").double().flatten() for r in (base, edited)]
    u0, u1 = [r.read("L0.u").double().flatten() for r in (base, edited)]
    s0, s1 = [torch.nn.functional.silu(g) for g in (g0, g1)]
    expected = {"gate": u0 * (s1 - s0), "up": s0 * (u1 - u0),
                "interaction": (s1 - s0) * (u1 - u0)}
    for name, term in expected.items():
        torch.testing.assert_close(tensors["mlp"].terms[name], term, rtol=1e-12, atol=1e-12)
    if kind in ("v", "sham"):
        assert tensors["source_parts"]["routing"].abs().max() == 0
    if kind == "k":
        assert tensors["source_parts"]["payload"].abs().max() == 0
    if kind == "sham":
        assert delta == 0
        assert all(t.abs().max() == 0 for t in tensors["mlp"].terms.values())


def test_refuse_unaligned_inputs_and_direct_intermediate_edits():
    dbg = open_tiny("qwen2", "fp32")
    base = dbg.run([1, 5, 9, 3]); base.until("logits")
    other = dbg.run([1, 6, 9, 3]); other.until("logits")
    with pytest.raises(Refusal, match="identical consumed tokens"):
        explain_changes(base, other, Margin(3, 4, 9), layer=0)
    other = dbg.run([1, 5, 9, 3]); other.set("L0.m", 0.0, "scale"); other.until("logits")
    with pytest.raises(Refusal, match="explicit edit-displacement"):
        explain_changes(base, other, Margin(3, 4, 9), layer=0)


def test_forecast_writer_runs_before_native_and_failure_prevents_release(monkeypatch):
    from mdb.analysis import propagate
    from mdb.core.records import verify_record

    dbg = open_tiny("qwen2", "fp32")
    run = dbg.run([1, 5, 9, 3]); run.until("logits")
    before = state_digest(run)
    edit = propagate.Edit("t3.L1.f", 0.0, "scale")
    events = []
    original = propagate._measure

    def measure(*args, **kwargs):
        events.append("native")
        return original(*args, **kwargs)

    def writer(record):
        verify_record(record)
        assert state_digest(run) == before
        events.append("persist")

    monkeypatch.setattr(propagate, "_measure", measure)
    propagate.whatif(run, edit, Margin(3, 4, 9), before_native=writer)
    assert events == ["persist", "native"]
    assert state_digest(run) == before

    def broken_writer(record):
        raise OSError("storage unavailable")

    events.clear()
    with pytest.raises(OSError, match="storage unavailable"):
        propagate.whatif(run, edit, Margin(3, 4, 9), before_native=broken_writer)
    assert events == []
    assert state_digest(run) == before
