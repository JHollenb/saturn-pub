"""The Qwen read-panel lane through the engine (``mdb_lanes.qwen_engine``) against the lane path
(``attention_port``), on a random tiny Qwen2 on the CPU.

Measured here (FP32): the engine's stepped run gives the lane's incremental logits bit for bit;
``mdb.trace``'s native ``p``, the stepped run's ``p`` and the lane's FP32 recompute agree bit
for bit; the engine's zero cut equals the lane's ``recompute`` zero cut bit for bit and differs
from the lane's ``delta`` zero cut in the last bits."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from mdb_lanes import attention_panels as P
from mdb_lanes import qwen_engine as QE

PLAN = (Path(__file__).resolve().parents[1] / "qualification" / "qwen_panels_20261112" / "full"
        / "read-float32.plan.json")
TOKENS = [1, 5, 9, 3, 6, 2]
PREFILL, LAST = 3, 5
HOT, COLD = 4, 9


@pytest.mark.skipif(not PLAN.exists(), reason="saved Qwen plan not in this checkout")
def test_expressible_arms_of_the_sealed_fp32_plan():
    panel = P.Panel.from_json(json.loads(PLAN.read_text()))
    table = {"receiver": [21], "own:freezing": [18], "own:icy": [2]}
    ok, reasons = {}, {}
    for arm in panel.arms:
        cuts, why = QE.expressible(arm, table, panel)
        (ok if cuts is not None else reasons)[arm.arm] = cuts if cuts is not None else why
    assert len(ok) == 25 and ok["native"] == []
    assert ok["z:L20H5:freezing/cut_recompute"] == [(20, (5,), 21, 18)]
    assert ok["z:L21H0@rows-icy:freezing/cut_recompute"] == [(21, (0,), 21, 2)]
    assert "delta" in reasons["z:L20H5:freezing"]
    assert "renormalising" in reasons["m:L20H5:freezing"]
    assert any("kind 'keep'" in r for r in reasons.values())
    assert sum("span layers" in r for r in reasons.values()) >= 8


@pytest.fixture(scope="module")
def tiny():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import mdb
    from transformers import DynamicCache
    model = mdb.open_tiny("qwen2", "fp32").model
    t = torch.tensor([TOKENS])
    parent = DynamicCache(config=model.config)
    with torch.inference_mode():          # the lane worker's order: rows 0..K-1, then one row at a time
        model(input_ids=t[:, :PREFILL], past_key_values=parent, use_cache=True)
        for j in range(PREFILL, LAST):
            model(input_ids=t[:, j:j + 1], past_key_values=parent, use_cache=True)

    def forward():
        with torch.inference_mode():
            return model(input_ids=t[:, LAST:LAST + 1], past_key_values=copy.deepcopy(parent),
                         use_cache=True).logits[0, -1]

    dbg, run = QE.attach_run(model, None, TOKENS, prefill=PREFILL)
    return torch, model, forward, run


def _lane_cut(torch, model, forward, layer, heads, source, zero_method):
    from mdb_lanes import attention_port as ap
    port = ap.AttentionPort(ap.QwenBackend(model), rows={"receiver": [LAST], "src": [source]},
                            zero_method=zero_method)
    cut = ap.cut(["receiver"], ["src"], heads=[(layer, h) for h in heads], renormalize=False)
    with port.apply(cut, forwards=1):
        return forward()


def test_base_logits_and_attention_agree_bit_for_bit(tiny):
    torch, model, forward, run = tiny
    from mdb_lanes import attention_port as ap
    layers = model.config.num_hidden_layers
    lane = forward()
    assert torch.equal(QE.run_logits(run, LAST), lane)
    traced, info, logits = QE.trace_attention(model, forward, layers=layers)
    assert torch.equal(logits, lane)                      # the trace is passive
    assert info["authority"] == "measured" and info["calls"] == 1
    assert traced == QE.run_attention(run, LAST, layers)
    port = ap.AttentionPort(ap.QwenBackend(model))
    obs = ap.observe(lambda read: read.probs()[0, :, -1, :].double().cpu(), steps={0})
    with port.apply(observers=[obs], forwards=1):
        forward()
    assert torch.equal(torch.as_tensor(obs.stack()[0]), torch.tensor(traced, dtype=torch.float64))


@pytest.mark.parametrize("layer,heads,source", [(1, (0,), 2), (0, (1,), 1), (1, (0, 1), 2)])
def test_engine_zero_cut_is_the_lane_recompute_cut(tiny, layer, heads, source):
    torch, model, forward, run = tiny
    from mdb.analysis.forecast import Margin
    (measured,) = QE.engine_cuts(run, [(layer, heads, LAST, source)], Margin(LAST, HOT, COLD))
    recompute = _lane_cut(torch, model, forward, layer, heads, source, "recompute")
    delta = _lane_cut(torch, model, forward, layer, heads, source, "delta")
    edited = measured["edited_logits"]
    assert torch.equal(edited, recompute)
    assert float((edited - delta).abs().max()) <= 1e-6    # delta rounds the native output again
    assert measured["verdict"] == "effect" and measured["residual_spacings"] == 0.0
    assert QE.margin_of(edited, HOT, COLD) == pytest.approx(
        QE.margin_of(measured["base_logits"], HOT, COLD) + measured["delta"], abs=1e-12)


def test_engine_rows_score_with_the_panel_controls(tiny):
    torch, model, forward, run = tiny
    if not PLAN.exists():
        pytest.skip("saved Qwen plan not in this checkout")
    panel = P.Panel.from_json(json.loads(PLAN.read_text()))
    keep = {"native", "z:L20H5:freezing/cut_recompute", "z:L20H5@rows-icy:freezing/cut_recompute",
            "z:L20H5:freezing", "m:L20H5:freezing"}
    arms = [a for a in panel.arms if a.arm in keep]
    arms = [replace(a, units=[[1, 0]]) if a.units else a for a in arms]       # remap onto the tiny model
    small = replace(panel, arms=arms)
    table = {"receiver": [LAST], "own:freezing": [2], "own:icy": [1]}
    rows, lane_only = QE.engine_rows(small, run, table, hot=HOT, cold=COLD, row=LAST)
    by = {r["arm"]: r for r in rows}
    assert set(by) == {"native", "z:L20H5:freezing/cut_recompute", "z:L20H5@rows-icy:freezing/cut_recompute"}
    assert set(lane_only) == {"z:L20H5:freezing", "m:L20H5:freezing"}
    assert by["native"]["margin"] == QE.margin_of(forward(), HOT, COLD)
    lane = _lane_cut(torch, model, forward, 1, (0,), 2, "recompute")
    assert by["z:L20H5:freezing/cut_recompute"]["margin"] == QE.margin_of(lane, HOT, COLD)


def test_contributions_are_the_engine_reads_numbers(tiny):
    torch, model, forward, run = tiny
    from mdb.analysis.forecast import Margin
    from mdb.analysis.reads import reads
    consumer = Margin(LAST, HOT, COLD)
    mine = QE.contributions(run, LAST, 2, consumer)
    theirs = reads(run, "t2", consumer=consumer, top=1, causal=False).tensors["entries"]
    assert len(mine) == len(theirs) == model.config.num_hidden_layers * model.config.num_attention_heads
    for entry in theirs:
        got = mine[(entry["layer"], int(entry["name"].rsplit(".h", 1)[1]))]
        assert got["mass"] == entry["mass"]["value"]
        assert got["to_consumer"] == entry["to_consumer"]["value"]
