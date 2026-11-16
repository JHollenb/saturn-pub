"""The mdb side of the stepping speedups: ``lookahead="auto"`` serves every port of an unedited
layer from one full-layer probe (bitwise the stop-probe values) and falls back to stop probes
once an edit is staged; ``each`` records the same per-item bytes through ``held_total``; the
MLP-weight analyses keep no float64 weight matrix across layers or calls and give the same
values with or without a shared cache."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402
from test_engine_ports import PROMPT, build_model  # noqa: E402

from mdb.analysis import symbols  # noqa: E402
from mdb.analysis.forecast import Margin  # noqa: E402
from mdb.analysis.symbols import readout  # noqa: E402
from mdb.engine import Run  # noqa: E402

CASES = [("qwen2", "bf16"), ("qwen3", "fp32")]


@pytest.fixture(scope="module", params=CASES, ids=lambda case: "-".join(case))
def adapter(request):
    return DecoderAdapter(build_model(*request.param), granularity="operation")


def _walk_ports(run, row):
    values = {}
    while run.where != f"t{row + 1}.start":
        run.step("port")
        if run._port is not None:
            values[run.where] = (run.read().clone(), list(run.status()["computed_past_cursor"]))
    return values


def test_auto_serves_a_row_from_one_probe_per_layer_bitwise(adapter):
    row = len(PROMPT) - 1
    walked = {}
    for mode in ("none", "layer", "auto"):
        run = Run(adapter, PROMPT, lookahead=mode)
        walked[mode] = (_walk_ports(run, row), dict(run.calls),
                        [m.fingerprint for m in run._trail])
    none, layer, auto = walked["none"], walked["layer"], walked["auto"]
    assert none[0].keys() == auto[0].keys() == layer[0].keys()
    for where, (value, past) in auto[0].items():
        assert torch.equal(value, none[0][where][0]) and value.dtype == none[0][where][0].dtype
        assert past == layer[0][where][1]  # what ran past the cursor is reported, as "layer"
    assert auto[1] == layer[1] and auto[1]["probe"] == adapter.layers  # one probe per layer
    assert none[1]["probe"] > auto[1]["probe"]
    assert auto[2] == none[2] == layer[2]  # the same cuts


def test_auto_with_a_staged_edit_stops_at_the_cursor_as_none(adapter):
    runs = {mode: Run(adapter, PROMPT, lookahead=mode) for mode in ("none", "auto")}
    for run in runs.values():
        run.until("L0.q")
        run.set("L0.m", 0.0, "scale")
        run.until("L0.k")
    none, auto = runs["none"], runs["auto"]
    assert auto._cache.stop == none._cache.stop == "k"
    assert auto.status()["computed_past_cursor"] == none.status()["computed_past_cursor"] == []
    assert torch.equal(auto.read(), none.read())
    for run in runs.values():
        run.until("logits")
    assert torch.equal(auto.read(), none.read())
    assert [m.fingerprint for m in auto._trail] == [m.fingerprint for m in none._trail]


def test_held_total_is_the_held_bytes_total(adapter):
    run = Run(adapter, PROMPT, prefill=1)
    run.until(f"t{len(PROMPT) - 1}.logits")
    assert run.held_total() == run.held_bytes()["total"]


def test_mlp_weight_analyses_cache_no_weight_matrix_and_agree_with_a_fresh_cache(adapter):
    prompt = [1, 5, 9, 3]
    run = Run(adapter, prompt)
    run.until("t3.logits")
    reader = readout(run, Margin(3, 4, 9))
    donor = Run(adapter, [2, 7, 9, 3])
    value = donor.read_cache(0, "v", staged=False).clone()[:, 0:1, 1:2]
    edited = Run(adapter, prompt)
    edited.set_cell(0, "v", value, rows=[1], kv=[0])
    edited.until("t3.logits")
    shared: dict = {}
    first = symbols.hop_edge(run, edited, reader, cell_layer=0, spectral=True, cache=shared)
    again = symbols.hop_edge(run, edited, reader, cell_layer=0, spectral=True, cache=shared)
    fresh = symbols.hop_edge(run, edited, reader, cell_layer=0, spectral=True)
    assert first == again == fresh
    forecast = symbols.gate_switch_forecast(run, edited, reader, cell_layer=0, ks=(1,),
                                            cache=shared)
    assert forecast == symbols.gate_switch_forecast(run, edited, reader, cell_layer=0, ks=(1,))
    d_ff = adapter.model.config.intermediate_size
    for key, held in shared.items():
        assert key[0] in ("cd", "perm")
        assert held.dim() == 1 and held.numel() in (d_ff, adapter.hidden_size)
