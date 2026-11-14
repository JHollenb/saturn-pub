"""``analysis.circuit_map`` (frozen cross-item map), ``summary.modes`` and the cell consumer of
``symbols.hop_edge`` on tiny models.

A slot binds the rows its rule names (R_argmax equals a hand argmax of the reader's heads), the
bound slots are ordinary declared symbols (the table still partitions the mass), ``freeze``
fills constants from ``summarize`` and ``modes``, the map's digest survives a JSON round trip,
the cell census equals a hand projection of each cell's unit write, and the cell ladder of
``hop_edge`` closes on the cell's native change.
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis import circuit_map as cm  # noqa: E402
from mdb.analysis import summary, symbols  # noqa: E402
from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
from mdb.analysis.reads import layer_module  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

PROMPT = [1, 5, 9, 3, 7, 2]
CONFIGS = [("qwen3", "fp32"), ("llama", "bf16")]


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


def _map(rule_layer=0, sets=None):
    slots = [cm.Slot("carrier", "carrier", "R_argmax", layers=(0,), kv=(0,),
                     rule_layer=rule_layer, rule_heads=(0, 1)),
             cm.Slot("desc", "descriptor", "rest")]
    edges = [cm.Edge("desc", "carrier", "read")]
    return cm.Map(slots, edges, {"kind": "margin"}, sets or {})


def test_slot_and_map_validate_and_round_trip():
    with pytest.raises(Refusal):
        cm.Slot("x", "nonsense", "gt")
    with pytest.raises(Refusal):
        cm.Slot("x", "carrier", "R_argmax")             # no rule layer / heads
    with pytest.raises(Refusal):
        cm.Slot("x", "carrier", "gt", layers=(1,))      # layers without kv
    with pytest.raises(Refusal):
        cm.Map([cm.Slot("a", "carrier", "gt")], [cm.Edge("a", "b", "read")], {}, {})
    m = _map(sets={"countries": [4, 9, 11]})
    again = cm.Map.from_json(json.loads(json.dumps(m.to_json())))
    assert again.to_json() == m.to_json()
    assert again.digest() == m.digest()
    m2 = cm.freeze(m, [{"x": {"value": 1.0, "authority": "measured"}, "best": 4}], ["best"])
    assert m2.digest() != m.digest()
    assert cm.Map.from_json(json.loads(json.dumps(m2.to_json()))).digest() == m2.digest()


def test_modes_counts_categorical_paths():
    results = [{"a": {"layer": 20, "kv": 6}, "t": [1, 2]}, {"a": {"layer": 20, "kv": 1}},
               {"a": {"layer": 18, "kv": 6}, "t": [1, 2]}, {"b": 1}]
    found = {m["path"]: m for m in summary.modes(results, ["a.layer", "a.kv", "t", "z"])}
    assert found["a.layer"]["mode"] == 20 and found["a.layer"]["count"] == 2
    assert found["a.layer"]["n"] == 3
    assert found["a.kv"]["mode"] == 6 and found["a.kv"]["count"] == 2
    assert found["t"]["mode"] == (1, 2) and found["t"]["n"] == 2
    assert found["z"]["n"] == 0 and found["z"]["mode"] is None
    assert summary.leaf({"x": [{"v": {"value": 3.0, "authority": "measured"}}]}, "x.0.v") == 3.0


def test_bind_r_argmax_matches_a_hand_argmax_and_the_table_still_partitions(debugger):
    run = debugger.run(PROMPT)
    run.until("t5.logits")
    item = {"name": "i", "span": [1, 2, 3, 4], "last": 5}
    m = _map()
    p = run.read("t5.L0.p").float().reshape(run.adapter.num_heads, -1)
    hand = max([1, 2, 3, 4], key=lambda r: (float(p[0, r] + p[1, r]), r))
    rows = cm.slot_rows(m.slot("carrier"), run, item)
    assert rows == [hand]
    report = cm.item_report(m, run, item, consumer=Margin(5, 4, 9))
    names = [b["symbol"] for b in report["table"]["bindings"]]
    assert names[:2] == ["carrier", "desc"]
    assert report["slots"]["carrier"] == [f"cache.L0.v[kv0,row={hand}]"]
    assert report["slots"]["desc"] == [f"t{r}" for r in [1, 2, 3, 4] if r != hand]
    assert report["table"]["accounting"]["within_bound"] is True
    assert symbols.declared(run) == {}                   # the slots are released
    frozen = cm.freeze(m, [report, report], ["table.bindings.carrier.symbol"])
    mode = frozen.constants["modes"][0]
    assert mode["mode"] == "carrier" and mode["count"] == 2
    assert frozen.constants["summary"]["n"] == 2


def test_freeze_lines_up_slots_by_name_when_an_item_binds_no_row_for_a_slot():
    """``bind`` skips a slot that binds no row, so page position is not a slot identity: the
    summary must key table bindings by symbol name (review 2026-11-14, summary.py positional
    paths)."""
    def claim(x):
        return {"value": x, "authority": "measured"}

    full = {"table": {"bindings": [{"symbol": "carrier", "mass": claim(1.0)},
                                   {"symbol": "desc", "mass": claim(5.0)}]}}
    no_carrier = {"table": {"bindings": [{"symbol": "desc", "mass": claim(7.0)}]}}
    frozen = cm.freeze(_map(), [full, no_carrier, full], ["table.bindings.desc.symbol"])
    paths = {row["path"]: row for row in frozen.constants["summary"]["numeric"]}
    assert "table.bindings.0.mass" not in paths
    assert "table.bindings.desc.mass" in paths              # present in every report
    assert "table.bindings.carrier.mass" not in paths       # present in only some: partial
    assert frozen.constants["summary"]["partial"] >= 1
    desc = paths["table.bindings.desc.mass"]
    vals = [v["value"] if isinstance(v, dict) else v for k, v in desc.items() if k in ("min", "max")]
    assert vals and min(vals) >= 5.0                         # never mixes the carrier's 1.0
    assert frozen.constants["modes"][0]["mode"] == "desc"


def test_cell_census_equals_a_hand_projection_of_each_cells_unit_write(debugger):
    run = debugger.run(PROMPT)
    run.until("t5.logits")
    ad = run.adapter
    final = getattr(ad.backbone, ad.spec.final_norm)
    tokens = [4, 9, 11]
    ro = {"set": final.weight.double()[None, :] * ad.model.lm_head.weight.double()[tokens]}
    values = {layer: run.read_cache(layer, "v", staged=False) for layer in range(ad.layers)}
    writers = {layer: cm.group_writers(layer_module(ad, layer).self_attn.o_proj.weight,
                                       ad.num_heads, ad.num_kv_heads) for layer in values}
    census = cm.cell_census(values, writers, [2, 4], ro)
    scores = census["scores"]["set"]
    assert scores.shape == (ad.layers, 2, ad.num_kv_heads, len(tokens))
    dh = ad.head_dim
    group = torch.arange(ad.num_heads) // (ad.num_heads // ad.num_kv_heads)
    for layer in range(ad.layers):
        w = layer_module(ad, layer).self_attn.o_proj.weight.double()
        for i, r in enumerate([2, 4]):
            for g in range(ad.num_kv_heads):
                v = values[layer][0, g, r].double()
                write = sum(w[:, h * dh:(h + 1) * dh] @ v for h in range(ad.num_heads)
                            if int(group[h]) == g)
                hand = ro["set"] @ write
                assert torch.allclose(scores[layer, i, g], hand, rtol=1e-10, atol=1e-12)
    assert cm.rank_of(torch.tensor([0.1, 0.3, 0.2]), tokens, 11) == 2
    assert cm.rank_of(torch.tensor([0.1, 0.3, 0.2]), tokens, 99) is None


def test_hop_edge_cell_consumer_closes_on_the_cells_native_change(debugger):
    ad = debugger.run(PROMPT).adapter
    top = ad.layers - 1
    if top < 1:
        pytest.skip("needs two layers")
    target = symbols.CellTarget(row=3, layer=top, kv=0, a=4, b=9)
    run = debugger.run(PROMPT, prefill=2)
    run.until("t5.logits")
    reader = symbols.cell_readout(run, target)
    assert reader.kind == "cell" and reader.layer == top
    v = run.read_cache(top, "v", staged=False)[0, 0, 3].double()
    final = getattr(ad.backbone, ad.spec.final_norm)
    d = final.weight.double() * (ad.model.lm_head.weight.double()[4] - ad.model.lm_head.weight.double()[9])
    w_g = cm.group_writers(layer_module(ad, top).self_attn.o_proj.weight, ad.num_heads,
                           ad.num_kv_heads)[0]
    assert reader.native == pytest.approx(float(d @ (w_g @ v)), rel=1e-12, abs=1e-12)
    h = run.read(f"t3.L{top}.h").double().reshape(-1)
    fp32 = ad.model.lm_head.weight.dtype == torch.float32
    assert float(reader.vector @ h) == pytest.approx(reader.native, rel=1e-4 if fp32 else 5e-2,
                                                     abs=1e-6 if fp32 else 5e-2)
    donor = debugger.run([2, 7, 9, 3, 7, 2])
    value = donor.read_cache(0, "v", staged=False).clone()[:, 0:1, 1:2]
    edited = debugger.run(PROMPT, prefill=2)
    edited.set_cell(0, "v", value, rows=[1], kv=[0])
    edited.until("t5.logits")
    edge = symbols.hop_edge(run, edited, reader, cell_layer=0, top_neurons=2)
    native = edge["consumer"]["native_delta"]["value"]
    assert native == pytest.approx(symbols.cell_value(edited, target) - reader.native, abs=1e-12)
    assert edge["consumer"]["kind"] == "cell"
    assert [e["layer"] for e in edge["mlp"]] == list(range(0, top))
    # The sum of the ladder's parts equals native_delta by construction (``rounding`` is the
    # remainder), so it is not checked (AUDIT-B1 §4). The real checks: the remainder is small,
    # i.e. the cache's bf16/fp32 V matches the float64 functional of the captured h, and one
    # ladder term recomputed by hand from the two runs' ports.
    tol = 1e-3 if fp32 else 5e-2
    assert abs(edge["rounding"]["value"]) < tol * max(1.0, abs(native))
    c = reader.vector
    d_o = run.read("t3.L0.o").double().reshape(-1) - edited.read("t3.L0.o").double().reshape(-1)
    assert edge["direct"]["value"] == pytest.approx(float(c @ -d_o), rel=1e-9, abs=1e-12)
    d_m = sum(edited.read(f"t3.L{x}.m").double().reshape(-1) - run.read(f"t3.L{x}.m").double().reshape(-1)
              for x in range(top))
    assert edge["totals"]["mlp"]["value"] == pytest.approx(float(c @ d_m), rel=1e-9, abs=1e-12)
    # The ladder's terms are a decomposition of c . Delta h at the consumer's layer: the
    # upstream change, the cell layer's direct write, every later layer's re-read and every
    # layer's captured MLP write. Recomputed by hand from the ports (residual adds h + o + m),
    # this checks that no layer term is missing or counted twice, which the closure (an
    # identity, AUDIT-B1 §5) cannot.
    dh = (edited.read(f"t3.L{top}.h").double().reshape(-1)
          - run.read(f"t3.L{top}.h").double().reshape(-1))
    terms = (edge["upstream"]["value"] + edge["direct"]["value"]
             + edge["totals"]["reread"]["value"] + edge["totals"]["mlp"]["value"])
    assert terms == pytest.approx(float(c @ dh), rel=1e-4 if fp32 else 5e-2,
                                  abs=1e-6 if fp32 else 5e-2)
    assert abs(edge["residual_add_rounding"]["value"]) <= (1e-4 if fp32 else 5e-2) * max(
        1.0, abs(float(c @ dh)))
    fast = symbols.hop_edge(run, edited, reader, cell_layer=0, neurons=False)
    for key in ("direct", "upstream", "norm", "rounding"):
        assert fast[key]["value"] == edge[key]["value"]
    for key in ("mlp", "reread"):
        assert fast["totals"][key]["value"] == edge["totals"][key]["value"]
    assert all(set(e) == {"layer", "captured"} for e in fast["mlp"])
    with pytest.raises(Refusal):
        symbols.hop_edge(run, edited, reader, cell_layer=top)


def test_cell_readout_is_affine_under_a_v_bias():
    """Qwen2 has a V bias: C = c . h + bias, with bias = d . W_g b_V (and the bias cancels in
    every difference, so the ladder is unchanged)."""
    from mdb.engine.models import tiny_model
    from mdb.ops.debugger import attach

    model = tiny_model("qwen2", "fp32", seed=113)
    bias = model.model.layers[1].self_attn.v_proj.bias
    if bias is None:
        pytest.skip("tiny qwen2 has no V bias")
    with torch.no_grad():                         # zero-initialised: set before attaching
        bias.copy_(torch.randn(bias.shape, generator=torch.Generator().manual_seed(0)) * 0.5)
    dbg = attach(model)
    run = dbg.run(PROMPT, prefill=2)
    run.until("t5.logits")
    reader = symbols.cell_readout(run, symbols.CellTarget(3, 1, 0, 4, 9))
    assert reader.bias != 0.0
    h = run.read("t3.L1.h").double().reshape(-1)
    assert float(reader.vector @ h) + reader.bias == pytest.approx(reader.native, rel=1e-4, abs=1e-6)


def test_hop_edge_top_neurons_zero_keeps_every_layer_on_the_neuron_path(debugger):
    """``top_neurons=0`` lists no neurons but still splits every layer exactly (AUDIT-SCALE:
    the list of listed neurons once shadowed the ``neurons`` flag, so an empty list sent every
    layer after the first down the ``neurons=False`` path)."""
    from mdb.analysis.reads import readout

    run = debugger.run(PROMPT, prefill=2)
    if run.adapter.layers < 2:
        pytest.skip("needs two layers")
    run.until("t5.logits")
    reader = readout(run, Margin(5, 4, 9))
    donor = debugger.run([2, 7, 9, 3, 7, 2])
    value = donor.read_cache(0, "v", staged=False).clone()[:, 0:1, 1:2]
    edited = debugger.run(PROMPT, prefill=2)
    edited.set_cell(0, "v", value, rows=[1], kv=[0])
    edited.until("t5.logits")
    none = symbols.hop_edge(run, edited, reader, cell_layer=0, top_neurons=0)
    some = symbols.hop_edge(run, edited, reader, cell_layer=0, top_neurons=2)
    layers = run.adapter.layers
    assert [e["layer"] for e in none["mlp"]] == list(range(layers))
    for a, b in zip(none["mlp"], some["mlp"]):
        assert a["neurons"] == [] and "exact" in a and "linear" in a
        assert a["exact"]["value"] == b["exact"]["value"]
        assert a["omitted"]["count"] == b["omitted"]["count"] + len(b["neurons"])
    assert none["totals"]["mlp_exact"]["value"] == some["totals"]["mlp_exact"]["value"]
