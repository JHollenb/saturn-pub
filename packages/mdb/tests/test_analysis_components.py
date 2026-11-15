"""Component bindings (B-a), feature addresses and feature watch predicates (B-b) and supplied
forecasts (B-c) on tiny Qwen2 (V bias, bf16) and Qwen3 (fp32) with random SAEs.

The components of a value cell add up to its float64 image exactly; their consumer shares add
up to the image's share; the native cell differs from the image by rounding only; row 0 is
refused; a feature predicate fires inside ``run_forward`` and sees an edit; a supplied forecast
is sealed as a verifiable record and judged against a native change with a floor."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis import components, features, sae, supplied, symbols  # noqa: E402
from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
from mdb.analysis.reads import head_mix, readout  # noqa: E402
from mdb.core.records import verify_record  # noqa: E402
from mdb.engine import breakpoints as bp  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

PROMPT = [1, 5, 9, 3, 7]
K, F = 6, 64


@pytest.fixture(scope="module", params=[("qwen2", "bf16"), ("qwen3", "fp32")],
                ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


@pytest.fixture(scope="module")
def basis(tmp_path_factory, debugger):
    d = tmp_path_factory.mktemp("sae")
    width = int(debugger.model.config.hidden_size)
    g = torch.Generator().manual_seed(3)
    for layer in range(debugger.adapter.layers):
        torch.save({"W_enc": torch.randn(F, width, generator=g),
                    "b_enc": torch.randn(F, generator=g) * 0.1,
                    "W_dec": torch.randn(width, F, generator=g) / 8,
                    "b_dec": torch.randn(width, generator=g) * 0.1}, d / f"layer{layer}.sae.pt")
    (d / "config.json").write_text(json.dumps({"k": K, "hook_point": "resid_post"}))
    return sae.open_set("s", str(d)).get(0)          # L0.h_out = L1.h


def _run(debugger, where="t4.logits"):
    run = debugger.run(PROMPT, prefill=1)
    run.until(where)
    return run


def test_components_close_on_the_cell_image_and_the_native_cell(debugger, basis):
    run = _run(debugger)
    cb = symbols.Binding.component(run, "carrier", "cache.L1.v[kv0,row=2]", basis)
    assert cb.kind == "component" and cb.cells == []
    names = [c.name for c in cb.components]
    assert names[-3:] == ["other_features", "bdec", "error"]
    assert len(names) - 3 == len(cb.active)
    closure = cb.closure()
    assert closure["within_f64"], closure
    # the image is the native cell up to its dtype's rounding
    tol = 1e-5 if debugger.adapter.dtype == torch.float32 else 2e-2
    assert closure["rounding_relative"]["value"] < tol
    # each feature component is a_f times the image of d_f
    first = cb.components[0]
    expect = cb.image(basis.direction(first.feature)) * first.activation
    assert torch.allclose(first.value, expect, rtol=0, atol=1e-12)


def test_consumer_shares_partition_the_cells_read(debugger, basis):
    run = _run(debugger)
    cb = components.component_binding(run, "c", "cache.L1.v[kv0,row=2]", basis,
                                      features=list(basis.encode(run.read("t2.L1.h"))["ids"][:2]))
    margin = Margin(4, 4, 9)
    report = components.component_table(run, cb, margin)
    assert report["partition"]["holds"], report["partition"]
    # the native cell's share equals the value-mix read of that cell (P06/P07)
    reader = readout(run, margin)
    mix = head_mix(run, 4, 1)
    parts = mix.scalar_parts(reader.vector)
    heads = [h for h, g in enumerate(mix.kappa) if g == 0]
    direct = float(sum(parts[h, 2] for h in heads))
    assert abs(direct - report["cell_to_consumer"]["value"]) <= 1e-9 * max(1, abs(direct))
    # a feature that is not active is a zero component
    cb2 = components.component_binding(run, "c2", "cache.L1.v[kv0,row=2]", basis,
                                       features=[f for f in range(F) if f not in cb.active][:1])
    assert float(cb2.components[0].value.abs().max()) == 0.0


def test_row_zero_and_wrong_sites_are_refused(debugger, basis):
    run = _run(debugger)
    with pytest.raises(Refusal, match="first token"):
        components.component_binding(run, "c", "cache.L1.v[kv0,row=0]", basis)
    with pytest.raises(Refusal, match="does not hold"):
        components.component_binding(run, "c", "cache.L2.v[kv0,row=2]", basis)
    with pytest.raises(Refusal, match="one value cell"):
        components.component_binding(run, "c", "t2.L1.v", basis)


def test_removing_a_component_is_a_native_cell_edit(debugger, basis):
    run = _run(debugger)
    cb = components.component_binding(run, "c", "cache.L1.v[kv0,row=2]", basis)
    base = run.read("t4.logits").reshape(-1).double()
    edited = debugger.run(PROMPT, prefill=1)
    edited.until("t4.L1.h")
    new = cb.edited(remove=("error",)).to(edited.read_cache(1, "v").dtype)
    edited.set_cell(1, "v", new.reshape(1, 1, 1, -1), rows=[2], kv=[0])
    edited.until("t4.logits")
    after = edited.read("t4.logits").reshape(-1).double()
    assert not torch.equal(base, after)
    held = edited.read_cache(1, "v", staged=False)[0, 0, 2]
    assert torch.equal(held, new)


def test_feature_address_parse_and_value(debugger, basis):
    a = features.parse_feature_address("feature[12]@t2.L1.h")
    assert (a.feature, a.row, a.layer, a.form) == (12, 2, 1, "h")
    c = features.parse_feature_address("feature[7]@cache.L1.v[kv0,row=2]")
    assert (c.form, c.kv, c.row, c.text) == ("cell", 0, 2, "feature[7]@cache.L1.v[kv0,row=2]")
    with pytest.raises(Refusal):
        features.parse_feature_address("feature[x]@t2.L1.h")
    run = _run(debugger)
    acts = features.activations(run, basis, 2, 1)
    f = max(acts, key=acts.get)
    assert features.feature_value(run, f"feature[{f}]@t2.L1.h", basis) == acts[f]
    vec = features.feature_value(run, f"feature[{f}]@cache.L1.v[kv0,row=2]", basis)
    assert vec.shape == (debugger.adapter.head_dim,)
    target = symbols.CellTarget(2, 1, 0, 4, 9)
    scalar = features.feature_value(run, f"feature[{f}]@cache.L1.v[kv0,row=2]", basis,
                                    target=target)
    assert isinstance(scalar, float)


def test_feature_watch_fires_in_run_forward_and_sees_edits(debugger, basis):
    run = _run(debugger, "t4.logits")
    acts = features.activations(run, basis, 2, 1)
    ranked = sorted(acts, key=acts.get, reverse=True)
    top, second = ranked[0], ranked[1]
    ctx = features.FeatureContext(basis, row=2, layer=1, kv=0,
                                  sets={"S": [top, second]}, labels={top: 100, second: 200},
                                  candidates={"C": [4, 9, 11]}, constants={"truth": 200})
    cond = features.FeatureCondition("best_feature(S) >= 0 and best_feature(S) != truth", ctx)
    assert cond.selector_texts() == []
    with pytest.raises(bp.ConditionError):
        features.FeatureCondition("best_feature(NOPE) == 1", ctx)
    with pytest.raises(bp.ConditionError):
        features.FeatureCondition("feature[x] > 0", ctx)
    fresh = debugger.run(PROMPT, prefill=1)
    table = bp.Table()
    watch = table.add_watch("t4.L1.h", fresh.program, mode="edge", compiled=cond)
    out = bp.run_forward(fresh, table, budget=8)
    assert out["reason"] == "watch" and out["where"] == "t4.L1.h", out
    assert table.items[watch.id].hits == 1
    # feature[f] and content(SET) evaluate at the stop
    env = bp._Probe(fresh, 4, "L1.h", None).env(watch.parsed)
    assert features.FeatureCondition(f"feature[{top}] > 0", ctx).evaluate(env)
    assert features.FeatureCondition("content(C) >= 4", ctx).evaluate(env)
    # truth = the top feature's label: the predicate no longer holds
    ctx2 = features.FeatureContext(basis, 2, 1, 0, sets={"S": [top, second]},
                                   labels={top: 100, second: 200}, constants={"truth": 100})
    assert not features.FeatureCondition("best_feature(S) != truth", ctx2).evaluate(env)


def test_supplied_forecast_seals_and_is_judged(debugger):
    run = _run(debugger)
    rec = supplied.seal(run, target={"kind": "margin", "row": 4, "indices": [4, 9]},
                        edit={"address": "t4.L1.m", "row": 4, "layer": 1, "op": "zero neuron 3"},
                        value=-0.75, library="nnsight", version="0.8.2",
                        address_map={"t4.L1.f": "model.model.layers[1].mlp.down_proj.input"},
                        method="grad x act", exact=-0.7, linear=-0.6)
    assert rec["kind"] == "supplied-forecast"
    verify_record(rec)
    out = supplied.judge(rec, -1.0, magnitude=10.0, dtype=torch.bfloat16)
    assert out["native_effect"] and out["sign_agrees"]
    assert abs(out["ratio"]["value"] - 0.75) < 1e-12
    small = supplied.judge(rec, 0.01, magnitude=10.0, dtype=torch.bfloat16)
    assert small["native_effect"] is False and small["sign_agrees"] is None


def test_offline_component_binding_equals_live_bitwise(debugger, basis, tmp_path):
    """SC5: a capture of h (and the value cache) at declared rows builds the same binding as the
    live run, bitwise, with no forward pass; the capture survives save/load; provenance says so."""
    run = _run(debugger)
    cell = "cache.L1.v[kv0,row=2]"
    live = components.component_binding(run, "c", cell, basis)
    cap = components.ResidualCapture.from_run(run, rows=[2, 3], layers=[1])
    path = str(tmp_path / "h.pt")
    cap.save(path)
    for source in (cap, components.ResidualCapture.load(path)):
        off = components.component_binding(None, "c", cell, basis, capture=source,
                                           adapter=debugger.adapter)
        assert [c.name for c in off.components] == [c.name for c in live.components]
        for a, b in zip(off.components, live.components):
            assert torch.equal(a.value, b.value) and torch.equal(a.residual, b.residual)
        assert torch.equal(off.native, live.native) and torch.equal(off.linear, live.linear)
        assert off.rho == live.rho and off.active == live.active
        assert off.provenance["offline"] and off.provenance["native_cell"] == "capture.values"
        assert "capture" in off.provenance["h"]
    assert not live.provenance["offline"] and live.provenance["h"] == "live Run.read"
    # h without values: the native cell is read from the run; without a run it is refused
    thin = components.ResidualCapture.from_run(run, rows=[2], layers=[1], values=False)
    mixed = components.component_binding(run, "c", cell, basis, capture=thin)
    assert torch.equal(mixed.native, live.native)
    with pytest.raises(Refusal, match="no value cache"):
        components.component_binding(None, "c", cell, basis, capture=thin,
                                     adapter=debugger.adapter)
    with pytest.raises(Refusal, match="holds no h"):
        components.component_binding(None, "c", "cache.L1.v[kv0,row=1]", basis, capture=cap,
                                     adapter=debugger.adapter)
    with pytest.raises(Refusal, match="not stepped"):
        components.ResidualCapture.from_run(run, rows=[0], layers=[1])


def test_symbol_table_record_holds_a_wide_feature_binding_by_pointer(tmp_path):
    """A feature binding with hundreds of active features (a k=50 SAE with content is already
    past the 64 KiB record bound) seals: the section's rows go behind a sha256 pointer, the
    report keeps them all."""
    from mdb.core.bounded import load_json

    debugger = open_tiny("qwen3", "fp32")
    width, wide_f, wide_k = int(debugger.model.config.hidden_size), 1024, 400
    g = torch.Generator().manual_seed(5)
    for layer in range(debugger.adapter.layers):
        torch.save({"W_enc": torch.randn(wide_f, width, generator=g),
                    "b_enc": torch.randn(wide_f, generator=g) * 0.1,
                    "W_dec": torch.randn(width, wide_f, generator=g) / 8,
                    "b_dec": torch.randn(width, generator=g) * 0.1},
                   tmp_path / f"layer{layer}.sae.pt")
    (tmp_path / "config.json").write_text(json.dumps({"k": wide_k, "hook_point": "resid_post"}))
    wide = sae.open_set("w", str(tmp_path)).get(0)
    run = _run(debugger)
    cb = symbols.Binding.component(run, "carrier", "cache.L1.v[kv0,row=2]", wide)
    tab = symbols.table(run, consumer=Margin(4, 4, 9), components=[cb])
    record = tab.records[0]
    verify_record(record)
    rows = tab.report["components"][0]["components"]
    assert len(rows) == len(cb.components) > wide_k / 2
    section = record["body"]["components"][0]
    assert "components" not in section and section["partition"]["holds"]
    full = load_json(section["components_full"], run=run)
    assert len(full) == len(rows)
    assert json.dumps(full, sort_keys=True) == json.dumps(json.loads(json.dumps(rows)),
                                                          sort_keys=True)
