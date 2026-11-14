"""Symbol-table additions (review 2026-11-14) on tiny models: name-keyed statistics and the
``freeze`` misalignment they fix, the contrast / mask_delta / non-sink columns against hand
arithmetic, roles and labels from a circuit map, candidate flags, the whole-subspace co-readout
and lexical null, the switched readout, causal liveness against ``rows.row_table``'s arm, the
long-frame export and JSON schema, sealed traces and maps, and ``MapTable``.
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis import candidates as cs  # noqa: E402
from mdb.analysis import circuit_map as cm  # noqa: E402
from mdb.analysis import frame as fr  # noqa: E402
from mdb.analysis import sealed, summary, symbols  # noqa: E402
from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
from mdb.analysis.reads import head_mix  # noqa: E402
from mdb.core.pointers import check_bounded  # noqa: E402
from mdb.core.records import verify_record  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

PROMPT = [1, 5, 9, 3, 7, 2]
CONFIGS = [("qwen3", "fp32"), ("qwen2", "bf16")]


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


class Tok:
    """A word -> ids table standing in for a tokenizer (``encode(" " + word)``)."""

    def __init__(self, table):
        self.table = table

    def encode(self, text, add_special_tokens=False):
        return list(self.table[text.strip()])

    def decode(self, ids):
        return "|".join(str(i) for i in ids)


def claim(x, a="measured"):
    return {"value": x, "authority": a}


# --------------------------------------------------------------------------- 1: name keys


def _three_slot_map():
    slots = [cm.Slot("A", "carrier", "gt", layers=(0,), kv=(0,)),
             cm.Slot("B", "descriptor", "R_last"),
             cm.Slot("C", "descriptor", "rest")]
    return cm.Map(slots, [cm.Edge("B", "A", "read")], {"kind": "margin"}, {})


def test_positional_paths_misalign_slots_when_a_slot_binds_no_row(debugger):
    """The review's latent bug, reproduced: ``bind`` skips B on the second item (its row is the
    carrier cell's row), so page position 1 is B on one item and C on the other. Positional
    paths merge them; name-keyed paths do not."""
    m = _three_slot_map()
    reports = []
    for gt in (1, 3):
        run = debugger.run(PROMPT[:4])
        run.until("t3.logits")
        reports.append(cm.item_report(m, run, {"name": f"g{gt}", "span": [1, 2, 3], "gt": gt},
                                      consumer=Margin(3, 4, 9)))
    assert [b["symbol"] for b in reports[0]["table"]["bindings"]] == ["A", "B", "C"]
    assert [b["symbol"] for b in reports[1]["table"]["bindings"]] == ["A", "C"]
    assert reports[1]["unbound_slots"] == ["B"]
    # before the fix: the positional path exists on both items and names different slots
    old = [summary.verdicts(r, positional=True) for r in reports]
    assert summary.leaf(reports[0], "table.bindings.1.symbol") == "B"
    assert summary.leaf(reports[1], "table.bindings.1.symbol") == "C"
    pos = {s["path"] for s in summary.summarize(
        [{"x": summary.claims(r, positional=True)} for r in reports])["numeric"]}
    del old, pos
    flat = [summary.claims(r, positional=True) for r in reports]
    assert "table.bindings.1.mass" in flat[0] and "table.bindings.1.mass" in flat[1]
    # after: name keys; B is partial (summarised nowhere), C lines up with C
    named = summary.summarize(reports)
    paths = {s["path"] for s in named["numeric"]}
    assert "table.bindings.C.mass" in paths and "table.bindings.A.mass" in paths
    assert not any(p.startswith("table.bindings.B.") for p in paths)
    assert not any(p.startswith("table.bindings.1.") for p in paths)
    frozen = cm.freeze(m, reports, ["table.bindings.C.symbol"])
    assert frozen.constants["unbound_slots"] == {"B": 1}
    assert frozen.constants["modes"][0]["mode"] == "C"
    assert summary.leaf(reports[1], "table.bindings.C.role") == "descriptor"


def test_modes_break_ties_by_value_and_report_them():
    rows = [{"x": 3}, {"x": 1}]
    first = summary.modes(rows, ["x"])[0]
    assert first["mode"] == 3 and first["tie"]["values"] == [3, 1]
    by_value = summary.modes(list(reversed(rows)), ["x"], ties="value")[0]
    assert by_value["mode"] == 1 == summary.modes(rows, ["x"], ties="value")[0]["mode"]
    assert "tie" not in summary.modes([{"x": 1}, {"x": 1}, {"x": 2}], ["x"])[0]


def test_summarize_by_groups_by_a_label():
    results = [{"labels": {"family": "city"}, "m": claim(1.0)},
               {"labels": {"family": "city"}, "m": claim(3.0)},
               {"labels": {"family": "river"}, "m": claim(5.0)}, {"m": claim(9.0)}]
    out = summary.summarize_by(results, "labels.family")
    assert out["unlabelled"] == 1
    groups = {g["value"]: g for g in out["groups"]}
    assert groups["city"]["n"] == 2
    assert groups["city"]["summary"]["numeric"][0]["mean"]["value"] == 2.0
    assert groups["river"]["summary"]["numeric"][0]["mean"]["value"] == 5.0


# --------------------------------------------------------------------------- 3: columns


def test_contrast_mask_delta_and_nonsink_mass_equal_hand_arithmetic(debugger):
    run = debugger.run(PROMPT)
    run.until("t5.logits")
    margin = Margin(5, 4, 9)
    tab = symbols.table(run, consumer=margin)
    check_bounded(tab.report)
    reader = __import__("mdb.analysis.reads", fromlist=["readout"]).readout(run, margin)
    layers, heads = run.adapter.layers, run.adapter.num_heads
    hand_c = {s: 0.0 for s in range(6)}
    hand_m = {s: 0.0 for s in range(6)}
    hand_ns = {s: 0.0 for s in range(6)}
    identity = 0.0
    for layer in range(layers):
        mix = head_mix(run, 5, layer, {}, lazy=True)
        parts = mix.scalar_parts(reader.vector)                     # (H, T) share effects
        p = mix.p
        for h in range(heads):
            total = float(parts[h].sum())
            identity += total * (1 - float(p[h].sum()))
            den = float(p[h].sum() - p[h, 0])
            for s in range(6):
                con = float(parts[h, s]) - float(p[h, s]) * total
                hand_c[s] += con
                hand_m[s] += -con / (1 - float(p[h, s]))
                if s != 0:
                    hand_ns[s] += float(p[h, s]) / den
    for view in tab.report["bindings"]:
        s = int(view["symbol"][1:])
        assert view["contrast"]["value"] == pytest.approx(hand_c[s], abs=1e-12)
        assert view["mask_delta"]["value"] == pytest.approx(hand_m[s], abs=1e-12)
        assert view["mass_nonsink"]["value"] == pytest.approx(hand_ns[s], abs=1e-12)
        assert view["causal"]["status"] == "unknown"
    # contrast sums to sum_h xbar_h (1 - sum_s p) over all bindings: 0 up to p's rounding
    total_c = sum(v["contrast"]["value"] for v in tab.report["bindings"])
    assert total_c == pytest.approx(identity, abs=1e-12)
    t0 = tab.report["bindings"][0]
    assert t0["mass_nonsink"]["value"] == 0.0 and "sink_ratio" in t0
    assert tab.report["sink"] == ["t0"]
    # mask_delta is the exact direct-path change of masking one read with renormalisation
    mix = head_mix(run, 5, 1, {}, lazy=True)
    parts = mix.scalar_parts(reader.vector)
    p, h, s = mix.p, 0, 2
    o = float(parts[h].sum())
    masked = (o - float(parts[h, s])) / (1 - float(p[h, s]))
    con = float(parts[h, s]) - float(p[h, s]) * o
    assert masked - o == pytest.approx(-con / (1 - float(p[h, s])), abs=1e-12)
    off = symbols.table(run, consumer=margin, sink=None)
    assert "mass_nonsink" not in off.report["bindings"][1] and "sink" not in off.report


def test_roles_labels_and_derived_labels_reach_the_table(debugger):
    run = debugger.run(PROMPT)
    run.until("t5.logits")
    m = cm.Map([cm.Slot("carrier", "carrier", "gt", layers=(0,), kv=(0,)),
                cm.Slot("desc", "descriptor", "rest")], [], {"kind": "margin"},
               {"small": [2, 4, 6, 8]})
    item = {"name": "i", "span": [1, 2, 3, 4], "gt": 2, "labels": {"family": "city"}}
    rep = cm.item_report(m, run, item, consumer=Margin(5, 4, 9), truths={"small": 4},
                         derive={"decoded": "content.small.best"})
    carrier = rep["table"]["bindings"][0]
    assert (carrier["slot"], carrier["role"], carrier["map_digest"]) == \
        ("carrier", "carrier", m.digest())
    assert carrier["labels"]["family"] == {"label": "city", "source": "supplied"}
    assert carrier["labels"]["decoded"]["authority"] == "derived-exact"
    assert carrier["labels"]["decoded"]["value"] == carrier["content"]["small"]["best"]
    assert rep["labels"]["family"]["label"] == "city"
    assert rep["table"]["derived_labels"] == {"decoded": "content.small.best"}
    tab = symbols.table(run, consumer=Margin(5, 4, 9), labels={"depth": 2})
    assert tab.report["labels"]["depth"]["value"] == 2
    verify_record(tab.records[0])


# --------------------------------------------------------------------------- 2: candidates


WORDS = {"France": [4], "Japan": [6, 11], "Japanese": [6, 12], "Peru": [8],
         "Paris": [2], "Tokyo": [10], "Lima": [3], "French": [4]}


def _sets():
    tok = Tok(WORDS)
    countries = cs.build("countries", ["France", "Japan", "Peru"], tok)
    languages = cs.build("languages", ["French", "Japanese"], tok)
    capitals = cs.pair(cs.build("capitals", ["Paris", "Tokyo", "Lima"], tok), countries)
    return cs.cross([countries, languages, capitals]), tok


def test_candidate_set_flags_provenance_and_pairs():
    (countries, languages, capitals), tok = _sets()
    assert countries.tokens == (4, 6, 8)
    assert countries.multi_piece(6) and not countries.multi_piece(4)
    assert countries.across[4] == ("languages",) and countries.across[6] == ("languages",)
    assert len(countries.sha256) == 64 and countries.describe()["multi_piece_words"] == 1
    assert capitals.pairs["Japan"] == "Tokyo"
    with pytest.raises(Refusal, match="single"):
        cs.build("x", ["Japan", "Peru"], tok, rule="single")
    collide = cs.build("c", ["France", "French"], tok)
    assert collide.collides_within("France")
    flags = cs.content_flags(countries.candidates("Japan"), 6,
                             [countries.candidates(), languages.candidates()])
    assert flags["first_subword"] is True and flags["kind"] == ["languages"]
    assert flags["collision"] is False


def test_co_readout_removes_the_whole_subspace_and_reports_the_lexical_null(debugger):
    run = debugger.run(PROMPT)
    run.until("t5.logits")
    (countries, languages, capitals), _ = _sets()
    sets = [countries.candidates("Japan"), capitals.candidates("Tokyo")]
    co = [cs.CoReadout("capitals", project_out="countries")]
    tab = symbols.table(run, consumer=Margin(5, 4, 9), sets=sets, co=co, lexical_null=True,
                        switch={"capitals": "countries"})
    verify_record(tab.records[0])
    check_bounded(tab.report)
    view = tab.report["bindings"][2]
    found = view["content"]["capitals"]
    co_out = found["co"]["countries"]
    assert co_out["rank"] == 3 and co_out["projection"]["authority"] == "supplied"
    assert "counts" in co_out
    assert view["content"]["countries"]["truth"] == countries.token("Japan")
    assert view["content"]["countries"]["flags"]["first_subword"] in (True, False)
    assert found["lexical_null"]["source_token"] == 10
    assert found["switched"]["on"] == "countries"
    # by hand: projecting the span out leaves every country score at 0 (whole subspace)
    rho, gamma = symbols.readout_basis(run, 5)
    W = run.adapter.model.lm_head.weight.double()
    q = cs.subspace(run, countries.tokens, gamma)
    rows = W[list(countries.tokens)] * gamma
    w = torch.randn(rows.shape[1], dtype=torch.float64)
    outside = w - q @ (q.T @ w)
    assert float((rows @ outside).abs().max()) < 1e-9
    one = cs.subspace(run, countries.tokens, gamma, k=1)
    assert one.shape[1] == 1
    # the switched truth is the capital paired with the decoded country
    decoded = view["content"]["countries"]["best"]
    word = countries.words_of(decoded)[0]
    assert found["switched"]["truth"] == capitals.token(capitals.pairs[word])


# --------------------------------------------------------------------------- 5: liveness


def test_causal_liveness_equals_the_row_table_arm(debugger):
    from mdb.analysis.rows import row_table

    run = debugger.run(PROMPT)
    run.until("t5.logits")
    tab = symbols.table(run, consumer=Margin(5, 4, 9), causal=True)
    check_bounded(tab.report)
    verify_record(tab.records[0])
    views = {v["symbol"]: v for v in tab.report["bindings"]}
    trace = row_table(run, {"x": [1]}, letters=[4, 9], X="A", R="B",
                      arms={f"t{s}": [s] for s in range(6)})
    last = trace.records[-1]
    for s in range(6):
        live = views[f"t{s}"]["causal"]
        assert live["status"] == "intervened" and live["scope"] == "set"
        native_delta = last["ablated"][f"t{s}"] - last["margin"]
        tol = 1e-4 if run.adapter.model.dtype == torch.float32 else 0.1
        assert live["delta"]["value"] == pytest.approx(native_delta, abs=tol)
        assert live["verdict"] in ("effect", "null", "below_floor")
    run = debugger.run(PROMPT)
    run.until("t5.L0.h")
    run.set("t5.L0.v", 0.25, "add")
    run.until("t5.logits")
    with pytest.raises(Refusal, match="edits"):
        symbols.table(run, consumer=Margin(5, 4, 9), causal=True)


# --------------------------------------------------------------------------- frames, records


def test_frames_arrow_round_trip_and_schema(debugger, tmp_path):
    run = debugger.run(PROMPT)
    run.until("t5.logits")
    tab = symbols.table(run, consumer=Margin(5, 4, 9), foreign=True)
    df = tab.to_frame()
    assert list(df["symbol"]) == [f"t{s}" for s in range(6)]
    assert (df["mass__auth"] == "derived-exact").all()
    assert df["record_id"].iloc[0] == tab.records[0]["id"]
    reads = tab.to_frame("read", foreign=True)
    for name, group in reads.groupby("binding"):
        view = next(v for v in tab.report["bindings"] if v["symbol"] == name)
        assert group["mass"].sum() == pytest.approx(view["mass"]["value"], abs=1e-12)
        assert group["effect"].sum() == pytest.approx(view["to_consumer"]["value"], abs=1e-12)
        assert group["contrast"].sum() == pytest.approx(view["contrast"]["value"], abs=1e-12)
    assert reads["tl_hook"].iloc[0].startswith("blocks.")
    heads = tab.to_frame("head")
    assert len(heads) == run.adapter.layers * run.adapter.num_heads
    path = tab.to_parquet(str(tmp_path / "t.parquet"))
    import pyarrow.parquet as pq

    back = pq.read_table(path)
    assert back.schema.metadata[b"mdb_schema"] == b"mdb-symbol-table-v1"
    rows = fr.from_frame(back.to_pandas())
    assert rows[1]["mass"] == {"value": pytest.approx(tab.report["bindings"][1]["mass"]["value"]),
                               "authority": "derived-exact"}
    schema = fr.json_schema()
    assert "bindings" in schema["properties"] and "contrast" in schema["$defs"]["binding"][
        "properties"]
    assert fr.json_schema("mdb-circuit-map-v1")["x-record-kind"]["kind"] == "circuit-map"
    assert "| `contrast` |" in fr.schema_markdown()
    view = tab.report["bindings"][1]
    assert view["foreign"]["cells"][0]["tl"]["index"] == "[:, 1, :, :]"
    assert view["readers"][0]["foreign"]["tl"]["hook"].endswith("hook_pattern")


def test_loo_position_mean_excludes_the_row_itself():
    import pandas as pd

    df = pd.DataFrame({"position": [0, 0, 0, 1], "v": [1.0, 2.0, 6.0, 5.0]})
    out = fr.loo_position_mean(df, "v")
    assert list(out[:3]) == [1.0 - 4.0, 2.0 - 3.5, 6.0 - 1.5]
    assert out.isna().iloc[3]


def test_sealed_trace_replays_from_its_record_and_map_round_trips(debugger):
    from mdb.analysis.rows import row_table
    from mdb.engine.breakpoints import replay

    run = debugger.run(PROMPT)
    run.until("t5.logits")
    trace = row_table(run, {"a": [0, 1, 2], "b": [3, 4, 5]}, letters=[4, 9], X="A", R="B")
    record = trace.seal(session_id="s1", run=run)
    verify_record(record)
    assert record["body"]["count"] == 6 and record["body"]["sha256"]
    again = sealed.load_trace(record, holder=run)
    assert again.records == json.loads(json.dumps(trace.records))
    hits = replay('mass["a"] > mass["b"]', trace)
    from_record = replay('mass["a"] > mass["b"]', again)
    assert hits["holds"] == from_record["holds"]
    watch = sealed.seal_watch_hits(hits, record, session_id="s1", holder=run)
    verify_record(watch)
    assert watch["body"]["trace"] == record["id"]
    frame = trace.to_frame()
    assert "mass.a" in frame.columns and len(frame) == 6
    m = _three_slot_map()
    m2 = cm.freeze(m, [{"x": claim(1.0)}], [],
                   supplied={"hop": cm.supplied_constant(0.3, source="rec123")})
    rec = sealed.seal_map(m2, session_id="s1", run=run)
    verify_record(rec)
    assert sealed.load_map(rec, holder=run).digest() == m2.digest() == rec["body"]["digest"]
    with pytest.raises(Refusal, match="source"):
        cm.freeze(m, [], [], supplied={"hop": {"value": 0.3}})


def test_map_table_rows_baselines_and_permutation_null(debugger):
    m = cm.Map([cm.Slot("carrier", "carrier", "gt", layers=(0,), kv=(0,)),
                cm.Slot("desc", "descriptor", "rest")], [], {"kind": "margin"},
               {"small": [2, 4, 6, 8]})
    reports = []
    for gt, fam in ((1, "city"), (2, "city"), (3, "river")):
        run = debugger.run(PROMPT)
        run.until("t5.logits")
        reports.append(cm.item_report(m, run, {"name": f"i{gt}", "span": [1, 2, 3, 4], "gt": gt,
                                               "labels": {"family": fam}},
                                      consumer=Margin(5, 4, 9), truths={"small": 4},
                                      lexical_null=True))
    mt = cm.MapTable.build(m, reports, switch="labels.family.label")
    carrier = next(r for r in mt.rows if r["slot"] == "carrier")
    assert carrier["n_bound"] == 3
    assert carrier["site_mode"]["n"] == 3
    assert carrier["content"]["small"]["answer_permutation"]["n"] == 3
    assert {g["value"] for g in carrier["by_switch"]} == {"city", "river"}
    df = mt.frame()
    assert set(df["slot"]) == {"carrier", "desc"} and "item_labels.family" in df.columns
    rec = sealed.seal_map(m, session_id="s1", run=run, table=mt)
    verify_record(rec)
    frozen = cm.freeze(m, reports, [], switch="labels.family.label")
    assert {g["value"] for g in frozen.constants["by_switch"]["groups"]} == {"city", "river"}
