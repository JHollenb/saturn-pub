"""``circuit test ID ref=NAME`` (issue M4): a circuit patched natively from a reference run.

Clean/corrupted pairs on tiny Qwen2 (bf16) and Qwen3 (fp32). Checked: a circuit holding every
component patches nothing in keep-only (keep == base bit for bit) and everything in knock-out
(ko == the full-row patch == the reference's consumer); an empty circuit is the mirror image;
for built circuits (depth 1-3) ko and keep equal the same patches typed by hand with ``set ... =
clean:...`` on a fresh run; ratios are refused when the reference does not move the consumer;
the op refuses a missing ``ref=``, this run, an unknown run and a carried circuit; a reference
of another model is refused; the session and the reference are unchanged; ``+N more`` groups
and their unexamined members are counted in the complement; a paged circuit (40 layers, top=8
depth=3) tests without a size refusal; ``ref=set:NAME`` patches the per-component mean; the
record is sealed, bounded and stored with a project, its component list sha256-checked.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_ops_predict import unlabelled  # noqa: E402

import mdb  # noqa: E402
from mdb.analysis import circuit_test, circuits  # noqa: E402
from mdb.analysis.circuits import Member, Node  # noqa: E402
from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
from mdb.analysis.propagate import state_digest  # noqa: E402
from mdb.core import index  # noqa: E402
from mdb.core.bounded import json_bytes, load_json  # noqa: E402
from mdb.core.pointers import MAX_RECORD_BYTES, check_bounded, tensor_pointer  # noqa: E402
from mdb.core.records import verify_record  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops import Refused, dispatch  # noqa: E402
from mdb.ops.debugger import attach, open_tiny  # noqa: E402
from mdb.ops.predict import session_digest  # noqa: E402

CLEAN, CORRUPT = [1, 7, 9, 3], [1, 5, 9, 3]     # row 1 differs
MARGIN = {"target": "M[4,9]"}
CONFIGS = [("qwen2", "bf16"), ("qwen3", "fp32")]


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda c: "-".join(c))
def tiny(request):
    return open_tiny(*request.param)


def _ctx(debugger, corrupt=CORRUPT, clean=CLEAN):
    ctx = debugger.ops()
    dispatch(ctx, "start", {"prompt": clean, "name": "clean"})
    dispatch(ctx, "until", {"target": f"t{len(clean) - 1}.logits"})
    dispatch(ctx, "start", {"prompt": corrupt})
    dispatch(ctx, "until", {"target": f"t{len(corrupt) - 1}.logits"})
    return ctx


def _test(ctx, ident, ref="clean"):
    return dispatch(ctx, "circuit", {"target": "test", "id": ident, "ref": ref})["test"]


def _margins(result):
    return {key: item["value"] for key, item in result["margins"].items()}


def _by_hand(debugger, prompt, table, which, ref="clean", ref_row=3):
    """The same patches typed with ``set ... = clean:...`` on a fresh run, one ``set`` per head
    or neuron (disjoint masked replaces), then the margin read at the logits."""
    ctx = debugger.ops()
    dispatch(ctx, "start", {"prompt": prompt, "name": "byhand"})
    for entry in table:
        chosen = (entry["circuit"] if which == "circuit" else
                  [i for i in range(entry["size"]) if i not in entry["circuit"]])
        if not chosen:
            continue
        port = entry["address"].split(".", 1)[1]           # L<l>.<port>
        dispatch(ctx, "until", {"target": port.rsplit(".", 1)[0] + ".h"})
        if entry["unit"] in ("embedding", "mlp"):
            dispatch(ctx, "set", {"target": port, "value": f"{ref}:t{ref_row}.{port}"})
            continue
        for i in chosen:
            sel = f"h{i}" if entry["unit"] == "head" else f"i={i}"
            dispatch(ctx, "set", {"target": f"{port}[{sel}]",
                                  "value": f"{ref}:t{ref_row}.{port}[{sel}]"})
    row = len(prompt) - 1
    dispatch(ctx, "until", {"target": f"t{row}.logits"})
    logits = ctx.run.read(f"t{row}.logits").double().reshape(-1)
    value = float(logits[4] - logits[9])
    del debugger.runs["byhand"]
    return value


def _table(ctx, ident):
    run = ctx.run
    ports, _ = circuit_test.components(run, circuits.ledger(run)[ident])
    return circuit_test.component_list(ports)


@pytest.mark.parametrize("shape", [(2, 1), (2, 2), (2, 3), (3, 2)], ids=lambda s: f"top{s[0]}d{s[1]}")
def test_ko_and_keep_equal_the_same_patches_typed_by_hand(tiny, shape):
    top, depth = shape
    ctx = _ctx(tiny)
    built = dispatch(ctx, "circuit", {**MARGIN, "top": top, "depth": depth})
    result = _test(ctx, built["id"])
    assert unlabelled(result) == []
    table = _table(ctx, built["id"])
    found = _margins(result)
    assert found["ko"] == _by_hand(tiny, CORRUPT, table, "circuit")
    assert found["keep"] == _by_hand(tiny, CORRUPT, table, "complement")
    assert found["full"] == found["ref"] and result["full_equals_ref"] is True
    span = found["base"] - found["ref"]
    assert span != 0
    ratios = result["ratios"]
    assert ratios["faithfulness"]["value"] == (found["keep"] - found["ref"]) / span
    assert ratios["knockout_share"]["value"] == (found["base"] - found["ko"]) / span
    assert ratios["faithfulness"]["authority"] == ratios["knockout_share"]["authority"] == (
        "measured")
    assert "verdict" not in result     # no pass/fail bar


def test_a_differing_last_token_patches_the_embedding_too(tiny):
    """The rows differ at the consumer row itself: L0.h is a component and is patched."""
    clean = [1, 7, 9, 5]
    ctx = _ctx(tiny, clean=clean)
    built = dispatch(ctx, "circuit", {**MARGIN, "top": 3, "depth": 2})
    result = _test(ctx, built["id"])
    table = _table(ctx, built["id"])
    found = _margins(result)
    assert found["ko"] == _by_hand(tiny, CORRUPT, table, "circuit")
    assert found["keep"] == _by_hand(tiny, CORRUPT, table, "complement")
    assert found["full"] == found["ref"]


def _everything(circuit):
    """Make ``circuit`` hold every component: each ``+N more`` member becomes a listed node."""
    def lift(node):
        kept = []
        for child in node.children:
            if child.kind == "more":
                kept += [Node(m.name, m.kind, m.port, None, "derived-exact", port=m.port,
                              index=m.index) for m in child.members]
            else:
                lift(child)
                kept.append(child)
        node.children = kept

    lift(circuit.root)
    return circuit


def _nothing(circuit):
    """Make ``circuit`` hold no component: every depth-1 node goes into one ``+N more``."""
    members = []
    fixed = []
    for child in circuit.root.children:
        if child.kind == "more":
            members += child.members
        elif child.kind in ("remainder", "bias", "edit"):
            fixed.append(child)
        else:
            members.append(Member(child.name, child.kind, child.port, child.index, None))
    group = Node(f"+{len(members)} more", "more", "x", None, "derived-exact", members=members)
    circuit.root.children = [group, *fixed]
    return circuit


def test_a_circuit_of_every_component_patches_nothing_in_keep_and_everything_in_ko(tiny):
    ctx = _ctx(tiny)
    run = ctx.run
    built = dispatch(ctx, "circuit", {**MARGIN, "top": 8, "depth": 1})
    circuit = circuits.ledger(run)[built["id"]]
    result = circuits.test(run, circuit, tiny.runs["clean"], name="clean")["result"]
    found = _margins(result)
    assert result["components"]["complement"]["value"] == 0
    assert found["keep"] == found["base"]            # nothing patched: bit for bit
    assert found["ko"] == found["full"] == found["ref"]
    # the same with neurons and heads listed one by one (depth 2, every member lifted)
    built = dispatch(ctx, "circuit", {**MARGIN, "top": 8, "depth": 2})
    circuit = _everything(circuits.ledger(run)[built["id"]])
    result = circuits.test(run, circuit, tiny.runs["clean"], name="clean")["result"]
    found = _margins(result)
    assert result["components"]["complement"]["value"] == 0
    assert found["keep"] == found["base"] and found["ko"] == found["full"] == found["ref"]
    assert result["cost"]["per_pass"]["keep"]["edits"]["value"] == 0


def test_an_empty_circuit_patches_nothing_in_ko_and_everything_in_keep(tiny):
    ctx = _ctx(tiny)
    run = ctx.run
    built = dispatch(ctx, "circuit", {**MARGIN, "top": 2, "depth": 2})
    circuit = _nothing(circuits.ledger(run)[built["id"]])
    result = circuits.test(run, circuit, tiny.runs["clean"], name="clean")["result"]
    found = _margins(result)
    assert result["components"]["circuit"]["value"] == 0
    assert found["ko"] == found["base"]
    assert found["keep"] == found["full"] == found["ref"]
    assert result["ratios"]["faithfulness"]["value"] == 0.0
    assert result["ratios"]["knockout_share"]["value"] == 0.0
    assert result["reading"]["knockout"] == "no measurable effect"


def test_session_and_reference_are_unchanged(tiny):
    ctx = _ctx(tiny)
    clean = tiny.runs["clean"]
    built = dispatch(ctx, "circuit", {**MARGIN, "top": 2, "depth": 3})
    before, where = session_digest(ctx), ctx.run.where
    ref_before, ref_where = state_digest(clean), clean.where
    result = _test(ctx, built["id"])
    assert session_digest(ctx) == before and ctx.run.where == where
    assert state_digest(clean) == ref_before and clean.where == ref_where
    assert result["session"]["before"] == result["session"]["after"]
    assert set(tiny.runs) == {"clean", "main"}


def test_groups_and_unexamined_members_are_counted_in_the_complement(tiny):
    ctx = _ctx(tiny)
    built = dispatch(ctx, "circuit", {**MARGIN, "top": 1, "depth": 2})
    result = _test(ctx, built["id"])
    comp = {k: v["value"] for k, v in result["components"].items()
            if isinstance(v, dict) and "value" in v}
    assert comp["complement"] == comp["total"] - comp["circuit"] > 0
    assert comp["from_groups"] == comp["complement"] and comp["not_listed"] == 0
    # top=1 keeps one depth-1 term: every other attn/mlp term is in the top group, never
    # expanded at depth 2, so its components are unexamined
    circuit = circuits.ledger(ctx.run)[built["id"]]
    heads = tiny.adapter.model.config.num_attention_heads
    top_group = next(c for c in circuit.root.children if c.kind == "more")
    expected = sum(heads if m.kind == "attn" else 1 for m in top_group.members
                   if m.kind in ("attn", "mlp"))
    assert comp["unexamined"] >= expected > 0
    assert result["nodes"]["group"]["refs"]
    shown = dispatch(ctx, "circuit", {"target": "show", "id": built["id"]})
    assert shown["ablated"]["complement"] == result["nodes"]["group"]["refs"]
    assert shown["ablated"]["retained"] == result["nodes"]["circuit"]["refs"]


def test_ratios_are_refused_when_the_reference_does_not_move_the_consumer(tiny):
    ctx = _ctx(tiny, corrupt=CLEAN)          # the reference is the same prompt
    built = dispatch(ctx, "circuit", {**MARGIN, "top": 2, "depth": 2})
    result = _test(ctx, built["id"])
    found = _margins(result)
    assert found["base"] == found["ref"] == found["ko"] == found["keep"]
    assert result["ratios"]["state"] == "refused"
    assert "does not change the consumer" in result["ratios"]["reason"]
    assert "pick another reference" in result["ratios"]["reason"]


def test_op_refusals(tiny):
    ctx = _ctx(tiny)
    built = dispatch(ctx, "circuit", {**MARGIN, "top": 2, "depth": 2})
    before = session_digest(ctx)
    with pytest.raises(Refused, match=r"needs a reference: circuit test ID ref=NAME.*"
                                      r"Zero-ablation is not offered"):
        dispatch(ctx, "circuit", {"target": "test", "id": built["id"]})
    with pytest.raises(Refused, match="is the current run"):
        _test(ctx, built["id"], ref="main")
    with pytest.raises(Refused, match=r"no run 'nope'.*runs: clean, main"):
        _test(ctx, built["id"], ref="nope")
    with pytest.raises(Refused, match="no prompt set 'nope'"):
        _test(ctx, built["id"], ref="set:nope")
    assert session_digest(ctx) == before
    assert circuits.ledger(ctx.run)[built["id"]].tests == []
    # a vector consumer has no ratio
    vector = dispatch(ctx, "circuit", {"target": "L1.h_out", "top": 1, "depth": 1})
    with pytest.raises(Refused, match="is a vector"):
        _test(ctx, vector["id"])
    # a reference that has not computed its last row's logits
    dispatch(ctx, "start", {"prompt": CLEAN, "name": "early"})
    dispatch(ctx, "start", {"prompt": CORRUPT})
    dispatch(ctx, "until", {"target": "t3.logits"})
    fresh = dispatch(ctx, "circuit", {**MARGIN, "top": 2, "depth": 1})
    with pytest.raises(Refused, match="early has not computed t3.logits"):
        _test(ctx, fresh["id"], ref="early")
    # a circuit carried from the run `start` replaced
    with pytest.raises(Refused, match="built by an earlier run"):
        _test(ctx, built["id"])
    del tiny.runs["early"]


def test_a_reference_of_another_model_is_refused():
    one, other = open_tiny("qwen2", "bf16"), open_tiny("qwen3", "fp32")
    run = one.run(CORRUPT)
    run.until("t3.logits")
    ref = other.run(CLEAN)
    ref.until("t3.logits")
    circuit = circuits.build(run, Margin(3, 4, 9), top=2, depth=2)
    with pytest.raises(Refusal, match="another model"):
        circuits.test(run, circuit, ref, name="other")


def test_a_longer_reference_is_read_at_its_last_row(tiny):
    longer = [1, 7, 2, 9, 3]
    ctx = _ctx(tiny, clean=longer)
    built = dispatch(ctx, "circuit", {**MARGIN, "top": 2, "depth": 2})
    result = _test(ctx, built["id"])
    assert result["reference"]["row"]["value"] == 4
    found = _margins(result)
    assert found["full"] == found["ref"]
    table = _table(ctx, built["id"])
    assert found["keep"] == _by_hand(tiny, CORRUPT, table, "complement", ref_row=4)


def test_set_mean_of_copies_of_the_reference_is_the_reference(tiny):
    ctx = _ctx(tiny)
    dispatch(ctx, "promptset", {"action": "define", "name": "same",
                                "arg": '[{"name": "a", "tokens": [1, 7, 9, 3]}, '
                                       '{"name": "b", "tokens": [1, 7, 9, 3]}]'})
    dispatch(ctx, "promptset", {"action": "define", "name": "mixed",
                                "arg": '[{"name": "a", "tokens": [1, 7, 9, 3]}, '
                                       '{"name": "b", "tokens": [1, 2, 6, 9, 3]}]'})
    built = dispatch(ctx, "circuit", {**MARGIN, "top": 2, "depth": 2})
    by_run = _margins(_test(ctx, built["id"]))
    before = session_digest(ctx)
    same = _test(ctx, built["id"], ref="set:same")
    assert same["reference"]["kind"] == "set-mean" and same["reference"]["items"]["value"] == 2
    found = _margins(same)
    # the mean of two equal values is that value: every patch equals the run reference's
    assert {k: found[k] for k in ("ko", "keep", "full")} == {
        k: by_run[k] for k in ("ko", "keep", "full")}
    assert found["ref"] == found["full"]
    mixed = _test(ctx, built["id"], ref="set:mixed")
    assert [r["value"] for r in mixed["reference"]["rows"]] == [3, 4]
    assert _margins(mixed)["ref"] == _margins(mixed)["full"]
    assert session_digest(ctx) == before and set(tiny.runs) == {"clean", "main"}


def test_record_is_sealed_bounded_and_stored_with_its_component_list(tmp_path):
    dbg = mdb.attach(tiny_model("qwen3", "fp32"), project=tmp_path / "p")
    ctx = _ctx(dbg)
    built = dispatch(ctx, "circuit", {**MARGIN, "top": 2, "depth": 2})
    out = dispatch(ctx, "circuit", {"target": "test", "id": built["id"], "ref": "clean"})
    assert out["stored"]
    record = index.get_record(dbg.project, dbg.session_id, out["stored"][0])
    assert record["kind"] == "circuit-test" and record["id"] == out["test"]["record"]
    body = verify_record(record)["body"]
    assert unlabelled(body) == []
    pointer = body["components"]["list"]
    assert pointer["state"] == "stored"
    tensor = tensor_pointer(pointer["ref"], sha256=pointer["tensor_sha256"],
                            shape=[int(pointer["bytes"]["value"])], dtype="uint8")
    table = load_json({"state": "stored", "sha256": pointer["sha256"]},
                      store=dbg.project.store, tensor=tensor)
    assert table == _table(ctx, built["id"])
    assert sum(len(e["circuit"]) for e in table) == body["components"]["circuit"]["value"]


def test_a_paged_circuit_tests_without_a_size_refusal():
    dbg = attach(tiny_model("qwen3", "fp32", layers=40))
    ctx = _ctx(dbg, corrupt=[1, 5, 9, 3, 7, 2, 11, 4, 6, 8, 10, 12],
               clean=[1, 5, 9, 3, 7, 2, 11, 4, 6, 8, 10, 2])
    built = dispatch(ctx, "circuit", {"target": "M[4,9]", "top": 8, "depth": 3})
    circuit = circuits.ledger(ctx.run)[built["id"]]
    assert circuit.pages, "expected a paged circuit"
    result = _test(ctx, built["id"])
    check_bounded(result)
    assert len(json_bytes(result)) < MAX_RECORD_BYTES
    comp = result["components"]
    assert comp["total"]["value"] == 1 + 40 * 4 + sum(
        64 if e["unit"] == "neuron" else 1 for e in _table(ctx, built["id"])
        if e["unit"] in ("mlp", "neuron"))
    assert comp["complement"]["value"] == comp["from_groups"]["value"] > 0
    found = _margins(result)
    assert found["full"] == found["ref"]
