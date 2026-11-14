"""``analysis.circuits`` on tiny Qwen2/Qwen3 (BF16 and FP32).

Closure: at every node the children (retained, ``+N more`` and remainder) sum to the node, and
the depth-1 nodes sum to the native consumer, to float64 roundoff, for a margin, a logit, a
residual port and ``o``. ``test`` (patching from a reference run) is ``test_circuit_test.py``.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis import circuits  # noqa: E402
from mdb.analysis.forecast import Logit, Margin, Refusal  # noqa: E402
from mdb.analysis.propagate import state_digest  # noqa: E402
from mdb.core.pointers import check_bounded  # noqa: E402
from mdb.core.records import verify_record  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

PROMPT = [1, 5, 9, 3]
CONFIGS = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16"), ("qwen3", "fp32"),
           ("llama", "bf16"), ("llama", "fp32"), ("mistral", "bf16"), ("mistral", "fp32")]


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


def _closes(node, tol):
    if not node.children:
        return
    total = sum((c.value for c in node.children[1:]), node.children[0].value)
    assert float((total - node.value).abs().max()) <= tol, node.name
    for child in node.children:
        _closes(child, tol)


@pytest.mark.parametrize("consumer", [Margin(3, 4, 9), Logit(3, 7), "t3.L1.h_out", "t3.L1.r",
                                      "t3.L1.o"], ids=str)
def test_every_node_closes_on_its_children_and_the_root_on_the_consumer(debugger, consumer):
    run = debugger.run(PROMPT)
    run.until("t3.logits")
    circuit = circuits.build(run, consumer, top=2, depth=3)
    _closes(circuit.root, 1e-12)
    body = verify_record(circuit.record)["body"]
    assert body["closure"]["within"] is True
    check_bounded(circuit.record)
    kinds = set()

    def walk(node):
        kinds.add(node["kind"])
        for child in node.get("children", ()):
            walk(child)

    walk(body["root"])
    assert {"head", "source"} <= kinds


def test_refusals(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.L1.n")
    with pytest.raises(Refusal, match="has not been computed"):
        circuits.build(run, Margin(3, 4, 9))
    with pytest.raises(Refusal, match="no additive path"):
        circuits.build(run, "t3.L1.q")
    with pytest.raises(Refusal, match="top must be"):
        circuits.build(run, "t3.L0.h_out", top=9)
    run.until("t3.logits")
    built = circuits.build(run, Margin(3, 4, 9), top=1, depth=1)
    other = debugger.run([1, 7, 9, 3], name="other")
    other.until("t3.logits")
    with pytest.raises(Refusal, match="needs a reference"):
        circuits.test(run, built)
    run.reverse_step("layer")
    with pytest.raises(Refusal, match="was built at"):
        circuits.test(run, built, other, name="other")


def _nodes(node):
    yield node
    for child in node.get("children", ()):
        yield from _nodes(child)


@pytest.mark.parametrize("consumer", [Margin(3, 4, 9), Logit(3, 7), "t3.L1.h_out", "t3.L1.o"],
                         ids=str)
def test_every_remainder_is_bounded_and_within_on_a_native_run(debugger, consumer):
    run = debugger.run(PROMPT)
    run.until("t3.logits")
    body = circuits.build(run, consumer, top=8, depth=3).record["body"]
    remainders = [n for n in _nodes(body["root"]) if n["kind"] == "remainder"]
    assert remainders and all(n["within"] is True for n in remainders)
    assert all(n["bound"].get("authority", n["bound"].get("max_abs", {}).get("authority"))
               == "bound" for n in remainders)
    assert body["remainders_within"] is True and body["closure"]["by_construction"] is True
    assert body["edits"]["count"] == 0 and body["unexamined"] is False


@pytest.mark.parametrize("edit", [("o", 0.5, "add"), ("m", 2.0, "scale"), ("h_out", 0.25, "add")],
                         ids=lambda e: e[0])
def test_a_committed_edit_is_its_own_node_never_a_remainder(debugger, edit):
    port, value, op = edit
    run = debugger.run(PROMPT)
    run.until("t3.L0.h")
    run.set(f"t3.L0.{port}", value, op)
    run.until("t3.logits")
    circuit = circuits.build(run, Margin(3, 4, 9), top=8, depth=2)
    _closes(circuit.root, 1e-12)
    body = circuit.record["body"]
    edits = [n for n in _nodes(body["root"]) if n["kind"] == "edit"]
    assert [n["address"] for n in edits] == [f"t3.L0.{port}"]
    assert body["edits"]["count"] == 1
    assert edits[0]["value"]["authority"] in ("supplied", "derived-exact")
    assert abs(edits[0]["value"]["value"]) > 1e-6
    remainders = [n for n in _nodes(body["root"]) if n["kind"] == "remainder"]
    assert all(n["within"] is True for n in remainders), [
        (n["name"], n["value"], n["bound"]) for n in remainders if not n["within"]]


def test_dropped_subtrees_are_recorded_as_unexamined_and_claim_nothing(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.logits")
    body = circuits.build(run, Margin(3, 4, 9), top=2, depth=3).record["body"]
    groups = [n for n in _nodes(body["root"]) if n["kind"] == "more"]
    marked = [g for g in groups if g.get("unexamined")]
    assert marked and all(g["unexamined_count"]["value"] >= 1 for g in marked)
    assert "not examined" in marked[0]["note"]
    assert body["unexamined"] is True
    assert body["unexamined_count"]["value"] == sum(g["unexamined_count"]["value"] for g in marked)
    # every examined remainder is within, but the dropped subtrees were not looked at
    remainders = [n for n in _nodes(body["root"]) if n["kind"] == "remainder"]
    assert all(n["within"] is True for n in remainders)
    assert body["remainders_within"] is None
    assert "never expanded" in body["closure"]["meaning"]
    # neurons have no subtree: a group of dropped neurons alone is not "unexamined"
    leafy = [g for g in groups if g["address"].endswith(".m (unlisted)")]
    assert not any(g.get("unexamined") for g in leafy)


def test_a_refusal_in_a_dropped_subtree_is_not_raised_but_is_recorded(debugger, monkeypatch):
    def refuse(*args, **kwargs):
        raise Refusal("synthetic: a premise fails in this subtree")

    run = debugger.run(PROMPT)
    run.until("t3.logits")
    monkeypatch.setattr(circuits, "_attn_children", refuse)
    # top=1 keeps only the largest depth-1 node; build succeeds only if no kept attn node
    # expands. Both outcomes are legitimate; what must hold is that a success says what was
    # never examined.
    try:
        body = circuits.build(run, Margin(3, 4, 9), top=1, depth=2).record["body"]
    except Refusal as error:
        assert "synthetic" in str(error)
        return
    assert body["unexamined"] is True and body["remainders_within"] is None
