"""``analysis.circuits`` on tiny Qwen2/Qwen3 (BF16 and FP32).

Closure: at every node the children (retained, ``+N more`` and remainder) sum to the node, and
the depth-1 nodes sum to the native consumer, to float64 roundoff, for a margin, a logit, a
residual port and ``o``. ``test`` measures exactly what a fresh run with the same ablation
edits committed gives, bit for bit, and leaves the session where it was.
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
CONFIGS = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16"), ("qwen3", "fp32")]


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


def _fresh(debugger, edits, consumer):
    run = debugger.run(PROMPT)
    for edit in sorted(edits, key=lambda e: run.parse(e.address)):
        entry = edit.address.rsplit(".", 1)[0] + ".h"
        if run.parse(run.where) < run.parse(entry):
            run.until(entry)
        run.set(edit.address, edit.value, edit.op, mask=edit.mask, selection=edit.selection)
    run.until("t3.logits")
    logits = run.read("t3.logits").double().reshape(-1)
    return logits[consumer.a] - logits[consumer.b]


def test_native_ablation_equals_a_fresh_committed_run_and_restores(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.logits")
    margin = Margin(3, 4, 9)
    circuit = circuits.build(run, margin, top=2, depth=2)
    before = state_digest(run)
    result = circuits.test(run, circuit)
    assert state_digest(run) == before
    base = _fresh(debugger, [], margin)
    assert result["base"]["value"] == float(base)
    retained = circuits._leaves(circuit.root)
    others = [m for g in circuits._groups(circuit.root) for m in g.members]
    for name, nodes in (("retained", retained), ("complement", others)):
        edits = circuits._merged([e for n in nodes for e in n.edits])
        assert result[name]["measured"]["value"] == float(_fresh(debugger, edits, margin))
        assert result[name]["delta"]["authority"] == "measured"
        assert result[name]["predicted_delta"]["authority"] == "derived-local"
    limit = 0.1 * abs(float(base))
    sufficient = abs(result["complement"]["delta"]["value"]) <= limit
    assert result["verdict"] == ("sufficient at this dose" if sufficient
                                 else "not sufficient at this dose")


def test_refusals(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.L1.n")
    with pytest.raises(Refusal, match="has not been computed"):
        circuits.build(run, Margin(3, 4, 9))
    with pytest.raises(Refusal, match="no additive path"):
        circuits.build(run, "t3.L1.q")
    with pytest.raises(Refusal, match="top must be"):
        circuits.build(run, "t3.L0.h_out", top=9)
    built = circuits.build(run, "t3.L0.h_out", top=1, depth=1)
    run.until("t3.L1.m")
    with pytest.raises(Refusal, match="was built at"):
        circuits.test(run, built)
