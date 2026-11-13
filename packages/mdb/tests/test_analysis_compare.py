"""``analysis.compare`` on tiny Qwen2/Qwen3 (BF16 and FP32).

Base, A, B and A+B are each measured natively inside one session; every value equals a fresh
run with the same edits committed, bit for bit, so the interaction ``(A+B) - A - B + base``
equals the one computed from four independent runs. The session digest is unchanged. A
checkpoint side measures that state and refuses A+B; another run is measured as it stands.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis.compare import Side, compare  # noqa: E402
from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
from mdb.analysis.propagate import Edit, state_digest  # noqa: E402
from mdb.core.pointers import check_bounded  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

PROMPT = [1, 5, 9, 3]
CONFIGS = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16"), ("qwen3", "fp32")]
MARGIN = Margin(3, 4, 9)
A = [Edit("t3.L1.m", 0.5, "add")]
B = [Edit("t3.L0.v", 0.25, "add")]


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


def _fresh(debugger, edits, consumer):
    run = debugger.run(PROMPT)
    for edit in sorted(edits, key=lambda e: run.parse(e.address)):
        run.until(edit.address.rsplit(".", 1)[0] + ".h")
        run.set(edit.address, edit.value, edit.op)
    if isinstance(consumer, Margin):
        run.until("t3.logits")
        logits = run.read("t3.logits").double().reshape(-1)
        return logits[consumer.a] - logits[consumer.b]
    run.until(consumer)
    return run.read(consumer).double().reshape(-1)


@pytest.mark.parametrize("cursor", ["t3.logits", "t3.L0.h"])
def test_four_native_measurements_equal_four_fresh_runs_bitwise(debugger, cursor):
    run = debugger.run(PROMPT)
    run.until(cursor)
    before = state_digest(run)
    result = compare(run, Side("A", edits=A), Side("B", edits=B), MARGIN)
    assert state_digest(run) == before and run.where == cursor
    report = check_bounded(result["report"])
    assert report["session"]["unchanged"] is True
    fresh = {"base": _fresh(debugger, [], MARGIN), "A": _fresh(debugger, A, MARGIN),
             "B": _fresh(debugger, B, MARGIN), "A+B": _fresh(debugger, A + B, MARGIN)}
    for name, value in fresh.items():
        assert float(result["values"][name]) == float(value), name
        assert report["measured"][name]["value"]["authority"] == "measured"
    reference = fresh["A+B"] - fresh["A"] - fresh["B"] + fresh["base"]
    assert report["interaction"]["value"]["value"] == float(reference)
    assert report["kind"] == "scalar"


def test_a_vector_consumer_is_compared_elementwise(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.L1.h_out")
    result = compare(run, Side("A", edits=[Edit("t3.L1.m", 0.5, "add")]), Side("B", edits=B),
                     "t3.L1.h_out")
    assert result["report"]["kind"] == "vector"
    for name, edits in (("A", [Edit("t3.L1.m", 0.5, "add")]), ("B", B)):
        assert torch.equal(result["values"][name], _fresh(debugger, edits, "t3.L1.h_out"))
    assert set(result["report"]["interaction"]["value"]) == {"max_abs", "norm"}


def test_checkpoint_and_other_run_sides_refuse_the_joint_branch(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.L1.h")
    clean = run.checkpoint()
    run.set("t3.L1.m", 0.5, "add")
    saved = run.checkpoint()
    run.restore(clean)
    other = debugger.run(PROMPT)
    other.until("t3.L0.h")
    other.set("t3.L0.v", 0.25, "add")
    base_run = debugger.run(PROMPT)
    base_run.until("t3.L1.h")
    result = compare(base_run, Side("A", edits=A), Side("other", run=other), MARGIN)
    assert result["report"]["interaction"]["state"] == "refused"
    assert float(result["values"]["B"]) == float(_fresh(debugger, B, MARGIN))
    assert float(result["values"]["A"]) == float(_fresh(debugger, A, MARGIN))
    held = compare(run, Side("saved", checkpoint=saved), Side("B", edits=B), MARGIN)
    assert "A+B needs both sides" in held["report"]["interaction"]["reason"]
    assert float(held["values"]["A"]) == float(_fresh(debugger, A, MARGIN))


def test_refusals(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.logits")
    with pytest.raises(Refusal, match="token id"):
        compare(run, Side("A", edits=A), Side("B", edits=B), "t3.sample")
