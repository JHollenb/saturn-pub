"""``whatif``, ``explain``, ``forecast`` and ``verify`` on tiny Qwen2/Qwen3 (BF16 and FP32).

What-if: the measured change equals a fresh run with the edit committed, bit for bit; the
first-order term agrees with a finite-difference slope; the session is unchanged (digests); a
consumer that is not downstream is refused. Every number of every result carries one valid
label, and a refused part carries none.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis import propagate  # noqa: E402
from mdb.analysis.forecast import Logit, Margin, Refusal  # noqa: E402
from mdb.analysis.propagate import Edit, state_digest, whatif  # noqa: E402
from mdb.core.pointers import AUTHORITIES  # noqa: E402
from mdb.ops import INTEGER_KEYS, Refused, dispatch  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402
from mdb.ops.predict import _session_digest  # noqa: E402

PROMPT = [1, 5, 9, 3]
CONFIGS = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16"), ("qwen3", "fp32")]

# (name, cursor, edit address, value, op, consumer, staged edit kept during the what-if)
SITES = [
    ("residual", "t3.L0.h", "t3.L0.h_out", 0.25, "add", Margin(3, 4, 9), None),
    ("attention", "t3.L0.h", "t3.L1.v", 0.5, "add", Margin(3, 4, 9), None),
    ("mlp", "t3.L0.h", "t3.L1.m", 1.5, "scale", Logit(3, 7), ("t3.L0.q", 0.01, "add")),
    ("rewind", "t3.L1.n", "t3.L0.o", 0.1, "add", "t3.L1.o", None),
    ("next-row", "t3.L0.h", "t3.L0.v", 0.3, "add", Margin(4, 4, 9), None),
]


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


def _stop(consumer) -> str:
    return consumer.text.split(".")[0] + ".logits" if not isinstance(consumer, str) else consumer


def _layer_entry(address: str) -> str:
    row, layer, _ = address.split(".")
    return f"{row}.{layer}.h"


def _fresh(debugger, edits, consumer_stop):
    """An independent run: each edit staged at its layer, then run natively to the consumer."""
    run = debugger.run(PROMPT)
    for address, value, op in edits:
        if run.parse(run.where) < run.parse(_layer_entry(address)):
            run.until(_layer_entry(address))
        run.set(address, value, op)
    run.until(consumer_stop)
    return run.read(consumer_stop)


def unlabelled(node, key=None, labelled=False, path=""):
    """Numeric leaves outside a ``{value, authority}`` claim with a valid label, except
    structural integers under ``INTEGER_KEYS``; plus labelled numbers inside a refused part."""
    bad = []
    if isinstance(node, dict):
        has = "authority" in node and "value" in node
        if has and node["authority"] not in AUTHORITIES:
            bad.append(f"{path}: bad label {node['authority']!r}")
        if node.get("state") == "refused" and _claims(node):
            bad.append(f"{path}: refused part carries a labelled number")
        for k, value in node.items():
            bad += unlabelled(value, k, has, f"{path}.{k}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            bad += unlabelled(value, key, labelled, f"{path}[{i}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        if not labelled and not (type(node) is int and key in INTEGER_KEYS):
            bad.append(path)
    return bad


def _claims(node) -> bool:
    if isinstance(node, dict):
        if node.get("authority") in AUTHORITIES and "value" in node:
            return True
        return any(_claims(v) for v in node.values())
    if isinstance(node, list):
        return any(_claims(v) for v in node)
    return False


@pytest.mark.parametrize("site", SITES, ids=lambda s: s[0])
def test_whatif_measures_a_fresh_committed_run_bitwise_and_leaves_the_session(debugger, site):
    _, cursor, address, value, op, consumer, staged = site
    run = debugger.run(PROMPT)
    if run.parse(cursor) > run.parse(run.where):
        run.until(cursor)
    if staged is not None:
        run.set(*staged)
    before, checkpoint = state_digest(run), run.checkpoint()
    result = whatif(run, Edit(address, value, op), consumer)
    assert state_digest(run) == before
    after = run.checkpoint()
    assert (after["where"], after["cut"], after["edits"], after["path_edits"]) == (
        checkpoint["where"], checkpoint["cut"], checkpoint["edits"], checkpoint["path_edits"])
    stop = _stop(consumer)
    kept = [staged] if staged is not None else []
    base = _fresh(debugger, kept, stop)
    edited = _fresh(debugger, sorted([*kept, (address, value, op)],
                                     key=lambda e: run.parse(e[0])), stop)
    assert torch.equal(result.tensors["native_base"], base)
    assert torch.equal(result.tensors["native_edited"], edited)
    report = result.report
    assert report["session"]["unchanged"] and report["record"] == result.record["id"]
    assert unlabelled(report) == [] and unlabelled(result.record["body"]) == []
    prediction = report["prediction"]
    assert prediction["state"] == "resolved"
    assert prediction["exact"]["value"]  # the exact term exists on every site here
    rounding = report["split"]["terms"][-1]
    assert rounding["name"] == "rounding" and rounding["value"]
    if rounding["within_bound"] is not None:
        assert rounding["within_bound"] is True
    # the split adds up to the native change, element by element
    tensors = result.tensors
    exact, at_realised = tensors["exact"], tensors["at_realised"]
    native = tensors["native_delta"].reshape(exact.shape)
    assert torch.allclose(exact + (at_realised - exact) + (native - at_realised), native,
                          rtol=0, atol=1e-12)


def test_linear_term_matches_the_finite_difference_slope(debugger):
    run = debugger.run(PROMPT)
    fp32 = run.adapter.dtype == torch.float32
    results = {}
    for eps in (1e-3, -1e-3):
        results[eps] = whatif(run, Edit("t3.L1.v", eps, "add"), Margin(3, 4, 9)).tensors
    linear = float(results[1e-3]["linear"])
    exact_slope = float(results[1e-3]["exact"] - results[-1e-3]["exact"]) / 2
    assert abs(exact_slope - linear) <= 1e-4 * abs(linear)
    assert float(results[-1e-3]["linear"]) == pytest.approx(-linear, rel=1e-12)
    if fp32:  # the native slope; bf16 rounding swamps a small edit
        native = {eps: whatif(run, Edit("t3.L1.v", eps, "add"), Margin(3, 4, 9)).tensors
                  for eps in (1e-2, -1e-2)}
        native_slope = float(native[1e-2]["native_delta"] - native[-1e-2]["native_delta"]) / 2
        assert native_slope == pytest.approx(float(native[1e-2]["linear"]), rel=2e-2)


def test_consumers_that_are_not_downstream_are_refused(debugger):
    run = debugger.run(PROMPT)
    with pytest.raises(Refusal, match="not downstream"):
        whatif(run, Edit("t3.L1.m", 0.5, "add"), "t3.L0.o")       # computed before the edit
    with pytest.raises(Refusal, match="no path"):
        whatif(run, Edit("t3.L0.v", 0.5, "add"), "t3.L0.qr")      # later, but not reached
    with pytest.raises(Refusal, match="token id"):
        whatif(run, Edit("t3.L0.v", 0.5, "add"), "t3.sample")
    reach = propagate.affected(run, 3, "L1.o", 4)
    assert "L1.kr" not in reach[3] and "L1.v" not in reach[3]    # o writes no cache row
    assert {"emb", "L0.kr", "logits"} <= reach[4]                # through the sampled token


def test_rewind_with_staged_edits_is_refused_before_anything_native(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.L1.n")
    run.set("t3.L1.n", 0.5, "add")
    before = state_digest(run)
    receipts = len(run.session.receipts)
    with pytest.raises(Refusal, match="drops the staged edits"):
        whatif(run, Edit("t3.L0.o", 0.1, "add"), "t3.L1.o")
    assert state_digest(run) == before and len(run.session.receipts) == receipts


def test_an_edit_on_the_path_refuses_the_prediction_but_still_measures(debugger):
    run = debugger.run(PROMPT)
    run.set("t3.L0.o", 0.2, "add")
    run.until("t3.L1.h")                       # commits layer 0 with the o edit
    result = whatif(run, Edit("t3.L0.v", 0.3, "add"), Margin(3, 4, 9))
    report = result.report
    assert report["prediction"]["state"] == "refused"
    assert "t3.L0.o" in report["prediction"]["reason"]
    assert report["split"]["state"] == "refused"
    assert unlabelled(report) == []
    edited = _fresh(debugger, [("t3.L0.v", 0.3, "add"), ("t3.L0.o", 0.2, "add")], "t3.logits")
    assert torch.equal(result.tensors["native_edited"], edited)


def test_ops_results_are_labelled_and_whatif_keeps_the_debugger_session(debugger):
    ctx = debugger.ops()
    dispatch(ctx, "start", {"prompt": PROMPT})
    dispatch(ctx, "break", {"pattern": "L*.p"})
    dispatch(ctx, "display", {"address": "L0.a[d=0:2]"})
    results = [dispatch(ctx, "forecast", {}), dispatch(ctx, "forecast", {"target": "M[4,9]"})]
    dispatch(ctx, "until", {"target": "t3.L0.k"})
    dispatch(ctx, "set", {"target": "L0.q", "value": 0.01, "op": "add"})
    before = _session_digest(ctx)
    results.append(dispatch(ctx, "whatif", {"target": "t3.L1.m=0.5", "op": "add",
                                            "consumer": "M[4,9]"}))
    results.append(dispatch(ctx, "whatif", {"target": "t3.L1.o[d=0:4]", "value": "L0.k[d=0:4]",
                                            "consumer": "t3.L1.h_out"}))
    assert _session_digest(ctx) == before
    assert results[-1]["edit"]["mask_count"] == 4 and results[-1]["edit"]["source"]
    with pytest.raises(Refused, match="origin resolver"):
        dispatch(ctx, "whatif", {"target": "L1.o=boiling:L1.o", "consumer": "M[4,9]"})
    with pytest.raises(Refused, match="not downstream"):
        dispatch(ctx, "whatif", {"target": "L1.m=0.5", "consumer": "L0.o"})
    dispatch(ctx, "delete", {})
    dispatch(ctx, "until", {"target": "t3.logits"})
    results.append(dispatch(ctx, "verify", {}))
    results += [dispatch(ctx, "explain", {"address": "M[4,9]"}),
                dispatch(ctx, "explain", {"address": "L1.o"})]
    with pytest.raises(Refused, match="until t4.logits"):
        dispatch(ctx, "explain", {"address": "t4.M[4,9]"})
    with pytest.raises(Refused, match="already computed"):
        dispatch(ctx, "forecast", {"target": "L1.o"})
    for result in results:
        assert unlabelled(result) == [], (result["op"], unlabelled(result))
    verdicts = [item["verdict"] for item in results[4]["forecasts"]]
    # L0.a is computed before the L0.q edit; the margin is downstream of it
    assert verdicts == ["within-bound", "edited"]
